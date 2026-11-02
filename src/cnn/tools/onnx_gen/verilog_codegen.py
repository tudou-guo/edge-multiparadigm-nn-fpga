"""E1 graph wrapper/ROM code generation (W3/W5).

Reuses the frozen npu_top.v.tpl and op_shell ABI.  It fills capacity, engine and
instruction-ROM placeholders; the instruction stream itself comes from
instr_codegen.py.

2026-10-24 DW-parameter work:
  * the generated top now carries DW_PC / DW_TAP_PARALLEL / DW_RQ_LANES and
    forwards them through npu_mixed_engine -> npu_engine_router -> the frozen
    depthwise_conv3x3_core parameter interface;
  * MAX_CH capacity is the graph-derived channel capacity widened by
    align8(DW_PC) when that is larger; the requested and allocated values are
    both recorded so DW_PC is never presented as the logical channel count;
  * the manifest records the resolved engine profile, target, per-engine gates
    and the exact generated parameter values.  Compile evidence alone never
    claims functional/OOC/platform verification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

from . import compute_profiles as validation
from . import engine_profiles, targets
from .graph_ir import E_CFG, CompileError, Graph
from .instr_codegen import generate_stream, write_stream
from .shape_infer import quant_storage  # noqa: F401  (re-exported for instr_codegen)

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "tools/npu_gen/npu_top.v.tpl"
GEN_VERSION = "onnx_gen-e1-graph-0.2"

# One catalog is shared by both graph codegen and the legacy ops generator.
from .source_library import rtl_sources, source_identity, write_binding

# Retained import name for older callers; resolution uses the selected root.
RTL_SOURCES = rtl_sources(ROOT)

# Record opcode -> engine identity (rtl/operator_library.json engines).  The
# opcode is authoritative; the op-name map is only a readable fallback.
OPCODE_ENGINE = {
    0x0001: "dense", 0x0002: "pool2x2", 0x0010: "eltwise", 0x0011: "eltwise",
    0x0012: "copy", 0x0013: "copy", 0x0014: "copy", 0x0015: "eltwise",
    0x0016: "reduce", 0x0017: "pool2x2", 0x0018: "eltwise", 0x0019: "resize",
    0x001A: "pool5", 0x001B: "activation", 0x001C: "depthwise", 0x001D: "maxpool2d",
}
OPNAME_ENGINE = {
    "Conv": "dense", "Conv3x3": "dense", "Conv1x1": "dense",
    "DepthwiseConv": "depthwise", "ActivationLUT": "activation",
    "MaxPool": "pool2x2", "AveragePool": "pool2x2",
    "Relu": "eltwise", "Add": "eltwise", "Clip": "eltwise", "Mul": "eltwise",
    "Pad": "copy", "Slice": "copy", "Concat": "copy",
    "GlobalAveragePool": "reduce", "Resize": "resize",
}
ALL_ENGINES = ("dense", "pool2x2", "eltwise", "copy", "reduce", "resize",
               "pool5", "activation", "depthwise", "maxpool2d")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def derive_capacities(graph: Graph) -> Dict[str, int]:
    """Actual MAX_CIN/MAX_COUT/MAX_CH/MAX_W derived from the lowered graph.

    This is the frozen 2026-10-22 algorithm, extracted so that the explicit
    compute-profile branch can validate the *actual* build capacity before any
    compile artifact is written.  A compute profile only bounds these values;
    it is never injected into the generated MAX parameters.
    """
    convs = [node for node in graph.nodes if node.op_type == "Conv"]
    max_cin = max([int(graph.tensors[node.inputs[0]].shape[1]) for node in convs] or [1])
    max_cout = max([int(graph.tensors[node.outputs[0]].shape[1]) for node in convs] or [1])
    rank4 = [tensor for tensor in graph.tensors.values()
             if tensor.dtype == "int8" and tensor.shape is not None and len(tensor.shape) == 4]
    def channel_alloc(value):
        return ((int(value) + 7) // 8) * 8
    max_ch = max([channel_alloc(max_cin), channel_alloc(max_cout)] +
                 [channel_alloc(int(t.shape[1])) for t in rank4] or [1])
    max_w = max([int(t.shape[2]) for t in rank4] + [int(t.shape[3]) for t in rank4] or [1])
    return {
        "max_cin": max_cin, "max_cout": max_cout,
        "max_ch": max_ch, "max_w": max_w,
    }


def derive_spatial_capacities(graph: Graph, allocated_h: int,
                               allocated_w: int) -> Dict[str, int]:
    """v0.2 spatial capacity: channel capacity from the graph, H/W explicitly.

    The legacy derive_capacities folds max(H,W) into one max_w field.  A spatial
    build keeps the channel derivation byte-identical but takes the two axes
    from the resolved spatial capacity, so H never widens a row buffer and W is
    not silently raised by a tall tensor.
    """
    legacy = derive_capacities(graph)
    return {
        "max_cin": int(legacy["max_cin"]),
        "max_cout": int(legacy["max_cout"]),
        "max_ch": int(legacy["max_ch"]),
        "max_h": int(allocated_h),
        "max_w": int(allocated_w),
    }


def dw_capacity_alloc(dw_pc: int) -> int:
    """Channel capacity needed by one DW engine with DW_PC MAC lanes.

    Rounding matches derive_capacities (channels padded to a multiple of 8).
    MAX_CH is a *capacity*, not the logical C of the graph.
    """
    return ((int(dw_pc) + 7) // 8) * 8


def allocate_capacity(capacities: Dict[str, int], dw_pc: int) -> Dict[str, int]:
    """MAX_CH = max(graph-derived MAX_CH, align8(DW_PC)); other capacities kept."""
    out = {key: int(value) for key, value in capacities.items()}
    if "max_ch" not in out:
        raise CompileError(E_CFG, "build capacity is missing max_ch")
    out["max_ch"] = max(int(out["max_ch"]), dw_capacity_alloc(dw_pc))
    return out


def _capacities(graph: Graph, tn: int, tm: int) -> Dict[str, int]:
    cap = {"tn": int(tn), "tm": int(tm)}
    cap.update(derive_capacities(graph))
    return cap


def resolve_repo_root(repo_root) -> Path:
    """Resolve the explicit candidate source root (None keeps the frozen ROOT)."""
    root = ROOT if repo_root is None else Path(repo_root)
    try:
        root = root.resolve()
    except OSError as exc:
        raise CompileError(E_CFG, "repo_root cannot be resolved: %s" % exc,
                           detail={"repo_root": str(repo_root)})
    if not root.is_dir():
        raise CompileError(E_CFG, "repo_root is not a directory",
                           detail={"repo_root": str(root)})
    return root


def template_path_for(repo_root) -> Path:
    """npu_top.v.tpl always comes from the *same* tree as the RTL sources."""
    root = ROOT if repo_root is None else Path(repo_root)
    return root / "tools/npu_gen/npu_top.v.tpl"


def rtl_sources_for(repo_root):
    return rtl_sources(ROOT if repo_root is None else repo_root)


PHYSICAL_BITS = {
    "dense": 0, "maxpool2": 1, "eltwise": 2, "copy": 3, "clip": 4,
    "gap": 5, "avgpool2": 6, "mul": 7, "resize": 8, "pool5": 9,
    "activation": 10, "depthwise": 11, "maxpool2d": 12,
}

# Physical instance owning each lowered opcode (authoritative for pruning).
# Separate from OPCODE_ENGINE on purpose: that map keeps the legacy *logical*
# families (Add/Relu/Clip/Mul = eltwise, MaxPool/AvgPool = pool2x2) and cannot
# decide which physical instance to elaborate.
OPCODE_PHYSICAL = {
    0x0001: "dense", 0x0002: "maxpool2", 0x0010: "eltwise", 0x0011: "eltwise",
    0x0012: "copy", 0x0013: "copy", 0x0014: "copy", 0x0015: "clip",
    0x0016: "gap", 0x0017: "avgpool2", 0x0018: "mul", 0x0019: "resize",
    0x001A: "pool5", 0x001B: "activation", 0x001C: "depthwise", 0x001D: "maxpool2d",
}

# These cores remain structurally instantiated; the rest are mask-gated.
PHYSICAL_STRUCTURAL = ("eltwise", "copy", "resize")
PHYSICAL_GATED = ("dense", "maxpool2", "clip", "gap", "avgpool2", "mul",
                  "pool5", "activation", "depthwise", "maxpool2d")
STRUCTURAL_MASK = sum(1 << PHYSICAL_BITS[k] for k in PHYSICAL_STRUCTURAL)


def _physical_required_mask(stream, strict=True):
    """Derive the physical-instance mask required by the lowered opcodes."""
    required = 0
    usage = {}
    unknown = []
    for record in stream.get("records") or []:
        opcode = int(record.get("opcode", -1))
        key = OPCODE_PHYSICAL.get(opcode)
        if key is None:
            unknown.append({"opcode": opcode, "op": record.get("op")})
            continue
        required |= 1 << PHYSICAL_BITS[key]
        usage[key] = usage.get(key, 0) + 1
    if unknown and strict:
        raise CompileError(E_CFG,
                           "engine pruning cannot classify opcode(s): %s" %
                           ", ".join("0x%04X" % row["opcode"] for row in unknown),
                           detail={"unknown_opcodes": unknown})
    return {"required_mask": required, "usage": usage, "unknown": unknown}


def _logical_instantiated(name, physical_mask):
    members = {
        "dense": ("dense",), "pool2x2": ("maxpool2", "avgpool2"),
        "eltwise": ("eltwise",), "copy": ("copy",), "reduce": ("gap",),
        "resize": ("resize",), "pool5": ("pool5",),
        "activation": ("activation",), "depthwise": ("depthwise",), "maxpool2d": ("maxpool2d",),
        "int8_mul": ("mul",), "maxpool2": ("maxpool2",),
        "avgpool2": ("avgpool2",), "clip": ("clip",), "gap": ("gap",),
    }.get(name)
    if not members:
        return True
    return any((physical_mask >> PHYSICAL_BITS[m]) & 1 for m in members)


def _physical_instances(physical_mask, required_mask, usage):
    rows = {}
    for key, bit in sorted(PHYSICAL_BITS.items(), key=lambda kv: kv[1]):
        rows[key] = {
            "bit": bit,
            "gated": key in PHYSICAL_GATED,
            "required_by_graph": bool((required_mask >> bit) & 1),
            "instantiated": bool((physical_mask >> bit) & 1),
            "records": int(usage.get(key, 0)),
        }
    return rows


# Files whose presence in sources.f is controlled by the physical mask.  A plain
# source-level prune: unused cores are not even compiled, so the generated netlist
# input contains only the operators the graph needs.  maxpool2x2_rowbuf is shared
# by both pool2x2 cores, so it stays while either one is kept.
_FILE_GATES = (
    ("rtl/ops/maxpool2d/maxpool2d_row_store.sv", (12,)),
    ("rtl/ops/maxpool2d/maxpool2d_reduce8.sv", (12,)),
    ("rtl/ops/maxpool2d/maxpool2d_core.sv", (12,)),
    ("rtl/ops/pool5/maxpool5x5_core.v", (9,)),
    ("rtl/ops/pool2x2/maxpool2x2_core.v", (1,)),
    ("rtl/ops/pool2x2/maxpool2x2_lane.v", (1,)),
    ("rtl/ops/pool2x2/avgpool2x2_core.v", (6,)),
    ("rtl/ops/pool2x2/avgpool2x2_lane.v", (6,)),
    ("rtl/ops/pool2x2/maxpool2x2_rowbuf.v", (1, 6)),
    ("rtl/ops/eltwise/clip_core.v", (4,)),
    ("rtl/ops/eltwise/int8_mul_core.v", (7,)),
    ("rtl/ops/reduce/global_avg_pool_core.v", (5,)),
    ("rtl/ops/activation/activation_lut_core.sv", (10,)),
    ("rtl/ops/activation/activation_lut_bank.sv", (10,)),
    ("rtl/ops/depthwise/depthwise_conv3x3_core.sv", (11,)),
)


def _filter_rtl_sources(rtl_sources, physical_mask):
    kept, pruned = [], []
    for rel in rtl_sources:
        pos = rel.replace("\\", "/")
        gate = None
        for path, bits in _FILE_GATES:
            if pos == path:
                gate = bits
                break
        if gate is not None and not any((physical_mask >> bit) & 1 for bit in gate):
            pruned.append(rel)
        else:
            kept.append(rel)
    return kept, pruned


def _engine_usage(stream: Dict[str, Any]) -> Dict[str, bool]:
    used = {engine: False for engine in ALL_ENGINES}
    for record in stream.get("records") or []:
        engine = OPCODE_ENGINE.get(int(record.get("opcode", -1)))
        if engine is None:
            engine = OPNAME_ENGINE.get(str(record.get("op")))
        if engine in used:
            used[engine] = True
    return used


def _engine_gates(engine: engine_profiles.ResolvedEngineProfile,
                  capacity: Dict[str, int], stream: Dict[str, Any],
                  allocated_max_ch: int,
                  spatial_engines: Optional[Dict[str, Any]] = None,
                  physical_mask: Optional[int] = None) -> Dict[str, Any]:
    """Per-engine capacity/parallelism/evidence rows; never a verification claim.

    spatial_engines (v0.2) replaces the single global MAX_W row for the used
    engines and pins the unused ones to their historical small capacity, so an
    unused core can neither block a large graph nor be inflated by it.
    """
    used = _engine_usage(stream)
    gates: Dict[str, Any] = {}
    for name in ALL_ENGINES:
        row: Dict[str, Any] = {
            "used_by_graph": bool(used[name]),
            "instantiated": True,   # refined below from the physical mask
            "capacity": None,
            "parallelism": "fixed",
            "evidence": "compile_generated_rtl_only",
            "gates": {"functional": "not_run", "ooc": "not_run",
                      "platform": "not_run", "board": "not_run"},
        }
        if name == "dense":
            row["capacity"] = {"max_cin": int(capacity["max_cin"]),
                               "max_cout": int(capacity["max_cout"]),
                               "max_w": int(capacity["max_w"])}
            row["parallelism"] = {"tn": int(engine.dense["tn"]),
                                  "tm": int(engine.dense["tm"]),
                                  "mac_impl": int(engine.dense["mac_impl"])}
        elif name == "depthwise":
            row["capacity"] = {"max_ch": int(allocated_max_ch),
                               "max_w": int(capacity["max_w"])}
            row["parallelism"] = {"dw_pc": int(engine.depthwise["pc"]),
                                  "dw_tap_parallel":
                                      int(engine.depthwise["tap_parallel"]),
                                  "dw_rq_lanes": int(engine.depthwise["rq_lanes"])}
        elif name == "activation":
            row["capacity"] = {"max_ch": int(allocated_max_ch)}
            row["parallelism"] = {"kind": engine.activation["kind"],
                                  "lookup_lanes": 8}
        gates[name] = row
    if spatial_engines:
        for name, info in sorted(spatial_engines.items()):
            row = gates.setdefault(name, {
                "used_by_graph": False, "instantiated": True,
                "parallelism": "fixed", "evidence": "compile_generated_rtl_only",
                "gates": {"functional": "not_run", "ooc": "not_run",
                          "platform": "not_run", "board": "not_run"},
            })
            row["used_by_graph"] = bool(info.get("used_by_graph", False))
            row["expanded"] = bool(info.get("expanded", False))
            if name == "depthwise" and spatial_engines is not None:
                # This is a claim about the generated RTL, not the capability
                # table: it is only set when the instance is really elaborated away.
                row["instantiated"] = bool(info.get("used_by_graph", False))
            row["streaming"] = bool(info.get("streaming", False))
            capacity_row = {key: int(info[key]) for key in
                            ("max_h", "max_w", "max_ch", "max_cin", "max_cout",
                             "max_row_bytes", "max_in_row_bytes",
                             "out_max_h", "out_max_w", "out_max_row_bytes",
                             "table_bytes", "input_axis_limit")
                            if key in info}
            row["capacity"] = capacity_row
            row["evidence"] = "compile_generated_rtl_only"
            if info.get("reason"):
                row["reason"] = str(info["reason"])
            if info.get("note"):
                row["note"] = str(info["note"])
    effective_mask = 0xFFF if physical_mask is None else (int(physical_mask) & 0x1FFF)
    for name, row in gates.items():
        row["instantiated"] = _logical_instantiated(name, effective_mask)
    return gates


def write_graph_rtl(graph: Graph, plan: Dict[str, Any], out_dir: Path, *,
                    run_id: str = "e1_w5", name: str = "e1_graph",
                    tn: int = 8, tm: int = 8, mac_impl: int = 3,
                    stream: Optional[Dict[str, Any]] = None,
                    capacities: Optional[Dict[str, int]] = None,
                    compute_profile: Optional[Dict[str, Any]] = None,
                    engine_profile=None, target=None,
                    dw_pc: Optional[int] = None,
                    dw_tap_parallel: Optional[int] = None,
                    dw_rq_lanes: Optional[int] = None,
                    repo_root: Optional[Path] = None,
                    spatial_capacity: Optional[Dict[str, Any]] = None,
                    spatial_engines: Optional[Dict[str, Any]] = None,
                    spatial_rows: Optional[Dict[str, Any]] = None,
                    small_engine_max_w: int = 32,
                    engine_pruning: str = "none") -> Dict[str, Any]:
    # Universal safety guard: every ONNX generation entry rejects tn/tm outside
    # 1..16 instead of relying on the conv3x3_calc*/conv3x3_core lane clamps.
    validation.validate_legacy_tile(tn, tm)
    # Resolve the complete engine configuration.  An already-resolved profile is
    # used verbatim (compile_pipeline validated conflicts before calling codegen, and
    # re-resolving would drop the original requested/provenance record);
    # otherwise this is the compatibility path whose legacy tn/tm/mac_impl
    # become the dense base and DW values default to dw_c8_t1_q1.
    if isinstance(engine_profile, engine_profiles.ResolvedEngineProfile):
        resolved_engine = engine_profile
    else:
        resolved_engine = engine_profiles.resolve(
            engine_profile,
            base_dense={"tn": int(tn), "tm": int(tm), "mac_impl": int(mac_impl)},
            dw_pc=dw_pc, dw_tap_parallel=dw_tap_parallel, dw_rq_lanes=dw_rq_lanes)
    resolved_target = (target if isinstance(target, targets.ResolvedTarget)
                       else targets.resolve(target))
    tn = int(resolved_engine.dense["tn"])
    tm = int(resolved_engine.dense["tm"])
    mac_impl = int(resolved_engine.dense["mac_impl"])
    dw_pc = int(resolved_engine.depthwise["pc"])
    dw_tap_parallel = int(resolved_engine.depthwise["tap_parallel"])
    dw_rq_lanes = int(resolved_engine.depthwise["rq_lanes"])
    resolved_root = resolve_repo_root(repo_root)
    source_root = resolved_root
    identity = source_identity(resolved_root)
    template_file = template_path_for(resolved_root)
    rtl_sources = identity["rtl_sources"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.finalize()
    if stream is None:
        stream = generate_stream(graph, plan, run_id=run_id)
    required_engines = {
        "ActivationLUT": "rtl/ops/activation/activation_lut_core.sv",
        "DepthwiseConv": "rtl/ops/depthwise/depthwise_conv3x3_core.sv",
    }
    for record in stream["records"]:
        required = ("rtl/ops/maxpool2d/maxpool2d_core.sv"
                    if int(record.get("opcode", -1)) == 0x001D
                    else required_engines.get(record["op"]))
        if required and required not in rtl_sources:
            raise CompileError(E_CFG, "selected source tree does not implement %s" % record["op"],
                               detail={"source_root": str(resolved_root), "missing": required})
    artifacts = write_stream(stream, out, name=name)
    if capacities is None:
        cap = _capacities(graph, tn, tm)
    else:
        cap = {"tn": int(tn), "tm": int(tm)}
        cap.update({key: int(capacities[key])
                    for key in ("max_cin", "max_cout", "max_ch", "max_w")})
        for _key in ("max_h", "dense_max_in_row_bytes", "dense_max_out_row_bytes",
                     "dense_max_weight_tiles", "dense_weight_read_ports"):
            if _key in capacities:
                cap[_key] = int(capacities[_key])
    requested_capacity = dict(cap)
    # v0.2 affine dense capacity: real row spans / tile product instead of
    # MAX_C*MAX_W and MAX_ICB*MAX_OCB (GAP-18/GAP-22).  A legacy build has none
    # of these keys, so every emitted value stays 0 (= legacy derivation).
    dense_line_row = int(cap["dense_max_in_row_bytes"]) \
        if (capacities is not None and "dense_max_in_row_bytes" in capacities) else 0
    dense_out_row = int(cap["dense_max_out_row_bytes"]) \
        if (capacities is not None and "dense_max_out_row_bytes" in capacities) else 0
    dense_tiles = int(cap["dense_max_weight_tiles"]) \
        if (capacities is not None and "dense_max_weight_tiles" in capacities) else 0
    dense_ports = int(cap["dense_weight_read_ports"]) \
        if (capacities is not None and "dense_weight_read_ports" in capacities) else 0
    cap = allocate_capacity(cap, dw_pc)
    if "max_h" in requested_capacity:
        cap["max_h"] = int(requested_capacity["max_h"])
    allocated_max_ch = int(cap["max_ch"])
    spatial = bool(spatial_capacity)
    max_h_param = int(cap.get("max_h", 0)) if spatial else 0
    pool_engines = (spatial_engines or {}).get("pool2x2", {})
    pool_max_w = int(pool_engines.get("max_w", 0)) if spatial else 0
    pool_max_h = int(pool_engines.get("max_h", 0)) if spatial else 0
    small_w = int(small_engine_max_w)
    # v0.2 batch 2: per-engine capacity for the feature-fusion families.  A
    # legacy build keeps every new parameter at 0 (= inherit the historical
    # MAX_W/MAX_H resolution inside the router), so its generated top is
    # unchanged.  A spatial build passes the capacity the engine really touches;
    # unused engines get the small historical capacity recorded in the manifest.
    def _engine_cap(name: str, key: str, fallback: int = 0) -> int:
        if not spatial:
            return 0
        row = (spatial_engines or {}).get(name) or {}
        value = row.get(key)
        return int(fallback if value is None else value)

    resize_max_w = _engine_cap("resize", "max_w", small_w)
    resize_max_h = _engine_cap("resize", "max_h", small_w)
    pool5_max_w = _engine_cap("pool5", "max_w", small_w)
    pool5_max_h = _engine_cap("pool5", "max_h", small_w)

    def _al8(v: int) -> int:
        return ((int(v) + 7) // 8) * 8

    def _al16(v: int) -> int:
        return ((int(v) + 15) // 16) * 16
    # Real per-engine channel / row capacity (v0.2 M3).  A legacy build keeps
    # these at 0 (= historical inheritance); an affine build passes the real
    # maxima so the resize/pool5 line buffers are not sized by MAX_CH*MAX_W.
    resize_max_ch = _engine_cap("resize", "max_ch", 0)
    pool5_max_ch = _engine_cap("pool5", "max_ch", 0)
    # Resize is nearest x2 only (frozen): the engine max_w is the *output*
    # width, so the real input row is half of it.  The old product form gave
    # 2x the real input line buffer.
    resize_in_w = max(1, resize_max_w // 2) if resize_max_w > 0 else 0
    resize_in_row = (_al16(_al8(resize_max_ch) * resize_in_w)
                     if (spatial and resize_max_ch > 0 and resize_in_w > 0) else 0)
    resize_out_row = (2 * resize_in_row) if resize_in_row > 0 else 0
    if spatial and resize_max_ch > 0:
        pool5_row = _al16(_al8(pool5_max_ch) * pool5_max_w)
    else:
        pool5_row = 0
    act_max_w = _engine_cap("activation", "max_w", small_w)
    act_max_h = _engine_cap("activation", "max_h", small_w)
    act_max_ch = _engine_cap("activation", "max_ch", int(cap["max_ch"]))
    copy_row_bytes = _engine_cap("copy", "max_row_bytes", 512)
    copy_max_h = _engine_cap("copy", "max_h", small_w)
    copy_max_w = _engine_cap("copy", "max_w", small_w)
    copy_max_ch = _engine_cap("copy", "max_ch", 16)
    # v0.2: an unused, not-yet-closed engine must not block a large graph.  Only
    # an engine whose capability row says used_by_graph=false is elaborated away
    # (ELABORATE gate); the manifest then records instantiated=false.
    depthwise_enabled = True
    if spatial:
        depthwise_row = (spatial_engines or {}).get("depthwise") or {}
        depthwise_enabled = bool(depthwise_row.get("used_by_graph", False))
    # v0.3 physical-instance pruning: derive the required physical mask from the
    # lowered opcodes (authoritative), then gate unused cores at elaboration.
    if engine_pruning not in ("none", "graph"):
        raise CompileError(E_CFG, "engine pruning policy must be 'none' or 'graph'",
                           detail={"engine_pruning": str(engine_pruning)})
    physical = _physical_required_mask(stream, strict=(engine_pruning == "graph"))
    required_mask = int(physical["required_mask"])
    from .maxpool2d_geometry import CAPABILITY_V1, capacity as pool_capacity, decode_payload
    pool_records = [r for r in stream["records"] if int(r["opcode"]) == 0x001D]
    pool_geometries = [decode_payload(r["payload"])[0] for r in pool_records]
    pool2d_capacity = (pool_capacity(pool_geometries) if pool_geometries else
                       {"max_in_row_bytes": 16, "max_out_row_bytes": 16,
                        "max_eh": 1, "cache_rows": 1, "logical_storage_bytes": 0})
    allowed_mask = (0xFFF if depthwise_enabled else 0x7FF) | (required_mask & 0x1000)
    if engine_pruning == "graph":
        missing = required_mask & ~allowed_mask
        if missing:
            raise CompileError(E_CFG, "graph requires physical engine(s) disabled by the build",
                               detail={"required_mask": required_mask,
                                       "allowed_mask": allowed_mask})
        physical_mask = (required_mask | STRUCTURAL_MASK) & allowed_mask
    else:
        physical_mask = allowed_mask
    if engine_pruning == "graph":
        kept_sources, pruned_sources = _filter_rtl_sources(rtl_sources, physical_mask)
    else:
        # 'none' keeps the historical full source list (byte-identical old builds).
        kept_sources = [s for s in rtl_sources if (required_mask & 0x1000) or
                        not s.startswith("rtl/ops/maxpool2d/")]
        pruned_sources = [s for s in rtl_sources if s not in kept_sources]
    template = template_file.read_text(encoding="utf-8")
    comments = []
    for record in stream["records"]:
        comments.append("// record %d: %s opcode=0x%04X payload_words=%d" %
                        (record["index"], record["op"], record["opcode"],
                         record["payload_words"]))
    top = template
    generated_parameters = {
        "DW_PC": dw_pc,
        "DW_TAP_PARALLEL": dw_tap_parallel,
        "DW_RQ_LANES": dw_rq_lanes,
        "TN": int(cap["tn"]),
        "TM": int(cap["tm"]),
        "MAX_CIN": int(cap["max_cin"]),
        "MAX_COUT": int(cap["max_cout"]),
        "MAX_CH": allocated_max_ch,
        "MAX_W": int(cap["max_w"]),
        # v0.2: 0 means "inherit MAX_W/MAX_H" (byte-identical legacy RTL).
        "MAX_H": int(max_h_param),
        "DENSE_LINE_ROW_BYTES": int(dense_line_row),
        "DENSE_OUT_ROW_BYTES": int(dense_out_row),
        "DENSE_MAX_WEIGHT_TILES": int(dense_tiles),
        "DENSE_WEIGHT_READ_PORTS": int(dense_ports),
        "POOL_MAX_W": int(pool_max_w),
        "POOL_MAX_H": int(pool_max_h),
        "RESIZE_MAX_W": int(resize_max_w),
        "RESIZE_MAX_H": int(resize_max_h),
        "RESIZE_MAX_CH": int(resize_max_ch),
        "RESIZE_MAX_IN_ROW_BYTES": int(resize_in_row),
        "RESIZE_MAX_OUT_ROW_BYTES": int(resize_out_row),
        "POOL5_MAX_W": int(pool5_max_w),
        "POOL5_MAX_H": int(pool5_max_h),
        "POOL5_MAX_CH": int(pool5_max_ch),
        "POOL5_MAX_ROW_BYTES": int(pool5_row),
        "POOL2D_MAX_IN_ROW_BYTES": pool2d_capacity["max_in_row_bytes"],
        "POOL2D_MAX_OUT_ROW_BYTES": pool2d_capacity["max_out_row_bytes"],
        "POOL2D_MAX_EH": pool2d_capacity["max_eh"],
        "POOL2D_CACHE_ROWS": pool2d_capacity["cache_rows"],
        "ACT_MAX_W": int(act_max_w),
        "ACT_MAX_H": int(act_max_h),
        "ACT_MAX_CH": int(act_max_ch),
        "COPY_MAX_ROW_BYTES": int(copy_row_bytes),
        "COPY_MAX_H": int(copy_max_h),
        "COPY_MAX_W": int(copy_max_w),
        "COPY_MAX_CH": int(copy_max_ch),
        "DW_MAX_W": (small_w if spatial else 0),
        "ENABLE_DEPTHWISE": int(depthwise_enabled),
        "PHYSICAL_ENGINE_MASK": int(physical_mask),
        "MAC_IMPL": mac_impl,
        "WEIGHT_MODE": 0,
        "CLAMP_MIN_EN": 1,
    }
    replacements = {
        "<NET_NAME>": name,
        "<NUM_LAYERS>": str(len(stream["records"])),
        "<RUN_ID>": run_id,
        "<GEN_VERSION>": GEN_VERSION,
        "<TN>": str(cap["tn"]),
        "<TM>": str(cap["tm"]),
        "<MAX_CIN>": str(cap["max_cin"]),
        "<MAX_COUT>": str(cap["max_cout"]),
        "<MAX_CH>": str(cap["max_ch"]),
        "<MAX_W>": str(cap["max_w"]),
        "<MAX_H>": str(int(max_h_param)),
        "<DENSE_LINE_ROW_BYTES>": str(int(dense_line_row)),
        "<DENSE_OUT_ROW_BYTES>": str(int(dense_out_row)),
        "<DENSE_MAX_WEIGHT_TILES>": str(int(dense_tiles)),
        "<DENSE_WEIGHT_READ_PORTS>": str(int(dense_ports)),
        "<POOL_MAX_W>": str(int(pool_max_w)),
        "<POOL_MAX_H>": str(int(pool_max_h)),
        "<RESIZE_MAX_W>": str(int(resize_max_w)),
        "<RESIZE_MAX_H>": str(int(resize_max_h)),
        "<RESIZE_MAX_CH>": str(int(resize_max_ch)),
        "<RESIZE_MAX_IN_ROW_BYTES>": str(int(resize_in_row)),
        "<RESIZE_MAX_OUT_ROW_BYTES>": str(int(resize_out_row)),
        "<POOL5_MAX_W>": str(int(pool5_max_w)),
        "<POOL5_MAX_H>": str(int(pool5_max_h)),
        "<POOL5_MAX_CH>": str(int(pool5_max_ch)),
        "<POOL2D_MAX_IN_ROW_BYTES>": str(pool2d_capacity["max_in_row_bytes"]),
        "<POOL2D_MAX_OUT_ROW_BYTES>": str(pool2d_capacity["max_out_row_bytes"]),
        "<POOL2D_MAX_EH>": str(pool2d_capacity["max_eh"]),
        "<POOL2D_CACHE_ROWS>": str(pool2d_capacity["cache_rows"]),
        "<ACT_MAX_W>": str(int(act_max_w)),
        "<ACT_MAX_H>": str(int(act_max_h)),
        "<ACT_MAX_CH>": str(int(act_max_ch)),
        "<COPY_MAX_ROW_BYTES>": str(int(copy_row_bytes)),
        "<COPY_MAX_H>": str(int(copy_max_h)),
        "<COPY_MAX_W>": str(int(copy_max_w)),
        "<COPY_MAX_CH>": str(int(copy_max_ch)),
        "<DW_MAX_W>": str(small_w if spatial else 0),
        "<ENABLE_DEPTHWISE>": str(int(depthwise_enabled)),
        "<PHYSICAL_ENGINE_MASK>": str(int(physical_mask)),
        "<DW_PC>": str(dw_pc),
        "<DW_TAP_PARALLEL>": str(dw_tap_parallel),
        "<DW_RQ_LANES>": str(dw_rq_lanes),
        "<CLAMP_MIN_EN>": "1",
        "<INSTR_WORDS>": str(stream["instr_words"]),
        "<INSTR_ADDR_W>": str(stream["instr_addr_w"]),
        "<INSTR_INIT>": stream["mem_file"],
        "<WEIGHT_MODE>": "0",
        "<MAC_IMPL>": str(mac_impl),
        "// <LAYER_COMMENTS>": "\n".join(comments),
    }
    for key, value in replacements.items():
        top = top.replace(key, value)
    for index in range(9):
        top = top.replace("<LOCAL_W%d>" % index, "0")
    if "<" in top and ">" in top:
        # Local weight placeholders are all handled; flag any accidental one.
        for token in ("<NET_NAME>", "<NUM_LAYERS>", "<INSTR_WORDS>", "<INSTR_INIT>",
                      "<DW_PC>", "<DW_TAP_PARALLEL>", "<DW_RQ_LANES>",
                  "<DENSE_LINE_ROW_BYTES>", "<DENSE_OUT_ROW_BYTES>",
                  "<DENSE_MAX_WEIGHT_TILES>", "<DENSE_WEIGHT_READ_PORTS>",
                      "<RESIZE_MAX_W>", "<RESIZE_MAX_H>", "<RESIZE_MAX_CH>",
                      "<RESIZE_MAX_IN_ROW_BYTES>", "<RESIZE_MAX_OUT_ROW_BYTES>",
                      "<POOL5_MAX_W>", "<POOL5_MAX_H>", "<POOL5_MAX_CH>",
                      "<ACT_MAX_W>", "<ACT_MAX_H>",
                      "<ACT_MAX_CH>", "<COPY_MAX_ROW_BYTES>", "<COPY_MAX_H>",
                      "<COPY_MAX_W>", "<COPY_MAX_CH>"):
            if token in top:
                raise CompileError(E_CFG, "template placeholder %s remains" % token)
    top_path = out / "npu_top.v"
    top_path.write_text(top, encoding="utf-8")
    sources_path = out / "sources.f"
    lines = ["# sources.f: E1 graph generated (%s)" % run_id, "-i rtl/common"]
    lines.extend(kept_sources)
    try:
        # Relative-path entries only: simulation runners resolve "rtl/..." lines
        # against SourceRoot and this generated top against the project root.
        top_relative = top_path.relative_to(ROOT).as_posix()
    except ValueError:
        top_relative = top_path.resolve().as_posix()
    lines.append(top_relative)
    sources_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema": "e1-graph-rtl-manifest-1",
        "run_id": run_id,
        "name": name,
        "graph_digest": graph.digest(),
        "capacities": cap,
        "records": len(stream["records"]),
        "opcodes": [record["opcode"] for record in stream["records"]],
        "quant_allocations": stream["quant_allocations"],
        "instr_words": stream["instr_words"],
        "instr_addr_w": stream["instr_addr_w"],
        "mem_file": stream["mem_file"],
        "instruction_stream": artifacts["instruction_stream"],
        "mem": artifacts["mem"],
        "npu_top": str(top_path),
        "sources_f": str(sources_path),
        "files_sha256": {str(top_path): _sha(top_path),
                         str(sources_path): _sha(sources_path),
                         artifacts["instruction_stream"]: _sha(Path(artifacts["instruction_stream"])),
                         artifacts["mem"]: _sha(Path(artifacts["mem"]))},
        "generated_parameters": generated_parameters,
        "maxpool2d": {"capability": dict(CAPABILITY_V1), "capacity": pool2d_capacity,
                      "opcode": "0x001D", "backend": "generic",
                      "instantiated": bool(physical_mask & 0x1000),
                      "qualification": {"functional": "not_run", "ooc": "not_run",
                                        "ip_package": "not_run"}},
        **({"spatial_capacity": spatial_capacity,
            "engine_capacity": spatial_engines,
            "engine_row_capacity": spatial_rows}
           if spatial else {}),
        "engine_profile": resolved_engine.manifest_document(),
        "target": resolved_target.manifest_document(),
        "dense": {"tn": tn, "tm": tm, "mac_impl": mac_impl},
        "depthwise": {
            "dw_pc": dw_pc,
            "dw_tap_parallel": dw_tap_parallel,
            "dw_rq_lanes": dw_rq_lanes,
            "capacity_max_ch": allocated_max_ch,
            "requested_capacity_max_ch": int(requested_capacity["max_ch"]),
            "allocated_lanes": {"mac": dw_pc, "rq": dw_rq_lanes},
            "ngroups_formula": "Ng = Hout*Wout*ceil(C/DW_PC)",
            "capacity_rule": "MAX_CH = max(graph-derived MAX_CH, align8(DW_PC))",
            "dw_used_by_graph": bool(_engine_usage(stream)["depthwise"]),
            "logical_channel_note": "DW_PC is a compute parallelism; it is not the "
                                    "logical channel count C of any graph layer",
        },
        "engine_gates": _engine_gates(resolved_engine, cap, stream,
                                      allocated_max_ch,
                                      spatial_engines if spatial else None,
                                      physical_mask),
        "pruning": {
            "policy": engine_pruning,
            "mask_schema": "e1-physical-engine-mask-1",
            "bit_assignments": dict(PHYSICAL_BITS),
            "required_mask": required_mask,
            "allowed_mask": allowed_mask,
            "effective_mask": physical_mask,
            "unknown_opcodes": physical["unknown"],
        },
        "physical_instances": _physical_instances(
            physical_mask, required_mask, physical["usage"]),
        "sources_f_pruned": pruned_sources,
        "qualification": "compile-generated RTL only; no functional/OOC/platform/"
                         "board claim is made by code generation",
    }
    if compute_profile is not None:
        # Explicit compute-profile builds only: the legacy/default manifest keeps
        # its historical field set byte-for-byte.
        manifest["compute_profile"] = compute_profile
    manifest.update(identity)
    manifest["new_core_sources"] = [rel for rel in kept_sources
                                    if rel.startswith(("rtl/ops/activation/",
                                                       "rtl/ops/depthwise/",
                                                       "rtl/ops/copy/",
                                                       "rtl/ops/resize/",
                                                       "rtl/ops/pool5/",
                                                       "rtl/ops/maxpool2d/"))]
    write_binding(out, identity, [top_path, sources_path,
                                 Path(artifacts["instruction_stream"]), Path(artifacts["mem"])])
    for path in (out / "source_root.txt", out / "source_hashes.tsv"):
        manifest["files_sha256"][str(path)] = _sha(path)
    (out / "rtl_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest
