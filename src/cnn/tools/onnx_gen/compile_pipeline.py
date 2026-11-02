"""End-to-end E1-P2 static graph compiler."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Tuple

from . import compute_profiles, engine_profiles, reference_cache as reference_cache_module
from . import spatial_capacity, targets
from .graph_ir import (CompileError, E_CFG, E_CP_CONFLICT, E_CP_OUTPUT,
                       E_CP_SCHEMA, E_CP_TARGET, E_SPATIAL_MEMORY)
from .graph_planner import build_graph_plan
from .graph_reference import evaluate_graph
from .instr_codegen import generate_stream
import numpy as np
from .layout import build_layout_plan
from .lowering import lower_graph
from .memory_planner import build_memory_plan
from .onnx_frontend import load_onnx
from .parameter_pack import (load_initializers, load_input_manifest, pack_activation,
                             pack_conv_bias, pack_conv_quant, pack_conv_weights,
                             pack_depthwise_bias, pack_depthwise_quant,
                             pack_depthwise_weights, sha256_bytes)
from .quant_contract import apply_contract
from .verilog_codegen import (allocate_capacity, derive_capacities,
                              derive_spatial_capacities, write_graph_rtl)

SCHEMA = "e1-p2-compile-manifest-1"
MAX_SIM_DDR_BYTES = 128 * 1024
#: affine-int8-v1 builds carry the real 640 model weights/image; the legacy tool
#: image limit is not a device DDR budget.
AFFINE_SIM_DDR_LIMIT_MIB = 256
# v0.2 spatial builds use the user-configured --sim-ddr-limit-mib budget; the
# legacy 128 KiB is a *tool* image limit, not a device DDR limit.
SPATIAL_SCHEMA = spatial_capacity.SCHEMA

# Module-private sentinel: distinguishes "argument not supplied" (legacy
# defaults 8/8/3) from an explicit value (which must agree with a resolved
# compute profile instead of being silently overridden).
_UNSET = object()

# Manifest keys derived by the compiler; explicit-compute-profile builds refuse
# extra_manifest overrides for these (technical plan section 4.1/5).
_MANIFEST_IDENTITY_KEYS = frozenset({
    "schema", "status", "profile", "compute_profile", "graph_name",
    "model_sha256", "quant_contract_sha256", "input_manifest_sha256",
    "source_graph_digest", "lowered_graph_digest", "quant", "lowering",
    "graph_plan", "layout_plan", "memory_plan", "parameter_manifest",
    "instruction_stream", "instruction_mem", "instruction_mem_sha256",
    "generated_rtl_manifest", "ddr_init", "ddr_init_sha256",
    "snapshot_manifest", "node_outputs", "records", "gates", "run_id",
    "source_root", "source_digest", "source_files", "activation_contract_sha256",
    "activation_deployment", "engine_profile", "target", "dense", "depthwise",
    "generated_parameters", "engine_gates", "pruning", "physical_instances",
    # v0.2 spatial-capacity identity: these are compiler-derived, never
    # overridable through extra_manifest.
    "spatial_capacity", "engine_capacity", "memory_limits", "simulation_budget",
    "generated", "tb_params",
    # explicit reference dependency cache evidence (task book 7): derived, never
    # overridable through extra_manifest.
    "reference_cache",
})


def _resolve_compute_profile(compute_profile, tn, tm, mac_impl):
    """Return (resolved_or_None, tn, tm, mac_impl, explicit_dense) for the two
    dense-tier branches.

    compute_profile is None keeps the historical legacy branch: defaults 8/8/3,
    explicit mac_impl 1/3 accepted, and the only tightened rule is that tn/tm
    outside 1..16 are rejected (E_CP_UNSUPPORTED_TILE) instead of being clamped
    by the RTL.  An explicit compute profile validates the name and any explicit
    legacy parameters first; capacity/geometry checks happen later, after
    lowering, but still before any compile artifact is written.

    explicit_dense lists the dense fields that came from a user-explicit source
    (an explicit compute profile counts for all three).  None values are treated
    as "not supplied" so callers such as the compile CLI never inject a default
    that could silently beat an explicit engine profile.
    """
    explicit_tn = None if tn is _UNSET or tn is None else tn
    explicit_tm = None if tm is _UNSET or tm is None else tm
    explicit_mac = None if mac_impl is _UNSET or mac_impl is None else mac_impl
    resolved = compute_profiles.resolve(compute_profile, tn=explicit_tn,
                                        tm=explicit_tm, mac_impl=explicit_mac)
    if resolved is not None:
        return (resolved, resolved.tn, resolved.tm, resolved.mac_impl,
                {"tn", "tm", "mac_impl"})
    legacy_tn = 8 if explicit_tn is None else explicit_tn
    legacy_tm = 8 if explicit_tm is None else explicit_tm
    legacy_mac = 3 if explicit_mac is None else explicit_mac
    if int(legacy_mac) not in (1, 3):
        # P3 keeps MAC_IMPL=3 for the delivered candidate.  MAC_IMPL=1 (the
        # behavioral int8 multiplier path) is accepted for functional RTL
        # replay only, where the two backends are already proven equivalent.
        raise CompileError(E_CFG, "mac_impl must be 1 (behavioral) or 3 (DSP)")
    legacy_tn, legacy_tm = compute_profiles.validate_legacy_tile(legacy_tn, legacy_tm)
    explicit_dense = set()
    if explicit_tn is not None:
        explicit_dense.add("tn")
    if explicit_tm is not None:
        explicit_dense.add("tm")
    if explicit_mac is not None:
        explicit_dense.add("mac_impl")
    return None, int(legacy_tn), int(legacy_tm), int(legacy_mac), explicit_dense


def _check_extra_manifest(extra_manifest) -> None:
    if extra_manifest is None:
        return
    if not isinstance(extra_manifest, dict):
        raise CompileError(E_CP_SCHEMA, "extra_manifest must be a dict")
    bad = sorted(set(extra_manifest) & _MANIFEST_IDENTITY_KEYS)
    if bad:
        raise CompileError(E_CP_CONFLICT,
                           "extra_manifest may not override derived identity fields",
                           detail={"fields": bad})


def _widen_depthwise_capacity(graph, capacities: Dict[str, int]) -> Dict[str, int]:
    """Make generated MAX_* capacity cover DepthwiseConv layers as well.

    derive_capacities only inspects dense Conv nodes; a depthwise-only or mixed
    graph must still instantiate the pipeline with enough channel lanes and
    row width for its 0x001C records.  Dense-only graphs are untouched (the
    historical capacity derivation stays byte-identical).
    """
    depthwise = [node for node in graph.nodes if node.op_type == "DepthwiseConv"]
    if not depthwise:
        return capacities
    widened = dict(capacities)
    max_c = 1
    max_hw = 1
    for node in depthwise:
        out_shape = graph.tensors[node.outputs[0]].shape
        in_shape = graph.tensors[node.inputs[0]].shape
        max_c = max(max_c, int(out_shape[1]))
        max_hw = max(max_hw, int(in_shape[2]), int(in_shape[3]))
    widened["max_cin"] = max(int(widened["max_cin"]), max_c)
    widened["max_cout"] = max(int(widened["max_cout"]), max_c)
    widened["max_ch"] = max(int(widened["max_ch"]), ((max_c + 7) // 8) * 8)
    widened["max_w"] = max(int(widened["max_w"]), max_hw)
    return widened


def _write_graph_rtl_with_repo_root(graph, plan, generated, *, repo_root, **kwargs):
    """Pass repo_root to verilog_codegen only when the live module supports it.

    rtl-int owns that additive parameter.  An explicit depthwise build needs the
    candidate source root so that new core files appear in sources.f; failing
    loudly is safer than emitting a compile directory that cannot be simulated.
    """
    rtl_root = Path(repo_root) if repo_root is not None else None
    if rtl_root is not None:
        try:
            parameters = inspect.signature(write_graph_rtl).parameters
        except (TypeError, ValueError):
            parameters = {}
        if "repo_root" not in parameters:
            raise CompileError(
                E_CFG,
                "verilog_codegen.write_graph_rtl has no repo_root support yet "
                "(rtl-int Phase A); explicit depthwise compile requires it so that "
                "sources.f references the new-run core files")
        kwargs["repo_root"] = rtl_root
    return write_graph_rtl(graph, plan, generated, **kwargs)


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _sha_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _put(image: bytearray, base: int, address: str, blob: bytes, label: str,
         intents: List[Dict[str, Any]]) -> None:
    offset = int(str(address), 16) - base
    if offset < 0 or offset + len(blob) > len(image):
        raise CompileError(E_CFG, "%s write escapes DDR image" % label)
    for previous in intents:
        if offset < previous["offset"] + previous["bytes"] and previous["offset"] < offset + len(blob):
            raise CompileError(E_CFG, "%s overlaps %s" % (label, previous["label"]))
    image[offset:offset + len(blob)] = blob
    intents.append({"label": label, "address": address, "offset": offset,
                    "bytes": len(blob), "sha256": sha256_bytes(blob)})


def _archive_inputs(compile_dir: Path, model_path: Path, contract_path: Path,
                    input_manifest_path: Path, input_manifest: Dict[str, Any]) -> None:
    model_dir = compile_dir / "model"
    _copy_file(model_path, model_dir / "model.onnx")
    _copy_file(contract_path, model_dir / "quant_contract.json")
    _copy_file(input_manifest_path, model_dir / "input_manifest.json")
    for name, row in input_manifest["inputs"].items():
        _copy_file(Path(row["path"]), model_dir / "input_data" / (name + ".bin"))

def compile_graph(model_path: Path, contract_path: Path, input_manifest_path: Path,
                  out_dir: Path, *, profile: str = "no_reuse", mac_impl=_UNSET,
                  tn=_UNSET, tm=_UNSET, compute_profile=None,
                  name: str | None = None,
                  run_id: str = "e1p2", schema: str = SCHEMA,
                  extra_manifest: Dict[str, Any] | None = None,
                  repo_root: Path | None = None,
                  activation_contract_path: Path | None = None,
                  engine_profile=None, target=None,
                  dw_pc=None, dw_tap_parallel=None,
                  dw_rq_lanes=None,
                  max_feature_width=None, max_feature_height=None,
                  sim_ddr_limit_mib=None,
                  reference_cache=None,
                  numeric_profile: str | None = None,
                  fragment_nodes=None,
                  fragment_input_rows=None,
                  input_u8=None,
                  engine_pruning: str = "none", maxpool_backend: str = "auto") -> Dict[str, Any]:
    # Stage timing for the cold/hot evidence (task book 7.3).  The reference
    # dependency cache is explicit and off by default: passing
    # reference_cache=None keeps the historical behaviour and manifest exactly.
    _compile_started = time.perf_counter()
    _stage_clock = [_compile_started]
    stage_timings: Dict[str, float] = {}

    def _stage(name: str) -> None:
        now = time.perf_counter()
        stage_timings[name] = round(now - _stage_clock[0], 4)
        _stage_clock[0] = now

    if profile not in ("no_reuse", "first_fit"):
        raise CompileError(E_CFG, "compile profile must be no_reuse or first_fit")
    # v0.2 spatial capacity: parsed before the frontend so the shape-inference
    # ceiling already matches the build the user asked for.  None keeps the
    # historical behaviour byte-for-byte.
    spatial_request = spatial_capacity.parse_request(
        max_feature_width=max_feature_width,
        max_feature_height=max_feature_height,
        sim_ddr_limit_mib=sim_ddr_limit_mib)
    spatial_infer_caps = (None if spatial_request is None
                          else spatial_capacity.shape_infer_caps(spatial_request))
    resolved, tn, tm, mac_impl, dense_explicit = _resolve_compute_profile(
        compute_profile, tn, tm, mac_impl)
    # New-style per-engine configuration.  Defaults are never treated as an
    # explicit selection: only an explicit engine profile/target/dw_* argument
    # can conflict with an explicit dense tier (and then it is rejected).
    engine_requested = (engine_profile is not None or target is not None or
                        dw_pc is not None or dw_tap_parallel is not None or
                        dw_rq_lanes is not None or spatial_request is not None)
    resolved_engine = engine_profiles.resolve(
        engine_profile,
        base_dense={"tn": int(tn), "tm": int(tm), "mac_impl": int(mac_impl)},
        explicit_dense=dense_explicit,
        dw_pc=dw_pc, dw_tap_parallel=dw_tap_parallel, dw_rq_lanes=dw_rq_lanes)
    resolved_target = targets.resolve(target)
    if resolved is not None:
        if resolved.part != resolved_target.part:
            raise CompileError(E_CP_TARGET,
                               "compute profile part disagrees with the target part",
                               detail={"profile_part": resolved.part,
                                       "target": resolved_target.id,
                                       "target_part": resolved_target.part})
        if float(resolved.clock_period_ns) != float(resolved_target.clock_ns):
            raise CompileError(E_CP_TARGET,
                               "compute profile clock disagrees with the target clock",
                               detail={"profile_clock_ns": resolved.clock_period_ns,
                                       "target": resolved_target.id,
                                       "target_clock_ns": resolved_target.clock_ns})
    tn = int(resolved_engine.dense["tn"])
    tm = int(resolved_engine.dense["tm"])
    mac_impl = int(resolved_engine.dense["mac_impl"])
    if resolved is not None or engine_requested:
        _check_extra_manifest(extra_manifest)
    model_path = Path(model_path).resolve()
    contract_path = (Path(contract_path).resolve()
                     if contract_path is not None else None)
    input_manifest_path = (Path(input_manifest_path).resolve()
                           if input_manifest_path is not None else None)
    compile_dir = Path(out_dir).resolve()
    explicit_output = (resolved is not None or engine_requested or
                       numeric_profile is not None)
    if not explicit_output:
        compile_dir.mkdir(parents=True, exist_ok=True)
    elif compile_dir.exists():
        # Explicit compute-profile builds are reserved by the caller and must
        # start from a fresh, empty compile directory (no silent overwrite).
        if not compile_dir.is_dir() or any(compile_dir.iterdir()):
            raise CompileError(E_CP_OUTPUT,
                               "explicit engine/compute-profile builds require a "
                               "fresh empty compile directory",
                               detail={"compile_dir": str(compile_dir)})
    affine = None
    affine_capacity = None
    if numeric_profile is not None:
        from . import affine_frontend
        if numeric_profile != "affine-int8-v1":
            raise CompileError(E_CFG, "unknown numeric profile %r" % (numeric_profile,))
        if activation_contract_path is not None:
            raise CompileError(E_CFG,
                               "an affine numeric profile build cannot also take a "
                               "legacy activation contract")
        compile_dir.mkdir(parents=True, exist_ok=True)
        affine = affine_frontend.prepare(model_path, fragment=fragment_nodes,
                                         input_u8=input_u8,
                                         input_rows=fragment_input_rows)
        # affine builds hand the shape-inference pass their versioned ceiling: the
        # legacy C<=64 / W<=256 / C16 identity gates are not model properties.
        spatial_infer_caps = affine_frontend.shape_caps()
        input_manifest_path = compile_dir / "inputs" / "input_manifest.json"
        affine_frontend.write_input_manifest(input_manifest_path, affine["input_values"])
        contract_path = compile_dir / "model" / "affine_contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(
            json.dumps(affine["contract"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    model_sha = _sha_file(model_path)
    activation_contract = None
    if activation_contract_path is not None:
        activation_contract_path = Path(activation_contract_path).resolve()
        activation_contract = json.loads(activation_contract_path.read_text(encoding="utf-8-sig"))
        if activation_contract.get("model_sha256") not in (None, model_sha):
            raise CompileError(E_CFG, "activation contract model_sha256 does not match model")
    if affine is not None:
        source = affine["graph"]
    else:
        source = (load_onnx(model_path, name=name, spatial_caps=spatial_infer_caps)
                  if activation_contract is None else
                  load_onnx(model_path, name=name, activation_contract=activation_contract,
                            spatial_caps=spatial_infer_caps))
    if (source.source or {}).get("model_sha256") != model_sha:
        raise CompileError(E_CFG, "frontend model digest mismatch")
    contract = (affine["contract"] if affine is not None
                else json.loads(contract_path.read_text(encoding="utf-8")))
    if activation_contract is not None:
        for tensor_name, row in activation_contract.get("tensors", {}).items():
            quant_row = contract.get("tensors", {}).get(tensor_name, {})
            if (row.get("dtype", "int8") != "int8" or
                    int(row.get("zero_point", 0)) != 0 or
                    row.get("materialized", True) is not True or
                    float(row.get("scale")) != float(quant_row.get("scale", -1))):
                raise CompileError(E_CFG,
                                   "activation and quant contracts disagree for %s" % tensor_name)
    if affine is not None:
        applied = affine["applied"]
        quant = {"schema": "v0.2-yolo-affine-contract-1",
                 "applied_activations": applied["tensors"],
                 "convs": applied["convs"], "adds": applied["adds"],
                 "concats": applied["concats"]}
        inputs = load_input_manifest(input_manifest_path, source)
        initializers = affine["initializers"]
    else:
        quant = apply_contract(source, contract)
        inputs = load_input_manifest(input_manifest_path, source)
        initializers = load_initializers(model_path)
    if activation_contract is not None:
        for node in source.nodes:
            if node.op_type in ("Conv", "DepthwiseConv"):
                for tensor_name, dtype in ((node.inputs[1], "int8"),
                                           (node.inputs[2], "int32")):
                    if initializers[tensor_name]["dtype"] != dtype:
                        raise CompileError(E_CFG,
                                           "LUT deployment requires explicit %s initializer %s; "
                                           "float parameter quantization is not implicit" %
                                           (dtype, tensor_name), node_id=node.id)
    lowered, lowering = lower_graph(source, spatial_caps=spatial_infer_caps)
    from .maxpool2d_geometry import select_backend
    for node in lowered.nodes:
        if node.op_type == "MaxPool":
            selected = select_backend(lowered, node, maxpool_backend)
            node.metadata["maxpool_backend"] = selected
            if selected == "pool5":
                node.lowering = "maxpool5x5_core_v1"
            elif selected == "generic":
                node.lowering = "maxpool2d_generic_v1"
    _stage("frontend_lowering")
    graph_name = name or source.name
    spatial_resolved = None
    spatial_engines = None
    spatial_rows = None
    if affine is not None:
        build_capacity = affine_frontend.derive_build_capacity(lowered)
        affine_engines = affine_frontend.derive_engine_capacity(lowered, build_capacity)
        affine_capacity = {
            "schema": "e1-affine-capacity-1",
            "capability_version": "affine-int8-v1",
            "requested": {"max_feature_height": "auto", "max_feature_width": "auto"},
            "required": {"height": int(build_capacity["max_h"]),
                         "width": int(build_capacity["max_w"])},
            "allocated": {"height": int(build_capacity["max_h"]),
                          "width": int(build_capacity["max_w"])},
            "mode": "graph_derived",
            "scope": "materialized_runtime_activation_tensors",
            "engine_capacity": affine_engines,
            "row_budget": {"dense_max_in_row_bytes":
                           int(build_capacity["dense_max_in_row_bytes"]),
                           "dense_max_out_row_bytes":
                           int(build_capacity["dense_max_out_row_bytes"]),
                           "dense_max_weight_tiles":
                           int(build_capacity["dense_max_weight_tiles"])},
        }
        spatial_engines = affine_engines
        spatial_rows = affine_frontend.check_affine_row_budget(build_capacity, affine_engines)
    elif spatial_request is not None:
        # First-batch engine capability gate, then the independent H/W axes.
        spatial_capacity.validate_graph_support(lowered)
        required = spatial_capacity.required_feature_hw(lowered)
        spatial_resolved = spatial_capacity.resolve(spatial_request, required)
        allocated_axis = spatial_resolved["allocated"]
        build_capacity = derive_spatial_capacities(
            lowered, allocated_axis["height"], allocated_axis["width"])
        spatial_engines = spatial_capacity.engine_capacity(
            lowered, allocated_axis, build_capacity)
        spatial_rows = spatial_capacity.check_row_capacity(
            spatial_engines, build_capacity,
            max_axis=spatial_capacity.TARGET_MAX_AXIS)
    else:
        build_capacity = derive_capacities(lowered)
    has_depthwise = any(node.op_type == "DepthwiseConv" for node in lowered.nodes)
    if has_depthwise and spatial_request is None:
        build_capacity = _widen_depthwise_capacity(lowered, build_capacity)
    # MAX_CH is a capacity: it covers both the graph channel demand and the
    # DW MAC lane count.  DW_PC widens capacity only when it is larger; it is
    # never presented as the logical channel count.
    allocated_capacity = allocate_capacity(build_capacity,
                                           int(resolved_engine.depthwise["pc"]))
    if spatial_request is not None:
        # keep the spatial height axis alive through the DW widening helper
        allocated_capacity["max_h"] = int(spatial_resolved["allocated"]["height"])
        build_capacity["max_h"] = int(spatial_resolved["allocated"]["height"])
    elif affine is not None:
        allocated_capacity["max_h"] = int(build_capacity["max_h"])
    profile_document = None
    rtl_capacities = None
    capacity_override = None
    if spatial_request is not None:
        # The compute profile's legacy capacity_limit.max_w=32 bounds the *legacy*
        # build geometry only; a spatial build passes its resolved width as a
        # versioned, recorded override instead of mutating the definition file.
        capacity_override = {"max_w": int(spatial_resolved["allocated"]["width"])}
    if resolved is not None:
        # Capacity/geometry/group-width validation happens before the first
        # compile artifact is written, so a rejected profile leaves no
        # half-complete compile directory behind.
        derived = compute_profiles.validate_build(
            resolved, allocated_capacity,
            capacity_limit_override=capacity_override)
        profile_document = compute_profiles.manifest_document(
            resolved, derived, capacity_limit_override=capacity_override)
        rtl_capacities = dict(build_capacity)
    elif has_depthwise or engine_requested:
        # Explicit capacity dict keeps the generated wrapper capacities
        # consistent with the widened derivation above.  verilog_codegen applies
        # the same allocate_capacity rule and records requested vs allocated.
        rtl_capacities = dict(build_capacity)
    if affine is not None:
        rtl_capacities = dict(build_capacity)
        # One weight-face read port for the <=8x8 profile; the 16x16 profile keeps
        # the four replicated ports.  A legacy build emits 0 (module default 4).
        rtl_capacities["dense_weight_read_ports"] = 1 if (tn <= 8 and tm <= 8) else 4
    compile_dir.mkdir(parents=True, exist_ok=True)
    _archive_inputs(compile_dir, model_path, contract_path, input_manifest_path, inputs)
    if activation_contract_path is not None:
        _copy_file(activation_contract_path, compile_dir / "model" / "activation_contract.json")
    _json(compile_dir / "source_graph_ir.json", source.document())
    _json(compile_dir / "lowered_graph_ir.json", lowered.document())
    _json(compile_dir / "lowering_manifest.json", lowering)
    graph_plan = build_graph_plan(lowered)
    layout_plan = build_layout_plan(lowered)
    memory_plan = build_memory_plan(lowered, graph_plan, layout_plan, profile=profile)
    for key, value in (("graph_plan", graph_plan), ("layout_plan", layout_plan),
                       ("memory_plan", memory_plan)):
        _json(compile_dir / (key + ".json"), value)
    plan = {"graph_digest": lowered.digest(), "graph_plan": graph_plan,
            "layout_plan": layout_plan, "memory_plan": memory_plan}
    _stage("planning")
    if spatial_resolved is not None:
        # v0.2 batch 2: payload builders (e.g. the LUT H/W gate) read the resolved
        # allocated axes from the plan instead of hard-coding a 32x32 batch limit.
        plan["spatial_capacity"] = spatial_resolved
    elif affine_capacity is not None:
        plan["spatial_capacity"] = affine_capacity
    stream = generate_stream(lowered, plan, run_id=run_id + ":" + graph_name)
    generated = compile_dir / "generated"
    rtl_manifest = _write_graph_rtl_with_repo_root(
        lowered, plan, generated, repo_root=repo_root, run_id=run_id,
        name=graph_name, tn=tn, tm=tm, mac_impl=int(mac_impl), stream=stream,
        capacities=rtl_capacities, compute_profile=profile_document,
        engine_profile=resolved_engine, target=resolved_target,
        spatial_capacity=(spatial_resolved if spatial_resolved is not None
                          else affine_capacity),
        spatial_engines=spatial_engines,
        spatial_rows=spatial_rows,
        small_engine_max_w=spatial_capacity.SMALL_ENGINE_MAX_W,
        engine_pruning=engine_pruning)
    _stage("codegen_rtl")
    params_dir = compile_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    parameter_rows = []
    param_blobs: List[Tuple[str, str, bytes]] = []
    lut_extents = []
    for node in lowered.nodes:
        if node.op_type == "Conv":
            blobs = (("weight", node.inputs[1], pack_conv_weights(node, lowered, initializers)),
                     ("bias", node.inputs[2], pack_conv_bias(node, lowered, initializers)),
                     ("quant", None, pack_conv_quant(node, lowered)))
        elif node.op_type == "DepthwiseConv":
            blobs = (("weight", node.inputs[1],
                      pack_depthwise_weights(node, lowered, initializers)),
                     ("bias", node.inputs[2],
                      pack_depthwise_bias(node, lowered, initializers)),
                     ("quant", None, pack_depthwise_quant(node, lowered)))
        elif node.op_type == "ActivationLUT":
            from .activation_lut import generate_table
            from .activation_oracle import oracle_table
            from .instr_codegen import _lut_allocations
            si = float(lowered.tensors[node.inputs[0]].quantization["scale"])
            so = float(lowered.tensors[node.outputs[0]].quantization["scale"])
            mode = str(node.attrs["mode"])
            alpha = node.attrs.get("alpha", 0.01) if mode == "LeakyRelu" else None
            table = generate_table(mode, si, so, alpha)
            if list(table) != list(oracle_table(mode, si, so, alpha)):
                raise CompileError(E_CFG, "LUT generator/oracle mismatch", node_id=node.id)
            blob = bytes(int(v) & 0xFF for v in table)
            alloc = _lut_allocations(lowered, plan, int(memory_plan["arena"]["alignment"]))[node.id]
            filename = "%s_table.bin" % node.id
            (params_dir / filename).write_bytes(blob)
            param_blobs.append((alloc["address"], filename, blob))
            lut_extents.append(int(alloc["offset"]) + int(alloc["allocated_bytes"]))
            parameter_rows.append({"node": node.id, "kind": "lut_table", "mode": mode,
                                   "address": alloc["address"], "bytes": len(blob),
                                   "sha256": sha256_bytes(blob), "file": "params/" + filename,
                                   "input_scale": si, "output_scale": so,
                                   "alpha": alpha, "oracle_matched": True})
            continue
        else:
            continue
        for kind, tensor_name, blob in blobs:
            filename = "%s_%s.bin" % (node.id, kind)
            (params_dir / filename).write_bytes(blob)
            if kind == "quant":
                alloc = stream["quant_allocations"][node.id]
                address, declared_bytes = alloc["address"], int(alloc["bytes"])
            else:
                binding = memory_plan["tensor_bindings"][tensor_name]
                address, declared_bytes = binding["address"], int(binding["bytes"])
            if len(blob) != declared_bytes:
                raise CompileError(E_CFG, "%s bytes differ from plan" % filename)
            param_blobs.append((address, filename, blob))
            parameter_rows.append({"node": node.id, "kind": kind, "address": address,
                                   "bytes": len(blob), "sha256": sha256_bytes(blob),
                                   "file": "params/" + filename,
                                   "source_initializer": tensor_name,
                                   "requant_records": node.metadata.get("requant_records", [])
                                   if kind == "quant" else None})
    # Same-shape INT8 initializers consumed as PL data (for example the Mul
    # constant operand) are packed in the activation ABI, not as Conv weights.
    activation_initializers = []
    for name, row in sorted(layout_plan["tensors"].items()):
        tensor = lowered.tensors.get(name)
        if tensor is None or tensor.kind != "initializer":
            continue
        if row["managed_by"] != "pl" or row["storage"].get("layout") != "NHWC8_A16":
            continue
        if row["storage"].get("initializer_activation") is not True:
            continue
        init_source = initializers.get(name)
        if init_source is None or init_source["dtype"] != "int8":
            raise CompileError(E_CFG, "data initializer %s must be INT8" % name)
        blob = pack_activation(tuple(int(v) for v in init_source["shape"]),
                               [int(v) for v in init_source["values"]], dtype="int8")
        binding = memory_plan["tensor_bindings"][name]
        if len(blob) != int(binding["bytes"]):
            raise CompileError(E_CFG, "data initializer %s bytes differ from plan" % name)
        filename = "%s_data.bin" % name
        (params_dir / filename).write_bytes(blob)
        param_blobs.append((binding["address"], filename, blob))
        parameter_rows.append({"node": None, "kind": "data_initializer", "address":
                               binding["address"], "bytes": len(blob),
                               "sha256": sha256_bytes(blob), "file": "params/" + filename,
                               "source_initializer": name, "requant_records": None})
        activation_initializers.append({"name": name, "address": binding["address"],
                                        "bytes": len(blob),
                                        "sha256": sha256_bytes(blob)})
    _json(compile_dir / "parameter_manifest.json",
          {"schema": "e1-p3-parameter-manifest-1", "graph_digest": lowered.digest(),
           "parameters": parameter_rows,
           "activation_initializers": activation_initializers})
    _stage("parameter_pack")

    base = int(memory_plan["arena"]["base"], 16)
    extents = [int(row["offset"]) + int(row["allocated_bytes"])
               for row in memory_plan["buffers"].values()]
    for row in stream["quant_allocations"].values():
        extents.append(int(row["offset"]) + int(row["allocated_bytes"]))
    extents.extend(lut_extents)
    image_size = max(extents, default=0)
    if affine is not None:
        image_limit = AFFINE_SIM_DDR_LIMIT_MIB * 1024 * 1024
    else:
        image_limit = (int(spatial_request.sim_ddr_limit_mib) * 1024 * 1024
                       if spatial_request is not None else MAX_SIM_DDR_BYTES)
    if image_size <= 0 or image_size > image_limit:
        raise CompileError(E_SPATIAL_MEMORY if spatial_request is not None else E_CFG,
                           "DDR image size %d is outside 1..%d" % (image_size, image_limit),
                           detail={"image_bytes": int(image_size),
                                   "limit_bytes": int(image_limit),
                                   "sim_ddr_limit_mib": (None if spatial_request is None
                                                         else spatial_request.sim_ddr_limit_mib)})
    image = bytearray(image_size)
    intents: List[Dict[str, Any]] = []
    for input_name, row in inputs["inputs"].items():
        blob = pack_activation(row["shape"], row["values"], dtype="int8")
        address = memory_plan["tensor_bindings"][input_name]["address"]
        _put(image, base, address, blob, "input:" + input_name, intents)
    for address, filename, blob in param_blobs:
        _put(image, base, address, blob, "param:" + filename, intents)
    ddr_path = compile_dir / "ddr_init.bin"
    ddr_path.write_bytes(bytes(image))
    _json(compile_dir / "ddr_init_manifest.json",
          {"schema": "e1-p2-ddr-init-1", "base": memory_plan["arena"]["base"],
           "bytes": len(image), "sha256": sha256_bytes(bytes(image)), "writes": intents})

    _stage("ddr_image")
    # Optional explicit reference dependency cache (task book 7).  With
    # reference_cache=None this performs exactly the historical two
    # evaluate_graph calls; with a cache it either materialises a verified entry
    # or recomputes and publishes one after the source/lowered check below.
    if affine is not None:
        from . import affine_frontend
        reference_session = None
        source_values = affine_frontend.reference_values(source, affine["reference"])
        seed = affine.get("seed_tensors") or None
        if seed:
            lowered_ref = affine_frontend.run_canonical_reference(
                lowered, initializers, seed_tensors=seed)
        else:
            lowered_ref = affine_frontend.run_canonical_reference(
                lowered, initializers,
                input_u8=np.asarray(input_u8, dtype=np.uint8))
        lowered_values = affine_frontend.reference_values(lowered, lowered_ref)
    else:
        reference_session = reference_cache_module.open_reference(
            reference_cache,
            model_path=model_path, contract_path=contract_path,
            input_manifest_path=input_manifest_path,
            activation_contract_path=activation_contract_path,
            source=source, lowered=lowered, initializers=initializers,
            inputs=inputs["inputs"], run_id=run_id, graph_name=graph_name)
        source_values = reference_session.source_values
        lowered_values = reference_session.lowered_values
    common_mismatches = []
    for tensor_name, left in source_values["values"].items():
        right = lowered_values["values"].get(tensor_name)
        if right is not None and left.values != right.values:
            common_mismatches.append(tensor_name)
    output_mismatches = [name for name in source.outputs
                         if source_values["values"][name].values != lowered_values["values"][name].values]
    if common_mismatches or output_mismatches:
        raise CompileError(E_CFG, "source/lowered reference mismatch: %s / %s" %
                           (common_mismatches, output_mismatches))
    if reference_session is not None:
        reference_session.commit({"common_mismatches": common_mismatches,
                                  "output_mismatches": output_mismatches})
    _stage("reference")
    golden_dir = compile_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    node_golden = []
    record_snapshots = []
    records_by_node = {}
    for record in stream["records"]:
        records_by_node.setdefault(record["node_id"], []).append(record["index"])
    for node in lowered.nodes:
        if node.placement == "ps" or node.metadata_only:
            continue
        for output_name in node.outputs:
            value = lowered_values["values"][output_name]
            if value.dtype != "int8" or len(value.shape) != 4:
                continue
            blob = pack_activation(value.shape, value.values, dtype=value.dtype)
            filename = "%s__%s.bin" % (node.id, output_name)
            (golden_dir / filename).write_bytes(blob)
            binding = memory_plan["tensor_bindings"][output_name]
            rec_index = records_by_node[node.id][0]
            snap = {"when": "record", "record_index": rec_index,
                    "name": "r%02d_%s" % (rec_index, output_name),
                    "node": node.id, "op": node.op_type, "output": output_name,
                    "address": binding["address"], "bytes": len(blob),
                    "golden": "golden/" + filename, "sha256": sha256_bytes(blob)}
            record_snapshots.append(snap)
            node_golden.append({"node": node.id, "op": node.op_type,
                                "output": output_name, "bytes": len(blob),
                                "sha256": sha256_bytes(blob),
                                "file": "golden/" + filename})
    output_golden = []
    final_snapshots = []
    for output_name in lowered.outputs:
        value = lowered_values["values"][output_name]
        if value.dtype != "int8" or len(value.shape) != 4:
            raise CompileError(E_CFG, "graph output %s is not rank-4 INT8" % output_name)
        blob = pack_activation(value.shape, value.values, dtype=value.dtype)
        filename = "output__%s.bin" % output_name
        (golden_dir / filename).write_bytes(blob)
        binding = memory_plan["tensor_bindings"][output_name]
        final_snapshots.append({"when": "final", "record_index": -1,
                                "name": "final_%s" % output_name,
                                "node": value_source_node(lowered, output_name),
                                "op": "GraphOutput", "output": output_name,
                                "address": binding["address"], "bytes": len(blob),
                                "golden": "golden/" + filename,
                                "sha256": sha256_bytes(blob)})
        output_golden.append({"output": output_name, "bytes": len(blob),
                              "sha256": sha256_bytes(blob), "file": "golden/" + filename})
    snapshots = record_snapshots + final_snapshots
    _json(compile_dir / "snapshot_manifest.json",
          {"schema": "e1-p2-snapshot-manifest-1", "snapshots": snapshots,
           "snapshot_count": len(snapshots)})
    snapshot_lines = []
    for row in snapshots:
        snapshot_lines.append("snap %s %d %s %s %d %s" %
                              (row["when"], row["record_index"], row["name"],
                               row["address"], row["bytes"], row["golden"]))
    (compile_dir / "snapshot_desc.txt").write_text("\n".join(snapshot_lines) + "\n",
                                                    encoding="ascii")
    _json(compile_dir / "node_outputs.json",
          {"schema": "e1-p2-node-outputs-1", "source_graph_digest": source.digest(),
           "lowered_graph_digest": lowered.digest(),
           "source_reference": source_values["node_outputs"],
           "lowered_reference": lowered_values["node_outputs"],
           "node_golden": node_golden, "output_golden": output_golden})
    _stage("golden_snapshot")
    spatial_memory = None
    spatial_cycles = None
    tb_rows = None
    if spatial_request is not None or affine is not None:
        snapshot_max_bytes = max([int(row["bytes"]) for row in snapshots] or [0])
        sim_limit = (int(spatial_request.sim_ddr_limit_mib)
                     if spatial_request is not None else AFFINE_SIM_DDR_LIMIT_MIB)
        spatial_memory = spatial_capacity.memory_limits(
            sim_ddr_limit_mib=sim_limit,
            arena_bytes=int(memory_plan["arena"]["used_bytes"]),
            image_bytes=int(image_size),
            snapshot_max_bytes=int(snapshot_max_bytes))
        spatial_cycles = spatial_capacity.estimate_cycles(lowered, stream)
        tb_doc = spatial_capacity.tb_parameters(spatial_memory, spatial_cycles,
                                                len(snapshots))
        tb_json = compile_dir / "generated" / "tb_params.json"
        tb_vh = compile_dir / "generated" / "tb_params.vh"
        _json(tb_json, tb_doc)
        tb_vh.write_text(spatial_capacity.tb_params_vh(tb_doc), encoding="ascii")
        # Convenience aliases used by the simulation runner: it only needs the
        # budget, the memory depth and the tb_params path (plus the derived
        # snapshot/watchdog values) to launch the same TB for any capacity.
        spatial_memory.update({
            "mem_words": int(tb_doc["TB_MEM_WORDS"]),
            "mem_bytes": int(tb_doc["TB_MEM_BYTES"]),
            "tb_params": "generated/tb_params.vh",
            "tb_max_snaps": int(tb_doc["TB_MAX_SNAPS"]),
            "tb_snap_chunk": int(tb_doc["TB_SNAP_CHUNK"]),
            "tb_watchdog_cycles": int(tb_doc["TB_WATCHDOG_CYCLES"]),
        })
        tb_rows = {
            "schema": tb_doc["schema"],
            "json": tb_json.resolve().relative_to(compile_dir).as_posix(),
            "json_sha256": _sha_file(tb_json),
            "vh": tb_vh.resolve().relative_to(compile_dir).as_posix(),
            "vh_sha256": _sha_file(tb_vh),
            "values": tb_doc,
        }
    _stage("simulation_budget")
    manifest = {
        "schema": schema,
        "status": "pass",
        "profile": profile,
        "graph_name": graph_name,
        "model_sha256": model_sha,
        "quant_contract_sha256": _sha_file(contract_path),
        "input_manifest_sha256": _sha_file(input_manifest_path),
        "source_graph_digest": source.digest(),
        "lowered_graph_digest": lowered.digest(),
        "quant": {"applied_activations": len(contract.get("tensors", {})),
                  "convs": quant["convs"]},
        "lowering": lowering,
        "graph_plan": "graph_plan.json",
        "layout_plan": "layout_plan.json",
        "memory_plan": "memory_plan.json",
        "parameter_manifest": "parameter_manifest.json",
        "instruction_stream": Path(rtl_manifest["instruction_stream"]).resolve()
                            .relative_to(compile_dir).as_posix(),
        "instruction_mem": Path(rtl_manifest["mem"]).resolve()
                           .relative_to(compile_dir).as_posix(),
        "instruction_mem_sha256": rtl_manifest["files_sha256"][rtl_manifest["mem"]],
        "generated_rtl_manifest": "generated/rtl_manifest.json",
        "ddr_init": "ddr_init.bin",
        "ddr_init_sha256": sha256_bytes(bytes(image)),
        "snapshot_manifest": "snapshot_manifest.json",
        "node_outputs": "node_outputs.json",
        "records": [{"index": row["index"], "node": row["node_id"], "op": row["op"],
                     "opcode": "0x%04X" % row["opcode"], "payload_words": row["payload_words"]}
                    for row in stream["records"]],
        "gates": {"frontend": "pass", "quant_contract": "pass", "lowering": "pass",
                  "reference": "pass", "parameter_pack": "pass", "memory_planning": "pass",
                  "codegen": "pass", "rtl": "not_run", "board": "not_run"},
    }
    legacy_depthwise_row = {
        "nodes": [node.id for node in lowered.nodes
                  if node.op_type == "DepthwiseConv"],
        "records": sum(1 for record in stream["records"]
                       if record["op"] == "DepthwiseConv"),
        "opcode": "0x001C",
        "source_root": str(repo_root) if repo_root is not None else None,
    }
    if engine_requested:
        # New-style builds: merge the resolved per-engine facts with the
        # historical node/record identity (empty node list for DW-free graphs).
        # Legacy builds keep the exact historical field set below.
        depthwise_row = dict(rtl_manifest["depthwise"])
        depthwise_row.update({key: value for key, value in
                              legacy_depthwise_row.items()
                              if key not in depthwise_row})
        manifest["depthwise"] = depthwise_row
    elif has_depthwise:
        manifest["depthwise"] = legacy_depthwise_row
    if engine_requested:
        # Per-engine configuration evidence for explicit --engine-profile /
        # --target / --dw-* builds.  Compile only: every implemented engine's
        # functional/ooc/platform/board gate stays not_run.
        manifest["engine_profile"] = rtl_manifest["engine_profile"]
        manifest["target"] = rtl_manifest["target"]
        manifest["dense"] = rtl_manifest["dense"]
        manifest["generated_parameters"] = rtl_manifest["generated_parameters"]
        manifest["engine_gates"] = rtl_manifest["engine_gates"]
        manifest["gates"]["engine_parameters"] = "pass"
        manifest["gates"]["engine_verification"] = "not_run"
        # Consumer-friendly relative artifact pointers (same shape the prior
        # LUT/DW batch compile wrapper used) so downstream OOC/platform flows do
        # not have to guess paths from the absolute rtl_manifest entries.
        manifest["generated"] = {
            "rtl_manifest": "generated/rtl_manifest.json",
            "sources_f": "generated/sources.f",
            "npu_top": "generated/npu_top.v",
            "mem": Path(rtl_manifest["mem"]).resolve()
                   .relative_to(compile_dir).as_posix(),
            "instruction_stream": Path(rtl_manifest["instruction_stream"])
                                  .resolve().relative_to(compile_dir).as_posix(),
            "source_files": rtl_manifest["source_files"],
            "new_core_sources": rtl_manifest.get("new_core_sources", []),
        }
    if affine is not None:
        manifest["numeric_profile"] = numeric_profile
        manifest["affine_capacity"] = dict(affine_capacity)
        manifest["affine_frontend"] = {
            "schema": affine_frontend.SCHEMA,
            "profile": numeric_profile,
            "fragment": ([str(v) for v in fragment_nodes]
                         if fragment_nodes else None),
            "fragment_input_rows": (int(fragment_input_rows)
                                    if fragment_input_rows else None),
            "inputs": list(source.inputs), "outputs": list(source.outputs),
            "nodes": len(source.nodes),
            "onnx_sha256": affine["model_sha256"],
            "frontend": affine["report"],
            "row_budget": affine_capacity["row_budget"],
        }
        manifest["spatial_capacity"] = dict(affine_capacity)
        manifest["spatial_capacity"]["engine_capacity"] = affine_engines
        manifest["spatial_capacity"]["engine_row_capacity"] = spatial_rows
        manifest["spatial_capacity"]["memory_limits"] = spatial_memory
        manifest["spatial_capacity"]["simulation_budget"] = spatial_cycles
        manifest["engine_capacity"] = affine_engines
        manifest["memory_limits"] = spatial_memory
        manifest["simulation_budget"] = spatial_cycles
        manifest["tb_params"] = tb_rows
        manifest["gates"]["spatial_capacity"] = "pass"
        manifest["gates"]["affine_frontend"] = "pass"
        manifest["gates"]["reserved_engines"] = "not_used_by_graph"
    if spatial_request is not None:
        # v0.2 spatial block: requested/required/allocated are separate, the
        # engine rows record exactly which capacity each core got, and the
        # simulation budget is derived from independent work counts.
        manifest["spatial_capacity"] = dict(spatial_resolved)
        manifest["spatial_capacity"]["capability"] = spatial_capacity.capability_document()
        manifest["spatial_capacity"]["engine_capacity"] = spatial_engines
        manifest["spatial_capacity"]["engine_row_capacity"] = spatial_rows
        manifest["spatial_capacity"]["memory_limits"] = spatial_memory
        manifest["spatial_capacity"]["simulation_budget"] = spatial_cycles
        manifest["engine_capacity"] = spatial_engines
        manifest["memory_limits"] = spatial_memory
        manifest["simulation_budget"] = spatial_cycles
        manifest["tb_params"] = tb_rows
        manifest["gates"]["spatial_capacity"] = "pass"
        manifest["gates"]["reserved_engines"] = "not_used_by_graph"
    for key in ("source_root", "source_digest", "source_files",
                "pruning", "physical_instances"):
        if key in rtl_manifest:
            manifest[key] = rtl_manifest[key]
    if activation_contract_path is not None:
        manifest["activation_contract_sha256"] = _sha_file(activation_contract_path)
        manifest["activation_deployment"] = source.metadata.get("activation_deployment", {})
    if resolved is not None:
        # Explicit compute-profile builds add the resolved tier metadata; the
        # legacy/default manifest keeps its historical field set byte-for-byte.
        manifest["compute_profile"] = profile_document
    if extra_manifest:
        manifest.update(extra_manifest)
    if reference_cache is not None:
        # Explicit cache evidence: a hot hit must show recompute_calls == 0 and
        # the full key document / stage timings live next to the compile output.
        manifest["reference_cache"] = reference_session.summary()
        _stage("compile_manifest")
        _json(compile_dir / "reference_cache.json", reference_session.document())
        _json(compile_dir / "stage_timings.json",
              {"schema": "e1-compile-stage-timings-1",
               "stages_seconds": dict(stage_timings),
               "total_seconds": round(time.perf_counter() - _compile_started, 4)})
    _json(compile_dir / "compile_manifest.json", manifest)
    return manifest


def value_source_node(graph, output_name):
    tensor = graph.tensors[output_name]
    return tensor.producer or "graph_input"
