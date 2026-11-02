"""Versioned spatial-capacity semantics for v0.2 (schema 'e1-spatial-capacity-1').

This module is the single rule source for the *feature-map capacity* of a build:

  * actual     -- every materialized runtime INT8 activation of the graph
  * requested  -- the user's --max-feature-width/height value (auto or N)
  * allocated  -- the capacity actually baked into the generated hardware

The three are deliberately separate from the compute parallelism (Tn/Tm, DW_PC):
raising the spatial capacity never changes the PE array, and choosing a PE tile
never changes the capacity.  This module is pure Python: it never touches the
graph contents, never writes to disk and never imports Vivado.

Second-batch implementation target range: 1..640 on each axis, H and W
resolved independently.  The range lives here (a versioned definition), not as
scattered constants inside the operator cores.  Batch 2 expands the feature
fusion families (Copy/Resize/Pool5/LUT) to the same versioned range while
keeping the per-engine row-cache capacity separate from the PE parallelism.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_ir import (
    CompileError,
    E_SPATIAL_AXIS,
    E_SPATIAL_ENGINE,
    E_SPATIAL_LIMIT,
    E_SPATIAL_MEMORY,
    E_SPATIAL_ROW,
    E_SPATIAL_SCHEMA,
    align_up,
)

SCHEMA = "e1-spatial-capacity-1"
CAPABILITY_VERSION = "v0.2-b2"
TARGET_MAX_AXIS = 640
MIN_AXIS = 1
AXIS_CEILING = TARGET_MAX_AXIS

DEFAULT_SIM_DDR_LIMIT_MIB = 32
MIN_SIM_DDR_LIMIT_MIB = 1
MAX_SIM_DDR_LIMIT_MIB = 512
DDR_GUARD_BYTES = 65536
SNAPSHOT_MAX_CHUNK = 8192

# Unused engines are elaborated with their historical small capacity so a large
# spatial graph never inflates a core it does not execute.  These are *not*
# expansion claims: the corresponding engine row records expanded=false.
SMALL_ENGINE_MAX_W = 32
SMALL_ENGINE_MAX_H = 32
STREAMING_ENGINES = ("eltwise", "reduce")
# Copy engine: unused builds keep a small row cache (align16(align8(16)*32)).
SMALL_COPY_ROW_BYTES = 512

# Batch-2 expanded operator set.  Everything else is a structured rejection in
# spatial mode with the remaining scope attached (the legacy compiler path
# keeps supporting those ops unchanged).
FIRST_BATCH_ENGINES = ("dense", "pool2x2", "eltwise", "reduce")
SECOND_BATCH_ENGINES = ("copy", "resize", "pool5", "activation", "maxpool2d")
EXPANDED_ENGINES = FIRST_BATCH_ENGINES + SECOND_BATCH_ENGINES
NOT_EXPANDED_ENGINES = {
    "depthwise": "DepthwiseConv is limited to C<=16/H,W<=32 and the working "
                 "candidate is not closed (independent item, not a prerequisite)",
    "int8_mul": "INT8 Mul large-size path is out of scope for the second batch",
    "ps": "PS-placed nodes need a host boundary; spatial graphs are fully PL "
          "(Gemm/FC and PS host I/O are out of scope here)",
}
# Batch-2 per-engine legal lowered operator subset (all other attributes are
# rejected by the frontend/op_registry before this gate).
COPY_OPS = ("Pad", "Slice", "Concat", "Split")
RESIZE_OPS = ("Resize",)
POOL5_OPS = ("MaxPool",)
ACTIVATION_OPS = ("ActivationLUT",)

OP_ENGINE = {
    "Conv": "dense",
    "DepthwiseConv": "depthwise",
    "AveragePool": "pool2x2",
    "Relu": "eltwise",
    "Add": "eltwise",
    "Clip": "eltwise",
    "GlobalAveragePool": "reduce",
    "Pad": "copy",
    "Slice": "copy",
    "Concat": "copy",
    "Split": "copy",
    "Resize": "resize",
    "Mul": "int8_mul",
    "ActivationLUT": "activation",
}
SPATIAL_ELTWISE_OPS = ("Relu", "Add")
MAXPOOL2_ENGINE = "pool2x2"
MAXPOOL5_ENGINE = "pool5"


@dataclass(frozen=True)
class SpatialRequest:
    """Parsed user request; None means the legacy (unchanged) compiler path."""

    width_value: Any            # "auto" or explicit int
    height_value: Any
    sim_ddr_limit_mib: Optional[int]

    @property
    def explicit(self) -> bool:
        return True

    def manifest_request(self) -> Dict[str, Any]:
        return {
            "max_feature_width": self.width_value,
            "max_feature_height": self.height_value,
            "sim_ddr_limit_mib": self.sim_ddr_limit_mib,
        }


def _strict_int(value: Any, what: str, code: str = E_SPATIAL_AXIS) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(code, "%s must be an integer (bool/float/numeric string "
                                 "are rejected)" % what,
                           detail={"field": what, "value": repr(value)})
    return int(value)


def parse_axis(value: Any, label: str) -> Any:
    """Return "auto" or a positive int for one feature-map axis."""
    if value is None:
        raise CompileError(E_SPATIAL_SCHEMA, "%s must be 'auto' or a positive integer" % label,
                           detail={"field": label, "value": repr(value)})
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        raise CompileError(E_SPATIAL_SCHEMA,
                           "%s must be 'auto' or a positive integer, got %r" % (label, value),
                           detail={"field": label, "value": value})
    number = _strict_int(value, label)
    if number < MIN_AXIS:
        raise CompileError(E_SPATIAL_AXIS, "%s=%d must be >= %d" % (label, number, MIN_AXIS),
                           detail={"field": label, "value": number})
    if number > AXIS_CEILING:
        raise CompileError(E_SPATIAL_AXIS,
                           "%s=%d exceeds the %s implementation range 1..%d" %
                           (label, number, CAPABILITY_VERSION, AXIS_CEILING),
                           detail={"field": label, "value": number,
                                   "limit": AXIS_CEILING, "capability": CAPABILITY_VERSION})
    return number


def parse_sim_ddr_limit(value: Any) -> int:
    number = _strict_int(value, "sim_ddr_limit_mib")
    if number < MIN_SIM_DDR_LIMIT_MIB or number > MAX_SIM_DDR_LIMIT_MIB:
        raise CompileError(E_SPATIAL_MEMORY,
                           "sim_ddr_limit_mib=%d is outside %d..%d MiB" %
                           (number, MIN_SIM_DDR_LIMIT_MIB, MAX_SIM_DDR_LIMIT_MIB),
                           detail={"value": number,
                                   "min": MIN_SIM_DDR_LIMIT_MIB,
                                   "max": MAX_SIM_DDR_LIMIT_MIB})
    return number


def parse_request(*, max_feature_width: Any = None, max_feature_height: Any = None,
                  sim_ddr_limit_mib: Any = None) -> Optional[SpatialRequest]:
    """Build the spatial request, or None when no spatial option was supplied."""
    if max_feature_width is None and max_feature_height is None and sim_ddr_limit_mib is None:
        return None
    width = "auto" if max_feature_width is None else parse_axis(max_feature_width,
                                                               "max_feature_width")
    height = "auto" if max_feature_height is None else parse_axis(max_feature_height,
                                                                  "max_feature_height")
    limit = (DEFAULT_SIM_DDR_LIMIT_MIB if sim_ddr_limit_mib is None
             else parse_sim_ddr_limit(sim_ddr_limit_mib))
    return SpatialRequest(width_value=width, height_value=height, sim_ddr_limit_mib=limit)


def capability_document() -> Dict[str, Any]:
    """Versioned spatial capability table consumed by the compiler and reports."""
    engines = {
        "dense": {"ops": ["Conv"], "expanded": True,
                  "max_h": TARGET_MAX_AXIS, "max_w": TARGET_MAX_AXIS,
                  "row_based": True},
        "pool2x2": {"ops": ["MaxPool2x2", "AveragePool2x2"], "expanded": True,
                    "max_h": TARGET_MAX_AXIS, "max_w": TARGET_MAX_AXIS,
                    "row_based": True},
        "eltwise": {"ops": ["Relu", "Add"], "expanded": True, "streaming": True,
                    "note": "16B streaming; no row buffer capacity to expand"},
        "reduce": {"ops": ["GlobalAveragePool"], "expanded": True, "streaming": True,
                   "note": "row-segment accumulate; only MAX_CH is a capacity"},
        # Batch-2 expansion.  All four are row/block based: the on-chip working
        # set is a bounded number of rows, never the image area.
        "copy": {"ops": ["Pad", "Slice", "Concat", "Split->Slice"],
                 "expanded": True, "row_based": True,
                 "max_h": TARGET_MAX_AXIS, "max_w": TARGET_MAX_AXIS,
                 "max_ch": 16,
                 "note": "row/block vector copy + channel reorder; 2..4 Concat "
                         "inputs share one output-row staging buffer"},
        "resize": {"ops": ["Resize nearest x2 asymmetric/floor"], "expanded": True,
                   "row_based": True, "max_input_h": TARGET_MAX_AXIS // 2,
                   "max_input_w": TARGET_MAX_AXIS // 2,
                   "max_output_h": TARGET_MAX_AXIS, "max_output_w": TARGET_MAX_AXIS,
                   "note": "input axis <=320 is derived from output<=640 at x2, "
                           "not a permanent core constant"},
        "pool5": {"ops": ["MaxPool5x5 s1 pad2 (SPPF block)"], "expanded": True,
                  "row_based": True, "max_h": TARGET_MAX_AXIS,
                  "max_w": TARGET_MAX_AXIS, "max_ch": 16,
                  "note": "5-row sliding window; padding -128 produced on chip; "
                          "SPPF is three serial Pool5 commands + Concat"},
        "activation": {"ops": ["ActivationLUT SiLU/Sigmoid/Tanh/LeakyRelu"],
                       "expanded": True, "streaming": True,
                       "max_h": TARGET_MAX_AXIS, "max_w": TARGET_MAX_AXIS,
                       "max_ch": 16,
                       "note": "16B streaming table lookup; 256B table load "
                               "counted separately; one loadable table engine"},
    }
    for name, reason in sorted(NOT_EXPANDED_ENGINES.items()):
        engines[name] = {"ops": [], "expanded": False, "reason": reason,
                         "max_w": SMALL_ENGINE_MAX_W}
    return {
        "schema": "e1-spatial-capability-1",
        "capability_version": CAPABILITY_VERSION,
        "target_max_axis": TARGET_MAX_AXIS,
        "axes_resolved_independently": True,
        "scope": "materialized_runtime_activation_tensors",
        "numerics_contract": "NHWC8_A16 / INT32 accumulate / unsigned M0 / "
                             "round-half-away-from-zero shift / sat INT8",
        "engines": engines,
    }


def shape_infer_caps(request: SpatialRequest) -> Dict[str, Any]:
    """Public alias: the geometry ceiling handed to shape_infer."""
    return _spatial_caps_for_shape_infer(request)


def _spatial_caps_for_shape_infer(request: SpatialRequest) -> Dict[str, Any]:
    """Limits handed to shape_infer so math/attribute inference can proceed."""
    return {
        "schema": SCHEMA,
        "feature_max_h": AXIS_CEILING,
        "feature_max_w": AXIS_CEILING,
        "global_avg_max_hw": AXIS_CEILING,
        "width_ceiling": AXIS_CEILING,
    }


def _node_engine(node) -> str:
    if node.placement == "ps":
        return "ps"
    if node.metadata_only:
        return "meta"
    op = str(node.op_type)
    if op == "MaxPool":
        if "legacy_layer_index" in node.metadata:
            return MAXPOOL2_ENGINE
        return MAXPOOL5_ENGINE if node.lowering == "maxpool5x5_core_v1" else "maxpool2d"
    return OP_ENGINE.get(op, "unknown")


def required_feature_hw(graph) -> Dict[str, Any]:
    """Max H/W over every *materialized runtime* INT8 rank-4 activation.

    Initializers (OIHW weights, shape constants, INT32 bias/scale records) are
    excluded on purpose: they are not feature maps.  Metadata aliases are
    included because they reference the same storage.
    """
    max_h = 0
    max_w = 0
    tensors: List[Dict[str, Any]] = []
    for name, tensor in sorted(graph.tensors.items()):
        if tensor.dtype != "int8" or tensor.shape is None or len(tensor.shape) != 4:
            continue
        if tensor.kind == "initializer":
            continue
        h, w = int(tensor.shape[2]), int(tensor.shape[3])
        if h <= 0 or w <= 0:
            continue
        max_h = max(max_h, h)
        max_w = max(max_w, w)
        tensors.append({"tensor": name, "kind": tensor.kind,
                        "shape": [int(v) for v in tensor.shape],
                        "producer": tensor.producer})
    if max_h <= 0 or max_w <= 0:
        raise CompileError(E_SPATIAL_SCHEMA,
                           "graph has no materialized rank-4 INT8 activation")
    return {"height": max_h, "width": max_w,
            "scope": "materialized_runtime_activation_tensors",
            "tensor_count": len(tensors),
            "widest": [row for row in tensors
                       if int(row["shape"][3]) == max_w][:8],
            "tallest": [row for row in tensors
                        if int(row["shape"][2]) == max_h][:8]}


def resolve(request: SpatialRequest, required: Dict[str, Any]) -> Dict[str, Any]:
    """Apply auto/explicit per axis; explicit values must cover the graph."""
    def axis(mode: Any, need: int, label: str) -> Tuple[int, str]:
        if mode == "auto":
            return int(need), "auto"
        value = int(mode)
        if value < int(need):
            raise CompileError(E_SPATIAL_AXIS,
                               "explicit %s=%d is smaller than the graph requirement %d" %
                               (label, value, need),
                               detail={"field": label, "allocated": value,
                                       "required": int(need)})
        return value, "explicit"

    height, height_mode = axis(request.height_value, required["height"],
                               "max_feature_height")
    width, width_mode = axis(request.width_value, required["width"],
                             "max_feature_width")
    # An auto-derived axis must obey the same versioned range: e.g. a Resize
    # input of 321 derives a 642 output, which is rejected here (E_SPATIAL_AXIS)
    # instead of silently generating an out-of-range build.
    for label, value in (("max_feature_height", int(height)),
                         ("max_feature_width", int(width))):
        if value > AXIS_CEILING:
            raise CompileError(E_SPATIAL_AXIS,
                               "%s=%d (graph requirement) exceeds the %s "
                               "implementation range 1..%d" %
                               (label, value, CAPABILITY_VERSION, AXIS_CEILING),
                               detail={"field": label, "allocated": value,
                                       "limit": AXIS_CEILING,
                                       "capability": CAPABILITY_VERSION})
    modes = {height_mode, width_mode}
    mode = "auto" if modes == {"auto"} else ("explicit" if modes == {"explicit"}
                                             else "mixed")
    return {
        "schema": SCHEMA,
        "capability_version": CAPABILITY_VERSION,
        "requested": {
            "max_feature_height": request.height_value,
            "max_feature_width": request.width_value,
            "sim_ddr_limit_mib": request.sim_ddr_limit_mib,
        },
        "required": {"height": int(required["height"]), "width": int(required["width"])},
        "required_evidence": required,
        "allocated": {"height": int(height), "width": int(width)},
        "mode": mode,
        "axis_modes": {"height": height_mode, "width": width_mode},
        "scope": "materialized_runtime_activation_tensors",
        "target_limit": {"height": TARGET_MAX_AXIS, "width": TARGET_MAX_AXIS,
                         "capability_version": CAPABILITY_VERSION},
    }


def validate_graph_support(graph) -> Dict[str, Any]:
    """Per-node engine capability gate for a spatial build.

    A spatial build supports exactly the first-batch operator set.  Using any
    not-yet-expanded engine is a structured rejection that names the node and
    the next-batch scope - it is never silently executed on a small engine.
    """
    used: Dict[str, bool] = {name: False for name in
                             list(EXPANDED_ENGINES) + list(NOT_EXPANDED_ENGINES)}
    rows: List[Dict[str, Any]] = []
    for node in graph.nodes:
        engine = _node_engine(node)
        if engine == "meta":
            continue
        if engine == "unknown":
            raise CompileError(E_SPATIAL_ENGINE,
                               "operator %s is outside the first-batch spatial "
                               "operator set" % node.op_type,
                               node_id=node.id,
                               detail={"op": str(node.op_type),
                                       "next_batch": "large feature fusion / SPPF"})
        if engine in used:
            used[engine] = True
        if engine in EXPANDED_ENGINES:
            op = str(node.op_type)
            if engine == "eltwise" and op not in SPATIAL_ELTWISE_OPS:
                raise CompileError(E_SPATIAL_ENGINE,
                                   "eltwise op %s is not in the supported spatial "
                                   "set (only Relu/Add)" % node.op_type,
                                   node_id=node.id,
                                   detail={"op": op, "engine": engine,
                                           "scope": CAPABILITY_VERSION})
            allowed = {"copy": COPY_OPS, "resize": RESIZE_OPS, "maxpool2d": POOL5_OPS,
                       "pool5": POOL5_OPS, "activation": ACTIVATION_OPS}.get(engine)
            if allowed is not None and op not in allowed:
                raise CompileError(E_SPATIAL_ENGINE,
                                   "engine %s does not implement %s in %s" %
                                   (engine, op, CAPABILITY_VERSION),
                                   node_id=node.id,
                                   detail={"op": op, "engine": engine,
                                           "allowed": list(allowed),
                                           "capability_version": CAPABILITY_VERSION})
            rows.append({"node": node.id, "engine": engine})
            continue
        reason = NOT_EXPANDED_ENGINES.get(engine, "engine is not expanded in this batch")
        raise CompileError(E_SPATIAL_ENGINE,
                           "%s (engine %s) is not expanded in %s: %s" %
                           (node.op_type, engine, CAPABILITY_VERSION, reason),
                           node_id=node.id,
                           detail={"op": str(node.op_type), "engine": engine,
                                   "capability_version": CAPABILITY_VERSION,
                                   "next_batch": "large feature fusion / SPPF batch",
                                   "reason": reason})
    # v0.2 numeric contract: every materialized runtime INT8 rank-4 activation
    # has logical C<=16 (weights/shape constants are excluded).  C17..C64 keeps
    # the legacy compiler behaviour but is a structured spatial rejection.
    pool_tensors = set()
    other_tensors = set()
    for node in graph.nodes:
        names = set(node.inputs) | set(node.outputs)
        (pool_tensors if _node_engine(node) == "maxpool2d" else other_tensors).update(names)
    pool_only = pool_tensors - other_tensors
    for tensor_name, tensor in sorted(graph.tensors.items()):
        if (tensor.dtype != "int8" or tensor.kind == "initializer" or
                tensor.shape is None or len(tensor.shape) != 4):
            continue
        channels = int(tensor.shape[1])
        if channels > 16 and tensor_name not in pool_only:
            raise CompileError(E_SPATIAL_LIMIT,
                               "activation tensor %s has C=%d; spatial builds support "
                               "logical C<=16 in %s" % (tensor_name, channels,
                                                        CAPABILITY_VERSION),
                               detail={"tensor": tensor_name, "channels": channels,
                                       "limit": 16,
                                       "capability_version": CAPABILITY_VERSION})
    return {"used_by_graph": used, "nodes": rows}


def _activation_hw(graph, names: Sequence[str]) -> Tuple[int, int]:
    h = w = 0
    for name in names:
        tensor = graph.tensors.get(name)
        if tensor is None or tensor.shape is None or len(tensor.shape) != 4:
            continue
        h = max(h, int(tensor.shape[2]))
        w = max(w, int(tensor.shape[3]))
    return h, w


def _tensor_hwc(tensors) -> Tuple[int, int, int, int]:
    """Max H, W, align8(C) and align16(align8(C)*W) over rank-4 INT8 tensors."""
    height = width = channels = row_bytes = 0
    for tensor in tensors:
        shape = tensor.shape
        if tensor.dtype != "int8" or shape is None or len(shape) != 4:
            continue
        th, tw = int(shape[2]), int(shape[3])
        ch_alloc = ((int(shape[1]) + 7) // 8) * 8
        height = max(height, th)
        width = max(width, tw)
        channels = max(channels, ch_alloc)
        row_bytes = max(row_bytes, align_up(ch_alloc * tw, 16))
    return height, width, channels, row_bytes


def _named_tensors(graph, names: Sequence[str]):
    rows = []
    for name in names:
        tensor = graph.tensors.get(name)
        if tensor is not None:
            rows.append(tensor)
    return rows


def _engine_nodes(graph) -> Dict[str, List[Any]]:
    nodes: Dict[str, List[Any]] = {name: [] for name in EXPANDED_ENGINES}
    for node in graph.nodes:
        engine = _node_engine(node)
        if engine in nodes:
            nodes[engine].append(node)
    return nodes


def engine_capacity(graph, allocated: Dict[str, int],
                    base_capacity: Dict[str, int]) -> Dict[str, Any]:
    """Per-engine capacity actually baked into the generated top.

    Used engines get the capacity they really touch (pool/resize/pool5 read
    their input side; copy sizes one row-staging buffer; LUT streams), while
    unused engines keep a small historical capacity and are marked
    used_by_graph=false so an unused core can neither force a large graph back
    to 32 nor inflate its own resource budget.
    """
    nodes = _engine_nodes(graph)
    dense_used = bool(nodes["dense"])
    eltwise_used = bool(nodes["eltwise"])
    reduce_used = bool(nodes["reduce"])
    pool_names: List[str] = []
    for node in nodes["pool2x2"]:
        pool_names.extend([node.inputs[0]] + list(node.outputs))
    pool_h, pool_w = _activation_hw(graph, pool_names)
    # Copy: all inputs and outputs of copy nodes share one row staging limit.
    copy_tensors = _named_tensors(
        graph, [n for node in nodes["copy"] for n in list(node.inputs) + list(node.outputs)])
    copy_h, copy_w, copy_ch, copy_row = _tensor_hwc(copy_tensors)
    # Resize: the row buffer is sized by the *input* side; the doubled output
    # must still be covered by the user-allocated axes (checked below).
    resize_in = _named_tensors(graph, [node.inputs[0] for node in nodes["resize"]])
    resize_h, resize_w, resize_ch, resize_row = _tensor_hwc(resize_in)
    resize_out = _named_tensors(graph, [node.outputs[0] for node in nodes["resize"]])
    resize_oh, resize_ow, _, resize_out_row = _tensor_hwc(resize_out)
    # Pool5: each input row enters the line cache once (output shape == input).
    p5_in = _named_tensors(graph, [node.inputs[0] for node in nodes["pool5"]])
    p5_h, p5_w, p5_ch, p5_row = _tensor_hwc(p5_in)
    from .maxpool2d_geometry import capacity as pool_capacity, geometry
    p2d_nodes = nodes["maxpool2d"]
    p2d_caps = (pool_capacity([geometry(graph.tensors[n.inputs[0]].shape, n.attrs)
                               for n in p2d_nodes]) if p2d_nodes else {})
    p2d_h, p2d_w, p2d_ch, _ = _tensor_hwc(_named_tensors(
        graph, [t for n in p2d_nodes for t in (n.inputs[0], n.outputs[0])]))
    # LUT activation: 16B streaming, identity shape.
    act_tensors = _named_tensors(
        graph, [n for node in nodes["activation"]
                for n in list(node.inputs) + list(node.outputs)])
    act_h, act_w, act_ch, act_row = _tensor_hwc(act_tensors)
    rows: Dict[str, Any] = {
        "dense": {
            "used_by_graph": dense_used,
            "expanded": True,
            "max_h": int(allocated["height"]) if dense_used else 0,
            "max_w": int(allocated["width"]) if dense_used else 0,
            "max_cin": int(base_capacity["max_cin"]),
            "max_cout": int(base_capacity["max_cout"]),
            "max_ch": int(base_capacity["max_ch"]),
            "row_capacity_rule": "align16(align8(MAX_CIN_or_MAX_COUT) * MAX_W)",
        },
        "pool2x2": {
            "used_by_graph": bool(pool_names),
            "expanded": True,
            "max_h": int(pool_h),
            "max_w": int(pool_w),
            "max_ch": int(base_capacity["max_ch"]),
            "row_capacity_rule": "align16(align8(MAX_CH) * POOL_MAX_W)",
        },
        "eltwise": {
            "used_by_graph": bool(eltwise_used),
            "expanded": True,
            "streaming": True,
            "max_h": int(allocated["height"]),
            "max_w": int(allocated["width"]),
            "note": "16B streaming; no row buffer",
        },
        "reduce": {
            "used_by_graph": bool(reduce_used),
            "expanded": True,
            "streaming": True,
            "max_ch": int(base_capacity["max_ch"]),
            "note": "row-segment accumulate; MAX_CH is the only capacity",
        },
        # ---- batch-2 feature fusion families --------------------------------
        "copy": {
            "used_by_graph": bool(nodes["copy"]),
            "expanded": True,
            "max_h": int(copy_h) if nodes["copy"] else SMALL_ENGINE_MAX_H,
            "max_w": int(copy_w) if nodes["copy"] else SMALL_ENGINE_MAX_W,
            "max_ch": int(copy_ch) if nodes["copy"] else 16,
            "max_row_bytes": (int(copy_row) if nodes["copy"]
                              else SMALL_COPY_ROW_BYTES),
            "row_capacity_rule": "max(in,out) align16(align8(C) * W); one input "
                                 "row buffer + one output row staging buffer",
            "note": "Pad/Slice/Concat/Split real copies; bounded row/block staging, "
                    "never a whole-tensor buffer",
        },
        "resize": {
            "used_by_graph": bool(nodes["resize"]),
            "expanded": True,
            "max_h": int(resize_h) if nodes["resize"] else SMALL_ENGINE_MAX_H,
            "max_w": int(resize_w) if nodes["resize"] else SMALL_ENGINE_MAX_W,
            "max_ch": int(resize_ch) if nodes["resize"] else 16,
            "max_row_bytes": (int(resize_row) if nodes["resize"]
                              else SMALL_COPY_ROW_BYTES),
            "out_max_h": int(resize_oh),
            "out_max_w": int(resize_ow),
            "out_max_row_bytes": int(resize_out_row),
            "input_axis_limit": TARGET_MAX_AXIS // 2,
            "note": "capacity is the input axis (<=320 at x2/640); the doubled "
                    "output is covered by the allocated axes",
        },
        "pool5": {
            "used_by_graph": bool(nodes["pool5"]),
            "expanded": True,
            "max_h": int(p5_h) if nodes["pool5"] else SMALL_ENGINE_MAX_H,
            "max_w": int(p5_w) if nodes["pool5"] else SMALL_ENGINE_MAX_W,
            "max_ch": int(p5_ch) if nodes["pool5"] else 16,
            "max_row_bytes": int(p5_row) if nodes["pool5"] else SMALL_COPY_ROW_BYTES,
            "row_capacity_rule": "align16(align8(C) * W); 5-row sliding cache",
            "note": "one read of each input row; -128 spatial padding on chip",
        },
        "maxpool2d": {
            "used_by_graph": bool(p2d_nodes), "expanded": True,
            "max_h": p2d_h, "max_w": p2d_w, "max_ch": p2d_ch,
            "max_in_row_bytes": p2d_caps.get("max_in_row_bytes", 0),
            "out_max_row_bytes": p2d_caps.get("max_out_row_bytes", 0),
            "max_eh": p2d_caps.get("max_eh", 0),
            "cache_rows": p2d_caps.get("cache_rows", 0),
            "logical_storage_bytes": p2d_caps.get("logical_storage_bytes", 0),
        },
        "activation": {
            "used_by_graph": bool(nodes["activation"]),
            "expanded": True,
            "streaming": True,
            "max_h": int(act_h) if nodes["activation"] else SMALL_ENGINE_MAX_H,
            "max_w": int(act_w) if nodes["activation"] else SMALL_ENGINE_MAX_W,
            "max_ch": int(act_ch) if nodes["activation"] else 16,
            "max_row_bytes": int(act_row) if nodes["activation"] else SMALL_COPY_ROW_BYTES,
            "table_bytes": 256,
            "note": "16B streaming lookup; 256B table load counted separately",
        },
    }
    for name, reason in sorted(NOT_EXPANDED_ENGINES.items()):
        rows[name] = {
            "used_by_graph": False,
            "expanded": False,
            "max_w": 0 if name in STREAMING_ENGINES else SMALL_ENGINE_MAX_W,
            "reason": reason,
        }
    return rows


def check_row_capacity(engines: Dict[str, Any], capacity: Dict[str, int],
                       *, max_axis: int) -> Dict[str, Any]:
    """Prove every per-engine row is inside both the RTL width and the 16-bit
    byte fields before a single compile artifact is written."""
    def row_bytes(channels: int, width: int) -> int:
        return align_up(((int(channels) + 7) // 8) * 8 * int(width), 16)

    dense = engines["dense"]
    pool = engines["pool2x2"]
    rows: Dict[str, Any] = {}
    if dense["used_by_graph"]:
        in_bytes = row_bytes(capacity["max_cin"], dense["max_w"])
        out_bytes = row_bytes(capacity["max_cout"], dense["max_w"])
        rows["dense"] = {"in_row_bytes": in_bytes, "out_row_bytes": out_bytes,
                         "in_row_beats": in_bytes // 16, "out_row_beats": out_bytes // 16}
        for key, value in (("in_row_bytes", in_bytes), ("out_row_bytes", out_bytes)):
            if value > 0xFFFF:
                raise CompileError(E_SPATIAL_ROW,
                                   "dense %s=%d exceeds the 16-bit core row field" %
                                   (key, value),
                                   detail={"engine": "dense", key: value, "limit": 0xFFFF})
        if in_bytes // 16 < 2 or out_bytes // 16 < 2:
            raise CompileError(E_SPATIAL_ROW,
                               "dense row capacity is below 2 AXI beats",
                               detail={"in_row_beats": in_bytes // 16,
                                       "out_row_beats": out_bytes // 16})
    if pool["used_by_graph"]:
        in_bytes = row_bytes(pool["max_ch"], pool["max_w"])
        out_bytes = row_bytes(pool["max_ch"], (align_up(int(pool["max_w"]), 2)) // 2)
        rows["pool2x2"] = {"in_row_bytes": in_bytes, "out_row_bytes": out_bytes}
        if in_bytes > 0xFFFF or out_bytes > 0xFFFF:
            raise CompileError(E_SPATIAL_ROW,
                               "pool row capacity exceeds the 16-bit core row field",
                               detail={"engine": "pool2x2", "in_row_bytes": in_bytes,
                                       "out_row_bytes": out_bytes, "limit": 0xFFFF})
    def _check_16bit(engine: str, key: str, value: int) -> None:
        if value > 0xFFFF:
            raise CompileError(E_SPATIAL_ROW,
                               "%s %s=%d exceeds the 16-bit core row field" %
                               (engine, key, value),
                               detail={"engine": engine, key: value, "limit": 0xFFFF})
        if value < 16:
            raise CompileError(E_SPATIAL_ROW,
                               "%s %s=%d is below one AXI beat" % (engine, key, value),
                               detail={"engine": engine, key: value, "limit": 16})

    copy = engines.get("copy") or {}
    if copy.get("used_by_graph"):
        value = int(copy["max_row_bytes"])
        _check_16bit("copy", "max_row_bytes", value)
        rows["copy"] = {"max_row_bytes": value, "row_beats": value // 16,
                        "max_h": int(copy["max_h"]), "max_w": int(copy["max_w"]),
                        "max_ch": int(copy["max_ch"])}
    resize = engines.get("resize") or {}
    if resize.get("used_by_graph"):
        in_bytes = int(resize["max_row_bytes"])
        out_bytes = int(resize["out_max_row_bytes"])
        _check_16bit("resize", "in_row_bytes", in_bytes)
        _check_16bit("resize", "out_row_bytes", out_bytes)
        # x2: the doubled output must fit the user-allocated axes, not only the
        # versioned ceiling (task: output <=640 AND inside allocated).
        out_h, out_w = int(resize["out_max_h"]), int(resize["out_max_w"])
        if out_h > int(capacity["max_h"]) or out_w > int(capacity["max_w"]):
            raise CompileError(E_SPATIAL_LIMIT,
                               "resize output %dx%d exceeds the allocated %dx%d" %
                               (out_h, out_w, int(capacity["max_h"]),
                                int(capacity["max_w"])),
                               detail={"engine": "resize", "out_h": out_h,
                                       "out_w": out_w,
                                       "allocated_h": int(capacity["max_h"]),
                                       "allocated_w": int(capacity["max_w"])})
        rows["resize"] = {"in_row_bytes": in_bytes, "out_row_bytes": out_bytes,
                          "in_row_beats": in_bytes // 16,
                          "out_row_beats": out_bytes // 16,
                          "max_in_h": int(resize["max_h"]),
                          "max_in_w": int(resize["max_w"]),
                          "out_max_h": out_h, "out_max_w": out_w}
    pool5 = engines.get("pool5") or {}
    if pool5.get("used_by_graph"):
        value = int(pool5["max_row_bytes"])
        _check_16bit("pool5", "row_bytes", value)
        rows["pool5"] = {"in_row_bytes": value, "row_beats": value // 16,
                         "max_h": int(pool5["max_h"]), "max_w": int(pool5["max_w"]),
                         "max_ch": int(pool5["max_ch"])}
    pool2d = engines.get("maxpool2d") or {}
    if pool2d.get("used_by_graph"):
        in_bytes, out_bytes = int(pool2d["max_in_row_bytes"]), int(pool2d["out_max_row_bytes"])
        _check_16bit("maxpool2d", "in_row_bytes", in_bytes)
        _check_16bit("maxpool2d", "out_row_bytes", out_bytes)
        rows["maxpool2d"] = {"in_row_bytes": in_bytes, "out_row_bytes": out_bytes,
                             "max_eh": pool2d["max_eh"], "cache_rows": pool2d["cache_rows"],
                             "logical_storage_bytes": pool2d["logical_storage_bytes"]}
    act = engines.get("activation") or {}
    if act.get("used_by_graph"):
        value = int(act["max_row_bytes"])
        _check_16bit("activation", "row_bytes", value)
        rows["activation"] = {"in_row_bytes": value, "row_beats": value // 16,
                              "max_h": int(act["max_h"]), "max_w": int(act["max_w"]),
                              "max_ch": int(act["max_ch"]),
                              "table_bytes": int(act.get("table_bytes", 256))}
    for engine, row in engines.items():
        if not row.get("used_by_graph"):
            continue
        width = int(row.get("max_w") or 0)
        height = int(row.get("max_h") or 0)
        if width > max_axis or height > max_axis:
            raise CompileError(E_SPATIAL_LIMIT,
                               "engine %s capacity %dx%d exceeds the allocated %d" %
                               (engine, height, width, max_axis),
                               detail={"engine": engine, "height": height,
                                       "width": width, "limit": max_axis})
    return rows


def memory_limits(*, sim_ddr_limit_mib: int, arena_bytes: int, image_bytes: int,
                  snapshot_max_bytes: int) -> Dict[str, Any]:
    limit_bytes = int(sim_ddr_limit_mib) * 1024 * 1024
    if image_bytes > limit_bytes:
        raise CompileError(E_SPATIAL_MEMORY,
                           "DDR image %d bytes exceeds the simulation budget %d bytes" %
                           (image_bytes, limit_bytes),
                           detail={"image_bytes": image_bytes,
                                   "sim_ddr_limit_bytes": limit_bytes})
    if arena_bytes > limit_bytes:
        raise CompileError(E_SPATIAL_MEMORY,
                           "memory arena %d bytes exceeds the simulation budget %d" %
                           (arena_bytes, limit_bytes),
                           detail={"arena_bytes": arena_bytes,
                                   "sim_ddr_limit_bytes": limit_bytes})
    tb_mem_bytes = align_up(max(image_bytes, arena_bytes) + DDR_GUARD_BYTES, DDR_GUARD_BYTES)
    if tb_mem_bytes > limit_bytes:
        tb_mem_bytes = align_up(max(image_bytes, arena_bytes), DDR_GUARD_BYTES)
    return {
        "schema": "e1-spatial-memory-limits-1",
        "sim_ddr_limit_bytes": limit_bytes,
        "sim_ddr_limit_mib": int(sim_ddr_limit_mib),
        "arena_bytes": int(arena_bytes),
        "image_bytes": int(image_bytes),
        "ddr_guard_bytes": DDR_GUARD_BYTES,
        "tb_mem_bytes": int(tb_mem_bytes),
        "tb_mem_words": int(tb_mem_bytes // 16),
        "snapshot_max_bytes": int(snapshot_max_bytes),
        "snapshot_chunk_bytes": SNAPSHOT_MAX_CHUNK,
    }


def estimate_cycles(graph, stream: Dict[str, Any]) -> Dict[str, Any]:
    """Independent work/transaction upper bound for the simulation watchdog.

    The number is only a *budget*: it is not a performance claim and it is never
    used to gate correctness.  Records keep the formula so measured cycles can
    be compared against it.
    """
    issues = 0
    transactions = 0
    for record in stream.get("records") or []:
        node = None
        for candidate in graph.nodes:
            if candidate.id == record.get("node_id"):
                node = candidate
                break
        if node is None or not node.outputs:
            continue
        out_shape = graph.tensors[node.outputs[0]].shape
        if out_shape is None or len(out_shape) != 4:
            continue
        out_h, out_w = int(out_shape[2]), int(out_shape[3])
        cout = int(out_shape[1])
        cin = int(graph.tensors[node.inputs[0]].shape[1])
        icb = max(1, (cin + 7) // 8)
        ocb = max(1, (cout + 7) // 8)
        issues += out_h * out_w * icb * ocb
        out_bytes = out_h * out_w * (((cout + 7) // 8) * 8)
        in_bytes = out_h * out_w * (((cin + 7) // 8) * 8)
        transactions += max(1, (in_bytes + out_bytes) // 16)
    estimate = 8 * (issues * 2 + transactions * 2 + 4096)
    watchdog = max(20_000_000, int(estimate))
    return {
        "work_issues_estimate": int(issues),
        "transaction_beats_estimate": int(transactions),
        "cycle_upper_bound_estimate": int(estimate),
        "watchdog_cycles": int(watchdog),
        "watchdog_rule": "watchdog = max(20e6, 8*(2*issue_estimate + "
                         "2*beat_estimate + 4096)); derived from independent "
                         "work/transaction counts, not tuned to a run",
        "performance_claim": "none; simulation budget only",
    }


def tb_parameters(memory: Dict[str, Any], cycles: Dict[str, Any],
                  snapshot_count: int) -> Dict[str, Any]:
    return {
        "schema": "e1-spatial-tb-params-1",
        "TB_MEM_BYTES": int(memory["tb_mem_bytes"]),
        "TB_MEM_WORDS": int(memory["tb_mem_words"]),
        "TB_MAX_SNAPS": max(64, int(snapshot_count) + 16),
        "TB_SNAP_CHUNK": int(memory["snapshot_chunk_bytes"]),
        "TB_WATCHDOG_CYCLES": int(cycles["watchdog_cycles"]),
        "TB_DDR_BASE": "0x10000000",
    }


def tb_params_vh(parameters: Dict[str, Any]) -> str:
    bt = chr(96)
    lines = ["// generated by tools/onnx_gen/spatial_capacity.py -- do not edit",
             bt + "ifndef TB_SPATIAL_PARAMS_VH",
             bt + "define TB_SPATIAL_PARAMS_VH"]
    for key in ("TB_MEM_BYTES", "TB_MEM_WORDS", "TB_MAX_SNAPS", "TB_SNAP_CHUNK",
                "TB_WATCHDOG_CYCLES"):
        lines.append(bt + "define %s %d" % (key, int(parameters[key])))
    lines.append(bt + "define TB_DDR_BASE 32'h1000_0000")
    lines.append(bt + "endif")
    return "\n".join(lines) + "\n"
