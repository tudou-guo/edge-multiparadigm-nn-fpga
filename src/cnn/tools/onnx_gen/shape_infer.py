"""Static shape/rank inference for the E1 operator subset.

No dynamic dimensions, unknown ranks or implicit ONNX broadcasting are accepted.
The legacy MNIST adapter also uses the ABI geometry helpers at the bottom of
this module so IR tensor bytes and the frozen NHWC8_A16 contract stay aligned.
"""
from __future__ import annotations

from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .graph_ir import (
    CompileError,
    E_CFG,
    E_DTYPE_MISMATCH,
    E_LIMIT,
    E_SPATIAL_LIMIT,
    E_QUANT_MISMATCH,
    E_SHAPE_MISMATCH,
    E_UNKNOWN_RANK,
    E_UNSUPPORTED_ATTR,
    E_UNSUPPORTED_BROADCAST,
    E_UNSUPPORTED_LAYOUT,
    E_UNSUPPORTED_OP,
    Tensor,
    align_up,
)
DTYPE_BYTES = {
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint16": 2,
    "uint8": 1,
    "bool": 1,
}

CHANNEL_BLOCK = 8
# Depthwise 3x3 hardware limits (S1_DW contract): 1 <= C <= 16, 1 <= H,W <= 32.
DW_MAX_CHANNELS = 16
DW_MAX_HW = 32
# Semantic-float parsing flag for LUT activation intake (S2).
# When True, _infer_mul additionally accepts float32/float16 same-shape Mul
# as semantic SiLU-group candidates. Hardware deployment path never sets it.
SEMANTIC_FLOAT_OK = False
ABI_ALIGNMENT = 16
MAX_CHANNELS = 64
MAX_WIDTH = 256
MAX_COPY_BYTES = 16384

#: explicit numeric profile id of the v0.2 real-YOLO affine build.  Profile-gated
#: checks below only fire when a tensor carries this id, so every legacy graph
#: (whose quantisation rows have no "profile" key) keeps its exact behaviour.
AFFINE_PROFILE = "affine-int8-v1"

# --- v0.2 affine-int8-v1 capacity context ------------------------------------
# The historical MAX_CHANNELS/MAX_COPY_BYTES gates stay the default so every
# legacy call keeps its exact accepted/rejected set.  An affine build supplies
# a versioned capability dict (channel_ceiling / pool5_max_ch / resize_max_ch)
# and the per-engine capacities are then enforced after lowering.
def channel_limit() -> int:
    caps = _ACTIVE_SPATIAL_CAPS
    return int(MAX_CHANNELS if caps is None else caps.get("channel_ceiling", MAX_CHANNELS))


def pool5_channel_limit() -> int:
    caps = _ACTIVE_SPATIAL_CAPS
    return int(16 if caps is None else caps.get("pool5_max_ch", 16))


def resize_channel_limit() -> int:
    caps = _ACTIVE_SPATIAL_CAPS
    return int(16 if caps is None else caps.get("resize_max_ch", 16))


# --- v0.2 spatial capability context -----------------------------------------
# The historical limits above stay the *default* so every legacy call keeps its
# exact accepted/rejected set.  A spatial build temporarily raises the pure
# shape/geometry ceiling; per-engine capability is then enforced after lowering
# by tools/onnx_gen/spatial_capacity.validate_graph_support (so the compiler
# rejects a not-yet-expanded engine with a structured reason instead of having
# the frontend silently clamp a legal static shape).
_ACTIVE_SPATIAL_CAPS = None


class spatial_caps:
    """Temporarily activate v0.2 spatial math limits for one inference pass."""

    def __init__(self, caps):
        self._caps = caps
        self._previous = None

    def __enter__(self):
        global _ACTIVE_SPATIAL_CAPS
        self._previous = _ACTIVE_SPATIAL_CAPS
        _ACTIVE_SPATIAL_CAPS = dict(self._caps)
        return _ACTIVE_SPATIAL_CAPS

    def __exit__(self, *exc_info):
        global _ACTIVE_SPATIAL_CAPS
        _ACTIVE_SPATIAL_CAPS = self._previous
        return False


def spatial_mode() -> bool:
    return _ACTIVE_SPATIAL_CAPS is not None


def width_limit() -> int:
    caps = _ACTIVE_SPATIAL_CAPS
    return int(MAX_WIDTH if caps is None else caps.get("width_ceiling", MAX_WIDTH))


def height_limit() -> int:
    # Legacy builds never bounded H in the frontend (only the 16-bit ABI field
    # does); keep that exactly.  A spatial build bounds it by the allocated axis.
    caps = _ACTIVE_SPATIAL_CAPS
    return int(65535 if caps is None else caps.get("feature_max_h", MAX_WIDTH))


def hw_limit() -> int:
    caps = _ACTIVE_SPATIAL_CAPS
    return int(MAX_WIDTH if caps is None else caps.get("feature_max_h", MAX_WIDTH))


def width_error_code() -> str:
    return E_SPATIAL_LIMIT if spatial_mode() else E_LIMIT


def prod(values: Iterable[int]) -> int:
    return reduce(mul, (int(v) for v in values), 1)


def logical_nbytes(shape: Sequence[int], dtype: str) -> Optional[int]:
    width = DTYPE_BYTES.get(dtype)
    return prod(shape) * width if width is not None else None


def layout_for_rank(rank: int) -> str:
    return {1: "C", 2: "NC", 3: "NCW", 4: "NCHW"}.get(int(rank), "LOGICAL")


def _require_rank(node, tensor: Tensor, rank: int) -> None:
    if tensor.shape is None:
        raise CompileError(E_UNKNOWN_RANK, "tensor %s has no static shape" % tensor.name,
                           node_id=node.id)
    if len(tensor.shape) != rank:
        raise CompileError(E_SHAPE_MISMATCH,
                           "tensor %s rank %d != required rank %d" %
                           (tensor.name, len(tensor.shape), rank), node_id=node.id)


def _out(tensor: Tensor, shape: Sequence[int], dtype: str, layout: Optional[str] = None) -> Dict[str, Any]:
    shape_t = tuple(int(v) for v in shape)
    return {
        "shape": shape_t,
        "dtype": dtype,
        "layout": layout or layout_for_rank(len(shape_t)),
        "logical_bytes": logical_nbytes(shape_t, dtype),
    }


def _infer_conv(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) not in (2, 3):
        raise CompileError(E_SHAPE_MISMATCH, "Conv requires input, weight and optional bias",
                           node_id=node.id)
    x, weight = inputs[0], inputs[1]
    if x.dtype not in ("float32", "float16", "int8", "uint8"):
        raise CompileError(E_DTYPE_MISMATCH, "Conv activation dtype %s is unsupported" % x.dtype,
                           node_id=node.id)
    _require_rank(node, x, 4)
    _require_rank(node, weight, 4)
    if weight.kind != "initializer":
        raise CompileError(E_CFG, "Conv weight must be an initializer", node_id=node.id)
    n, cin, h, w = x.shape
    if n != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "Conv batch N must be 1", node_id=node.id,
                           attribute="N")
    group = int(node.attrs.get("group", 1))
    if group != 1:
        # Standard ONNX grouped Conv is accepted only as depthwise
        # (group == Cin == Cout, multiplier 1); everything else is rejected with
        # a structured error.  The node is rewritten to DepthwiseConv later by
        # the lowering pass once the source semantics are proven here.
        return _infer_depthwise_conv(node, x, weight, inputs)
    cout, wcin, kh, kw = weight.shape
    if (kh, kw) != tuple(node.attrs["kernel_shape"]):
        raise CompileError(E_SHAPE_MISMATCH, "Conv weight kernel %dx%d != attr %s" %
                           (kh, kw, node.attrs["kernel_shape"]), node_id=node.id)
    if wcin != cin:
        raise CompileError(E_SHAPE_MISMATCH, "Conv weight Cin %d != input C %d" % (wcin, cin),
                           node_id=node.id)
    if weight.dtype != x.dtype:
        raise CompileError(E_DTYPE_MISMATCH, "Conv input/weight dtype mismatch: %s vs %s" %
                           (x.dtype, weight.dtype), node_id=node.id)
    if len(inputs) == 3:
        bias = inputs[2]
        _require_rank(node, bias, 1)
        if bias.kind != "initializer":
            raise CompileError(E_CFG, "Conv bias must be an initializer", node_id=node.id)
        if bias.shape != (cout,):
            raise CompileError(E_SHAPE_MISMATCH, "Conv bias shape %s != (%d,)" %
                               (bias.shape, cout), node_id=node.id)
        if bias.dtype not in (x.dtype, "int32"):
            raise CompileError(E_DTYPE_MISMATCH, "Conv bias dtype %s is incompatible with %s" %
                               (bias.dtype, x.dtype), node_id=node.id)
    ch_lim = channel_limit()
    if not (1 <= cin <= ch_lim and 1 <= cout <= ch_lim):
        raise CompileError(E_LIMIT, "Conv channels Cin=%d Cout=%d exceed 1..%d" %
                           (cin, cout, ch_lim), node_id=node.id)
    width_cap = width_limit()
    if w > width_cap:
        raise CompileError(width_error_code(), "Conv W=%d exceeds %d" % (w, width_cap),
                           node_id=node.id,
                           detail={"width": w, "limit": width_cap,
                                   "spatial_mode": spatial_mode()})
    height_cap = height_limit()
    if h > height_cap:
        raise CompileError(width_error_code(), "Conv H=%d exceeds %d" % (h, height_cap),
                           node_id=node.id,
                           detail={"height": h, "limit": height_cap,
                                   "spatial_mode": spatial_mode()})
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    if hout <= 0 or wout <= 0:
        raise CompileError(E_CFG, "Conv output shape is degenerate", node_id=node.id)
    return _out(x, (1, cout, hout, wout), x.dtype, "NCHW")


def _depthwise_geometry(node, n, c, h, w, kh, kw):
    """Validate the frozen depthwise contract and return the output geometry."""
    auto_pad = str(node.attrs.get("auto_pad", "NOTSET"))
    if auto_pad != "NOTSET":
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "depthwise auto_pad=%s is unsupported" % auto_pad,
                           node_id=node.id, attribute="auto_pad")
    if (kh, kw) != (3, 3):
        raise CompileError(E_UNSUPPORTED_ATTR, "depthwise kernel must be 3x3",
                           node_id=node.id, attribute="kernel_shape")
    if [int(v) for v in node.attrs.get("kernel_shape", [3, 3])] != [3, 3]:
        raise CompileError(E_UNSUPPORTED_ATTR, "depthwise kernel_shape must be [3,3]",
                           node_id=node.id, attribute="kernel_shape")
    dilation = [int(v) for v in node.attrs.get("dilations", [1, 1])]
    if dilation != [1, 1]:
        raise CompileError(E_UNSUPPORTED_ATTR, "depthwise dilation must be 1",
                           node_id=node.id, attribute="dilations")
    strides = [int(v) for v in node.attrs.get("strides", [1, 1])]
    if len(strides) != 2 or strides[0] != strides[1] or strides[0] not in (1, 2):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "depthwise stride must be symmetric 1 or 2",
                           node_id=node.id, attribute="strides")
    pads = [int(v) for v in node.attrs.get("pads", [0, 0, 0, 0])]
    if len(pads) != 4 or len(set(pads)) != 1 or pads[0] not in (0, 1):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "depthwise pad must be symmetric 0 or 1",
                           node_id=node.id, attribute="pads")
    if n != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "depthwise batch N must be 1", node_id=node.id,
                           attribute="N")
    if not (1 <= c <= DW_MAX_CHANNELS):
        raise CompileError(E_LIMIT, "depthwise C=%d exceeds 1..%d" %
                           (c, DW_MAX_CHANNELS), node_id=node.id)
    if not (1 <= h <= DW_MAX_HW and 1 <= w <= DW_MAX_HW):
        raise CompileError(E_LIMIT, "depthwise H/W=%dx%d exceeds 1..%d" %
                           (h, w, DW_MAX_HW), node_id=node.id)
    stride, pad = strides[0], pads[0]
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    if hout <= 0 or wout <= 0:
        raise CompileError(E_CFG, "depthwise output shape is degenerate", node_id=node.id)
    return {"stride": stride, "pad": pad, "hout": hout, "wout": wout,
            "dilations": dilation, "strides": strides, "pads": pads}


def _require_depthwise_bias(node, inputs: Sequence[Tensor], cout: int) -> None:
    if len(inputs) != 3:
        raise CompileError(E_CFG,
                           "depthwise Conv requires an INT32 bias initializer (inputs: %d)" %
                           len(inputs), node_id=node.id)
    bias = inputs[2]
    _require_rank(node, bias, 1)
    if bias.kind != "initializer":
        raise CompileError(E_CFG, "depthwise bias must be an initializer", node_id=node.id)
    if tuple(bias.shape) != (int(cout),):
        raise CompileError(E_SHAPE_MISMATCH,
                           "depthwise bias shape %s != (%d,)" % (bias.shape, cout),
                           node_id=node.id)
    if bias.dtype != "int32":
        raise CompileError(E_DTYPE_MISMATCH,
                           "depthwise bias must be INT32, got %s" % bias.dtype,
                           node_id=node.id)


def _infer_depthwise_conv(node, x: Tensor, weight: Tensor,
                          inputs: Sequence[Tensor]) -> Dict[str, Any]:
    """Validate source ONNX grouped Conv as depthwise and infer its output.

    Only group == Cin == Cout == C with weight [C, 1, 3, 3] is a depthwise
    convolution; any other grouped shape is a structured rejection (no guessed
    interpretation and no silent dense fallback).
    """
    n, cin, h, w = x.shape
    cout, wcin, kh, kw = weight.shape
    group = int(node.attrs.get("group", 1))
    if not (group == cin == cout):
        raise CompileError(
            E_UNSUPPORTED_ATTR,
            "grouped Conv is supported only as depthwise (group==Cin==Cout); "
            "got group=%d Cin=%d Cout=%d" % (group, cin, cout),
            node_id=node.id, attribute="group")
    if wcin != cin // group:
        raise CompileError(E_SHAPE_MISMATCH,
                           "depthwise weight shape %s must be [%d, 1, 3, 3]" %
                           (tuple(weight.shape), cout), node_id=node.id)
    if x.dtype != "int8" or weight.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "depthwise requires INT8 activation/weight, got %s/%s" %
                           (x.dtype, weight.dtype), node_id=node.id)
    geometry = _depthwise_geometry(node, n, cin, h, w, kh, kw)
    _require_depthwise_bias(node, inputs, cout)
    node.metadata["depthwise_candidate"] = True
    node.metadata["depthwise_group"] = group
    node.metadata["depthwise_geometry"] = dict(geometry)
    return _out(x, (1, cout, geometry["hout"], geometry["wout"]), x.dtype, "NCHW")


def _infer_depthwise(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    """Shape inference for the lowered DepthwiseConv node (PL hardware op)."""
    if len(inputs) != 3:
        raise CompileError(E_SHAPE_MISMATCH,
                           "DepthwiseConv requires input, weight and bias", node_id=node.id)
    x, weight = inputs[0], inputs[1]
    _require_rank(node, x, 4)
    _require_rank(node, weight, 4)
    if weight.kind != "initializer":
        raise CompileError(E_CFG, "DepthwiseConv weight must be an initializer",
                           node_id=node.id)
    n, c, h, w = x.shape
    cout, wcin, kh, kw = weight.shape
    if int(node.attrs.get("group", 1)) != c:
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "DepthwiseConv group %s must equal C=%d" %
                           (node.attrs.get("group"), c), node_id=node.id, attribute="group")
    if tuple(weight.shape) != (c, 1, kh, kw):
        raise CompileError(E_SHAPE_MISMATCH,
                           "DepthwiseConv weight shape %s must be [%d, 1, 3, 3]" %
                           (tuple(weight.shape), c), node_id=node.id)
    if x.dtype != "int8" or weight.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "DepthwiseConv requires INT8 activation/weight, got %s/%s" %
                           (x.dtype, weight.dtype), node_id=node.id)
    geometry = _depthwise_geometry(node, n, c, h, w, kh, kw)
    _require_depthwise_bias(node, inputs, cout)
    return _out(x, (1, cout, geometry["hout"], geometry["wout"]), x.dtype, "NCHW")


def _infer_maxpool5(node, x: Tensor) -> Dict[str, Any]:
    """5x5 stride 1 pads 2 shape-preserving signed INT8 max window."""
    if [int(v) for v in node.attrs.get("strides", [1, 1])] != [1, 1]:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool 5x5 stride must be [1,1]",
                           node_id=node.id, attribute="strides")
    if [int(v) for v in node.attrs.get("pads", [2, 2, 2, 2])] != [2, 2, 2, 2]:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool 5x5 pads must be [2,2,2,2]",
                           node_id=node.id, attribute="pads")
    if [int(v) for v in node.attrs.get("dilations", [1, 1])] != [1, 1]:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool 5x5 dilation must be 1",
                           node_id=node.id, attribute="dilations")
    if int(node.attrs.get("ceil_mode", 0)) != 0:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool 5x5 ceil_mode must be 0",
                           node_id=node.id, attribute="ceil_mode")
    if int(node.attrs.get("storage_order", 0)) != 0:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool 5x5 storage_order must be 0",
                           node_id=node.id, attribute="storage_order")
    n, c, h, w = (int(v) for v in x.shape)
    p5_lim = pool5_channel_limit()
    if not (1 <= c <= p5_lim):
        raise CompileError(E_LIMIT, "MaxPool 5x5 C=%d is outside 1..%d" % (c, p5_lim),
                           node_id=node.id)
    if not (1 <= h and 1 <= w):
        raise CompileError(E_LIMIT, "MaxPool 5x5 H/W=%dx%d is degenerate" % (h, w),
                           node_id=node.id)
    h_cap = height_limit()
    if h > h_cap:
        raise CompileError(width_error_code(),
                           "MaxPool 5x5 H=%d exceeds %d" % (h, h_cap), node_id=node.id,
                           detail={"height": h, "limit": h_cap,
                                   "spatial_mode": spatial_mode()})
    if w > width_limit():
        raise CompileError(width_error_code(),
                           "MaxPool 5x5 W=%d exceeds %d" % (w, width_limit()),
                           node_id=node.id,
                           detail={"width": w, "limit": width_limit(),
                                   "spatial_mode": spatial_mode()})
    node.metadata.update({
        "pool_kernel": [5, 5],
        "pool_strides": [1, 1],
        "pool_pads": [2, 2, 2, 2],
        "ceil_mode": 0,
        "padding_value": -128,
    })
    node.lowering = "maxpool5x5_core_v1"
    return _out(x, (1, c, h, w), x.dtype, "NCHW")


def _infer_maxpool(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "MaxPool requires exactly one input", node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    if len(node.outputs) != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool indices output is unsupported", node_id=node.id)
    from .maxpool2d_geometry import geometry
    g = geometry(x.shape, node.attrs)
    if "legacy_layer_index" not in node.metadata:
        node.lowering = "maxpool2d_generic_v1"
    return _out(x, (1, g.channels, g.hout, g.wout), x.dtype, "NCHW")


RESIZE_FACTOR = 2
MAX_RESIZE_IN_HW = 16
MAX_RESIZE_OUT_HW = 32


def _infer_resize(node, inputs: Sequence[Tensor],
                  numeric_constants: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    """Validate the identity-arithmetic nearest/asymmetric/floor x2 upsample.

    The compile-time roi/scales/sizes values are looked up through the ONNX
    positional slots recorded by the frontend, never through node.inputs order,
    because an omitted optional slot shifts the positional list.
    """
    if not (1 <= len(inputs) <= 4):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Resize requires a data input and optional roi/scales/sizes",
                           node_id=node.id)
    x = inputs[0]
    if x.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "Resize supports signed INT8 only, got %s" % x.dtype,
                           node_id=node.id)
    _require_rank(node, x, 4)
    n, c, h, w = (int(v) for v in x.shape)
    if n != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "Resize batch N must be 1", node_id=node.id,
                           attribute="N")
    rsz_lim = resize_channel_limit()
    if not (1 <= c <= rsz_lim):
        raise CompileError(E_LIMIT, "Resize C=%d is outside 1..%d" % (c, rsz_lim),
                           node_id=node.id)
    if spatial_mode():
        # v0.2 batch 2: the legal range is derived from the versioned output
        # ceiling (output = 2*input <= allocated axis).  Input 320 is not a new
        # permanent architecture constant; 321 (output 642) is rejected here and
        # the independent auto-derived axis ceiling in spatial_capacity.resolve
        # rejects any other construction that would exceed 640.
        in_h_cap = height_limit() // RESIZE_FACTOR
        in_w_cap = width_limit() // RESIZE_FACTOR
        if not (1 <= h <= in_h_cap and 1 <= w <= in_w_cap):
            raise CompileError(E_SPATIAL_LIMIT,
                               "Resize input H/W=%dx%d outside 1..%dx%d "
                               "(x2 output must stay within the allocated axes)" %
                               (h, w, in_h_cap, in_w_cap), node_id=node.id,
                               detail={"input_h": h, "input_w": w,
                                       "input_h_limit": in_h_cap,
                                       "input_w_limit": in_w_cap,
                                       "output_h": RESIZE_FACTOR * h,
                                       "output_w": RESIZE_FACTOR * w,
                                       "spatial_mode": True})
    else:
        if not (1 <= h <= MAX_RESIZE_IN_HW and 1 <= w <= MAX_RESIZE_IN_HW):
            raise CompileError(E_LIMIT,
                               "Resize input H/W=%dx%d is outside 1..%d (output H/W <= %d)" %
                               (h, w, MAX_RESIZE_IN_HW, MAX_RESIZE_OUT_HW), node_id=node.id)
        if w > MAX_WIDTH:
            raise CompileError(E_LIMIT, "Resize input W=%d exceeds %d" % (w, MAX_WIDTH),
                               node_id=node.id)
    slots = node.metadata.get("resize_slots")
    if not isinstance(slots, dict):
        raise CompileError(E_CFG,
                           "Resize node lacks resize_slots metadata from the ONNX frontend",
                           node_id=node.id)
    roi_name = slots.get("roi")
    scales_name = slots.get("scales")
    sizes_name = slots.get("sizes")
    if roi_name is not None:
        roi = _constant_numeric(node, numeric_constants, roi_name, "Resize roi")
        if any(float(value) != 0.0 for value in roi):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize roi must be absent, empty or all zeros for the "
                               "asymmetric mapping", node_id=node.id, attribute="roi")
    scales = sizes = None
    if scales_name is not None:
        scales = _constant_numeric(node, numeric_constants, scales_name, "Resize scales")
        if len(scales) != 4:
            raise CompileError(E_SHAPE_MISMATCH,
                               "Resize scales must contain 4 values, got %d" % len(scales),
                               node_id=node.id)
        expected_scales = [1.0, 1.0, float(RESIZE_FACTOR), float(RESIZE_FACTOR)]
        if [float(value) for value in scales] != expected_scales:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize scales must be exactly [1,1,2,2] (x2 only), got %s" %
                               [float(value) for value in scales],
                               node_id=node.id, attribute="scales")
    if sizes_name is not None:
        sizes = _constant_ints(node, numeric_constants, sizes_name, "Resize sizes")
        if len(sizes) != 4:
            raise CompileError(E_SHAPE_MISMATCH,
                               "Resize sizes must contain 4 values, got %d" % len(sizes),
                               node_id=node.id)
        expected_sizes = [1, c, RESIZE_FACTOR * h, RESIZE_FACTOR * w]
        if sizes != expected_sizes:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize sizes must be [1,C,2H,2W]=%s, got %s" %
                               (expected_sizes, sizes), node_id=node.id, attribute="sizes")
    if scales is None and sizes is None:
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Resize requires compile-time scales or sizes metadata",
                           node_id=node.id)
    out_h, out_w = RESIZE_FACTOR * h, RESIZE_FACTOR * w
    if spatial_mode():
        if out_h > height_limit() or out_w > width_limit():
            raise CompileError(E_SPATIAL_LIMIT,
                               "Resize output H/W=%dx%d exceeds allocated %dx%d" %
                               (out_h, out_w, height_limit(), width_limit()),
                               node_id=node.id,
                               detail={"output_h": out_h, "output_w": out_w,
                                       "height_limit": height_limit(),
                                       "width_limit": width_limit()})
    elif out_h > MAX_RESIZE_OUT_HW or out_w > MAX_RESIZE_OUT_HW:
        raise CompileError(E_LIMIT, "Resize output H/W=%dx%d exceeds %d" %
                           (out_h, out_w, MAX_RESIZE_OUT_HW), node_id=node.id)
    node.metadata.update({
        "resize_scale_hw": [RESIZE_FACTOR, RESIZE_FACTOR],
        "resize_in_hw": [h, w],
        "resize_out_hw": [out_h, out_w],
        "copy_required": False,
    })
    node.lowering = "resize_nearest2_core_v1"
    return _out(x, (1, c, out_h, out_w), x.dtype, "NCHW")


def _infer_average_pool(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "AveragePool requires exactly one input",
                           node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    n, c, h, w = x.shape
    if n != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool batch N must be 1", node_id=node.id,
                           attribute="N")
    if c > channel_limit():
        raise CompileError(E_LIMIT, "AveragePool C=%d exceeds %d" % (c, channel_limit()),
                           node_id=node.id)
    width_cap = width_limit()
    if w > width_cap:
        raise CompileError(width_error_code(),
                           "AveragePool W=%d exceeds %d" % (w, width_cap),
                           node_id=node.id,
                           detail={"width": w, "limit": width_cap,
                                   "spatial_mode": spatial_mode()})
    if h % 2 or w % 2:
        # Both restrictions are explicit: the core never drops a partial window.
        raise CompileError(E_CFG,
                           "AveragePool 2x2/s2 pad0 requires even H/W (H=%d W=%d)" % (h, w),
                           node_id=node.id)
    node.metadata.update({
        "pool_kernel": [2, 2],
        "pool_strides": [2, 2],
        "pool_pads": [0, 0, 0, 0],
        "ceil_mode": 0,
        "count_include_pad": 0,
        "rounding": "ties_away_div4",
    })
    node.lowering = "avgpool2x2_core_v1"
    return _out(x, (1, c, h // 2, w // 2), x.dtype, "NCHW")


def _infer_mul(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 2:
        raise CompileError(E_SHAPE_MISMATCH, "Mul requires exactly two inputs", node_id=node.id)
    a, b = inputs
    _require_rank(node, a, 4)
    _require_rank(node, b, 4)
    if (node.metadata or {}).get("semantic_float") is True or SEMANTIC_FLOAT_OK:
        # Semantic SiLU-group float Mul: shapes must match, dtypes float32/float16, N=1.
        if a.dtype not in ("float32", "float16") or b.dtype not in ("float32", "float16"):
            raise CompileError(E_DTYPE_MISMATCH,
                               "semantic Mul supports float32/float16 only, got %s x %s" % (a.dtype, b.dtype),
                               node_id=node.id)
        if tuple(a.shape) != tuple(b.shape):
            raise CompileError(E_UNSUPPORTED_BROADCAST,
                               "semantic Mul requires identical static shapes, got %s vs %s" %
                               (tuple(a.shape), tuple(b.shape)), node_id=node.id)
        if int(a.shape[0]) != 1:
            raise CompileError(E_UNSUPPORTED_ATTR, "semantic Mul batch N must be 1", node_id=node.id,
                               attribute="N")
        node.lowering = "semantic_mul_float"
        return _out(a, tuple(a.shape), str(a.dtype), "NCHW")
    if a.dtype != "int8" or b.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "Mul supports signed INT8 only, got %s x %s" % (a.dtype, b.dtype),
                           node_id=node.id)
    if tuple(a.shape) != tuple(b.shape):
        raise CompileError(E_UNSUPPORTED_BROADCAST,
                           "Mul requires identical static shapes, got %s vs %s" %
                           (tuple(a.shape), tuple(b.shape)), node_id=node.id)
    if int(a.shape[0]) != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "Mul batch N must be 1", node_id=node.id,
                           attribute="N")
    if int(a.shape[1]) > channel_limit() or int(a.shape[3]) > MAX_WIDTH:
        raise CompileError(E_LIMIT, "Mul shape exceeds core channel/width limit",
                           node_id=node.id)
    node.lowering = "int8_mul_core_v1"
    return _out(a, tuple(a.shape), a.dtype, "NCHW")


def _infer_batch_norm(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 5:
        raise CompileError(E_SHAPE_MISMATCH,
                           "BatchNormalization requires X, scale, B, mean and var",
                           node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    if x.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "BatchNormalization fold supports INT8 X only, got %s" % x.dtype,
                           node_id=node.id)
    if int(x.shape[0]) != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "BatchNormalization batch N must be 1",
                           node_id=node.id, attribute="N")
    cout = int(x.shape[1])
    for tensor, what in zip(inputs[1:], ("scale", "B", "mean", "var")):
        if tensor.kind not in ("initializer", "constant"):
            if (tensor.source or {}).get("kind") != "compile_time_constant":
                raise CompileError(E_UNSUPPORTED_ATTR,
                                   "BatchNormalization %s must be a static initializer" % what,
                                   node_id=node.id, attribute=what)
        if tensor.shape is None or len(tensor.shape) != 1 or int(tensor.shape[0]) != cout:
            raise CompileError(E_SHAPE_MISMATCH,
                               "BatchNormalization %s shape %s is not (%d,)" %
                               (what, tensor.shape, cout), node_id=node.id)
    node.metadata.update({"bn_epsilon": float(node.attrs["epsilon"]),
                          "bn_channels": cout,
                          "bn_fold_required": True})
    node.lowering = "fold_into_conv_requant"
    return _out(x, tuple(x.shape), x.dtype, "NCHW")


def _require_p1_activation(node, tensor: Tensor, what: str) -> None:
    if tensor.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "%s requires INT8, got %s" % (what, tensor.dtype),
                           node_id=node.id)
    _require_rank(node, tensor, 4)
    if int(tensor.shape[0]) != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s batch N must be 1" % what,
                           node_id=node.id, attribute="N")


def _constant_numeric(node, numeric_constants: Dict[str, Sequence[float]], name: str,
                      what: str) -> List[float]:
    if name not in numeric_constants:
        raise CompileError(E_CFG, "%s %s is not a static initializer constant" % (what, name),
                           node_id=node.id)
    return list(numeric_constants[name])


def _constant_ints(node, numeric_constants: Dict[str, Sequence[float]], name: str,
                   what: str) -> List[int]:
    values = _constant_numeric(node, numeric_constants, name, what)
    out: List[int] = []
    for value in values:
        if isinstance(value, bool):
            raise CompileError(E_UNSUPPORTED_ATTR, "%s %s must be integers" % (what, name),
                               node_id=node.id)
        if isinstance(value, int):
            out.append(int(value))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise CompileError(E_UNSUPPORTED_ATTR, "%s %s must be integers" % (what, name),
                               node_id=node.id)
        if not number.is_integer():
            raise CompileError(E_UNSUPPORTED_ATTR, "%s %s must be integers" % (what, name),
                               node_id=node.id)
        out.append(int(number))
    return out


def _affine_concat_requant(node, quant: Sequence[Dict[str, Any]]) -> bool:
    """True when this Concat is an explicit affine-profile per-input requantisation.

    affine-int8-v1 Concat is a copy-v2 record: every input keeps its own real
    domain and the engine applies one explicit rescale per input (the contract
    metadata node.metadata["concat_quant"] carries zp_in/scale_in/M0/SHIFT for
    each one), so uint8-domain scale/zero-point equality is *not* required in
    that profile.  Encoding (signed INT8, materialised) must still be uniform,
    and the legacy profile keeps the historical same-domain gate byte for byte.
    """
    if node.op_type != "Concat" or node.metadata.get("affine") is not True:
        return False
    if not isinstance(node.metadata.get("concat_quant"), dict):
        return False
    for item in quant:
        if (item.get("profile") != AFFINE_PROFILE
                or item.get("dtype") != "int8"
                or item.get("materialized") is not True):
            return False
    return True


def _check_quant_compatible(node, tensors: Sequence[Tensor]) -> None:
    quant = [tensor.quantization for tensor in tensors]
    if all(item is None for item in quant):
        return
    if any(item is None for item in quant):
        raise CompileError(E_QUANT_MISMATCH,
                           "%s inputs mix known and unknown quantization metadata" % node.op_type,
                           node_id=node.id)
    if _affine_concat_requant(node, quant):
        # different real domains are legal here: the recompute is explicit
        return
    for key in ("scale", "zero_point", "dtype"):
        values = {item.get(key) for item in quant}
        if len(values) != 1:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s input quant mismatch on %s" % (node.op_type, key),
                               node_id=node.id)


def _affine_identity_domain_error(node, label: str, detail: Dict[str, Any]) -> None:
    raise CompileError(
        E_QUANT_MISMATCH,
        "%s is identity-arithmetic in profile %s: input and output must share one "
        "real domain (same scale, same signed zero point, same signed-INT8 encoding); "
        "differing domains need an explicit requantisation, which this op does not "
        "implement" % (label, AFFINE_PROFILE),
        node_id=node.id, detail=detail)


def _check_affine_identity_domain(graph, node, label: str) -> None:
    """Profile gate: Resize / MaxPool are identity-arithmetic in affine-int8-v1.

    Both ops copy or max the *same* signed INT8 codes, so the real numeric domain
    (scale, real signed zero point, storage encoding) must be identical on input
    and output.  A nonzero signed zero point is legal: the identity only needs
    equality, not zp == 0 (that legacy restriction belongs to the signed
    zero-point-0 contract, not to this profile).  Legacy graphs (no "profile" on
    the quantisation row) are not affected at all.
    """
    if len(node.inputs) < 1 or len(node.outputs) < 1:
        return
    x = graph.tensors.get(node.inputs[0])
    y = graph.tensors.get(node.outputs[0])
    if x is None or y is None:
        return
    qx = x.quantization or {}
    qy = y.quantization or {}
    if qx.get("profile") != AFFINE_PROFILE or qy.get("profile") != AFFINE_PROFILE:
        return
    detail = {
        "op": node.op_type,
        "input": x.name, "output": y.name,
        "input_dtype": x.dtype, "output_dtype": y.dtype,
        "input_scale": qx.get("scale"), "output_scale": qy.get("scale"),
        "input_zero_point_s8": qx.get("zero_point_s8"),
        "output_zero_point_s8": qy.get("zero_point_s8"),
        "profile": AFFINE_PROFILE,
    }
    if x.dtype != "int8" or y.dtype != "int8":
        _affine_identity_domain_error(node, label, dict(detail, reason="dtype"))
    if qx.get("materialized") is False or qy.get("materialized") is False:
        _affine_identity_domain_error(node, label, dict(detail, reason="materialized"))
    if "scale" not in qx or "scale" not in qy:
        _affine_identity_domain_error(node, label, dict(detail, reason="missing_scale"))
    if ("zero_point_s8" not in qx or "zero_point_s8" not in qy):
        _affine_identity_domain_error(
            node, label, dict(detail, reason="missing_zero_point_s8"))
    if float(qx["scale"]) != float(qy["scale"]):
        _affine_identity_domain_error(node, label, dict(detail, reason="scale"))
    if int(qx["zero_point_s8"]) != int(qy["zero_point_s8"]):
        _affine_identity_domain_error(node, label, dict(detail, reason="zero_point"))


def _check_copy_core_size(node, name: str, shape: Sequence[int]) -> int:
    size = int(nhwc8_a16_storage(shape)["bytes"])
    if size <= 0:
        raise CompileError(E_CFG, "%s tensor %s has empty storage" %
                           (node.op_type, name), node_id=node.id)
    # v0.2 batch 2: the copy engine is a bounded row/block staging engine, so
    # the legacy whole-tensor 16KiB staging limit no longer applies in spatial
    # mode.  The row-byte/16-bit-field budget is enforced independently by
    # spatial_capacity.check_row_capacity after lowering.
    if size > MAX_COPY_BYTES and not spatial_mode():
        raise CompileError(E_LIMIT,
                           "%s tensor %s storage %d exceeds P1 copy-core limit %d" %
                           (node.op_type, name, size, MAX_COPY_BYTES),
                           node_id=node.id)
    return size


def _infer_pad(node, inputs: Sequence[Tensor],
               numeric_constants: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    if len(inputs) not in (2, 3):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Pad requires data and pads, with optional constant_value",
                           node_id=node.id)
    x, pads_tensor = inputs[0], inputs[1]
    _require_p1_activation(node, x, "Pad input")
    if pads_tensor.kind != "initializer":
        raise CompileError(E_CFG, "Pad pads must be a static initializer", node_id=node.id)
    _require_rank(node, pads_tensor, 1)
    pads = _constant_ints(node, numeric_constants, pads_tensor.name, "Pad pads")
    if len(pads) != 8:
        raise CompileError(E_SHAPE_MISMATCH, "Pad pads must contain 8 integers", node_id=node.id)
    n0, c0, h0, w0, n1, c1, h1, w1 = pads
    if any(value != 0 for value in (n0, c0, n1, c1)):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Pad batch/channel padding must be 0", node_id=node.id,
                           attribute="pads")
    if min(h0, w0, h1, w1) < 0:
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Pad negative padding/cropping is unsupported", node_id=node.id,
                           attribute="pads")
    if len(inputs) == 3:
        value_tensor = inputs[2]
        if value_tensor.kind != "initializer":
            raise CompileError(E_CFG, "Pad constant_value must be a static initializer",
                               node_id=node.id)
        values = _constant_numeric(node, numeric_constants, value_tensor.name,
                                   "Pad constant_value")
        if len(values) != 1 or float(values[0]) != 0.0:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Pad constant_value must equal zero_point=0", node_id=node.id,
                               attribute="constant_value")
    _, c, h, w = x.shape
    if c > channel_limit():
        raise CompileError(E_LIMIT, "Pad C=%d exceeds %d" % (c, channel_limit()),
                           node_id=node.id)
    hout = h + h0 + h1
    wout = w + w0 + w1
    if hout <= 0 or wout <= 0:
        raise CompileError(E_CFG, "Pad output shape is degenerate", node_id=node.id)
    width_cap = width_limit()
    height_cap = height_limit()
    if wout > width_cap:
        raise CompileError(width_error_code(),
                           "Pad output W=%d exceeds %d" % (wout, width_cap),
                           node_id=node.id,
                           detail={"width": wout, "limit": width_cap,
                                   "spatial_mode": spatial_mode()})
    if hout > height_cap:
        raise CompileError(width_error_code(),
                           "Pad output H=%d exceeds %d" % (hout, height_cap),
                           node_id=node.id,
                           detail={"height": hout, "limit": height_cap,
                                   "spatial_mode": spatial_mode()})
    copy_required = any(value != 0 for value in (h0, w0, h1, w1))
    if copy_required:
        _check_copy_core_size(node, x.name, x.shape)
        _check_copy_core_size(node, node.outputs[0], (1, c, hout, wout))
    node.metadata.update({
        "pads": [n0, c0, h0, w0, n1, c1, h1, w1],
        "constant_value": 0,
        "copy_required": bool(copy_required),
        "view_kind": "none" if copy_required else "identity",
    })
    node.metadata_only = not copy_required
    node.lowering = "spatial_pad_copy" if copy_required else "metadata_pad_noop"
    return _out(x, (1, c, hout, wout), x.dtype, "NCHW")


def _infer_slice(node, inputs: Sequence[Tensor],
                 numeric_constants: Dict[str, Sequence[float]]) -> Dict[str, Any]:
    if len(inputs) not in (3, 4, 5):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Slice requires data, starts, ends and optional axes/steps",
                           node_id=node.id)
    x = inputs[0]
    _require_p1_activation(node, x, "Slice input")
    for tensor in inputs[1:]:
        if tensor.kind != "initializer":
            raise CompileError(E_CFG, "Slice %s must be a static initializer" % tensor.name,
                               node_id=node.id)
    starts = _constant_ints(node, numeric_constants, inputs[1].name, "Slice starts")
    ends = _constant_ints(node, numeric_constants, inputs[2].name, "Slice ends")
    if not starts or len(starts) != len(ends):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Slice starts/ends must be non-empty and equal length", node_id=node.id)
    if len(inputs) >= 4:
        axes = _constant_ints(node, numeric_constants, inputs[3].name, "Slice axes")
    else:
        axes = list(range(len(starts)))
    if len(axes) != len(starts):
        raise CompileError(E_SHAPE_MISMATCH, "Slice axes length must match starts", node_id=node.id)
    if len(inputs) == 5:
        steps = _constant_ints(node, numeric_constants, inputs[4].name, "Slice steps")
    else:
        steps = [1] * len(starts)
    if len(steps) != len(starts):
        raise CompileError(E_SHAPE_MISMATCH, "Slice steps length must match starts", node_id=node.id)
    if any(step != 1 for step in steps):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Slice steps must all be 1", node_id=node.id, attribute="steps")
    normalized_axes = []
    for axis in axes:
        value = int(axis)
        if value < 0:
            value += 4
        if value not in (1, 2, 3):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Slice axis %d is unsupported (H/W/C only)" % axis,
                               node_id=node.id, attribute="axes")
        normalized_axes.append(value)
    if len(set(normalized_axes)) != len(normalized_axes):
        raise CompileError(E_UNSUPPORTED_ATTR, "Slice has duplicate axes", node_id=node.id,
                           attribute="axes")
    shape = [int(v) for v in x.shape]
    starts_out = [0, 0, 0, 0]
    ends_out = list(shape)
    for start, end, axis in zip(starts, ends, normalized_axes):
        dim = shape[axis]
        value_start = int(start)
        value_end = int(end)
        if value_start < 0:
            value_start += dim
        if value_end < 0:
            value_end += dim
        value_start = max(0, min(dim, value_start))
        value_end = max(0, min(dim, value_end))
        if value_end <= value_start:
            raise CompileError(E_CFG,
                               "Slice produces an empty output on axis %d" % axis,
                               node_id=node.id)
        starts_out[axis] = value_start
        ends_out[axis] = value_end
    out_shape = (1, ends_out[1] - starts_out[1], ends_out[2] - starts_out[2],
                 ends_out[3] - starts_out[3])
    if out_shape[1] <= 0 or out_shape[2] <= 0 or out_shape[3] <= 0:
        raise CompileError(E_CFG, "Slice produces an empty output", node_id=node.id)
    if out_shape[1] > channel_limit():
        raise CompileError(E_LIMIT, "Slice output C=%d exceeds %d" %
                           (out_shape[1], channel_limit()), node_id=node.id)
    if out_shape[3] > width_limit():
        raise CompileError(width_error_code(), "Slice output W=%d exceeds %d" %
                           (out_shape[3], width_limit()), node_id=node.id,
                           detail={"width": out_shape[3], "limit": width_limit()})
    if out_shape[2] > height_limit():
        raise CompileError(width_error_code(), "Slice output H=%d exceeds %d" %
                           (out_shape[2], height_limit()), node_id=node.id,
                           detail={"height": out_shape[2], "limit": height_limit()})
    c_changed = starts_out[1] != 0 or ends_out[1] != shape[1]
    h_changed = starts_out[2] != 0 or ends_out[2] != shape[2]
    w_changed = starts_out[3] != 0 or ends_out[3] != shape[3]
    if not c_changed and not h_changed and not w_changed:
        view_kind = "identity"
    elif h_changed and not c_changed and not w_changed:
        view_kind = "h_view"
    else:
        view_kind = "copy"
    if view_kind == "copy":
        _check_copy_core_size(node, x.name, x.shape)
        _check_copy_core_size(node, node.outputs[0], out_shape)
    node.metadata.update({
        "axes": normalized_axes,
        "starts": starts_out,
        "ends": ends_out,
        "steps": steps,
        "view_kind": view_kind,
        "copy_required": view_kind == "copy",
    })
    node.metadata_only = view_kind != "copy"
    node.lowering = ("slice_copy" if view_kind == "copy"
                     else "metadata_slice_" + view_kind)
    return _out(x, out_shape, x.dtype, "NCHW")


def _infer_concat(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) < 2 or len(inputs) > 4:
        raise CompileError(E_SHAPE_MISMATCH,
                           "Concat requires 2..4 inputs", node_id=node.id)
    first = inputs[0]
    _require_p1_activation(node, first, "Concat input")
    height, width = int(first.shape[2]), int(first.shape[3])
    for tensor in inputs[1:]:
        _require_p1_activation(node, tensor, "Concat input")
        if tensor.dtype != first.dtype:
            raise CompileError(E_DTYPE_MISMATCH,
                               "Concat dtype mismatch: %s vs %s" % (first.dtype, tensor.dtype),
                               node_id=node.id)
        if tensor.layout != first.layout:
            raise CompileError(E_UNSUPPORTED_LAYOUT,
                               "Concat layout mismatch: %s vs %s" % (first.layout, tensor.layout),
                               node_id=node.id)
        if (int(tensor.shape[2]), int(tensor.shape[3])) != (height, width):
            raise CompileError(E_SHAPE_MISMATCH,
                               "Concat H/W mismatch: %s vs %s" %
                               (first.shape[2:], tensor.shape[2:]), node_id=node.id)
    _check_quant_compatible(node, inputs)
    channels = [int(tensor.shape[1]) for tensor in inputs]
    if any(value > channel_limit() for value in channels):
        raise CompileError(E_LIMIT, "Concat input channels exceed %d" % channel_limit(),
                           node_id=node.id)
    total_channels = sum(channels)
    if total_channels > channel_limit():
        raise CompileError(E_LIMIT, "Concat output C=%d exceeds %d" %
                           (total_channels, MAX_CHANNELS), node_id=node.id)
    width_cap = width_limit()
    height_cap = height_limit()
    if width > width_cap:
        raise CompileError(width_error_code(),
                           "Concat W=%d exceeds %d" % (width, width_cap),
                           node_id=node.id,
                           detail={"width": width, "limit": width_cap,
                                   "spatial_mode": spatial_mode()})
    if height > height_cap:
        raise CompileError(width_error_code(),
                           "Concat H=%d exceeds %d" % (height, height_cap),
                           node_id=node.id,
                           detail={"height": height, "limit": height_cap,
                                   "spatial_mode": spatial_mode()})
    out_shape = (1, total_channels, height, width)
    _check_copy_core_size(node, node.outputs[0], out_shape)
    for tensor in inputs:
        _check_copy_core_size(node, tensor.name, tensor.shape)
    node.metadata.update({
        "axis": 1,
        "input_channels": channels,
        "channel_offsets": [sum(channels[:index]) for index in range(len(channels))],
        "copy_required": True,
        "view_kind": "none",
    })
    node.metadata_only = False
    node.lowering = "channel_concat_copy"
    return _out(first, out_shape, first.dtype, "NCHW")


def _infer_relu(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "Relu requires exactly one input", node_id=node.id)
    x = inputs[0]
    return _out(x, x.shape, x.dtype, x.layout)


def _infer_add(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 2:
        raise CompileError(E_SHAPE_MISMATCH, "Add requires exactly two inputs", node_id=node.id)
    a, b = inputs
    if a.shape != b.shape:
        raise CompileError(E_UNSUPPORTED_BROADCAST,
                           "Add requires identical shapes; got %s and %s" % (a.shape, b.shape),
                           node_id=node.id)
    if a.dtype != b.dtype:
        raise CompileError(E_DTYPE_MISMATCH, "Add dtype mismatch: %s vs %s" %
                           (a.dtype, b.dtype), node_id=node.id)
    if a.layout != b.layout:
        raise CompileError(E_UNSUPPORTED_LAYOUT, "Add layout mismatch: %s vs %s" %
                           (a.layout, b.layout), node_id=node.id)
    return _out(a, a.shape, a.dtype, a.layout)


def _shape_values(node, constants: Dict[str, Sequence[int]], name: str) -> List[int]:
    if name not in constants:
        raise CompileError(E_CFG, "Reshape shape input %s is not a constant" % name,
                           node_id=node.id)
    try:
        values = [int(v) for v in constants[name]]
    except (TypeError, ValueError):
        raise CompileError(E_CFG, "Reshape shape input %s is not an integer vector" % name,
                           node_id=node.id)
    if not values:
        raise CompileError(E_UNSUPPORTED_ATTR, "Reshape target rank 0 is unsupported",
                           node_id=node.id)
    return values


def _infer_constant(node, inputs: Sequence[Tensor]) -> List[Dict[str, Any]]:
    if inputs:
        raise CompileError(E_SHAPE_MISMATCH, "Constant has no runtime inputs", node_id=node.id)
    shape = node.metadata.get("constant_shape")
    dtype = node.metadata.get("constant_dtype")
    values = node.metadata.get("constant_values")
    if shape is None or dtype is None or values is None:
        raise CompileError(E_CFG, "Constant value metadata is missing", node_id=node.id)
    shape_t = tuple(int(v) for v in shape)
    count = prod(shape_t) if shape_t else 1
    if len(values) != count:
        raise CompileError(E_CFG, "Constant value count %d != shape elements %d" %
                           (len(values), count), node_id=node.id)
    if dtype not in ("int8", "int32", "int64", "float32", "float16"):
        raise CompileError(E_DTYPE_MISMATCH,
                           "Constant dtype %s is unsupported" % dtype, node_id=node.id)
    return [{"shape": shape_t, "dtype": dtype,
             "layout": layout_for_rank(len(shape_t)) if shape_t else "LOGICAL",
             "logical_bytes": logical_nbytes(shape_t, dtype)}]


def _infer_identity(node, inputs: Sequence[Tensor]) -> List[Dict[str, Any]]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "Identity requires exactly one input", node_id=node.id)
    x = inputs[0]
    node.metadata.update({"view_kind": "identity", "alias_required": True})
    node.lowering = "metadata_identity_alias"
    return [_out(x, x.shape, x.dtype, x.layout)]


def _scalar_int_threshold(node, value_name: str,
                          numeric_constants: Dict[str, Sequence[float]]) -> int:
    values = _constant_numeric(node, numeric_constants, value_name, "Clip bound")
    if len(values) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "Clip bound %s must be scalar" % value_name,
                           node_id=node.id)
    value = float(values[0])
    if not value.is_integer() or not (-128.0 <= value <= 127.0):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Clip bound %s must be an integer INT8 threshold" % value_name,
                           node_id=node.id, attribute=value_name)
    return int(value)


def _infer_clip(node, inputs: Sequence[Tensor],
                numeric_constants: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    if len(inputs) not in (1, 2, 3):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Clip requires data and optional min/max", node_id=node.id)
    x = inputs[0]
    _require_p1_activation(node, x, "Clip input")
    qmin = int(node.attrs.get("min", -128))
    qmax = int(node.attrs.get("max", 127))
    if len(inputs) >= 2:
        if inputs[1].kind not in ("initializer", "intermediate"):
            raise CompileError(E_CFG, "Clip min must be compile-time constant", node_id=node.id)
        qmin = _scalar_int_threshold(node, inputs[1].name, numeric_constants)
    if len(inputs) == 3:
        if inputs[2].kind not in ("initializer", "intermediate"):
            raise CompileError(E_CFG, "Clip max must be compile-time constant", node_id=node.id)
        qmax = _scalar_int_threshold(node, inputs[2].name, numeric_constants)
    if qmin > qmax:
        raise CompileError(E_UNSUPPORTED_ATTR, "Clip min exceeds max", node_id=node.id)
    node.metadata.update({"clip_min": qmin, "clip_max": qmax,
                          "clip_has_min": bool(len(inputs) >= 2 or "min" in node.attrs),
                          "clip_has_max": bool(len(inputs) == 3 or "max" in node.attrs)})
    node.lowering = "clip_core_v1"
    return [_out(x, x.shape, x.dtype, "NCHW")]


def _infer_global_average_pool(node, inputs: Sequence[Tensor]) -> List[Dict[str, Any]]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH,
                           "GlobalAveragePool requires one input", node_id=node.id)
    x = inputs[0]
    _require_p1_activation(node, x, "GlobalAveragePool input")
    n, c, h, w = x.shape
    if not (1 <= c <= channel_limit()):
        raise CompileError(E_LIMIT, "GlobalAveragePool C=%d exceeds %d" %
                           (c, channel_limit()), node_id=node.id)
    hw_cap = hw_limit()
    if h <= 0 or w <= 0 or h > hw_cap or w > hw_cap:
        raise CompileError(width_error_code(),
                           "GlobalAveragePool H/W=%dx%d exceeds capacity %d" %
                           (h, w, hw_cap),
                           node_id=node.id,
                           detail={"height": h, "width": w, "limit": hw_cap,
                                   "spatial_mode": spatial_mode()})
    node.metadata.update({"reduction": "sum_exact_round_away", "reduction_n": int(h * w)})
    node.lowering = "global_avg_pool_core_v1"
    return [_out(x, (1, c, 1, 1), x.dtype, "NCHW")]


def _infer_split(node, inputs: Sequence[Tensor],
                 numeric_constants: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    if len(inputs) not in (1, 2):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Split requires data and optional split lengths", node_id=node.id)
    x = inputs[0]
    _require_p1_activation(node, x, "Split input")
    channels = int(x.shape[1])
    outputs = list(node.outputs)
    if not (2 <= len(outputs) <= 4):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Split requires 2..4 static outputs", node_id=node.id)
    if len(inputs) == 2:
        split = _constant_ints(node, numeric_constants, inputs[1].name, "Split lengths")
    else:
        if channels % len(outputs):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Split without lengths requires equal division", node_id=node.id)
        split = [channels // len(outputs)] * len(outputs)
    if len(split) != len(outputs) or any(value <= 0 for value in split):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Split lengths must be positive and match output count", node_id=node.id)
    if sum(split) != channels:
        raise CompileError(E_SHAPE_MISMATCH,
                           "Split length sum %d != C %d" % (sum(split), channels), node_id=node.id)
    offsets = []
    cursor = 0
    specs = []
    for index, length in enumerate(split):
        offsets.append(cursor)
        specs.append({"index": index, "output": outputs[index],
                      "start": cursor, "length": length})
        cursor += length
    node.metadata.update({"split_lengths": split, "channel_offsets": offsets,
                          "split_specs": specs, "view_kind": "copy",
                          "copy_required": True, "lowered_op": "Slice"})
    node.lowering = "static_split_to_slice_records"
    return [_out(x, (1, length, int(x.shape[2]), int(x.shape[3])), x.dtype, "NCHW")
            for length in split]


def _metadata_axes(node, inputs: Sequence[Tensor], rank_for_normalization: int,
                   numeric_constants: Dict[str, Sequence[float]]) -> List[int]:
    if "axes" in node.attrs:
        axes = [int(v) for v in node.attrs["axes"]]
    elif len(inputs) >= 2:
        axes = _constant_ints(node, numeric_constants, inputs[1].name, "%s axes" % node.op_type)
    else:
        axes = []
    normalized = []
    for axis in axes:
        value = int(axis)
        if value < 0:
            value += rank_for_normalization
        if value < 0 or value >= rank_for_normalization:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "%s axis %d outside rank %d" %
                               (node.op_type, axis, rank_for_normalization),
                               node_id=node.id, attribute="axes")
        normalized.append(value)
    if len(normalized) != len(set(normalized)):
        raise CompileError(E_UNSUPPORTED_ATTR, "%s has duplicate axes" % node.op_type,
                           node_id=node.id, attribute="axes")
    return normalized


def _infer_squeeze(node, inputs: Sequence[Tensor],
                   numeric_constants: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    if len(inputs) not in (1, 2):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Squeeze requires data and optional axes", node_id=node.id)
    x = inputs[0]
    if x.shape is None:
        raise CompileError(E_UNKNOWN_RANK, "Squeeze input has no shape", node_id=node.id)
    if "axes" in node.attrs or len(inputs) == 2:
        axes = _metadata_axes(node, inputs, len(x.shape), numeric_constants)
    else:
        axes = [index for index, dim in enumerate(x.shape) if int(dim) == 1]
    for axis in axes:
        if int(x.shape[axis]) != 1:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Squeeze axis %d has size %d != 1" %
                               (axis, x.shape[axis]), node_id=node.id, attribute="axes")
    out_shape = tuple(dim for index, dim in enumerate(x.shape) if index not in set(axes))
    node.metadata.update({"axes": sorted(axes), "view_kind": "identity",
                          "alias_required": True})
    node.lowering = "metadata_squeeze_alias"
    return [_out(x, out_shape, x.dtype)]


def _infer_unsqueeze(node, inputs: Sequence[Tensor],
                     numeric_constants: Dict[str, Sequence[float]]) -> List[Dict[str, Any]]:
    if len(inputs) not in (1, 2):
        raise CompileError(E_SHAPE_MISMATCH,
                           "Unsqueeze requires data and axes", node_id=node.id)
    x = inputs[0]
    if x.shape is None:
        raise CompileError(E_UNKNOWN_RANK, "Unsqueeze input has no shape", node_id=node.id)
    if "axes" not in node.attrs and len(inputs) != 2:
        raise CompileError(E_UNSUPPORTED_ATTR, "Unsqueeze requires static axes", node_id=node.id)
    out_rank = len(x.shape) + (len(node.attrs.get("axes", [])) if "axes" in node.attrs else
                               len(_constant_ints(node, numeric_constants, inputs[1].name,
                                                  "Unsqueeze axes")))
    axes = _metadata_axes(node, inputs, out_rank, numeric_constants)
    out: List[int] = []
    source_index = 0
    axis_set = set(axes)
    for index in range(out_rank):
        if index in axis_set:
            out.append(1)
        else:
            out.append(int(x.shape[source_index]))
            source_index += 1
    node.metadata.update({"axes": sorted(axes), "view_kind": "identity",
                          "alias_required": True})
    node.lowering = "metadata_unsqueeze_alias"
    return [_out(x, tuple(out), x.dtype)]


def _infer_reshape(node, inputs: Sequence[Tensor], constants: Dict[str, Sequence[int]]) -> Dict[str, Any]:
    if len(inputs) != 2:
        raise CompileError(E_SHAPE_MISMATCH, "Reshape requires data and shape inputs", node_id=node.id)
    x, shape_tensor = inputs
    if shape_tensor.kind != "initializer":
        raise CompileError(E_CFG, "Reshape shape must be an initializer", node_id=node.id)
    _require_rank(node, shape_tensor, 1)
    target = _shape_values(node, constants, shape_tensor.name)
    source = list(x.shape)
    out: List[int] = []
    infer_index = None
    for index, dim in enumerate(target):
        if dim == -1:
            if infer_index is not None:
                raise CompileError(E_CFG, "Reshape permits only one -1 dimension", node_id=node.id)
            infer_index = index
            out.append(-1)
        elif dim == 0:
            if index >= len(source):
                raise CompileError(E_CFG, "Reshape zero dimension index exceeds source rank",
                                   node_id=node.id)
            out.append(source[index])
        elif dim > 0:
            out.append(dim)
        else:
            raise CompileError(E_UNSUPPORTED_ATTR, "Reshape dimension %d is unsupported" % dim,
                               node_id=node.id)
    total = prod(source)
    known = prod(dim for dim in out if dim > 0)
    if infer_index is not None:
        if known == 0 or total % known:
            raise CompileError(E_CFG, "Reshape cannot infer -1 dimension", node_id=node.id)
        out[infer_index] = total // known
    if prod(out) != total:
        raise CompileError(E_SHAPE_MISMATCH, "Reshape element count %d != source %d" %
                           (prod(out), total), node_id=node.id)
    return _out(x, out, x.dtype)


def _infer_flatten(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "Flatten requires exactly one input", node_id=node.id)
    x = inputs[0]
    rank = len(x.shape)
    axis = int(node.attrs["axis"])
    if axis < 0:
        axis += rank
    if not (0 <= axis <= rank):
        raise CompileError(E_UNSUPPORTED_ATTR, "Flatten axis %d is outside [-%d,%d]" %
                           (node.attrs["axis"], rank, rank), node_id=node.id, attribute="axis")
    return _out(x, (prod(x.shape[:axis]), prod(x.shape[axis:])), x.dtype, "NC")


def _infer_gemm(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) not in (2, 3):
        raise CompileError(E_SHAPE_MISMATCH, "Gemm requires A, B and optional C", node_id=node.id)
    a, b = inputs[0], inputs[1]
    _require_rank(node, a, 2)
    _require_rank(node, b, 2)
    if a.dtype != b.dtype:
        raise CompileError(E_DTYPE_MISMATCH, "Gemm A/B dtype mismatch: %s vs %s" %
                           (a.dtype, b.dtype), node_id=node.id)
    m, k = a.shape
    if int(node.attrs["transB"]):
        n, kb = b.shape
    else:
        kb, n = b.shape
    if kb != k:
        raise CompileError(E_SHAPE_MISMATCH, "Gemm reduction dimension %d != %d" % (k, kb),
                           node_id=node.id)
    if len(inputs) == 3:
        c = inputs[2]
        if c.shape not in ((n,), (1, n)):
            raise CompileError(E_SHAPE_MISMATCH, "Gemm C shape %s is not (%d,) or (1,%d)" %
                               (c.shape, n, n), node_id=node.id)
        if c.dtype != a.dtype:
            raise CompileError(E_DTYPE_MISMATCH, "Gemm C dtype %s != A dtype %s" %
                               (c.dtype, a.dtype), node_id=node.id)
    return _out(a, (m, n), a.dtype, "NC")


def _require_lut_shape(node, shape) -> None:
    if len(tuple(shape)) != 4:
        raise CompileError(E_SHAPE_MISMATCH, "ActivationLUT requires rank-4 NCHW", node_id=node.id)
    n, c, h, w = (int(v) for v in shape)
    if n != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "ActivationLUT batch N must be 1", node_id=node.id,
                           attribute="N")
    if not (1 <= c <= 16):
        raise CompileError(E_LIMIT, "ActivationLUT channels C=%d outside 1..16" % c, node_id=node.id)
    if spatial_mode():
        h_cap, w_cap = height_limit(), width_limit()
        if not (1 <= h <= h_cap and 1 <= w <= w_cap):
            raise CompileError(width_error_code(),
                               "ActivationLUT H/W=%dx%d outside 1..%d/%d" %
                               (h, w, h_cap, w_cap), node_id=node.id,
                               detail={"height": h, "width": w,
                                       "height_limit": h_cap, "width_limit": w_cap,
                                       "spatial_mode": True})
    elif not (1 <= h <= 32 and 1 <= w <= 32):
        raise CompileError(E_LIMIT, "ActivationLUT H/W=%dx%d outside 1..32" % (h, w), node_id=node.id)


def _infer_activation_lut(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "ActivationLUT requires exactly one input", node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    if x.dtype != "int8":
        raise CompileError(E_DTYPE_MISMATCH,
                           "ActivationLUT supports signed INT8 only, got %s" % x.dtype,
                           node_id=node.id)
    _require_lut_shape(node, tuple(x.shape))
    mode = str((node.attrs or {}).get("mode", ""))
    if mode not in ("Sigmoid", "Tanh", "SiLU", "LeakyRelu"):
        raise CompileError(E_UNSUPPORTED_ATTR, "ActivationLUT mode must be Sigmoid/Tanh/SiLU/LeakyRelu",
                           node_id=node.id, attribute="mode")
    if mode == "LeakyRelu":
        try:
            alpha = float((node.attrs or {}).get("alpha", 0.01))
        except Exception:
            raise CompileError(E_UNSUPPORTED_ATTR, "ActivationLUT alpha must be numeric", node_id=node.id,
                               attribute="alpha")
        import math as _m
        if not _m.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise CompileError(E_UNSUPPORTED_ATTR, "ActivationLUT alpha outside 0..1", node_id=node.id,
                               attribute="alpha")
    node.lowering = "activation_lut_table_v1"
    return _out(x, tuple(x.shape), "int8", "NCHW")


def _infer_sigmoid_tanh_semantic(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "%s requires exactly one input" % node.op_type, node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    if x.dtype in ("float32", "float16"):
        if int(x.shape[0]) != 1:
            raise CompileError(E_UNSUPPORTED_ATTR, "%s batch N must be 1" % node.op_type, node_id=node.id,
                               attribute="N")
        node.lowering = "semantic_activation_float"
        return _out(x, tuple(x.shape), str(x.dtype), "NCHW")
    raise CompileError(E_DTYPE_MISMATCH,
                       "%s semantic supports float32/float16 only in pre-deployment IR; got %s; use ActivationLUT deployment" %
                       (node.op_type, x.dtype), node_id=node.id)


def _infer_leaky_semantic(node, inputs: Sequence[Tensor]) -> Dict[str, Any]:
    if len(inputs) != 1:
        raise CompileError(E_SHAPE_MISMATCH, "LeakyRelu requires exactly one input", node_id=node.id)
    x = inputs[0]
    _require_rank(node, x, 4)
    if x.dtype in ("float32", "float16"):
        if int(x.shape[0]) != 1:
            raise CompileError(E_UNSUPPORTED_ATTR, "LeakyRelu batch N must be 1", node_id=node.id,
                               attribute="N")
        node.lowering = "semantic_activation_float"
        return _out(x, tuple(x.shape), str(x.dtype), "NCHW")
    raise CompileError(E_DTYPE_MISMATCH,
                       "LeakyRelu semantic supports float32/float16 only; got %s; use ActivationLUT deployment" % x.dtype,
                       node_id=node.id)


def infer_node_outputs(node, inputs: Sequence[Tensor],
                       constants: Optional[Dict[str, Sequence[int]]] = None,
                       numeric_constants: Optional[Dict[str, Sequence[float]]] = None) -> List[Dict[str, Any]]:
    for tensor in inputs:
        if tensor.shape is None:
            raise CompileError(E_UNKNOWN_RANK, "input tensor %s has no inferred shape" % tensor.name,
                               node_id=node.id)
    constants = constants or {}
    numeric_constants = numeric_constants or {}
    single = None
    if node.op_type == "Conv":
        single = _infer_conv(node, inputs)
    elif node.op_type == "DepthwiseConv":
        single = _infer_depthwise(node, inputs)
    elif node.op_type == "MaxPool":
        single = _infer_maxpool(node, inputs)
    elif node.op_type == "Resize":
        single = _infer_resize(node, inputs, numeric_constants)
    elif node.op_type == "AveragePool":
        single = _infer_average_pool(node, inputs)
    elif node.op_type == "Mul":
        single = _infer_mul(node, inputs)
    elif node.op_type == "ActivationLUT":
        single = _infer_activation_lut(node, inputs)
    elif node.op_type in ("Sigmoid", "Tanh"):
        single = _infer_sigmoid_tanh_semantic(node, inputs)
    elif node.op_type == "LeakyRelu":
        single = _infer_leaky_semantic(node, inputs)
    elif node.op_type == "BatchNormalization":
        single = _infer_batch_norm(node, inputs)
    elif node.op_type == "Relu":
        single = _infer_relu(node, inputs)
    elif node.op_type == "Add":
        single = _infer_add(node, inputs)
    elif node.op_type == "Pad":
        single = _infer_pad(node, inputs, numeric_constants)
    elif node.op_type == "Slice":
        single = _infer_slice(node, inputs, numeric_constants)
    elif node.op_type == "Concat":
        single = _infer_concat(node, inputs)
    elif node.op_type == "Reshape":
        single = _infer_reshape(node, inputs, constants)
    elif node.op_type == "Flatten":
        single = _infer_flatten(node, inputs)
    elif node.op_type == "Gemm":
        single = _infer_gemm(node, inputs)
    elif node.op_type == "Constant":
        return _infer_constant(node, inputs)
    elif node.op_type == "Identity":
        return _infer_identity(node, inputs)
    elif node.op_type == "Clip":
        return _infer_clip(node, inputs, numeric_constants)
    elif node.op_type == "GlobalAveragePool":
        return _infer_global_average_pool(node, inputs)
    elif node.op_type == "Split":
        return _infer_split(node, inputs, numeric_constants)
    elif node.op_type == "Squeeze":
        return _infer_squeeze(node, inputs, numeric_constants)
    elif node.op_type == "Unsqueeze":
        return _infer_unsqueeze(node, inputs, numeric_constants)
    else:
        raise CompileError(E_UNSUPPORTED_OP,
                           "operator %r has no shape inference" % node.op_type,
                           node_id=node.id)
    return [single]


def infer_node(node, inputs: Sequence[Tensor],
               constants: Optional[Dict[str, Sequence[int]]] = None,
               numeric_constants: Optional[Dict[str, Sequence[float]]] = None) -> Dict[str, Any]:
    results = infer_node_outputs(node, inputs, constants, numeric_constants)
    if len(results) != 1:
        raise CompileError(E_CFG, "operator %s has multiple outputs" % node.op_type,
                           node_id=node.id)
    return results[0]


def infer_graph(graph, constants: Optional[Dict[str, Sequence[int]]] = None,
                numeric_constants: Optional[Dict[str, Sequence[float]]] = None,
                caps=None):
    """Infer every node output, then finalize producer/consumer relations.

    caps=None keeps the historical (legacy) limits; a spatial build passes its
    versioned capability dict so only the geometry ceiling changes.
    """
    constants = constants or {}
    numeric_constants = numeric_constants or {}
    if caps is not None:
        with spatial_caps(caps):
            return infer_graph(graph, constants, numeric_constants, caps=None)
    graph.nodes = graph._topological_nodes()
    for node in graph.nodes:
        inputs = [graph.tensors[name] for name in node.inputs]
        results = infer_node_outputs(node, inputs, constants, numeric_constants)
        if len(results) != len(node.outputs):
            raise CompileError(E_SHAPE_MISMATCH,
                               "%s inferred %d outputs but declares %d" %
                               (node.op_type, len(results), len(node.outputs)), node_id=node.id)
        for output_name, result in zip(node.outputs, results):
            tensor = graph.tensors.get(output_name)
            if tensor is None:
                tensor = Tensor(output_name, dtype=result["dtype"], shape=result["shape"])
                graph.tensors[output_name] = tensor
            elif tensor.shape is not None and tensor.shape != result["shape"]:
                raise CompileError(E_SHAPE_MISMATCH, "declared shape %s != inferred %s for %s" %
                                   (tensor.shape, result["shape"], tensor.name), node_id=node.id)
            tensor.shape = tuple(result["shape"])
            tensor.dtype = result["dtype"]
            tensor.layout = result["layout"]
            tensor.logical_bytes = result["logical_bytes"]
        if node.op_type in ("Resize", "MaxPool"):
            # affine-int8-v1 profile gate: identity-arithmetic ops must keep one
            # real numeric domain (checked on the contract tensors attached by the
            # affine frontend, never on the signed-reencoded legacy view).
            _check_affine_identity_domain(graph, node, node.op_type)
    graph.finalize()
    return graph


def nhwc8_a16_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 4:
        raise CompileError(E_CFG, "NHWC8_A16 storage requires rank-4 NCHW shape")
    n, c, h, w = (int(v) for v in shape)
    if n != 1:
        raise CompileError(E_CFG, "NHWC8_A16 storage requires N=1")
    channel_alloc = align_up(c, CHANNEL_BLOCK)
    row_stride = align_up(channel_alloc * w, ABI_ALIGNMENT)
    return {
        "layout": "NHWC8_A16",
        "channel_alloc": channel_alloc,
        "row_stride_bytes": row_stride,
        "bytes": align_up(h * row_stride, ABI_ALIGNMENT),
        "alignment": ABI_ALIGNMENT,
    }


def weight_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 4:
        raise CompileError(E_CFG, "Conv weight storage requires OIHW rank 4")
    cout, cin, kh, kw = (int(v) for v in shape)
    if (kh, kw) != (3, 3):
        raise CompileError(E_CFG, "Conv weight storage only defines 3x3 kernels")
    cout_alloc = align_up(cout, CHANNEL_BLOCK)
    cin_alloc = align_up(cin, CHANNEL_BLOCK)
    blocks = (cout_alloc // CHANNEL_BLOCK) * (cin_alloc // CHANNEL_BLOCK)
    return {
        "layout": "OIHW8_KB_IC_OC",
        "cout_alloc": cout_alloc,
        "cin_alloc": cin_alloc,
        "bytes": blocks * 9 * 64,
        "alignment": ABI_ALIGNMENT,
    }


def bias_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 1:
        raise CompileError(E_CFG, "Conv bias storage requires a vector")
    cout_alloc = align_up(int(shape[0]), CHANNEL_BLOCK)
    return {"layout": "I32_C", "bytes": align_up(cout_alloc * 4, ABI_ALIGNMENT),
            "alignment": ABI_ALIGNMENT}


def quant_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 1:
        raise CompileError(E_CFG, "quant storage requires an output-channel vector")
    cout_alloc = align_up(int(shape[0]), CHANNEL_BLOCK)
    return {"layout": "M0_SHIFT_ZP", "record_bytes": 8,
            "bytes": align_up(cout_alloc * 8, ABI_ALIGNMENT),
            "alignment": ABI_ALIGNMENT}


def depthwise_weight_storage(shape: Sequence[int]) -> Dict[str, Any]:
    """Depthwise 0x001C weight: C blocks of 16B (9 int8 taps + 7 zero bytes)."""
    if len(shape) != 4:
        raise CompileError(E_CFG, "depthwise weight storage requires rank-4 [C,1,3,3]")
    cout, cin, kh, kw = (int(v) for v in shape)
    if (cin, kh, kw) != (1, 3, 3):
        raise CompileError(E_CFG,
                           "depthwise weight storage requires [C,1,3,3], got %s" %
                           (tuple(shape),))
    if not (1 <= cout <= DW_MAX_CHANNELS):
        raise CompileError(E_LIMIT, "depthwise C=%d exceeds 1..%d" %
                           (cout, DW_MAX_CHANNELS))
    return {"layout": "DW_OC16_K9", "channel_count": cout, "block_bytes": 16,
            "bytes": cout * 16, "alignment": ABI_ALIGNMENT}


def depthwise_bias_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 1:
        raise CompileError(E_CFG, "depthwise bias storage requires a vector")
    c = int(shape[0])
    if not (1 <= c <= DW_MAX_CHANNELS):
        raise CompileError(E_LIMIT, "depthwise C=%d exceeds 1..%d" %
                           (c, DW_MAX_CHANNELS))
    return {"layout": "I32_C", "bytes": align_up(4 * c, ABI_ALIGNMENT),
            "alignment": ABI_ALIGNMENT}


def depthwise_quant_storage(shape: Sequence[int]) -> Dict[str, Any]:
    if len(shape) != 1:
        raise CompileError(E_CFG, "depthwise quant storage requires a channel vector")
    c = int(shape[0])
    if not (1 <= c <= DW_MAX_CHANNELS):
        raise CompileError(E_LIMIT, "depthwise C=%d exceeds 1..%d" %
                           (c, DW_MAX_CHANNELS))
    return {"layout": "M0_SHIFT_ZP", "record_bytes": 8,
            "bytes": align_up(8 * c, ABI_ALIGNMENT), "alignment": ABI_ALIGNMENT}
