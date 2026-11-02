"""Generic integer graph interpreter for the E1-P2 static CNN subset."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .exact_quant import bn_channel_terms, exact, round_away

from .graph_ir import CompileError, E_CFG, E_UNSUPPORTED_OP, Graph
from .parameter_pack import pack_activation

# Thread-local instrumentation for the optional reference dependency cache
# (tools/onnx_gen/reference_cache.py).  A hot cache hit must be provable: the
# compile records recompute_calls == 0, which means evaluate_graph never ran on
# that compile.  Counters are thread-local so a multi-threaded driver (for
# example the compile matrix with --jobs 2) cannot contaminate another compile's
# cold/hot evidence.  Pure counters only; no behaviour change.
_EVALUATE_STATS_LOCAL = threading.local()


def _evaluate_counters() -> Dict[str, int]:
    counters = getattr(_EVALUATE_STATS_LOCAL, "counters", None)
    if counters is None:
        counters = {"evaluate_graph_calls": 0, "nodes_evaluated": 0}
        _EVALUATE_STATS_LOCAL.counters = counters
    return counters


def evaluate_stats() -> Dict[str, int]:
    """Return a copy of this thread's evaluate_graph call counters."""
    return dict(_evaluate_counters())


def attach_lut_scales(graph: Graph) -> None:
    """Attach LUT scales from tensor quantization for reference determinism.

    evaluate_graph used to perform this mutation inline.  The reference cache
    materialises verified values without calling evaluate_graph, so the same
    metadata attachment must happen there as well; otherwise a cold compute and
    a hot hit would record different canonical graph digests for identical
    values.  Idempotent (sets the same float values).
    """
    for node in graph.nodes:
        if node.op_type != "ActivationLUT":
            continue
        try:
            tensors = graph.tensors
            in_scale = (tensors[node.inputs[0]].quantization or {}).get("scale")
            out_scale = (tensors[node.outputs[0]].quantization or {}).get("scale")
            if in_scale is not None:
                node.metadata["lut_input_scale"] = float(in_scale)
            if out_scale is not None:
                node.metadata["lut_output_scale"] = float(out_scale)
        except (KeyError, AttributeError, TypeError):
            continue


@dataclass
class TensorValue:
    shape: Tuple[int, ...]
    dtype: str
    values: List[int]
    # Raw signed INT32 Conv accumulators, set by _conv for a fused Conv->BN
    # boundary.  They are reference-only and never reach the hardware ABI.
    accumulators: Optional[List[int]] = None
    accumulator_shape: Optional[Tuple[int, ...]] = None
    materialized: bool = True


def _signed8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _sat8(value: int) -> int:
    return max(-128, min(127, int(value)))


def _index(shape: Sequence[int], n: int, c: int, h: int, w: int) -> int:
    _, cc, hh, ww = (int(v) for v in shape)
    return ((n * cc + c) * hh + h) * ww + w


def _requant(acc: int, record: Dict[str, Any]) -> int:
    m0 = int(record["M0"])
    shift = int(record["SHIFT"])
    p = int(acc) * m0
    if shift == 0:
        rounded = p
    elif p >= 0:
        rounded = (p + (1 << (shift - 1))) >> shift
    else:
        rounded = -(((-p) + (1 << (shift - 1))) >> shift)
    return _sat8(rounded + int(record.get("out_zero_point", 0)))


def _conv(node, x: TensorValue, initializers: Dict[str, Dict[str, Any]]) -> TensorValue:
    n, cin, h, w = x.shape
    group = int(node.attrs.get("group", 1))
    if group != 1:
        # Source ONNX grouped Conv: only the depthwise form is supported and is
        # evaluated by an independent path from the lowered DepthwiseConv node.
        return _depthwise_conv_source(node, x, initializers)
    weight = initializers[node.inputs[1]]
    src_shape = tuple(int(v) for v in weight["shape"])
    wvals = [int(v) for v in weight["values"]]
    kh, kw = (int(v) for v in node.attrs["kernel_shape"])
    cout = int(src_shape[0])
    if src_shape[1] != cin or src_shape[2] not in (1, 3) or src_shape[3] not in (1, 3):
        raise CompileError(E_CFG, "Conv reference weight shape mismatch", node_id=node.id)
    if (kh, kw) == (3, 3) and src_shape[2:] == (1, 1):
        expanded = [0] * (cout * cin * 9)
        for oc in range(cout):
            for ic in range(cin):
                expanded[((oc * cin + ic) * 3 + 1) * 3 + 1] = wvals[(oc * cin + ic)]
        wvals = expanded
    elif src_shape != (cout, cin, kh, kw):
        raise CompileError(E_CFG, "Conv reference unsupported source/lowest kernel", node_id=node.id)
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    bias = [0] * cout
    folded_bias = node.metadata.get("folded_bias_values")
    if folded_bias is not None:
        if len(folded_bias) != cout:
            raise CompileError(E_CFG, "folded Conv bias length mismatch", node_id=node.id)
        bias = [int(v) for v in folded_bias]
    elif len(node.inputs) >= 3:
        bsrc = initializers.get(node.inputs[2])
        if bsrc is None:
            raise CompileError(E_CFG, "Conv bias missing", node_id=node.id)
        bvals = [int(v) for v in bsrc["values"]]
        if len(bvals) != cout:
            raise CompileError(E_CFG, "Conv bias length mismatch", node_id=node.id)
        bias = bvals
    boundary = node.metadata.get("fold_pending") is True
    records = node.metadata.get("requant_records")
    if not boundary and (not isinstance(records, list) or len(records) != cout):
        raise CompileError(E_CFG, "Conv requant records missing", node_id=node.id)
    out = [0] * (cout * hout * wout)
    accumulators = [0] * (cout * hout * wout)
    for oy in range(hout):
        for ox in range(wout):
            for oc in range(cout):
                acc = int(bias[oc])
                for ky in range(kh):
                    iy = oy * stride + ky - pad
                    if iy < 0 or iy >= h:
                        continue
                    for kx in range(kw):
                        ix = ox * stride + kx - pad
                        if ix < 0 or ix >= w:
                            continue
                        for ic in range(cin):
                            xq = x.values[_index(x.shape, 0, ic, iy, ix)]
                            wq = int(wvals[((oc * cin + ic) * kh + ky) * kw + kx])
                            acc += xq * wq
                if not (-(2 ** 31) <= acc <= 2 ** 31 - 1):
                    raise CompileError(E_CFG, "Conv INT32 accumulator overflow", node_id=node.id)
                position = (oc * hout + oy) * wout + ox
                accumulators[position] = acc
                if boundary:
                    # Fused Conv->BN boundary: no INT8 rounding happens here.
                    # BatchNormalization consumes the raw accumulator instead.
                    out[position] = 0
                else:
                    out[position] = _requant(acc, records[oc])
    return TensorValue((1, cout, hout, wout), "int8", out,
                       accumulators=accumulators,
                       accumulator_shape=(cout, hout, wout),
                       materialized=not boundary)

def _depthwise_conv_source(node, x: TensorValue,
                            initializers: Dict[str, Dict[str, Any]]) -> TensorValue:
    """Independent source-level reference for ONNX Conv(group==Cin==Cout==C).

    Evaluates the grouped-Conv definition on the INT8 logical NCHW tensor and
    the explicit per-output-channel M0/SHIFT records derived by the quant
    contract.  It is written separately from the lowered DepthwiseConv path so
    that compile_pipeline's source/lowered comparison is a real cross-check.
    """
    n, cin, h, w = x.shape
    group = int(node.attrs.get("group", 1))
    if not (group == cin):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "source grouped Conv reference supports depthwise only "
                           "(group %d != Cin %d)" % (group, cin), node_id=node.id)
    weight = initializers.get(node.inputs[1])
    if weight is None:
        raise CompileError(E_CFG, "source depthwise Conv weight missing", node_id=node.id)
    if tuple(int(v) for v in weight["shape"]) != (cin, 1, 3, 3):
        raise CompileError(E_SHAPE_MISMATCH,
                           "source depthwise weight shape %s != [%d,1,3,3]" %
                           (tuple(int(v) for v in weight["shape"]), cin), node_id=node.id)
    wvals = [int(v) for v in weight["values"]]
    kh, kw = (int(v) for v in node.attrs["kernel_shape"])
    if (kh, kw) != (3, 3):
        raise CompileError(E_CFG, "source depthwise kernel must be 3x3", node_id=node.id)
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    if hout <= 0 or wout <= 0:
        raise CompileError(E_CFG, "source depthwise output is degenerate", node_id=node.id)
    bias_values = node.metadata.get("folded_bias_values")
    if bias_values is None:
        if len(node.inputs) < 3:
            raise CompileError(E_CFG, "source depthwise Conv requires an INT32 bias",
                               node_id=node.id)
        bsrc = initializers.get(node.inputs[2])
        if bsrc is None:
            raise CompileError(E_CFG, "source depthwise bias missing", node_id=node.id)
        if bsrc["dtype"] != "int32":
            raise CompileError(E_CFG, "source depthwise bias must be INT32",
                               node_id=node.id)
        bias_values = [int(v) for v in bsrc["values"]]
    bias_list = [int(v) for v in bias_values]
    if len(bias_list) != cin:
        raise CompileError(E_CFG, "source depthwise bias length mismatch", node_id=node.id)
    records = node.metadata.get("requant_records")
    if not isinstance(records, list) or len(records) != cin:
        raise CompileError(E_CFG, "source depthwise requant records missing",
                           node_id=node.id)
    out = [0] * (cin * hout * wout)
    for oy in range(hout):
        for ox in range(wout):
            for channel in range(cin):
                acc = bias_list[channel]
                for ky in range(3):
                    iy = oy * stride + ky - pad
                    if iy < 0 or iy >= h:
                        continue
                    for kx in range(3):
                        ix = ox * stride + kx - pad
                        if ix < 0 or ix >= w:
                            continue
                        acc += (x.values[_index(x.shape, 0, channel, iy, ix)] *
                                wvals[channel * 9 + ky * 3 + kx])
                if not (-(2 ** 31) <= acc <= 2 ** 31 - 1):
                    raise CompileError(E_CFG, "depthwise INT32 accumulator overflow",
                                       node_id=node.id)
                out[(channel * hout + oy) * wout + ox] = _requant(acc, records[channel])
    return TensorValue((1, cin, hout, wout), "int8", out)


def _dw_requant(acc: int, record: Dict[str, Any]) -> int:
    """Depthwise round-half-away-from-zero + sat8 (independent implementation)."""
    m0 = int(record["M0"])
    shift = int(record["SHIFT"])
    product = int(acc) * m0
    magnitude = abs(product)
    if shift > 0:
        magnitude = (magnitude + (1 << (shift - 1))) >> shift
    value = -magnitude if product < 0 else magnitude
    return _sat8(value + int(record.get("out_zero_point", 0)))


def _depthwise(node, x: TensorValue,
               initializers: Dict[str, Dict[str, Any]]) -> TensorValue:
    """Lowered DepthwiseConv reference (separate from the source grouped path)."""
    _, cin, h, w = x.shape
    group = int(node.attrs.get("group", 1))
    if group != cin:
        raise CompileError(E_CFG, "lowered DepthwiseConv group must equal C",
                           node_id=node.id)
    weight = initializers.get(node.inputs[1])
    if weight is None:
        raise CompileError(E_CFG, "lowered DepthwiseConv weight missing", node_id=node.id)
    if tuple(int(v) for v in weight["shape"]) != (cin, 1, 3, 3):
        raise CompileError(E_CFG, "lowered DepthwiseConv weight shape mismatch",
                           node_id=node.id)
    wvals = [int(v) for v in weight["values"]]
    bias_list = node.metadata.get("folded_bias_values")
    if bias_list is None:
        bsrc = initializers.get(node.inputs[2]) if len(node.inputs) >= 3 else None
        if bsrc is None or bsrc["dtype"] != "int32":
            raise CompileError(E_CFG, "lowered DepthwiseConv INT32 bias missing",
                               node_id=node.id)
        bias_list = [int(v) for v in bsrc["values"]]
    bias_list = [int(v) for v in bias_list]
    if len(bias_list) != cin:
        raise CompileError(E_CFG, "lowered DepthwiseConv bias length mismatch",
                           node_id=node.id)
    records = node.metadata.get("requant_records")
    if not isinstance(records, list) or len(records) != cin:
        raise CompileError(E_CFG, "lowered DepthwiseConv requant records missing",
                           node_id=node.id)
    kh = int(node.attrs.get("kernel_shape", [3, 3])[0])
    kw = int(node.attrs.get("kernel_shape", [3, 3])[1])
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    hout = (h + 2 * pad - kh) // stride + 1
    wout = (w + 2 * pad - kw) // stride + 1
    out = [0] * (cin * hout * wout)
    for channel in range(cin):
        for oy in range(hout):
            for ox in range(wout):
                acc = bias_list[channel]
                for ky in range(kh):
                    iy = oy * stride + ky - pad
                    if iy < 0 or iy >= h:
                        continue
                    for kx in range(kw):
                        ix = ox * stride + kx - pad
                        if ix < 0 or ix >= w:
                            continue
                        acc += (x.values[_index(x.shape, 0, channel, iy, ix)] *
                                wvals[channel * 9 + ky * 3 + kx])
                if not (-(2 ** 31) <= acc <= 2 ** 31 - 1):
                    raise CompileError(E_CFG, "depthwise INT32 accumulator overflow",
                                       node_id=node.id)
                out[(channel * hout + oy) * wout + ox] = _dw_requant(acc, records[channel])
    return TensorValue((1, cin, hout, wout), "int8", out)


def _average_pool(node, x: TensorValue) -> TensorValue:
    """2x2 stride 2 pad 0 exact mean: INT32 sum, ties-away by 4, INT8 saturate."""
    _, c, h, w = x.shape
    if h % 2 or w % 2:
        raise CompileError(E_CFG, "AveragePool requires even H/W", node_id=node.id)
    hout, wout = h // 2, w // 2
    out = [0] * (c * hout * wout)
    for oy in range(hout):
        for ox in range(wout):
            for ch in range(c):
                total = (x.values[_index(x.shape, 0, ch, 2 * oy, 2 * ox)] +
                         x.values[_index(x.shape, 0, ch, 2 * oy, 2 * ox + 1)] +
                         x.values[_index(x.shape, 0, ch, 2 * oy + 1, 2 * ox)] +
                         x.values[_index(x.shape, 0, ch, 2 * oy + 1, 2 * ox + 1)])
                q, rem = divmod(abs(total), 4)
                if 2 * rem >= 4:
                    q += 1
                out[(ch * hout + oy) * wout + ox] = _sat8(-q if total < 0 else q)
    return TensorValue((1, c, hout, wout), x.dtype, out)


def _resize(node, x: TensorValue) -> TensorValue:
    """Independent nearest/asymmetric/floor x2 reference of ONNX Resize.

    Logical NCHW INT8, identity scale: y[c,2y+dy,2x+dx] = x[c,y,x] for
    dy,dx in {0,1}.  Derived from the node attributes/shape only, so a wrong
    frontend mapping cannot hide behind shared code.
    """
    mode = str(node.attrs.get("mode", "nearest"))
    coordinate = str(node.attrs.get("coordinate_transformation_mode", "asymmetric"))
    nearest = str(node.attrs.get("nearest_mode", "floor"))
    if mode != "nearest" or coordinate != "asymmetric" or nearest != "floor":
        raise CompileError(E_UNSUPPORTED_OP,
                           "Resize reference implements nearest/asymmetric/floor only, got "
                           "%s/%s/%s" % (mode, coordinate, nearest), node_id=node.id)
    _, c, h, w = x.shape
    out_h, out_w = 2 * h, 2 * w
    out = [0] * (c * out_h * out_w)
    for ch in range(c):
        for y in range(h):
            for xx in range(w):
                value = x.values[_index(x.shape, 0, ch, y, xx)]
                for dy in (0, 1):
                    for dx in (0, 1):
                        out[(ch * out_h + 2 * y + dy) * out_w + 2 * xx + dx] = value
    return TensorValue((1, c, out_h, out_w), x.dtype, out)


def _maxpool5(node, x: TensorValue) -> TensorValue:
    """5x5 stride 1 pad 2 signed INT8 window max with -128 padding.

    Deliberately does *not* clamp the result to 0: an all-negative window must
    stay negative.  The legacy 2x2 branch keeps its historical max(0, ...).
    """
    if [int(v) for v in node.attrs.get("kernel_shape", [5, 5])] != [5, 5]:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires kernel [5,5]",
                           node_id=node.id)
    if [int(v) for v in node.attrs.get("strides", [1, 1])] != [1, 1]:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires stride 1",
                           node_id=node.id)
    if [int(v) for v in node.attrs.get("pads", [2, 2, 2, 2])] != [2, 2, 2, 2]:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires pads 2",
                           node_id=node.id)
    if [int(v) for v in node.attrs.get("dilations", [1, 1])] != [1, 1]:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires dilation 1",
                           node_id=node.id)
    if int(node.attrs.get("ceil_mode", 0)) != 0:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires ceil_mode 0",
                           node_id=node.id)
    if int(node.attrs.get("storage_order", 0)) != 0:
        raise CompileError(E_UNSUPPORTED_OP, "MaxPool5 reference requires storage_order 0",
                           node_id=node.id)
    _, c, h, w = x.shape
    out = [0] * (c * h * w)
    for ch in range(c):
        for oy in range(h):
            for ox in range(w):
                best = -128
                for ky in range(5):
                    iy = oy + ky - 2
                    for kx in range(5):
                        ix = ox + kx - 2
                        if iy < 0 or iy >= h or ix < 0 or ix >= w:
                            value = -128
                        else:
                            value = x.values[_index(x.shape, 0, ch, iy, ix)]
                        if value > best:
                            best = value
                out[(ch * h + oy) * w + ox] = best
    return TensorValue((1, c, h, w), x.dtype, out)


def _maxpool_generic(node, x: TensorValue) -> TensorValue:
    from .maxpool2d_geometry import geometry
    g = geometry(x.shape, node.attrs)
    out = [-128] * (g.channels * g.hout * g.wout)
    for ch in range(g.channels):
        for oy in range(g.hout):
            for ox in range(g.wout):
                best = -128
                for ky in range(g.kh):
                    for kx in range(g.kw):
                        iy = oy * g.sh - g.pt + ky * g.dh
                        ix = ox * g.sw - g.pl + kx * g.dw
                        if 0 <= iy < g.hin and 0 <= ix < g.win:
                            best = max(best, x.values[_index(x.shape, 0, ch, iy, ix)])
                out[(ch * g.hout + oy) * g.wout + ox] = best
    return TensorValue((1, g.channels, g.hout, g.wout), x.dtype, out)


def _activation_lut(node, x: TensorValue) -> TensorValue:
    # Deployment INT8 ActivationLUT reference via independent oracle (not generator tables).
    # This keeps graph-level reference independent of parameter_pack generator.
    from .activation_oracle import oracle_table
    mode = str((node.attrs or {}).get("mode", ""))
    if mode not in ("Sigmoid", "Tanh", "SiLU", "LeakyRelu"):
        raise CompileError(E_CFG, "ActivationLUT mode unsupported in reference", node_id=node.id)
    in_q = (node.metadata or {}).get("lut_input_scale")
    out_q = (node.metadata or {}).get("lut_output_scale")
    # Fallback to tensor quantization when metadata not yet populated (S2 unit tests).
    try:
        from .graph_ir import CompileError as _CE
    except Exception:
        pass
    # Resolve scales from node metadata or attached graph tensors is done by caller;
    # here we require metadata scales for determinism.
    if in_q is None or out_q is None:
        raise CompileError(E_CFG, "ActivationLUT reference missing lut scales", node_id=node.id)
    alpha = (node.attrs or {}).get("alpha", 0.01) if mode == "LeakyRelu" else None
    table = oracle_table(mode, float(in_q), float(out_q), alpha)
    lut = {q: table[q + 128] for q in range(-128, 128)}
    out = [int(lut[int(v)]) for v in x.values]
    return TensorValue(x.shape, "int8", out)


def _mul(node, a: TensorValue, b: TensorValue) -> TensorValue:
    record = node.metadata.get("mul_requant")
    if not isinstance(record, dict):
        raise CompileError(E_CFG, "Mul requant record missing", node_id=node.id)
    if a.shape != b.shape:
        raise CompileError(E_CFG, "Mul shape mismatch", node_id=node.id)
    out = []
    for x, y in zip(a.values, b.values):
        exact_value = Fraction(int(x) * int(y) * int(record["M0"]),
                               1 << int(record["SHIFT"]))
        out.append(_sat8(round_away(exact_value)))
    return TensorValue(a.shape, a.dtype, out)


def _batch_norm(node, x: TensorValue, initializers: Dict[str, Dict[str, Any]],
                graph: Graph) -> TensorValue:
    """Independent reference of the *unfolded* Conv->BN integer semantics.

    The BN sees the real accumulator value acc * s_input * weight_scale and
    rounds (A*acc + B)/s_out once, ties away from zero.  Equality with the
    folded Conv is a property of the exact parameters, not of the samples.
    """
    if x.accumulators is None or x.accumulator_shape is None:
        raise CompileError(E_CFG,
                           "BatchNormalization reference needs the fused Conv "
                           "accumulators", node_id=node.id)
    producer_name = graph.tensors[node.inputs[0]].producer
    conv = next((row for row in graph.nodes
                 if row.id == producer_name or node.inputs[0] in row.outputs), None)
    if conv is None or conv.op_type != "Conv":
        raise CompileError(E_CFG, "BatchNormalization reference needs a Conv producer",
                           node_id=node.id)
    s_in = conv.metadata.get("quant_input_scale")
    weight_scales = conv.metadata.get("weight_scales")
    if s_in is None or not weight_scales:
        raise CompileError(E_CFG, "Conv lacks contract scales for BN reference", node_id=conv.id)
    out_quant = graph.tensors[node.outputs[0]].quantization or {}
    if "scale" not in out_quant:
        raise CompileError(E_CFG, "BatchNormalization output has no scale", node_id=node.id)
    s_out = out_quant["scale"]
    gamma = [float(v) for v in initializers[node.inputs[1]]["values"]]
    beta = [float(v) for v in initializers[node.inputs[2]]["values"]]
    mean = [float(v) for v in initializers[node.inputs[3]]["values"]]
    var = [float(v) for v in initializers[node.inputs[4]]["values"]]
    epsilon = float(node.attrs["epsilon"])
    cout, hout, wout = x.accumulator_shape
    out = [0] * (cout * hout * wout)
    for ch in range(cout):
        terms = bn_channel_terms(gamma[ch], beta[ch], mean[ch], var[ch], epsilon)
        a_value = exact(terms["a_float"])
        b_value = exact(terms["b_float"])
        s_acc = exact(s_in) * exact(weight_scales[ch])
        denominator = exact(s_out)
        for position in range(ch * hout * wout, (ch + 1) * hout * wout):
            value = (Fraction(x.accumulators[position]) * s_acc * a_value + b_value) / denominator
            out[position] = _sat8(round_away(value))
    return TensorValue((1, cout, hout, wout), "int8", out)


def _pad(node, x: TensorValue) -> TensorValue:
    pads = [int(v) for v in node.metadata["pads"]]
    top, left, bottom, right = pads[2], pads[3], pads[6], pads[7]
    _, c, h, w = x.shape
    hout, wout = h + top + bottom, w + left + right
    out = [0] * (c * hout * wout)
    for y in range(h):
        for xx in range(w):
            for ch in range(c):
                out[(ch * hout + (y + top)) * wout + (xx + left)] = \
                    x.values[_index(x.shape, 0, ch, y, xx)]
    return TensorValue((1, c, hout, wout), x.dtype, out)


def _slice(node, x: TensorValue) -> TensorValue:
    starts = [int(v) for v in node.metadata["starts"]]
    ends = [int(v) for v in node.metadata["ends"]]
    _, c, h, w = x.shape
    c0, h0, w0 = starts[1], starts[2], starts[3]
    co, ho, wo = ends[1] - c0, ends[2] - h0, ends[3] - w0
    out = [0] * (co * ho * wo)
    for y in range(ho):
        for xx in range(wo):
            for ch in range(co):
                out[(ch * ho + y) * wo + xx] = \
                    x.values[_index(x.shape, 0, c0 + ch, h0 + y, w0 + xx)]
    return TensorValue((1, co, ho, wo), x.dtype, out)


def _concat(node, inputs: Sequence[TensorValue]) -> TensorValue:
    _, _, h, w = inputs[0].shape
    total_c = sum(int(v.shape[1]) for v in inputs)
    out = [0] * (total_c * h * w)
    offset = 0
    for value in inputs:
        _, c, _, _ = value.shape
        for y in range(h):
            for xx in range(w):
                for ch in range(c):
                    out[((offset + ch) * h + y) * w + xx] = \
                        value.values[_index(value.shape, 0, ch, y, xx)]
        offset += c
    return TensorValue((1, total_c, h, w), inputs[0].dtype, out)


def _gap(x: TensorValue) -> TensorValue:
    _, c, h, w = x.shape
    n = h * w
    out = []
    for ch in range(c):
        total = 0
        for y in range(h):
            for xx in range(w):
                total += x.values[_index(x.shape, 0, ch, y, xx)]
        a = abs(total)
        q, r = divmod(a, n)
        if 2 * r >= n:
            q += 1
        out.append(_sat8(-q if total < 0 else q))
    return TensorValue((1, c, 1, 1), "int8", out)


def _split(node, x: TensorValue) -> List[TensorValue]:
    _, c, h, w = x.shape
    outputs = []
    for spec in node.metadata["split_specs"]:
        start = int(spec["start"])
        length = int(spec["length"])
        out = [0] * (length * h * w)
        for y in range(h):
            for xx in range(w):
                for ch in range(length):
                    out[(ch * h + y) * w + xx] = \
                        x.values[_index(x.shape, 0, start + ch, y, xx)]
        outputs.append(TensorValue((1, length, h, w), x.dtype, out))
    return outputs


def evaluate_graph(graph: Graph, initializers: Dict[str, Dict[str, Any]],
                   inputs: Dict[str, Any]) -> Dict[str, Any]:
    _evaluate_counters()["evaluate_graph_calls"] += 1
    values: Dict[str, TensorValue] = {}
    for name, meta in initializers.items():
        if name in graph.tensors:
            values[name] = TensorValue(tuple(int(v) for v in meta["shape"]),
                                       str(meta["dtype"]), [int(v) for v in meta["values"]])
    for name, meta in inputs.items():
        values[name] = TensorValue(tuple(int(v) for v in meta["shape"]), "int8",
                                   [_signed8(int(v)) for v in meta["values"]])
    node_outputs: Dict[str, List[Dict[str, Any]]] = {}
    for node in graph.nodes:
        if node.op_type == "Constant":
            out = TensorValue(tuple(int(v) for v in node.metadata["constant_shape"]),
                              str(node.metadata["constant_dtype"]),
                              [int(v) for v in node.metadata["constant_values"]])
            values[node.outputs[0]] = out
            node_outputs[node.id] = [{"name": node.outputs[0], "shape": list(out.shape),
                                      "dtype": out.dtype, "values": list(out.values)}]
            continue
        ins = [values[name] for name in node.inputs if name in values]
        if node.op_type == "Conv":
            results = [_conv(node, ins[0], initializers)]
        elif node.op_type == "MaxPool":
            if "legacy_layer_index" in node.metadata:
                # Legacy 2x2/stride2/pad0 branch: unchanged max(0, ...) behavior.
                x = ins[0]
                _, c, h, w = x.shape
                out = [0] * (c * (h // 2) * (w // 2))
                for oy in range(h // 2):
                    for ox in range(w // 2):
                        for ch in range(c):
                            m = max(x.values[_index(x.shape, 0, ch, 2 * oy + dy, 2 * ox + dx)]
                                    for dy in (0, 1) for dx in (0, 1))
                            out[(ch * (h // 2) + oy) * (w // 2) + ox] = max(0, m)
                results = [TensorValue((1, c, h // 2, w // 2), x.dtype, out)]
            else:
                results = [_maxpool_generic(node, ins[0])]
        elif node.op_type == "Resize":
            results = [_resize(node, ins[0])]
        elif node.op_type == "DepthwiseConv":
            results = [_depthwise(node, ins[0], initializers)]
        elif node.op_type == "AveragePool":
            results = [_average_pool(node, ins[0])]
        elif node.op_type == "Mul":
            results = [_mul(node, ins[0], ins[1])]
        elif node.op_type == "ActivationLUT":
            attach_lut_scales(graph)
            results = [_activation_lut(node, ins[0])]
        elif node.op_type == "BatchNormalization":
            results = [_batch_norm(node, ins[0], initializers, graph)]
        elif node.op_type == "Relu":
            x = ins[0]
            results = [TensorValue(x.shape, x.dtype, [max(0, v) for v in x.values])]
        elif node.op_type == "Add":
            a, b = ins[0], ins[1]
            if a.shape != b.shape:
                raise CompileError(E_CFG, "Add shape mismatch", node_id=node.id)
            results = [TensorValue(a.shape, a.dtype,
                                   [_sat8(x + y) for x, y in zip(a.values, b.values)])]
        elif node.op_type == "Pad":
            results = [_pad(node, ins[0])]
        elif node.op_type == "Slice":
            results = [_slice(node, ins[0])]
        elif node.op_type == "Concat":
            results = [_concat(node, ins)]
        elif node.op_type == "Clip":
            x = ins[0]
            lo, hi = int(node.metadata["clip_min"]), int(node.metadata["clip_max"])
            results = [TensorValue(x.shape, x.dtype, [max(lo, min(hi, v)) for v in x.values])]
        elif node.op_type == "GlobalAveragePool":
            results = [_gap(ins[0])]
        elif node.op_type == "Split":
            results = _split(node, ins[0])
        elif node.op_type in ("Identity", "Squeeze", "Unsqueeze", "Flatten", "Reshape"):
            x = ins[0]
            shape = graph.tensors[node.outputs[0]].shape
            results = [TensorValue(tuple(int(v) for v in shape), x.dtype, list(x.values))]
        elif node.op_type == "Gemm":
            raise CompileError(E_UNSUPPORTED_OP,
                               "Gemm is a PS boundary and is not in the P2 graphs", node_id=node.id)
        else:
            raise CompileError(E_UNSUPPORTED_OP,
                               "reference has no implementation for %s" % node.op_type,
                               node_id=node.id)
        if len(results) != len(node.outputs):
            raise CompileError(E_CFG, "%s output count mismatch" % node.op_type, node_id=node.id)
        rows = []
        for name, result in zip(node.outputs, results):
            values[name] = result
            rows.append({"name": name, "shape": list(result.shape), "dtype": result.dtype,
                         "values": list(result.values),
                         "sha256_values": hashlib.sha256(
                             bytes((int(v) & 0xFF) for v in result.values)).hexdigest()})
        node_outputs[node.id] = rows
    missing = [name for name in graph.outputs if name not in values]
    if missing:
        raise CompileError(E_CFG, "reference missing graph outputs: %s" % ", ".join(missing))
    _evaluate_counters()["nodes_evaluated"] += len(graph.nodes)
    return {"values": values, "node_outputs": node_outputs}


def physical_outputs(result: Dict[str, Any], names: Sequence[str]) -> Dict[str, bytes]:
    out = {}
    for name in names:
        value = result["values"][name]
        if value.dtype == "int8" and len(value.shape) == 4:
            out[name] = pack_activation(value.shape, value.values, dtype=value.dtype)
    return out
