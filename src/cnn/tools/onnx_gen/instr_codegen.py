"""Instruction stream v2 generation for E1 W3.

Opcode 1/2 records retain the frozen 17-word Conv/Pool payload.  Relu/Add use
opcodes 0x0010/0x0011 and a 7-word elementwise payload.  E1-P1 adds Pad/Slice/
Concat with opcodes 0x0012/0x0013/0x0014 and a fixed 27-word copy/transform
payload.  PS-only nodes and metadata alias nodes emit no hardware record.

E1 direct-operator extension adds Resize (opcode 0x0019, 9-word payload) and
MaxPool 5x5/s1/pad2 (opcode 0x001A, 17-word payload).  Legacy MaxPool 2x2 keeps
opcode 0x0002 and its byte-identical payload.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .graph_ir import (
    E_CFG,
    E_QUANT_MISMATCH,
    E_SPATIAL_LIMIT,
    E_UNSUPPORTED_OP,
    CompileError,
    Graph,
    align_up,
)
from .shape_infer import depthwise_quant_storage, quant_storage

OPCODE_CONV = 0x0001
OPCODE_MAXPOOL = 0x0002
OPCODE_RELU = 0x0010
OPCODE_ADD = 0x0011
OPCODE_PAD = 0x0012
OPCODE_SLICE = 0x0013
OPCODE_CONCAT = 0x0014
OPCODE_CLIP = 0x0015
OPCODE_GAP = 0x0016
OPCODE_AVGPOOL = 0x0017
OPCODE_MUL = 0x0018
OPCODE_ACTLUT = 0x001B
OPCODE_RESIZE = 0x0019
OPCODE_MAXPOOL5 = 0x001A
OPCODE_DEPTHWISE = 0x001C
OPCODE_MAXPOOL2D = 0x001D

CONV_PAYLOAD_WORDS = 17
ELEMENTWISE_PAYLOAD_WORDS = 7
#: affine-int8-v1 Add: 7 legacy words + m0_a, m0_b, (shift, zp_a, zp_b, zp_out)
ELEMENTWISE_PAYLOAD_WORDS_AFFINE = 10
COPY_PAYLOAD_WORDS = 27
#: affine-int8-v1 copy/transform v2: versioned 40-word record (word 23 bit16 =
#: 1).  It keeps the v1 address/geometry words 0..21 and adds 16-bit channel /
#: channel-alloc / pixel-stride fields plus the per-input rescale parameters.
#: The copy config port is widened to 6 bits / 64 words for this record.
COPY_PAYLOAD_WORDS_V2 = 40
COPY_FORMAT_V2_BIT = 16
REDUCE_PAYLOAD_WORDS = 9
RESIZE_PAYLOAD_WORDS = 9
ACTLUT_PAYLOAD_WORDS = 17
DEPTHWISE_PAYLOAD_WORDS = 17

COPY_MODE_PAD = 1
COPY_MODE_SLICE = 2
COPY_MODE_CONCAT = 3
STREAM_SCHEMA = "e1-instruction-stream-2"
MEM_BASE = 0x10000000


def _hex(value: int) -> str:
    return "0x%08X" % (int(value) & 0xFFFFFFFF)


def _addr(value: Any) -> int:
    if value is None:
        raise CompileError(E_CFG, "required allocation address is missing")
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text, 16) if text.lower().startswith("0x") else int(text)


def _binding(plan: Dict[str, Any], name: str) -> Dict[str, Any]:
    bindings = plan["memory_plan"]["tensor_bindings"]
    if name not in bindings:
        raise CompileError(E_CFG, "tensor %s has no memory binding" % name)
    return bindings[name]


def _storage(plan: Dict[str, Any], name: str) -> Dict[str, Any]:
    tensors = plan["layout_plan"]["tensors"]
    if name not in tensors:
        raise CompileError(E_CFG, "tensor %s has no layout entry" % name)
    return tensors[name]["storage"]


def _shape(graph: Graph, name: str) -> Tuple[int, ...]:
    shape = graph.tensors[name].shape
    if shape is None:
        raise CompileError(E_CFG, "tensor %s has no shape" % name)
    return tuple(int(v) for v in shape)


def _conv_cfg(node) -> int:
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    cfg = (0 | ((stride - 1) << 2) | (pad << 4) | (1 << 10)) & 0xFFFFFFFF
    # affine-int8-v1 native K1: conv_cfg[11] tells the dense core that the weight
    # buffer holds only the center-tap group of a logically 1x1 Conv (1/9 of the
    # frozen nine-tap layout).  Legacy builds never set this reserved bit.
    if (node.metadata.get("affine") and
            [int(v) for v in node.metadata.get("source_kernel_shape", [])] == [1, 1]):
        cfg |= (1 << 11)
    return cfg & 0xFFFFFFFF


def _quant_record(scale: float, zero_point: int = 0) -> Dict[str, Any]:
    if zero_point != 0:
        raise CompileError(E_QUANT_MISMATCH, "W3 elementwise requires zero_point=0")
    # Integer requantization record used by the Conv core: M0/SHIFT.
    if scale <= 0:
        raise CompileError(E_QUANT_MISMATCH, "quantization scale must be positive")
    shift = 30
    m0 = int(round(scale * (1 << shift)))
    if not (0 <= m0 <= 0xFFFFFFFF):
        raise CompileError(E_QUANT_MISMATCH, "quantization scale out of M0 range")
    return {"M0": m0, "SHIFT": shift, "out_zero_point": 0, "reserved": 0}


def _validate_elementwise_quant(graph: Graph, node) -> None:
    tensors = [graph.tensors[name] for name in node.inputs + node.outputs]
    for tensor in tensors:
        if tensor.dtype != "int8":
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires INT8 tensors (node %s)" % (node.op_type, node.id),
                               node_id=node.id)
        if not tensor.quantization or int(tensor.quantization.get("zero_point", 0)) != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires explicit zero_point=0 on %s" % (node.op_type, tensor.name),
                               node_id=node.id)
    q0 = graph.tensors[node.inputs[0]].quantization
    qout = graph.tensors[node.outputs[0]].quantization
    for key in ("scale", "zero_point", "dtype"):
        if q0.get(key) != qout.get(key):
            raise CompileError(E_QUANT_MISMATCH,
                               "%s input/output quant mismatch on %s" % (node.op_type, key),
                               node_id=node.id)
    if node.op_type == "Add":
        q1 = graph.tensors[node.inputs[1]].quantization
        for key in ("scale", "zero_point", "dtype"):
            if q0.get(key) != q1.get(key):
                raise CompileError(E_QUANT_MISMATCH,
                                   "Add input quant mismatch on %s" % key, node_id=node.id)


def _side_quant_allocations(graph: Graph, plan: Dict[str, Any], alignment: int) -> Dict[str, Dict[str, Any]]:
    """Return quant buffers for generic plans; legacy plans already contain them."""
    bindings = plan["memory_plan"]["tensor_bindings"]
    base = _addr(plan["memory_plan"]["arena"]["base"])
    cursor = int(plan["memory_plan"]["arena"]["used_bytes"])
    result: Dict[str, Dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op_type == "DepthwiseConv":
            c = int(_shape(graph, node.outputs[0])[1])
            size = int(depthwise_quant_storage((c,))["bytes"])
            offset = align_up(cursor, alignment)
            cursor = offset + align_up(size, alignment)
            result[node.id] = {"name": "dwc%02d.quant" % node.order,
                               "address": _hex(base + offset), "offset": offset,
                               "bytes": size, "allocated_bytes": align_up(size, alignment)}
            continue
        if node.op_type != "Conv":
            continue
        layer_index = int(node.metadata.get("legacy_layer_index", node.order))
        names = ["layer%02d.quant" % layer_index, "conv%02d.quant" % node.order]
        existing = next((name for name in names if name in bindings and bindings[name]["address"]), None)
        if existing is not None:
            row = bindings[existing]
            result[node.id] = {"name": existing, "address": row["address"],
                               "offset": int(row["offset"]), "bytes": int(row["bytes"]),
                               "allocated_bytes": int(row["allocated_bytes"])}
            continue
        cout = int(_shape(graph, node.outputs[0])[1])
        size = int(quant_storage((cout,))["bytes"])
        offset = align_up(cursor, alignment)
        cursor = offset + align_up(size, alignment)
        result[node.id] = {"name": "conv%02d.quant" % node.order,
                           "address": _hex(base + offset), "offset": offset,
                           "bytes": size, "allocated_bytes": align_up(size, alignment)}
    return result


def _conv_affine_fields(graph: Graph, node) -> Tuple[int, int]:
    """Return (quant_cfg, act_cfg) for a Conv record.

    affine-int8-v1: quant_cfg[7:0] carries the input zero point zs_in (the router
    forwards it to cfg_x_zero_point; the core pads with it and removes
    (128 + zs_in)*sum(w) itself), and act_cfg[0] requests the fused ReLU clamp at
    zs_out.  Legacy builds keep 0/0 byte-for-byte.
    """
    if not node.metadata.get("affine"):
        return 0, 0
    zs_in = int(node.metadata.get("x_zero_point_s8", 0))
    if not (-128 <= zs_in <= 127):
        raise CompileError(E_QUANT_MISMATCH, "Conv %s zs_in outside int8" % node.id,
                           node_id=node.id)
    relu = bool(node.metadata.get("relu", False))
    return (zs_in & 0xFF), (1 if relu else 0)


def _conv_payload(graph: Graph, node, plan: Dict[str, Any], quant_row: Dict[str, Any]) -> List[int]:
    if len(node.inputs) != 3:
        raise CompileError(E_CFG, "W3 Conv codegen requires input, weight and bias", node_id=node.id)
    x_name, w_name, b_name = node.inputs
    out_name = node.outputs[0]
    x_shape = _shape(graph, x_name)
    out_shape = _shape(graph, out_name)
    w_storage = _storage(plan, w_name)
    b_storage = _storage(plan, b_name)
    x_binding = _binding(plan, x_name)
    w_binding = _binding(plan, w_name)
    b_binding = _binding(plan, b_name)
    out_binding = _binding(plan, out_name)
    x_store = _storage(plan, x_name)
    out_store = _storage(plan, out_name)
    quant_cfg_word, act_cfg_word = _conv_affine_fields(graph, node)
    payload = [
        _addr(x_binding["address"]) & 0xFFFFFFFF,
        _addr(w_binding["address"]) & 0xFFFFFFFF,
        _addr(b_binding["address"]) & 0xFFFFFFFF,
        _addr(quant_row["address"]) & 0xFFFFFFFF,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        (x_shape[2] & 0xFFFF) | ((x_shape[3] & 0xFFFF) << 16),
        (x_shape[1] & 0xFFFF) | ((out_shape[1] & 0xFFFF) << 16),
        _conv_cfg(node),
        quant_cfg_word,
        act_cfg_word,
        int(x_store["row_stride_bytes"]) & 0xFFFFFFFF,
        int(out_store["row_stride_bytes"]) & 0xFFFFFFFF,
        int(x_store["bytes"]) & 0xFFFFFFFF,
        int(w_storage["bytes"]) & 0xFFFFFFFF,
        int(b_storage["bytes"]) & 0xFFFFFFFF,
        int(quant_row["bytes"]) & 0xFFFFFFFF,
        int(out_store["bytes"]) & 0xFFFFFFFF,
    ]
    if len(payload) != CONV_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _depthwise_payload(graph: Graph, node, plan: Dict[str, Any],
                       quant_row: Dict[str, Any]) -> List[int]:
    """0x001C 17-word DepthwiseConv payload (frozen S3 section 1.2)."""
    if node.op_type != "DepthwiseConv":
        raise CompileError(E_CFG, "depthwise payload requires DepthwiseConv",
                           node_id=node.id)
    if len(node.inputs) != 3:
        raise CompileError(E_CFG, "DepthwiseConv requires input, weight and bias",
                           node_id=node.id)
    x_name, w_name, b_name = node.inputs
    out_name = node.outputs[0]
    x_shape = _shape(graph, x_name)
    out_shape = _shape(graph, out_name)
    if len(x_shape) != 4 or len(out_shape) != 4:
        raise CompileError(E_CFG, "DepthwiseConv requires rank-4 input/output",
                           node_id=node.id)
    n, c, h, w = (int(v) for v in x_shape)
    cout, hout, wout = int(out_shape[1]), int(out_shape[2]), int(out_shape[3])
    if n != 1 or cout != c:
        raise CompileError(E_CFG, "DepthwiseConv requires N=1 and Cin==Cout",
                           node_id=node.id)
    if not (1 <= c <= 16 and 1 <= h <= 32 and 1 <= w <= 32):
        raise CompileError(E_CFG, "DepthwiseConv shape outside frozen bounds",
                           node_id=node.id)
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    if stride not in (1, 2) or pad not in (0, 1):
        raise CompileError(E_CFG, "DepthwiseConv stride/pad outside {1,2}/{0,1}",
                           node_id=node.id)
    expected_hout = (h + 2 * pad - 3) // stride + 1
    expected_wout = (w + 2 * pad - 3) // stride + 1
    if (hout, wout) != (expected_hout, expected_wout) or hout <= 0 or wout <= 0:
        raise CompileError(E_CFG, "DepthwiseConv output geometry mismatch", node_id=node.id)

    x_binding = _binding(plan, x_name)
    w_binding = _binding(plan, w_name)
    b_binding = _binding(plan, b_name)
    out_binding = _binding(plan, out_name)
    x_store = _storage(plan, x_name)
    w_store = _storage(plan, w_name)
    b_store = _storage(plan, b_name)
    out_store = _storage(plan, out_name)
    c_alloc = int(x_store.get("channel_alloc", ((c + 7) // 8) * 8))
    if str(w_store.get("layout")) != "DW_OC16_K9":
        raise CompileError(E_CFG, "DepthwiseConv weight storage is not DW_OC16_K9",
                           node_id=node.id)
    if int(w_store.get("bytes", 0)) != c * 16:
        raise CompileError(E_CFG, "DepthwiseConv weight bytes != C*16", node_id=node.id)
    if int(b_store.get("bytes", 0)) != align_up(4 * c, 16):
        raise CompileError(E_CFG, "DepthwiseConv bias bytes != align16(4C)",
                           node_id=node.id)
    if int(quant_row.get("bytes", 0)) != align_up(8 * c, 16) or \
            int(quant_row.get("bytes", 0)) != int(depthwise_quant_storage((c,))["bytes"]):
        raise CompileError(E_CFG, "DepthwiseConv quant bytes != align16(8C)",
                           node_id=node.id)
    in_stride = int(x_store["row_stride_bytes"])
    out_stride = int(out_store["row_stride_bytes"])
    if in_stride != align_up(w * c_alloc, 16):
        raise CompileError(E_CFG, "DepthwiseConv input stride mismatch", node_id=node.id)
    if out_stride != align_up(wout * c_alloc, 16):
        raise CompileError(E_CFG, "DepthwiseConv output stride mismatch", node_id=node.id)
    in_bytes = h * in_stride
    out_bytes = hout * out_stride
    if int(x_store["bytes"]) != in_bytes:
        raise CompileError(E_CFG, "DepthwiseConv input bytes != H*in_stride",
                           node_id=node.id)
    if int(out_store["bytes"]) != out_bytes:
        raise CompileError(E_CFG, "DepthwiseConv output bytes != Hout*out_stride",
                           node_id=node.id)

    conv_cfg = (0x00000400 | ((stride - 1) << 2) | (pad << 4)) & 0xFFFFFFFF
    expected_cfg = (0x00000400 | ((stride - 1) << 2) | (pad << 4)) & 0xFFFFFFFF
    if conv_cfg != expected_cfg:
        raise CompileError(E_CFG, "DepthwiseConv conv_cfg mismatch", node_id=node.id)
    payload = [
        _addr(x_binding["address"]) & 0xFFFFFFFF,
        _addr(w_binding["address"]) & 0xFFFFFFFF,
        _addr(b_binding["address"]) & 0xFFFFFFFF,
        _addr(quant_row["address"]) & 0xFFFFFFFF,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        (int(h) & 0xFFFF) | ((int(w) & 0xFFFF) << 16),
        (int(c) & 0xFFFF) | ((int(cout) & 0xFFFF) << 16),
        conv_cfg,
        0x00000000,
        0x00000000,
        in_stride & 0xFFFFFFFF,
        out_stride & 0xFFFFFFFF,
        in_bytes & 0xFFFFFFFF,
        int(w_store["bytes"]) & 0xFFFFFFFF,
        int(b_store["bytes"]) & 0xFFFFFFFF,
        int(quant_row["bytes"]) & 0xFFFFFFFF,
        out_bytes & 0xFFFFFFFF,
    ]
    if len(payload) != DEPTHWISE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    if payload[7] != (0x00000400 | ((stride - 1) << 2) | (pad << 4)):
        raise CompileError(E_CFG, "DepthwiseConv conv_cfg encoding mismatch",
                           node_id=node.id)
    # five relocatable buffers: 16B aligned, pairwise disjoint, no 32-bit wrap
    ranges = [(payload[0], payload[12]), (payload[1], payload[13]),
              (payload[2], payload[14]), (payload[3], payload[15]),
              (payload[4], payload[16])]
    for index, (address, size) in enumerate(ranges):
        if address % 16:
            raise CompileError(E_CFG, "DepthwiseConv buffer %d not 16B aligned" % index,
                               node_id=node.id)
        if address + size > 0x100000000:
            raise CompileError(E_CFG, "DepthwiseConv buffer %d wraps 32-bit address" % index,
                               node_id=node.id)
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            a_start, a_size = ranges[i]
            b_start, b_size = ranges[j]
            if a_start < b_start + b_size and b_start < a_start + a_size:
                raise CompileError(E_CFG,
                                   "DepthwiseConv buffers %d/%d overlap" % (i, j),
                                   node_id=node.id)
    return payload


def _pool_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    x_name = node.inputs[0]
    out_name = node.outputs[0]
    x_shape = _shape(graph, x_name)
    out_shape = _shape(graph, out_name)
    x_binding = _binding(plan, x_name)
    out_binding = _binding(plan, out_name)
    x_store = _storage(plan, x_name)
    out_store = _storage(plan, out_name)
    stride = int(node.attrs["strides"][0])
    pad = int(node.attrs["pads"][0])
    conv_cfg = (0 | ((stride - 1) << 2) | (pad << 4) | (1 << 10)) & 0xFFFFFFFF
    payload = [
        _addr(x_binding["address"]) & 0xFFFFFFFF, 0, 0, 0,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        (x_shape[2] & 0xFFFF) | ((x_shape[3] & 0xFFFF) << 16),
        (x_shape[1] & 0xFFFF) | ((out_shape[1] & 0xFFFF) << 16),
        conv_cfg, 0, 0,
        int(x_store["row_stride_bytes"]) & 0xFFFFFFFF,
        int(out_store["row_stride_bytes"]) & 0xFFFFFFFF,
        int(x_store["bytes"]) & 0xFFFFFFFF,
        0, 0, 0, int(out_store["bytes"]) & 0xFFFFFFFF,
    ]
    if len(payload) != CONV_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _pool2d_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """Generic pure-max payload, with the full per-layer V1 geometry."""
    from .maxpool2d_geometry import encode_payload, geometry
    x_name, out_name = node.inputs[0], node.outputs[0]
    x, out = graph.tensors[x_name], graph.tensors[out_name]
    qx, qo = x.quantization or {}, out.quantization or {}
    if (x.dtype != "int8" or out.dtype != "int8" or not qx or not qo or
            qx.get("materialized") is False or qo.get("materialized") is False or
            "scale" not in qx or "scale" not in qo):
        raise CompileError(E_QUANT_MISMATCH, "MaxPool requires materialized INT8 domains", node_id=node.id)
    zx = qx.get("zero_point_s8", qx.get("zero_point", 0))
    zo = qo.get("zero_point_s8", qo.get("zero_point", 0))
    if float(qx["scale"]) <= 0 or qx["scale"] != qo["scale"] or int(zx) != int(zo):
        raise CompileError(E_QUANT_MISMATCH, "MaxPool input/output domains must match", node_id=node.id)
    g = geometry(_shape(graph, x_name), node.attrs)
    if _shape(graph, out_name) != (1, g.channels, g.hout, g.wout):
        raise CompileError(E_CFG, "MaxPool output shape is inconsistent", node_id=node.id)
    xs, ys = _storage(plan, x_name), _storage(plan, out_name)
    if (xs.get("layout") != "NHWC8_A16" or ys.get("layout") != "NHWC8_A16" or
            xs.get("channel_alloc") != g.cout_alloc or ys.get("channel_alloc") != g.cout_alloc or
            xs.get("row_stride_bytes") != g.input_row_bytes or ys.get("row_stride_bytes") != g.output_row_bytes or
            xs.get("bytes") != g.input_span_bytes or ys.get("bytes") != g.output_span_bytes):
        raise CompileError(E_CFG, "MaxPool storage geometry is inconsistent", node_id=node.id)
    return encode_payload(g, _addr(_binding(plan, x_name)["address"]),
                          _addr(_binding(plan, out_name)["address"]))


def _avgpool_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """17-word pool payload; input and output must share one zero-point-0 scale."""
    x_name, out_name = node.inputs[0], node.outputs[0]
    q = graph.tensors[x_name].quantization
    qo = graph.tensors[out_name].quantization
    if (not q or not qo or q.get("materialized") is False or
            qo.get("materialized") is False or q.get("scale") != qo.get("scale") or
            int(q.get("zero_point", 0)) != 0 or int(qo.get("zero_point", 0)) != 0):
        raise CompileError(E_QUANT_MISMATCH,
                           "AveragePool requires equal zero-point-0 input/output scales",
                           node_id=node.id)
    return _pool_payload(graph, node, plan)


def _require_identity_int8_quant(graph: Graph, node, in_name: str,
                                  out_name: str, label: str) -> None:
    """Resize / MaxPool5 are identity-arithmetic: same INT8 scale and zp 0."""
    for name in (in_name, out_name):
        tensor = graph.tensors[name]
        quant = tensor.quantization
        if (tensor.dtype != "int8" or not quant or
                quant.get("materialized") is False):
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires an explicit INT8 quantization on %s" %
                               (label, tensor.name), node_id=node.id)
        if int(quant.get("zero_point", 0)) != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires zero_point=0 on %s" % (label, tensor.name),
                               node_id=node.id)
    qin = graph.tensors[in_name].quantization
    qout = graph.tensors[out_name].quantization
    for key in ("scale", "zero_point", "dtype"):
        if qin.get(key) != qout.get(key):
            raise CompileError(E_QUANT_MISMATCH,
                               "%s input/output quant mismatch on %s" % (label, key),
                               node_id=node.id)


def _resize_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """9-word Resize nearest x2 payload (opcode 0x0019, contract B.1).

    ONNX Resize keeps the present roi/scales/sizes names in node.inputs after the
    frontend slot resolution; those are compile-time metadata with no DDR
    binding, so only node.inputs[0] (the data tensor) is addressed here.
    """
    if not 1 <= len(node.inputs) <= 4:
        raise CompileError(E_CFG,
                           "Resize codegen requires 1..4 positional inputs, got %d" %
                           len(node.inputs), node_id=node.id)
    x_name, out_name = node.inputs[0], node.outputs[0]
    _require_identity_int8_quant(graph, node, x_name, out_name, "Resize")
    x_shape = _shape(graph, x_name)
    out_shape = _shape(graph, out_name)
    if len(x_shape) != 4 or len(out_shape) != 4:
        raise CompileError(E_CFG, "Resize codegen requires rank-4 tensors",
                           node_id=node.id)
    in_h, in_w, channels = int(x_shape[2]), int(x_shape[3]), int(x_shape[1])
    if (int(out_shape[1]) != channels or
            (int(out_shape[2]), int(out_shape[3])) != (2 * in_h, 2 * in_w)):
        raise CompileError(E_CFG, "Resize lowering requires output (1,C,2H,2W)",
                           node_id=node.id)
    x_binding, out_binding = _binding(plan, x_name), _binding(plan, out_name)
    x_store, out_store = _storage(plan, x_name), _storage(plan, out_name)
    channel_alloc = int(x_store.get("channel_alloc") or align_up(channels, 8))
    out_alloc = int(out_store.get("channel_alloc") or channel_alloc)
    if channel_alloc != align_up(channels, 8) or out_alloc != channel_alloc:
        raise CompileError(E_CFG,
                           "Resize requires NHWC8_A16 channel_alloc %d on both sides "
                           "(C=%d, in=%d, out=%d)" % (align_up(channels, 8), channels,
                                                      channel_alloc, out_alloc),
                           node_id=node.id)
    payload = [
        _addr(x_binding["address"]) & 0xFFFFFFFF,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        int(x_store["bytes"]) & 0xFFFFFFFF,
        int(out_store["bytes"]) & 0xFFFFFFFF,
        int(x_store["row_stride_bytes"]) & 0xFFFFFFFF,
        int(out_store["row_stride_bytes"]) & 0xFFFFFFFF,
        (in_h & 0xFFFF) | ((in_w & 0xFFFF) << 16),
        (channels & 0xFFFF) | ((channels & 0xFFFF) << 16),
        channel_alloc & 0xFFFFFFFF,
    ]
    if len(payload) != RESIZE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _pool5_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """17-word MaxPool 5x5/s1/pad2 payload (opcode 0x001A, contract B.2).

    Same layout as the legacy pool payload except word 8 carries channel_alloc
    (align8(C)) instead of zero.
    """
    kernel = [int(v) for v in node.attrs.get("kernel_shape", [2, 2])]
    if kernel != [5, 5]:
        raise CompileError(E_UNSUPPORTED_OP,
                           "MaxPool5 codegen requires kernel_shape [5,5]",
                           node_id=node.id)
    if _conv_cfg(node) != 0x00000420:
        raise CompileError(E_UNSUPPORTED_OP,
                           "MaxPool5 codegen requires stride 1 / pad 2 / ACT_BYPASS",
                           node_id=node.id)
    x_name, out_name = node.inputs[0], node.outputs[0]
    _require_identity_int8_quant(graph, node, x_name, out_name, "MaxPool5")
    payload = _pool_payload(graph, node, plan)
    payload[8] = align_up(int(_shape(graph, x_name)[1]), 8) & 0xFFFFFFFF
    return payload


def _mul_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """7-word elementwise payload; word 6 carries M0[23:0] and SHIFT[28:24]."""
    record = node.metadata.get("mul_requant")
    if not isinstance(record, dict):
        raise CompileError(E_QUANT_MISMATCH, "Mul requant record is missing", node_id=node.id)
    m0, shift = int(record["M0"]), int(record["SHIFT"])
    if not (1 <= m0 <= 0xFFFFFF):
        raise CompileError(E_QUANT_MISMATCH,
                           "Mul M0=0x%X outside 1..0xFFFFFF" % m0, node_id=node.id)
    if not (0 <= shift <= 31):
        raise CompileError(E_QUANT_MISMATCH,
                           "Mul SHIFT=%d outside 0..31" % shift, node_id=node.id)
    a_name, b_name, out_name = node.inputs[0], node.inputs[1], node.outputs[0]
    a_shape, b_shape = _shape(graph, a_name), _shape(graph, b_name)
    if a_shape != b_shape:
        raise CompileError(E_UNSUPPORTED_BROADCAST,
                           "Mul payload requires identical input shapes", node_id=node.id)
    qa = graph.tensors[a_name].quantization
    qb = graph.tensors[b_name].quantization
    qo = graph.tensors[out_name].quantization
    for what, quant in (("input0", qa), ("input1", qb), ("output", qo)):
        if not quant or quant.get("materialized") is False or int(quant.get("zero_point", 0)) != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "Mul %s requires an explicit zero-point-0 scale" % what,
                               node_id=node.id)
    ab = _binding(plan, a_name), _binding(plan, b_name), _binding(plan, out_name)
    asb = _storage(plan, a_name), _storage(plan, b_name), _storage(plan, out_name)
    sizes = [int(row["bytes"]) for row in asb]
    if len(set(sizes)) != 1:
        raise CompileError(E_QUANT_MISMATCH, "Mul input/output byte sizes must match",
                           node_id=node.id)
    if sizes[0] % 16:
        raise CompileError(E_CFG, "Mul byte count %d is not 16-byte aligned" % sizes[0],
                           node_id=node.id)
    payload = [
        _addr(ab[0]["address"]) & 0xFFFFFFFF,
        _addr(ab[1]["address"]) & 0xFFFFFFFF,
        _addr(ab[2]["address"]) & 0xFFFFFFFF,
        sizes[0] & 0xFFFFFFFF,
        sizes[1] & 0xFFFFFFFF,
        sizes[2] & 0xFFFFFFFF,
        (m0 & 0xFFFFFF) | ((shift & 0x1F) << 24),
    ]
    if len(payload) != ELEMENTWISE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _add_affine_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """10-word affine Add record: two multipliers, one final rounding.

    Word 6 keeps the legacy flags slot and carries 0x00000002, which selects the
    rescale path in the elementwise core; words 7..9 carry m0_a, m0_b and the
    (shift, zp_a, zp_b, zp_out) pack.
    """
    params = node.metadata.get("add_requant")
    if not isinstance(params, dict):
        raise CompileError(E_QUANT_MISMATCH,
                           "affine Add %s has no add_requant parameters" % node.id,
                           node_id=node.id)
    if len(node.inputs) != 2:
        raise CompileError(E_CFG, "Add %s expects two inputs" % node.id, node_id=node.id)
    out = node.outputs[0]
    in0, in1 = node.inputs
    b0 = _binding(plan, in0)
    b1 = _binding(plan, in1)
    bo = _binding(plan, out)
    s0 = _storage(plan, in0)
    s1 = _storage(plan, in1)
    so = _storage(plan, out)
    m0_a = int(params["m0_a"])
    m0_b = int(params["m0_b"])
    shift = int(params["shift"])
    zp_a = int(params["zero_points_in_s8"][0])
    zp_b = int(params["zero_points_in_s8"][1])
    zp_o = int(params["zero_point_out_s8"])
    if not (1 <= m0_a <= 0xFFFFFFFF and 1 <= m0_b <= 0xFFFFFFFF):
        raise CompileError(E_QUANT_MISMATCH, "Add %s multiplier out of u32" % node.id,
                           node_id=node.id)
    if not (0 <= shift <= 63):
        raise CompileError(E_QUANT_MISMATCH, "Add %s shift out of 0..63" % node.id,
                           node_id=node.id)
    for value, what in ((zp_a, "zp_a"), (zp_b, "zp_b"), (zp_o, "zp_out")):
        if not (-128 <= value <= 127):
            raise CompileError(E_QUANT_MISMATCH, "Add %s %s outside int8" % (node.id, what),
                               node_id=node.id)
    payload = [
        _addr(b0["address"]) & 0xFFFFFFFF,
        _addr(b1["address"]) & 0xFFFFFFFF,
        _addr(bo["address"]) & 0xFFFFFFFF,
        int(s0["bytes"]) & 0xFFFFFFFF,
        int(s1["bytes"]) & 0xFFFFFFFF,
        int(so["bytes"]) & 0xFFFFFFFF,
        0x00000002,
        m0_a & 0xFFFFFFFF,
        m0_b & 0xFFFFFFFF,
        (shift & 0xFF) | ((zp_a & 0xFF) << 8) | ((zp_b & 0xFF) << 16) |
        ((zp_o & 0xFF) << 24),
    ]
    if len(payload) != ELEMENTWISE_PAYLOAD_WORDS_AFFINE:
        raise AssertionError(len(payload))
    return payload


def _elementwise_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    _validate_elementwise_quant(graph, node)
    in0 = node.inputs[0]
    out = node.outputs[0]
    in0_binding = _binding(plan, in0)
    out_binding = _binding(plan, out)
    in0_bytes = int(_storage(plan, in0)["bytes"])
    out_bytes = int(_storage(plan, out)["bytes"])
    if node.op_type == "Relu":
        if len(node.inputs) != 1:
            raise CompileError(E_CFG, "Relu expects one input", node_id=node.id)
        in1_addr = 0
        in1_bytes = 0
    elif node.op_type == "Add":
        if len(node.inputs) != 2:
            raise CompileError(E_CFG, "Add expects two inputs", node_id=node.id)
        in1 = node.inputs[1]
        in1_binding = _binding(plan, in1)
        in1_addr = _addr(in1_binding["address"])
        in1_bytes = int(_storage(plan, in1)["bytes"])
        if in1_bytes != in0_bytes or out_bytes != in0_bytes:
            raise CompileError(E_QUANT_MISMATCH, "Add byte sizes must match", node_id=node.id)
    else:
        raise CompileError(E_UNSUPPORTED_OP, "not an elementwise op", node_id=node.id)
    payload = [
        _addr(in0_binding["address"]) & 0xFFFFFFFF,
        in1_addr & 0xFFFFFFFF,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        in0_bytes & 0xFFFFFFFF,
        in1_bytes & 0xFFFFFFFF,
        out_bytes & 0xFFFFFFFF,
        0x00000001,
    ]
    if len(payload) != ELEMENTWISE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _validate_copy_quant(graph: Graph, node, names: List[str]) -> None:
    tensors = [graph.tensors[name] for name in names]
    for tensor in tensors:
        if tensor.dtype != "int8":
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires INT8 tensors (node %s)" % (node.op_type, node.id),
                               node_id=node.id)
        quant = tensor.quantization
        if not quant or int(quant.get("zero_point", 0)) != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires explicit zero_point=0 on %s" %
                               (node.op_type, tensor.name), node_id=node.id)
    first = tensors[0].quantization
    for tensor in tensors[1:]:
        for key in ("scale", "zero_point", "dtype"):
            if tensor.quantization.get(key) != first.get(key):
                raise CompileError(E_QUANT_MISMATCH,
                                   "%s quant mismatch on %s (%s)" %
                                   (node.op_type, key, tensor.name), node_id=node.id)


def _pack_nibbles(values: List[int], name: str) -> int:
    if len(values) > 4 or any(int(v) < 0 or int(v) > 255 for v in values):
        raise CompileError(E_CFG, "%s cannot be packed into four bytes" % name)
    word = 0
    for index, value in enumerate(values):
        word |= (int(value) & 0xFF) << (index * 8)
    return word & 0xFFFFFFFF


def _copy_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    out_name = node.outputs[0]
    out_shape = _shape(graph, out_name)
    out_store = _storage(plan, out_name)
    out_binding = _binding(plan, out_name)
    out_bytes = int(out_store["bytes"])
    out_row = int(out_store["row_stride_bytes"])
    out_c = int(out_shape[1])
    out_c_alloc = (out_c + 7) // 8 * 8

    in_names: List[str]
    mode: int
    src_h = src_w = src_c = 0
    dst_h = dst_w = dst_c = 0
    ch_offsets = [0, 0, 0, 0]
    zero_point = 0
    if node.op_type == "Pad":
        if len(node.inputs) not in (2, 3):
            raise CompileError(E_CFG, "Pad codegen requires data, pads and optional value",
                               node_id=node.id)
        if node.metadata.get("copy_required") is not True:
            raise CompileError(E_CFG, "Pad codegen requires an explicit copy record",
                               node_id=node.id)
        mode = COPY_MODE_PAD
        in_names = [node.inputs[0]]
        pads = [int(v) for v in node.metadata["pads"]]
        dst_h, dst_w = pads[2], pads[3]
        zero_point = int(node.metadata.get("constant_value", 0))
    elif node.op_type == "Slice":
        if len(node.inputs) not in (3, 4, 5):
            raise CompileError(E_CFG, "Slice codegen requires data/starts/ends",
                               node_id=node.id)
        if node.metadata.get("copy_required") is not True:
            raise CompileError(E_CFG, "Slice codegen requires an explicit copy record",
                               node_id=node.id)
        mode = COPY_MODE_SLICE
        in_names = [node.inputs[0]]
        starts = [int(v) for v in node.metadata["starts"]]
        src_h, src_w, src_c = starts[2], starts[3], starts[1]
    elif node.op_type == "Concat":
        if len(node.inputs) < 2 or len(node.inputs) > 4:
            raise CompileError(E_CFG, "Concat codegen requires 2..4 inputs", node_id=node.id)
        mode = COPY_MODE_CONCAT
        in_names = list(node.inputs)
        ch_offsets = [int(v) for v in node.metadata["channel_offsets"]]
    else:
        raise CompileError(E_UNSUPPORTED_OP, "not a copy/transform op", node_id=node.id)

    _validate_copy_quant(graph, node, in_names + [out_name])
    if zero_point != 0:
        raise CompileError(E_QUANT_MISMATCH, "copy/transform requires zero_point=0",
                           node_id=node.id)

    in_shapes = [_shape(graph, name) for name in in_names]
    in_h, in_w = int(in_shapes[0][2]), int(in_shapes[0][3])
    if node.op_type == "Concat":
        if any((int(shape[2]), int(shape[3])) != (in_h, in_w) for shape in in_shapes):
            raise CompileError(E_CFG, "Concat inputs disagree on H/W", node_id=node.id)
    in_c = int(in_shapes[0][1])
    in_c_alloc = (in_c + 7) // 8 * 8

    in_addrs: List[int] = []
    in_bytes_list: List[int] = []
    in_rows: List[int] = []
    in_cs: List[int] = []
    in_cas: List[int] = []
    for name, shape in zip(in_names, in_shapes):
        binding = _binding(plan, name)
        storage = _storage(plan, name)
        in_addrs.append(_addr(binding["address"]) & 0xFFFFFFFF)
        in_bytes_list.append(int(storage["bytes"]))
        in_rows.append(int(storage["row_stride_bytes"]))
        in_cs.append(int(shape[1]))
        in_cas.append((int(shape[1]) + 7) // 8 * 8)
    while len(in_addrs) < 4:
        in_addrs.append(0)
        in_bytes_list.append(0)
        in_rows.append(0)
        in_cs.append(0)
        in_cas.append(0)

    payload = [
        in_addrs[0], in_addrs[1], in_addrs[2], in_addrs[3],
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        in_bytes_list[0], in_bytes_list[1], in_bytes_list[2], in_bytes_list[3],
        out_bytes,
        in_rows[0], in_rows[1], in_rows[2], in_rows[3], out_row,
        ((in_h & 0xFFFF) << 16) | (in_w & 0xFFFF),
        ((out_shape[2] & 0xFFFF) << 16) | (out_shape[3] & 0xFFFF),
        ((out_c & 0xFFFF) << 16) | (in_c & 0xFFFF),
        ((out_c_alloc & 0xFFFF) << 16) | (in_c_alloc & 0xFFFF),
        ((src_h & 0xFFFF) << 16) | (src_w & 0xFFFF),
        ((dst_h & 0xFFFF) << 16) | (src_c & 0xFFFF),
        ((dst_c & 0xFFFF) << 16) | (dst_w & 0xFFFF),
        _pack_nibbles(ch_offsets, "Concat channel offsets"),
        ((len(in_names) & 0xFF) << 8) | (mode & 0xFF),
        _pack_nibbles(in_cs, "input channels"),
        _pack_nibbles(in_cas, "input channel allocations"),
        zero_point & 0xFFFFFFFF,
    ]
    if len(payload) != COPY_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _affine_tensor_domain(graph: Graph, name: str, label: str, node) -> Tuple[float, int]:
    """Real affine-int8-v1 activation domain (scale, signed zero point)."""
    tensor = graph.tensors[name]
    quant = tensor.quantization or {}
    if tensor.dtype != "int8" or not quant or quant.get("materialized") is False:
        raise CompileError(E_QUANT_MISMATCH,
                           "%s requires an explicit INT8 quantisation on %s" %
                           (label, name), node_id=node.id)
    scale = quant.get("scale")
    zp = quant.get("zero_point_s8")
    if scale is None or zp is None:
        raise CompileError(E_QUANT_MISMATCH,
                           "%s requires an affine-int8-v1 domain on %s" % (label, name),
                           node_id=node.id)
    scale = float(scale)
    zp = int(zp)
    if not scale > 0.0:
        raise CompileError(E_QUANT_MISMATCH, "%s scale must be positive on %s" %
                           (label, name), node_id=node.id)
    if not (-128 <= zp <= 127):
        raise CompileError(E_QUANT_MISMATCH, "%s signed zero point outside int8 on %s" %
                           (label, name), node_id=node.id)
    return scale, zp


def _copy_v2_rescale_params(graph: Graph, node, in_names: List[str],
                            out_name: str) -> Dict[str, Any]:
    """Per-input requantisation for the copy v2 record.

    Concat consumes the frozen affine contract rows (node.metadata["concat_quant"]);
    Pad/Slice derive the same single-rounding rescale from their real domains.
    Same-domain inputs are passthrough (no arithmetic).
    """
    out_scale, out_zp = _affine_tensor_domain(graph, out_name, node.op_type, node)
    rows: List[Dict[str, Any]] = []
    if node.op_type == "Concat":
        params = node.metadata.get("concat_quant")
        if not isinstance(params, dict):
            raise CompileError(E_QUANT_MISMATCH,
                               "affine Concat %s has no concat_quant parameters" % node.id,
                               node_id=node.id)
        if int(params.get("zero_point_out_s8", 0)) != int(out_zp):
            raise CompileError(E_QUANT_MISMATCH,
                               "affine Concat %s output zero point disagrees with the graph" %
                               node.id, node_id=node.id)
        contract_inputs = params.get("inputs") or []
        if len(contract_inputs) != len(in_names):
            raise CompileError(E_QUANT_MISMATCH,
                               "affine Concat %s input-parameter count mismatch" % node.id,
                               node_id=node.id)
        for index, name in enumerate(in_names):
            scale, zp = _affine_tensor_domain(graph, name, "Concat input", node)
            contract = contract_inputs[index]
            if int(contract.get("zero_point_in_s8")) != int(zp):
                raise CompileError(E_QUANT_MISMATCH,
                                   "affine Concat %s input %d zero point disagrees with the "
                                   "contract" % (node.id, index), node_id=node.id)
            passthrough = bool(contract.get("passthrough"))
            if passthrough:
                if not (scale == out_scale and zp == out_zp):
                    raise CompileError(E_QUANT_MISMATCH,
                                       "affine Concat %s input %d claims passthrough with a "
                                       "different domain" % (node.id, index), node_id=node.id)
                rows.append({"m0": 0, "shift": 0, "zs_in": zp, "bypass": True})
            else:
                m0_shift = contract.get("m0_shift") or {}
                m0 = int(m0_shift.get("m0", 0))
                shift = int(m0_shift.get("shift", 0))
                if not (1 <= m0 <= 0xFFFFFFFF):
                    raise CompileError(E_QUANT_MISMATCH,
                                       "affine Concat %s input %d M0 out of u32" %
                                       (node.id, index), node_id=node.id)
                if not (0 <= shift <= 62):
                    raise CompileError(E_QUANT_MISMATCH,
                                       "affine Concat %s input %d SHIFT out of 0..62" %
                                       (node.id, index), node_id=node.id)
                rows.append({"m0": m0, "shift": shift, "zs_in": zp, "bypass": False})
    else:
        from .affine_quant import choose_m0_shift
        for index, name in enumerate(in_names):
            scale, zp = _affine_tensor_domain(graph, name, node.op_type, node)
            if scale == out_scale and zp == out_zp:
                rows.append({"m0": 0, "shift": 0, "zs_in": zp, "bypass": True})
            else:
                params = choose_m0_shift(scale, 1.0, out_scale)
                rows.append({"m0": int(params["m0"]), "shift": int(params["shift"]),
                             "zs_in": zp, "bypass": False})
    return {"out_zs": int(out_zp), "inputs": rows}


def _copy_payload_v2(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    """40-word affine-int8-v1 copy/transform record (versioned v2).

    Word 23 bit16 marks the v2 format; words 0..21 keep the v1 address/geometry
    meaning (with wide 16-bit channel fields in words 22..31) and words 32..39
    carry the per-input rescale parameters.  The legacy 27-word record is never
    produced for an affine node, and a v1 consumer must reject the v2 marker
    instead of decoding truncated fields.
    """
    out_name = node.outputs[0]
    out_shape = _shape(graph, out_name)
    out_store = _storage(plan, out_name)
    out_binding = _binding(plan, out_name)
    out_bytes = int(out_store["bytes"])
    out_row = int(out_store["row_stride_bytes"])
    out_c = int(out_shape[1])
    out_ca = int(out_store.get("channel_alloc") or align_up(out_c, 8))
    if out_ca != align_up(out_c, 8):
        raise CompileError(E_CFG, "copy v2 output channel_alloc %d != align8(C=%d)" %
                           (out_ca, out_c), node_id=node.id)
    in_names: List[str]
    mode: int
    src_h = src_w = src_c = 0
    dst_h = dst_w = dst_c = 0
    ch_offsets = [0, 0, 0, 0]
    if node.op_type == "Pad":
        if len(node.inputs) not in (2, 3):
            raise CompileError(E_CFG, "Pad codegen requires data, pads and optional value",
                               node_id=node.id)
        if node.metadata.get("copy_required") is not True:
            raise CompileError(E_CFG, "Pad codegen requires an explicit copy record",
                               node_id=node.id)
        mode = COPY_MODE_PAD
        in_names = [node.inputs[0]]
        pads = [int(v) for v in node.metadata["pads"]]
        dst_h, dst_w = pads[2], pads[3]
        if int(node.metadata.get("constant_value", 0)) != 0:
            raise CompileError(E_UNSUPPORTED_OP,
                               "affine Pad supports only the encoded-zero fill value",
                               node_id=node.id)
    elif node.op_type == "Slice":
        if len(node.inputs) not in (3, 4, 5):
            raise CompileError(E_CFG, "Slice codegen requires data/starts/ends",
                               node_id=node.id)
        if node.metadata.get("copy_required") is not True:
            raise CompileError(E_CFG, "Slice codegen requires an explicit copy record",
                               node_id=node.id)
        mode = COPY_MODE_SLICE
        in_names = [node.inputs[0]]
        starts = [int(v) for v in node.metadata["starts"]]
        src_h, src_w, src_c = starts[2], starts[3], starts[1]
    elif node.op_type == "Concat":
        if len(node.inputs) < 2 or len(node.inputs) > 4:
            raise CompileError(E_CFG, "Concat codegen requires 2..4 inputs", node_id=node.id)
        mode = COPY_MODE_CONCAT
        in_names = list(node.inputs)
        ch_offsets = [int(v) for v in node.metadata["channel_offsets"]]
    else:
        raise CompileError(E_UNSUPPORTED_OP, "not a copy/transform op", node_id=node.id)

    if mode in (COPY_MODE_CONCAT, COPY_MODE_SLICE):
        if out_ca % 16:
            raise CompileError(E_UNSUPPORTED_OP,
                               "copy v2 %s requires a 16-byte pixel stride (out C_alloc=%d)" %
                               (node.op_type, out_ca), node_id=node.id)
    in_shapes = [_shape(graph, name) for name in in_names]
    in_h, in_w = int(in_shapes[0][2]), int(in_shapes[0][3])
    if mode == COPY_MODE_CONCAT:
        if any((int(shape[2]), int(shape[3])) != (in_h, in_w) for shape in in_shapes):
            raise CompileError(E_CFG, "Concat inputs disagree on H/W", node_id=node.id)
    in_c = int(in_shapes[0][1])
    in_ca = int(_storage(plan, in_names[0]).get("channel_alloc") or align_up(in_c, 8))

    in_addrs: List[int] = []
    in_bytes_list: List[int] = []
    in_rows: List[int] = []
    in_cs: List[int] = []
    in_cas: List[int] = []
    in_pss: List[int] = []
    for name, shape in zip(in_names, in_shapes):
        binding = _binding(plan, name)
        storage = _storage(plan, name)
        in_addrs.append(_addr(binding["address"]) & 0xFFFFFFFF)
        in_bytes_list.append(int(storage["bytes"]))
        in_rows.append(int(storage["row_stride_bytes"]))
        c = int(shape[1])
        ca = int(storage.get("channel_alloc") or align_up(c, 8))
        if ca != align_up(c, 8):
            raise CompileError(E_CFG, "copy v2 input channel_alloc %d != align8(C=%d)" %
                               (ca, c), node_id=node.id)
        in_cs.append(c)
        in_cas.append(ca)
        in_pss.append(ca)
    if mode == COPY_MODE_PAD:
        xs = out_shape
        if (in_c, in_ca, in_pss[0]) != (out_c, out_ca, out_ca):
            raise CompileError(E_UNSUPPORTED_OP,
                               "copy v2 Pad requires identical C/C_alloc/pixel stride on "
                               "input and output (in %d/%d/%d, out %d/%d/%d)" %
                               (in_c, in_ca, in_pss[0], out_c, out_ca, out_ca),
                               node_id=node.id)
    if mode == COPY_MODE_CONCAT:
        if sum(in_cs) != out_c:
            raise CompileError(E_CFG, "Concat channel sum %d != output C %d" %
                               (sum(in_cs), out_c), node_id=node.id)
        for index in range(len(in_cs)):
            off = ch_offsets[index]
            if off + in_cs[index] > out_ca:
                raise CompileError(E_CFG,
                                   "copy v2 Concat input %d channel range %d..%d exceeds "
                                   "output C_alloc %d" % (index, off, off + in_cs[index], out_ca),
                                   node_id=node.id)
            if index > 0 and off != ch_offsets[index - 1] + in_cs[index - 1]:
                raise CompileError(E_CFG, "Concat channel offsets are not contiguous",
                                   node_id=node.id)
    while len(in_addrs) < 4:
        in_addrs.append(0)
        in_bytes_list.append(0)
        in_rows.append(0)
        in_cs.append(0)
        in_cas.append(0)
        in_pss.append(0)
    while len(ch_offsets) < 4:
        ch_offsets.append(0)

    rescale = _copy_v2_rescale_params(graph, node, list(node.inputs[:len(node.inputs)])
                                      if node.op_type == "Concat"
                                      else [node.inputs[0]], out_name)
    rows = list(rescale["inputs"])
    while len(rows) < 4:
        rows.append({"m0": 0, "shift": 0, "zs_in": 0, "bypass": True})

    def pack16(values: List[int]) -> List[int]:
        return [((int(values[0]) & 0xFFFF) | ((int(values[1]) & 0xFFFF) << 16)),
                ((int(values[2]) & 0xFFFF) | ((int(values[3]) & 0xFFFF) << 16))]

    ch_pack = pack16(ch_offsets)
    c_pack = pack16(in_cs)
    ca_pack = pack16(in_cas)
    ps_pack = pack16(in_pss)
    out_zs = int(rescale["out_zs"])
    fill = out_zs
    payload = [
        in_addrs[0], in_addrs[1], in_addrs[2], in_addrs[3],
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        in_bytes_list[0], in_bytes_list[1], in_bytes_list[2], in_bytes_list[3],
        out_bytes,
        in_rows[0], in_rows[1], in_rows[2], in_rows[3], out_row,
        ((in_h & 0xFFFF) << 16) | (in_w & 0xFFFF),
        ((out_shape[2] & 0xFFFF) << 16) | (out_shape[3] & 0xFFFF),
        ((out_c & 0xFFFF) << 16) | (in_c & 0xFFFF),
        ((out_ca & 0xFFFF) << 16) | (in_ca & 0xFFFF),
        ((src_h & 0xFFFF) << 16) | (src_w & 0xFFFF),
        ((dst_h & 0xFFFF) << 16) | (src_c & 0xFFFF),
        ((dst_c & 0xFFFF) << 16) | (dst_w & 0xFFFF),
        ch_pack[0],
        ((len(in_names) & 0xFF) << 8) | (mode & 0xFF) | (1 << COPY_FORMAT_V2_BIT),
        ch_pack[1],
        c_pack[0], c_pack[1],
        ca_pack[0], ca_pack[1],
        ps_pack[0], ps_pack[1],
        ((out_zs & 0xFF) << 16) | ((fill & 0xFF) << 24) | (out_ca & 0xFFFF),
        int(rows[0]["m0"]) & 0xFFFFFFFF,
        int(rows[1]["m0"]) & 0xFFFFFFFF,
        int(rows[2]["m0"]) & 0xFFFFFFFF,
        int(rows[3]["m0"]) & 0xFFFFFFFF,
        ((int(rows[0]["shift"]) & 0xFF) |
         ((int(rows[0]["zs_in"]) & 0xFF) << 8) |
         ((0 if rows[0]["bypass"] else 1) << 16) |
         ((1 if rows[0]["bypass"] else 0) << 17)),
        ((int(rows[1]["shift"]) & 0xFF) |
         ((int(rows[1]["zs_in"]) & 0xFF) << 8) |
         ((0 if rows[1]["bypass"] else 1) << 16) |
         ((1 if rows[1]["bypass"] else 0) << 17)),
        ((int(rows[2]["shift"]) & 0xFF) |
         ((int(rows[2]["zs_in"]) & 0xFF) << 8) |
         ((0 if rows[2]["bypass"] else 1) << 16) |
         ((1 if rows[2]["bypass"] else 0) << 17)),
        ((int(rows[3]["shift"]) & 0xFF) |
         ((int(rows[3]["zs_in"]) & 0xFF) << 8) |
         ((0 if rows[3]["bypass"] else 1) << 16) |
         ((1 if rows[3]["bypass"] else 0) << 17)),
    ]
    if len(payload) != COPY_PAYLOAD_WORDS_V2:
        raise AssertionError(len(payload))
    return payload


def _common_activation_storage(graph: Graph, plan: Dict[str, Any], name: str) -> Dict[str, Any]:
    return _storage(plan, name)


def _clip_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    if len(node.inputs) not in (1, 2, 3):
        raise CompileError(E_CFG, "Clip payload requires data and optional bounds", node_id=node.id)
    x_name, out_name = node.inputs[0], node.outputs[0]
    q = graph.tensors[x_name].quantization
    qo = graph.tensors[out_name].quantization
    if not q or not qo or q.get("scale") != qo.get("scale") or             int(q.get("zero_point", 0)) != 0 or int(qo.get("zero_point", 0)) != 0:
        raise CompileError(E_QUANT_MISMATCH, "Clip requires equal zero-point-0 quant scales",
                           node_id=node.id)
    xb, ob = _binding(plan, x_name), _binding(plan, out_name)
    xs, os = _common_activation_storage(graph, plan, x_name), _common_activation_storage(graph, plan, out_name)
    x_shape, o_shape = _shape(graph, x_name), _shape(graph, out_name)
    if x_shape != o_shape:
        raise CompileError(E_CFG, "Clip input/output shapes differ", node_id=node.id)
    qmin, qmax = int(node.metadata["clip_min"]), int(node.metadata["clip_max"])
    payload = [
        _addr(xb["address"]) & 0xFFFFFFFF,
        _addr(ob["address"]) & 0xFFFFFFFF,
        int(xs["bytes"]) & 0xFFFFFFFF,
        int(os["bytes"]) & 0xFFFFFFFF,
        int(xs["row_stride_bytes"]) & 0xFFFFFFFF,
        int(os["row_stride_bytes"]) & 0xFFFFFFFF,
        (int(x_shape[2]) & 0xFFFF) | ((int(x_shape[3]) & 0xFFFF) << 16),
        (int(x_shape[1]) & 0xFFFF) | ((int(xs["channel_alloc"]) & 0xFFFF) << 16),
        (qmin & 0xFF) | ((qmax & 0xFF) << 8),
    ]
    if len(payload) != REDUCE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _gap_payload(graph: Graph, node, plan: Dict[str, Any]) -> List[int]:
    if len(node.inputs) != 1:
        raise CompileError(E_CFG, "GlobalAveragePool payload requires one input", node_id=node.id)
    x_name, out_name = node.inputs[0], node.outputs[0]
    q = graph.tensors[x_name].quantization
    qo = graph.tensors[out_name].quantization
    if not q or not qo or q.get("scale") != qo.get("scale") or             int(q.get("zero_point", 0)) != 0 or int(qo.get("zero_point", 0)) != 0:
        raise CompileError(E_QUANT_MISMATCH,
                           "GlobalAveragePool requires equal zero-point-0 quant scales",
                           node_id=node.id)
    xb, ob = _binding(plan, x_name), _binding(plan, out_name)
    xs, os = _common_activation_storage(graph, plan, x_name), _common_activation_storage(graph, plan, out_name)
    x_shape = _shape(graph, x_name)
    payload = [
        _addr(xb["address"]) & 0xFFFFFFFF,
        _addr(ob["address"]) & 0xFFFFFFFF,
        int(xs["bytes"]) & 0xFFFFFFFF,
        int(os["bytes"]) & 0xFFFFFFFF,
        int(xs["row_stride_bytes"]) & 0xFFFFFFFF,
        (int(x_shape[2]) & 0xFFFF) | ((int(x_shape[3]) & 0xFFFF) << 16),
        (int(x_shape[1]) & 0xFFFF) | ((int(xs["channel_alloc"]) & 0xFFFF) << 16),
        int(x_shape[2]) * int(x_shape[3]) & 0xFFFFFFFF,
        0,
    ]
    if len(payload) != REDUCE_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    return payload


def _lut_allocations(graph: Graph, plan: Dict[str, Any], alignment: int) -> Dict[str, Dict[str, Any]]:
    bindings = plan["memory_plan"]["tensor_bindings"]
    base = _addr(plan["memory_plan"]["arena"]["base"])
    cursor = int(plan["memory_plan"]["arena"]["used_bytes"])
    # Account for Conv quant rows already reserved by _side_quant_allocations
    # by advancing cursor past them (same computation as _side_quant_allocations).
    for node in graph.nodes:
        if node.op_type == "DepthwiseConv":
            c = int(_shape(graph, node.outputs[0])[1])
            size = int(depthwise_quant_storage((c,))["bytes"])
            offset = align_up(cursor, alignment)
            cursor = offset + align_up(size, alignment)
            continue
        if node.op_type != "Conv":
            continue
        layer_index = int(node.metadata.get("legacy_layer_index", node.order))
        names = ["layer%02d.quant" % layer_index, "conv%02d.quant" % node.order]
        existing = next((name for name in names if name in bindings and bindings[name]["address"]), None)
        if existing is not None:
            row = bindings[existing]
            end = int(row["offset"]) + int(row["allocated_bytes"])
            if end > cursor:
                cursor = end
            continue
        from .verilog_codegen import quant_storage as _qs
        cout = int(_shape(graph, node.outputs[0])[1])
        size = int(_qs((cout,))["bytes"])
        offset = align_up(cursor, alignment)
        cursor = offset + align_up(size, alignment)
    result: Dict[str, Dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op_type != "ActivationLUT":
            continue
        size = 256
        offset = align_up(cursor, 16)
        cursor = offset + align_up(size, 16)
        result[node.id] = {"name": "lut_%s.table" % node.id, "address": _hex(base + offset),
                           "offset": offset, "bytes": size, "allocated_bytes": align_up(size, 16)}
    # stash extended arena usage for DDR image sizing (caller merges)
    result["_cursor_end"] = {"offset": cursor, "bytes": 0, "allocated_bytes": 0, "address": _hex(base + cursor), "name": "_cursor"}
    return result


def _actlut_payload(graph: Graph, node, plan: Dict[str, Any], lut_row: Dict[str, Any]) -> List[int]:
    if node.op_type != "ActivationLUT":
        raise CompileError(E_CFG, "LUT payload requires ActivationLUT", node_id=node.id)
    mode = str((node.attrs or {}).get("mode", ""))
    if mode not in ("Sigmoid", "Tanh", "SiLU", "LeakyRelu"):
        raise CompileError(E_CFG, "LUT payload mode unsupported", node_id=node.id)
    in_name, out_name = node.inputs[0], node.outputs[0]
    in_shape = _shape(graph, in_name)
    out_shape = _shape(graph, out_name)
    if list(in_shape) != list(out_shape):
        raise CompileError(E_CFG, "LUT payload requires identical I/O shapes", node_id=node.id)
    n, c, h, w = (int(v) for v in in_shape)
    # v0.2 batch 2: a spatial build carries its resolved capacity in the plan and
    # may stream feature maps up to the allocated axes (<=640).  A legacy build
    # (no plan["spatial_capacity"]) keeps the frozen H,W<=32 payload gate.
    spatial_capacity = plan.get("spatial_capacity") or {}
    if spatial_capacity:
        allocated = spatial_capacity.get("allocated") or {}
        h_cap = int(allocated.get("height") or 640)
        w_cap = int(allocated.get("width") or 640)
        if n != 1 or not (1 <= c <= 16) or not (1 <= h <= h_cap) or not (1 <= w <= w_cap):
            raise CompileError(E_SPATIAL_LIMIT,
                               "LUT payload shape [N,C,H,W]=[%d,%d,%d,%d] outside the "
                               "allocated capacity C<=16 H<=%d W<=%d" %
                               (n, c, h, w, h_cap, w_cap), node_id=node.id,
                               detail={"n": n, "c": c, "h": h, "w": w,
                                       "h_cap": h_cap, "w_cap": w_cap})
    elif n != 1 or not (1 <= c <= 16) or not (1 <= h <= 32) or not (1 <= w <= 32):
        raise CompileError(E_CFG, "LUT payload shape outside batch range", node_id=node.id)
    for what, tn in (("input", in_name), ("output", out_name)):
        q = graph.tensors[tn].quantization
        if not q or q.get("materialized") is False or int(q.get("zero_point", 0)) != 0 or "scale" not in q:
            raise CompileError(E_QUANT_MISMATCH, "LUT %s requires explicit zp0 scale" % what, node_id=node.id)
    in_binding = _binding(plan, in_name)
    out_binding = _binding(plan, out_name)
    in_store = _storage(plan, in_name)
    out_store = _storage(plan, out_name)
    c_alloc = int(in_store.get("channel_alloc", ((c + 7) // 8) * 8))
    in_stride = int(in_store["row_stride_bytes"])
    out_stride = int(out_store["row_stride_bytes"])
    in_bytes = int(in_store["bytes"])
    out_bytes = int(out_store["bytes"])
    if in_bytes % 16 or out_bytes % 16:
        raise CompileError(E_CFG, "LUT packed bytes must be 16B aligned", node_id=node.id)
    payload = [
        _addr(in_binding["address"]) & 0xFFFFFFFF,
        0x00000000,
        0x00000000,
        _addr(lut_row["address"]) & 0xFFFFFFFF,
        _addr(out_binding["address"]) & 0xFFFFFFFF,
        (int(h) & 0xFFFF) | ((int(w) & 0xFFFF) << 16),
        (int(c) & 0xFFFF) | ((int(c_alloc) & 0xFFFF) << 16),
        0x00000000,
        0x00000000,
        0x00000000,
        in_stride & 0xFFFFFFFF,
        out_stride & 0xFFFFFFFF,
        in_bytes & 0xFFFFFFFF,
        0x00000000,
        0x00000000,
        256 & 0xFFFFFFFF,
        out_bytes & 0xFFFFFFFF,
    ]
    if len(payload) != ACTLUT_PAYLOAD_WORDS:
        raise AssertionError(len(payload))
    # reserved-word check (non-zero rejected at launch; codegen never emits non-zero)
    return payload


def generate_stream(graph: Graph, plan: Dict[str, Any], *, run_id: str = "e1_w3") -> Dict[str, Any]:
    graph.finalize()
    plan_graph_digest = plan.get("graph_digest")
    if plan_graph_digest is not None and plan_graph_digest != graph.digest():
        raise CompileError(E_CFG, "plan graph digest does not match graph")
    quant_rows = _side_quant_allocations(graph, plan, int(plan["memory_plan"]["arena"]["alignment"]))
    # affine-int8-v1 copy/transform records use the versioned v2 layout; the
    # legacy profile keeps the byte-identical 27-word record.
    affine_copy = (str((graph.metadata or {}).get("numeric_profile") or "")
                   == "affine-int8-v1")
    global _LUT_ROWS
    _LUT_ROWS = _lut_allocations(graph, plan, int(plan["memory_plan"]["arena"]["alignment"]))
    records: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.placement == "ps" or node.metadata_only:
            continue
        if node.op_type == "Conv":
            opcode = OPCODE_CONV
            payload = _conv_payload(graph, node, plan, quant_rows[node.id])
            payload_words = CONV_PAYLOAD_WORDS
        elif node.op_type == "MaxPool":
            if "legacy_layer_index" in node.metadata:
                opcode = OPCODE_MAXPOOL
                payload = _pool_payload(graph, node, plan)
                payload_words = CONV_PAYLOAD_WORDS
            elif node.metadata.get("maxpool_backend") == "pool5":
                opcode = OPCODE_MAXPOOL5
                payload = _pool5_payload(graph, node, plan)
                payload_words = CONV_PAYLOAD_WORDS
            else:
                opcode = OPCODE_MAXPOOL2D
                payload = _pool2d_payload(graph, node, plan)
                payload_words = 16
        elif node.op_type == "Relu":
            opcode = OPCODE_RELU
            payload = _elementwise_payload(graph, node, plan)
            payload_words = ELEMENTWISE_PAYLOAD_WORDS
        elif node.op_type == "Add":
            opcode = OPCODE_ADD
            if node.metadata.get("affine"):
                payload = _add_affine_payload(graph, node, plan)
                payload_words = ELEMENTWISE_PAYLOAD_WORDS_AFFINE
            else:
                payload = _elementwise_payload(graph, node, plan)
                payload_words = ELEMENTWISE_PAYLOAD_WORDS
        elif node.op_type == "Pad":
            opcode = OPCODE_PAD
            payload = (_copy_payload_v2(graph, node, plan) if affine_copy
                       else _copy_payload(graph, node, plan))
            payload_words = COPY_PAYLOAD_WORDS_V2 if affine_copy else COPY_PAYLOAD_WORDS
        elif node.op_type == "Slice":
            opcode = OPCODE_SLICE
            payload = (_copy_payload_v2(graph, node, plan) if affine_copy
                       else _copy_payload(graph, node, plan))
            payload_words = COPY_PAYLOAD_WORDS_V2 if affine_copy else COPY_PAYLOAD_WORDS
        elif node.op_type == "Concat":
            opcode = OPCODE_CONCAT
            payload = (_copy_payload_v2(graph, node, plan) if affine_copy
                       else _copy_payload(graph, node, plan))
            payload_words = COPY_PAYLOAD_WORDS_V2 if affine_copy else COPY_PAYLOAD_WORDS
        elif node.op_type == "Clip":
            opcode = OPCODE_CLIP
            payload = _clip_payload(graph, node, plan)
            payload_words = REDUCE_PAYLOAD_WORDS
        elif node.op_type == "GlobalAveragePool":
            opcode = OPCODE_GAP
            payload = _gap_payload(graph, node, plan)
            payload_words = REDUCE_PAYLOAD_WORDS
        elif node.op_type == "AveragePool":
            opcode = OPCODE_AVGPOOL
            payload = _avgpool_payload(graph, node, plan)
            payload_words = CONV_PAYLOAD_WORDS
        elif node.op_type == "Resize":
            opcode = OPCODE_RESIZE
            payload = _resize_payload(graph, node, plan)
            payload_words = RESIZE_PAYLOAD_WORDS
        elif node.op_type == "Mul":
            opcode = OPCODE_MUL
            payload = _mul_payload(graph, node, plan)
            payload_words = ELEMENTWISE_PAYLOAD_WORDS
        elif node.op_type == "DepthwiseConv":
            opcode = OPCODE_DEPTHWISE
            payload = _depthwise_payload(graph, node, plan, quant_rows[node.id])
            payload_words = DEPTHWISE_PAYLOAD_WORDS
        elif node.op_type == "ActivationLUT":
            opcode = OPCODE_ACTLUT
            _lut_rows = globals().get("_LUT_ROWS_CACHE", None)
            # _lut_rows is set below before the loop; fallback computes on demand
            try:
                _lr = _LUT_ROWS[node.id]
            except Exception:
                _lr = _lut_allocations(graph, plan, int(plan["memory_plan"]["arena"]["alignment"]))[node.id]
            payload = _actlut_payload(graph, node, plan, _lr)
            payload_words = ACTLUT_PAYLOAD_WORDS
        else:
            raise CompileError(E_UNSUPPORTED_OP,
                               "W3 codegen cannot lower %s (node %s)" % (node.op_type, node.id),
                               node_id=node.id)
        records.append({
            "index": len(records),
            "node_id": node.id,
            "op": node.op_type,
            "opcode": opcode,
            "header": ((opcode & 0xFFFF) << 16) | payload_words,
            "payload_words": payload_words,
            "payload": payload,
        })
    body: List[int] = []
    for record in records:
        body.append(record["header"])
        body.extend(record["payload"])
    word_count = len(body) + 1
    words = [0xFFFFFFFF, word_count] + body + [0xFFFFFFFF]
    return {
        "schema": STREAM_SCHEMA,
        "run_id": run_id,
        "graph_digest": graph.digest(),
        "framing": {"start_marker": "0xFFFFFFFF", "word_count": word_count,
                    "terminator": "0xFFFFFFFF"},
        "opcodes": {"0x0001": "Conv", "0x0002": "MaxPool",
                    "0x0010": "Relu", "0x0011": "Add",
                    "0x0012": "Pad", "0x0013": "Slice", "0x0014": "Concat",
                    "0x0015": "Clip", "0x0016": "GlobalAveragePool",
                    "0x0017": "AveragePool", "0x0018": "Mul",
                    "0x0019": "Resize", "0x001A": "MaxPool5",
                    "0x001B": "ActivationLUT", "0x001C": "DepthwiseConv",
                    "0x001D": "MaxPool2DGeneric"},
        "payload_fields": {
            "0x001D": ["version_flags", "in_addr", "out_addr", "hin_win", "c_calloc",
                       "hout_wout", "kh_kw", "sh_sw", "dh_dw", "pt_pl", "pb_pr",
                       "in_row_stride", "out_row_stride", "in_bytes", "out_bytes", "reserved"],
            "0x0001": ["in_addr", "weight_addr", "bias_addr", "quant_addr", "out_addr",
                        "h_w", "cin_cout", "conv_cfg", "quant_cfg", "act_cfg",
                        "in_row_stride", "out_row_stride", "in_bytes", "weight_bytes",
                        "bias_bytes", "quant_bytes", "out_bytes"],
            "0x0002": ["in_addr", "weight_addr", "bias_addr", "quant_addr", "out_addr",
                        "h_w", "cin_cout", "conv_cfg", "quant_cfg", "act_cfg",
                        "in_row_stride", "out_row_stride", "in_bytes", "weight_bytes",
                        "bias_bytes", "quant_bytes", "out_bytes"],
            "0x0010": ["in0_addr", "in1_addr", "out_addr", "in0_bytes", "in1_bytes",
                        "out_bytes", "flags"],
            "0x0011": ["in0_addr", "in1_addr", "out_addr", "in0_bytes", "in1_bytes",
                        "out_bytes", "flags"],
            "0x0012": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "channel_offsets", "mode_n_inputs",
                        "input_channels", "input_channel_allocs", "zero_point"],
            "0x0013": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "channel_offsets", "mode_n_inputs",
                        "input_channels", "input_channel_allocs", "zero_point"],
            "0x0015": ["in_addr", "out_addr", "in_bytes", "out_bytes",
                        "in_row_stride", "out_row_stride", "h_w", "c_calloc", "bounds"],
            "0x0016": ["in_addr", "out_addr", "in_bytes", "out_bytes",
                        "in_row_stride", "h_w", "c_calloc", "reduction_n", "reserved"],
            "0x0017": ["in_addr", "weight_addr", "bias_addr", "quant_addr", "out_addr",
                        "h_w", "cin_cout", "conv_cfg", "quant_cfg", "act_cfg",
                        "in_row_stride", "out_row_stride", "in_bytes", "weight_bytes",
                        "bias_bytes", "quant_bytes", "out_bytes"],
            "0x0018": ["in0_addr", "in1_addr", "out_addr", "in0_bytes", "in1_bytes",
                        "out_bytes", "m0_shift"],
            "0x0019": ["in_addr", "out_addr", "in_bytes", "out_bytes",
                        "in_row_stride", "out_row_stride", "h_w", "c_c",
                        "channel_alloc"],
            "0x001A": ["in_addr", "weight_addr", "bias_addr", "quant_addr", "out_addr",
                        "h_w", "cin_cout", "conv_cfg", "channel_alloc", "act_cfg",
                        "in_row_stride", "out_row_stride", "in_bytes", "weight_bytes",
                        "bias_bytes", "quant_bytes", "out_bytes"],
            "0x001C": ["in_addr", "weight_addr", "bias_addr", "quant_addr", "out_addr",
                        "h_w", "cin_cout", "conv_cfg", "rsv8", "rsv9",
                        "in_row_stride", "out_row_stride", "in_bytes", "weight_bytes",
                        "bias_bytes", "quant_bytes", "out_bytes"],
            "0x001B": ["in_addr", "rsv1", "rsv2", "lut_addr", "out_addr",
                        "h_w", "c_calloc", "rsv7", "rsv8", "rsv9",
                        "in_row_stride", "out_row_stride", "in_bytes", "rsv13", "rsv14",
                        "lut_bytes", "out_bytes"],
            "0x0014": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "channel_offsets", "mode_n_inputs",
                        "input_channels", "input_channel_allocs", "zero_point"],
        },
        "payload_fields_v2": {
            "0x0012": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "ch_off01", "mode_n_inputs_format", "ch_off23",
                        "in_c_i01", "in_c_i23", "in_ca_i01", "in_ca_i23",
                        "in_ps01", "in_ps23", "out_ps_zs_fill",
                        "m0_in0", "m0_in1", "m0_in2", "m0_in3",
                        "rs0", "rs1", "rs2", "rs3"],
            "0x0013": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "ch_off01", "mode_n_inputs_format", "ch_off23",
                        "in_c_i01", "in_c_i23", "in_ca_i01", "in_ca_i23",
                        "in_ps01", "in_ps23", "out_ps_zs_fill",
                        "m0_in0", "m0_in1", "m0_in2", "m0_in3",
                        "rs0", "rs1", "rs2", "rs3"],
            "0x0014": ["in0_addr", "in1_addr", "in2_addr", "in3_addr", "out_addr",
                        "in0_bytes", "in1_bytes", "in2_bytes", "in3_bytes", "out_bytes",
                        "in0_row_stride", "in1_row_stride", "in2_row_stride",
                        "in3_row_stride", "out_row_stride",
                        "in_h_w", "out_h_w", "in_c_out_c", "in_ca_out_ca",
                        "src_h_w", "src_c_dst_h", "dst_w_c",
                        "ch_off01", "mode_n_inputs_format", "ch_off23",
                        "in_c_i01", "in_c_i23", "in_ca_i01", "in_ca_i23",
                        "in_ps01", "in_ps23", "out_ps_zs_fill",
                        "m0_in0", "m0_in1", "m0_in2", "m0_in3",
                        "rs0", "rs1", "rs2", "rs3"],
        },
        "records": records,
        "quant_allocations": quant_rows,
        "word_count": len(words),
        "words": [_hex(word) for word in words],
    }


def write_stream(stream: Dict[str, Any], out_dir: Union[str, Path], name: str = "e1_graph") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    words = [int(value, 16) for value in stream["words"]]
    instr_words = 1
    while instr_words < len(words):
        instr_words <<= 1
    instr_words = max(instr_words, 256)
    mem_name = name + "_instr.mem"
    lines = ["%08x" % (word & 0xFFFFFFFF) for word in words]
    while len(lines) < instr_words:
        lines.append("ffffffff")
    mem_blob = ("\n".join(lines) + "\n").encode()
    (out / mem_name).write_bytes(mem_blob)
    stream["mem_file"] = mem_name
    stream["mem_sha256"] = hashlib.sha256(mem_blob).hexdigest()
    stream["instr_words"] = instr_words
    stream["instr_addr_w"] = max(1, (instr_words - 1).bit_length())
    (out / "instruction_stream.json").write_text(
        json.dumps(stream, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"instruction_stream": str(out / "instruction_stream.json"),
            "mem": str(out / mem_name), "mem_sha256": stream["mem_sha256"]}
