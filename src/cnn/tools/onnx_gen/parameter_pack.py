"""Real ONNX initializer loading and ABI byte packing for E1-P2."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any, Dict, Sequence

import onnx
from onnx import numpy_helper

from .graph_ir import CompileError, E_CFG, E_QUANT_MISMATCH, Graph, align_up
from .layout import (NATIVE_K1_WEIGHT_FORMAT, conv_weight_format,
                     dense_weight_storage)
from .shape_infer import (ABI_ALIGNMENT, CHANNEL_BLOCK, depthwise_bias_storage,
                          depthwise_quant_storage, depthwise_weight_storage,
                          nhwc8_a16_storage, weight_storage)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_initializers(model_path: Path) -> Dict[str, Dict[str, Any]]:
    model = onnx.load(str(model_path))
    result: Dict[str, Dict[str, Any]] = {}
    for tensor in model.graph.initializer:
        array = numpy_helper.to_array(tensor)
        values = array.reshape(-1).tolist()
        result[str(tensor.name)] = {
            "dtype": str(array.dtype),
            "shape": [int(v) for v in array.shape],
            "values": [int(v) if str(array.dtype).startswith("int") or
                       str(array.dtype).startswith("uint") else float(v) for v in values],
            "sha256": sha256_bytes(tensor.SerializeToString()),
        }
    return result


def pack_activation(shape: Sequence[int], values: Sequence[int], *, dtype: str = "int8") -> bytes:
    if dtype != "int8":
        raise CompileError(E_CFG, "pack_activation supports INT8 only")
    n, c, h, w = (int(v) for v in shape)
    if n != 1:
        raise CompileError(E_CFG, "pack_activation requires N=1")
    if len(values) != n * c * h * w:
        raise CompileError(E_CFG, "pack_activation value count %d != shape %s" %
                           (len(values), tuple(shape)))
    geometry = nhwc8_a16_storage(shape)
    out = bytearray(int(geometry["bytes"]))
    ca, row = int(geometry["channel_alloc"]), int(geometry["row_stride_bytes"])
    for y in range(h):
        for x in range(w):
            for ch in range(c):
                value = int(values[(ch * h + y) * w + x])
                if not (-128 <= value <= 255):
                    raise CompileError(E_CFG,
                                       "activation value %d outside INT8 byte domain" % value)
                out[y * row + x * ca + ch] = value & 0xFF
    return bytes(out)


def _pack_conv_weights_k1(cout: int, cin: int, src_shape, values) -> bytes:
    """Native K1: one 64-byte center-tap group per (ocb, icb) block.

    The lowered 1x1 Conv keeps the frozen nine-tap *geometry* (pad/stride are
    unchanged and the center tap samples exactly the 1x1 input), but the
    hardware weight buffer only holds the center-tap group.  The block index
    is therefore [ocb][icb] with four 128-bit words (64 B) of
    [ici*8+oci] int8 lanes, i.e. weight bytes = icb*ocb*64.
    """
    storage = dense_weight_storage((cout, cin, 3, 3), NATIVE_K1_WEIGHT_FORMAT)
    blob = bytearray(int(storage["bytes"]))
    cout_alloc, cin_alloc = int(storage["cout_alloc"]), int(storage["cin_alloc"])
    ic_blocks = cin_alloc // CHANNEL_BLOCK
    for oc in range(cout_alloc):
        for ic in range(cin_alloc):
            value = 0
            if oc < cout and ic < cin:
                if tuple(src_shape) == (cout, cin, 1, 1):
                    value = int(values[oc * cin + ic])
                else:
                    value = int(values[((oc * cin + ic) * 3 + 1) * 3 + 1])
            if not (-128 <= value <= 127):
                raise CompileError(E_CFG, "Conv weight outside INT8")
            ocb, oci = divmod(oc, CHANNEL_BLOCK)
            icb, ici = divmod(ic, CHANNEL_BLOCK)
            offset = ((ocb * ic_blocks + icb) * 8 + ici) * 8 + oci
            blob[offset] = value & 0xFF
    return bytes(blob)


def pack_conv_weights(node, graph: Graph, initializers: Dict[str, Dict[str, Any]]) -> bytes:
    weight_name = node.inputs[1]
    source = initializers.get(weight_name)
    if source is None:
        raise CompileError(E_CFG, "Conv weight initializer %s is missing" % weight_name,
                           node_id=node.id)
    out_shape = graph.tensors[weight_name].shape
    if out_shape is None or len(out_shape) != 4:
        raise CompileError(E_CFG, "lowered Conv weight must be rank 4", node_id=node.id)
    cout, cin, kh, kw = (int(v) for v in out_shape)
    if source["dtype"] != "int8":
        raise CompileError(E_CFG, "Conv weight %s must be INT8, got %s" %
                           (weight_name, source["dtype"]), node_id=node.id)
    src_shape = tuple(int(v) for v in source["shape"])
    values = [int(v) for v in source["values"]]
    if conv_weight_format(node) == NATIVE_K1_WEIGHT_FORMAT:
        # affine-int8-v1 native K1: store the center tap only.  The lowered
        # tensor is 3x3/center-tap but the physical buffer is 1/9 of the
        # legacy nine-tap blob, and the core's conv_cfg bit 11 tells it so.
        if src_shape not in ((cout, cin, 1, 1), (cout, cin, 3, 3)):
            raise CompileError(E_CFG, "Conv weight source shape %s incompatible with "
                               "native K1 lowering" % (src_shape,), node_id=node.id)
        return _pack_conv_weights_k1(cout, cin, src_shape, values)
    if src_shape == (cout, cin, 3, 3):
        if kh != 3 or kw != 3:
            raise CompileError(E_CFG, "Conv weight shape mismatch", node_id=node.id)
        source_values = values
    elif src_shape == (cout, cin, 1, 1) and (kh, kw) == (3, 3):
        source_values = [0] * (cout * cin * 9)
        for oc in range(cout):
            for ic in range(cin):
                source_values[((oc * cin + ic) * 3 + 1) * 3 + 1] = values[(oc * cin + ic)]
    else:
        raise CompileError(E_CFG, "Conv weight source shape %s incompatible with lowered %s" %
                           (src_shape, out_shape), node_id=node.id)
    storage = weight_storage(out_shape)
    blob = bytearray(int(storage["bytes"]))
    cout_alloc, cin_alloc = int(storage["cout_alloc"]), int(storage["cin_alloc"])
    ic_blocks = cin_alloc // CHANNEL_BLOCK
    for oc in range(cout_alloc):
        for ky in range(3):
            for kx in range(3):
                for ic in range(cin_alloc):
                    value = 0
                    if oc < cout and ic < cin:
                        value = int(source_values[((oc * cin + ic) * 3 + ky) * 3 + kx])
                    if not (-128 <= value <= 127):
                        raise CompileError(E_CFG, "Conv weight outside INT8", node_id=node.id)
                    ocb, oci = divmod(oc, CHANNEL_BLOCK)
                    icb, ici = divmod(ic, CHANNEL_BLOCK)
                    offset = ((((ocb * 3 + ky) * 3 + kx) * ic_blocks + icb) * 8 + ici) * 8 + oci
                    blob[offset] = value & 0xFF
    return bytes(blob)


def pack_conv_bias(node, graph: Graph, initializers: Dict[str, Dict[str, Any]]) -> bytes:
    if len(node.inputs) < 3:
        raise CompileError(E_CFG, "P2 Conv hardware requires an INT32 bias", node_id=node.id)
    bias_name = node.inputs[2]
    source = initializers.get(bias_name)
    if source is None:
        raise CompileError(E_CFG, "Conv bias initializer %s is missing" % bias_name,
                           node_id=node.id)
    if source["dtype"] != "int32":
        raise CompileError(E_CFG, "Conv bias %s must be INT32, got %s" %
                           (bias_name, source["dtype"]), node_id=node.id)
    values = [int(v) for v in source["values"]]
    folded = node.metadata.get("folded_bias_values")
    if folded is not None:
        # E1-P3: the BN additive term was folded into the INT32 accumulator
        # bias exactly (see lowering._fold_batch_norm).
        values = [int(v) for v in folded]
    cout = int(graph.tensors[node.outputs[0]].shape[1])
    if len(values) != cout:
        raise CompileError(E_CFG, "Conv bias length %d != Cout %d" % (len(values), cout),
                           node_id=node.id)
    cout_alloc = align_up(cout, CHANNEL_BLOCK)
    blob = bytearray(align_up(cout_alloc * 4, ABI_ALIGNMENT))
    for oc, value in enumerate(values):
        struct.pack_into("<i", blob, oc * 4, int(value))
    return bytes(blob)


def pack_conv_quant(node, graph: Graph) -> bytes:
    records = node.metadata.get("requant_records")
    if not isinstance(records, list) or not records:
        raise CompileError(E_QUANT_MISMATCH, "Conv requant records are missing", node_id=node.id)
    cout = int(graph.tensors[node.outputs[0]].shape[1])
    if len(records) != cout:
        raise CompileError(E_QUANT_MISMATCH, "Conv requant record count mismatch", node_id=node.id)
    cout_alloc = align_up(cout, CHANNEL_BLOCK)
    blob = bytearray(align_up(cout_alloc * 8, ABI_ALIGNMENT))
    for record in records:
        oc = int(record["oc"])
        struct.pack_into("<IBbh", blob, oc * 8, int(record["M0"]), int(record["SHIFT"]),
                         int(record["out_zero_point"]), 0)
    return bytes(blob)


def _depthwise_channel_count(node, graph: Graph) -> int:
    shape = graph.tensors[node.outputs[0]].shape
    if shape is None or len(shape) != 4:
        raise CompileError(E_CFG, "DepthwiseConv output must be rank 4", node_id=node.id)
    c = int(shape[1])
    if int(graph.tensors[node.inputs[0]].shape[1]) != c:
        raise CompileError(E_CFG, "DepthwiseConv Cin != Cout", node_id=node.id)
    return c


def pack_depthwise_weights(node, graph: Graph,
                           initializers: Dict[str, Dict[str, Any]]) -> bytes:
    """C blocks of 16B: 9 row-major int8 taps followed by 7 zero bytes."""
    weight_name = node.inputs[1]
    source = initializers.get(weight_name)
    if source is None:
        raise CompileError(E_CFG, "depthwise weight initializer %s is missing" % weight_name,
                           node_id=node.id)
    shape = graph.tensors[weight_name].shape
    if shape is None or tuple(int(v) for v in shape) != (int(shape[0]), 1, 3, 3):
        raise CompileError(E_CFG, "depthwise weight must be [C,1,3,3]", node_id=node.id)
    c = _depthwise_channel_count(node, graph)
    if tuple(int(v) for v in shape)[0] != c:
        raise CompileError(E_CFG, "depthwise weight channel count mismatch", node_id=node.id)
    if source["dtype"] != "int8":
        raise CompileError(E_CFG, "depthwise weight %s must be INT8, got %s" %
                           (weight_name, source["dtype"]), node_id=node.id)
    src_shape = tuple(int(v) for v in source["shape"])
    if src_shape != (c, 1, 3, 3):
        raise CompileError(E_CFG,
                           "depthwise weight source shape %s != [%d,1,3,3]" %
                           (src_shape, c), node_id=node.id)
    values = [int(v) for v in source["values"]]
    if len(values) != c * 9:
        raise CompileError(E_CFG, "depthwise weight value count mismatch", node_id=node.id)
    storage = depthwise_weight_storage((c, 1, 3, 3))
    blob = bytearray(int(storage["bytes"]))
    for channel in range(c):
        for tap in range(9):
            value = values[channel * 9 + tap]
            if not (-128 <= value <= 127):
                raise CompileError(E_CFG, "depthwise weight outside INT8", node_id=node.id)
            blob[channel * 16 + tap] = value & 0xFF
    return bytes(blob)


def pack_depthwise_bias(node, graph: Graph,
                        initializers: Dict[str, Dict[str, Any]]) -> bytes:
    """align16(4*C) bytes: C int32 LE bias values, then zero padding."""
    if len(node.inputs) < 3:
        raise CompileError(E_CFG, "DepthwiseConv requires an INT32 bias", node_id=node.id)
    bias_name = node.inputs[2]
    source = initializers.get(bias_name)
    if source is None:
        raise CompileError(E_CFG, "depthwise bias initializer %s is missing" % bias_name,
                           node_id=node.id)
    if source["dtype"] != "int32":
        raise CompileError(E_CFG, "depthwise bias %s must be INT32, got %s" %
                           (bias_name, source["dtype"]), node_id=node.id)
    c = _depthwise_channel_count(node, graph)
    values = [int(v) for v in source["values"]]
    folded = node.metadata.get("folded_bias_values")
    if folded is not None:
        values = [int(v) for v in folded]
    if len(values) != c:
        raise CompileError(E_CFG, "depthwise bias length %d != C %d" %
                           (len(values), c), node_id=node.id)
    storage = depthwise_bias_storage((c,))
    blob = bytearray(int(storage["bytes"]))
    for channel, value in enumerate(values):
        if not (-(2 ** 31) <= int(value) <= 2 ** 31 - 1):
            raise CompileError(E_CFG, "depthwise bias outside INT32", node_id=node.id)
        struct.pack_into("<i", blob, channel * 4, int(value))
    return bytes(blob)


def pack_depthwise_quant(node, graph: Graph) -> bytes:
    """align16(8*C) bytes: C <M0 u32,SHIFT u8,zp i8,reserved u16> records."""
    records = node.metadata.get("requant_records")
    if not isinstance(records, list) or not records:
        raise CompileError(E_QUANT_MISMATCH, "depthwise requant records are missing",
                           node_id=node.id)
    c = _depthwise_channel_count(node, graph)
    if len(records) != c:
        raise CompileError(E_QUANT_MISMATCH,
                           "depthwise requant record count %d != C %d" %
                           (len(records), c), node_id=node.id)
    storage = depthwise_quant_storage((c,))
    blob = bytearray(int(storage["bytes"]))
    for record in records:
        channel = int(record["oc"])
        m0, shift = int(record["M0"]), int(record["SHIFT"])
        if not (0 <= channel < c):
            raise CompileError(E_QUANT_MISMATCH, "depthwise requant oc out of range",
                               node_id=node.id)
        if not (1 <= m0 <= 0xFFFFFFFF):
            raise CompileError(E_QUANT_MISMATCH, "depthwise M0 outside u32",
                               node_id=node.id)
        if not (0 <= shift <= 31):
            raise CompileError(E_QUANT_MISMATCH, "depthwise SHIFT outside 0..31",
                               node_id=node.id)
        if int(record.get("out_zero_point", 0)) != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "depthwise output zero_point must be 0", node_id=node.id)
        struct.pack_into("<IBbh", blob, channel * 8, m0, shift, 0, 0)
    return bytes(blob)


def load_input_manifest(manifest_path: Path, graph: Graph) -> Dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    if doc.get("schema") != "e1-input-manifest-1":
        raise CompileError(E_CFG, "unsupported input manifest schema")
    if doc.get("model_sha256") not in (None, (graph.source or {}).get("model_sha256")):
        raise CompileError(E_CFG, "input manifest model_sha256 does not match model")
    rows = doc.get("inputs")
    if not isinstance(rows, dict):
        raise CompileError(E_CFG, "input manifest inputs must be an object")
    result = {}
    for name in graph.inputs:
        row = rows.get(name)
        if not isinstance(row, dict):
            raise CompileError(E_CFG, "input manifest has no entry for %s" % name)
        path = (manifest_path.parent / str(row["path"])).resolve()
        if not path.is_file():
            raise CompileError(E_CFG, "input file missing for %s: %s" % (name, path))
        data = path.read_bytes()
        shape = tuple(int(v) for v in row.get("shape", graph.tensors[name].shape))
        if shape != tuple(graph.tensors[name].shape):
            raise CompileError(E_CFG, "input %s shape %s != graph %s" %
                               (name, shape, graph.tensors[name].shape))
        if row.get("format") != "nchw_int8":
            raise CompileError(E_CFG, "input %s format must be nchw_int8" % name)
        expected = 1
        for dim in shape:
            expected *= int(dim)
        if len(data) != expected:
            raise CompileError(E_CFG, "input %s length %d != %d" % (name, len(data), expected))
        actual_sha = sha256_bytes(data)
        if row.get("sha256") is not None and str(row["sha256"]).lower() != actual_sha:
            raise CompileError(E_CFG, "input %s SHA256 mismatch" % name)
        result[name] = {
            "path": str(path),
            "relative_path": str(path.relative_to(manifest_path.parent)).replace("\\", "/"),
            "shape": shape,
            "format": "nchw_int8",
            "sha256": actual_sha,
            "bytes": len(data),
            "values": list(data),
        }
    return {"schema": "e1-input-manifest-1", "path": str(manifest_path), "inputs": result}
