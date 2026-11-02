"""Legacy MNIST ops.json <-> canonical E1 graph IR adapter.

The adapter never calls npu_gen or mutates the frozen generator.  It converts the
existing ordered conv/pool chain to IR and can reconstruct the exact legacy
schema for an instruction-generation equivalence check.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_ir import (
    CompileError,
    E_CFG,
    E_UNSUPPORTED_OP,
    Graph,
    Node,
    Tensor,
    _json_copy,
)
from .op_registry import (
    legacy_conv_attributes,
    legacy_maxpool_attributes,
)
from .shape_infer import (
    bias_storage,
    infer_graph,
    nhwc8_a16_storage,
    quant_storage,
    weight_storage,
)

MAX_LAYERS = 8
ACT_DTYPE = "int8"
CONV_OPS = ("conv3x3_dense",)
POOL_OPS = ("maxpool2x2_s2",)
ALL_OPS = CONV_OPS + POOL_OPS

CONV_KNOWN_KEYS = {
    "op", "h", "w", "cin", "cout", "stride", "pad", "kernel",
    "groups", "n", "dilation",
}
POOL_KNOWN_KEYS = {
    "op", "h", "w", "cin", "cout", "kernel", "stride", "pad",
    "groups", "n", "dilation",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive_int(layer: Dict[str, Any], key: str, index: int) -> int:
    value = layer.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompileError(E_CFG, "legacy layer %d %s must be a positive integer" % (index, key))
    return int(value)


def _check_common(layer: Dict[str, Any], index: int) -> Tuple[str, int, int, int]:
    op = layer.get("op")
    if op not in ALL_OPS:
        raise CompileError(E_UNSUPPORTED_OP,
                           "legacy layer %d op %r is not Conv/MaxPool" % (index, op))
    return (str(op), _positive_int(layer, "h", index),
            _positive_int(layer, "w", index), _positive_int(layer, "cin", index))


def _activation_tensor(name: str, shape: Sequence[int]) -> Tensor:
    shape_t = tuple(int(v) for v in shape)
    return Tensor(
        name=name,
        dtype=ACT_DTYPE,
        shape=shape_t,
        kind="intermediate",
        layout="NCHW",
        storage=nhwc8_a16_storage(shape_t),
        logical_bytes=int(shape_t[0] * shape_t[1] * shape_t[2] * shape_t[3]),
        source={"kind": "legacy_activation_geometry"},
    )


def parse_legacy_document(document: Dict[str, Any], *,
                          source_sha256: Optional[str] = None,
                          name: Optional[str] = None) -> Graph:
    """Convert an ordered legacy ops.json document into finalized graph IR."""
    if not isinstance(document, dict):
        raise CompileError(E_CFG, "legacy ops document must be a JSON object")
    layers = document.get("layers")
    if not isinstance(layers, list) or not (1 <= len(layers) <= MAX_LAYERS):
        raise CompileError(E_CFG, "legacy layers must be a list with 1..%d entries" % MAX_LAYERS)
    if not any(layer.get("op") in CONV_OPS for layer in layers if isinstance(layer, dict)):
        raise CompileError(E_CFG, "legacy mixed chain must contain at least one Conv layer")
    if not all(isinstance(layer, dict) for layer in layers):
        raise CompileError(E_CFG, "every legacy layer must be a JSON object")
    build_profile_present = "build_profile" in document
    if build_profile_present and not isinstance(document.get("build_profile"), dict):
        raise CompileError(E_CFG, "legacy build_profile must be a JSON object")
    build_profile = dict(document.get("build_profile", {}))
    top_extra = {k: _json_copy(v) for k, v in document.items()
                 if k not in ("name", "layers", "build_profile")}

    graph_name = name or str(document.get("name", "legacy_graph"))
    op0, h0, w0, c0 = _check_common(layers[0], 0)
    if op0 not in CONV_OPS:
        raise CompileError(E_CFG, "legacy chain must start with conv3x3_dense")
    current_shape = (1, c0, h0, w0)
    current_name = "input"
    tensors: Dict[str, Tensor] = {
        current_name: _activation_tensor(current_name, current_shape),
    }
    tensors[current_name].kind = "input"
    nodes: List[Node] = []

    for index, layer in enumerate(layers):
        op, h, w, cin = _check_common(layer, index)
        if (h, w, cin) != (current_shape[2], current_shape[3], current_shape[1]):
            raise CompileError(
                E_CFG,
                "legacy layer %d input (%d,%d,c%d) does not match previous output %s" %
                (index, h, w, cin, current_shape),
            )
        if op in CONV_OPS:
            cout = _positive_int(layer, "cout", index)
            stride = int(layer.get("stride", 1))
            pad = int(layer.get("pad", 0))
            if stride not in (1, 2):
                raise CompileError(E_UNSUPPORTED_OP, "legacy layer %d stride must be 1/2" % index)
            if pad not in (0, 1):
                raise CompileError(E_UNSUPPORTED_OP, "legacy layer %d pad must be 0/1" % index)
            if int(layer.get("kernel", 3)) != 3 or int(layer.get("groups", 1)) != 1:
                raise CompileError(E_UNSUPPORTED_OP,
                                   "legacy layer %d must be dense 3x3 groups=1" % index)
            if int(layer.get("n", 1)) != 1 or int(layer.get("dilation", 1)) != 1:
                raise CompileError(E_UNSUPPORTED_OP,
                                   "legacy layer %d requires N=1 dilation=1" % index)
            hout = (h + 2 * pad - 3) // stride + 1
            wout = (w + 2 * pad - 3) // stride + 1
            out_shape = (1, cout, hout, wout)
            weight_name = "layer%02d.weight" % index
            bias_name = "layer%02d.bias" % index
            wshape = (cout, cin, 3, 3)
            bshape = (cout,)
            tensors[weight_name] = Tensor(
                weight_name, dtype=ACT_DTYPE, shape=wshape, kind="initializer",
                layout="OIHW", storage=weight_storage(wshape),
                source={"kind": "legacy_geometry", "blob": "not_supplied"},
            )
            tensors[bias_name] = Tensor(
                bias_name, dtype="int32", shape=bshape, kind="initializer",
                layout="C", storage=bias_storage(bshape),
                source={"kind": "legacy_geometry", "blob": "not_supplied"},
            )
            out_name = "layer%02d_conv_out" % index
            tensors[out_name] = _activation_tensor(out_name, out_shape)
            node_id = "n%04d" % index
            extra = {k: _json_copy(v) for k, v in layer.items() if k not in CONV_KNOWN_KEYS}
            nodes.append(Node(
                id=node_id, op_type="Conv", name="legacy_%s_%d" % (op, index),
                inputs=[current_name, weight_name, bias_name], outputs=[out_name],
                attrs=legacy_conv_attributes(stride, pad), placement="pl",
                lowering="npu_gen_v1_conv", source_index=index,
                metadata={
                    "legacy_layer_index": index,
                    "legacy_op": op,
                    "legacy_layer_extra": extra,
                    "opcode": 0x0001,
                    "weight_storage": _json_copy(tensors[weight_name].storage),
                    "bias_storage": _json_copy(tensors[bias_name].storage),
                    "quant_storage": quant_storage(bshape),
                },
            ))
        else:
            stride = int(layer.get("stride", 2))
            pad = int(layer.get("pad", 0))
            kernel = int(layer.get("kernel", 2))
            if (kernel, stride, pad) != (2, 2, 0):
                raise CompileError(E_UNSUPPORTED_OP,
                                   "legacy layer %d must be maxpool 2x2/s2/p0" % index)
            if h % 2 or w % 2:
                raise CompileError(E_CFG, "legacy layer %d pool requires even H/W" % index)
            if int(layer.get("cout", cin)) != cin:
                raise CompileError(E_CFG, "legacy layer %d pool cout must equal cin" % index)
            out_shape = (1, cin, h // 2, w // 2)
            out_name = "layer%02d_pool_out" % index
            tensors[out_name] = _activation_tensor(out_name, out_shape)
            extra = {k: _json_copy(v) for k, v in layer.items() if k not in POOL_KNOWN_KEYS}
            node_id = "n%04d" % index
            nodes.append(Node(
                id=node_id, op_type="MaxPool", name="legacy_%s_%d" % (op, index),
                inputs=[current_name], outputs=[out_name],
                attrs=legacy_maxpool_attributes(), placement="pl",
                lowering="npu_gen_v1_pool", source_index=index,
                metadata={
                    "legacy_layer_index": index,
                    "legacy_op": op,
                    "legacy_layer_extra": extra,
                    "opcode": 0x0002,
                },
            ))
        current_name = out_name
        current_shape = out_shape

    graph = Graph(
        name=graph_name,
        inputs=["input"],
        outputs=[current_name],
        initializers=[name for name, tensor in tensors.items() if tensor.kind == "initializer"],
        tensors=tensors,
        nodes=nodes,
        opset_imports=[],
        layout_policy={
            "logical_layout": "NCHW",
            "target_layout": "NHWC8_A16",
            "reshape_policy": "metadata_only_if_layout_compatible",
        },
        source={
            "format": "legacy_ops_json",
            "sha256": source_sha256 or hashlib.sha256(
                json.dumps(document, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
            "legacy_name": str(document.get("name", graph_name)),
        },
        metadata={
            "legacy_adapter": "e1-legacy-0.1",
            "legacy_build_profile": build_profile,
            "legacy_build_profile_present": build_profile_present,
            "legacy_top_level_extra": top_extra,
            "shape_inference": "pass",
            "topological_sort": "pass",
            "layout": "NHWC8_A16_geometry_only",
            "memory_planning": "not_run",
            "codegen": "not_run",
            "rtl": "not_run",
            "board": "not_run",
        },
    )
    infer_graph(graph)
    for tensor in graph.tensors.values():
        if tensor.kind != "initializer" and tensor.dtype == ACT_DTYPE and len(tensor.shape) == 4:
            tensor.storage = nhwc8_a16_storage(tensor.shape)
            tensor.layout = "NCHW"
    output = graph.tensors[current_name]
    output.kind = "output"
    return graph


def load_legacy_ops(path: Path, *, name: Optional[str] = None) -> Graph:
    path = Path(path)
    data = path.read_bytes()
    document = json.loads(data.decode("utf-8-sig"))
    return parse_legacy_document(document, source_sha256=_sha256(data), name=name)


def _chain_inputs(graph: Graph) -> List[Tensor]:
    if len(graph.inputs) != 1:
        raise CompileError(E_UNSUPPORTED_OP,
                           "legacy downgrade requires exactly one graph input")
    return [graph.tensors[graph.inputs[0]]]


def to_legacy_ops(graph: Graph) -> Dict[str, Any]:
    """Downgrade a pure Conv/MaxPool linear IR back to legacy ops.json.

    Any IR that contains Relu, Add, metadata nodes or a PS boundary is rejected;
    those graphs require the future E1 planner/codegen path rather than the old
    sequential npu_gen backend.
    """
    if not graph.nodes:
        raise CompileError(E_CFG, "cannot downgrade an empty graph")
    layers: List[Dict[str, Any]] = []
    expected_input = graph.inputs[0]
    for index, node in enumerate(graph.nodes):
        if node.op_type not in ("Conv", "MaxPool"):
            raise CompileError(E_UNSUPPORTED_OP,
                               "legacy backend cannot lower %s nodes (node %s)" %
                               (node.op_type, node.id), node_id=node.id)
        if node.inputs[0] != expected_input:
            raise CompileError(E_CFG,
                               "legacy downgrade requires a linear activation chain; "
                               "node %s consumes %s, expected %s" %
                               (node.id, node.inputs[0], expected_input), node_id=node.id)
        x = graph.tensors[node.inputs[0]]
        if x.shape is None or len(x.shape) != 4:
            raise CompileError(E_CFG, "legacy node %s input is not rank-4" % node.id)
        _, cin, h, w = x.shape
        extra = dict(node.metadata.get("legacy_layer_extra", {}))
        if node.op_type == "Conv":
            weight = graph.tensors[node.inputs[1]]
            if weight.shape is None or len(weight.shape) != 4:
                raise CompileError(E_CFG, "legacy Conv weight %s is not rank-4" % weight.name,
                                   node_id=node.id)
            cout = int(weight.shape[0])
            stride = int(node.attrs["strides"][0])
            pad = int(node.attrs["pads"][0])
            row = {
                "op": "conv3x3_dense",
                "h": int(h), "w": int(w), "cin": int(cin), "cout": cout,
                "stride": stride, "pad": pad, "kernel": 3, "groups": 1,
                "n": 1, "dilation": 1,
            }
        else:
            stride = int(node.attrs["strides"][0])
            pad = int(node.attrs["pads"][0])
            kernel = int(node.attrs["kernel_shape"][0])
            row = {
                "op": "maxpool2x2_s2",
                "h": int(h), "w": int(w), "cin": int(cin), "cout": int(cin),
                "kernel": kernel, "stride": stride, "pad": pad,
                "groups": 1, "n": 1, "dilation": 1,
            }
        canonical = dict(extra)
        canonical.update(row)
        layers.append(canonical)
        expected_input = node.outputs[0]
    if expected_input not in graph.outputs:
        raise CompileError(E_CFG, "legacy downgrade must end at a graph output")

    output: Dict[str, Any] = {}
    for key, value in sorted(graph.metadata.get("legacy_top_level_extra", {}).items()):
        output[key] = _json_copy(value)
    output["name"] = graph.name
    output["layers"] = layers
    if graph.metadata.get("legacy_build_profile_present", True):
        output["build_profile"] = _json_copy(graph.metadata.get("legacy_build_profile", {}))
    return output


def roundtrip_document(document: Dict[str, Any]) -> Dict[str, Any]:
    return to_legacy_ops(parse_legacy_document(document))
