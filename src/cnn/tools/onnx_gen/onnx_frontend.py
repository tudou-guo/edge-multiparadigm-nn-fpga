"""Static ONNX -> canonical E1 graph IR frontend.

Supported default-domain opset 11..19 and operators: Conv, MaxPool, Relu, Add,
Pad, Slice, Concat, Reshape, Flatten and PS-only Gemm.  Shapes must be fully
static.  Unsupported operators, attributes, domains and dynamic dimensions are
rejected before any lowering or code generation.

Pad pads / constant_value and Slice starts / ends / axes / steps are compile-time
initializers.  They are recorded as control inputs and are never runtime data.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_ir import (
    CompileError,
    E_CFG,
    E_DYNAMIC_SHAPE,
    E_GRAPH,
    E_SHAPE_MISMATCH,
    E_UNSUPPORTED_ATTR,
    E_UNSUPPORTED_OP,
    E_UNSUPPORTED_OPSET,
    E_UNKNOWN_RANK,
    Graph,
    Node,
    Tensor,
)
from .op_registry import (
    MAX_OPSET,
    MIN_OPSET,
    control_input_indices,
    get_spec,
    normalize_attributes,
    normalize_domain,
)
from .shape_infer import infer_graph, layout_for_rank

try:  # Keep graph_ir/legacy adapter usable in environments without ONNX.
    import onnx
    from onnx import AttributeProto, TensorProto
except ImportError:  # pragma: no cover - exercised only by dependency failure
    onnx = None
    AttributeProto = None
    TensorProto = None


ACTIVATION_DTYPES = {
    "float32", "float16", "int8", "uint8",
}
DTYPE_NAMES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    9: "bool",
    10: "float16",
    11: "float64",
}


def _require_onnx() -> None:
    if onnx is None:
        raise CompileError(E_CFG,
                           "Python package onnx is required for ONNX parsing; "
                           "use E:\\miniconda\\python.exe")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_dtype(tensor_type: Any) -> str:
    elem_type = int(tensor_type.elem_type)
    dtype = DTYPE_NAMES.get(elem_type)
    if dtype is None:
        raise CompileError(E_CFG, "unsupported ONNX elem_type %d" % elem_type)
    return dtype


def _static_shape(value_info: Any, *, require_shape: bool, what: str) -> Optional[Tuple[int, ...]]:
    if not value_info.type.HasField("tensor_type"):
        raise CompileError(E_CFG, "%s is not a tensor" % what)
    shape_proto = value_info.type.tensor_type.shape
    dims = list(shape_proto.dim)
    if not dims:
        if require_shape:
            raise CompileError(E_UNKNOWN_RANK, "%s has no static shape" % what)
        return None
    shape: List[int] = []
    for index, dim in enumerate(dims):
        if not dim.HasField("dim_value"):
            raise CompileError(E_DYNAMIC_SHAPE,
                               "%s dimension %d is dynamic (%r)" %
                               (what, index, dim.dim_param or "unknown"))
        value = int(dim.dim_value)
        if value <= 0:
            raise CompileError(E_DYNAMIC_SHAPE,
                               "%s dimension %d is non-positive (%d)" % (what, index, value))
        shape.append(value)
    return tuple(shape)


def _value_info(value_info: Any, *, require_shape: bool, what: str) -> Dict[str, Any]:
    if not value_info.type.HasField("tensor_type"):
        raise CompileError(E_CFG, "%s is not a tensor" % what)
    dtype = _tensor_dtype(value_info.type.tensor_type)
    shape = _static_shape(value_info, require_shape=require_shape, what=what)
    return {"name": value_info.name, "dtype": dtype, "shape": shape}


def _attr_value(attribute: Any) -> Any:
    attr_type = int(attribute.type)
    if attr_type == int(AttributeProto.INT):
        return int(attribute.i)
    if attr_type == int(AttributeProto.FLOAT):
        return float(attribute.f)
    if attr_type == int(AttributeProto.STRING):
        return bytes(attribute.s).decode("utf-8")
    if attr_type == int(AttributeProto.INTS):
        return [int(v) for v in attribute.ints]
    if attr_type == int(AttributeProto.FLOATS):
        return [float(v) for v in attribute.floats]
    if attr_type == int(AttributeProto.STRINGS):
        return [bytes(v).decode("utf-8") for v in attribute.strings]
    if attr_type == int(AttributeProto.TENSOR):
        tensor = attribute.t
        dtype = DTYPE_NAMES.get(int(tensor.data_type))
        if dtype is None:
            raise CompileError(E_CFG,
                               "attribute %s tensor dtype is unsupported" % attribute.name,
                               attribute=attribute.name)
        values = _numeric_vector(tensor)
        if values is None:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "attribute %s tensor data is not a compact dense numeric value" %
                               attribute.name, attribute=attribute.name)
        return {
            "__tensor__": True,
            "dtype": dtype,
            "shape": [int(v) for v in tensor.dims],
            "values": values,
            "sha256": _initializer_bytes_hash(tensor),
        }
    raise CompileError(E_UNSUPPORTED_ATTR,
                       "attribute %s has unsupported ONNX type %d" %
                       (attribute.name, attr_type), attribute=attribute.name)


def _node_attributes(node_proto: Any) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for attribute in node_proto.attribute:
        name = str(attribute.name)
        if name in attrs:
            raise CompileError(E_UNSUPPORTED_ATTR, "duplicate attribute %s" % name,
                               attribute=name)
        attrs[name] = _attr_value(attribute)
    return attrs


def _trim_optional_names(names: Sequence[str], what: str) -> List[str]:
    values = [str(name) for name in names]
    while values and values[-1] == "":
        values.pop()
    if any(name == "" for name in values):
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "%s contains an empty optional name before the last input" % what)
    return values


_RESIZE_SLOT_KEYS = ("roi", "scales", "sizes")


def _resize_input_names(names: Sequence[str]) -> Tuple[List[str], Dict[str, Optional[str]]]:
    """Resolve ONNX Resize positional inputs X, roi, scales, sizes.

    Resize is the only operator in the supported subset whose optional inputs
    may be omitted from the *middle* of the positional list (a missing roi is
    normally spelled as "").  It therefore bypasses the generic trailing-only
    truncation rule and records which slots are present as node metadata.
    """
    raw = [str(name) for name in names]
    while raw and raw[-1] == "":
        raw.pop()
    if not raw:
        raise CompileError(E_UNSUPPORTED_ATTR, "Resize requires a data input")
    if len(raw) > 4:
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "Resize has at most four positional inputs (X, roi, scales, sizes)")
    data = raw[0]
    if not data:
        raise CompileError(E_UNSUPPORTED_ATTR, "Resize data input name is empty")
    slots: Dict[str, Optional[str]] = {key: None for key in _RESIZE_SLOT_KEYS}
    inputs = [data]
    for key, name in zip(_RESIZE_SLOT_KEYS, raw[1:]):
        if name == "":
            slots[key] = None
            continue
        slots[key] = name
        inputs.append(name)
    return inputs, slots


def _initializer_bytes_hash(tensor: Any) -> str:
    clone = TensorProto()
    clone.CopyFrom(tensor)
    clone.name = ""
    clone.doc_string = ""
    return _sha256(clone.SerializeToString(deterministic=True))


def _int_vector(tensor: Any) -> Optional[List[int]]:
    if int(tensor.data_type) not in (int(TensorProto.INT64), int(TensorProto.INT32)):
        return None
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    if int(tensor.data_type) == int(TensorProto.INT64):
        if tensor.raw_data:
            if len(tensor.raw_data) != count * 8:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return list(struct.unpack("<%dq" % count, tensor.raw_data))
        values = list(tensor.int64_data)
    else:
        if tensor.raw_data:
            if len(tensor.raw_data) != count * 4:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return list(struct.unpack("<%di" % count, tensor.raw_data))
        values = list(tensor.int32_data)
    if len(values) != count:
        raise CompileError(E_CFG, "initializer %s integer data length mismatch" % tensor.name)
    return [int(v) for v in values]


def _numeric_vector(tensor: Any) -> Optional[List[float]]:
    """Return small numeric initializer values for compile-time metadata.

    Only compact control tensors (<= 64 elements) are materialized.  Large
    weights are deliberately represented by shape/hash only so the canonical IR
    never embeds model blobs.
    """
    count = 1
    for dim in tensor.dims:
        count *= int(dim)
    if count > 64:
        return None
    data_type = int(tensor.data_type)
    raw = bytes(tensor.raw_data)
    if data_type == int(TensorProto.FLOAT):
        if raw:
            if len(raw) != count * 4:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return [float(v) for v in struct.unpack("<%df" % count, raw)]
        values = list(tensor.float_data)
    elif data_type == int(TensorProto.FLOAT16):
        if raw:
            if len(raw) != count * 2:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return [float(v) for v in struct.unpack("<%de" % count, raw)]
        return None
    elif data_type == int(TensorProto.DOUBLE):
        if raw:
            if len(raw) != count * 8:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return [float(v) for v in struct.unpack("<%dd" % count, raw)]
        values = list(tensor.double_data)
    elif data_type == int(TensorProto.INT64):
        if raw:
            if len(raw) != count * 8:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            return [int(v) for v in struct.unpack("<%dq" % count, raw)]
        values = list(tensor.int64_data)
    elif data_type in (int(TensorProto.INT32), int(TensorProto.INT16),
                       int(TensorProto.INT8), int(TensorProto.UINT8),
                       int(TensorProto.UINT16), int(TensorProto.BOOL)):
        if raw:
            widths = {int(TensorProto.INT32): 4, int(TensorProto.INT16): 2,
                      int(TensorProto.INT8): 1, int(TensorProto.UINT8): 1,
                      int(TensorProto.UINT16): 2, int(TensorProto.BOOL): 1}
            width = widths[data_type]
            if len(raw) != count * width:
                raise CompileError(E_CFG, "initializer %s raw_data length mismatch" % tensor.name)
            fmt = {4: "<%di", 2: "<%dh", 1: "<%db"}[width]
            return [int(v) for v in struct.unpack(fmt % count, raw)]
        values = list(tensor.int32_data)
    else:
        return None
    if len(values) != count:
        raise CompileError(E_CFG, "initializer %s numeric data length mismatch" % tensor.name)
    if data_type in (int(TensorProto.FLOAT), int(TensorProto.FLOAT16), int(TensorProto.DOUBLE)):
        return [float(v) for v in values]
    return [int(v) for v in values]


def _parse_opset(model: Any) -> Tuple[int, List[Dict[str, Any]]]:
    default_versions = []
    imports: List[Dict[str, Any]] = []
    for entry in model.opset_import:
        domain = normalize_domain(entry.domain)
        if domain not in ("", "ai.onnx"):
            raise CompileError(E_UNSUPPORTED_OPSET,
                               "operator domain %r is not supported" % domain)
        version = int(entry.version)
        default_versions.append(version)
        imports.append({"domain": "", "version": version})
    unique_versions = sorted(set(default_versions))
    if len(unique_versions) != 1:
        raise CompileError(E_UNSUPPORTED_OPSET,
                           "expected exactly one default-domain opset import, got %s" %
                           unique_versions)
    opset = unique_versions[0]
    if not (MIN_OPSET <= opset <= MAX_OPSET):
        raise CompileError(E_UNSUPPORTED_OPSET,
                           "ONNX opset %d is outside supported range %d..%d" %
                           (opset, MIN_OPSET, MAX_OPSET))
    return opset, imports


def parse_onnx_bytes(data: bytes, *, name: Optional[str] = None,
                     activation_contract: Optional[Dict[str, Any]] = None,
                     spatial_caps: Optional[Dict[str, Any]] = None) -> Graph:
    """Parse one serialized static ONNX model into a finalized graph IR."""
    _require_onnx()
    model = onnx.load_model_from_string(bytes(data))
    return parse_model(model, model_sha256=_sha256(bytes(data)), name=name,
                       activation_contract=activation_contract,
                       spatial_caps=spatial_caps)


def load_onnx(path: Path, *, name: Optional[str] = None,
              activation_contract: Optional[Dict[str, Any]] = None,
              spatial_caps: Optional[Dict[str, Any]] = None) -> Graph:
    _require_onnx()
    path = Path(path)
    data = path.read_bytes()
    return parse_onnx_bytes(data, name=name, activation_contract=activation_contract,
                            spatial_caps=spatial_caps)


def parse_model(model: Any, *, model_sha256: Optional[str] = None,
                name: Optional[str] = None,
                activation_contract: Optional[Dict[str, Any]] = None,
                spatial_caps: Optional[Dict[str, Any]] = None) -> Graph:
    """Parse an in-memory onnx.ModelProto.  The result is deterministic.

    spatial_caps activates the v0.2 spatial math ceiling for the shape-inference
    pass only; per-engine capability is checked after lowering.
    """
    _require_onnx()
    if any(node.op_type == "MaxPool" and node.domain in ("", "ai.onnx")
           for node in model.graph.node):
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as exc:
            raise CompileError(E_CFG, "ONNX MaxPool model failed schema/type checking: %s" % exc)
    if len(model.functions):
        raise CompileError(E_UNSUPPORTED_OP, "ONNX local functions are not supported")
    if len(model.graph.sparse_initializer):
        raise CompileError(E_UNSUPPORTED_OP, "ONNX sparse initializers are not supported")
    opset, opset_imports = _parse_opset(model)

    tensors: Dict[str, Tensor] = {}
    constants: Dict[str, Sequence[int]] = {}
    numeric_constants: Dict[str, Sequence[float]] = {}
    initializer_names: List[str] = []
    for init in model.graph.initializer:
        if int(init.data_location) == int(TensorProto.EXTERNAL):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "external initializer %s is not supported" % init.name,
                               attribute="external_data")
        dims = tuple(int(v) for v in init.dims)
        if any(v <= 0 for v in dims):
            raise CompileError(E_DYNAMIC_SHAPE,
                               "initializer %s has non-static shape %s" % (init.name, dims))
        dtype = DTYPE_NAMES.get(int(init.data_type))
        if dtype is None:
            raise CompileError(E_CFG, "initializer %s has unsupported dtype" % init.name)
        if init.name in tensors:
            raise CompileError(E_GRAPH, "duplicate tensor/initializer name %s" % init.name)
        tensors[init.name] = Tensor(
            init.name, dtype=dtype, shape=dims, kind="initializer", layout="LOGICAL",
            source={"kind": "onnx_initializer", "sha256": _initializer_bytes_hash(init),
                    "serialized_bytes": int(init.ByteSize())},
            logical_bytes=None,
        )
        initializer_names.append(init.name)
        values = _int_vector(init)
        if values is not None:
            constants[init.name] = values
        numeric = _numeric_vector(init)
        if numeric is not None:
            numeric_constants[init.name] = numeric
            tensors[init.name].source["values"] = list(numeric)

    runtime_inputs: List[str] = []
    declared_inputs: Dict[str, Tuple[int, ...]] = {}
    overridable_inputs: List[str] = []
    for value_info in model.graph.input:
        info = _value_info(value_info, require_shape=True, what="graph input %s" % value_info.name)
        tensor_name = info["name"]
        if not tensor_name:
            raise CompileError(E_GRAPH, "graph input has an empty name")
        if info["dtype"] not in ACTIVATION_DTYPES:
            raise CompileError(E_CFG, "graph input %s dtype %s is unsupported" %
                               (tensor_name, info["dtype"]))
        if tensor_name in tensors:
            if tensor_name in initializer_names:
                overridable_inputs.append(tensor_name)
                continue
            raise CompileError(E_GRAPH, "duplicate graph input %s" % tensor_name)
        tensors[tensor_name] = Tensor(
            tensor_name, dtype=info["dtype"], shape=info["shape"], kind="input",
            layout=layout_for_rank(len(info["shape"])),
            logical_bytes=None,
        )
        runtime_inputs.append(tensor_name)
        declared_inputs[tensor_name] = info["shape"]
    if not runtime_inputs:
        raise CompileError(E_GRAPH, "graph has no runtime inputs")

    declared_value_shapes: Dict[str, Tuple[int, ...]] = {}
    for value_info in model.graph.value_info:
        info = _value_info(value_info, require_shape=False,
                           what="graph value %s" % value_info.name)
        if info["shape"] is not None:
            declared_value_shapes[info["name"]] = info["shape"]

    declared_outputs: Dict[str, Tuple[int, ...]] = {}
    output_names: List[str] = []
    for value_info in model.graph.output:
        info = _value_info(value_info, require_shape=False,
                           what="graph output %s" % value_info.name)
        tensor_name = info["name"]
        if not tensor_name:
            raise CompileError(E_GRAPH, "graph output has an empty name")
        if info["dtype"] not in ACTIVATION_DTYPES | {"int32", "int64"}:
            raise CompileError(E_CFG, "graph output %s dtype %s is unsupported" %
                               (tensor_name, info["dtype"]))
        existing = tensors.get(tensor_name)
        if existing is None:
            tensors[tensor_name] = Tensor(
                tensor_name, dtype=info["dtype"], shape=info["shape"], kind="output",
                layout=(layout_for_rank(len(info["shape"])) if info["shape"] else "LOGICAL"),
                logical_bytes=None,
            )
            existing = tensors[tensor_name]
        else:
            if existing.dtype == "unknown":
                existing.dtype = info["dtype"]
            if info["shape"] is not None and existing.shape is None:
                existing.shape = info["shape"]
                existing.layout = layout_for_rank(len(info["shape"]))
            elif info["shape"] is not None and existing.shape != info["shape"]:
                raise CompileError(E_SHAPE_MISMATCH,
                                   "graph output %s declared twice with different shapes" %
                                   tensor_name)
        if info["shape"] is not None:
            declared_outputs[tensor_name] = info["shape"]
        output_names.append(tensor_name)
    if not output_names:
        raise CompileError(E_GRAPH, "graph has no outputs")

    nodes: List[Node] = []
    constant_values_by_name: Dict[str, List[float]] = {}
    constant_ints_by_name: Dict[str, List[int]] = {}
    for source_index, node_proto in enumerate(model.graph.node):
        op_type = str(node_proto.op_type)
        domain = normalize_domain(node_proto.domain)
        spec = get_spec(op_type, opset=opset, domain=domain)
        raw_attrs = _node_attributes(node_proto)
        constant_attr = None
        if op_type == "Constant":
            constant_attr = raw_attrs.pop("value", None)
            if not isinstance(constant_attr, dict) or not constant_attr.get("__tensor__"):
                raise CompileError(E_UNSUPPORTED_ATTR,
                                   "Constant requires one dense Tensor value attribute",
                                   attribute="value")
        attrs = normalize_attributes(op_type, raw_attrs)
        resize_slots: Optional[Dict[str, Optional[str]]] = None
        if op_type == "Resize":
            inputs, resize_slots = _resize_input_names(node_proto.input)
        else:
            inputs = _trim_optional_names(node_proto.input, "node %s inputs" % node_proto.op_type)
        outputs = _trim_optional_names(node_proto.output, "node %s outputs" % node_proto.op_type)
        if not outputs:
            raise CompileError(E_GRAPH,
                               "node %d %s has no outputs" %
                               (source_index, node_proto.op_type))
        for output_name in outputs:
            if output_name not in tensors:
                tensors[output_name] = Tensor(output_name, dtype="unknown", shape=None)
        node_id = "n%04d" % source_index
        controls = sorted(set(range(len(inputs))) &
                          set(control_input_indices(str(node_proto.op_type))))
        metadata = {"onnx_node_index": source_index,
                    "opcode": spec.opcode,
                    "capability": spec.capability}
        if controls:
            metadata["control_inputs"] = controls
        if resize_slots is not None:
            metadata["resize_slots"] = resize_slots
            metadata["control_inputs"] = list(range(1, len(inputs)))
        if constant_attr is not None:
            if len(outputs) != 1:
                raise CompileError(E_GRAPH, "Constant must have exactly one output",
                                   node_id=node_id)
            output_name = outputs[0]
            values = list(constant_attr["values"])
            metadata.update({
                "constant_dtype": str(constant_attr["dtype"]),
                "constant_shape": list(constant_attr["shape"]),
                "constant_values": values,
                "constant_sha256": str(constant_attr["sha256"]),
            })
            constant_values_by_name[output_name] = values
            if str(constant_attr["dtype"]) in ("int8", "int32", "int64"):
                constant_ints_by_name[output_name] = [int(v) for v in values]
            tensor = tensors[output_name]
            tensor.dtype = str(constant_attr["dtype"])
            tensor.shape = tuple(int(v) for v in constant_attr["shape"])
            tensor.layout = layout_for_rank(len(tensor.shape)) if tensor.shape else "LOGICAL"
            tensor.source = {"kind": "compile_time_constant",
                             "sha256": str(constant_attr["sha256"])}
        nodes.append(Node(
            id=node_id,
            op_type=op_type,
            name=str(node_proto.name) if node_proto.name else op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            placement=spec.placement,
            lowering=spec.lowering,
            source_index=source_index,
            metadata_only=spec.metadata_only,
            metadata=metadata,
        ))

    graph = Graph(
        name=name or model.graph.name or "onnx_graph",
        inputs=runtime_inputs,
        outputs=output_names,
        initializers=initializer_names,
        tensors=tensors,
        nodes=nodes,
        opset_imports=opset_imports,
        layout_policy={
            "logical_layout": "ONNX_row_major_NCHW",
            "target_layout": "NHWC8_A16",
            "reshape_policy": "metadata_only_if_layout_compatible",
        },
        source={
            "format": "onnx",
            "model_sha256": model_sha256 or _sha256(model.SerializeToString(deterministic=True)),
            "ir_version": int(model.ir_version),
            "producer_name": str(model.producer_name),
            "producer_version": str(model.producer_version),
            "opset": opset,
        },
        metadata={
            "supported_opset_range": [MIN_OPSET, MAX_OPSET],
            "overridable_initializer_inputs": sorted(overridable_inputs),
            "declared_input_shapes": {k: list(v) for k, v in sorted(declared_inputs.items())},
            "declared_output_shapes": {k: list(v) for k, v in sorted(declared_outputs.items())},
            "declared_value_shapes": {k: list(v) for k, v in sorted(declared_value_shapes.items())},
            "shape_inference": "pass",
            "topological_sort": "pass",
            "layout_planning": "not_run",
            "memory_planning": "not_run",
            "codegen": "not_run",
            "rtl": "not_run",
            "board": "not_run",
        },
    )
    constants.update(constant_ints_by_name)
    numeric_constants.update(constant_values_by_name)
    if activation_contract is not None:
        # Explicit deployment conversion runs before hardware shape gates so
        # mixed graphs need no fixture-specific ONNX segmentation. Every
        # resulting operator still passes the ordinary strict hardware gate.
        from .activation_contract import build_deployment_ir
        graph, deployment_report = build_deployment_ir(
            graph, activation_contract, infer_shapes=False, caps=spatial_caps)
        graph.metadata.update({"activation_deployment": deployment_report,
                               "declared_output_shapes": {
                                   k: list(v) for k, v in sorted(declared_outputs.items())}})
    infer_graph(graph, constants, numeric_constants, caps=spatial_caps)
    for name, declared in declared_value_shapes.items():
        if name in graph.tensors and graph.tensors[name].shape != declared:
            raise CompileError(E_SHAPE_MISMATCH, "value %s declared %s but inferred %s" %
                               (name, declared, graph.tensors[name].shape))
    for output_name, declared in declared_outputs.items():
        actual = graph.tensors[output_name].shape
        if actual != declared:
            raise CompileError(E_SHAPE_MISMATCH, "graph output %s declared %s but inferred %s" %
                               (output_name, declared, actual))
    return graph
