"""Canonical, deterministic graph IR for the E1 ONNX subset compiler.

The IR deliberately contains no timestamps, absolute paths, protobuf objects or
raw tensor blobs.  ONNX initializers are represented by shape/dtype identity and
a content hash; layout/allocation planning is added by later passes.

Schema: e1-graph-ir-1
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "e1-graph-ir-1"
IR_VERSION = "0.1.0"

E_GRAPH = "E_GRAPH"
E_UNSUPPORTED_OP = "E_UNSUPPORTED_OP"
E_UNSUPPORTED_ATTR = "E_UNSUPPORTED_ATTR"
E_UNSUPPORTED_OPSET = "E_UNSUPPORTED_OPSET"
E_DYNAMIC_SHAPE = "E_DYNAMIC_SHAPE"
E_UNKNOWN_RANK = "E_UNKNOWN_RANK"
E_SHAPE_MISMATCH = "E_SHAPE_MISMATCH"
E_UNSUPPORTED_BROADCAST = "E_UNSUPPORTED_BROADCAST"
E_DTYPE_MISMATCH = "E_DTYPE_MISMATCH"
E_UNSUPPORTED_LAYOUT = "E_UNSUPPORTED_LAYOUT"
E_LAYOUT_MISMATCH = "E_LAYOUT_MISMATCH"
E_QUANT_MISMATCH = "E_QUANT_MISMATCH"
E_LIMIT = "E_LIMIT"
E_CFG = "E_CFG"

# Compute-profile (elaboration-time tn/tm tier) rejection codes.  These are
# appended for the 2026-10-23 compute-profile work; the historical codes above
# keep their exact meaning.
E_CP_UNKNOWN = "E_CP_UNKNOWN"
E_CP_SCHEMA = "E_CP_SCHEMA"
E_CP_UNSUPPORTED_TILE = "E_CP_UNSUPPORTED_TILE"
E_CP_TILE_SET = "E_CP_TILE_SET"
E_CP_TILE_GROUP = "E_CP_TILE_GROUP"
E_CP_CAPACITY = "E_CP_CAPACITY"
E_CP_GEOMETRY = "E_CP_GEOMETRY"
E_CP_BACKEND = "E_CP_BACKEND"
E_CP_CONFLICT = "E_CP_CONFLICT"
E_CP_TARGET = "E_CP_TARGET"
E_CP_OUTPUT = "E_CP_OUTPUT"
E_CP_IDENTITY = "E_CP_IDENTITY"
E_CP_RESOURCE = "E_CP_RESOURCE"
E_CP_NOT_QUALIFIED = "E_CP_NOT_QUALIFIED"

# --- v0.2 spatial capacity (feature-map H/W, independent of Tn/Tm) -----------
E_SPATIAL_SCHEMA = "E_SPATIAL_SCHEMA"
E_SPATIAL_AXIS = "E_SPATIAL_AXIS"
E_SPATIAL_LIMIT = "E_SPATIAL_LIMIT"
E_SPATIAL_ENGINE = "E_SPATIAL_ENGINE"
E_SPATIAL_MEMORY = "E_SPATIAL_MEMORY"
E_SPATIAL_ROW = "E_SPATIAL_ROW"


class CompileError(ValueError):
    """Structured compile-time rejection with a stable machine-readable code."""

    def __init__(self, code: str, message: str, *, node_id: Optional[str] = None,
                 attribute: Optional[str] = None, detail: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.node_id = node_id
        self.attribute = attribute
        self.detail = dict(detail or {})

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.node_id is not None:
            out["node_id"] = self.node_id
        if self.attribute is not None:
            out["attribute"] = self.attribute
        if self.detail:
            out["detail"] = self.detail
        return out


def align_up(value: int, alignment: int) -> int:
    value = int(value)
    alignment = int(alignment)
    if value < 0 or alignment <= 0:
        raise ValueError("align_up requires value >= 0 and alignment > 0")
    return ((value + alignment - 1) // alignment) * alignment


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    """Return a JSON-round-tripped copy so callers cannot mutate our IR."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _check_name(value: str, what: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise CompileError(E_GRAPH, "%s must be a non-empty string" % what)
    return value


@dataclass
class Tensor:
    """One SSA value in the graph.

    ``shape`` is the canonical logical ONNX shape (NCHW for rank-4 activation
    tensors).  ``storage`` is optional target-layout metadata supplied by the
    legacy adapter; the ONNX frontend leaves it unset until layout planning.
    """

    name: str
    dtype: str
    shape: Optional[Tuple[int, ...]]
    kind: str = "intermediate"
    layout: str = "unknown"
    producer: Optional[str] = None
    consumers: List[Dict[str, Any]] = field(default_factory=list)
    storage: Optional[Dict[str, Any]] = None
    quantization: Optional[Dict[str, Any]] = None
    storage_id: Optional[int] = None
    source: Optional[Dict[str, Any]] = None
    logical_bytes: Optional[int] = None

    def validate(self, *, require_shape: bool = True) -> None:
        _check_name(self.name, "tensor name")
        _check_name(self.dtype, "tensor dtype for %s" % self.name)
        if self.kind not in ("input", "initializer", "intermediate", "output"):
            raise CompileError(E_GRAPH, "tensor %s has invalid kind %r" % (self.name, self.kind))
        if require_shape and self.shape is None:
            raise CompileError(E_UNKNOWN_RANK, "tensor %s has no inferred static shape" % self.name)
        if self.shape is not None:
            if not isinstance(self.shape, tuple):
                raise CompileError(E_GRAPH, "tensor %s shape must be a tuple" % self.name)
            if len(self.shape) == 0 and self.kind != "initializer" and not (
                    self.source and self.source.get("kind") == "compile_time_constant"):
                raise CompileError(E_UNKNOWN_RANK,
                                   "tensor %s has scalar rank 0 (only compile-time constants "
                                   "and initializers may be scalar)" % self.name)
            for dim in self.shape:
                if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                    raise CompileError(E_DYNAMIC_SHAPE,
                                       "tensor %s has non-static dimension %r" % (self.name, dim))
        if self.storage_id is not None and (not isinstance(self.storage_id, int) or self.storage_id < 0):
            raise CompileError(E_GRAPH, "tensor %s has invalid storage_id" % self.name)
        for consumer in self.consumers:
            if not isinstance(consumer.get("node"), str) or not isinstance(consumer.get("input_index"), int):
                raise CompileError(E_GRAPH, "tensor %s has malformed consumer entry" % self.name)

    def to_dict(self) -> Dict[str, Any]:
        self.validate(require_shape=True)
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape) if self.shape is not None else None,
            "kind": self.kind,
            "layout": self.layout,
            "producer": self.producer,
            "consumers": _json_copy(sorted(self.consumers,
                                            key=lambda c: (c["node"], c["input_index"]))),
            "storage": _json_copy(self.storage) if self.storage is not None else None,
            "quantization": _json_copy(self.quantization) if self.quantization is not None else None,
            "storage_id": self.storage_id,
            "source": _json_copy(self.source) if self.source is not None else None,
            "logical_bytes": self.logical_bytes,
        }


@dataclass
class Node:
    """One operator invocation in canonical topological order."""

    id: str
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)
    placement: str = "unplanned"
    lowering: str = "unplanned"
    source_index: int = 0
    order: int = -1
    metadata_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _check_name(self.id, "node id")
        _check_name(self.op_type, "node op_type")
        if self.placement not in ("pl", "ps", "metadata", "unplanned"):
            raise CompileError(E_GRAPH, "node %s has invalid placement %r" % (self.id, self.placement))
        if len(self.outputs) != len(set(self.outputs)):
            raise CompileError(E_GRAPH, "node %s has duplicate output names" % self.id)
        if not self.outputs:
            raise CompileError(E_GRAPH, "node %s has no outputs" % self.id)
        for name in list(self.inputs) + list(self.outputs):
            _check_name(name, "node %s tensor name" % self.id)
        if self.order < 0:
            raise CompileError(E_GRAPH, "node %s has not been topologically ordered" % self.id)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "id": self.id,
            "op_type": self.op_type,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": _json_copy(self.attrs),
            "placement": self.placement,
            "lowering": self.lowering,
            "metadata_only": bool(self.metadata_only),
            "source_index": int(self.source_index),
            "order": int(self.order),
            "metadata": _json_copy(self.metadata),
        }


class Graph:
    """Canonical DAG container with deterministic topological ordering."""

    def __init__(self, *, name: str, inputs: Sequence[str], outputs: Sequence[str],
                 initializers: Sequence[str], tensors: Dict[str, Tensor], nodes: Sequence[Node],
                 opset_imports: Optional[Sequence[Dict[str, Any]]] = None,
                 layout_policy: Optional[Dict[str, Any]] = None,
                 source: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = _check_name(name, "graph name")
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.initializers = list(initializers)
        self.tensors: Dict[str, Tensor] = dict(tensors)
        self.nodes: List[Node] = list(nodes)
        self.opset_imports = [dict(x) for x in (opset_imports or [])]
        self.layout_policy = dict(layout_policy or {
            "logical_layout": "ONNX_row_major",
            "target_layout": "NHWC8_A16",
            "reshape_policy": "metadata_only_if_layout_compatible",
        })
        self.source = dict(source or {})
        self.metadata = dict(metadata or {})
        self._finalized = False

    def _topological_nodes(self) -> List[Node]:
        ids = [node.id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise CompileError(E_GRAPH, "duplicate node id in graph")
        available = set(self.inputs) | set(self.initializers)
        pending = sorted(self.nodes, key=lambda n: (n.source_index, n.id))
        ordered: List[Node] = []
        while pending:
            ready = [n for n in pending if all(name in available for name in n.inputs)]
            if not ready:
                unresolved = {n.id: [name for name in n.inputs if name not in available]
                              for n in pending}
                raise CompileError(E_GRAPH, "graph contains a cycle or unresolved SSA inputs",
                                   detail={"unresolved": unresolved})
            node = ready[0]
            pending.remove(node)
            ordered.append(node)
            for output in node.outputs:
                if output in available:
                    raise CompileError(E_GRAPH, "SSA value %s has more than one producer" % output,
                                       node_id=node.id)
                available.add(output)
        for index, node in enumerate(ordered):
            node.order = index
        return ordered

    def _rebuild_tensor_relations(self) -> None:
        for node in self.nodes:
            for input_name in node.inputs:
                if input_name not in self.tensors:
                    raise CompileError(E_GRAPH, "node %s references unknown tensor %s" %
                                       (node.id, input_name), node_id=node.id)
            for output_name in node.outputs:
                tensor = self.tensors.get(output_name)
                if tensor is None:
                    tensor = Tensor(output_name, dtype="unknown", shape=None)
                    self.tensors[output_name] = tensor
                if tensor.producer is not None:
                    raise CompileError(E_GRAPH, "tensor %s has multiple producers" % output_name,
                                       node_id=node.id)
                tensor.producer = node.id

        for tensor in self.tensors.values():
            tensor.consumers = []

        order_by_id = {node.id: node.order for node in self.nodes}
        for node in self.nodes:
            for input_index, input_name in enumerate(node.inputs):
                self.tensors[input_name].consumers.append({
                    "node": node.id,
                    "input_index": int(input_index),
                })
        for tensor in self.tensors.values():
            tensor.consumers.sort(key=lambda c: (order_by_id.get(c["node"], 10 ** 9),
                                                 c["input_index"]))

        inputs = set(self.inputs)
        outputs = set(self.outputs)
        initializers = set(self.initializers)
        for name, tensor in self.tensors.items():
            if name in initializers:
                tensor.kind = "initializer"
            elif name in inputs:
                tensor.kind = "input"
            elif name in outputs:
                tensor.kind = "output"
            else:
                tensor.kind = "intermediate"

    def finalize(self) -> "Graph":
        if not self.inputs or not self.outputs:
            raise CompileError(E_GRAPH, "graph must have at least one input and one output")
        for name in self.inputs + self.outputs + self.initializers:
            if name not in self.tensors:
                raise CompileError(E_GRAPH, "graph references unknown tensor %s" % name)
        self.nodes = self._topological_nodes()
        for tensor in self.tensors.values():
            tensor.producer = None
            tensor.consumers = []
        self._rebuild_tensor_relations()
        for tensor in self.tensors.values():
            tensor.validate(require_shape=True)
        for node in self.nodes:
            node.validate()
        self._finalized = True
        return self

    def document(self) -> Dict[str, Any]:
        if not self._finalized:
            self.finalize()
        opsets = sorted((dict(x) for x in self.opset_imports),
                        key=lambda x: (str(x.get("domain", "")), int(x.get("version", 0))))
        return {
            "schema": SCHEMA,
            "ir_version": IR_VERSION,
            "name": self.name,
            "source": _json_copy(self.source),
            "opset_imports": opsets,
            "layout_policy": _json_copy(self.layout_policy),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "initializers": sorted(self.initializers),
            "nodes": [node.to_dict() for node in self.nodes],
            "tensors": {name: self.tensors[name].to_dict() for name in sorted(self.tensors)},
            "metadata": _json_copy(self.metadata),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def pretty_json(self) -> str:
        return json.dumps(self.document(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Graph":
        if value.get("schema") != SCHEMA:
            raise CompileError(E_GRAPH, "unsupported graph IR schema %r" % value.get("schema"))
        tensors: Dict[str, Tensor] = {}
        for name, raw in value.get("tensors", {}).items():
            shape = raw.get("shape")
            tensors[name] = Tensor(
                name=name,
                dtype=raw["dtype"],
                shape=tuple(shape) if shape is not None else None,
                kind=raw.get("kind", "intermediate"),
                layout=raw.get("layout", "unknown"),
                storage=raw.get("storage"),
                quantization=raw.get("quantization"),
                storage_id=raw.get("storage_id"),
                source=raw.get("source"),
                logical_bytes=raw.get("logical_bytes"),
            )
        nodes: List[Node] = []
        for index, raw in enumerate(value.get("nodes", [])):
            nodes.append(Node(
                id=raw["id"], op_type=raw["op_type"], name=raw.get("name", raw["id"]),
                inputs=list(raw["inputs"]), outputs=list(raw["outputs"]),
                attrs=dict(raw.get("attrs", {})),
                placement=raw.get("placement", "unplanned"),
                lowering=raw.get("lowering", "unplanned"),
                source_index=int(raw.get("source_index", index)),
                order=int(raw.get("order", -1)),
                metadata_only=bool(raw.get("metadata_only", False)),
                metadata=dict(raw.get("metadata", {})),
            ))
        graph = cls(
            name=value["name"], inputs=value.get("inputs", []),
            outputs=value.get("outputs", []), initializers=value.get("initializers", []),
            tensors=tensors, nodes=nodes, opset_imports=value.get("opset_imports", []),
            layout_policy=value.get("layout_policy"), source=value.get("source"),
            metadata=value.get("metadata"),
        )
        graph.finalize()
        return graph


def tensor_names(tensors: Iterable[Tensor]) -> List[str]:
    return [tensor.name for tensor in tensors]