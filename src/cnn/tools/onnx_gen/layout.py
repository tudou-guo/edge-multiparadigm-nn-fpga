"""Logical-shape to storage-layout planning for E1 W2.

INT8 rank-4 activations use the frozen NHWC8_A16 geometry.  Non-INT8 structural
IR is still plannable for DAG/liveness diagnostics, but is marked
abi_compatible=false and must not be presented as hardware ABI storage.
Reshape/Flatten are metadata aliases only; this pass does not move bytes.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from .graph_ir import (
    CompileError,
    E_GRAPH,
    E_LAYOUT_MISMATCH,
    Graph,
    Tensor,
    align_up,
)
from .op_registry import control_input_indices
from .shape_infer import (
    ABI_ALIGNMENT,
    CHANNEL_BLOCK,
    bias_storage,
    depthwise_bias_storage,
    depthwise_weight_storage,
    layout_for_rank,
    logical_nbytes,
    nhwc8_a16_storage,
    prod,
    weight_storage,
)

LAYOUT_SCHEMA = "e1-layout-plan-1"
STRUCT_ALIGNMENT = 16

#: Dense Conv weight formats.
#:
#: ``oihw8_kb_ic_oc_k9`` is the frozen ABI 1.0 nine-tap layout
#: ([ocb][tap(9)][icb] blocks of 64 B, bytes = blocks * 9 * 64).  Every legacy
#: build keeps it byte-for-byte.
#:
#: ``k1_center_tap_compact`` is the affine-int8-v1 native 1x1 path: the logical
#: kernel stays 1 (the lowering keeps the mathematically identical center-tap
#: window) and only the center-tap group is stored/loaded, bytes = blocks * 64.
#: It is selected by the conv_cfg bit 11 emitted for affine Conv nodes whose
#: lowering record says the source kernel was 1x1.
LEGACY_DENSE_WEIGHT_FORMAT = "oihw8_kb_ic_oc_k9"
NATIVE_K1_WEIGHT_FORMAT = "k1_center_tap_compact"


def conv_weight_format(node) -> str:
    """Hardware weight format required by a lowered dense Conv node.

    Only the affine-int8-v1 path opts into the compact native K1 format; a
    legacy node (no ``affine`` metadata) always keeps the frozen nine-tap
    layout so legacy instruction/parameter bytes cannot change.
    """
    if node is None:
        return LEGACY_DENSE_WEIGHT_FORMAT
    if (node.metadata.get("affine") and
            [int(v) for v in node.metadata.get("source_kernel_shape", [])] == [1, 1]):
        return NATIVE_K1_WEIGHT_FORMAT
    return LEGACY_DENSE_WEIGHT_FORMAT


def dense_weight_storage(shape, weight_format: str = LEGACY_DENSE_WEIGHT_FORMAT):
    """Storage geometry of a dense Conv weight blob for ``weight_format``.

    ``shape`` is the *lowered* OIHW shape (always 3x3 after the center-tap
    lowering).  The returned ``bytes`` is exactly the ABI blob the parameter
    packer must produce; it drives the memory plan and the payload
    ``weight_bytes`` word, so layout and packer cannot disagree.
    """
    if len(shape) != 4:
        raise CompileError(E_CFG, "Conv weight storage requires OIHW rank 4")
    cout, cin, kh, kw = (int(v) for v in shape)
    if (kh, kw) != (3, 3):
        raise CompileError(E_CFG, "Conv weight storage only defines 3x3 kernels")
    if weight_format not in (LEGACY_DENSE_WEIGHT_FORMAT, NATIVE_K1_WEIGHT_FORMAT):
        raise CompileError(E_CFG, "unknown dense weight format %r" % (weight_format,))
    cout_alloc = align_up(cout, CHANNEL_BLOCK)
    cin_alloc = align_up(cin, CHANNEL_BLOCK)
    blocks = (cout_alloc // CHANNEL_BLOCK) * (cin_alloc // CHANNEL_BLOCK)
    tap_groups = 1 if weight_format == NATIVE_K1_WEIGHT_FORMAT else 9
    return {
        "layout": "OIHW8_KB_IC_OC",
        "weight_format": weight_format,
        "tap_groups": tap_groups,
        "cout_alloc": cout_alloc,
        "cin_alloc": cin_alloc,
        "blocks": blocks,
        "bytes": blocks * tap_groups * 64,
        "alignment": ABI_ALIGNMENT,
    }


def _host_storage() -> Dict[str, Any]:
    return {
        "layout": "HOST_PS",
        "bytes": 0,
        "alignment": 0,
        "abi_compatible": False,
        "managed_by_host": True,
    }


def _logical_storage(tensor: Tensor) -> Dict[str, Any]:
    if tensor.shape is None:
        raise CompileError(E_GRAPH, "tensor %s has no shape for layout planning" % tensor.name)
    raw_bytes = logical_nbytes(tensor.shape, tensor.dtype)
    if raw_bytes is None:
        raise CompileError(E_GRAPH, "tensor %s dtype %s has unknown byte width" %
                           (tensor.name, tensor.dtype))
    row_elems = int(tensor.shape[-1]) if tensor.shape else 1
    elem_bytes = raw_bytes // max(1, prod(tensor.shape))
    row_stride = align_up(row_elems * elem_bytes, STRUCT_ALIGNMENT)
    rows = max(1, prod(tensor.shape[:-1]))
    return {
        "layout": "LOGICAL_CONTIGUOUS",
        "logical_layout": layout_for_rank(len(tensor.shape)),
        "bytes": align_up(rows * row_stride, STRUCT_ALIGNMENT),
        "raw_logical_bytes": int(raw_bytes),
        "alignment": STRUCT_ALIGNMENT,
        "row_stride_bytes": row_stride,
        "elem_bytes": elem_bytes,
        "abi_compatible": False,
        "managed_by_host": False,
    }


def _activation_storage(tensor: Tensor) -> Dict[str, Any]:
    if tensor.shape is not None and len(tensor.shape) == 4 and tensor.dtype in ("int8", "uint8"):
        storage = nhwc8_a16_storage(tensor.shape)
        storage.update({
            "abi_compatible": True,
            "managed_by_host": False,
            "elem_bytes": 1,
            "logical_layout": "NCHW",
        })
        return storage
    return _logical_storage(tensor)


def _initializer_storage(tensor: Tensor, *, conv_weight: bool, conv_bias: bool,
                        conv_weight_node=None) -> Dict[str, Any]:
    if tensor.storage is not None:
        storage = dict(tensor.storage)
        storage.setdefault("abi_compatible", storage.get("layout") in
                           ("NHWC8_A16", "OIHW8_KB_IC_OC", "I32_C", "M0_SHIFT_ZP"))
        storage.setdefault("managed_by_host", False)
        return storage
    if conv_weight and tensor.shape is not None and tensor.dtype in ("int8", "uint8"):
        weight_format = conv_weight_format(conv_weight_node)
        if weight_format == NATIVE_K1_WEIGHT_FORMAT:
            storage = dense_weight_storage(tensor.shape, weight_format)
        else:
            # Legacy nine-tap layout: byte-identical to the frozen weight_storage.
            storage = weight_storage(tensor.shape)
        storage.update({"abi_compatible": True, "managed_by_host": False, "elem_bytes": 1})
        return storage
    if conv_bias and tensor.shape is not None and tensor.dtype == "int32":
        storage = bias_storage(tensor.shape)
        storage.update({"abi_compatible": True, "managed_by_host": False, "elem_bytes": 4})
        return storage
    return _logical_storage(tensor)


def _conv_initializer_roles(graph: Graph):
    weights: Set[str] = set()
    biases: Set[str] = set()
    for node in graph.nodes:
        if node.op_type == "Conv":
            if len(node.inputs) >= 2:
                weights.add(node.inputs[1])
            if len(node.inputs) >= 3:
                biases.add(node.inputs[2])
    return weights, biases


def _depthwise_initializer_roles(graph: Graph):
    """Depthwise 0x001C weight/bias tensors use DW_OC16_K9 / I32_C storage."""
    weights: Set[str] = set()
    biases: Set[str] = set()
    for node in graph.nodes:
        if node.op_type == "DepthwiseConv":
            if len(node.inputs) >= 2:
                weights.add(node.inputs[1])
            if len(node.inputs) >= 3:
                biases.add(node.inputs[2])
    return weights, biases


def _all_consumers_are_host(graph: Graph, tensor_name: str) -> bool:
    consumers = graph.tensors[tensor_name].consumers
    if not consumers:
        return False
    node_by_id = {node.id: node for node in graph.nodes}
    return all(node_by_id[row["node"]].placement == "ps" for row in consumers)


def _all_consumers_are_metadata(graph: Graph, tensor_name: str) -> bool:
    consumers = graph.tensors[tensor_name].consumers
    if not consumers:
        return False
    node_by_id = {node.id: node for node in graph.nodes}
    return all(node_by_id[row["node"]].metadata_only for row in consumers)


def _constant_storage() -> Dict[str, Any]:
    return {
        "layout": "COMPILE_TIME_CONSTANT",
        "bytes": 0,
        "alignment": 0,
        "abi_compatible": False,
        "managed_by_host": False,
    }


def _data_initializers(graph: Graph) -> Set[str]:
    """Rank-4 INT8 initializers consumed as *data* by a PL operator.

    E1-P3 Mul may take a same-shape constant operand.  Such a tensor is not a
    Conv weight: it must be stored in the activation ABI (NHWC8_A16), otherwise
    a byte-wise elementwise core would read the wrong bytes.
    """
    names: Set[str] = set()
    for node in graph.nodes:
        if node.placement != "pl" or node.metadata_only:
            continue
        control = control_input_indices(node.op_type)
        for index, name in enumerate(node.inputs):
            if index in control:
                continue
            if node.op_type in ("Conv", "DepthwiseConv") and index in (1, 2):
                continue
            tensor = graph.tensors.get(name)
            if tensor is not None and tensor.kind == "initializer":
                names.add(name)
    return names


def build_layout_plan(graph: Graph) -> Dict[str, Any]:
    graph.finalize()
    conv_weights, conv_biases = _conv_initializer_roles(graph)
    conv_weight_nodes = {node.inputs[1]: node for node in graph.nodes
                         if node.op_type == "Conv" and len(node.inputs) >= 2}
    dw_weights, dw_biases = _depthwise_initializer_roles(graph)
    data_initializers = _data_initializers(graph)

    control_initializers: Set[str] = set()
    for node in graph.nodes:
        for index in control_input_indices(node.op_type):
            if index < len(node.inputs):
                control_initializers.add(node.inputs[index])

    host_outputs: Set[str] = set()
    for node in graph.nodes:
        if node.placement == "ps":
            host_outputs.update(node.outputs)

    tensors: Dict[str, Dict[str, Any]] = {}
    for name, tensor in sorted(graph.tensors.items()):
        if name in host_outputs:
            storage = _host_storage()
            managed_by = "host"
        elif tensor.kind == "initializer" and _all_consumers_are_host(graph, name):
            storage = _host_storage()
            managed_by = "host"
        elif tensor.kind == "initializer" and name in control_initializers:
            storage = _constant_storage()
            managed_by = "constant"
        elif tensor.kind == "initializer" and _all_consumers_are_metadata(graph, name):
            storage = _constant_storage()
            managed_by = "constant"
        elif tensor.kind == "initializer" and name in data_initializers:
            storage = _activation_storage(tensor)
            storage["initializer_activation"] = True
            managed_by = "pl"
        elif tensor.kind == "initializer" and name in dw_weights:
            storage = depthwise_weight_storage(tuple(int(v) for v in tensor.shape))
            storage.update({"abi_compatible": True, "managed_by_host": False,
                            "elem_bytes": 1})
            managed_by = "pl"
        elif tensor.kind == "initializer" and name in dw_biases:
            storage = depthwise_bias_storage(tuple(int(v) for v in tensor.shape))
            storage.update({"abi_compatible": True, "managed_by_host": False,
                            "elem_bytes": 4})
            managed_by = "pl"
        elif tensor.kind == "initializer":
            storage = _initializer_storage(tensor, conv_weight=name in conv_weights,
                                           conv_bias=name in conv_biases,
                                           conv_weight_node=conv_weight_nodes.get(name))
            managed_by = "pl"
        else:
            storage = _activation_storage(tensor)
            managed_by = "pl"
        tensors[name] = {
            "tensor": name,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape) if tensor.shape is not None else None,
            "logical_layout": layout_for_rank(len(tensor.shape)) if tensor.shape else "unknown",
            "storage": storage,
            "managed_by": managed_by,
            "alias_of": None,
            "alias_group": None,
            "view_offset_bytes": 0,
            "view_bytes": int(storage.get("bytes", 0)),
            "view_kind": "none",
        }

    root_of: Dict[str, str] = {}
    alias_groups: Dict[str, List[str]] = {}
    aliases: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if not node.metadata_only:
            continue
        if not node.inputs or len(node.outputs) != 1:
            raise CompileError(E_GRAPH,
                               "metadata node %s must have a data input and one output" % node.id,
                               node_id=node.id)
        source_name = node.inputs[0]
        output_name = node.outputs[0]
        for aux_name in node.inputs[1:]:
            aux = graph.tensors.get(aux_name)
            if aux is None or aux.kind != "initializer":
                raise CompileError(E_GRAPH,
                                   "metadata node %s control input %s is not an initializer" %
                                   (node.id, aux_name), node_id=node.id)
        root = root_of.get(source_name, source_name)
        source_row = tensors[source_name]
        root_row = tensors[root]
        source_tensor = graph.tensors[root]
        output_tensor = graph.tensors[output_name]
        if source_tensor.dtype != output_tensor.dtype:
            raise CompileError(E_LAYOUT_MISMATCH,
                               "metadata alias dtype mismatch: %s -> %s" %
                               (source_tensor.dtype, output_tensor.dtype), node_id=node.id)

        view_kind = str(node.metadata.get("view_kind", "identity"))
        view_offset = int(source_row.get("view_offset_bytes", 0) or 0)
        if node.op_type == "Slice" and view_kind == "h_view":
            row_stride = int(root_row["storage"].get("row_stride_bytes", 0))
            if row_stride <= 0 or output_tensor.shape is None:
                raise CompileError(E_LAYOUT_MISMATCH,
                                   "Slice H-view lacks row stride/shape", node_id=node.id)
            start_h = int(node.metadata["starts"][2])
            rows = int(output_tensor.shape[2])
            view_offset += start_h * row_stride
            view_bytes = align_up(rows * row_stride, 16)
        else:
            if prod(source_tensor.shape) != prod(output_tensor.shape):
                raise CompileError(E_LAYOUT_MISMATCH,
                                   "metadata alias element count mismatch: %s -> %s" %
                                   (source_tensor.shape, output_tensor.shape), node_id=node.id)
            view_bytes = int(source_row["storage"].get("bytes", 0))

        root_bytes = int(root_row["storage"].get("bytes", 0))
        if view_offset < 0 or view_bytes <= 0 or view_offset + view_bytes > root_bytes:
            raise CompileError(E_LAYOUT_MISMATCH,
                               "metadata view [%d,%d) escapes root %s bytes %d" %
                               (view_offset, view_offset + view_bytes, root, root_bytes),
                               node_id=node.id)
        root_storage = root_row["storage"]
        if root_storage.get("managed_by_host"):
            storage = _host_storage()
            managed_by = "host"
        else:
            storage = dict(root_storage)
            storage["bytes"] = view_bytes
            managed_by = ("view" if (view_offset != 0 or view_bytes != root_bytes)
                          else "alias")
            if managed_by == "view":
                storage["view_of"] = root
                storage["view_offset_bytes"] = view_offset
                storage["view_bytes"] = view_bytes
        tensors[output_name]["storage"] = storage
        tensors[output_name]["managed_by"] = managed_by
        tensors[output_name]["alias_of"] = root
        tensors[output_name]["alias_group"] = root
        tensors[output_name]["view_offset_bytes"] = view_offset
        tensors[output_name]["view_bytes"] = view_bytes
        tensors[output_name]["view_kind"] = view_kind
        root_of[output_name] = root
        alias_groups.setdefault(root, []).append(output_name)
        aliases.append({"node": node.id, "order": node.order,
                        "input": source_name, "output": output_name, "root": root,
                        "view_kind": view_kind, "view_offset_bytes": view_offset,
                        "view_bytes": view_bytes})

    for node in graph.nodes:
        if node.op_type == "Add":
            a = tensors[node.inputs[0]]["storage"]
            b = tensors[node.inputs[1]]["storage"]
            if (a.get("layout") != b.get("layout") or int(a.get("bytes", 0)) != int(b.get("bytes", 0))):
                raise CompileError(E_LAYOUT_MISMATCH,
                                   "Add inputs have incompatible storage layouts", node_id=node.id)

    lowerings: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.placement == "ps":
            continue
        if node.metadata_only:
            kind = "noop" if node.op_type == "Pad" else "view"
            lowerings.append({"node": node.id, "op_type": node.op_type, "kind": kind,
                              "view_kind": node.metadata.get("view_kind", "identity"),
                              "copy_required": False})
            continue
        if node.op_type in ("Pad", "Slice", "Concat"):
            if not node.metadata.get("copy_required"):
                raise CompileError(E_GRAPH,
                                   "%s node %s is neither a view nor an explicit copy" %
                                   (node.op_type, node.id), node_id=node.id)
            output_row = tensors[node.outputs[0]]
            if output_row["managed_by"] != "pl":
                raise CompileError(E_LAYOUT_MISMATCH,
                                   "%s copy output %s must own a PL buffer" %
                                   (node.op_type, node.outputs[0]), node_id=node.id)
            if output_row["storage"].get("layout") != "NHWC8_A16":
                raise CompileError(E_LAYOUT_MISMATCH,
                                   "%s copy output %s is not NHWC8_A16" %
                                   (node.op_type, node.outputs[0]), node_id=node.id)
            lowerings.append({"node": node.id, "op_type": node.op_type, "kind": "copy",
                              "view_kind": "none", "copy_required": True,
                              "output_storage": output_row["storage"].get("layout")})
            continue
        lowerings.append({"node": node.id, "op_type": node.op_type, "kind": "compute",
                          "view_kind": "none", "copy_required": False})

    node_by_id = {node.id: node for node in graph.nodes}
    for node in graph.nodes:
        if node.placement != "ps":
            continue
        for output_name in node.outputs:
            for consumer in graph.tensors[output_name].consumers:
                if node_by_id[consumer["node"]].placement != "ps":
                    raise CompileError(E_LAYOUT_MISMATCH,
                                       "PS output %s is consumed by a PL node" % output_name,
                                       node_id=node.id)

    host_boundaries = []
    for node in graph.nodes:
        if node.placement == "ps":
            host_boundaries.append({
                "node": node.id,
                "op_type": node.op_type,
                "inputs": list(node.inputs),
                "outputs": list(node.outputs),
                "allocation": "host_managed_no_pl_buffer",
            })

    stats = {
        "tensors": len(tensors),
        "pl_managed": sum(1 for row in tensors.values() if row["managed_by"] == "pl"),
        "alias_tensors": sum(1 for row in tensors.values() if row["managed_by"] == "alias"),
        "view_tensors": sum(1 for row in tensors.values() if row["managed_by"] == "view"),
        "host_tensors": sum(1 for row in tensors.values() if row["managed_by"] == "host"),
        "constant_tensors": sum(1 for row in tensors.values() if row["managed_by"] == "constant"),
        "abi_compatible_tensors": sum(1 for row in tensors.values()
                                      if row["storage"].get("abi_compatible")),
        "logical_only_tensors": sum(1 for row in tensors.values()
                                    if not row["storage"].get("abi_compatible") and
                                    row["managed_by"] not in ("host", "constant")),
        "copy_nodes": sum(1 for row in lowerings if row["kind"] == "copy"),
        "view_nodes": sum(1 for row in lowerings if row["kind"] == "view"),
        "noop_nodes": sum(1 for row in lowerings if row["kind"] == "noop"),
    }
    return {
        "schema": LAYOUT_SCHEMA,
        "graph_digest": graph.digest(),
        "policy": "int8_nhwc8_a16; explicit_view_copy; logical_fallback_for_non_int8",
        "tensors": tensors,
        "aliases": aliases,
        "alias_groups": {root: sorted(members) for root, members in sorted(alias_groups.items())},
        "lowerings": lowerings,
        "host_boundaries": host_boundaries,
        "stats": stats,
        "gates": {"layout_planning": "pass", "alias_planning": "pass",
                  "view_copy_planning": "pass", "memory_planning": "not_run"},
    }
