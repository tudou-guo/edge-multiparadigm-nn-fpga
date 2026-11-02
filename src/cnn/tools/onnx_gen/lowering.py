"""Static lowering passes between source ONNX IR and hardware planning.

The source graph keeps ONNX identity (Conv1x1, Split, Constant, metadata ops).
The lowered graph is the exact graph consumed by layout/memory/codegen:

* Conv1x1 becomes the existing dense 3x3 core with a zero-padded center tap.
* Split becomes one explicit Slice copy per output.
* Constant nodes become compile-time initializers.
* Identity/Squeeze/Unsqueeze/reshape-style nodes remain metadata aliases.

Schema: e1-p2-lowering-1
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List, Tuple

from .exact_quant import ExactQuantError, bn_fold_channel, exact
from .graph_ir import CompileError, Graph, Node, Tensor
from .shape_infer import infer_graph, layout_for_rank

LOWERING_SCHEMA = "e1-p3-lowering-1"


def _source_values(tensor: Tensor) -> List[Any]:
    values = (tensor.source or {}).get("values")
    if values is None:
        raise KeyError(tensor.name)
    return list(values)


def _fold_batch_norm(graph: Graph) -> Dict[str, Any]:
    """Fold inference BatchNormalization into its producing Conv or refuse.

    The integer-equivalence contract is exact, not sample based:

    * the Conv output tensor must be declared non-materialized, so no INT8
      rounding happens between Conv and BN;
    * the accumulator-domain scale is s_acc = s_input * weight_scale[oc];
    * BN acts on that real value:  y = A*acc + B,  A = gamma/sqrt(var+eps),
      B = beta - A*mean;
    * folding requires A*s_acc/s_out == M0/2**SHIFT exactly and
      B/(A*s_acc) to be an exact integer, so the folded integer rounding is
      applied to the very same rational number the original BN would produce.
    """
    producers: Dict[str, Node] = {}
    consumers: Dict[str, List[Tuple[Node, int]]] = {}
    for node in graph.nodes:
        for index, name in enumerate(node.inputs):
            if name:
                consumers.setdefault(name, []).append((node, index))
        for name in node.outputs:
            producers[name] = node
    folded: Dict[str, Any] = {}
    removed: Dict[str, Any] = {}
    for node in list(graph.nodes):
        if node.op_type != "BatchNormalization":
            continue
        conv_in = node.inputs[0]
        producer = producers.get(conv_in)
        if producer is None or producer.op_type != "Conv":
            raise CompileError(
                "E_UNSUPPORTED_OP",
                "BatchNormalization %s is not preceded by Conv; only Conv->BN "
                "folding with exact integer equivalence is supported" % node.id,
                node_id=node.id)
        if len(producer.outputs) != 1:
            raise CompileError("E_CFG", "Conv %s feeding BN has several outputs" %
                               producer.id, node_id=node.id)
        uses = consumers.get(conv_in, [])
        if len(uses) != 1 or uses[0][0] is not node:
            raise CompileError(
                "E_UNSUPPORTED_OP",
                "Conv output %s has %d consumer(s); a foldable BN boundary must be "
                "single-consumer and must not be a graph output" % (conv_in, len(uses)),
                node_id=node.id)
        if conv_in in graph.outputs:
            raise CompileError(
                "E_UNSUPPORTED_OP",
                "Conv output %s is a graph output; its INT8 rounding is materialized "
                "and cannot be removed by BN folding" % conv_in, node_id=node.id)
        boundary = graph.tensors[conv_in]
        quant = boundary.quantization or {}
        if quant.get("materialized") is not False:
            raise CompileError(
                "E_QUANT_MISMATCH",
                "Conv output %s is materialized (contract scale present); folding BN "
                "would delete a declared INT8 rounding point" % conv_in,
                node_id=node.id)
        if not producer.metadata.get("fold_pending"):
            raise CompileError("E_CFG", "Conv %s has no pending fold state" %
                               producer.id, node_id=node.id)
        output_tensor = graph.tensors[node.outputs[0]]
        out_quant = output_tensor.quantization or {}
        if out_quant.get("materialized") is False or "scale" not in out_quant:
            raise CompileError(
                "E_QUANT_MISMATCH",
                "BatchNormalization output %s must be a materialized INT8 tensor with "
                "an explicit scale" % node.outputs[0], node_id=node.id)
        params = {}
        for name, key in zip(node.inputs[1:5], ("gamma", "beta", "mean", "var")):
            tensor = graph.tensors[name]
            if tensor.kind != "initializer":
                raise CompileError(
                    "E_UNSUPPORTED_ATTR",
                    "BatchNormalization %s parameter %s is not a static initializer" %
                    (key, name), node_id=node.id, attribute=key)
            try:
                params[key] = _source_values(tensor)
            except KeyError:
                raise CompileError("E_CFG", "BatchNormalization %s has no static values" %
                                   name, node_id=node.id)
        cout = int(graph.tensors[conv_in].shape[1])
        for key, values in params.items():
            if len(values) != cout:
                raise CompileError(
                    "E_SHAPE_MISMATCH",
                    "BatchNormalization %s length %d != Cout %d" % (key, len(values), cout),
                    node_id=node.id)
        epsilon = float(node.attrs["epsilon"])
        s_in = float(producer.metadata["quant_input_scale"])
        s_out = float(out_quant["scale"])
        weight_scales = [float(v) for v in producer.metadata["weight_scales"]]
        bias_name = producer.inputs[2] if len(producer.inputs) >= 3 else None
        if bias_name is None:
            raise CompileError("E_CFG", "foldable Conv %s requires an INT32 bias" %
                               producer.id, node_id=node.id)
        bias_values = _source_values(graph.tensors[bias_name])
        if len(bias_values) != cout:
            raise CompileError("E_CFG", "Conv bias length mismatch", node_id=producer.id)
        per_channel = []
        records = []
        folded_bias = []
        try:
            for oc in range(cout):
                info = bn_fold_channel(
                    gamma=params["gamma"][oc], beta=params["beta"][oc],
                    mean=params["mean"][oc], var=params["var"][oc], epsilon=epsilon,
                    input_scale=s_in, weight_scale=weight_scales[oc],
                    output_scale=s_out, bias_int=int(bias_values[oc]))
                per_channel.append({
                    "oc": oc,
                    "A": info["a_float"], "B": info["b_float"],
                    "accumulator_scale": float(info["s_acc"]),
                    "ratio_numerator": info["ratio"].numerator,
                    "ratio_denominator": info["ratio"].denominator,
                    "M0": info["M0"], "SHIFT": info["SHIFT"],
                    "bias_delta": info["bias_delta"], "bias_int": info["bias_int"],
                    "folded_weight_scale": float(Fraction(weight_scales[oc]) *
                                                 Fraction(info["a_float"])),
                })
                records.append({
                    "oc": oc,
                    "weight_scale": float(Fraction(weight_scales[oc]) *
                                          Fraction(info["a_float"])),
                    "bias_scale": float(info["s_acc"]),
                    "multiplier": float(info["ratio"]),
                    "M0": info["M0"], "SHIFT": info["SHIFT"],
                    "out_zero_point": 0,
                    "folded_from_bn": node.id,
                })
                folded_bias.append(info["bias_int"])
        except ExactQuantError as error:
            raise CompileError(error.code, "BatchNormalization %s cannot be folded "
                               "exactly: %s" % (node.id, error.message), node_id=node.id)
        producer.metadata["folded_bias_values"] = folded_bias
        producer.metadata["folded_weight_scale"] = [row["weight_scale"] for row in records]
        producer.metadata["requant_records"] = records
        producer.metadata["fold_pending"] = False
        producer.metadata["folded_batch_norm"] = {
            "source_bn_node": node.id,
            "epsilon": epsilon,
            "input_scale": s_in,
            "output_scale": s_out,
            "channels": per_channel,
            "disposition": "folded_into_conv_requant",
        }
        producer.metadata["source_op_type"] = "Conv+BatchNormalization"
        producer.lowering = "conv3x3_folded_bn"
        producer.outputs[0] = node.outputs[0]
        folded[node.id] = producer.metadata["folded_batch_norm"]
        removed[node.id] = {"bn_node": node.id, "conv_node": producer.id,
                            "conv_output": conv_in, "bn_output": node.outputs[0],
                            "channels": cout}
        # The fused boundary tensor and the BN parameter tensors disappear from
        # the lowered graph; hardware planning must never allocate them.
        graph.tensors.pop(conv_in, None)
        for name in node.inputs[1:5]:
            graph.tensors.pop(name, None)
            if name in graph.initializers:
                graph.initializers.remove(name)
    if removed:
        graph.nodes = [node for node in graph.nodes if node.id not in removed]
    return {"folded": folded, "removed": removed}



def _recognize_depthwise(graph: Graph) -> List[Dict[str, Any]]:
    """Rewrite depthwise grouped source Conv nodes into the DepthwiseConv PL op.

    Shape inference already proved group == Cin == Cout == C and the frozen
    depthwise geometry; this pass re-checks the same contract from the inferred
    graph (defense in depth) and switches the operator identity consumed by
    layout/memory/codegen.  No other grouped form is accepted.
    """
    contract = graph.metadata.get("quant_contract") or {}
    contract_convs = contract.get("convs") or {}
    recognized: List[Dict[str, Any]] = []
    for node in graph.nodes:
        if node.op_type != "Conv":
            continue
        group = int(node.attrs.get("group", 1))
        row = contract_convs.get(node.id)
        explicit_depthwise = isinstance(row, dict) and bool(row.get("depthwise"))
        if group == 1 and not explicit_depthwise:
            # Dense group=1 Conv stays dense.  The C=1 depthwise geometry is
            # numerically identical to a dense 1->1 Conv, so an explicit
            # contract marker is required instead of guessing.
            continue
        if group > 1 and isinstance(row, dict) and row.get("depthwise") is False:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "contract marks Conv %s as dense but group=%d" %
                               (node.id, group), node_id=node.id, attribute="group")
        x_shape = graph.tensors[node.inputs[0]].shape
        out_shape = graph.tensors[node.outputs[0]].shape
        weight_shape = graph.tensors[node.inputs[1]].shape
        if x_shape is None or out_shape is None or weight_shape is None:
            raise CompileError(E_CFG, "depthwise Conv %s lacks static shapes" % node.id,
                               node_id=node.id)
        cin, cout = int(x_shape[1]), int(out_shape[1])
        if not (group == cin == cout):
            raise CompileError(
                E_UNSUPPORTED_ATTR,
                "grouped Conv is supported only as depthwise (group==Cin==Cout); "
                "got group=%d Cin=%d Cout=%d" % (group, cin, cout),
                node_id=node.id, attribute="group")
        if tuple(int(v) for v in weight_shape) != (cout, 1, 3, 3):
            raise CompileError(E_SHAPE_MISMATCH,
                               "depthwise weight shape %s must be [%d, 1, 3, 3]" %
                               (tuple(weight_shape), cout), node_id=node.id)
        if len(node.inputs) != 3:
            raise CompileError(E_CFG,
                               "depthwise Conv %s requires an INT32 bias" % node.id,
                               node_id=node.id)
        kernel = [int(v) for v in node.attrs.get("kernel_shape", [3, 3])]
        dilation = [int(v) for v in node.attrs.get("dilations", [1, 1])]
        strides = [int(v) for v in node.attrs.get("strides", [1, 1])]
        pads = [int(v) for v in node.attrs.get("pads", [0, 0, 0, 0])]
        if kernel != [3, 3] or dilation != [1, 1]:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "depthwise requires 3x3 kernel and dilation 1",
                               node_id=node.id)
        if len(strides) != 2 or strides[0] != strides[1] or strides[0] not in (1, 2):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "depthwise stride must be symmetric 1 or 2",
                               node_id=node.id, attribute="strides")
        if len(pads) != 4 or len(set(pads)) != 1 or pads[0] not in (0, 1):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "depthwise pad must be symmetric 0 or 1",
                               node_id=node.id, attribute="pads")
        node.op_type = "DepthwiseConv"
        node.lowering = "depthwise_conv3x3_core_v1"
        node.attrs = dict(node.attrs)
        node.metadata.update({
            "source_op_type": "Conv (depthwise)",
            "depthwise": True,
            "depthwise_group": group,
            "lowering": "depthwise_recognition",
        })
        recognized.append({
            "source_node": node.id,
            "group": group,
            "Cin": cin,
            "Cout": cout,
            "stride": strides[0],
            "pad": pads[0],
        })
    return recognized


def _constant_scalar_tensor(graph: Graph, name: str, value: int, *,
                            source_node: str) -> None:
    tensor = Tensor(
        name=name,
        dtype="int64",
        shape=(1,),
        kind="initializer",
        layout="C",
        source={"kind": "compile_time_constant", "source_node": source_node,
                "values": [int(value)]},
    )
    graph.tensors[name] = tensor
    if name not in graph.initializers:
        graph.initializers.append(name)


def lower_graph(source: Graph, *, spatial_caps=None) -> Tuple[Graph, Dict[str, Any]]:
    """Return a deterministic lowered graph and source-to-lowered mapping.

    spatial_caps is forwarded to the post-lowering shape inference so a v0.2
    spatial build keeps its raised geometry ceiling; the legacy call (None)
    keeps the historical limits exactly.
    """
    source_digest = source.digest()
    graph = Graph.from_dict(source.document())
    graph.metadata["source_graph_digest"] = source_digest

    fold = _fold_batch_norm(graph)
    folded_bn = fold["folded"]
    depthwise_recognized = _recognize_depthwise(graph)

    lowered_nodes: List[Node] = []
    mapping: List[Dict[str, Any]] = []
    numeric_constants: Dict[str, List[float]] = {}
    integer_constants: Dict[str, List[int]] = {}
    removed_nodes: List[Dict[str, Any]] = []
    for tensor_name, tensor in graph.tensors.items():
        if tensor.kind != "initializer":
            continue
        source_values = (tensor.source or {}).get("values")
        if source_values is not None:
            numeric_constants[tensor_name] = [float(v) for v in source_values]
            if tensor.dtype in ("int8", "int32", "int64"):
                integer_constants[tensor_name] = [int(v) for v in source_values]

    for node in graph.nodes:
        if node.op_type == "Constant":
            output = node.outputs[0]
            values = list(node.metadata.get("constant_values", []))
            if output not in graph.tensors:
                raise KeyError(output)
            if output not in graph.initializers:
                graph.initializers.append(output)
            numeric_constants[output] = [float(v) for v in values]
            if graph.tensors[output].dtype in ("int8", "int32", "int64"):
                integer_constants[output] = [int(v) for v in values]
            entry = {"source_node": node.id, "source_op": "Constant",
                     "lowered_nodes": [], "disposition": "folded_initializer",
                     "outputs": [output]}
            mapping.append(entry)
            removed_nodes.append(entry)
            continue

        if node.op_type == "BatchNormalization":
            removed_row = fold["removed"].get(node.id)
            if removed_row is None:  # pragma: no cover - _fold_batch_norm raises first
                raise CompileError("E_CFG", "unfolded BatchNormalization %s" % node.id,
                                   node_id=node.id)
            entry = folded_bn[node.id]
            mapping.append({"source_node": node.id, "source_op": "BatchNormalization",
                            "lowered_nodes": [],
                            "folded_into": removed_row["conv_node"],
                            "disposition": "folded_into_conv_requant",
                            "outputs": list(node.outputs),
                            "channels": len(entry["channels"]),
                            "epsilon": entry["epsilon"]})
            continue

        if node.op_type == "Conv" and list(node.attrs.get("kernel_shape", [])) == [1, 1]:
            if len(node.inputs) < 2:
                raise ValueError("Conv1x1 is missing weight")
            weight = graph.tensors[node.inputs[1]]
            if weight.shape is None or len(weight.shape) != 4:
                raise ValueError("Conv1x1 weight must be OIHW rank 4")
            cout, cin, kh, kw = weight.shape
            if (kh, kw) != (1, 1):
                raise ValueError("Conv1x1 weight shape is not 1x1")
            weight.source = dict(weight.source or {})
            weight.source.update({
                "expanded_from_onnx_initializer": node.inputs[1],
                "expanded_from_shape": [cout, cin, 1, 1],
                "expansion": "center_tap_3x3",
            })
            weight.shape = (cout, cin, 3, 3)
            node.metadata.update({
                "source_op_type": "Conv",
                "source_kernel_shape": [1, 1],
                "lowered_kernel_shape": [3, 3],
                "lowering": "conv1x1_to_3x3_center_tap",
            })
            node.attrs = dict(node.attrs)
            node.attrs["kernel_shape"] = [3, 3]
            node.attrs["pads"] = [1, 1, 1, 1]
            node.lowering = "conv1x1_to_3x3_center_tap"
            lowered_nodes.append(node)
            mapping.append({"source_node": node.id, "source_op": "Conv1x1",
                            "lowered_nodes": [node.id], "disposition": "expanded_3x3",
                            "outputs": list(node.outputs)})
            continue

        if node.op_type == "Split":
            specs = list(node.metadata.get("split_specs", []))
            if not specs:
                raise ValueError("Split lowering requires split_specs metadata")
            lowered_ids: List[str] = []
            for spec in specs:
                index = int(spec["index"])
                output = str(spec["output"])
                start = int(spec["start"])
                length = int(spec["length"])
                starts = "%s.slice%d.starts" % (node.id, index)
                ends = "%s.slice%d.ends" % (node.id, index)
                axes = "%s.slice%d.axes" % (node.id, index)
                _constant_scalar_tensor(graph, starts, start, source_node=node.id)
                _constant_scalar_tensor(graph, ends, start + length, source_node=node.id)
                _constant_scalar_tensor(graph, axes, 1, source_node=node.id)
                numeric_constants[starts] = [float(start)]
                numeric_constants[ends] = [float(start + length)]
                numeric_constants[axes] = [1.0]
                integer_constants[starts] = [start]
                integer_constants[ends] = [start + length]
                integer_constants[axes] = [1]
                lowered_id = "%s.slice%d" % (node.id, index)
                lowered = Node(
                    id=lowered_id,
                    op_type="Slice",
                    name="%s.slice%d" % (node.name, index),
                    inputs=[node.inputs[0], starts, ends, axes],
                    outputs=[output],
                    attrs={},
                    placement="pl",
                    lowering="slice_copy",
                    source_index=int(node.source_index),
                    metadata={
                        "source_node_id": node.id,
                        "source_op_type": "Split",
                        "split_output_index": index,
                        "source_output": output,
                        "lowering": "split_to_slice",
                    },
                )
                lowered_nodes.append(lowered)
                lowered_ids.append(lowered_id)
            mapping.append({"source_node": node.id, "source_op": "Split",
                            "lowered_nodes": lowered_ids, "disposition": "split_to_slices",
                            "outputs": list(node.outputs),
                            "split_lengths": [int(s["length"]) for s in specs],
                            "channel_offsets": [int(s["start"]) for s in specs]})
            continue

        lowered_nodes.append(node)
        if node.op_type == "DepthwiseConv":
            disposition = "depthwise_direct"
        elif node.metadata_only:
            disposition = "metadata_alias"
        else:
            disposition = "direct"
        mapping.append({"source_node": node.id, "source_op": node.op_type,
                        "lowered_nodes": [node.id],
                        "disposition": disposition,
                        "outputs": list(node.outputs)})

    graph.nodes = lowered_nodes
    graph.metadata["lowering_map"] = mapping
    graph.metadata["removed_nodes"] = removed_nodes
    graph.metadata["lowering_passes"] = [
        "batch_norm_exact_fold", "constant_fold", "conv1x1_center_expand",
        "static_split_to_slice", "depthwise_recognition"
    ]
    infer_graph(graph, integer_constants, numeric_constants, caps=spatial_caps)
    graph.metadata["lowering_schema"] = LOWERING_SCHEMA
    # Re-finalize so metadata added after shape inference is reflected in the digest.
    graph.finalize()
    manifest = {
        "schema": LOWERING_SCHEMA,
        "source_graph_digest": source_digest,
        "lowered_graph_digest": graph.digest(),
        "passes": list(graph.metadata["lowering_passes"]),
        "mapping": mapping,
        "source_nodes": len(source.nodes),
        "lowered_nodes": len(graph.nodes),
        "constant_nodes_folded": sum(1 for row in mapping if row["disposition"] == "folded_initializer"),
        "conv1x1_nodes_expanded": sum(1 for row in mapping if row["disposition"] == "expanded_3x3"),
        "split_nodes_lowered": sum(1 for row in mapping if row["disposition"] == "split_to_slices"),
        "batch_norm_folded": list(folded_bn.values()),
        "batch_norm_folded_count": len(folded_bn),
        "depthwise_recognized": depthwise_recognized,
        "depthwise_recognized_count": len(depthwise_recognized),
    }
    return graph, manifest
