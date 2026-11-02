# -*- coding: utf-8 -*-
"""QDQ + affine-int8-v1 -> canonical graph IR bridge for the mainline compiler.

This is the *frontend* half of the v0.2 real-YOLO compiler wiring.  It reuses
tools/onnx_gen/qdq_frontend.py (the only QDQ pattern reader) and
tools/onnx_gen/affine_contract.py (the only affine-int8-v1 parameter layer), and
converts the normalised deployment graph into the canonical graph_ir.Graph /
Tensor / Node structures that every later pass (lowering, layout, memory,
instruction codegen, parameter packing, RTL generation) already consumes.
There is no second compile pipeline: compile_pipeline.compile_graph selects this
frontend only when an explicit numeric profile is requested.

Design rules (task book S1/S2):

* no model identity assertions here: the example-specific six-head checks stay
  in the example asset checker (qdq_frontend strict mode).  This module accepts
  any graph the QDQ reader recognises and that the operator registry supports.
* every activation is stored re-encoded as signed INT8 (qs = qu - 128) and the
  affine contract attaches scale / zero points.  The legacy signed zero-point-0
  contract is untouched.
* the Conv bias is kept raw INT32: the core applies the (128 + zs_in) * sum(w)
  correction itself, so folding the bias here as well would double-correct.
  bias_folded is still exposed (for the independent reference and evidence).

Schema: v0.2-yolo-affine-frontend-1
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import affine_contract
from .affine_quant import PROFILE, u8_to_s8
from .graph_ir import CompileError, E_CFG, E_UNSUPPORTED_OP, Graph, Node, Tensor
from .op_registry import get_spec, normalize_attributes
from .qdq_frontend import parse_qdq_onnx
from .shape_infer import CHANNEL_BLOCK, ABI_ALIGNMENT, align_up

SCHEMA = "v0.2-yolo-affine-frontend-1"
#: build-time capacity keys this frontend derives from the real graph
CAPACITY_KEYS = ("max_cin", "max_cout", "max_ch", "max_w", "max_h",
                 "dense_max_in_row_bytes", "dense_max_out_row_bytes",
                 "dense_max_weight_tiles")

#: shape-inference caps for the affine build (no C<=16 / W<=32 legacy ceiling)
def shape_caps() -> Dict[str, Any]:
    return {
        "schema": "e1-affine-shape-caps-1",
        "channel_ceiling": 512,
        "width_ceiling": 640,
        "feature_max_h": 640,
        "feature_max_w": 640,
        "global_avg_max_hw": 640,
        "resize_in_max_hw": 320,
        "resize_out_max_hw": 640,
        "pool5_max_ch": 256,
        "resize_max_ch": 256,
    }


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chw(shape_chw: Sequence[int]) -> Tuple[int, int, int, int]:
    return (1, int(shape_chw[0]), int(shape_chw[1]), int(shape_chw[2]))


def _norm_attrs(op: str, dep_attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Map deployment-graph attributes onto the canonical registry attributes."""
    if op == "Conv":
        k = int(dep_attrs.get("kernel", 3))
        s = int(dep_attrs.get("stride", 1))
        p = int(dep_attrs.get("pad", 0))
        return normalize_attributes("Conv", {
            "kernel_shape": [k, k], "strides": [s, s], "pads": [p] * 4,
            "dilations": [1, 1], "group": int(dep_attrs.get("groups", 1))})
    if op == "MaxPool":
        if "kernel_shape" in dep_attrs:
            if int(dep_attrs.get("format_version", 1)) != 1:
                raise CompileError(E_CFG, "unsupported MaxPool deployment format")
            return normalize_attributes("MaxPool", {k: v for k, v in dep_attrs.items()
                                                     if k != "format_version"})
        k = int(dep_attrs.get("kernel", 2))
        s = int(dep_attrs.get("stride", 1))
        p = int(dep_attrs.get("pad", 0))
        return normalize_attributes("MaxPool", {
            "kernel_shape": [k, k], "strides": [s, s], "pads": [p] * 4,
            "ceil_mode": 0, "dilations": [1, 1], "storage_order": 0})
    if op == "Concat":
        return normalize_attributes("Concat", {"axis": int(dep_attrs.get("axis", 1))})
    if op == "Split":
        return normalize_attributes("Split", {"axis": int(dep_attrs.get("axis", 1))})
    if op == "Resize":
        return normalize_attributes("Resize", {"mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor", "antialias": 0, "exclude_outside": 0,
            "keep_aspect_ratio_policy": "stretch"})
    if op == "Add":
        return {}
    raise CompileError(E_UNSUPPORTED_OP,
                       "deployment operator %r has no canonical mapping" % op)


def _init_tensor(name: str, dtype: str, shape: Sequence[int], values: Sequence[Any]) -> Tensor:
    return Tensor(name, dtype=dtype, shape=tuple(int(v) for v in shape),
                  kind="initializer", layout="C",
                  source={"kind": "affine_initializer", "values": list(values)},
                  logical_bytes=int(np.prod([int(v) for v in shape]) *
                                    (1 if dtype in ("int8", "uint8") else 4)))


def _pack_init(name: str, dtype: str, shape: Sequence[int], array: np.ndarray) -> Dict[str, Any]:
    flat = np.ascontiguousarray(array).reshape(-1)
    values = [int(v) for v in flat] if dtype != "float32" else [float(v) for v in flat]
    return {"dtype": dtype, "shape": [int(v) for v in shape], "values": values,
            "sha256": _sha_bytes(np.ascontiguousarray(array).tobytes()),
            "array": array}


# --------------------------------------------------------------------------- #
# deployment graph -> canonical graph
# --------------------------------------------------------------------------- #
def _select_nodes(dep: Dict[str, Any], fragment: Optional[Sequence[str]]):
    nodes = list(dep["nodes"])
    if fragment is None:
        return nodes
    by_id = {str(node["id"]): node for node in nodes}
    selected = []
    for name in fragment:
        node = by_id.get(str(name))
        if node is None:
            raise CompileError(E_CFG, "fragment node %r is not in the deployment graph" % name)
        selected.append(node)
    # keep the original topological order regardless of the fragment spelling
    order = {str(node["id"]): index for index, node in enumerate(nodes)}
    selected.sort(key=lambda node: order[str(node["id"])])
    return selected


def build_canonical_graph(dep: Dict[str, Any], quant_table: Dict[str, Any],
                          weights: Dict[str, np.ndarray], *, model_sha256: Optional[str],
                          fragment: Optional[Sequence[str]] = None,
                          graph_name: Optional[str] = None,
                          input_rows: Optional[int] = None) -> Dict[str, Any]:
    """Convert a parsed QDQ deployment graph into the canonical CLI-consumed IR."""
    q_tensors = quant_table.get("tensors") or {}
    q_convs = quant_table.get("conv") or {}
    selected = _select_nodes(dep, fragment)
    if not selected:
        raise CompileError(E_CFG, "fragment selects no nodes")

    # ---- tensor bookkeeping ------------------------------------------------
    selected_ids = {str(node["id"]) for node in selected}
    produced: Dict[str, str] = {}
    for node in selected:
        nid = str(node["id"])
        produced[nid] = nid
        if node["op"] == "Split":
            for entry in node["attrs"]["outputs"]:
                produced[str(entry["name"])] = nid

    def domain_of(tensor: str) -> Tuple[float, int]:
        if tensor == "__input__":
            row = dep["input"]
            return float(row["scale"]), int(row["zero_point"])
        row = q_tensors.get(tensor)
        if row is None:
            raise CompileError(E_CFG, "no affine domain for activation %r" % tensor)
        return float(row["out_scale"]), int(row["out_zero_point"])

    def shape_of(tensor: str) -> Tuple[int, ...]:
        if tensor == "__input__":
            return _chw(dep["input"]["shape"][1:])
        if tensor in dep["tensors"]:
            return _chw(dep["tensors"][tensor]["shape"])
        raise CompileError(E_CFG, "no shape for tensor %r" % tensor)

    tensors: Dict[str, Tensor] = {}
    for node in selected:
        nid = str(node["id"])
        if node["op"] == "Split":
            for entry in node["attrs"]["outputs"]:
                name = str(entry["name"])
                tensors[name] = Tensor(name, "int8", _chw(dep["tensors"][name]["shape"]),
                                       layout="NCHW")
        else:
            tensors[nid] = Tensor(nid, "int8", _chw(node["out_shape"]), layout="NCHW")

    # ---- boundary inputs / outputs ----------------------------------------
    consumed = set()
    for node in selected:
        consumed.update(str(name) for name in node["inputs"])
    boundary: List[str] = []
    for node in selected:
        for name in node["inputs"]:
            name = str(name)
            if name == "__input__" or name in produced:
                continue
            if name in self_outputs(selected):
                continue
            if name not in boundary:
                boundary.append(name)
    graph_inputs: List[str] = []
    if "__input__" in [str(name) for node in selected for name in node["inputs"]]:
        graph_inputs.append("__input__")
    graph_inputs.extend(boundary)
    for name in graph_inputs:
        shape = shape_of(name)
        if fragment is not None and input_rows is not None and (name in boundary or name == "__input__"):
            if not (1 <= int(input_rows) <= int(shape[2])):
                raise CompileError(E_CFG, "fragment input rows %r outside 1..%d" %
                                   (input_rows, int(shape[2])))
            shape = (int(shape[0]), int(shape[1]), int(input_rows), int(shape[3]))
        tensors[name] = Tensor(name, "int8", shape, kind="input", layout="NCHW")

    # ---- fragment shape re-derivation (small-H short tasks) ----------------
    if fragment is not None and input_rows is not None:
        shape_of_local: Dict[str, Tuple[int, ...]] = {}
        for name in boundary:
            shape_of_local[name] = tensors[name].shape
        if "__input__" in tensors:
            # fragment starting at a node that consumes the graph input: seed the
            # fragment-local shape map with __input__ (same shape as tensors entry)
            shape_of_local["__input__"] = tensors["__input__"].shape
        for node in selected:
            nid = str(node["id"])
            x = shape_of_local[str(node["inputs"][0])]
            if node["op"] == "Conv":
                k = int(node["attrs"]["kernel"]); s_ = int(node["attrs"]["stride"])
                p_ = int(node["attrs"]["pad"])
                out = (1, int(node["attrs"]["cout"]),
                       (int(x[2]) + 2 * p_ - k) // s_ + 1,
                       (int(x[3]) + 2 * p_ - k) // s_ + 1)
            elif node["op"] == "MaxPool":
                from .maxpool2d_geometry import geometry
                g = geometry(x, _norm_attrs("MaxPool", node["attrs"]))
                out = (1, g.channels, g.hout, g.wout)
            elif node["op"] == "Resize":
                out = (1, int(x[1]), int(x[2]) * 2, int(x[3]) * 2)
            elif node["op"] == "Add":
                out = x
            elif node["op"] == "Concat":
                out = (1, sum(int(shape_of_local[str(name)][1]) for name in node["inputs"]),
                       int(x[2]), int(x[3]))
            elif node["op"] == "Split":
                sizes = [int(v) for v in node["attrs"]["sizes"]]
                cursor = 0
                for entry, size in zip(node["attrs"]["outputs"], sizes):
                    name = str(entry["name"])
                    shape_of_local[name] = (1, size, int(x[2]), int(x[3]))
                    tensors[name].shape = shape_of_local[name]
                    cursor += size
                continue
            else:
                raise CompileError(E_UNSUPPORTED_OP, "fragment shape rule missing for %s" %
                                   node["op"])
            shape_of_local[nid] = out
            tensors[nid].shape = out


    if fragment is None:
        graph_outputs = [str(name) for name in dep["outputs"]]
    else:
        outs: List[str] = []
        for node in selected:
            nid = str(node["id"])
            names = ([str(entry["name"]) for entry in node["attrs"]["outputs"]]
                     if node["op"] == "Split" else [nid])
            for name in names:
                if name not in consumed and name not in outs:
                    outs.append(name)
        graph_outputs = outs
    for name in graph_outputs:
        if name not in tensors:
            raise CompileError(E_CFG, "graph output %r has no tensor" % name)

    # ---- initializers ------------------------------------------------------
    initializers: Dict[str, Tensor] = {}
    pack_initializers: Dict[str, Dict[str, Any]] = {}
    init_arrays: Dict[str, np.ndarray] = {}
    for node in selected:
        nid = str(node["id"])
        if node["op"] != "Conv":
            continue
        w = np.asarray(weights[nid + ".w"], dtype=np.int8)
        b = np.asarray(weights[nid + ".b"], dtype=np.int32)
        attrs = node["attrs"]
        if tuple(w.shape) != (int(attrs["cout"]), int(attrs["cin"]),
                              int(attrs["kernel"]), int(attrs["kernel"])):
            raise CompileError(E_CFG, "Conv %s weight shape %s != declared C=%dx%d K=%d" %
                               (nid, tuple(w.shape), int(attrs["cout"]), int(attrs["cin"]),
                                int(attrs["kernel"])))
        wname, bname = nid + ".weight", nid + ".bias"
        initializers[wname] = _init_tensor(wname, "int8", w.shape, [])
        initializers[bname] = _init_tensor(bname, "int32", b.shape, [])
        init_arrays[wname] = w
        init_arrays[bname] = b
        pack_initializers[wname] = _pack_init(wname, "int8", w.shape, w)
        pack_initializers[bname] = _pack_init(bname, "int32", b.shape, b)
    for node in selected:
        nid = str(node["id"])
        if node["op"] == "Split":
            sizes = np.asarray([int(v) for v in node["attrs"]["sizes"]], dtype=np.int64)
            name = nid + ".sizes"
            initializers[name] = _init_tensor(name, "int64", sizes.shape,
                                              [int(v) for v in sizes])
            init_arrays[name] = sizes
            pack_initializers[name] = _pack_init(name, "int64", sizes.shape, sizes)
        elif node["op"] == "Resize":
            scales = np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
            name = nid + ".scales"
            initializers[name] = _init_tensor(name, "float32", scales.shape,
                                              [float(v) for v in scales])
            init_arrays[name] = scales
            pack_initializers[name] = _pack_init(name, "float32", scales.shape, scales)

    # ---- nodes -------------------------------------------------------------
    nodes: List[Node] = []
    split_sizes: Dict[str, List[int]] = {}
    for index, node in enumerate(selected):
        nid = str(node["id"])
        op = str(node["op"])
        spec = get_spec(op, opset=int(dep.get("opset", 13)) if False else 13)
        attrs = _norm_attrs(op, node["attrs"])
        metadata: Dict[str, Any] = {"affine_profile": PROFILE,
                                    "deployment_node_id": nid}
        if op == "Split":
            sizes = [int(v) for v in node["attrs"]["sizes"]]
            split_sizes[nid] = sizes
            inputs = [str(node["inputs"][0]), nid + ".sizes"]
            outputs = [str(entry["name"]) for entry in node["attrs"]["outputs"]]
            offset = 0
            specs = []
            for index, (size, out_name) in enumerate(zip(sizes, outputs)):
                specs.append({"index": index, "output": out_name,
                              "start": offset, "length": size})
                offset += size
            metadata.update({
                "split_lengths": sizes,
                "channel_offsets": [int(row["start"]) for row in specs],
                "split_specs": specs, "view_kind": "copy",
                "copy_required": True, "lowered_op": "Slice"})
        elif op == "Resize":
            inputs = [str(node["inputs"][0]), nid + ".scales"]
            metadata["resize_slots"] = {"roi": None, "scales": nid + ".scales",
                                        "sizes": None}
            metadata["control_inputs"] = [1]
            outputs = [nid]
        elif op == "Conv":
            inputs = [str(node["inputs"][0]), nid + ".weight", nid + ".bias"]
            metadata["relu"] = bool(node["attrs"].get("relu", False))
            metadata["source_kernel"] = int(node["attrs"].get("kernel", 3))
            metadata["logical_cin"] = int(node["attrs"]["cin"])
            metadata["logical_cout"] = int(node["attrs"]["cout"])
            outputs = [nid]
        else:
            inputs = [str(name) for name in node["inputs"]]
            outputs = [nid]
        placement = spec.placement if spec.placement in ("pl", "ps") else "pl"
        nodes.append(Node(id=nid, op_type=op, name=nid, inputs=inputs, outputs=outputs,
                          attrs=attrs, placement=placement, lowering=spec.lowering,
                          source_index=index, metadata=metadata))

    graph = Graph(
        name=graph_name or ("affine-" + str(dep.get("model", {}).get("name", "graph"))),
        inputs=graph_inputs, outputs=graph_outputs,
        initializers=sorted(initializers),
        tensors={**tensors, **initializers}, nodes=nodes,
        opset_imports=[{"domain": "", "version": 13}],
        layout_policy={"logical_layout": "ONNX_row_major_NCHW",
                       "target_layout": "NHWC8_A16",
                       "reshape_policy": "metadata_only_if_layout_compatible"},
        source={"format": "onnx_qdq", "model_sha256": model_sha256,
                "profile": PROFILE, "frontend_schema": SCHEMA},
        metadata={"numeric_profile": PROFILE, "frontend_schema": SCHEMA,
                  "fragment_nodes": ([str(v) for v in fragment] if fragment else None),
                  "split_sizes": split_sizes,
                  "layout_planning": "not_run", "codegen": "not_run"},
    )
    graph.finalize()

    # ---- affine contract ---------------------------------------------------
    contract_tensors: Dict[str, Dict[str, Any]] = {}
    for name, tensor in sorted(graph.tensors.items()):
        if tensor.kind == "initializer":
            continue
        scale, zu = domain_of(name)
        contract_tensors[name] = {"scale": float(scale), "zero_point": int(zu),
                                  "storage": "signed_int8_reencoded"}
    contract_convs: Dict[str, Dict[str, Any]] = {}
    for node in graph.nodes:
        if node.op_type != "Conv":
            continue
        row = q_convs.get(node.id)
        if row is None:
            raise CompileError(E_CFG, "quant table has no Conv entry for %s" % node.id)
        contract_convs[node.id] = {"w_scale": float(row["w_scale"]),
                                   "w_zero_point": int(row.get("w_zero_point", 0)),
                                   "w_dtype": "int8", "per_output_channel": False,
                                   "bias": "int32"}
    contract = {
        "schema": affine_contract.SCHEMA,
        "profile": PROFILE,
        "model_sha256": model_sha256,
        "rounding": "ties_away_from_zero",
        "requant": {"m0_bits": 31, "shift_max": 62},
        "tensors": contract_tensors,
        "convs": contract_convs,
    }
    applied = affine_contract.apply_affine_contract(graph, contract, init_arrays)
    _bridge_metadata(graph, applied)
    return {"graph": graph, "contract": contract, "applied": applied,
            "initializers": pack_initializers, "init_arrays": init_arrays,
            "fragment": list(fragment) if fragment else None,
            "graph_outputs": list(graph_outputs), "graph_inputs": list(graph_inputs)}


def self_outputs(selected: Sequence[Dict[str, Any]]) -> set:
    outs = set()
    for node in selected:
        if node["op"] == "Split":
            outs.update(str(entry["name"]) for entry in node["attrs"]["outputs"])
        else:
            outs.add(str(node["id"]))
    return outs


def _bridge_metadata(graph: Graph, applied: Dict[str, Any]) -> None:
    """Attach the legacy-consumed per-node parameter metadata.

    The canonical passes (instr_codegen, parameter_pack, layout) read these keys;
    they are *derived* from the affine contract result, never re-derived here.
    """
    for node in graph.nodes:
        if node.op_type == "Conv":
            row = applied["convs"][node.id]
            relu = bool(node.metadata.get("relu", False)) or bool(row["relu"])
            cout = int(graph.tensors[node.outputs[0]].shape[1])
            w_scale = float(row["w_scale"])
            records = []
            for oc in range(cout):
                records.append({
                    "oc": oc,
                    "weight_scale": w_scale,
                    "bias_scale": float(row["m0_shift"]["ratio_float"]) * 0.0 + 0.0,
                    "M0": int(row["m0"]), "SHIFT": int(row["shift"]),
                    "out_zero_point": int(row["zs_out"]),
                    "relu": bool(row["relu"]),
                })
            node.metadata.update({
                "affine": True,
                "affine_profile": PROFILE,
                "quant_input_scale": float(graph.tensors[node.inputs[0]].quantization["scale"]),
                "weight_scales": [w_scale] * cout,
                "requant_records": records,
                "affine_bias_folded": [int(v) for v in row["bias_folded"]],
                "affine_sum_w": [int(v) for v in row["sum_w"]],
                "x_zero_point_s8": int(row["zs_in"]),
                "out_zero_point_s8": int(row["zs_out"]),
                "relu": relu,
                "relu_threshold_s8": (int(row["zs_out"]) if relu else None),
                "acc_bound": int(row["acc_bound"]),
            })
        elif node.op_type == "Add":
            params = dict(applied["adds"][node.id])
            node.metadata.update({"affine": True, "add_requant": params})
        elif node.op_type == "Concat":
            node.metadata.update({"affine": True,
                                  "concat_quant": dict(applied["concats"][node.id])})
        elif node.op_type in ("MaxPool", "Resize", "Split"):
            node.metadata.update({"affine": True})
    for tensor in graph.tensors.values():
        if tensor.quantization is not None:
            tensor.quantization["affine"] = True
            tensor.quantization["zero_point"] = 0  # legacy code sees the signed re-encode


# --------------------------------------------------------------------------- #
# build capacity derivation (real per-engine rows, never MAX_C*MAX_W)
# --------------------------------------------------------------------------- #
def derive_build_capacity(graph: Graph) -> Dict[str, int]:
    """Per-engine capacity from the real lowered graph.

    The dense row budget is the *largest real row span* of the graph, not
    align8(MAX_CIN)*MAX_W; the weight face window is the largest real
    (ICB*OCB) tile product, not the global product (GAP-18/GAP-22).
    """
    def al8(v: int) -> int:
        return ((int(v) + 7) // 8) * 8

    def al16(v: int) -> int:
        return ((int(v) + 15) // 16) * 16

    convs = [node for node in graph.nodes if node.op_type == "Conv"]
    max_cin = max([int(graph.tensors[n.inputs[0]].shape[1]) for n in convs] or [1])
    max_cout = max([int(graph.tensors[n.outputs[0]].shape[1]) for n in convs] or [1])
    max_in_row = 0
    max_out_row = 0
    max_tiles = 1
    max_w = 1
    max_h = 1
    for node in convs:
        xs = graph.tensors[node.inputs[0]].shape
        os_ = graph.tensors[node.outputs[0]].shape
        cin, cout = int(xs[1]), int(os_[1])
        k = int(node.attrs["kernel_shape"][0])
        stride = int(node.attrs["strides"][0])
        w_in = int(xs[3])
        w_out = int(os_[3])
        max_in_row = max(max_in_row, al16(al8(cin) * w_in))
        max_out_row = max(max_out_row, al16(al8(cout) * w_out))
        max_tiles = max(max_tiles, (al8(cin) // 8) * (al8(cout) // 8))
        max_w = max(max_w, w_in)
        max_h = max(max_h, int(xs[2]))
    rank4 = [t for t in graph.tensors.values()
             if t.dtype == "int8" and t.shape is not None and len(t.shape) == 4
             and t.kind != "initializer"]
    max_ch = max([al8(int(t.shape[1])) for t in rank4] or [1])
    max_w = max(max_w, max([int(t.shape[3]) for t in rank4] or [1]))
    max_h = max(max_h, max([int(t.shape[2]) for t in rank4] or [1]))

    # Real per-node row extents of the feature-fusion families.  The storage
    # requirement of one engine row is align16(align8(C) * W) of one real
    # node input/output tensor; the product MAX_CH*MAX_W (or any whole-image
    # product) is never a real row and must never size a buffer/budget.
    def row_bytes(name: str) -> int:
        tensor = graph.tensors.get(name)
        if tensor is None or tensor.shape is None or len(tensor.shape) != 4:
            return 0
        return al16(al8(int(tensor.shape[1])) * int(tensor.shape[3]))

    def family_rows(nodes: Sequence[Node]) -> Tuple[int, int]:
        inputs = [row_bytes(name) for node in nodes for name in node.inputs]
        outputs = [row_bytes(name) for node in nodes for name in node.outputs]
        return max(inputs or [0]), max(outputs or [0])

    copy_nodes = [n for n in graph.nodes if n.op_type in ("Pad", "Slice", "Concat", "Split")]
    resize_nodes = [n for n in graph.nodes if n.op_type == "Resize"]
    pool5_nodes = [n for n in graph.nodes
                   if n.op_type == "MaxPool"
                   and n.lowering == "maxpool5x5_core_v1"]
    generic_nodes = [n for n in graph.nodes if n.op_type == "MaxPool"
                     and "legacy_layer_index" not in n.metadata
                     and n.lowering != "maxpool5x5_core_v1"]
    from .maxpool2d_geometry import capacity as pool_capacity, geometry
    pool_caps = (pool_capacity([geometry(graph.tensors[n.inputs[0]].shape, n.attrs)
                               for n in generic_nodes]) if generic_nodes else None)
    copy_in_row, copy_out_row = family_rows(copy_nodes)
    resize_in_row, resize_out_row = family_rows(resize_nodes)
    pool5_in_row, pool5_out_row = family_rows(pool5_nodes)
    return {
        "max_cin": max_cin, "max_cout": max_cout, "max_ch": max_ch,
        "max_w": max_w, "max_h": max_h,
        "dense_max_in_row_bytes": int(max_in_row),
        "dense_max_out_row_bytes": int(max_out_row),
        "dense_max_weight_tiles": int(max_tiles),
        "copy_in_row_bytes": int(copy_in_row), "copy_out_row_bytes": int(copy_out_row),
        "resize_in_row_bytes": int(resize_in_row),
        "resize_out_row_bytes": int(resize_out_row),
        "pool5_in_row_bytes": int(pool5_in_row),
        "pool5_out_row_bytes": int(pool5_out_row),
        "pool2d_max_in_row_bytes": int((pool_caps or {}).get("max_in_row_bytes", 0)),
        "pool2d_max_out_row_bytes": int((pool_caps or {}).get("max_out_row_bytes", 0)),
        "pool2d_max_eh": int((pool_caps or {}).get("max_eh", 0)),
        "pool2d_cache_rows": int((pool_caps or {}).get("cache_rows", 0)),
        "pool2d_logical_storage_bytes": int((pool_caps or {}).get("logical_storage_bytes", 0)),
    }


def derive_engine_capacity(graph: Graph, capacity: Dict[str, int]) -> Dict[str, Any]:
    """Per-engine capability rows in the shape verilog_codegen/route expect."""
    def hwc(names: Sequence[str]) -> Tuple[int, int, int]:
        h = w = c = 1
        for name in names:
            tensor = graph.tensors.get(name)
            if tensor is None or tensor.shape is None or len(tensor.shape) != 4:
                continue
            h = max(h, int(tensor.shape[2]))
            w = max(w, int(tensor.shape[3]))
            c = max(c, int(tensor.shape[1]))
        return h, w, c

    engines: Dict[str, Any] = {}
    used: Dict[str, bool] = {}
    for node in graph.nodes:
        used[node.op_type] = True
    # copy family: Pad/Slice/Concat (Split is materialised to Slice copies)
    copy_nodes = [n for n in graph.nodes
                  if n.op_type in ("Pad", "Slice", "Concat", "Split")]
    copy_names: List[str] = []
    for node in copy_nodes:
        copy_names.extend(list(node.inputs) + list(node.outputs))
    ch, cw, cc = hwc(copy_names)
    copy_in_row = int(capacity.get("copy_in_row_bytes", 0))
    copy_out_row = int(capacity.get("copy_out_row_bytes", 0))
    engines["copy"] = {
        "max_h": ch, "max_w": cw, "max_ch": int(cc),
        "in_row_bytes": copy_in_row, "out_row_bytes": copy_out_row,
        "max_row_bytes": int(max(copy_in_row, copy_out_row)),
        "row_capacity_rule": "max over real Pad/Slice/Concat in+out rows of "
                             "align16(align8(C)*W); never MAX_CH*MAX_W",
        "used_by_graph": bool(copy_nodes), "expanded": True, "streaming": False}
    # resize: the line buffer is sized by the *input* row; the x2 output row is
    # a separate real extent (power-of-two width only where the graph asks for it)
    rs = [n for n in graph.nodes if n.op_type == "Resize"]
    rh, rw, rc = hwc([n.inputs[0] for n in rs])
    out_h, out_w, out_c = hwc([n.outputs[0] for n in rs])
    engines["resize"] = {"max_h": rh, "max_w": rw,
                         "max_ch": int(max(rc, out_c)),
                         "out_max_h": out_h, "out_max_w": out_w,
                         "in_row_bytes": int(capacity.get("resize_in_row_bytes", 0)),
                         "out_row_bytes": int(capacity.get("resize_out_row_bytes", 0)),
                         "max_row_bytes": int(capacity.get("resize_in_row_bytes", 0)),
                         "row_capacity_rule": "input row align16(align8(C)*W) "
                                              "line buffer + x2 output row extent",
                         "used_by_graph": bool(rs), "expanded": True,
                         "streaming": False,
                         "input_axis_limit": capacity.get("resize_in_max_hw", 320)}
    # pool5: input/output shape identical (5x5 s1 p2), one real row extent
    p5 = [n for n in graph.nodes
          if n.op_type == "MaxPool"
          and n.lowering == "maxpool5x5_core_v1"]
    ph, pw, pc = hwc([t for n in p5 for t in (n.inputs[0], n.outputs[0])])
    engines["pool5"] = {"max_h": ph, "max_w": pw, "max_ch": ((int(pc) + 7) // 8) * 8,
                        "in_row_bytes": int(capacity.get("pool5_in_row_bytes", 0)),
                        "out_row_bytes": int(capacity.get("pool5_out_row_bytes", 0)),
                        "max_row_bytes": int(max(capacity.get("pool5_in_row_bytes", 0),
                                                 capacity.get("pool5_out_row_bytes", 0))),
                        "row_capacity_rule": "real align16(align8(C)*W) row",
                        "used_by_graph": bool(p5), "expanded": True,
                        "streaming": False}
    p2d = [n for n in graph.nodes if n.op_type == "MaxPool"
           and "legacy_layer_index" not in n.metadata and n.lowering != "maxpool5x5_core_v1"]
    gh, gw, gc = hwc([t for n in p2d for t in (n.inputs[0], n.outputs[0])])
    pcaps = {"max_in_row_bytes": capacity.get("pool2d_max_in_row_bytes", 0),
             "max_out_row_bytes": capacity.get("pool2d_max_out_row_bytes", 0),
             "max_eh": capacity.get("pool2d_max_eh", 0),
             "cache_rows": capacity.get("pool2d_cache_rows", 0),
             "logical_storage_bytes": capacity.get("pool2d_logical_storage_bytes", 0)}
    engines["maxpool2d"] = {"used_by_graph": bool(p2d), "expanded": True,
                            "streaming": False, "max_h": gh, "max_w": gw, "max_ch": gc,
                            "in_row_bytes": int(pcaps.get("max_in_row_bytes", 0)),
                            "out_row_bytes": int(pcaps.get("max_out_row_bytes", 0)),
                            "max_eh": int(pcaps.get("max_eh", 0)),
                            "cache_rows": int(pcaps.get("cache_rows", 0)),
                            "logical_storage_bytes": int(pcaps.get("logical_storage_bytes", 0))}
    # eltwise: streaming, no row buffer; MAX_CH only
    ew = [n for n in graph.nodes if n.op_type in ("Relu", "Add", "Clip", "Mul")]
    eh, ecc, ecw = hwc([t for n in ew for t in (n.inputs[0],) + tuple(n.outputs)])
    engines["eltwise"] = {"max_h": eh, "max_w": ecw, "max_ch": int(ecc),
                          "used_by_graph": bool(ew), "expanded": True,
                          "streaming": True}
    engines["dense"] = {"max_cin": int(capacity["max_cin"]),
                        "max_cout": int(capacity["max_cout"]),
                        "max_w": int(capacity["max_w"]),
                        "max_h": int(capacity["max_h"]),
                        "used_by_graph": bool(convs(graph)), "expanded": True}
    for name in ("pool2x2", "reduce", "activation", "depthwise", "int8_mul"):
        engines[name] = {"used_by_graph": False, "expanded": False,
                         "reason": "not used by the affine graph",
                         "max_w": 32, "max_h": 0, "max_ch": 8}
    return engines


def check_affine_row_budget(capacity: Dict[str, int],
                            engines: Dict[str, Any]) -> Dict[str, Any]:
    """Static 16-bit row-field budget check for the affine build (GAP-20).

    Nothing is simulated here; an over-wide legal combination must be rejected
    at compile time instead of silently truncating a row stride in the RTL.
    """
    def al16(v: int) -> int:
        return ((int(v) + 15) // 16) * 16

    def check(name: str, value: int, *, minimum: int = 16) -> None:
        if value > 0xFFFF:
            raise CompileError(E_CFG, "affine build %s=%d exceeds the 16-bit row field"
                               % (name, value), detail={"engine": name, "bytes": value})
        if value < minimum:
            raise CompileError(E_CFG, "affine build %s=%d is below one AXI beat"
                               % (name, value), detail={"engine": name, "bytes": value})

    rows: Dict[str, Any] = {}
    in_row = int(capacity["dense_max_in_row_bytes"])
    out_row = int(capacity["dense_max_out_row_bytes"])
    dense = engines.get("dense") or {}
    if dense.get("used_by_graph"):
        check("dense_in_row_bytes", in_row)
        check("dense_out_row_bytes", out_row)
        rows["dense"] = {"in_row_bytes": in_row, "out_row_bytes": out_row,
                         "in_row_beats": in_row // 16, "out_row_beats": out_row // 16,
                         "max_weight_tiles": int(capacity["dense_max_weight_tiles"])}
    else:
        rows["dense"] = {"in_row_bytes": 0, "out_row_bytes": 0,
                         "in_row_beats": 0, "out_row_beats": 0,
                         "max_weight_tiles": 1}
    # The feature-fusion families are budgeted by the real per-node row extents
    # derived from the graph (never by a MAX_CH*MAX_W product): one staging row
    # per input side and per output side is all the RTL allocates.
    for name in ("copy", "resize", "pool5", "maxpool2d"):
        row = engines.get(name) or {}
        if not row.get("used_by_graph"):
            continue
        in_row = int(row.get("in_row_bytes", 0))
        out_row = int(row.get("out_row_bytes", 0))
        check("%s_in_row_bytes" % name, in_row)
        check("%s_out_row_bytes" % name, out_row)
        rows[name] = {"in_row_bytes": in_row, "in_row_beats": in_row // 16,
                      "out_row_bytes": out_row, "out_row_beats": out_row // 16,
                      "max_h": int(row.get("max_h", 1)),
                      "max_w": int(row.get("max_w", 1)),
                      "max_ch": int(row.get("max_ch", 1))}
        if name == "maxpool2d":
            rows[name].update({"max_eh": row["max_eh"], "cache_rows": row["cache_rows"],
                               "logical_storage_bytes": row["logical_storage_bytes"]})
        if name == "resize":
            out_h = int(row.get("out_max_h", 0))
            out_w = int(row.get("out_max_w", 0))
            rows[name].update({"out_max_h": out_h, "out_max_w": out_w})
            if out_h > 640 or out_w > 640:
                raise CompileError(E_CFG,
                                   "affine Resize output %dx%d exceeds 640" %
                                   (out_h, out_w), detail={"engine": "resize"})
    for name in ("dense", "copy", "resize", "pool5", "maxpool2d", "eltwise"):
        row = engines.get(name) or {}
        if not row.get("used_by_graph"):
            continue
        for key in ("max_h", "max_w"):
            value = int(row.get(key) or 0)
            if value > 640:
                raise CompileError(E_CFG, "affine engine %s %s=%d exceeds 640" %
                                   (name, key, value), detail={"engine": name})
    return rows


def convs(graph: Graph) -> List[Node]:
    return [node for node in graph.nodes if node.op_type == "Conv"]


# --------------------------------------------------------------------------- #
# independent integer reference on the canonical graph
# --------------------------------------------------------------------------- #
class _ArrView:
    """Cheap numpy-backed sequence: pack_activation/comparisons never copy to a list."""

    __slots__ = ("a",)

    def __init__(self, array):
        self.a = np.asarray(array)

    def __len__(self):
        return int(self.a.size)

    def __getitem__(self, index):
        return int(self.a.reshape(-1)[index])

    def __eq__(self, other):
        other = other.a if isinstance(other, _ArrView) else np.asarray(other)
        return bool(np.array_equal(self.a, other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def tolist(self):
        return self.a.tolist()


class CanonicalTensorValue:
    """Minimal TensorValue-compatible view over a numpy CHW int8 array."""

    __slots__ = ("shape", "dtype", "numpy")

    def __init__(self, array):
        self.numpy = np.ascontiguousarray(np.asarray(array, dtype=np.int8))
        self.shape = (1,) + tuple(int(v) for v in self.numpy.shape)
        self.dtype = "int8"

    @property
    def values(self):
        return _ArrView(self.numpy)


def _expand_1x1(weight: np.ndarray, k_lowered: int, k_source: int) -> np.ndarray:
    if k_lowered == k_source:
        return np.asarray(weight)
    if (k_source, k_lowered) != (1, 3):
        raise CompileError(E_CFG, "weight expansion %dx%d is not supported" %
                           (k_source, k_lowered))
    cout, cin = int(weight.shape[0]), int(weight.shape[1])
    out = np.zeros((cout, cin, 3, 3), dtype=np.int8)
    out[:, :, 1, 1] = np.asarray(weight, dtype=np.int8).reshape(cout, cin)
    return out


def canonical_to_ref_graph(graph: Graph, initializers: Dict[str, Dict[str, Any]],
                           *, name: str = "affine_ref") -> Dict[str, Any]:
    """Serialise a canonical (source or lowered) graph into the deployment schema.

    The independent numpy IntegerReference then becomes the single arithmetic
    source for golden values on both the source and lowered graph; no second
    integer math implementation is added.
    """
    ref_weights: Dict[str, np.ndarray] = {}
    ref_nodes: List[Dict[str, Any]] = []
    ref_tensors: Dict[str, Any] = {}
    split_members: Dict[str, List[Node]] = {}
    for node in graph.nodes:
        if node.op_type == "Slice" and node.metadata.get("source_node_id"):
            split_members.setdefault(str(node.metadata["source_node_id"]), []).append(node)
    emitted_splits = set()
    for node in graph.nodes:
        nid = node.id
        out_shape = list(graph.tensors[node.outputs[0]].shape)
        out_shape = out_shape[1:]
        out_q = graph.tensors[node.outputs[0]].quantization or {}
        in_q = [graph.tensors[name].quantization or {} for name in node.inputs]
        if node.op_type == "Conv":
            w = initializers[node.inputs[1]]["array"] if "array" in initializers[node.inputs[1]] \
                else np.asarray(initializers[node.inputs[1]]["values"], dtype=np.int8)
            b = initializers[node.inputs[2]]["array"] if "array" in initializers[node.inputs[2]] \
                else np.asarray(initializers[node.inputs[2]]["values"], dtype=np.int32)
            k_low = int(node.attrs["kernel_shape"][0])
            k_src = int(node.metadata.get("source_kernel", k_low))
            w = _expand_1x1(np.asarray(w, dtype=np.int8), k_low, k_src)
            ref_weights[nid + ".w"] = np.ascontiguousarray(w)
            ref_weights[nid + ".b"] = np.ascontiguousarray(b)
            ref_nodes.append({
                "id": nid, "op": "Conv", "inputs": [node.inputs[0]],
                "attrs": {"kernel": k_low, "stride": int(node.attrs["strides"][0]),
                          "pad": int(node.attrs["pads"][0]),
                          "groups": int(node.attrs.get("group", 1)),
                          "relu": bool(node.metadata.get("relu", False)),
                          "cin": int(w.shape[1]), "cout": int(w.shape[0])},
                "out_shape": out_shape,
                "quant": {"out_scale": float(out_q["scale"]),
                          "out_zero_points": [int(out_q["zero_point_u8"])],
                          "in_scales": [float(in_q[0]["scale"])],
                          "in_zero_points": [int(in_q[0]["zero_point_u8"])],
                          "w_scale": float(node.metadata["weight_scales"][0])},
            })
        elif node.op_type in ("Add", "Concat"):
            ref_nodes.append({
                "id": nid, "op": node.op_type,
                "inputs": [name for name in node.inputs if name in graph.tensors
                           and graph.tensors[name].kind != "initializer"],
                "attrs": ({"axis": 1} if node.op_type == "Concat" else {}),
                "out_shape": out_shape,
                "quant": {"out_scale": float(out_q["scale"]),
                          "out_zero_points": [int(out_q["zero_point_u8"])],
                          "in_scales": [float(q["scale"]) for q in in_q
                                        if q and "scale" in q],
                          "in_zero_points": [int(q["zero_point_u8"]) for q in in_q
                                             if q and "zero_point_u8" in q]},
            })
        elif node.op_type in ("MaxPool", "Resize"):
            ref_nodes.append({
                "id": nid, "op": node.op_type, "inputs": [node.inputs[0]],
                "attrs": ({"format_version": 1,
                           "kernel_shape": list(node.attrs["kernel_shape"]),
                           "strides": list(node.attrs["strides"]),
                           "pads": list(node.attrs["pads"]),
                           "dilations": list(node.attrs["dilations"])}
                          if node.op_type == "MaxPool" else
                          {"mode": "nearest", "scale": 2}),
                "out_shape": out_shape,
                "quant": {"out_scale": float(out_q["scale"]),
                          "out_zero_points": [int(out_q["zero_point_u8"])],
                          "in_scales": [float(in_q[0]["scale"])],
                          "in_zero_points": [int(in_q[0]["zero_point_u8"])]},
            })
        elif node.op_type == "Split":
            sizes = [int(v) for v in
                     initializers[node.inputs[1]]["values"]]
            offset = 0
            outs = []
            for index, name in enumerate(node.outputs):
                outs.append({"name": name, "channel_offset": offset,
                             "channels": sizes[index]})
                offset += sizes[index]
            ref_nodes.append({
                "id": nid, "op": "Split", "inputs": [node.inputs[0]],
                "attrs": {"sizes": sizes, "axis": 1, "outputs": outs},
                "out_shape": list(graph.tensors[node.inputs[0]].shape[1:]),
                "quant": {"out_scale": float(out_q["scale"]),
                          "out_zero_points": [int(out_q["zero_point_u8"])],
                          "in_scales": [float(in_q[0]["scale"])],
                          "in_zero_points": [int(in_q[0]["zero_point_u8"])]},
            })
            for index, name in enumerate(node.outputs):
                ref_tensors[name] = {"shape": [sizes[index],
                                               int(graph.tensors[name].shape[2]),
                                               int(graph.tensors[name].shape[3])]}
            continue
        elif node.op_type == "Slice" and node.metadata.get("source_node_id"):
            src = str(node.metadata["source_node_id"])
            if src in emitted_splits:
                continue
            emitted_splits.add(src)
            members = split_members[src]
            spans = []
            for member in members:
                starts = [int(v) for v in member.metadata["starts"]]
                ends = [int(v) for v in member.metadata["ends"]]
                spans.append((starts[1], ends[1] - starts[1], member.outputs[0]))
            spans.sort()
            ref_nodes.append({
                "id": src, "op": "Split", "inputs": [members[0].inputs[0]],
                "attrs": {"sizes": [span[1] for span in spans], "axis": 1,
                          "outputs": [{"name": span[2], "channel_offset": span[0],
                                       "channels": span[1]} for span in spans]},
                "out_shape": list(graph.tensors[members[0].inputs[0]].shape[1:]),
                "quant": {"out_scale": float(out_q["scale"]),
                          "out_zero_points": [int(out_q["zero_point_u8"])],
                          "in_scales": [float(in_q[0]["scale"])],
                          "in_zero_points": [int(in_q[0]["zero_point_u8"])]},
            })
            for span in spans:
                ref_tensors[span[2]] = {"shape": [span[1], int(out_shape[1]),
                                                  int(out_shape[2])],
                                        "local_c": span[1]}
        else:
            raise CompileError(E_UNSUPPORTED_OP,
                               "canonical reference does not implement %s" % node.op_type,
                               node_id=nid)
        if node.op_type != "Slice":
            ref_tensors[nid] = {"shape": out_shape}
        for name in node.outputs:
            if name not in ref_tensors and name != nid:
                ref_tensors[name] = {"shape": list(graph.tensors[name].shape[1:])}
    # graph input domain
    input_name = graph.inputs[0]
    in_q = graph.tensors[input_name].quantization
    ref_graph = {
        "schema": "v0.2-yolo-graph-1", "profile": PROFILE,
        "model": {"name": name},
        "input": {"name": input_name, "dtype": "uint8",
                  "shape": [int(v) for v in graph.tensors[input_name].shape],
                  "scale": float(in_q["scale"]),
                  "zero_point": int(in_q["zero_point_u8"])},
        "outputs": list(graph.outputs), "nodes": ref_nodes, "tensors": ref_tensors,
    }
    ref_quant = {"schema": "v0.2-yolo-quant-table-1", "profile": PROFILE,
                 "rounding": "ties_away_from_zero", "tensors": {}, "conv": {}}
    for tensor_name, tensor in graph.tensors.items():
        q = tensor.quantization or {}
        if tensor.kind == "initializer" or "scale" not in q:
            continue
        ref_quant["tensors"][tensor_name] = {
            "out_scale": float(q["scale"]),
            "out_zero_point": int(q["zero_point_u8"]),
            "logical_c": int(tensor.shape[1]) if len(tensor.shape) == 4 else int(tensor.shape[0]),
        }
    for node in graph.nodes:
        if node.op_type == "Conv":
            ref_quant["conv"][node.id] = {
                "w_scale": float(node.metadata["weight_scales"][0]),
                "w_zero_point": 0,
                "sum_w": [int(v) for v in node.metadata.get("affine_sum_w", [])],
                "acc_bound": int(node.metadata.get("acc_bound", 0)),
                "k_size": int(graph.tensors[node.inputs[1]].shape[1]) *
                          int(graph.tensors[node.inputs[1]].shape[2]) *
                          int(graph.tensors[node.inputs[1]].shape[3]),
            }
    return {"graph": ref_graph, "quant": ref_quant, "weights": ref_weights}


def run_canonical_reference(graph: Graph, initializers: Dict[str, Dict[str, Any]],
                            *, seed_tensors: Optional[Dict[str, np.ndarray]] = None,
                            input_u8: Optional[np.ndarray] = None,
                            collect_stats: bool = False) -> Dict[str, Any]:
    """Run the independent IntegerReference on a canonical graph."""
    from .int_reference import IntegerReference
    ref = canonical_to_ref_graph(graph, initializers)
    engine = IntegerReference(ref["graph"], ref["quant"], ref["weights"])
    if seed_tensors is not None:
        result = engine.run(initial=seed_tensors, collect_stats=collect_stats)
    else:
        result = engine.run(input_u8=input_u8, collect_stats=collect_stats)
    return result


def reference_values(graph: Graph, result: Dict[str, Any]) -> Dict[str, Any]:
    values = {}
    for tensor_name, tensor in graph.tensors.items():
        arr = result["tensors"].get(tensor_name)
        if arr is None:
            continue
        value = CanonicalTensorValue(arr)
        if tensor.shape is not None and tuple(tensor.shape) != value.shape:
            raise CompileError(E_CFG, "reference shape %s != graph shape %s for %s" %
                               (value.shape, tensor.shape, tensor_name))
        values[tensor_name] = value
    return {"values": values, "node_outputs": {}}


# --------------------------------------------------------------------------- #
# input manifests for the affine build
# --------------------------------------------------------------------------- #
def write_input_manifest(out_path: Path, inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Write an e1-input-manifest-1 for signed-INT8 NCHW seed tensors."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = out_path.parent / (out_path.stem + "_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    rows: Dict[str, Any] = {}
    for name, array in inputs.items():
        arr = np.ascontiguousarray(np.asarray(array, dtype=np.int8))
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4 or arr.shape[0] != 1:
            raise CompileError(E_CFG, "input %s must be N=1 NCHW" % name)
        blob = arr.reshape(-1).tobytes()
        path = data_dir / (name.replace("/", "_") + ".bin")
        path.write_bytes(blob)
        rows[name] = {"path": str(path.resolve().relative_to(out_path.parent.resolve())).replace("\\", "/"),
                      "shape": [int(v) for v in arr.shape], "format": "nchw_int8",
                      "sha256": _sha_bytes(blob)}
    doc = {"schema": "e1-input-manifest-1", "model_sha256": None, "inputs": rows}
    out_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


def prepare(model_path: Path, *, fragment: Optional[Sequence[str]] = None,
            input_u8: Optional[np.ndarray] = None,
            manifest_path: Optional[Path] = None,
            input_rows: Optional[int] = None) -> Dict[str, Any]:
    """Parse the QDQ model and return everything compile_pipeline needs for the affine path."""
    parsed = parse_qdq_onnx(Path(model_path), strict_counts=False)
    dep = parsed["graph"]
    report = parsed.get("report") or {}
    model_sha = (report.get("onnx") or {}).get("sha256")
    built = build_canonical_graph(dep, parsed["quant"], parsed["weights"],
                                  model_sha256=model_sha, fragment=fragment,
                                  input_rows=input_rows)
    graph = built["graph"]
    # boundary seed tensors / runtime input data
    seeds: Dict[str, np.ndarray] = {}
    input_image = None
    if fragment is not None:
        # Run the independent reference once on the full deployment graph and
        # seed the fragment's boundary inputs from its intermediates.
        from .int_reference import IntegerReference
        engine = IntegerReference(dep, parsed["quant"], parsed["weights"])
        full = engine.run(input_u8=input_u8)
        for name in graph.inputs:
            arr = full["tensors"].get(name)
            if arr is None:
                raise CompileError(E_CFG, "fragment boundary input %r was not produced "
                                          "by the integer reference" % name)
            import numpy as _np
            arr = _np.ascontiguousarray(arr)
            in_tensor = graph.tensors[name]
            if arr.ndim == 3 and (int(input_rows) if input_rows else arr.shape[1]) != arr.shape[1]:
                arr = arr[:, :int(input_rows), :]
            if arr.ndim == 3 and tuple(arr.shape) != tuple(in_tensor.shape[1:]):
                raise CompileError(E_CFG, "fragment seed shape %s does not match graph input %s" %
                                   (arr.shape, in_tensor.shape))
            seeds[name] = arr
        initial_values = {name: seeds[name] for name in graph.inputs}
    else:
        # The deployment graph input file carries signed-INT8 codes (qs = qu - 128).
        # The RTL activation storage is the signed two's-complement byte, so the
        # input manifest (and hence ddr_init) must be recoded here, not left as raw
        # uint8 codes.
        initial_values = {"__input__": u8_to_s8(np.asarray(input_u8, dtype=np.uint8))}
    built["seed_tensors"] = seeds
    built["input_values"] = initial_values
    if fragment is not None:
        built["reference"] = run_canonical_reference(graph, built["initializers"],
                                                     seed_tensors=seeds)
    else:
        built["reference"] = run_canonical_reference(
            graph, built["initializers"],
            input_u8=np.asarray(input_u8, dtype=np.uint8))
    built["model_sha256"] = model_sha
    built["report"] = report
    return built
