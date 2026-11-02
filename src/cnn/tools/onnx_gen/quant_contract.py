"""Explicit INT8 quantization contract for the E1-P2 compiler.

No implicit scale, zero-point or weight quantization is invented. Every
runtime activation must be named in the contract and every Conv must provide a
per-output-channel weight scale. The ABI layout's requant M0/SHIFT is derived
here and frozen into the compile manifest.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Tuple

from .exact_quant import ExactQuantError, exact, ratio_to_m0_shift
from .graph_ir import CompileError, E_CFG, E_QUANT_MISMATCH, Graph

SCHEMA = "e1-quant-contract-1"
MAX_M0 = 0xFFFFFFFF


def _positive_float(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CompileError(E_QUANT_MISMATCH, "%s must be numeric" % what)
    if not math.isfinite(number):
        raise CompileError(E_QUANT_MISMATCH, "%s must be finite" % what)
    if number <= 0.0:
        raise CompileError(E_QUANT_MISMATCH, "%s must be positive" % what)
    return number


def _zero_point(value: Any, what: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise CompileError(E_QUANT_MISMATCH, "%s must be integer" % what)
    if number != 0:
        raise CompileError(E_QUANT_MISMATCH,
                           "%s must be 0 for the current signed INT8 ABI" % what)
    return 0


def multiplier_to_m0_shift(multiplier: float) -> Tuple[int, int]:
    """Choose deterministic M0/SHIFT for acc*multiplier."""
    multiplier = _positive_float(multiplier, "requant multiplier")
    for shift in range(30, -1, -1):
        m0 = int(round(multiplier * (1 << shift)))
        if 1 <= m0 <= MAX_M0:
            return m0, shift
    raise CompileError(E_QUANT_MISMATCH,
                       "requant multiplier %.17g cannot be represented by M0/SHIFT" %
                       multiplier)


def _is_compile_time_tensor(graph: Graph, name: str) -> bool:
    tensor = graph.tensors[name]
    if tensor.kind == "initializer":
        return True
    return (tensor.source or {}).get("kind") == "compile_time_constant"


def _resize_control_tensors(graph: Graph) -> set:
    """roi/scales/sizes are compile-time control metadata, never activations."""
    controls = set()
    for node in graph.nodes:
        if node.op_type != "Resize":
            continue
        slots = node.metadata.get("resize_slots")
        if not isinstance(slots, dict):
            continue
        for key in ("roi", "scales", "sizes"):
            name = slots.get(key)
            if name:
                controls.add(str(name))
    return controls


def _identity_quant(graph: Graph, node, what: str, *, allow_nonzero: bool = False) -> Dict[str, Any]:
    """Enforce one explicit INT8 zero-point-0 scale for input and output.

    Resize only copies bytes and Pool5 only selects bytes, so no requantization
    record exists and both sides must name the same quantization point.
    """
    quants = []
    for name in (node.inputs[0], node.outputs[0]):
        tensor = graph.tensors[name]
        if tensor.dtype != "int8":
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires INT8 input/output; %s is %s" %
                               (what, name, tensor.dtype), node_id=node.id)
        quant = tensor.quantization or {}
        if quant.get("materialized") is False or "scale" not in quant:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s requires a materialized INT8 quantization entry for %s" %
                               (what, name), node_id=node.id)
        zero = int(quant.get("zero_point", 0))
        if zero != 0 and not allow_nonzero:
            raise CompileError(E_QUANT_MISMATCH,
                               "%s %s must use zero_point 0, got %d" % (what, name, zero),
                               node_id=node.id)
        quants.append(quant)
    if (quants[0]["scale"] != quants[1]["scale"] or
            int(quants[0].get("zero_point", 0)) != int(quants[1].get("zero_point", 0))):
        raise CompileError(
            E_QUANT_MISMATCH,
            "%s requires equal input/output scales (identity arithmetic); "
            "got %.17g and %.17g" %
            (what, quants[0]["scale"], quants[1]["scale"]), node_id=node.id)
    return {"scale": quants[0]["scale"], "zero_point": int(quants[0].get("zero_point", 0)),
            "identity": True}


def apply_contract(graph: Graph, contract: Dict[str, Any]) -> Dict[str, Any]:
    if contract.get("schema") != SCHEMA:
        raise CompileError(E_CFG, "unsupported quant contract schema %r" % contract.get("schema"))
    expected_model = contract.get("model_sha256")
    actual_model = (graph.source or {}).get("model_sha256")
    if expected_model is not None and expected_model != actual_model:
        raise CompileError(E_CFG, "quant contract model_sha256 does not match graph source")
    entries = contract.get("tensors")
    if not isinstance(entries, dict):
        raise CompileError(E_CFG, "quant contract tensors must be an object")
    resize_controls = _resize_control_tensors(graph)
    # Nonzero signed zero points are permitted only where every touching node
    # is generic MaxPool; this does not relax other legacy engine contracts.
    pool_names = set()
    non_pool_names = set()
    for node in graph.nodes:
        names = set(node.inputs) | set(node.outputs)
        if node.op_type == "MaxPool" and "legacy_layer_index" not in node.metadata:
            pool_names.update(names)
        else:
            non_pool_names.update(names)
    nonzero_pool_names = pool_names - non_pool_names
    applied = {}
    for name, tensor in sorted(graph.tensors.items()):
        if tensor.dtype != "int8":
            continue
        if name in resize_controls:
            # roi/scales/sizes initializers must never receive an activation
            # quantization row, even if the contract happens to supply one.
            continue
        if _is_compile_time_tensor(graph, name):
            # Initializers normally need no activation entry.  A same-shape
            # constant operand of Mul does: its scale participates in the exact
            # requantization ratio, so an explicit entry is honoured here.
            row = entries.get(name)
            if not isinstance(row, dict):
                continue
            if row.get("materialized") is False:
                raise CompileError(E_QUANT_MISMATCH,
                                   "initializer %s cannot be non-materialized" % name)
            scale = _positive_float(row.get("scale"), "scale for %s" % name)
            zero = _zero_point(row.get("zero_point", 0), "zero_point for %s" % name)
            quant = {"dtype": "int8", "scale": scale, "zero_point": zero,
                     "materialized": True, "compile_time": True}
            tensor.quantization = quant
            applied[name] = dict(quant)
            continue
        row = entries.get(name)
        if not isinstance(row, dict):
            raise CompileError(E_QUANT_MISMATCH,
                               "quant contract has no activation entry for %s" % name)
        dtype = str(row.get("dtype", "int8"))
        if dtype != "int8":
            raise CompileError(E_QUANT_MISMATCH,
                               "activation %s dtype %s is not INT8" % (name, dtype))
        if name in nonzero_pool_names:
            try:
                zero = int(row.get("zero_point", 0))
            except (TypeError, ValueError):
                raise CompileError(E_QUANT_MISMATCH, "zero point for %s must be an integer" % name)
            if not -128 <= zero <= 127:
                raise CompileError(E_QUANT_MISMATCH, "zero point for %s is outside INT8" % name)
        else:
            zero = _zero_point(row.get("zero_point", 0), "zero_point for %s" % name)
        materialized = row.get("materialized", True)
        if materialized is False:
            # E1-P3: a fused Conv->BN boundary declares that no INT8 rounding
            # happens at this tensor.  Such a tensor must not carry a scale:
            # the only quantization point is the following BatchNormalization
            # output, whose own contract entry supplies the output scale.
            if "scale" in row:
                raise CompileError(
                    E_QUANT_MISMATCH,
                    "non-materialized tensor %s must not declare a scale; the "
                    "following BatchNormalization output is the quantization point" % name)
            quant = {"dtype": "int8", "zero_point": zero, "materialized": False}
            tensor.quantization = quant
            applied[name] = dict(quant)
            continue
        scale = _positive_float(row.get("scale"), "scale for %s" % name)
        quant = {"dtype": "int8", "scale": scale, "zero_point": zero,
                 "materialized": True}
        tensor.quantization = quant
        applied[name] = dict(quant)

    conv_rows = contract.get("convs", {})
    if conv_rows is None:
        conv_rows = {}
    if not isinstance(conv_rows, dict):
        raise CompileError(E_CFG, "quant contract convs must be an object")
    derived_convs = {}
    for node in graph.nodes:
        if node.op_type not in ("Conv", "DepthwiseConv"):
            continue
        row = conv_rows.get(node.id)
        explicit_depthwise = isinstance(row, dict) and bool(row.get("depthwise"))
        is_depthwise = (node.op_type == "DepthwiseConv" or explicit_depthwise or
                        int((node.attrs or {}).get("group", 1)) != 1)
        if not isinstance(row, dict):
            raise CompileError(E_QUANT_MISMATCH,
                               "quant contract has no Conv entry for %s (%s)" %
                               (node.id, node.name), node_id=node.id)
        cin = int(graph.tensors[node.inputs[0]].shape[1])
        cout = int(graph.tensors[node.outputs[0]].shape[1])
        if is_depthwise and cin != cout:
            raise CompileError(E_QUANT_MISMATCH,
                               "DepthwiseConv %s requires Cin==Cout, got %d/%d" %
                               (node.id, cin, cout), node_id=node.id)
        scales = row.get("weight_scale")
        if not isinstance(scales, list):
            scales = [scales] * cout
        if len(scales) != cout:
            raise CompileError(E_QUANT_MISMATCH,
                               "Conv %s weight_scale length %d != Cout %d" %
                               (node.id, len(scales), cout), node_id=node.id)
        input_q = graph.tensors[node.inputs[0]].quantization
        output_q = graph.tensors[node.outputs[0]].quantization
        if input_q is None or output_q is None:
            raise CompileError(E_QUANT_MISMATCH,
                               "Conv %s lacks input/output activation quantization" % node.id,
                               node_id=node.id)
        if input_q.get("materialized") is False:
            raise CompileError(E_QUANT_MISMATCH,
                               "Conv %s consumes a non-materialized tensor; only a "
                               "BatchNormalization may consume the fused boundary" % node.id,
                               node_id=node.id)
        if is_depthwise:
            # Frozen S1_DW contract: per-tensor activation scales are finite
            # positive values in 2^-16..16.  This check is depthwise-only so the
            # historical dense/activation contract behaviour is unchanged.
            for what, quant in (("input", input_q), ("output", output_q)):
                if quant.get("materialized") is False or "scale" not in quant:
                    continue
                scale = float(quant["scale"])
                if not (2.0 ** -16 <= scale <= 16.0):
                    raise CompileError(
                        E_QUANT_MISMATCH,
                        "DepthwiseConv %s %s activation scale %.17g is outside the "
                        "frozen 2^-16..16 range" % (node.id, what, scale),
                        node_id=node.id)
        node.metadata["quant_input_scale"] = input_q["scale"]
        node.metadata["weight_scales"] = [float(v) for v in
                                          (_positive_float(raw, "weight_scale[%d] for %s" %
                                                           (oc, node.id))
                                           for oc, raw in enumerate(scales))]
        if output_q.get("materialized") is False:
            # The requantization point moves into the fold pass.  No record is
            # derived here because the output scale is not defined yet.
            node.metadata["fold_pending"] = True
            node.metadata.pop("requant_records", None)
            derived_convs[node.id] = {
                "input_scale": input_q["scale"],
                "output_scale": None,
                "Cin": cin,
                "Cout": cout,
                "weight_scale": node.metadata["weight_scales"],
                "fold_pending": True,
                "records": [],
            }
            continue
        records = []
        for oc, raw_scale in enumerate(scales):
            weight_scale = _positive_float(raw_scale,
                                           "weight_scale[%d] for %s" % (oc, node.id))
            multiplier = input_q["scale"] * weight_scale / output_q["scale"]
            m0, shift = multiplier_to_m0_shift(multiplier)
            records.append({
                "oc": oc,
                "weight_scale": weight_scale,
                "bias_scale": input_q["scale"] * weight_scale,
                "multiplier": multiplier,
                "M0": m0,
                "SHIFT": shift,
                "out_zero_point": 0,
            })
        derived_convs[node.id] = {
            "input_scale": input_q["scale"],
            "output_scale": output_q["scale"],
            "Cin": cin,
            "Cout": cout,
            "records": records,
        }
        if is_depthwise:
            derived_convs[node.id]["depthwise"] = True
        node.metadata["requant_records"] = records
    derived_mul = {}
    for node in graph.nodes:
        if node.op_type != "Mul":
            continue
        q0 = graph.tensors[node.inputs[0]].quantization or {}
        q1 = graph.tensors[node.inputs[1]].quantization or {}
        qo = graph.tensors[node.outputs[0]].quantization or {}
        for what, quant in (("input0", q0), ("input1", q1), ("output", qo)):
            if quant.get("materialized") is False or "scale" not in quant:
                raise CompileError(E_QUANT_MISMATCH,
                                   "Mul %s requires a materialized INT8 tensor with an "
                                   "explicit scale (declare same-shape initializer operands "
                                   "in the contract tensors map)" % what, node_id=node.id)
        if exact(q0["scale"]) != exact(q1["scale"]):
            raise CompileError(E_QUANT_MISMATCH,
                               "Mul requires equal input scales; got %.17g and %.17g" %
                               (q0["scale"], q1["scale"]), node_id=node.id)
        try:
            ratio = exact(q0["scale"]) * exact(q1["scale"]) / exact(qo["scale"])
            m0, shift = ratio_to_m0_shift(ratio)
        except ExactQuantError as error:
            raise CompileError(error.code, "Mul %s: %s" % (node.id, error.message),
                               node_id=node.id)
        node.metadata["mul_requant"] = {
            "M0": m0, "SHIFT": shift, "ratio": float(ratio),
            "ratio_numerator": ratio.numerator, "ratio_denominator": ratio.denominator,
            "input_scale": q0["scale"], "output_scale": qo["scale"],
            "max_product_abs": 128 * 128,
            "max_requant_temp": 128 * 128 * m0 + (1 << (shift - 1) if shift else 0),
            "max_requant_bits": (128 * 128 * m0 + (1 << (shift - 1) if shift else 0)).bit_length(),
        }
        derived_mul[node.id] = node.metadata["mul_requant"]
    derived_resize = {}
    derived_pool5 = {}
    derived_pool2d = {}
    for node in graph.nodes:
        if node.op_type == "Resize":
            derived_resize[node.id] = _identity_quant(graph, node, "Resize")
        elif node.op_type == "MaxPool" and "legacy_layer_index" not in node.metadata:
            derived_pool2d[node.id] = _identity_quant(graph, node, "MaxPool2D", allow_nonzero=True)
    graph.metadata["quant_contract_schema"] = SCHEMA
    graph.metadata["quant_contract"] = contract
    graph.metadata["quant_applied_activations"] = applied
    graph.metadata["conv_requant"] = derived_convs
    graph.metadata["mul_requant"] = derived_mul
    if derived_resize:
        graph.metadata["resize_quant"] = derived_resize
    if derived_pool5:
        graph.metadata["pool5_quant"] = derived_pool5
    if derived_pool2d:
        graph.metadata["maxpool2d_quant"] = derived_pool2d
    return {"schema": SCHEMA, "applied_activations": applied, "convs": derived_convs,
            "muls": derived_mul, "resize": derived_resize, "maxpool5": derived_pool5}


def contract_from_fixture(model_sha256: str, tensors: Dict[str, Any],
                          convs: Dict[str, Any]) -> Dict[str, Any]:
    return {"schema": SCHEMA, "model_sha256": model_sha256,
            "tensors": tensors, "convs": convs}
