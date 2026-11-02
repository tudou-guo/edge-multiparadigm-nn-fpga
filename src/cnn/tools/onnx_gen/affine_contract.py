# -*- coding: utf-8 -*-
"""Affine quantisation contract for the v0.2 real-YOLO model (affine-int8-v1).

This is the compiler-side contract layer for the new numeric profile.  It is
explicitly versioned and completely separate from the legacy
e1-quant-contract-1 module (which keeps its signed-INT8 zero-point-0 semantics,
byte for byte).  The contract:

* validates the declared per-tensor affine domains and rejects anything the
  hardware is not being built for (uint8 storage without the re-encode mapping,
  non-zero weight zero points, per-channel weight scales, unknown ops);
* attaches the signed re-encoded domain to the graph tensors
  (qs = qu - 128, zs = zu - 128);
* derives, per Conv/Add/Concat, the exact packing quantities the compiler and
  the RTL need: folded bias, INT32 accumulator bound, M0/SHIFT requantisation,
  padding value, ReLU threshold, and the single-rounding Add/Concat params.

The arithmetic itself comes from tools/onnx_gen/affine_quant.py - this module
never re-implements it, and it never reads RTL output or a compiled artifact.

Contract JSON shape (see docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md):

    {"schema": "v0.2-quant-contract-affine-1", "profile": "affine-int8-v1",
     "model_sha256": "...", "rounding": "ties_away_from_zero",
     "tensors": {name: {"scale": float, "zero_point": int,      # uint8 domain
                        "storage": "signed_int8_reencoded"}},
     "convs":   {node_id: {"w_scale": float, "w_zero_point": 0,
                           "w_dtype": "int8", "per_output_channel": false,
                           "bias": "int32"}},
     "requant": {"m0_bits": 31, "shift_max": 62}}
"""
from __future__ import annotations

from typing import Any, Dict

from .affine_quant import (PROFILE, check_acc_int32, check_rescale_int64,
                           choose_add_requant, choose_m0_shift, conv_bias_fold,
                           zero_point_u8_to_s8)
from .graph_ir import CompileError, E_CFG, E_QUANT_MISMATCH, Graph

SCHEMA = "v0.2-quant-contract-affine-1"
SUPPORTED_OPS = ("Conv", "Add", "Concat", "Split", "MaxPool", "Resize")


def _scale(value, what, node_id=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CompileError(E_QUANT_MISMATCH, "%s must be numeric" % what, node_id=node_id)
    if not number > 0.0 or number != number or number in (float("inf"), float("-inf")):
        raise CompileError(E_QUANT_MISMATCH, "%s must be finite and positive" % what,
                           node_id=node_id)
    return number


def _u8_zero_point(value, what, node_id=None):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise CompileError(E_QUANT_MISMATCH, "%s must be integer" % what, node_id=node_id)
    if not (0 <= number <= 255):
        raise CompileError(E_QUANT_MISMATCH,
                           "%s must be in [0,255] (uint8 domain), got %d" % (what, number),
                           node_id=node_id)
    return number


def apply_affine_contract(graph, contract, initializers=None):
    """Validate the affine contract, attach domains and derive packing quantities.

    Returns a dict with:
        tensors : name -> {scale, zero_point_u8, zero_point_s8, storage}
        convs   : node_id -> {w_scale, zs_in, zs_out, sum_w, bias_folded, acc_bound,
                             m0, shift, m0_shift, padding_value_s8, relu_threshold_s8,
                             relu, acc_bound_int32_ok, rescale_int64_ok}
        adds    : node_id -> choose_add_requant(...) params + zero points
        concats : node_id -> per-input rescale parameters (passthrough when same domain)
    """
    if not isinstance(contract, dict):
        raise CompileError(E_CFG, "quant contract must be an object")
    if contract.get("schema") != SCHEMA:
        raise CompileError(E_CFG, "unsupported quant contract schema %r (expected %r)"
                           % (contract.get("schema"), SCHEMA))
    if contract.get("profile") != PROFILE:
        raise CompileError(E_CFG, "contract profile %r is not %r"
                           % (contract.get("profile"), PROFILE))
    if contract.get("rounding", "ties_away_from_zero") != "ties_away_from_zero":
        raise CompileError(E_CFG, "profile rounding must be ties_away_from_zero")
    expected_model = contract.get("model_sha256")
    actual_model = (graph.source or {}).get("model_sha256")
    if expected_model is not None and actual_model is not None and expected_model != actual_model:
        raise CompileError(E_CFG, "contract model_sha256 does not match the graph source")

    requant = contract.get("requant") or {}
    m0_bits = int(requant.get("m0_bits", 31))
    shift_max = int(requant.get("shift_max", 62))

    entries = contract.get("tensors")
    if not isinstance(entries, dict) or not entries:
        raise CompileError(E_CFG, "affine contract tensors must be a non-empty object")
    tensors = {}
    for name, tensor in sorted(graph.tensors.items()):
        if tensor.dtype != "int8" or tensor.kind == "initializer":
            # weight/bias initializers are compile-time data: their quantisation is the
            # per-conv w_scale entry, not an activation domain row
            continue
        row = entries.get(name)
        if not isinstance(row, dict):
            raise CompileError(E_QUANT_MISMATCH,
                               "affine contract has no entry for activation %s" % name)
        storage = str(row.get("storage", "signed_int8_reencoded"))
        if storage != "signed_int8_reencoded":
            raise CompileError(E_QUANT_MISMATCH,
                               "activation %s storage %r is not signed_int8_reencoded "
                               "(changing only the dtype label is not an affine contract)"
                               % (name, storage))
        scale = _scale(row.get("scale"), "scale for %s" % name)
        zu = _u8_zero_point(row.get("zero_point", 0), "zero_point for %s" % name)
        zs = zero_point_u8_to_s8(zu)
        tensors[name] = {"scale": scale, "zero_point_u8": zu, "zero_point_s8": zs,
                         "storage": storage}
        tensor.quantization = {"dtype": "int8", "profile": PROFILE, "scale": scale,
                               "zero_point_u8": zu, "zero_point_s8": zs,
                               "materialized": True}

    conv_rows = contract.get("convs") or {}
    if not isinstance(conv_rows, dict):
        raise CompileError(E_CFG, "affine contract convs must be an object")
    convs = {}
    for node in graph.nodes:
        if node.op_type not in SUPPORTED_OPS:
            raise CompileError(E_QUANT_MISMATCH,
                               "op %s is not part of the affine profile" % node.op_type,
                               node_id=node.id)
        if node.op_type != "Conv":
            continue
        row = conv_rows.get(node.id)
        if not isinstance(row, dict):
            raise CompileError(E_QUANT_MISMATCH,
                               "affine contract has no Conv entry for %s" % node.id,
                               node_id=node.id)
        if row.get("per_output_channel"):
            raise CompileError(E_QUANT_MISMATCH,
                               "per-output-channel weight scales are not supported by "
                               "affine-int8-v1 (node %s)" % node.id, node_id=node.id)
        if str(row.get("w_dtype", "int8")) != "int8":
            raise CompileError(E_QUANT_MISMATCH, "weight dtype must be int8 (node %s)"
                               % node.id, node_id=node.id)
        w_zp = int(row.get("w_zero_point", 0))
        if w_zp != 0:
            raise CompileError(E_QUANT_MISMATCH,
                               "weight zero point must be 0, got %d (node %s)"
                               % (w_zp, node.id), node_id=node.id)
        bias_kind = str(row.get("bias", "int32"))
        if bias_kind not in ("int32", "none"):
            raise CompileError(E_QUANT_MISMATCH,
                               "bias kind %r is not supported (node %s)"
                               % (bias_kind, node.id), node_id=node.id)
        w_scale = _scale(row.get("w_scale"), "w_scale for %s" % node.id, node_id=node.id)
        in_name = node.inputs[0]
        out_name = node.outputs[0]
        if in_name not in tensors or out_name not in tensors:
            raise CompileError(E_QUANT_MISMATCH,
                               "Conv %s input/output domains are missing" % node.id,
                               node_id=node.id)
        w_int8 = _conv_weight_int8(graph, node, w_scale, initializers)
        bias_int32 = _conv_bias_int32(graph, node, row, tensors[in_name]["scale"], w_scale,
                                      initializers)
        zs_in = tensors[in_name]["zero_point_s8"]
        zs_out = tensors[out_name]["zero_point_s8"]
        folding = conv_bias_fold(w_int8, bias_int32, zs_in,
                                 cin=int(graph.tensors[in_name].shape[1]))
        check_acc_int32(int(folding["acc_bound"]))
        m0s = choose_m0_shift(tensors[in_name]["scale"], w_scale,
                              tensors[out_name]["scale"], m0_bits=m0_bits,
                              shift_max=shift_max)
        check_rescale_int64(int(folding["acc_bound"]), int(m0s["m0"]))
        relu = bool((node.attrs or {}).get("relu", False))
        convs[node.id] = {
            "w_scale": w_scale, "zs_in": zs_in, "zs_out": zs_out,
            "sum_w": [int(v) for v in folding["sum_w"]],
            "bias_folded": [int(v) for v in folding["bias_folded"]],
            "acc_bound": int(folding["acc_bound"]),
            "acc_bound_int32_ok": True, "rescale_int64_ok": True,
            "m0": int(m0s["m0"]), "shift": int(m0s["shift"]), "m0_shift": m0s,
            "padding_value_s8": zs_in,
            "relu": relu,
            "relu_threshold_s8": zs_out if relu else None,
        }

    adds = {}
    concats = {}
    for node in graph.nodes:
        if node.op_type not in ("Add", "Concat"):
            continue
        out = tensors[node.outputs[0]]
        ins = [tensors[name] for name in node.inputs]
        if node.op_type == "Add":
            if len(ins) != 2:
                raise CompileError(E_QUANT_MISMATCH,
                                   "Add %s must have exactly two inputs" % node.id,
                                   node_id=node.id)
            params = choose_add_requant(ins[0]["scale"], ins[1]["scale"], out["scale"],
                                        m0_bits=m0_bits, shift_max=shift_max)
            params.update({"zero_point_out_s8": out["zero_point_s8"],
                           "zero_points_in_s8": [ins[0]["zero_point_s8"],
                                                 ins[1]["zero_point_s8"]]})
            adds[node.id] = params
        else:
            per_input = []
            for t in ins:
                same = (t["scale"] == out["scale"]
                        and t["zero_point_s8"] == out["zero_point_s8"])
                per_input.append({"zero_point_in_s8": t["zero_point_s8"],
                                  "scale_in": t["scale"],
                                  "passthrough": same,
                                  "m0_shift": None if same else
                                  choose_m0_shift(t["scale"], 1.0, out["scale"],
                                                  m0_bits=m0_bits,
                                                  shift_max=shift_max)})
            concats[node.id] = {"zero_point_out_s8": out["zero_point_s8"],
                                "out_scale": out["scale"], "inputs": per_input}

    return {"schema": SCHEMA, "profile": PROFILE, "tensors": tensors,
            "convs": convs, "adds": adds, "concats": concats}


def _conv_weight_int8(graph, node, w_scale, initializers=None):
    """Read the Conv weight initializer as signed int8 (per-tensor, zero point 0).

    The mainline IR deliberately stores no raw tensor blobs (initializers are
    shape/dtype identity + content hash), so the caller passes the materialised
    initializers (name -> ndarray) exactly like the ONNX frontend does.
    """
    import numpy as np
    weight_name = node.inputs[1]
    tensor = graph.tensors[weight_name]
    if tensor.dtype != "int8":
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s weight %s dtype %r is not int8"
                           % (node.id, weight_name, tensor.dtype), node_id=node.id)
    quant = tensor.quantization or {}
    declared = quant.get("scale")
    if declared is not None and float(declared) != float(w_scale):
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s weight scale %.17g != contract %.17g"
                           % (node.id, float(declared), float(w_scale)), node_id=node.id)
    if initializers is None or weight_name not in initializers:
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s weight %s was not supplied by the initializer provider"
                           % (node.id, weight_name), node_id=node.id)
    arr = np.asarray(initializers[weight_name])
    if arr.dtype != np.int8:
        if arr.dtype.kind == "i" and arr.size and arr.min() >= -128 and arr.max() <= 127:
            arr = arr.astype(np.int8)
        else:
            raise CompileError(E_QUANT_MISMATCH,
                               "Conv %s weight is not representable as int8" % node.id,
                               node_id=node.id)
    return arr


def _conv_bias_int32(graph, node, row, s_in, w_scale, initializers=None):
    """Recover the int32 bias from the graph real float bias and the scales."""
    import numpy as np
    if str(row.get("bias", "int32")) == "none":
        cout = int(graph.tensors[node.outputs[0]].shape[1])
        return np.zeros((cout,), dtype=np.int32)
    if len(node.inputs) < 3:
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s declares an int32 bias but has no bias input" % node.id,
                           node_id=node.id)
    bias_name = node.inputs[2]
    if initializers is None or bias_name not in initializers:
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s bias %s was not supplied by the initializer provider"
                           % (node.id, bias_name), node_id=node.id)
    arr = np.asarray(initializers[bias_name])
    if arr.dtype.kind == "i":
        return arr.astype(np.int32)
    b_scale = float(s_in) * float(w_scale)
    frac = arr.astype(np.float64) / b_scale
    dev = float(np.abs(frac - np.rint(frac)).max()) if frac.size else 0.0
    if dev > 0.5 + 1e-9:
        raise CompileError(E_QUANT_MISMATCH,
                           "Conv %s float bias is not quantisable with s_in*s_w "
                           "(max fractional deviation %.6f)" % (node.id, dev),
                           node_id=node.id)
    return np.rint(frac).astype(np.int32)
