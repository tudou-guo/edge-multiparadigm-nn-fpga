# -*- coding: utf-8 -*-
"""Static QDQ ONNX -> normalized affine deployment IR frontend (model-agnostic).

What this module is
--------------------
The mainline compiler frontend (tools/onnx_gen/onnx_frontend.py) accepts plain
float ONNX only.  The real deployment models are exported as standard static QDQ
graphs: float32 input -> QuantizeLinear(uint8) -> DequantizeLinear -> Conv(int8
weight, int32 bias, both DequantizeLinear) -> optional Clip(0, FLT_MAX) ->
QuantizeLinear(u8) -> DequantizeLinear -> ...  Add/Concat/MaxPool/Resize are
expressed the same way and Split works directly on the u8 tensor.

This frontend folds those Q/DQ pairs back into the normalized deployment IR
(schema v0.2-yolo-graph-1, profile affine-int8-v1) that the compiler and the
independent integer reference (tools/onnx_gen/int_reference.py) consume::

    graph.json        topology (topological node order), shapes, per-node domains
    quant_table.json  per-conv weight scale / folded bias / M0-SHIFT evidence
    weights.npz       <id>.w int8 (true shape), <id>.b int32, <id>.bq int64

Parsing is model-agnostic: topology comes from the real ONNX producer/consumer
edges and tensor identity.  IR node/tensor names are *labels* derived
deterministically from the source names when available (a "_q"-suffixed u8
tensor, or a split output/name family) and from a stable generic per-op fallback
otherwise; recognition, validation and the IR never depend on them.  The profile
keeps its single float32 image input -> QuantizeLinear boundary, but the graph
may carry any number of outputs, and a graph output may be produced by any
supported op (Conv including fused ReLU, Add, Concat, MaxPool, Resize, Split) as
long as the dtype/quantisation boundary is explicit (uint8 Q output).
The deployment graph name comes from the ONNX graph metadata (or a stable
generic name), and model identity assertions (six detection heads,
num_classes/reg_max, fixed node/tensor counts, head geometry) are example-level
checks, not frontend rules - see examples/yolov8n_relu_voc20/identity_check.py.

Every recognised pattern is checked against the contract and every unexpected
shape, attribute, dtype, domain or dangling Q/DQ is a structured rejection
(QDQReject with code/node/reason) - there is no silent fallback and no
best-effort guessing.  All profile maths comes from tools/onnx_gen/affine_quant.py
(the single source of truth for affine-int8-v1).

Supported patterns (the generic operator subset):
  * one float32 graph input -> QuantizeLinear(u8) -> ... -> DequantizeLinear;
  * Conv: int8 weight initializer + DequantizeLinear(w_scale, int8 zp 0),
    int32 bias initializer + DequantizeLinear(s_in*w_scale, int32 zp 0),
    optional Clip(0, FLT_MAX) (ReLU), then QuantizeLinear(u8) -> DequantizeLinear;
  * Add/Concat/MaxPool(pads=2)/Resize(nearest, asymmetric, floor, x2), same QDQ
    shell, no Clip (fused ReLU is a Conv-only contract);
  * Split(axis=1) applied directly to the producer u8 tensor; a per-tensor
    DequantizeLinear emitted for a split input is accepted and recorded as
    dangling (never as data);
  * any number of graph outputs, each a uint8 QuantizeLinear output (or a Split
    output) of a supported op.

CLI::

    python -B -m tools.onnx_gen.qdq_frontend --onnx <model.onnx> \
        --out-dir <dir> [--compare-graph <graph.json>] [--compare-weights weights.npz] \
        [--input-u8 <npy> --heads-ref <heads_int_ref.npz>] [--report <json>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from tools.onnx_gen.affine_quant import (
    M0_BITS,
    PROFILE,
    ROUNDING,
    SHIFT_MAX,
    check_acc_int32,
    check_rescale_int64,
    choose_m0_shift,
    conv_bias_fold,
)

GRAPH_SCHEMA = "v0.2-yolo-graph-1"
QUANT_SCHEMA = "v0.2-yolo-quant-table-1"
FRONTEND_SCHEMA = "v0.2-yolo-qdq-frontend-1"

#: rejection codes (structured, stable strings)
E_QDQ_DEPENDENCY = "E_QDQ_DEPENDENCY"
E_QDQ_UNSUPPORTED_OPSET = "E_QDQ_UNSUPPORTED_OPSET"
E_QDQ_GRAPH_IO = "E_QDQ_GRAPH_IO"
E_QDQ_DYNAMIC_SHAPE = "E_QDQ_DYNAMIC_SHAPE"
E_QDQ_UNSUPPORTED_OP = "E_QDQ_UNSUPPORTED_OP"
E_QDQ_UNSUPPORTED_ATTR = "E_QDQ_UNSUPPORTED_ATTR"
E_QDQ_PATTERN = "E_QDQ_PATTERN"
E_QDQ_DANGLING = "E_QDQ_DANGLING"
E_QDQ_DOMAIN_MISMATCH = "E_QDQ_DOMAIN_MISMATCH"
E_QDQ_WEIGHT_TYPE = "E_QDQ_WEIGHT_TYPE"
E_QDQ_BIAS_SCALE = "E_QDQ_BIAS_SCALE"
E_QDQ_NAMING = "E_QDQ_NAMING"
E_QDQ_SHAPE_MISMATCH = "E_QDQ_SHAPE_MISMATCH"
E_QDQ_ORDER = "E_QDQ_ORDER"
E_QDQ_COUNTS = "E_QDQ_COUNTS"
E_QDQ_SHAPE_INFERENCE = "E_QDQ_SHAPE_INFERENCE"

#: operator subset the affine QDQ profile can lower (no other op is silently skipped)
COMPUTE_OPS = ("Conv", "Add", "Concat", "MaxPool", "Resize", "Split")

#: E_QDQ_COUNTS has a generic meaning only: the graph carries no supported compute
#: node at all.  Model-specific counts (63 convs/95 nodes/...) are example identity
#: checks (examples/yolov8n_relu_voc20/identity_check.py), never frontend rules.
#: E_QDQ_NAMING is reserved for IR identity collisions (the parser derives names
#: instead of demanding a naming convention; a collision is still a structured
#: rejection rather than a silent overwrite).

#: elem types used by the QDQ contract
FLOAT = 1
UINT8 = 2
INT8 = 3
INT32 = 6
INT64 = 7

FLT_MAX_F32 = float(np.finfo(np.float32).max)

#: IR-label helpers.  These only derive stable *labels* for the canonical IR; no
#: recognition/validation decision ever reads a source name.
_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")
_SPLIT_NAME = re.compile(r"(.+)_(\d+)")


class QDQReject(RuntimeError):
    """Structured rejection: an unrecognised pattern is never silently skipped."""

    def __init__(self, code, reason, node=None, detail=None):
        self.code = str(code)
        self.reason = str(reason)
        self.node = None if node is None else str(node)
        self.detail = detail
        RuntimeError.__init__(self, "%s%s: %s" % (
            self.code, (" [" + self.node + "]") if self.node else "", self.reason))

    def as_dict(self):
        out = {"code": self.code, "node": self.node, "reason": self.reason}
        if self.detail is not None:
            out["detail"] = self.detail
        return out


def _reject(code, reason, node=None, detail=None):
    raise QDQReject(code, reason, node=node, detail=detail)


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _sha256_array(arr):
    return _sha256_bytes(np.ascontiguousarray(arr).tobytes())


# --------------------------------------------------------------------------- #
# ONNX helpers
# --------------------------------------------------------------------------- #
def _import_onnx():
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:  # pragma: no cover - dependency failure
        _reject(E_QDQ_DEPENDENCY, "python package onnx is not importable: %r" % (exc,))
    return onnx, numpy_helper


def _node_label(node):
    if node is None:
        return None
    if node.name:
        return node.name
    if node.output:
        return node.output[0]
    return "<unnamed %s>" % node.op_type


def _attrs(node):
    import onnx
    out = {}
    for a in node.attribute:
        out[a.name] = onnx.helper.get_attribute_value(a)
    return out


def _scalar_float(inits, name, node, what):
    if name not in inits:
        _reject(E_QDQ_PATTERN, "%s %r is not a constant initializer" % (what, name), node)
    arr = np.asarray(inits[name])
    if arr.size != 1:
        _reject(E_QDQ_PATTERN, "%s %r must be scalar, got shape %r" % (what, name, arr.shape), node)
    if arr.dtype.kind != "f":
        _reject(E_QDQ_PATTERN, "%s %r must be float, got %r" % (what, name, arr.dtype), node)
    return float(arr.reshape(-1)[0])


def _scalar_int(inits, name, node, what, dtype_hint=None):
    if name not in inits:
        _reject(E_QDQ_PATTERN, "%s %r is not a constant initializer" % (what, name), node)
    arr = np.asarray(inits[name])
    if arr.size != 1:
        _reject(E_QDQ_PATTERN, "%s %r must be scalar, got shape %r" % (what, name, arr.shape), node)
    if arr.dtype.kind not in "iu":
        _reject(E_QDQ_PATTERN, "%s %r must be integer, got %r" % (what, name, arr.dtype), node)
    if dtype_hint is not None and arr.dtype != dtype_hint:
        _reject(E_QDQ_PATTERN, "%s %r must be %r, got %r"
                % (what, name, np.dtype(dtype_hint), arr.dtype), node)
    return int(arr.reshape(-1)[0])


def _tail_dims(shape, rank, node, what):
    """Return (C, H, W) from a static NCHW shape with batch 1."""
    if len(shape) != rank:
        _reject(E_QDQ_GRAPH_IO, "%s must be rank %d, got shape %r" % (what, rank, shape), node)
    for d in shape:
        if d is None or int(d) <= 0:
            _reject(E_QDQ_DYNAMIC_SHAPE, "%s has a dynamic/zero dimension: shape %r"
                    % (what, shape), node)
    if rank == 4:
        if int(shape[0]) != 1:
            _reject(E_QDQ_DYNAMIC_SHAPE, "%s batch must be 1, got %r" % (what, shape), node)
        return [int(shape[1]), int(shape[2]), int(shape[3])]
    return [int(d) for d in shape]


def _conv_out_hw(h, w, k, stride, pad):
    if h + 2 * pad < k or w + 2 * pad < k:
        return None
    return [(h + 2 * pad - k) // stride + 1, (w + 2 * pad - k) // stride + 1]


# --------------------------------------------------------------------------- #
# frontend
# --------------------------------------------------------------------------- #
class QDQFrontend(object):
    """One-shot parser: build once with the ONNX model, then call parse()."""

    def __init__(self, model, onnx_path=None):
        onnx, numpy_helper = _import_onnx()
        self._onnx = onnx
        self._numpy_helper = numpy_helper
        self.model = model
        self.onnx_path = None if onnx_path is None else str(onnx_path)
        self.graph_pb = model.graph
        self.nodes = list(self.graph_pb.node)
        self.inits = {}
        self.init_proto = {}
        for init in self.graph_pb.initializer:
            self.init_proto[init.name] = init
            self.inits[init.name] = numpy_helper.to_array(init)
        self.producers = {}
        self.consumers = {}
        for idx, node in enumerate(self.nodes):
            for out in node.output:
                if out in self.producers:
                    _reject(E_QDQ_PATTERN, "tensor %r produced twice" % out, _node_label(node))
                self.producers[out] = idx
            for i, t in enumerate(node.input):
                if t == "":
                    continue
                self.consumers.setdefault(t, []).append((idx, i))
        self.notes = []
        self.dangling_dq = []
        self.shape_checks = []
        self.domain_checks = []
        self.used_q = set()
        self.used_dq = set()
        self.used_clip = set()
        self.used_inits = set()
        self._conv_pack = {}
        self.counts = {}
        self.domains = {}          # u8 tensor name -> {"scale","zero_point"}
        self.float_from_u8 = {}    # float tensor name -> u8 tensor name
        self.u8_of_float = {}
        self.dq_index_of_float = {}
        # generic IR identity: u8 tensor -> deployment IR name; never derived from
        # a naming convention (source names are advisory labels with a fallback).
        self._ir_name_of_u8 = {}   # u8 tensor -> IR node id / split output name
        self._ir_id_of_node = {}   # ONNX node index -> deployment IR node id
        self._node_u8_out = {}     # ONNX node index -> its u8 Q output tensor
        self._node_relu = {}       # ONNX node index -> fused-ReLU flag (Conv only)
        self._split_out_names = {} # ONNX node index -> generated split output names
        self._used_ir_names = {"__input__"}
        self.source_nodes = []     # report-only source mapping (IR id -> ONNX node)

    # -- small accessors ---------------------------------------------------- #
    def node_at(self, idx):
        return self.nodes[idx]

    def consumer_list(self, tensor):
        return self.consumers.get(tensor, [])

    def single_consumer(self, tensor, node, what):
        cons = self.consumer_list(tensor)
        if len(cons) != 1:
            _reject(E_QDQ_PATTERN, "%s %r must have exactly one consumer, got %d"
                    % (what, tensor, len(cons)), _node_label(node))
        return cons[0]

    def init_dtype(self, name):
        return int(self.init_proto[name].data_type)

    def infer_types(self):
        onnx = self._onnx
        try:
            inferred = onnx.shape_inference.infer_shapes(self.model, strict_mode=False)
        except Exception as exc:  # noqa: BLE001 - any inference failure is a rejection
            _reject(E_QDQ_SHAPE_INFERENCE, "onnx.shape_inference failed: %r" % (exc,))
        self.type_map = {}
        collections = [inferred.graph.input, inferred.graph.output, inferred.graph.value_info]
        for coll in collections:
            for value in coll:
                tt = value.type.tensor_type
                dims = [int(d.dim_value) if d.HasField("dim_value") else None
                        for d in tt.shape.dim]
                self.type_map[value.name] = (int(tt.elem_type), dims)
        for name in self.inits:
            self.type_map.setdefault(name, (self.init_dtype(name),
                                            [int(d) for d in self.init_proto[name].dims]))

    def elem_type(self, name, node, what):
        if name not in self.type_map:
            _reject(E_QDQ_PATTERN, "%s tensor %r has no inferred type" % (what, name),
                    _node_label(node))
        return int(self.type_map[name][0])

    def inferred_chw(self, name, node, what):
        if name not in self.type_map:
            _reject(E_QDQ_PATTERN, "%s tensor %r has no inferred shape" % (what, name),
                    _node_label(node))
        return _tail_dims(self.type_map[name][1], 4, _node_label(node), what)

    # -- opset / graph io --------------------------------------------------- #
    def check_opset(self):
        default_versions = [int(op.version) for op in self.model.opset_import
                            if op.domain in ("", "ai.onnx")]
        if len(default_versions) != 1:
            _reject(E_QDQ_UNSUPPORTED_OPSET,
                    "expected exactly one default-domain opset, got %r" % (default_versions,))
        version = default_versions[0]
        if not (13 <= version <= 19):
            _reject(E_QDQ_UNSUPPORTED_OPSET,
                    "opset %d not supported (QDQ graph requires 13..19)" % version)
        self.opset = version

    def graph_input(self):
        data_inputs = [i for i in self.graph_pb.input if i.name not in self.inits]
        if len(data_inputs) != 1:
            _reject(E_QDQ_GRAPH_IO, "expected exactly one non-initializer graph input, got %d"
                    % len(data_inputs))
        return data_inputs[0]

    # -- domain detection --------------------------------------------------- #
    def collect_domains(self):
        for idx, node in enumerate(self.nodes):
            if node.op_type != "QuantizeLinear":
                continue
            if len(node.input) != 3:
                _reject(E_QDQ_PATTERN, "QuantizeLinear must have (x, scale, zero_point)",
                        _node_label(node))
            scale = _scalar_float(self.inits, node.input[1], _node_label(node), "quant scale")
            zp = _scalar_int(self.inits, node.input[2], _node_label(node), "quant zero point",
                             dtype_hint=np.uint8)
            if zp < 0 or zp > 255:
                _reject(E_QDQ_PATTERN, "u8 zero point out of range: %r" % (zp,), _node_label(node))
            out = node.output[0]
            if self.elem_type(out, node, "QuantizeLinear output") != UINT8:
                _reject(E_QDQ_PATTERN, "QuantizeLinear output %r is not uint8" % out,
                        _node_label(node))
            if scale <= 0.0:
                _reject(E_QDQ_PATTERN, "quant scale must be positive, got %r" % (scale,),
                        _node_label(node))
            self.domains[out] = {"scale": scale, "zero_point": zp}
            self.used_inits.add(node.input[1])
            self.used_inits.add(node.input[2])
        if not self.domains:
            _reject(E_QDQ_PATTERN, "no QuantizeLinear nodes: this is not a QDQ deployment graph")

    def collect_split_domains(self):
        """Split outputs keep their producer's domain; register them as u8 tensors.

        The export emits a DequantizeLinear for every u8 tensor, including the 16
        Split outputs, so their domains must be known before the Q/DQ pairs are
        resolved.  A Split output is a channel view of its input, therefore it has
        the input's scale/zero point (contract section 2.4).
        """
        seen = set()

        def domain_of(tensor, node):
            if tensor in self.domains:
                return self.domains[tensor]
            if tensor in seen:
                _reject(E_QDQ_PATTERN, "Split domain resolution cycle at %r" % tensor,
                        _node_label(node))
            seen.add(tensor)
            idx = self.producers.get(tensor)
            if idx is None:
                _reject(E_QDQ_PATTERN, "Split input %r has no producer" % tensor,
                        _node_label(node))
            prod = self.nodes[idx]
            if prod.op_type != "Split":
                _reject(E_QDQ_PATTERN, "Split input %r is not quantised (producer %s)"
                        % (tensor, prod.op_type), _node_label(node))
            if len(prod.input) < 1 or prod.input[0] == "":
                _reject(E_QDQ_PATTERN, "Split %r has no input" % _node_label(prod),
                        _node_label(prod))
            return domain_of(prod.input[0], prod)

        for node in self.nodes:
            if node.op_type != "Split":
                continue
            if len(node.input) < 1 or node.input[0] == "":
                _reject(E_QDQ_PATTERN, "Split %r has no input" % _node_label(node),
                        _node_label(node))
            domain = domain_of(node.input[0], node)
            for out in node.output:
                if not out:
                    continue
                if out in self.domains:
                    _reject(E_QDQ_PATTERN, "Split output %r already has a domain" % out,
                            _node_label(node))
                self.domains[out] = {"scale": domain["scale"],
                                     "zero_point": domain["zero_point"],
                                     "from_split": _node_label(node)}

    def resolve_dequantisers(self):
        """Map every DequantizeLinear output to its u8 source tensor (validated)."""
        for idx, node in enumerate(self.nodes):
            if node.op_type != "DequantizeLinear":
                continue
            if len(node.input) != 3:
                _reject(E_QDQ_PATTERN, "DequantizeLinear must have (x, scale, zero_point)",
                        _node_label(node))
            src = node.input[0]
            out = node.output[0]
            if src in self.domains:
                scale = _scalar_float(self.inits, node.input[1], _node_label(node),
                                      "dequant scale")
                zp = _scalar_int(self.inits, node.input[2], _node_label(node),
                                 "dequant zero point", dtype_hint=np.uint8)
                domain = self.domains[src]
                if scale != domain["scale"] or zp != domain["zero_point"]:
                    _reject(E_QDQ_DOMAIN_MISMATCH,
                            "DequantizeLinear %r domain (%r,%r) != producer domain (%r,%r)"
                            % (out, scale, zp, domain["scale"], domain["zero_point"]),
                            _node_label(node))
                if self.elem_type(out, node, "dequantised activation") != FLOAT:
                    _reject(E_QDQ_PATTERN, "dequantised activation %r is not float32" % out,
                            _node_label(node))
                self.float_from_u8[out] = src
                self.u8_of_float[out] = src
                self.dq_index_of_float[out] = idx
                self.used_inits.add(node.input[1])
                self.used_inits.add(node.input[2])
        if not self.float_from_u8:
            _reject(E_QDQ_PATTERN, "no u8 DequantizeLinear nodes found")

    def weight_dequant(self, tensor, node):
        """Resolve a Conv weight input: (int8 initializer, w_scale)."""
        prod_idx = self.producers.get(tensor)
        if prod_idx is None:
            _reject(E_QDQ_PATTERN, "Conv weight input %r is not produced by any node" % tensor,
                    _node_label(node))
        prod = self.nodes[prod_idx]
        if prod.op_type != "DequantizeLinear":
            _reject(E_QDQ_PATTERN, "Conv weight %r is not a DequantizeLinear output" % tensor,
                    _node_label(node))
        src = prod.input[0]
        if src not in self.inits:
            _reject(E_QDQ_WEIGHT_TYPE, "Conv weight %r is not an initializer" % src,
                    _node_label(node))
        if self.init_dtype(src) != INT8:
            _reject(E_QDQ_WEIGHT_TYPE, "Conv weight %r must be int8, got elem_type %d"
                    % (src, self.init_dtype(src)), _node_label(node))
        w_scale = _scalar_float(self.inits, prod.input[1], _node_label(node), "weight scale")
        w_zp = _scalar_int(self.inits, prod.input[2], _node_label(node), "weight zero point",
                           dtype_hint=np.int8)
        if w_zp != 0:
            _reject(E_QDQ_WEIGHT_TYPE,
                    "Conv weight %r zero point must be 0 (symmetric), got %d" % (src, w_zp),
                    _node_label(node))
        if self.elem_type(prod.output[0], prod, "weight dequant") != FLOAT:
            _reject(E_QDQ_WEIGHT_TYPE, "weight DequantizeLinear output is not float32",
                    _node_label(prod))
        self.used_inits.add(src)
        self.used_inits.add(prod.input[1])
        self.used_inits.add(prod.input[2])
        self.used_dq.add(prod_idx)
        return {"name": src, "w": np.asarray(self.inits[src], dtype=np.int8),
                "w_scale": w_scale, "w_zero_point": w_zp, "node": prod}

    def bias_dequant(self, tensor, node):
        prod_idx = self.producers.get(tensor)
        if prod_idx is None:
            _reject(E_QDQ_PATTERN, "Conv bias input %r is not produced by any node" % tensor,
                    _node_label(node))
        prod = self.nodes[prod_idx]
        if prod.op_type != "DequantizeLinear":
            _reject(E_QDQ_PATTERN, "Conv bias %r is not a DequantizeLinear output" % tensor,
                    _node_label(node))
        src = prod.input[0]
        if src not in self.inits:
            _reject(E_QDQ_PATTERN, "Conv bias %r is not an initializer" % src, _node_label(node))
        if self.init_dtype(src) != INT32:
            _reject(E_QDQ_PATTERN, "Conv bias %r must be int32, got elem_type %d"
                    % (src, self.init_dtype(src)), _node_label(node))
        b_scale = _scalar_float(self.inits, prod.input[1], _node_label(node), "bias scale")
        b_zp = _scalar_int(self.inits, prod.input[2], _node_label(node), "bias zero point",
                           dtype_hint=np.int32)
        if b_zp != 0:
            _reject(E_QDQ_PATTERN, "Conv bias %r zero point must be 0, got %d" % (src, b_zp),
                    _node_label(node))
        self.used_inits.add(src)
        self.used_inits.add(prod.input[1])
        self.used_inits.add(prod.input[2])
        self.used_dq.add(prod_idx)
        return {"name": src, "b": np.asarray(self.inits[src], dtype=np.int32),
                "b_scale": b_scale, "node": prod}

    # -- QDQ chain walking -------------------------------------------------- #
    def activation_u8(self, float_tensor, node, what):
        """Resolve a compute-op activation input (a DequantizeLinear output) to its u8 source."""
        if float_tensor in self.float_from_u8:
            if float_tensor in self.dq_index_of_float:
                self.used_dq.add(self.dq_index_of_float[float_tensor])
            return self.float_from_u8[float_tensor]
        _reject(E_QDQ_PATTERN, "%s input %r is not a DequantizeLinear output of a u8 tensor"
                % (what, float_tensor), _node_label(node))

    def output_qdq(self, node, allow_relu):
        """Follow op float output -> [Clip] -> QuantizeLinear; return (u8 tensor, relu)."""
        out = node.output[0]
        relu = False
        idx, port = self.single_consumer(out, node, "op output")
        nxt = self.nodes[idx]
        if nxt.op_type == "Clip":
            if not allow_relu:
                _reject(E_QDQ_UNSUPPORTED_ATTR,
                        "Clip after %s is not part of this graph family" % node.op_type,
                        _node_label(nxt))
            if port != 0:
                _reject(E_QDQ_PATTERN, "Clip consumes the op output on port %d" % port,
                        _node_label(nxt))
            if len(nxt.input) != 3:
                _reject(E_QDQ_PATTERN, "Clip must carry explicit min and max initializers",
                        _node_label(nxt))
            clip_min = _scalar_float(self.inits, nxt.input[1], _node_label(nxt), "Clip min")
            clip_max = _scalar_float(self.inits, nxt.input[2], _node_label(nxt), "Clip max")
            if clip_min != 0.0 or clip_max < 0.99 * FLT_MAX_F32:
                _reject(E_QDQ_UNSUPPORTED_ATTR,
                        "Clip must be the ReLU form Clip(0, FLT_MAX), got (%r, %r)"
                        % (clip_min, clip_max), _node_label(nxt))
            self.used_inits.add(nxt.input[1])
            self.used_inits.add(nxt.input[2])
            self.used_clip.add(idx)
            relu = True
            out = nxt.output[0]
            idx, port = self.single_consumer(out, nxt, "Clip output")
            nxt = self.nodes[idx]
        if nxt.op_type != "QuantizeLinear":
            _reject(E_QDQ_PATTERN,
                    "op output %r must be followed by QuantizeLinear, got %s"
                    % (out, nxt.op_type), _node_label(nxt))
        if port != 0:
            _reject(E_QDQ_PATTERN, "QuantizeLinear consumes the op output on port %d" % port,
                    _node_label(nxt))
        u8 = nxt.output[0]
        if u8 not in self.domains:
            _reject(E_QDQ_DOMAIN_MISMATCH, "QuantizeLinear output %r has no recorded domain"
                    % u8, _node_label(nxt))
        self.used_q.add(idx)
        return u8, relu

    # -- generic IR identity (names are labels, never recognition rules) ---- #
    @classmethod
    def _sanitize_id(cls, text):
        """Turn a source name into a stable IR label; None when unusable."""
        if text is None:
            return None
        name = _ID_CHARS.sub("_", str(text).strip()).strip("._")
        if not name or name == "__input__":
            return None
        return name

    def _allocate_ir_name(self, candidates, op, idx):
        """First free sanitized source-name label, else a stable generic fallback."""
        for cand in candidates:
            name = self._sanitize_id(cand)
            if name is not None and name not in self._used_ir_names:
                self._used_ir_names.add(name)
                return name
        base = "%s_%d" % (op.lower(), idx)
        name = base
        suffix = 1
        while name in self._used_ir_names:
            name = "%s_%d" % (base, suffix)
            suffix += 1
        self._used_ir_names.add(name)
        return name

    @staticmethod
    def _node_id_candidates(node, u8_out):
        """Advisory label hints from the producer node's own tensor/name."""
        cands = []
        if u8_out.endswith("_q") and len(u8_out) > 2:
            cands.append(u8_out[:-2])
        if node.name:
            cands.append(node.name)
        return cands

    @staticmethod
    def _split_id_candidates(node, outputs):
        """A split keeps its own stable name; outputs are '<id>_<index>'."""
        cands = []
        matches = [_SPLIT_NAME.fullmatch(out) for out in outputs]
        if outputs and all(m is not None for m in matches):
            prefixes = {m.group(1) for m in matches}
            indices = [int(m.group(2)) for m in matches]
            if len(prefixes) == 1 and indices == list(range(len(outputs))):
                cands.append(prefixes.pop())
        if node.name:
            cands.append(node.name)
        return cands

    def prepare_ir_ids(self):
        """Assign deployment IR names before pattern parsing (edge-based only).

        Every supported compute node gets an IR id from its own ONNX node/tensor
        identity: a source-name label when usable, otherwise a stable
        '<op>_<onnx node index>' fallback.  Split output names are generated from
        the split id + output index.  The conventional names of the exported
        assets are reproduced when present, but recognition/validation never reads
        them and no naming convention is required.
        """
        for idx, node in enumerate(self.nodes):
            op = node.op_type
            if op == "Split":
                outputs = [out for out in node.output if out]
                nid = self._allocate_ir_name(self._split_id_candidates(node, outputs), op, idx)
                out_names = []
                for j in range(len(outputs)):
                    name = "%s_%d" % (nid, j)
                    if name in self._used_ir_names:
                        _reject(E_QDQ_NAMING,
                                "generated split output %r collides with an IR name" % name,
                                _node_label(node))
                    self._used_ir_names.add(name)
                    out_names.append(name)
                self._ir_id_of_node[idx] = nid
                self._split_out_names[idx] = out_names
                for out, name in zip(outputs, out_names):
                    self._ir_name_of_u8[out] = name
            elif op in COMPUTE_OPS:
                u8_out, relu = self.output_qdq(node, allow_relu=(op == "Conv"))
                self._node_u8_out[idx] = u8_out
                self._node_relu[idx] = relu
                nid = self._allocate_ir_name(self._node_id_candidates(node, u8_out), op, idx)
                self._ir_id_of_node[idx] = nid
                self._ir_name_of_u8[u8_out] = nid
            if op in COMPUTE_OPS:
                self.source_nodes.append({"ir": self._ir_id_of_node[idx], "op": op,
                                          "onnx_node": _node_label(node),
                                          "onnx_outputs": list(node.output)})

    def ir_name_of_u8(self, u8, node=None, what="activation"):
        """Resolve a u8 tensor to its deployment IR name (structural, name-free)."""
        name = self._ir_name_of_u8.get(u8)
        if name is None:
            _reject(E_QDQ_PATTERN,
                    "%s tensor %r is not produced by a supported quantised pattern"
                    % (what, u8), _node_label(node))
        return name

    def shapes(self):
        return self._shapes

    # -- main parse --------------------------------------------------------- #
    def parse(self, strict_counts=True):
        """Parse the QDQ graph into the normalized deployment IR.

        strict_counts is accepted for backward compatibility only: the frontend no
        longer applies model identity rules (six heads, fixed node/tensor counts,
        num_classes/reg_max, naming conventions).  Those assertions are example
        asset checks (examples/yolov8n_relu_voc20/identity_check.py), so both flag
        values parse identically while unsupported patterns stay structured
        rejections in every mode.
        """
        self.check_opset()
        self.infer_types()
        data_input = self.graph_input()
        input_dtype = int(data_input.type.tensor_type.elem_type)
        if input_dtype != FLOAT:
            _reject(E_QDQ_GRAPH_IO, "graph input must be float32, got elem_type %d" % input_dtype)
        input_shape = [int(d.dim_value) if d.HasField("dim_value") else None
                       for d in data_input.type.tensor_type.shape.dim]
        input_chw = _tail_dims(input_shape, 4, data_input.name, "graph input")
        self.input_name = data_input.name

        self.collect_domains()
        # the float graph input must be quantised at the boundary by exactly one Q
        cons = self.consumer_list(self.input_name)
        if len(cons) != 1 or self.nodes[cons[0][0]].op_type != "QuantizeLinear":
            _reject(E_QDQ_GRAPH_IO, "graph input %r must feed exactly one QuantizeLinear"
                    % self.input_name)
        input_q = self.nodes[cons[0][0]].output[0]
        self.used_q.add(cons[0][0])
        self.input_u8 = input_q
        self._ir_name_of_u8[input_q] = "__input__"
        self.collect_split_domains()
        self.resolve_dequantisers()
        self.prepare_ir_ids()

        self._shapes = {"__input__": list(input_chw)}
        self._produced = set()
        self.nodes_ir = []
        self.tensors_ir = {"__input__": {"shape": list(input_chw), "domain_node": "__input__",
                                         "logical_c": int(input_chw[0])}}
        self._node_by_ir_id = {}
        self.split_outputs = []

        for idx, node in enumerate(self.nodes):
            op = node.op_type
            if op in ("QuantizeLinear", "DequantizeLinear", "Clip"):
                continue
            if op == "Conv":
                ir_node = self._parse_conv(idx, node)
            elif op in ("Add", "Concat", "MaxPool", "Resize"):
                ir_node = self._parse_eltwise(idx, node)
            elif op == "Split":
                ir_node = self._parse_split(idx, node)
            else:
                _reject(E_QDQ_UNSUPPORTED_OP,
                        "operator %r is not part of the supported QDQ subset" % op,
                        _node_label(node))
            nid = ir_node["id"]
            for name in ir_node["inputs"]:
                if name not in self._produced and name != "__input__":
                    _reject(E_QDQ_ORDER, "node %r consumes %r before it is produced"
                            % (nid, name), _node_label(node))
            self._produced.add(nid)
            if ir_node["op"] == "Split":
                for entry in ir_node["attrs"]["outputs"]:
                    self._produced.add(entry["name"])
            self._node_by_ir_id[nid] = ir_node
            self.nodes_ir.append(ir_node)

        if not self.nodes_ir:
            _reject(E_QDQ_COUNTS, "graph contains no supported compute nodes")
        self._check_dangling()
        self._settle_outputs()
        graph = self._build_graph()
        quant, weights = self._build_quant_and_weights()
        report = self._build_report()
        return {"ok": True, "graph": graph, "quant": quant, "weights": weights,
                "report": report}

    # -- per-op parsers ----------------------------------------------------- #
    def _common_eltwise(self, idx, node, n_data_inputs):
        u8s = []
        for i in range(n_data_inputs):
            if i >= len(node.input) or node.input[i] == "":
                _reject(E_QDQ_PATTERN, "%s is missing data input %d" % (node.op_type, i),
                        _node_label(node))
            u8s.append(self.activation_u8(node.input[i], node, node.op_type))
        # the QDQ output pattern (and the Conv-only fused-ReLU contract) is validated
        # once in prepare_ir_ids(); the eltwise parsers just reuse the resolved tensor.
        return u8s, self._node_u8_out[idx]

    def _parse_conv(self, idx, node):
        ins = list(node.input)
        if len(ins) < 2:
            _reject(E_QDQ_PATTERN, "Conv needs at least (x, weight)", _node_label(node))
        u8_in = self.activation_u8(ins[0], node, "Conv")
        w = self.weight_dequant(ins[1], node)
        attrs = _attrs(node)
        kernel_shape = attrs.get("kernel_shape")
        strides = attrs.get("strides", [1, 1])
        pads = attrs.get("pads", [0, 0, 0, 0])
        group = int(attrs.get("group", 1))
        dilations = attrs.get("dilations", [1, 1])
        if kernel_shape is None or len(kernel_shape) != 2:
            _reject(E_QDQ_UNSUPPORTED_ATTR, "Conv kernel_shape must be 2-D, got %r"
                    % (kernel_shape,), _node_label(node))
        kh, kw = int(kernel_shape[0]), int(kernel_shape[1])
        if kh != kw:
            _reject(E_QDQ_UNSUPPORTED_ATTR, "non-square kernel %r is not in this graph"
                    % (kernel_shape,), _node_label(node))
        if len(strides) != 2 or int(strides[0]) != int(strides[1]):
            _reject(E_QDQ_UNSUPPORTED_ATTR, "non-uniform strides %r are not in this graph"
                    % (strides,), _node_label(node))
        if len(pads) != 4 or not (pads[0] == pads[1] == pads[2] == pads[3]):
            _reject(E_QDQ_UNSUPPORTED_ATTR, "asymmetric pads %r are not in this graph"
                    % (pads,), _node_label(node))
        if any(int(d) != 1 for d in dilations):
            _reject(E_QDQ_UNSUPPORTED_ATTR, "dilations %r are not supported" % (dilations,),
                    _node_label(node))
        kernel, stride, pad = kh, int(strides[0]), int(pads[0])
        u8_out = self._node_u8_out[idx]
        relu = self._node_relu[idx]
        nid = self._ir_id_of_node[idx]
        in_shape = self._shapes[self.ir_name_of_u8(u8_in, node, "Conv input")]
        cout = int(w["w"].shape[0])
        cin = int(w["w"].shape[1])
        if cin != in_shape[0]:
            _reject(E_QDQ_SHAPE_MISMATCH, "Conv %r weight Cin %d != input C %d"
                    % (nid, cin, in_shape[0]), _node_label(node))
        hw = _conv_out_hw(in_shape[1], in_shape[2], kernel, stride, pad)
        if hw is None:
            _reject(E_QDQ_SHAPE_MISMATCH, "Conv %r kernel %d does not fit input %r"
                    % (nid, kernel, in_shape), _node_label(node))
        out_shape = [cout, hw[0], hw[1]]
        bias = None
        if len(ins) >= 3 and ins[2] != "":
            bias = self.bias_dequant(ins[2], node)
        else:
            bias = {"name": None, "b": np.zeros([cout], dtype=np.int32), "b_scale": None}
        s_in = self.domains[u8_in]["scale"]
        z_in = self.domains[u8_in]["zero_point"]
        expected_b_scale = float(np.float32(s_in) * np.float32(w["w_scale"]))
        if bias["b_scale"] is not None:
            rel = abs(bias["b_scale"] - expected_b_scale) / max(expected_b_scale, 1e-30)
            if rel > 1e-5:
                _reject(E_QDQ_BIAS_SCALE,
                        "Conv %r bias scale %r != s_in*w_scale %r (rel %.3e)"
                        % (nid, bias["b_scale"], expected_b_scale, rel), _node_label(node))
        self._record_shape(nid, u8_out, out_shape, node)
        ir = {
            "id": nid,
            "op": "Conv",
            "inputs": [self.ir_name_of_u8(u8_in, node, "Conv input")],
            "attrs": {"kernel": kernel, "stride": stride, "pad": pad, "groups": group,
                      "relu": bool(relu), "cin": cin, "cout": cout},
            "out_shape": out_shape,
            "quant": {"out_scale": self.domains[u8_out]["scale"],
                      "out_zero_points": [self.domains[u8_out]["zero_point"]],
                      "in_scales": [s_in], "in_zero_points": [z_in],
                      "w_scale": w["w_scale"]},
        }
        self._conv_pack[nid] = {"w": w["w"], "b": bias["b"], "w_scale": w["w_scale"],
                                "w_zero_point": w["w_zero_point"],
                                "bias_present": bias["b_scale"] is not None}
        return ir

    def _parse_eltwise(self, idx, node):
        op = node.op_type
        if op == "Add":
            u8s, u8_out = self._common_eltwise(idx, node, 2)
            shapes = [self._shapes[self.ir_name_of_u8(u)] for u in u8s]
            if shapes[0] != shapes[1]:
                _reject(E_QDQ_SHAPE_MISMATCH, "Add inputs have different shapes %r" % (shapes,),
                        _node_label(node))
            out_shape = list(shapes[0])
            attrs = {}
        elif op == "Concat":
            n = len(node.input)
            u8s, u8_out = self._common_eltwise(idx, node, n)
            a = _attrs(node)
            axis = int(a.get("axis", 0))
            if axis != 1:
                _reject(E_QDQ_UNSUPPORTED_ATTR, "Concat axis %d != 1" % axis, _node_label(node))
            shapes = [self._shapes[self.ir_name_of_u8(u)] for u in u8s]
            hw = shapes[0][1:]
            for s in shapes[1:]:
                if s[1:] != hw:
                    _reject(E_QDQ_SHAPE_MISMATCH, "Concat spatial shapes differ: %r" % (shapes,),
                            _node_label(node))
            out_shape = [int(sum(s[0] for s in shapes)), hw[0], hw[1]]
            attrs = {"axis": 1}
        elif op == "MaxPool":
            u8s, u8_out = self._common_eltwise(idx, node, 1)
            a = _attrs(node)
            if len([name for name in node.output if name]) != 1:
                _reject(E_QDQ_UNSUPPORTED_ATTR, "MaxPool indices output is unsupported", _node_label(node))
            from .maxpool2d_geometry import geometry, normalize_attributes
            from .graph_ir import CompileError
            in_shape = self._shapes[self.ir_name_of_u8(u8s[0])]
            try:
                attrs = normalize_attributes(a)
                geom = geometry((1, *in_shape), attrs)
            except CompileError as exc:
                _reject(E_QDQ_UNSUPPORTED_ATTR, str(exc), _node_label(node))
            out_shape = [in_shape[0], geom.hout, geom.wout]
            attrs["format_version"] = 1
        else:  # Resize
            u8s, u8_out = self._common_eltwise(idx, node, 1)
            a = _attrs(node)
            mode = a.get("mode", b"nearest")
            if isinstance(mode, bytes):
                mode = mode.decode("ascii")
            coord = a.get("coordinate_transformation_mode", b"half_pixel")
            if isinstance(coord, bytes):
                coord = coord.decode("ascii")
            nearest = a.get("nearest_mode", b"round_prefer_floor")
            if isinstance(nearest, bytes):
                nearest = nearest.decode("ascii")
            if mode != "nearest" or coord != "asymmetric" or nearest != "floor":
                _reject(E_QDQ_UNSUPPORTED_ATTR,
                        "Resize must be nearest/asymmetric/floor, got %r/%r/%r"
                        % (mode, coord, nearest), _node_label(node))
            if len(node.input) != 3:
                _reject(E_QDQ_PATTERN, "Resize must carry roi and scales initializers",
                        _node_label(node))
            roi = self.inits.get(node.input[1])
            if roi is None:
                _reject(E_QDQ_PATTERN, "Resize roi %r is not an initializer" % node.input[1],
                        _node_label(node))
            scales = self.inits.get(node.input[2])
            if scales is None:
                _reject(E_QDQ_PATTERN, "Resize scales %r is not an initializer" % node.input[2],
                        _node_label(node))
            scales = np.asarray(scales, dtype=np.float64).reshape(-1)
            if scales.size != 4 or not np.allclose(scales, [1.0, 1.0, 2.0, 2.0], rtol=0.0, atol=0.0):
                _reject(E_QDQ_UNSUPPORTED_ATTR, "Resize scales must be [1,1,2,2], got %r"
                        % (scales.tolist(),), _node_label(node))
            self.used_inits.add(node.input[1])
            self.used_inits.add(node.input[2])
            in_shape = self._shapes[self.ir_name_of_u8(u8s[0])]
            out_shape = [in_shape[0], in_shape[1] * 2, in_shape[2] * 2]
            attrs = {"mode": "nearest", "scale": 2}
        nid = self._ir_id_of_node[idx]
        in_ir = [self.ir_name_of_u8(u, node, op + " input") for u in u8s]
        self._record_shape(nid, u8_out, out_shape, node)
        ir = {
            "id": nid,
            "op": op,
            "inputs": in_ir,
            "attrs": attrs,
            "out_shape": out_shape,
            "quant": {"out_scale": self.domains[u8_out]["scale"],
                      "out_zero_points": [self.domains[u8_out]["zero_point"]],
                      "in_scales": [self.domains[u]["scale"] for u in u8s],
                      "in_zero_points": [self.domains[u]["zero_point"] for u in u8s]},
        }
        return ir

    def _parse_split(self, idx, node):
        if len(node.input) < 2:
            _reject(E_QDQ_PATTERN, "Split must carry an explicit split-sizes initializer",
                    _node_label(node))
        u8_in = node.input[0]
        if u8_in not in self.domains:
            _reject(E_QDQ_PATTERN, "Split input %r is not a u8 quantised tensor" % u8_in,
                    _node_label(node))
        sizes_arr = self.inits.get(node.input[1])
        if sizes_arr is None:
            _reject(E_QDQ_PATTERN, "Split sizes %r is not an initializer" % node.input[1],
                    _node_label(node))
        sizes = [int(v) for v in np.asarray(sizes_arr).reshape(-1)]
        a = _attrs(node)
        axis = int(a.get("axis", 0))
        if axis != 1:
            _reject(E_QDQ_UNSUPPORTED_ATTR, "Split axis %d != 1" % axis, _node_label(node))
        if "split" in a:
            _reject(E_QDQ_UNSUPPORTED_ATTR, "Split as an attribute is not supported",
                    _node_label(node))
        self.used_inits.add(node.input[1])
        outputs = [o for o in node.output if o]
        if len(sizes) != len(outputs):
            _reject(E_QDQ_PATTERN, "Split sizes %r vs %d outputs" % (sizes, len(outputs)),
                    _node_label(node))
        # IR ids/output names come from prepare_ir_ids() (structural, not the source
        # names); the ONNX-level names are still used for shape-inference lookups.
        nid = self._ir_id_of_node[idx]
        names = self._split_out_names[idx]
        in_ir = self.ir_name_of_u8(u8_in, node, "Split input")
        in_shape = self._shapes[in_ir]
        if int(sum(sizes)) != in_shape[0]:
            _reject(E_QDQ_SHAPE_MISMATCH, "Split sizes %r do not sum to input C %d"
                    % (sizes, in_shape[0]), _node_label(node))
        domain = self.domains[u8_in]
        out_entries = []
        offset = 0
        for j, size in enumerate(sizes):
            name = names[j]
            out_entries.append({"name": name, "channel_offset": offset, "channels": size})
            tshape = [size, in_shape[1], in_shape[2]]
            inferred = self.inferred_chw(outputs[j], node, "Split output")
            self.shape_checks.append({"node": name, "derived": list(tshape),
                                      "inferred": list(inferred),
                                      "ok": list(tshape) == list(inferred)})
            if list(tshape) != list(inferred):
                _reject(E_QDQ_SHAPE_MISMATCH,
                        "Split output %r derived shape %r != ONNX inferred %r"
                        % (name, tshape, inferred), _node_label(node))
            self._shapes[name] = tshape
            self.tensors_ir[name] = {"shape": tshape, "domain_node": in_ir,
                                     "logical_c": int(size)}
            self.split_outputs.append(name)
            offset += size
        ir = {
            "id": nid,
            "op": "Split",
            "inputs": [in_ir],
            "attrs": {"sizes": sizes, "axis": 1, "outputs": out_entries},
            "out_shape": list(in_shape),
            "quant": {"out_scale": domain["scale"],
                      "out_zero_points": [domain["zero_point"]],
                      "in_scales": [domain["scale"]],
                      "in_zero_points": [domain["zero_point"]]},
        }
        return ir

    # -- bookkeeping -------------------------------------------------------- #
    def _record_shape(self, nid, u8_out, out_shape, node):
        inferred = self.inferred_chw(u8_out, node, "%s output" % node.op_type)
        self.shape_checks.append({"node": nid, "derived": list(out_shape),
                                  "inferred": list(inferred),
                                  "ok": list(out_shape) == list(inferred)})
        if list(out_shape) != list(inferred):
            _reject(E_QDQ_SHAPE_MISMATCH, "node %r derived shape %r != ONNX inferred %r"
                    % (nid, list(out_shape), list(inferred)), _node_label(node))
        self._shapes[nid] = [int(v) for v in out_shape]
        self.tensors_ir[nid] = {"shape": [int(v) for v in out_shape],
                                "domain_node": nid, "logical_c": int(out_shape[0])}

    def _check_dangling(self):
        self.graph_output_u8 = {o.name for o in self.graph_pb.output}
        for idx, node in enumerate(self.nodes):
            if node.op_type == "QuantizeLinear" and idx not in self.used_q:
                _reject(E_QDQ_DANGLING, "QuantizeLinear node is not part of any pattern",
                        _node_label(node))
            if node.op_type == "Clip" and idx not in self.used_clip:
                _reject(E_QDQ_DANGLING, "Clip node is not part of any pattern",
                        _node_label(node))
            if node.op_type == "DequantizeLinear" and idx not in self.used_dq:
                src = node.input[0]
                out = node.output[0]
                consumers = [c for c in self.consumer_list(out)]
                if consumers or src not in self.domains:
                    _reject(E_QDQ_DANGLING, "DequantizeLinear node is not part of any pattern",
                            _node_label(node))
                split_consumer = any(self.nodes[i].op_type == "Split"
                                     for i, _ in self.consumer_list(src))
                graph_output = src in self.graph_output_u8
                if not split_consumer and not graph_output:
                    _reject(E_QDQ_DANGLING,
                            "unused DequantizeLinear whose u8 source is neither split-consumed "
                            "nor a raw graph output", _node_label(node))
                reason = ("a per-tensor DQ may be emitted for a Split input; the split path "
                          "consumes the u8 tensor directly")
                if graph_output and not split_consumer:
                    reason = ("raw graph output: the u8 QuantizeLinear output is the graph "
                              "boundary, so the per-tensor DQ has no integer consumer")
                self.dangling_dq.append({"dq_node": _node_label(node), "u8_tensor": src,
                                         "kind": ("split_input" if split_consumer else "graph_output"),
                                         "reason": reason})

    def _settle_outputs(self):
        """Resolve graph outputs structurally.

        Any graph output produced by a supported op is accepted (Conv including its
        fused ReLU, Add, Concat, MaxPool, Resize, or a Split output); what is
        enforced is the explicit boundary: the tensor must be uint8, part of a
        recognised Q/DQ pattern and carry a recorded quantisation domain.  No
        output count, op-kind or head-geometry rule lives here.
        """
        outputs = []
        for out in self.graph_pb.output:
            dtype = self.elem_type(out.name, None, "graph output")
            if dtype != UINT8:
                _reject(E_QDQ_GRAPH_IO, "graph output %r must be uint8, got elem_type %d"
                        % (out.name, dtype))
            name = self._ir_name_of_u8.get(out.name)
            if name is None or name == "__input__":
                _reject(E_QDQ_GRAPH_IO,
                        "graph output %r is not produced by a supported quantised op"
                        % out.name)
            if out.name not in self.domains:
                _reject(E_QDQ_DOMAIN_MISMATCH,
                        "graph output %r has no explicit quantisation domain" % out.name)
            outputs.append(name)
        self.outputs_ir = outputs
    # -- asset builders ----------------------------------------------------- #
    def graph_name(self):
        """Deployment graph name from the ONNX metadata (stable generic fallback)."""
        name = (self.graph_pb.name or "").strip()
        if not name:
            name = (self.model.producer_name or "").strip()
        name = _ID_CHARS.sub("-", name).strip("-._")
        return name or "qdq-graph"

    def _build_graph(self):
        self.counts = {}
        for node in self.nodes_ir:
            self.counts[node["op"]] = self.counts.get(node["op"], 0) + 1
        return {
            "schema": GRAPH_SCHEMA,
            "profile": PROFILE,
            "model": {"name": self.graph_name(),
                      "input_h": int(self._shapes["__input__"][1]),
                      "input_w": int(self._shapes["__input__"][2])},
            "input": {"name": self.input_name, "dtype": "uint8",
                      "shape": [1, int(self._shapes["__input__"][0]),
                                int(self._shapes["__input__"][1]),
                                int(self._shapes["__input__"][2])],
                      "scale": self.domains[self.input_u8]["scale"],
                      "zero_point": self.domains[self.input_u8]["zero_point"]},
            "outputs": list(self.outputs_ir),
            "nodes": [dict(n) for n in self.nodes_ir],
            "tensors": {k: dict(v) for k, v in self.tensors_ir.items()},
        }

    def _build_quant_and_weights(self):
        weights = {}
        conv_entries = {}
        tensors = {}
        for node in self.nodes_ir:
            nid = node["id"]
            q = node["quant"]
            if node["op"] == "Conv":
                pack = self._conv_pack[nid]
                w = np.ascontiguousarray(pack["w"], dtype=np.int8)
                b = np.ascontiguousarray(pack["b"], dtype=np.int32)
                zs_in = int(q["in_zero_points"][0]) - 128
                folding = conv_bias_fold(w, b, zs_in, cin=int(node["attrs"]["cin"]))
                check_acc_int32(int(folding["acc_bound"]))
                m0s = choose_m0_shift(float(q["in_scales"][0]), float(q["w_scale"]),
                                      float(q["out_scale"]))
                check_rescale_int64(int(folding["acc_bound"]), int(m0s["m0"]))
                bq = folding["bias_folded"].astype(np.int64)
                weights[nid + ".w"] = w
                weights[nid + ".b"] = b
                weights[nid + ".bq"] = np.ascontiguousarray(bq, dtype=np.int64)
                conv_entries[nid] = {
                    "w_scale": pack["w_scale"],
                    "w_zero_point": int(pack["w_zero_point"]),
                    "sum_w": [int(v) for v in folding["sum_w"].reshape(-1)],
                    "acc_bound": int(folding["acc_bound"]),
                    "k_size": int(folding["k_size"]),
                    "bias_int32_sha256": _sha256_array(b),
                    "bias_folded_sha256": _sha256_array(bq),
                    "bias_present": bool(pack["bias_present"]),
                    "bias_frac_dev": None,
                    "m0_shift": {k: (int(v) if isinstance(v, (int, np.integer)) and not isinstance(v, bool)
                                     else (float(v) if isinstance(v, (float, np.floating)) else v))
                                 for k, v in m0s.items()},
                }
                tensors[nid] = {"out_scale": q["out_scale"],
                                "out_zero_point": int(q["out_zero_points"][0]),
                                "logical_c": int(node["out_shape"][0])}
            elif node["op"] != "Split":
                tensors[nid] = {"out_scale": q["out_scale"],
                                "out_zero_point": int(q["out_zero_points"][0]),
                                "logical_c": int(node["out_shape"][0])}
        for name in self.split_outputs:
            t = self.tensors_ir[name]
            domain_node = t["domain_node"]
            node = self._node_by_ir_id[domain_node]
            tensors[name] = {"out_scale": node["quant"]["out_scale"],
                             "out_zero_point": int(node["quant"]["out_zero_points"][0]),
                             "logical_c": int(t["logical_c"])}
        quant = {
            "schema": QUANT_SCHEMA,
            "profile": PROFILE,
            "rounding": ROUNDING,
            "requant": {"m0_bits": int(M0_BITS), "shift_max": int(SHIFT_MAX)},
            "tensors": tensors,
            "conv": conv_entries,
            "frontend": {"source": FRONTEND_SCHEMA,
                         "note": "packing evidence re-derived from model.onnx; the exporter"
                                 " capture fields (bias_frac_dev/bias_present) are not"
                                 " recoverable from a quantised ONNX and stay null/derived"},
        }
        return quant, weights

    def _build_report(self):
        onnx_info = {"opset": int(self.opset),
                     "ir_version": int(self.model.ir_version),
                     "producer": "%s %s" % (self.model.producer_name, self.model.producer_version),
                     "n_nodes": len(self.nodes),
                     "n_initializers": len(self.inits),
                     "input": self.input_name,
                     "outputs": list(self.outputs_ir)}
        if self.onnx_path is not None:
            raw = Path(self.onnx_path).read_bytes()
            onnx_info.update({"path": self.onnx_path, "bytes": len(raw),
                              "sha256": _sha256_bytes(raw)})
        source_map = {
            "input": {"ir": "__input__", "onnx_input": self.input_name,
                      "onnx_u8_tensor": self.input_u8},
            "outputs": [{"ir": ir, "onnx_u8_tensor": out.name}
                        for ir, out in zip(self.outputs_ir, self.graph_pb.output)],
            "nodes": [dict(entry) for entry in self.source_nodes],
            "tensors": {ir: u8 for u8, ir in self._ir_name_of_u8.items()},
        }
        return {
            "schema": FRONTEND_SCHEMA,
            "profile": PROFILE,
            "onnx": onnx_info,
            "model": {"name": self.graph_name(),
                      "input_h": int(self._shapes["__input__"][1]),
                      "input_w": int(self._shapes["__input__"][2])},
            "counts": {k: int(v) for k, v in sorted(self.counts.items())},
            "nodes": len(self.nodes_ir),
            "tensors": len(self.tensors_ir),
            "conv_packed": len(self._conv_pack),
            "shape_checks": {"checked": len(self.shape_checks),
                             "mismatches": [c for c in self.shape_checks if not c["ok"]]},
            "dangling_dq": self.dangling_dq,
            "unused_initializers": sorted(set(self.init_proto) - self.used_inits),
            "source_map": source_map,
            "rejections": [],
        }


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def parse_qdq_onnx(source, strict_counts=True):
    """Parse a QDQ ONNX model (path or onnx.ModelProto) into the deployment IR.

    Returns {"ok": True, "graph": ..., "quant": ..., "weights": ..., "report": ...}.
    Raises QDQReject (structured) on any unsupported pattern.  Never falls back.
    The parser is model-agnostic; strict_counts is a deprecated compatibility flag
    (model identity assertions live in examples/yolov8n_relu_voc20/identity_check.py).
    """
    onnx, _ = _import_onnx()
    onnx_path = None
    if isinstance(source, (str, Path)):
        onnx_path = str(source)
        model = onnx.load(onnx_path)
    else:
        model = source
    if any(n.op_type == "MaxPool" and n.domain in ("", "ai.onnx") for n in model.graph.node):
        try:
            onnx.checker.check_model(model, full_check=True)
        except Exception as exc:
            _reject(E_QDQ_UNSUPPORTED_ATTR, "ONNX MaxPool schema/type check failed: %s" % exc)
    frontend = QDQFrontend(model, onnx_path=onnx_path)
    return frontend.parse(strict_counts=strict_counts)


def write_assets(parsed, out_dir):
    """Write graph.json / quant_table.json / weights.npz (the normalized IR assets)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, payload in (("graph.json", parsed["graph"]), ("quant_table.json", parsed["quant"])):
        raw = (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8")
        (out / name).write_bytes(raw)
        written[name] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    weights = parsed["weights"]
    npz_path = out / "weights.npz"
    np.savez(str(npz_path), **{k: weights[k] for k in sorted(weights)})
    raw = npz_path.read_bytes()
    written["weights.npz"] = {"bytes": len(raw), "sha256": _sha256_bytes(raw)}
    return written


def compare_graph(front, ref, max_items=12):
    """Field-level comparison of two deployment graphs (parsed vs reference asset)."""
    diffs = []
    if list(front.keys()) != list(ref.keys()):
        diffs.append({"where": "top_keys", "front": list(front.keys()), "ref": list(ref.keys())})
    for key in ("schema", "profile", "model", "input", "outputs"):
        if front.get(key) != ref.get(key):
            diffs.append({"where": key, "front": front.get(key), "ref": ref.get(key)})
        if len(diffs) >= max_items:
            return {"ok": not diffs, "diffs": diffs}
    f_nodes = front.get("nodes", [])
    r_nodes = ref.get("nodes", [])
    if len(f_nodes) != len(r_nodes):
        diffs.append({"where": "nodes.length", "front": len(f_nodes), "ref": len(r_nodes)})
    for i, (fn, rn) in enumerate(zip(f_nodes, r_nodes)):
        for field in ("id", "op", "inputs", "attrs", "out_shape", "quant"):
            if fn.get(field) != rn.get(field):
                diffs.append({"where": "nodes[%d].%s" % (i, field), "node": rn.get("id"),
                              "front": fn.get(field), "ref": rn.get(field)})
                if len(diffs) >= max_items:
                    return {"ok": False, "diffs": diffs}
    f_t, r_t = front.get("tensors", {}), ref.get("tensors", {})
    if set(f_t) != set(r_t):
        diffs.append({"where": "tensors.names",
                      "missing": sorted(set(r_t) - set(f_t)),
                      "extra": sorted(set(f_t) - set(r_t))})
    for name in sorted(set(f_t) & set(r_t)):
        if f_t[name] != r_t[name]:
            diffs.append({"where": "tensors[%r]" % name, "front": f_t[name], "ref": r_t[name]})
            if len(diffs) >= max_items:
                return {"ok": False, "diffs": diffs}
    return {"ok": not diffs, "diffs": diffs}


def compare_graph_topology(front, ref, max_items=12):
    """Compare deployment topology + domains; report the model metadata block apart.

    The model metadata block is deliberately not a topology identity: the generic
    frontend takes the graph name from the ONNX metadata and does not derive
    num_classes/reg_max (example identity checks own those).  Everything else
    (schema, profile, input, outputs, node order/ids/ops/inputs/attrs/shapes/
    domains, tensor map) is compared exactly.
    """
    front = dict(front)
    ref = dict(ref)
    front_model = front.pop("model", None)
    ref_model = ref.pop("model", None)
    result = compare_graph(front, ref, max_items=max_items)
    result["model"] = {"front": front_model, "ref": ref_model,
                       "delta": front_model != ref_model}
    result["topology_has_model_key"] = bool(front_model is not None and ref_model is not None)
    return result


def compare_weights(front, ref, max_items=8):
    """Compare two weight dicts (arrays) key by key: names, dtype, shape, content."""
    diffs = []
    f_keys, r_keys = set(front), set(ref)
    if f_keys != r_keys:
        diffs.append({"where": "keys", "missing": sorted(r_keys - f_keys),
                      "extra": sorted(f_keys - r_keys)})
    for name in sorted(f_keys & r_keys):
        a = np.asarray(front[name])
        b = np.asarray(ref[name])
        if a.dtype != b.dtype or a.shape != b.shape:
            diffs.append({"where": name, "front": [str(a.dtype), list(a.shape)],
                          "ref": [str(b.dtype), list(b.shape)]})
        elif not np.array_equal(a, b):
            diffs.append({"where": name, "mismatch": int((a != b).sum()),
                          "max_abs_diff": int(np.abs(a.astype(np.int64) - b.astype(np.int64)).max())})
        if len(diffs) >= max_items:
            break
    return {"ok": not diffs, "keys": len(f_keys & r_keys), "diffs": diffs}


def run_integer_reference(parsed, input_u8):
    """Execute the frontend IR with the independent integer reference; return u8 heads."""
    from tools.onnx_gen.int_reference import IntegerReference
    ref = IntegerReference(parsed["graph"], parsed["quant"], parsed["weights"])
    result = ref.run(input_u8)
    return {name: np.asarray(h["u8"], dtype=np.uint8) for name, h in result["heads"].items()}


def compare_heads(heads_u8, ref_npz_path):
    """Compare computed uint8 heads against a heads npz (exact byte equality)."""
    ref = dict(np.load(str(ref_npz_path), allow_pickle=False))
    report = {}
    ok = True
    for name in sorted(ref):
        want = np.asarray(ref[name], dtype=np.uint8)
        got = heads_u8.get(name)
        if got is None:
            report[name] = {"present": False, "equal": False}
            ok = False
            continue
        got = np.asarray(got, dtype=np.uint8)
        diff = got.astype(np.int64) - want.astype(np.int64)
        equal = bool(got.shape == want.shape and np.array_equal(got, want))
        report[name] = {"present": True, "equal": equal, "shape": list(got.shape),
                        "mismatch": int((diff != 0).sum()),
                        "max_abs_diff": int(np.abs(diff).max()) if diff.size else 0,
                        "u8_sha256": _sha256_array(got)}
        ok = ok and equal
    return {"ok": ok, "heads": report}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Static QDQ ONNX -> normalized affine "
                                 "deployment IR frontend (model-agnostic)")
    ap.add_argument("--onnx", required=True, help="QDQ ONNX model path")
    ap.add_argument("--out-dir", default=None, help="write graph.json/quant_table.json/weights.npz")
    ap.add_argument("--compare-graph", default=None,
                    help="reference graph.json: compare topology/domains exactly and "
                         "report the model metadata block separately")
    ap.add_argument("--compare-quant", default=None, help="reference quant_table.json")
    ap.add_argument("--compare-weights", default=None, help="reference weights.npz")
    ap.add_argument("--input-u8", default=None, help="uint8 input .npy for the int reference run")
    ap.add_argument("--heads-ref", default=None, help="reference heads npz (uint8)")
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    out = {"schema": FRONTEND_SCHEMA, "profile": PROFILE, "onnx": args.onnx}
    try:
        parsed = parse_qdq_onnx(args.onnx)
    except QDQReject as exc:
        out.update({"ok": False, "rejected": exc.as_dict()})
        if args.report:
            Path(args.report).parent.mkdir(parents=True, exist_ok=True)
            Path(args.report).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(out, indent=2))
        return 2
    out["ok"] = True
    out["parse"] = parsed["report"]
    if args.out_dir:
        out["written"] = write_assets(parsed, args.out_dir)
    if args.compare_graph:
        ref = json.loads(Path(args.compare_graph).read_text(encoding="utf-8"))
        out["compare_graph"] = compare_graph_topology(parsed["graph"], ref)
    if args.compare_quant:
        ref_quant = json.loads(Path(args.compare_quant).read_text(encoding="utf-8"))
        fields = ("w_scale", "w_zero_point", "sum_w", "acc_bound", "k_size",
                  "bias_folded_sha256")
        bad = []
        for nid, entry in parsed["quant"]["conv"].items():
            ref_entry = ref_quant.get("conv", {}).get(nid, {})
            for field in fields:
                if entry.get(field) != ref_entry.get(field):
                    bad.append({"node": nid, "field": field, "front": entry.get(field),
                                "ref": ref_entry.get(field)})
            for key in ("m0", "shift", "ratio_float", "rel_err"):
                if entry["m0_shift"].get(key) != ref_entry.get("m0_shift", {}).get(key):
                    bad.append({"node": nid, "field": "m0_shift." + key})
        tensors_ok = parsed["quant"]["tensors"] == ref_quant.get("tensors")
        out["compare_quant"] = {"ok": not bad and tensors_ok, "diffs": bad[:20],
                                "tensors_equal": tensors_ok}
    if args.compare_weights:
        ref_w = dict(np.load(args.compare_weights, allow_pickle=False))
        out["compare_weights"] = compare_weights(parsed["weights"], ref_w)
    if args.input_u8 and args.heads_ref:
        x = np.load(args.input_u8)
        if x.ndim == 4:
            x = x[0]
        heads = run_integer_reference(parsed, x)
        out["heads_vs_reference"] = compare_heads(heads, args.heads_ref)
    ok = all(out[k]["ok"] for k in ("compare_graph", "compare_quant", "compare_weights",
                                    "heads_vs_reference") if k in out)
    out["ok"] = bool(out["ok"] and ok)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if not args.quiet:
        summary = {k: out[k] for k in out if k in ("ok", "compare_graph", "compare_quant",
                                                   "compare_weights", "heads_vs_reference",
                                                   "written")}
        print(json.dumps(summary, indent=2)[:6000])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
