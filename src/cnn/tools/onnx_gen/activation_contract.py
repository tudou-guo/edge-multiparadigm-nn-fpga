"""Explicit LUT activation contract and SiLU-group deployment IR (S2).

New module for E1 LUT activation intake. Old no-extension load_onnx/ordinary-Mul
paths are unchanged; all entry points here require an explicit
e1-lut-activation-contract-1 document.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

from . import shape_infer as _shape_infer
from .graph_ir import CompileError, E_CFG, E_QUANT_MISMATCH, E_UNSUPPORTED_OP, Graph, Node, Tensor
from .onnx_frontend import load_onnx as _load_onnx_hardware

SCHEMA = "e1-lut-activation-contract-1"
SCALE_LO = Fraction(1, 65536)
SCALE_HI = Fraction(16, 1)

def _frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    f = float(v)
    if not math.isfinite(f):
        raise CompileError(E_QUANT_MISMATCH, "non-finite scale/alpha %r" % (v,))
    return Fraction.from_float(f)

def _check_scale(v, what: str) -> Fraction:
    fr = _frac(v)
    if not (SCALE_LO <= fr <= SCALE_HI):
        raise CompileError(E_QUANT_MISMATCH, "%s=%s outside 2^-16..16" % (what, fr))
    if fr <= 0:
        raise CompileError(E_QUANT_MISMATCH, "%s must be positive" % what)
    return fr

def _check_alpha(v) -> Fraction:
    fr = _frac(v)
    if not (Fraction(0) <= fr <= Fraction(1)):
        raise CompileError(E_QUANT_MISMATCH, "alpha=%s outside 0..1" % fr)
    return fr

def load_semantic(path: Path, *, name=None) -> Graph:
    """Parse ONNX with float SiLU-group members allowed (semantic IR only)."""
    prev = _shape_infer.SEMANTIC_FLOAT_OK
    _shape_infer.SEMANTIC_FLOAT_OK = True
    try:
        g = _load_onnx_hardware(path, name=name)
    finally:
        _shape_infer.SEMANTIC_FLOAT_OK = prev
    # Mark semantic float Mul nodes explicitly for downstream traceability.
    for n in g.nodes:
        if n.op_type == "Mul":
            try:
                a = g.tensors[n.inputs[0]]
                b = g.tensors[n.inputs[1]]
                if a.dtype in ("float32", "float16") and b.dtype in ("float32", "float16"):
                    n.metadata["semantic_float"] = True
            except Exception:
                pass
    return g

def _consumers(graph: Graph):
    cons: Dict[str, List[Tuple[str, int]]] = {}
    for n in graph.nodes:
        for idx, tn in enumerate(n.inputs):
            cons.setdefault(tn, []).append((n.id, idx))
    return cons

def _producers(graph: Graph):
    prod: Dict[str, str] = {}
    for n in graph.nodes:
        for o in n.outputs:
            prod[o] = n.id
    return prod

def recognize_silu_groups(graph: Graph):
    """Find Sigmoid->Mul restricted groups. Returns (groups, by_sigmoid, by_mul)."""
    cons = _consumers(graph)
    prod = _producers(graph)
    by_id = {n.id: n for n in graph.nodes}
    groups = []
    for n in graph.nodes:
        if n.op_type != "Sigmoid":
            continue
        s = n.outputs[0]
        x = n.inputs[0]
        # s must have exactly one consumer, a Mul, not a graph output
        uses = cons.get(s, [])
        if len(uses) != 1:
            continue
        mul_id, _ = uses[0]
        m = by_id.get(mul_id)
        if m is None or m.op_type != "Mul":
            continue
        if s in graph.outputs:
            continue
        # Mul inputs must be (x,s) in either order, same tensor x
        if not ((m.inputs[0] == x and m.inputs[1] == s) or (m.inputs[0] == s and m.inputs[1] == x)):
            continue
        # x shapes/dtypes: both Sigmoid in/out and Mul out share shape (checked by infer)
        # s must not be graph output (already) and must not be consumed elsewhere (single use ok)
        groups.append({"sigmoid": n.id, "mul": m.id, "x": x, "s": s, "y": m.outputs[0],
                       "sigmoid_node": n, "mul_node": m})
    return groups

def validate_activation_contract(graph: Graph, contract: Dict[str, Any], groups=None):
    if contract.get("schema") != SCHEMA:
        raise CompileError(E_CFG, "unsupported activation contract schema %r" % contract.get("schema"))
    tensors = contract.get("tensors")
    if not isinstance(tensors, dict):
        raise CompileError(E_CFG, "activation contract tensors must be an object")
    lut_groups = contract.get("activation_lut_groups", [])
    if not isinstance(lut_groups, list):
        raise CompileError(E_CFG, "activation_lut_groups must be a list")
    groups = groups if groups is not None else recognize_silu_groups(graph)
    # Every declared group must correspond to a recognized group; every recognized group that is
    # intended to fuse must be declared (no silent fusion).
    recognized_pairs = {(g["sigmoid"], g["mul"]) for g in groups}
    for entry in lut_groups:
        if entry.get("kind") != "SiLU" or entry.get("semantic") != "unrounded_silu":
            raise CompileError(E_CFG, "only SiLU/unrounded_silu groups supported in this batch")
        # find matching recognized group by source_nodes or x/y names
        src = entry.get("source_nodes", [])
        # allow matching by (sigmoid,mul) ids or by x/y tensor names
        matched = None
        for g in groups:
            if g["sigmoid"] in src and g["mul"] in src:
                matched = g
                break
            if entry.get("input") == g["x"] and entry.get("output") == g["y"]:
                matched = g
                break
        if matched is None:
            raise CompileError(E_CFG, "declared SiLU group has no recognized Sigmoid->Mul pair: %r" % entry)
        # intermediate must be the group's s and must have NO tensor contract entry
        if entry.get("intermediate") != matched["s"]:
            raise CompileError(E_CFG, "SiLU intermediate must be Sigmoid output %s" % matched["s"])
        if matched["s"] in tensors:
            raise CompileError(E_QUANT_MISMATCH, "SiLU intermediate %s must not have activation scale" % matched["s"])
        # x/y must have explicit INT8 entries
        for tn in (matched["x"], matched["y"]):
            row = tensors.get(tn)
            if not isinstance(row, dict):
                raise CompileError(E_QUANT_MISMATCH, "missing activation contract for %s" % tn)
            if str(row.get("dtype", "int8")) != "int8":
                raise CompileError(E_QUANT_MISMATCH, "tensor %s must be int8" % tn)
            if int(row.get("zero_point", 0)) != 0:
                raise CompileError(E_QUANT_MISMATCH, "tensor %s zp must be 0" % tn)
            _check_scale(row.get("scale"), "scale for %s" % tn)
    # No other tensor may claim non-materialized s scale; s absence already checked.
    return {"groups": groups, "declared": lut_groups}

def build_deployment_ir(graph: Graph, contract: Dict[str, Any], *, infer_shapes=True,
                        caps: Optional[Dict[str, Any]] = None):
    """Build INT8 deployment IR with canonical ActivationLUT nodes.

    Returns (deployed_graph, report). Semantic float Sigmoid/Mul groups become
    single ActivationLUT(SiLU); independent float Sigmoid/Tanh/LeakyRelu become
    single ActivationLUT of that mode. Old default paths without this contract
    never enter here.
    """
    from copy import deepcopy
    groups = recognize_silu_groups(graph)
    validation = validate_activation_contract(graph, contract, groups)
    tensors_cfg = contract.get("tensors", {})
    lut_groups = contract.get("activation_lut_groups", [])
    # Map sigmoid/mul ids that are fused
    fused_sigmoid = set()
    fused_mul = set()
    group_by_pair = {}
    for entry in lut_groups:
        for g in groups:
            if (g["sigmoid"] in entry.get("source_nodes", []) and g["mul"] in entry.get("source_nodes", [])) or \
               (entry.get("input") == g["x"] and entry.get("output") == g["y"] and entry.get("intermediate") == g["s"]):
                fused_sigmoid.add(g["sigmoid"])
                fused_mul.add(g["mul"])
                group_by_pair[g["mul"]] = g
                break
    # Clone tensors: deployment tensors are INT8 for contracted names
    new_tensors: Dict[str, Tensor] = {}
    for name, t in graph.tensors.items():
        nt = Tensor(name, dtype=t.dtype, shape=t.shape, kind=t.kind, layout=t.layout,
                    producer=None, consumers=[], storage=None, quantization=None,
                    storage_id=None, source=dict(t.source or {}), logical_bytes=None)
        new_tensors[name] = nt
    # Apply INT8 quantization to contracted activation tensors
    for name, row in tensors_cfg.items():
        if name not in new_tensors:
            raise CompileError(E_CFG, "contract tensor %s not in graph" % name)
        nt = new_tensors[name]
        # Only rank-4 activations are supported for the LUT batch.  The legacy
        # limit is C<=16/H,W<=32; a v0.2 spatial build passes its versioned caps
        # (H/W up to the allocated/640 ceiling) so the explicit SiLU table path
        # works on large feature maps without weakening the legacy gate.
        h_cap, w_cap, ch_cap = 32, 32, 16
        if caps:
            h_cap = int(caps.get("feature_max_h", h_cap))
            w_cap = int(caps.get("feature_max_w", caps.get("width_ceiling", w_cap)))
        if nt.shape is not None and len(nt.shape) == 4:
            if not (nt.shape[0] == 1 and 1 <= nt.shape[1] <= ch_cap and
                    1 <= nt.shape[2] <= h_cap and 1 <= nt.shape[3] <= w_cap):
                raise CompileError(E_CFG, "contract tensor %s shape %s outside the "
                                          "activation capacity C<=%d H<=%d W<=%d" %
                                   (name, nt.shape, ch_cap, h_cap, w_cap))
        nt.dtype = "int8"
        nt.layout = "NCHW"
        nt.quantization = {"dtype": "int8", "scale": float(_frac(row.get("scale"))), "zero_point": 0, "materialized": True}
    # Build nodes: replace fused pairs and independent activations
    new_nodes: List[Node] = []
    # index independent activation nodes (not fused)
    for n in graph.nodes:
        if n.id in fused_sigmoid:
            continue  # eliminated
        if n.id in fused_mul:
            g = group_by_pair[n.id]
            # emit canonical ActivationLUT SiLU
            new_nodes.append(Node(id="lut_%s" % n.id, op_type="ActivationLUT", name="SiLU_%s" % n.id,
                                  inputs=[g["x"]], outputs=[g["y"]],
                                  attrs={"mode": "SiLU"}, placement="pl",
                                  lowering="activation_lut_table_v1",
                                  source_index=n.source_index, metadata_only=False,
                                  metadata={"source_nodes": [g["sigmoid"], g["mul"]],
                                            "mode": "SiLU", "semantic": "unrounded_silu",
                                            "opcode": 0x001B}))
            continue
        if n.op_type in ("Sigmoid", "Tanh", "LeakyRelu"):
            # independent: must not be a fused sigmoid (already skipped); convert to LUT
            mode = n.op_type
            attrs = {"mode": mode}
            if mode == "LeakyRelu":
                alpha = float(n.attrs.get("alpha", 0.01))
                _check_alpha(alpha)
                attrs["alpha"] = alpha
            new_nodes.append(Node(id="lut_%s" % n.id, op_type="ActivationLUT", name="%s_%s" % (mode, n.id),
                                  inputs=list(n.inputs), outputs=list(n.outputs),
                                  attrs=attrs, placement="pl",
                                  lowering="activation_lut_table_v1",
                                  source_index=n.source_index, metadata_only=False,
                                  metadata={"source_nodes": [n.id], "mode": mode, "opcode": 0x001B}))
            continue
        if n.op_type == "Mul":
            # Unfused Mul stays as hardware Mul (must be INT8 deployment; semantic float Mul
            # that was not fused is an error: it would mean a float Mul leaked to deployment).
            # Check dtypes: deployment tensors for its inputs must be int8 now.
            a = new_tensors[n.inputs[0]]
            b = new_tensors[n.inputs[1]]
            if a.dtype != "int8" or b.dtype != "int8":
                raise CompileError(E_QUANT_MISMATCH,
                                   "unfused Mul %s has non-INT8 deployment inputs; declare SiLU group or fix contract" % n.id,
                                   node_id=n.id)
            # keep node as-is (hardware Mul gate will validate scales later)
            new_nodes.append(Node(id=n.id, op_type=n.op_type, name=n.name, inputs=list(n.inputs),
                                  outputs=list(n.outputs), attrs=dict(n.attrs), placement=n.placement,
                                  lowering=n.lowering, source_index=n.source_index,
                                  metadata_only=n.metadata_only, metadata=dict(n.metadata)))
            continue
        # other ops (Conv/Concat/GAP etc.): keep, but deployment dtypes for activations already INT8
        # Conv weights (initializers) stay as-is; explicit INT8 mapping is validated in S2 compile.
        new_nodes.append(Node(id=n.id, op_type=n.op_type, name=n.name, inputs=list(n.inputs),
                              outputs=list(n.outputs), attrs=dict(n.attrs), placement=n.placement,
                              lowering=n.lowering, source_index=n.source_index,
                              metadata_only=n.metadata_only, metadata=dict(n.metadata)))
    # Remove dead intermediate s tensors that were fused (keep trace in report, not in IR)
    fused_s_tensors = {g["s"] for g in groups if g["sigmoid"] in fused_sigmoid}
    for s in fused_s_tensors:
        new_tensors.pop(s, None)
    # Fix initializer/input/output lists: drop removed s if it was initializer (should not be)
    new_initializers = [x for x in graph.initializers if x in new_tensors]
    # Validate no dangling inputs
    deployed = Graph(name=graph.name + "_deployed_lut", inputs=list(graph.inputs), outputs=list(graph.outputs),
                     initializers=new_initializers, tensors=new_tensors, nodes=new_nodes,
                     opset_imports=list(graph.opset_imports),
                     layout_policy=dict(graph.layout_policy),
                     source={"format": "onnx-deployment-lut", "model_sha256": (graph.source or {}).get("model_sha256"),
                             "semantic_name": graph.name, "contract_schema": SCHEMA},
                     metadata={"deployment": "lut_activation_v1", "fused_groups": len(fused_sigmoid)})
    # Hardware shape gate for deployment (int8 only, no semantic flag)
    from .shape_infer import infer_graph as _infer_hw
    if infer_shapes:
        _infer_hw(deployed)
    report = {"fused": len(fused_sigmoid), "groups": [{"sigmoid": g["sigmoid"], "mul": g["mul"], "x": g["x"], "s": g["s"], "y": g["y"]} for g in groups if g["sigmoid"] in fused_sigmoid],
              "eliminated_tensors": sorted(fused_s_tensors)}
    return deployed, report
