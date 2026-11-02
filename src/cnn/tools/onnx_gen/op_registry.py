"""Operator capability registry for the E1 static ONNX subset.

Registration is intentionally data-driven.  Adding an operator requires a
registry entry, canonical attribute normalization, shape inference, and (later)
lowering/reference hooks.  Unregistered operators never silently fall back.

E1-P1 adds the static Pad / Slice / Concat subset.  Their hardware opcodes are
reserved here; lowering and payload details live in the planner/codegen.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .graph_ir import (
    CompileError,
    E_UNSUPPORTED_ATTR,
    E_UNSUPPORTED_OP,
    E_UNSUPPORTED_OPSET,
)

MIN_OPSET = 11
MAX_OPSET = 19


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    min_opset: int
    max_opset: int
    placement: str
    lowering: str
    metadata_only: bool
    allowed_attrs: tuple
    opcode: Optional[int]
    capability: str
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "opset": [self.min_opset, self.max_opset],
            "placement": self.placement,
            "lowering": self.lowering,
            "metadata_only": self.metadata_only,
            "allowed_attrs": list(self.allowed_attrs),
            "opcode": self.opcode,
            "capability": self.capability,
            "summary": self.summary,
        }


_DEFAULT_SPECS = (
    OperatorSpec(
        "Conv", MIN_OPSET, MAX_OPSET, "pl", "dense3x3_core_v1", False,
        ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"),
        0x0001, "frontend_lowering_codegen_rtl_verified",
        "dense 3x3; 1x1 lowered to centered 3x3; stride 1/2, pad 0/1, groups=1, N=1",
    ),
    OperatorSpec(
        "MaxPool", MIN_OPSET, MAX_OPSET, "pl", "maxpool2x2_core_v1", False,
        ("auto_pad", "ceil_mode", "dilations", "kernel_shape", "pads",
         "storage_order", "strides"),
        0x0002, "frontend_codegen_rtl_verified",
        "2x2 stride 2 pad 0 even H/W, or 5x5 stride 1 pad 2 shape-preserving; N=1",
    ),
    OperatorSpec(
        "Relu", MIN_OPSET, MAX_OPSET, "pl", "eltwise_core_v1", False,
        tuple(), 0x0010, "frontend_codegen_rtl_verified",
        "elementwise signed INT8 clamp(min=0), zero-point 0",
    ),
    OperatorSpec(
        "Add", MIN_OPSET, MAX_OPSET, "pl", "eltwise_same_shape_add", False,
        tuple(), 0x0011, "frontend_codegen_rtl_verified",
        "same shape/dtype/layout signed INT8 saturating add; no ONNX broadcasting",
    ),
    OperatorSpec(
        "Pad", MIN_OPSET, MAX_OPSET, "pl", "spatial_pad_copy", False,
        ("mode",), 0x0012, "frontend_lowering_codegen_rtl_verified",
        "constant zero-point pad through copy core; rank-4 NCHW N=1 INT8; spatial H/W only",
    ),
    OperatorSpec(
        "Slice", MIN_OPSET, MAX_OPSET, "pl", "slice_view_or_copy", False,
        tuple(), 0x0013, "frontend_lowering_codegen_rtl_verified",
        "static step=1 slice on H/W/C; H-row view or explicit copy",
    ),
    OperatorSpec(
        "Concat", MIN_OPSET, MAX_OPSET, "pl", "channel_concat_copy", False,
        ("axis",), 0x0014, "frontend_lowering_codegen_rtl_verified",
        "channel axis=1 concat of 2..4 same-layout/quant INT8 inputs",
    ),
    OperatorSpec(
        "Clip", MIN_OPSET, MAX_OPSET, "pl", "clip_core_v1", False,
        ("min", "max"), 0x0015, "frontend_lowering_codegen_rtl",
        "INT8 min/max clamp with static bounds; zero-point 0; active bytes only",
    ),
    OperatorSpec(
        "Split", MIN_OPSET, MAX_OPSET, "pl", "static_split_to_slice_records", False,
        ("axis",), None, "frontend_lowering_codegen",
        "static rank-4 N=1 channel split into 2..4 positive Slice copies",
    ),
    OperatorSpec(
        "GlobalAveragePool", MIN_OPSET, MAX_OPSET, "pl", "global_avg_pool_core_v1", False,
        tuple(), 0x0016, "frontend_lowering_codegen_rtl",
        "rank-4 N=1 INT8 channel mean; same input/output scale; exact rounding",
    ),
    OperatorSpec(
        "AveragePool", MIN_OPSET, MAX_OPSET, "pl", "avgpool2x2_core_v1", False,
        ("auto_pad", "ceil_mode", "count_include_pad", "dilations", "kernel_shape",
         "pads", "strides"),
        0x0017, "frontend_lowering_codegen_rtl",
        "2x2 stride 2 pad 0, even H/W, N=1; INT32 sum of four, ties-away by 4, INT8 saturate",
    ),
    OperatorSpec(
        "Mul", MIN_OPSET, MAX_OPSET, "pl", "int8_mul_core_v1", False,
        tuple(), 0x0018, "frontend_lowering_codegen_rtl",
        "same shape/layout/scale INT8 x*y; exact M0<=0xFFFFFF, SHIFT<=31 requant; no broadcast",
    ),
    OperatorSpec(
        "Resize", MIN_OPSET, MAX_OPSET, "pl", "resize_nearest2_core_v1", False,
        ("antialias", "axes", "coordinate_transformation_mode", "cubic_coeff_a",
         "exclude_outside", "extrapolation_value", "keep_aspect_ratio_policy",
         "mode", "nearest_mode"),
        0x0019, "frontend_codegen_rtl",
        "nearest x2 asymmetric/floor upsample, same scale/zp0, output H/W <= 32",
    ),
    OperatorSpec(
        "BatchNormalization", MIN_OPSET, MAX_OPSET, "pl", "fold_into_conv_requant", False,
        ("epsilon", "momentum", "spatial", "training_mode"),
 None,
        "frontend_lowering_only",
        "inference-only Conv->BN fold with exact integer equivalence; emits no hardware record",
    ),
    OperatorSpec(
        "Sigmoid", MIN_OPSET, MAX_OPSET, "pl", "activation_lut_table_v1", False,
        tuple(), 0x001B, "frontend_semantic_deployment_only",
        "default-domain Sigmoid; semantic float, deployment INT8 LUT via explicit contract",
    ),
    OperatorSpec(
        "Tanh", MIN_OPSET, MAX_OPSET, "pl", "activation_lut_table_v1", False,
        tuple(), 0x001B, "frontend_semantic_deployment_only",
        "default-domain Tanh; semantic float, deployment INT8 LUT via explicit contract",
    ),
    OperatorSpec(
        "LeakyRelu", MIN_OPSET, MAX_OPSET, "pl", "activation_lut_table_v1", False,
        ("alpha",), 0x001B, "frontend_semantic_deployment_only",
        "default-domain LeakyRelu static alpha 0..1; semantic float, deployment INT8 LUT",
    ),
    OperatorSpec(
        "ActivationLUT", MIN_OPSET, MAX_OPSET, "pl", "activation_lut_table_v1", False,
        ("mode", "alpha"), 0x001B, "deployment_canonical",
        "canonical INT8 ActivationLUT record; mode Sigmoid/Tanh/SiLU/LeakyRelu; hw 0x001B 17-word",
    ),
    OperatorSpec(
        "DepthwiseConv", MIN_OPSET, MAX_OPSET, "pl", "depthwise_conv3x3_core_v1", False,
        ("auto_pad", "dilations", "group", "kernel_shape", "pads", "strides"),
        0x001C, "frontend_lowering_codegen",
        "depthwise 3x3 group=Cin=Cout=C multiplier 1; stride 1/2 pad 0/1; int32 bias; "
        "N=1 C<=16 H/W<=32; hw 0x001C 17-word",
    ),
    OperatorSpec(
        "Constant", MIN_OPSET, MAX_OPSET, "metadata", "compile_time_constant_fold", True,
        tuple(), None, "frontend_lowering",
        "dense compile-time tensor/scalar; folded before hardware planning",
    ),
    OperatorSpec(
        "Identity", MIN_OPSET, MAX_OPSET, "metadata", "metadata_identity_alias", True,
        tuple(), None, "frontend_lowering",
        "dtype/layout-preserving alias; no hardware record",
    ),
    OperatorSpec(
        "Squeeze", MIN_OPSET, MAX_OPSET, "metadata", "metadata_squeeze_alias", True,
        ("axes",), None, "frontend_lowering",
        "static removal of size-1 axes; PL use requires canonical descriptor",
    ),
    OperatorSpec(
        "Unsqueeze", MIN_OPSET, MAX_OPSET, "metadata", "metadata_unsqueeze_alias", True,
        ("axes",), None, "frontend_lowering",
        "static insertion of size-1 axes; PL use requires canonical descriptor",
    ),
    OperatorSpec(
        "Reshape", MIN_OPSET, MAX_OPSET, "metadata", "metadata_static_alias", True,
        ("allowzero",), None, "frontend_lowering",
        "static shape alias; physical compatibility checked by layout planning",
    ),
    OperatorSpec(
        "Flatten", MIN_OPSET, MAX_OPSET, "metadata", "metadata_only_candidate", True,
        ("axis",), None, "frontend_lowering",
        "row-major shape-only flatten; physical compatibility checked by layout planning",
    ),
    OperatorSpec(
        "Gemm", MIN_OPSET, MAX_OPSET, "ps", "ps_host", False,
        ("alpha", "beta", "transA", "transB"), None, "host_ps",
        "PS-only rank-2 Gemm; no PL opcode in E1",
    ),
)

REGISTRY: Dict[str, OperatorSpec] = {spec.name: spec for spec in _DEFAULT_SPECS}

# Inputs that are compile-time metadata, not runtime activation buffers.
# Index 0 is always the data input for these operators.
CONTROL_INPUT_INDICES = {
    "Resize": (1, 2, 3),  # roi, scales, sizes (compile-time metadata only)
    "Pad": (1, 2),       # pads, constant_value
    "Slice": (1, 2, 3, 4),  # starts, ends, axes, steps
    "Reshape": (1,),     # shape
    "Clip": (1, 2),      # min, max
    "Split": (1,),       # split lengths
    "Squeeze": (1,),     # axes
    "Unsqueeze": (1,),   # axes
}


def control_input_indices(op_type: str) -> tuple:
    return tuple(CONTROL_INPUT_INDICES.get(op_type, ()))


def normalize_domain(domain: Optional[bytes]) -> str:
    if domain is None:
        return ""
    if isinstance(domain, bytes):
        domain = domain.decode("utf-8")
    return str(domain)


def get_spec(op_type: str, *, opset: int, domain: str = "") -> OperatorSpec:
    if domain not in ("", "ai.onnx"):
        raise CompileError(E_UNSUPPORTED_OPSET,
                           "custom operator domain %r is not supported" % domain)
    spec = REGISTRY.get(op_type)
    if spec is None:
        raise CompileError(E_UNSUPPORTED_OP,
                           "operator %r is not registered in the E1 subset" % op_type)
    if not (spec.min_opset <= int(opset) <= spec.max_opset):
        raise CompileError(E_UNSUPPORTED_OPSET,
                           "%s opset %s is outside supported range %d..%d" %
                           (op_type, opset, spec.min_opset, spec.max_opset))
    return spec


def registry_document() -> Dict[str, Any]:
    return {
        "schema": "e1-op-registry-1",
        "opset_range": [MIN_OPSET, MAX_OPSET],
        "operators": {name: REGISTRY[name].to_dict() for name in sorted(REGISTRY)},
    }


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must be an integer" % name, attribute=name)
    return int(value)


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must be numeric" % name, attribute=name)
    return float(value)


def _pair(value: Any, name: str, default: tuple) -> tuple:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must contain two integers" % name, attribute=name)
    out = tuple(_as_int(v, name) for v in value)
    if out[0] != out[1]:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must be symmetric" % name, attribute=name)
    return out


def _pads(value: Any, name: str, default: tuple) -> tuple:
    if value is None:
        return default
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must contain four integers" % name, attribute=name)
    out = tuple(_as_int(v, name) for v in value)
    if len(set(out)) != 1:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must be symmetric on H/W" % name, attribute=name)
    return out


def _check_attrs(op_type: str, attrs: Dict[str, Any], allowed: Iterable[str]) -> None:
    unknown = sorted(set(attrs) - set(allowed))
    if unknown:
        raise CompileError(E_UNSUPPORTED_ATTR,
                           "%s attribute(s) not implemented: %s" % (op_type, ", ".join(unknown)),
                           attribute=unknown[0])


def normalize_attributes(op_type: str, attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and materialize canonical defaults for one registered operator."""
    raw = dict(attrs or {})
    spec = REGISTRY.get(op_type)
    if spec is None:
        raise CompileError(E_UNSUPPORTED_OP, "operator %r is not registered" % op_type)
    _check_attrs(op_type, raw, spec.allowed_attrs)

    if op_type == "DepthwiseConv":
        auto_pad = str(raw.get("auto_pad", "NOTSET"))
        if auto_pad != "NOTSET":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "DepthwiseConv auto_pad=%s is unsupported" % auto_pad,
                               attribute="auto_pad")
        kernel = _pair(raw.get("kernel_shape"), "kernel_shape", (3, 3))
        if kernel != (3, 3):
            raise CompileError(E_UNSUPPORTED_ATTR, "DepthwiseConv kernel must be 3x3",
                               attribute="kernel_shape")
        stride = _pair(raw.get("strides"), "strides", (1, 1))
        if stride[0] not in (1, 2):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "DepthwiseConv stride must be 1 or 2", attribute="strides")
        dilation = _pair(raw.get("dilations"), "dilations", (1, 1))
        if dilation != (1, 1):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "DepthwiseConv dilation must be 1", attribute="dilations")
        pads = _pads(raw.get("pads"), "pads", (0, 0, 0, 0))
        if pads[0] not in (0, 1):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "DepthwiseConv pad must be 0 or 1", attribute="pads")
        group = _as_int(raw.get("group", 1), "group")
        if group < 1:
            raise CompileError(E_UNSUPPORTED_ATTR, "DepthwiseConv group must be positive",
                               attribute="group")
        return {
            "auto_pad": "NOTSET",
            "kernel_shape": list(kernel),
            "strides": list(stride),
            "pads": list(pads),
            "dilations": list(dilation),
            "group": int(group),
        }

    if op_type == "Conv":
        auto_pad = str(raw.get("auto_pad", "NOTSET"))
        if auto_pad != "NOTSET":
            raise CompileError(E_UNSUPPORTED_ATTR, "Conv auto_pad=%s is unsupported" % auto_pad,
                               attribute="auto_pad")
        kernel = _pair(raw.get("kernel_shape"), "kernel_shape", (3, 3))
        if kernel not in ((3, 3), (1, 1)):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Conv kernel must be 3x3 or 1x1", attribute="kernel_shape")
        stride = _pair(raw.get("strides"), "strides", (1, 1))
        if stride[0] not in (1, 2):
            raise CompileError(E_UNSUPPORTED_ATTR, "Conv stride must be 1 or 2", attribute="strides")
        dilation = _pair(raw.get("dilations"), "dilations", (1, 1))
        if dilation != (1, 1):
            raise CompileError(E_UNSUPPORTED_ATTR, "Conv dilation must be 1", attribute="dilations")
        pads = _pads(raw.get("pads"), "pads", (0, 0, 0, 0))
        if kernel == (1, 1) and pads[0] != 0:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Conv1x1 requires pad=0 before lowering", attribute="pads")
        if pads[0] not in (0, 1):
            raise CompileError(E_UNSUPPORTED_ATTR, "Conv pad must be 0 or 1", attribute="pads")
        group = _as_int(raw.get("group", 1), "group")
        if group < 1:
            raise CompileError(E_UNSUPPORTED_ATTR, "Conv group must be positive",
                               attribute="group")
        if group != 1:
            # Depthwise candidate: standard ONNX grouped Conv is accepted here
            # only when shape inference later proves group == Cin == Cout and the
            # weight is [Cout, 1, 3, 3].  Non-depthwise grouped Conv is rejected
            # there with a structured error; dense Conv (group=1) is unchanged.
            if kernel != (3, 3):
                raise CompileError(E_UNSUPPORTED_ATTR,
                                   "grouped Conv is supported only as depthwise 3x3",
                                   attribute="kernel_shape")
        return {
            "auto_pad": "NOTSET",
            "kernel_shape": list(kernel),
            "strides": list(stride),
            "pads": list(pads),
            "dilations": list(dilation),
            "group": int(group),
        }

    if op_type == "MaxPool":
        from .maxpool2d_geometry import normalize_attributes as normalize_maxpool
        return normalize_maxpool(raw)

    if op_type == "Resize":
        # Only the exact identity-arithmetic nearest-neighbour x2 upsample is
        # accepted.  ONNX defaults are applied first so an omitted attribute is
        # rejected with its real (unsupported) default instead of being silently
        # reinterpreted as the asymmetric mapping.
        mode = str(raw.get("mode", "nearest"))
        if mode != "nearest":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize mode=%s is unsupported (nearest only)" % mode,
                               attribute="mode")
        coordinate = str(raw.get("coordinate_transformation_mode", "half_pixel"))
        if coordinate != "asymmetric":
            raise CompileError(
                E_UNSUPPORTED_ATTR,
                "Resize coordinate_transformation_mode=%s is unsupported (asymmetric only)" %
                coordinate, attribute="coordinate_transformation_mode")
        nearest_mode = str(raw.get("nearest_mode", "round_prefer_floor"))
        if nearest_mode != "floor":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize nearest_mode=%s is unsupported (floor only)" %
                               nearest_mode, attribute="nearest_mode")
        if _as_int(raw.get("antialias", 0), "antialias") != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "Resize antialias must be 0",
                               attribute="antialias")
        if _as_int(raw.get("exclude_outside", 0), "exclude_outside") != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "Resize exclude_outside must be 0",
                               attribute="exclude_outside")
        policy = str(raw.get("keep_aspect_ratio_policy", "stretch"))
        if policy != "stretch":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Resize keep_aspect_ratio_policy=%s is unsupported (stretch only)"
                               % policy, attribute="keep_aspect_ratio_policy")
        axes = raw.get("axes")
        if axes is not None:
            if not isinstance(axes, (list, tuple)) or len(axes) != 0:
                raise CompileError(E_UNSUPPORTED_ATTR,
                                   "Resize axes must be absent or empty", attribute="axes")
        # cubic_coeff_a only affects cubic mode and extrapolation_value only
        # affects tf_crop_and_resize; they are accepted and ignored.
        if "cubic_coeff_a" in raw:
            _as_float(raw["cubic_coeff_a"], "cubic_coeff_a")
        if "extrapolation_value" in raw:
            _as_float(raw["extrapolation_value"], "extrapolation_value")
        return {
            "mode": "nearest",
            "coordinate_transformation_mode": "asymmetric",
            "nearest_mode": "floor",
            "antialias": 0,
            "exclude_outside": 0,
            "keep_aspect_ratio_policy": "stretch",
        }

    if op_type == "AveragePool":
        auto_pad = str(raw.get("auto_pad", "NOTSET"))
        if auto_pad != "NOTSET":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "AveragePool auto_pad=%s is unsupported" % auto_pad,
                               attribute="auto_pad")
        kernel = _pair(raw.get("kernel_shape"), "kernel_shape", (2, 2))
        if kernel != (2, 2):
            raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool kernel must be 2x2",
                               attribute="kernel_shape")
        stride = _pair(raw.get("strides"), "strides", (2, 2))
        if stride != (2, 2):
            raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool stride must be 2",
                               attribute="strides")
        pads = _pads(raw.get("pads"), "pads", (0, 0, 0, 0))
        if pads != (0, 0, 0, 0):
            raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool pad must be 0",
                               attribute="pads")
        if _as_int(raw.get("ceil_mode", 0), "ceil_mode") != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool ceil_mode must be 0",
                               attribute="ceil_mode")
        dilation = _pair(raw.get("dilations"), "dilations", (1, 1))
        if dilation != (1, 1):
            raise CompileError(E_UNSUPPORTED_ATTR, "AveragePool dilation must be 1",
                               attribute="dilations")
        # With pads=0 every window is complete, so ONNX count_include_pad=0 and
        # count_include_pad=1 are the same function.  Both spellings are
        # accepted and normalized to 0; no padding algorithm is added.
        count_include_pad = _as_int(raw.get("count_include_pad", 0), "count_include_pad")
        if count_include_pad not in (0, 1):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "AveragePool count_include_pad must be 0 or 1",
                               attribute="count_include_pad")
        return {
            "auto_pad": "NOTSET",
            "kernel_shape": list(kernel),
            "strides": list(stride),
            "pads": list(pads),
            "ceil_mode": 0,
            "count_include_pad": 0,
            "dilations": list(dilation),
        }

    if op_type == "Mul":
        return {}

    if op_type == "BatchNormalization":
        epsilon = _as_float(raw.get("epsilon", 1e-5), "epsilon")
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "BatchNormalization epsilon must be finite and positive",
                               attribute="epsilon")
        momentum = _as_float(raw.get("momentum", 0.9), "momentum")
        if not math.isfinite(momentum):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "BatchNormalization momentum must be finite",
                               attribute="momentum")
        if _as_int(raw.get("training_mode", 0), "training_mode") != 0:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "BatchNormalization training_mode=1 is unsupported; "
                               "only inference statistics can be folded",
                               attribute="training_mode")
        spatial = _as_int(raw.get("spatial", 1), "spatial")
        if spatial != 1:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "BatchNormalization spatial=0 is unsupported",
                               attribute="spatial")
        return {"epsilon": epsilon, "momentum": momentum,
                "training_mode": 0, "spatial": 1}

    if op_type == "Pad":
        mode = str(raw.get("mode", "constant"))
        if mode != "constant":
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Pad mode=%s is unsupported (constant only)" % mode,
                               attribute="mode")
        return {"mode": "constant"}

    if op_type == "Slice":
        return {}

    if op_type == "Concat":
        axis_value = raw.get("axis")
        if axis_value is None:
            raise CompileError(E_UNSUPPORTED_ATTR, "Concat requires an axis attribute",
                               attribute="axis")
        axis = _as_int(axis_value, "axis")
        # The supported subset is rank-4 NCHW, so -3 names the channel axis.
        if axis in (-3, 1):
            axis = 1
        else:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Concat axis=%d is unsupported; only channel axis 1 is allowed" % axis,
                               attribute="axis")
        return {"axis": 1}

    if op_type == "Clip":
        bounds = {}
        for key in ("min", "max"):
            if key in raw:
                value = _as_float(raw[key], key)
                if not value.is_integer():
                    raise CompileError(E_UNSUPPORTED_ATTR,
                                       "Clip %s must be an integer INT8 threshold" % key,
                                       attribute=key)
                bounds[key] = int(value)
        if "min" in bounds and not (-128 <= bounds["min"] <= 127):
            raise CompileError(E_UNSUPPORTED_ATTR, "Clip min outside INT8", attribute="min")
        if "max" in bounds and not (-128 <= bounds["max"] <= 127):
            raise CompileError(E_UNSUPPORTED_ATTR, "Clip max outside INT8", attribute="max")
        if bounds.get("min", -128) > bounds.get("max", 127):
            raise CompileError(E_UNSUPPORTED_ATTR, "Clip min exceeds max")
        return bounds

    if op_type == "Split":
        axis_value = raw.get("axis")
        axis = 1 if axis_value is None else _as_int(axis_value, "axis")
        if axis not in (-3, 1):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "Split only supports channel axis 1/-3", attribute="axis")
        return {"axis": 1}

    if op_type in ("GlobalAveragePool", "Identity"):
        return {}

    if op_type in ("Squeeze", "Unsqueeze"):
        axes = raw.get("axes")
        if axes is None:
            return {}
        if not isinstance(axes, (list, tuple)):
            raise CompileError(E_UNSUPPORTED_ATTR, "%s axes must be a list" % op_type,
                               attribute="axes")
        values = [_as_int(value, "axes") for value in axes]
        if len(values) != len(set(values)):
            raise CompileError(E_UNSUPPORTED_ATTR, "%s axes contain duplicates" % op_type,
                               attribute="axes")
        return {"axes": values}

    if op_type == "Constant":
        return {}

    if op_type in ("Relu", "Add"):
        return {}

    if op_type == "Reshape":
        allowzero = _as_int(raw.get("allowzero", 0), "allowzero")
        if allowzero != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "Reshape allowzero=1 is unsupported",
                               attribute="allowzero")
        return {"allowzero": 0}

    if op_type == "Flatten":
        axis = _as_int(raw.get("axis", 1), "axis")
        return {"axis": axis}

    if op_type in ("Sigmoid", "Tanh"):
        return {}

    if op_type == "LeakyRelu":
        alpha = _as_float(raw.get("alpha", 0.01), "alpha")
        if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "LeakyRelu alpha must be finite in 0..1, got %r" % (raw.get("alpha"),),
                               attribute="alpha")
        return {"alpha": float(alpha)}

    if op_type == "ActivationLUT":
        mode = str(raw.get("mode", ""))
        if mode not in ("Sigmoid", "Tanh", "SiLU", "LeakyRelu"):
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "ActivationLUT mode must be Sigmoid/Tanh/SiLU/LeakyRelu",
                               attribute="mode")
        if mode == "LeakyRelu":
            alpha = _as_float(raw.get("alpha", 0.01), "alpha")
            if not math.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
                raise CompileError(E_UNSUPPORTED_ATTR,
                                   "ActivationLUT LeakyRelu alpha must be finite in 0..1",
                                   attribute="alpha")
            return {"mode": mode, "alpha": float(alpha)}
        if "alpha" in raw:
            raise CompileError(E_UNSUPPORTED_ATTR,
                               "ActivationLUT non-Leaky modes must not carry alpha",
                               attribute="alpha")
        return {"mode": mode}

    if op_type == "Gemm":
        alpha = _as_float(raw.get("alpha", 1.0), "alpha")
        beta = _as_float(raw.get("beta", 1.0), "beta")
        trans_a = _as_int(raw.get("transA", 0), "transA")
        trans_b = _as_int(raw.get("transB", 0), "transB")
        if alpha != 1.0 or beta != 1.0:
            raise CompileError(E_UNSUPPORTED_ATTR, "Gemm alpha/beta must be 1.0")
        if trans_a != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "Gemm transA must be 0", attribute="transA")
        if trans_b not in (0, 1):
            raise CompileError(E_UNSUPPORTED_ATTR, "Gemm transB must be 0 or 1",
                               attribute="transB")
        return {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": int(trans_b)}

    raise CompileError(E_UNSUPPORTED_OP, "operator %r has no attribute normalizer" % op_type)


def legacy_conv_attributes(stride: int, pad: int) -> Dict[str, Any]:
    return normalize_attributes("Conv", {"strides": [int(stride), int(stride)],
                                         "pads": [int(pad)] * 4})


def legacy_maxpool_attributes() -> Dict[str, Any]:
    return normalize_attributes("MaxPool", {"kernel_shape": [2, 2], "strides": [2, 2],
                                             "pads": [0, 0, 0, 0]})
