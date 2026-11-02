# -*- coding: utf-8 -*-
"""CPU six-head postprocess for the real YOLOv8n-ReLU VOC20 model (affine-int8-v1).

This is the deployment CPU decoder of docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md
section 2.5.  It is driven *only* by the model manifest
<run>/model/heads.json (schema "v0.2-yolo-heads-1", contract section 3.4):
head names, dtypes, scales, zero points, logical/physical channels, H/W, row
strides, byte sizes and output-image offsets all come from that file.  Nothing
about the model (no scale, stride, grid size, class count, reg_max or class
name) is copied into the decode path here; a missing field is a hard failure
(HeadsManifestError) instead of a silent default.

Pipeline (contract 2.5)::

    u8 head codes (logical C,H,W)
      -> explicit dtype-aware dequantisation: real = scale * (q - zero_point)
      -> DFL: softmax over the reg_max bins + expectation per box side (4 sides)
      -> anchor/stride decode: xywh in input-image pixels
      -> num_classes sigmoid (single-label argmax, as the reference decoder)
      -> confidence threshold (>=, as the reference) + per-class greedy NMS
      -> deterministically sorted detection list

Anchors are re-implemented from the read-only reference
D:/yolov5/3.0/python/yolov8-pytorch-master/utils/utils_bbox.py
make_anchors(strides, grid_cell_offset=0.5) (PyTorch reference project) with
attribution.  The reference function is never imported at runtime, so the
deployment path has no dependency on the reference tree.

Explicit signedness (contract 2.5): the manifest declares the head dtype.
uint8 codes are used as unsigned integers; int8 codes are the signed re-encoding
of the same affine domain (qs = qu - 128, zs = zu - 128) and are dequantised as
real = scale * (qs - zs).  This module never reinterprets bytes through C
char/np.int8 views; the dtype must match the manifest or the call is rejected.

Non-square grids, empty detection lists and several detections on one image
location after NMS are handled explicitly (see the synthetic self-test).  The
module contains no image id and no threshold default: thresholds are explicit
call arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROFILE = "affine-int8-v1"
HEADS_SCHEMA = "v0.2-yolo-heads-1"
LAYOUT_NHWC8_A16 = "NHWC8_A16"
CHANNEL_BLOCK = 8
ABI_ALIGNMENT = 16
HEAD_DTYPES = ("uint8", "int8")
HEAD_ROLES = ("reg", "cls")


def align_up(value: int, alignment: int) -> int:
    """Round value up to a multiple of alignment (non-negative ints)."""
    if alignment <= 0:
        raise ValueError("alignment must be positive, got %r" % (alignment,))
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class HeadsManifestError(RuntimeError):
    """heads.json is missing a required field or violates the frozen schema."""


# --------------------------------------------------------------------------- #
# manifest (contract 3.4, field-level)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeadSpec:
    """One detection head as declared by model/heads.json."""

    node: str
    role: str
    dtype: str
    scale: float
    zero_point: int
    logical_c: int
    phys_c: int
    layout: str
    h: int
    w: int
    row_stride: int
    bytes: int
    #: per-head stride is optional in the manifest (contract 3.4 lists it at the
    #: top level only); when absent it is derived from the head pair position and
    #: validated against strides[].  It is always set on a parsed manifest.
    stride: Optional[int] = None
    offset_bytes: Optional[int] = None
    sample_npy: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def shape_chw(self) -> Tuple[int, int, int]:
        return (self.logical_c, self.h, self.w)

    @property
    def num_anchors(self) -> int:
        return self.h * self.w

    @property
    def is_reg(self) -> bool:
        return self.role == "reg"


@dataclass(frozen=True)
class HeadsManifest:
    """Parsed + validated model/heads.json."""

    schema: str
    profile: str
    num_classes: int
    reg_max: int
    input_h: int
    input_w: int
    class_names: Optional[Tuple[str, ...]]
    strides: Tuple[int, ...]
    heads: Tuple[HeadSpec, ...]
    source_path: Optional[str] = None
    source_sha256: Optional[str] = None
    warnings: Tuple[str, ...] = ()
    raw: Dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def nodes(self) -> Tuple[str, ...]:
        return tuple(head.node for head in self.heads)

    def head(self, node: str) -> HeadSpec:
        for spec in self.heads:
            if spec.node == node:
                return spec
        raise KeyError("head %r is not in the manifest (have %s)" % (node, self.nodes))

    def scale_pairs(self) -> Tuple[Tuple[HeadSpec, HeadSpec], ...]:
        """(reg, cls) head pairs in manifest order, one pair per stride."""
        return tuple((self.heads[2 * i], self.heads[2 * i + 1])
                     for i in range(len(self.strides)))


def _require(container: Any, key: str, errors: List[str], prefix: str = "") -> Any:
    name = "%s.%s" % (prefix, key) if prefix else key
    if not isinstance(container, dict) or key not in container:
        errors.append("missing required field %r" % name)
        return None
    return container[key]


def _as_int(value: Any, name: str, errors: List[str],
            minimum: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append("field %r must be an integer, got %r" % (name, value))
        return None
    if minimum is not None and value < minimum:
        errors.append("field %r must be >= %d, got %d" % (name, minimum, value))
        return None
    return value


def _as_float(value: Any, name: str, errors: List[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append("field %r must be a number, got %r" % (name, value))
        return None
    out = float(value)
    if not math.isfinite(out):
        errors.append("field %r must be finite, got %r" % (name, value))
        return None
    return out


def _parse_head(raw: Any, index: int, errors: List[str]) -> Optional[HeadSpec]:
    prefix = "heads[%d]" % index
    if not isinstance(raw, dict):
        errors.append("%s must be an object, got %r" % (prefix, type(raw).__name__))
        return None
    node = _require(raw, "node", errors, prefix)
    role = _require(raw, "role", errors, prefix)
    stride = raw.get("stride")
    if stride is not None:
        stride = _as_int(stride, prefix + ".stride", errors, minimum=1)
    dtype = _require(raw, "dtype", errors, prefix)
    scale = _as_float(_require(raw, "scale", errors, prefix), prefix + ".scale", errors)
    zero_point = _as_int(_require(raw, "zero_point", errors, prefix),
                         prefix + ".zero_point", errors)
    logical_c = _as_int(_require(raw, "logical_c", errors, prefix),
                        prefix + ".logical_c", errors, minimum=1)
    phys_c = _as_int(_require(raw, "phys_c", errors, prefix), prefix + ".phys_c",
                     errors, minimum=1)
    layout = _require(raw, "layout", errors, prefix)
    h = _as_int(_require(raw, "h", errors, prefix), prefix + ".h", errors, minimum=1)
    w = _as_int(_require(raw, "w", errors, prefix), prefix + ".w", errors, minimum=1)
    row_stride = _as_int(_require(raw, "row_stride", errors, prefix),
                         prefix + ".row_stride", errors, minimum=1)
    nbytes = _as_int(_require(raw, "bytes", errors, prefix), prefix + ".bytes",
                     errors, minimum=1)
    offset_bytes = raw.get("offset_bytes")
    if offset_bytes is not None:
        offset_bytes = _as_int(offset_bytes, prefix + ".offset_bytes", errors, minimum=0)
    sample_npy = raw.get("sample_npy")
    if sample_npy is not None and not isinstance(sample_npy, dict):
        errors.append("%s.sample_npy must be an object when present" % prefix)
        sample_npy = None

    if not isinstance(node, str) or not node:
        errors.append("%s.node must be a non-empty string" % prefix)
    if role not in HEAD_ROLES:
        errors.append("%s.role must be one of %s, got %r" % (prefix, HEAD_ROLES, role))
    if dtype not in HEAD_DTYPES:
        errors.append("%s.dtype must be one of %s, got %r" % (prefix, HEAD_DTYPES, dtype))
    if layout != LAYOUT_NHWC8_A16:
        errors.append("%s.layout must be %r, got %r" % (prefix, LAYOUT_NHWC8_A16, layout))
    if scale is not None and scale <= 0.0:
        errors.append("%s.scale must be > 0, got %r" % (prefix, scale))
    if dtype == "uint8" and zero_point is not None and not 0 <= zero_point <= 255:
        errors.append("%s.zero_point %d outside uint8 range 0..255" % (prefix, zero_point))
    if dtype == "int8" and zero_point is not None and not -128 <= zero_point <= 127:
        errors.append("%s.zero_point %d outside int8 range -128..127"
                      % (prefix, zero_point))
    if all(v is not None for v in (logical_c, phys_c, h, w, row_stride, nbytes)):
        expect_phys = align_up(logical_c, CHANNEL_BLOCK)
        expect_row = align_up(w * expect_phys, ABI_ALIGNMENT)
        expect_bytes = h * expect_row
        if phys_c != expect_phys:
            errors.append("%s.phys_c=%d but align8(logical_c=%d)=%d"
                          % (prefix, phys_c, logical_c, expect_phys))
        if row_stride != expect_row:
            errors.append("%s.row_stride=%d but align16(w*phys_c=%d*%d)=%d"
                          % (prefix, row_stride, w, expect_phys, expect_row))
        if nbytes != expect_bytes:
            errors.append("%s.bytes=%d but h*row_stride=%d" % (prefix, nbytes, expect_bytes))
    if any(v is None for v in (node, role, dtype, scale, zero_point, logical_c,
                               phys_c, layout, h, w, row_stride, nbytes)):
        return None
    return HeadSpec(node=node, role=role, stride=stride, dtype=dtype, scale=scale,
                    zero_point=zero_point, logical_c=logical_c, phys_c=phys_c,
                    layout=layout, h=h, w=w, row_stride=row_stride, bytes=nbytes,
                    offset_bytes=offset_bytes, sample_npy=sample_npy, raw=dict(raw))


def parse_heads_manifest(raw: Any, *, source_path: Optional[str] = None,
                         source_sha256: Optional[str] = None) -> HeadsManifest:
    """Validate a decoded heads.json object; hard-fail on missing/invalid fields."""
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(raw, dict):
        raise HeadsManifestError("heads.json must be a JSON object, got %r"
                                 % (type(raw).__name__,))
    schema = _require(raw, "schema", errors)
    profile = _require(raw, "profile", errors)
    model = _require(raw, "model", errors)
    strides_raw = _require(raw, "strides", errors)
    heads_raw = _require(raw, "heads", errors)

    num_classes = reg_max = input_h = input_w = None
    class_names: Optional[Tuple[str, ...]] = None
    if model is not None and not isinstance(model, dict):
        errors.append("field 'model' must be an object, got %r" % (type(model).__name__,))
        model = None
    if isinstance(model, dict):
        num_classes = _as_int(_require(model, "num_classes", errors, "model"),
                              "model.num_classes", errors, minimum=1)
        reg_max = _as_int(_require(model, "reg_max", errors, "model"),
                          "model.reg_max", errors, minimum=1)
        input_h = _as_int(_require(model, "input_h", errors, "model"),
                          "model.input_h", errors, minimum=1)
        input_w = _as_int(_require(model, "input_w", errors, "model"),
                          "model.input_w", errors, minimum=1)
        names = model.get("class_names")
        if names is not None:
            if (not isinstance(names, list) or not names
                    or any(not isinstance(n, str) or not n for n in names)):
                errors.append("model.class_names must be a non-empty list of strings")
            else:
                class_names = tuple(names)

    strides: Tuple[int, ...] = ()
    if strides_raw is not None:
        if not isinstance(strides_raw, list) or not strides_raw:
            errors.append("field 'strides' must be a non-empty list")
        else:
            parsed = []
            for idx, value in enumerate(strides_raw):
                item = _as_int(value, "strides[%d]" % idx, errors, minimum=1)
                if item is not None:
                    parsed.append(item)
            strides = tuple(parsed)

    heads: List[HeadSpec] = []
    heads_count = 0
    if heads_raw is not None:
        if not isinstance(heads_raw, list) or not heads_raw:
            errors.append("field 'heads' must be a non-empty list")
        else:
            heads_count = len(heads_raw)
            for idx, head_raw in enumerate(heads_raw):
                spec = _parse_head(head_raw, idx, errors)
                if spec is not None:
                    heads.append(spec)

    if schema is not None and schema != HEADS_SCHEMA:
        errors.append("unsupported schema %r (this decoder implements %r)"
                      % (schema, HEADS_SCHEMA))
    if profile is not None and profile != PROFILE:
        errors.append("unsupported profile %r (this decoder implements %r)"
                      % (profile, PROFILE))
    if class_names is not None and num_classes is not None and len(class_names) != num_classes:
        errors.append("model.class_names has %d entries but num_classes=%d"
                      % (len(class_names), num_classes))

    # pairing / order / domain consistency of the 2*len(strides) heads
    if strides and len(heads) == heads_count:
        if len(heads) != 2 * len(strides):
            errors.append("expected %d heads (2 per stride), got %d"
                          % (2 * len(strides), len(heads)))
        for i, stride in enumerate(strides):
            if 2 * i + 1 >= len(heads):
                break
            reg, cls = heads[2 * i], heads[2 * i + 1]
            pair = "heads[%d]/heads[%d]" % (2 * i, 2 * i + 1)
            if reg.role != "reg":
                errors.append("%s (node %s) must have role 'reg'" % (pair, reg.node))
            if cls.role != "cls":
                errors.append("%s (node %s) must have role 'cls'" % (pair, cls.node))
            if ((reg.stride is not None and reg.stride != stride)
                    or (cls.stride is not None and cls.stride != stride)):
                errors.append("%s declared strides %r/%r do not match manifest stride %d"
                              % (pair, reg.stride, cls.stride, stride))
            if (reg.h, reg.w) != (cls.h, cls.w):
                errors.append("%s grid mismatch reg %dx%d vs cls %dx%d"
                              % (pair, reg.h, reg.w, cls.h, cls.w))
            if reg_max is not None and reg.logical_c != 4 * reg_max:
                errors.append("%s reg logical_c=%d but 4*reg_max=%d"
                              % (pair, reg.logical_c, 4 * reg_max))
            if num_classes is not None and cls.logical_c != num_classes:
                errors.append("%s cls logical_c=%d but num_classes=%d"
                              % (pair, cls.logical_c, num_classes))
        if len(heads) == 2 * len(strides):
            # per-head stride is optional in the manifest: derive it from the pair
            # position (top-level strides order) and validate any declared value
            filled: List[HeadSpec] = []
            for i, stride in enumerate(strides):
                filled.append(replace(heads[2 * i], stride=stride))
                filled.append(replace(heads[2 * i + 1], stride=stride))
            heads = filled
    if heads:
        nodes = [head.node for head in heads]
        if len(set(nodes)) != len(nodes):
            errors.append("duplicate head node names in heads.json: %s" % (nodes,))
        offsets = [head.offset_bytes for head in heads]
        if all(value is not None for value in offsets):
            for i, value in enumerate(offsets):
                if value % ABI_ALIGNMENT != 0:
                    errors.append("heads[%d].offset_bytes=%d is not %dB aligned"
                                  % (i, value, ABI_ALIGNMENT))
                if i and offsets[i - 1] is not None and value != offsets[i - 1] + heads[i - 1].bytes:
                    errors.append("heads[%d].offset_bytes=%d does not follow heads[%d] end %d"
                                  % (i, value, i - 1, offsets[i - 1] + heads[i - 1].bytes))
        else:
            warnings.append("not all heads declare offset_bytes; "
                            "unpack_heads_from_image() is unavailable")

    if errors:
        raise HeadsManifestError(
            "heads.json violates the v0.2-yolo-heads-1 contract:\n  - "
            + "\n  - ".join(errors))

    assert schema is not None and profile is not None
    assert num_classes is not None and reg_max is not None
    assert input_h is not None and input_w is not None and strides
    return HeadsManifest(schema=schema, profile=profile, num_classes=num_classes,
                         reg_max=reg_max, input_h=input_h, input_w=input_w,
                         class_names=class_names, strides=strides,
                         heads=tuple(heads), source_path=source_path,
                         source_sha256=source_sha256, warnings=tuple(warnings),
                         raw=dict(raw))


def load_heads_manifest(source: Any) -> HeadsManifest:
    """Load + validate model/heads.json (accepts the file or the model dir)."""
    path = Path(source)
    if path.is_dir():
        path = path / "heads.json"
    if not path.is_file():
        raise FileNotFoundError("missing heads manifest: %s" % path)
    blob = path.read_bytes()
    try:
        raw = json.loads(blob.decode("utf-8"))
    except ValueError as exc:
        raise HeadsManifestError("heads manifest %s is not valid JSON: %r" % (path, exc))
    return parse_heads_manifest(raw, source_path=str(path.resolve()),
                                source_sha256=sha256_bytes(blob))


# --------------------------------------------------------------------------- #
# dequantisation (explicit dtype handling, contract 2.5)
# --------------------------------------------------------------------------- #
def dequantize_head(codes: np.ndarray, spec: HeadSpec) -> np.ndarray:
    """real = scale * (q - zero_point) in float32, explicit about signedness.

    uint8 codes are used as unsigned integers; int8 codes are the signed
    re-encoding of the same domain (qs = qu - 128, zs = zu - 128) and are
    dequantised against the signed zero point.  The array dtype must match the
    manifest dtype - a silent view(np.int8) would corrupt every code above 127,
    so it is rejected instead.
    """
    arr = np.asarray(codes)
    expected = np.uint8 if spec.dtype == "uint8" else np.int8
    if arr.dtype != expected:
        raise ValueError("head %r dtype %s does not match manifest dtype %r "
                         "(pass the codes in the manifest domain, never a byte view)"
                         % (spec.node, arr.dtype, spec.dtype))
    if tuple(arr.shape) != spec.shape_chw:
        raise ValueError("head %r shape %s does not match manifest logical shape %s"
                         % (spec.node, tuple(arr.shape), spec.shape_chw))
    q = arr.astype(np.int32)  # widen before any arithmetic; never rely on char signedness
    real = (q - int(spec.zero_point)).astype(np.float32) * np.float32(spec.scale)
    return real


# --------------------------------------------------------------------------- #
# anchors (re-implemented from the read-only reference make_anchors)
# --------------------------------------------------------------------------- #
def make_anchors(strides: Sequence[int], grids: Sequence[Tuple[int, int]],
                 offset: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Anchor points + per-point strides, in the reference order.

    Re-implemented from the PyTorch reference
    utils/utils_bbox.py::make_anchors(feats, strides, grid_cell_offset=0.5)
    (read-only reference project) with attribution; the reference module is not
    imported.  grids is one (h, w) per stride, in scale order; points are
    row-major y * w + x with (x + 0.5, y + 0.5) in grid units.
    """
    if len(strides) != len(grids):
        raise ValueError("strides/grids length mismatch: %d vs %d"
                         % (len(strides), len(grids)))
    points: List[np.ndarray] = []
    stride_rows: List[np.ndarray] = []
    for stride, (h, w) in zip(strides, grids):
        h, w, stride = int(h), int(w), int(stride)
        if h <= 0 or w <= 0 or stride <= 0:
            raise ValueError("invalid grid/stride (%d, %d, %d)" % (h, w, stride))
        sx = np.arange(w, dtype=np.float32) + np.float32(offset)
        sy = np.arange(h, dtype=np.float32) + np.float32(offset)
        gx, gy = np.meshgrid(sx, sy)  # indexing='xy': gx[i, j] = sx[j], gy[i, j] = sy[i]
        points.append(np.stack((gx.ravel(), gy.ravel()), axis=1))
        stride_rows.append(np.full((h * w, 1), np.float32(stride), dtype=np.float32))
    return np.concatenate(points, axis=0), np.concatenate(stride_rows, axis=0)


# --------------------------------------------------------------------------- #
# DFL (16-bin softmax + expectation)
# --------------------------------------------------------------------------- #
def dfl_expectation(reg_real: np.ndarray, reg_max: int) -> np.ndarray:
    """DFL softmax over the reg_max bins and its expectation per box side.

    reg_real is (4 * reg_max, N) float32; returns (4, N) distances in grid units
    (ltrb), matching the reference DFL module (softmax over the 16 bins of each
    side, then the weighted sum with bin indices 0..reg_max-1).
    """
    reg_max = int(reg_max)
    if reg_max <= 0:
        raise ValueError("reg_max must be positive, got %r" % (reg_max,))
    arr = np.asarray(reg_real, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 4 * reg_max:
        raise ValueError("reg tensor must be (4*reg_max=%d, N), got %s"
                         % (4 * reg_max, tuple(arr.shape)))
    n = arr.shape[1]
    z = arr.reshape(4, reg_max, n)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
    return np.sum(p * bins, axis=1).astype(np.float32)


def dist2bbox_xywh(dist_ltrb: np.ndarray, anchors_xy: np.ndarray,
                   strides: np.ndarray) -> np.ndarray:
    """Decode ltrb distances to xywh in input pixels (reference dist2bbox)."""
    lt = dist_ltrb[0:2]
    rb = dist_ltrb[2:4]
    x1 = anchors_xy[:, 0] - lt[0]
    y1 = anchors_xy[:, 1] - lt[1]
    x2 = anchors_xy[:, 0] + rb[0]
    y2 = anchors_xy[:, 1] + rb[1]
    c_xy = np.stack(((x1 + x2) * np.float32(0.5),
                     (y1 + y2) * np.float32(0.5)), axis=1)
    wh = np.stack((x2 - x1, y2 - y1), axis=1)
    xywh = np.concatenate((c_xy, wh), axis=1).astype(np.float32) * strides
    return xywh.astype(np.float32)


def xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = (boxes_xywh[:, i] for i in range(4))
    return np.stack((cx - w * np.float32(0.5), cy - h * np.float32(0.5),
                     cx + w * np.float32(0.5), cy + h * np.float32(0.5)), axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """float32 sigmoid (same expression as the reference cls.sigmoid())."""
    arr = np.asarray(x, dtype=np.float32)
    with np.errstate(over="ignore"):
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-arr))).astype(np.float32)


# --------------------------------------------------------------------------- #
# decode + detection assembly
# --------------------------------------------------------------------------- #
@dataclass
class DecodedHeads:
    """All anchors decoded from the six heads, before thresholding/NMS."""

    boxes_xyxy: np.ndarray        # (N, 4) float32, input pixels
    scores: np.ndarray            # (N, num_classes) float32 sigmoid
    anchor_index: np.ndarray      # (N,) int32, global index in the 8/16/32 order
    stride: np.ndarray            # (N,) int32
    scale_index: np.ndarray       # (N,) int8
    anchor_xy: np.ndarray         # (N, 2) float32, grid units
    grids: Tuple[Tuple[int, int], ...]
    strides: Tuple[int, ...]

    @property
    def num_anchors(self) -> int:
        return int(self.boxes_xyxy.shape[0])


def _codes_for(heads: Mapping[str, np.ndarray], spec: HeadSpec) -> np.ndarray:
    if spec.node not in heads:
        raise KeyError("missing head %r in the supplied head arrays (have %s)"
                       % (spec.node, sorted(heads)))
    return dequantize_head(heads[spec.node], spec)


def decode_heads(heads: Mapping[str, np.ndarray],
                 manifest: HeadsManifest) -> DecodedHeads:
    """Dequantise + DFL + anchor decode for all six heads (manifest order).

    heads maps each manifest head node to its logical uint8/int8 (C, H, W)
    codes.  Returns per-anchor boxes in input pixels, sigmoid class scores, and
    the anchor bookkeeping needed by detect() and by verification.
    """
    grids = tuple((spec.h, spec.w) for spec, _ in manifest.scale_pairs())
    points, stride_rows = make_anchors(manifest.strides, grids)
    starts = [0]
    for h, w in grids:
        starts.append(starts[-1] + h * w)
    boxes_xyxy: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    for scale_index, (reg, cls) in enumerate(manifest.scale_pairs()):
        reg_real = _codes_for(heads, reg).reshape(reg.logical_c, -1)
        cls_real = _codes_for(heads, cls).reshape(cls.logical_c, -1)
        if reg_real.shape[1] != reg.h * reg.w or cls_real.shape[1] != reg.h * reg.w:
            raise ValueError("head %r/%r spatial size does not match manifest %dx%d"
                             % (reg.node, cls.node, reg.h, reg.w))
        dist = dfl_expectation(reg_real, manifest.reg_max)
        local_points = points[starts[scale_index]:starts[scale_index + 1]]
        local_strides = stride_rows[starts[scale_index]:starts[scale_index + 1]]
        boxes_xywh = dist2bbox_xywh(dist, local_points, local_strides)
        boxes_xyxy.append(xywh_to_xyxy(boxes_xywh))
        scores.append(sigmoid(np.transpose(cls_real, (1, 0))))
    n = starts[-1]
    anchor_index = np.arange(n, dtype=np.int32)
    stride_arr = stride_rows[:, 0].astype(np.int32)
    scale_index_arr = np.concatenate(
        [np.full((h * w,), i, dtype=np.int8) for i, (h, w) in enumerate(grids)])
    return DecodedHeads(boxes_xyxy=np.concatenate(boxes_xyxy, axis=0).astype(np.float32),
                        scores=np.concatenate(scores, axis=0).astype(np.float32),
                        anchor_index=anchor_index, stride=stride_arr,
                        scale_index=scale_index_arr, anchor_xy=points,
                        grids=grids, strides=tuple(manifest.strides))


@dataclass(frozen=True)
class Detection:
    """One post-NMS detection in input-image pixels (640x640 contract input)."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: Optional[str]
    anchor_index: int
    stride: int
    scale_index: int

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, Any]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "score": self.score, "class_id": self.class_id,
                "class_name": self.class_name, "anchor_index": self.anchor_index,
                "stride": self.stride, "scale_index": self.scale_index}


def iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against (M, 4) xyxy boxes (float64 internally)."""
    box = np.asarray(box, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None) * \
        np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
    union = area_a + area_b - inter
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0.0)
    return out


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix (len(a), len(b)) for xyxy boxes."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for i in range(a.shape[0]):
        out[i] = iou_xyxy(a[i], b)
    return out


def nms_per_class(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                  iou_thres: float) -> np.ndarray:
    """Greedy per-class NMS; returns kept indices (score-sorted within a class).

    Suppression rule matches torchvision.ops.nms: a lower-scoring box is dropped
    when its IoU with a kept box of the same class is **greater than** iou_thres.
    Different classes never suppress each other, so overlapping detections of
    different classes at the same image location can both remain.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    class_ids = np.asarray(class_ids).reshape(-1)
    if not (boxes.shape[0] == scores.shape[0] == class_ids.shape[0]):
        raise ValueError("boxes/scores/class_ids length mismatch")
    if not (0.0 <= float(iou_thres) <= 1.0):
        raise ValueError("iou_thres must be in [0, 1], got %r" % (iou_thres,))
    kept: List[int] = []
    for class_id in np.unique(class_ids):
        idx = np.nonzero(class_ids == class_id)[0]
        order = np.lexsort((idx, -scores[idx]))  # primary -score, secondary index
        remaining = idx[order]  # map positions back to global indices
        while remaining.size:
            best = int(remaining[0])
            kept.append(best)
            if remaining.size == 1:
                break
            rest = remaining[1:]
            ious = iou_xyxy(boxes[best], boxes[rest])
            remaining = rest[ious <= float(iou_thres)]
    return np.asarray(kept, dtype=np.int64)


def detect(decoded: DecodedHeads, manifest: HeadsManifest, conf_thres: float,
           iou_thres: float, max_det: Optional[int] = None) -> List[Detection]:
    """Confidence threshold + per-class NMS + stable sort (manifest-driven).

    conf_thres/iou_thres are explicit arguments: the reference yolov8n_q.py
    defaults are 0.5 / 0.3 and callers must pass them; this module has no tuned
    default.  The threshold comparison is >= like the reference decoder.  The
    returned list is sorted by (-score, class_id, anchor_index), so equal-score
    ties are deterministic and repeatable.
    """
    conf_thres = float(conf_thres)
    iou_thres = float(iou_thres)
    if not (0.0 <= conf_thres <= 1.0):
        raise ValueError("conf_thres must be in [0, 1], got %r" % (conf_thres,))
    scores = np.asarray(decoded.scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != manifest.num_classes:
        raise ValueError("score tensor must be (N, num_classes=%d), got %s"
                         % (manifest.num_classes, tuple(scores.shape)))
    class_ids = np.argmax(scores, axis=1)
    class_conf = scores[np.arange(scores.shape[0]), class_ids]
    keep = class_conf >= np.float32(conf_thres)
    if not keep.any():
        return []  # explicit empty-detection path
    idx = np.nonzero(keep)[0]
    boxes = decoded.boxes_xyxy[idx]
    kept_local = nms_per_class(boxes, class_conf[idx], class_ids[idx], iou_thres)
    picked = idx[kept_local]
    order = np.lexsort((decoded.anchor_index[picked], class_ids[picked],
                        -class_conf[picked]))
    picked = picked[order]
    if max_det is not None:
        max_det = int(max_det)
        if max_det < 0:
            raise ValueError("max_det must be >= 0, got %r" % (max_det,))
        picked = picked[:max_det]
    names = manifest.class_names
    detections: List[Detection] = []
    for i in picked:
        cid = int(class_ids[i])
        x1, y1, x2, y2 = (float(v) for v in decoded.boxes_xyxy[i])
        detections.append(Detection(
            x1=x1, y1=y1, x2=x2, y2=y2, score=float(class_conf[i]),
            class_id=cid, class_name=(names[cid] if names else None),
            anchor_index=int(decoded.anchor_index[i]), stride=int(decoded.stride[i]),
            scale_index=int(decoded.scale_index[i])))
    return detections


def postprocess(heads: Mapping[str, np.ndarray], manifest: HeadsManifest,
                conf_thres: float, iou_thres: float,
                max_det: Optional[int] = None) -> Tuple[List[Detection], DecodedHeads]:
    """Convenience wrapper: decode then detect."""
    decoded = decode_heads(heads, manifest)
    return detect(decoded, manifest, conf_thres, iou_thres, max_det=max_det), decoded


def detections_to_json(detections: Sequence[Detection]) -> List[Dict[str, Any]]:
    return [det.to_dict() for det in detections]


# --------------------------------------------------------------------------- #
# NPU output-image (NHWC8_A16) unpacking, manifest-driven
# --------------------------------------------------------------------------- #
def unpack_nhwc8_a16(blob: Any, spec: HeadSpec) -> np.ndarray:
    """Unpack one head from the packed NHWC8_A16 output image to logical CHW.

    Geometry comes from the manifest: phys_c (align8(C)), row_stride
    (align16(w * phys_c)) and bytes (h * row_stride).  Padded channels are
    dropped; nothing is assumed about the padding byte value.
    """
    if isinstance(blob, np.ndarray):
        data = np.ascontiguousarray(blob).reshape(-1)
    else:
        data = np.frombuffer(bytes(blob), dtype=np.uint8)
    if data.dtype != np.uint8:
        raise ValueError("packed head %r must be uint8 bytes, got %s"
                         % (spec.node, data.dtype))
    if int(data.size) != spec.bytes:
        raise ValueError("packed head %r has %d bytes, manifest declares %d"
                         % (spec.node, int(data.size), spec.bytes))
    if spec.layout != LAYOUT_NHWC8_A16:
        raise ValueError("head %r layout %r is not %s"
                         % (spec.node, spec.layout, LAYOUT_NHWC8_A16))
    if spec.row_stride != align_up(spec.w * spec.phys_c, ABI_ALIGNMENT):
        raise ValueError("head %r row_stride %d is not align16(w*phys_c=%d)"
                         % (spec.node, spec.row_stride, spec.w * spec.phys_c))
    rows = data.reshape(spec.h, spec.row_stride)[:, :spec.w * spec.phys_c]
    pixels = rows.reshape(spec.h, spec.w, spec.phys_c)
    return np.ascontiguousarray(np.transpose(pixels[:, :, :spec.logical_c], (2, 0, 1)))


def unpack_heads_from_image(image: Any,
                            manifest: HeadsManifest) -> Dict[str, np.ndarray]:
    """Unpack all six heads from the 16B-aligned output mirror image."""
    offsets = [spec.offset_bytes for spec in manifest.heads]
    if any(value is None for value in offsets):
        raise ValueError("heads.json does not declare offset_bytes for every head")
    if isinstance(image, np.ndarray):
        data = np.ascontiguousarray(image).reshape(-1)
    else:
        data = np.frombuffer(bytes(image), dtype=np.uint8)
    if data.dtype != np.uint8:
        raise ValueError("output image must be uint8 bytes, got %s" % (data.dtype,))
    end = max(int(offset) + spec.bytes
              for offset, spec in zip(offsets, manifest.heads))
    if int(data.size) < end:
        raise ValueError("output image has %d bytes, manifest needs at least %d"
                         % (int(data.size), end))
    return {spec.node: unpack_nhwc8_a16(data[int(offset):int(offset) + spec.bytes], spec)
            for offset, spec in zip(offsets, manifest.heads)}


# --------------------------------------------------------------------------- #
# synthetic self-test (never part of the real decode path)
# --------------------------------------------------------------------------- #
def _selftest_manifest_raw(grids=((80, 80), (40, 40), (20, 20)),
                           strides=(8, 16, 32), num_classes=20, reg_max=16,
                           scales=(0.1334732771, 0.1708103716, 0.1190009490,
                                   0.2256305218, 0.1321256608, 0.2798818350),
                           zero_points=(41, 127, 44, 120, 47, 118),
                           class_names=None) -> Dict[str, Any]:
    heads = []
    offset = 0
    for i, (h, w) in enumerate(grids):
        for role, logical_c, scale, zp, node_suffix in (
                ("reg", 4 * reg_max, scales[2 * i], zero_points[2 * i], 0),
                ("cls", num_classes, scales[2 * i + 1], zero_points[2 * i + 1], 3)):
            phys_c = align_up(logical_c, CHANNEL_BLOCK)
            row_stride = align_up(w * phys_c, ABI_ALIGNMENT)
            nbytes = h * row_stride
            heads.append({"node": "conv%d" % (48 + 6 * i + node_suffix),
                          "role": role, "stride": strides[i], "dtype": "uint8",
                          "scale": scale, "zero_point": zp, "logical_c": logical_c,
                          "phys_c": phys_c, "layout": LAYOUT_NHWC8_A16, "h": h, "w": w,
                          "row_stride": row_stride, "bytes": nbytes,
                          "offset_bytes": offset})
            offset += nbytes
    model = {"name": "synthetic-selftest", "num_classes": num_classes,
             "reg_max": reg_max, "input_h": 640, "input_w": 640}
    if class_names is not None:
        model["class_names"] = class_names
    return {"schema": HEADS_SCHEMA, "profile": PROFILE, "model": model,
            "strides": list(strides), "heads": heads}


def _selftest_pack(codes: np.ndarray, spec: HeadSpec) -> bytes:
    """Test-only packer mirroring the frozen NHWC8_A16 geometry."""
    c, h, w = codes.shape
    if (c, h, w) != spec.shape_chw:
        raise ValueError("pack shape mismatch")
    out = np.zeros((h, spec.row_stride), dtype=np.uint8)
    pixels = np.zeros((h, w, spec.phys_c), dtype=np.uint8)
    pixels[:, :, :c] = codes.transpose(1, 2, 0)
    out[:, :w * spec.phys_c] = pixels.reshape(h, w * spec.phys_c)
    return out.tobytes()


def _selftest_check(checks: List[Dict[str, Any]], name: str, ok: bool,
                    detail: Any = None) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def self_test() -> Dict[str, Any]:
    """Synthetic-manifest self-test of the decoder (not evidence about the model)."""
    checks: List[Dict[str, Any]] = []
    rng = np.random.default_rng(20261026)

    # 1) manifest parse + geometry formulas
    raw = _selftest_manifest_raw()
    manifest = parse_heads_manifest(raw)
    _selftest_check(checks, "manifest_parse",
                    len(manifest.heads) == 6
                    and manifest.nodes == ("conv48", "conv51", "conv54",
                                           "conv57", "conv60", "conv63")
                    and [s.stride for s in manifest.heads] == [8, 8, 16, 16, 32, 32]
                    and manifest.heads[1].phys_c == 24
                    and manifest.heads[0].row_stride == 5120
                    and manifest.heads[0].bytes == 409600,
                    {"nodes": manifest.nodes,
                     "phys_c": [s.phys_c for s in manifest.heads]})

    # 1b) per-head stride is optional: derived from the top-level strides order
    raw_no_stride = _selftest_manifest_raw()
    for head in raw_no_stride["heads"]:
        head.pop("stride")
    derived = parse_heads_manifest(raw_no_stride)
    _selftest_check(checks, "stride_derived_from_top_level_strides",
                    [h.stride for h in derived.heads] == [8, 8, 16, 16, 32, 32],
                    {"strides": [h.stride for h in derived.heads]})

    # 2) missing field is a hard failure naming the field
    broken = _selftest_manifest_raw()
    del broken["heads"][0]["scale"]
    try:
        parse_heads_manifest(broken)
        missing_ok = False
    except HeadsManifestError as exc:
        missing_ok = "scale" in str(exc) and "heads[0]" in str(exc)
    _selftest_check(checks, "missing_field_hard_fail", missing_ok)

    # 3) wrong geometry / schema / pairing are rejected
    bad_geom = _selftest_manifest_raw()
    bad_geom["heads"][1]["phys_c"] = 32
    bad_schema = _selftest_manifest_raw()
    bad_schema["schema"] = "v0.2-yolo-heads-0"
    bad_pair = _selftest_manifest_raw()
    bad_pair["heads"][0], bad_pair["heads"][1] = bad_pair["heads"][1], bad_pair["heads"][0]
    bad_stride = _selftest_manifest_raw()
    bad_stride["heads"][0]["stride"] = 16  # declared value contradicts strides[0]=8
    rejected = 0
    for payload in (bad_geom, bad_schema, bad_pair, bad_stride):
        try:
            parse_heads_manifest(payload)
        except HeadsManifestError:
            rejected += 1
    _selftest_check(checks, "invalid_manifest_rejected", rejected == 4,
                    {"rejected": rejected, "of": 4})

    # 4) explicit dtype handling: u8 and signed re-encoding dequantise identically
    spec_u8 = HeadSpec(node="conv48", role="reg", stride=8, dtype="uint8",
                       scale=0.1334732771, zero_point=41, logical_c=64, phys_c=64,
                       layout=LAYOUT_NHWC8_A16, h=4, w=5,
                       row_stride=align_up(5 * 64, 16), bytes=4 * align_up(5 * 64, 16))
    codes_u8 = rng.integers(0, 256, size=spec_u8.shape_chw, dtype=np.uint8)
    real_u8 = dequantize_head(codes_u8, spec_u8)
    spec_s8 = HeadSpec(node="conv48", role="reg", stride=8, dtype="int8",
                       scale=spec_u8.scale, zero_point=spec_u8.zero_point - 128,
                       logical_c=64, phys_c=64, layout=LAYOUT_NHWC8_A16, h=4, w=5,
                       row_stride=spec_u8.row_stride, bytes=spec_u8.bytes)
    codes_s8 = (codes_u8.astype(np.int16) - 128).astype(np.int8)
    real_s8 = dequantize_head(codes_s8, spec_s8)
    dtype_guard = False
    try:
        dequantize_head(codes_u8, spec_s8)  # uint8 array against int8 manifest
    except ValueError:
        dtype_guard = True
    _selftest_check(checks, "dtype_explicit_signed_equivalence",
                    np.array_equal(real_u8, real_s8) and dtype_guard,
                    {"max_abs_diff": float(np.abs(real_u8 - real_s8).max())})

    # 5) DFL: uniform logits -> 7.5; a dominant bin -> that bin
    uniform = np.zeros((4 * 16, 3), dtype=np.float32)
    uni = dfl_expectation(uniform, 16)
    peak = np.full((4 * 16, 3), -20.0, dtype=np.float32)
    peak[16 + 3, :] = 20.0  # side 1, bin 3
    peaked = dfl_expectation(peak, 16)
    _selftest_check(checks, "dfl_uniform_and_peak",
                    np.allclose(uni, 7.5, atol=1e-5)
                    and abs(float(peaked[1, 0]) - 3.0) < 1e-3,
                    {"uniform": float(uni[0, 0]), "peak_side1": float(peaked[1, 0])})

    # 6) anchors: counts, first/last, scale boundaries, non-square grid safety
    points, stride_rows = make_anchors((8, 16, 32), ((80, 80), (40, 40), (20, 20)))
    anchors_ok = (points.shape == (8400, 2)
                  and np.allclose(points[0], (0.5, 0.5))
                  and np.allclose(points[6399], (79.5, 79.5))
                  and np.allclose(points[6400], (0.5, 0.5))
                  and np.allclose(points[8400 - 1], (19.5, 19.5))
                  and np.allclose(stride_rows[0, 0], 8.0)
                  and np.allclose(stride_rows[6400, 0], 16.0)
                  and np.allclose(stride_rows[8400 - 1, 0], 32.0))
    ns_points, _ = make_anchors((8, 16), ((40, 80), (10, 20)))
    nonsquare_ok = (ns_points.shape == (3200 + 200, 2)
                    and np.allclose(ns_points[0], (0.5, 0.5))
                    and np.allclose(ns_points[79], (79.5, 0.5))
                    and np.allclose(ns_points[80], (0.5, 1.5))
                    and np.allclose(ns_points[-1], (19.5, 9.5)))
    _selftest_check(checks, "anchors_reference_geometry", anchors_ok and nonsquare_ok)

    # 7) full decode on synthetic heads: shapes + non-square grid decode
    def synth_heads(grids, strides, reg_max=16, num_classes=20):
        out = {}
        for i, (h, w) in enumerate(grids):
            reg_codes = rng.integers(0, 256, size=(4 * reg_max, h * w), dtype=np.uint8)
            cls_codes = rng.integers(0, 256, size=(num_classes, h * w), dtype=np.uint8)
            out["conv%d" % (48 + 6 * i)] = reg_codes.reshape(4 * reg_max, h, w)
            out["conv%d" % (48 + 6 * i + 3)] = cls_codes.reshape(num_classes, h, w)
        return out

    heads = synth_heads(((80, 80), (40, 40), (20, 20)), (8, 16, 32))
    decoded = decode_heads(heads, manifest)
    _selftest_check(checks, "decode_shapes",
                    decoded.num_anchors == 8400
                    and decoded.boxes_xyxy.shape == (8400, 4)
                    and decoded.scores.shape == (8400, 20)
                    and int(decoded.scale_index[0]) == 0
                    and int(decoded.scale_index[6400]) == 1
                    and int(decoded.scale_index[8400 - 1]) == 2,
                    {"anchors": decoded.num_anchors})
    ns_manifest = parse_heads_manifest(_selftest_manifest_raw(
        grids=((40, 80), (10, 20)), strides=(8, 16)))
    ns_heads = synth_heads(((40, 80), (10, 20)), (8, 16))
    ns_decoded = decode_heads(ns_heads, ns_manifest)
    _selftest_check(checks, "decode_non_square_grid",
                    ns_decoded.num_anchors == 3400
                    and ns_decoded.boxes_xyxy.shape == (3400, 4),
                    {"anchors": ns_decoded.num_anchors})

    # 8) empty detections: all class codes in the far-negative domain
    small_manifest = parse_heads_manifest(_selftest_manifest_raw(
        grids=((8, 8),), strides=(8,), zero_points=(41, 127, 0, 0, 0, 0)))
    empty_heads = {}
    for spec in small_manifest.heads:
        if spec.role == "reg":
            empty_heads[spec.node] = np.full(spec.shape_chw, 200, dtype=np.uint8)
        else:
            empty_heads[spec.node] = np.zeros(spec.shape_chw, dtype=np.uint8)
    empty_dets = detect(decode_heads(empty_heads, small_manifest), small_manifest,
                        0.5, 0.3)
    _selftest_check(checks, "empty_detections", empty_dets == [],
                    {"count": len(empty_dets)})

    # 9) several detections after NMS on the same image location (different
    #    classes are never suppressed by each other; single-label argmax keeps
    #    one class per anchor, as the reference decoder does)
    overlap_decoded = DecodedHeads(
        boxes_xyxy=np.array([[100, 100, 200, 200],
                             [100, 100, 200, 200],
                             [300, 300, 400, 400]], dtype=np.float32),
        scores=np.zeros((3, 20), dtype=np.float32),
        anchor_index=np.array([0, 1, 2], dtype=np.int32),
        stride=np.array([8, 8, 16], dtype=np.int32),
        scale_index=np.array([0, 0, 1], dtype=np.int8),
        anchor_xy=np.zeros((3, 2), dtype=np.float32),
        grids=((8, 8),), strides=(8,))
    overlap_decoded.scores[0, 0] = 0.9
    overlap_decoded.scores[1, 3] = 0.9
    overlap_decoded.scores[2, 0] = 0.8
    overlap_dets = detect(overlap_decoded, small_manifest, 0.5, 0.3)
    same_place = [d for d in overlap_dets
                  if (d.x1, d.y1, d.x2, d.y2) == (100.0, 100.0, 200.0, 200.0)]
    _selftest_check(checks, "multiple_detections_after_nms",
                    len(overlap_dets) == 3 and len(same_place) == 2
                    and sorted(d.class_id for d in same_place) == [0, 3],
                    {"count": len(overlap_dets),
                     "same_location_classes": [d.class_id for d in same_place]})

    # 10) per-class NMS suppression + cross-class survival
    boxes = np.array([[0, 0, 100, 100], [10, 10, 110, 110], [500, 500, 600, 600]],
                     dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    same_class = nms_per_class(boxes, scores, np.array([1, 1, 1]), 0.3)
    cross_class = nms_per_class(boxes, scores, np.array([1, 2, 1]), 0.3)
    _selftest_check(checks, "nms_per_class",
                    same_class.tolist() == [0, 2]
                    and sorted(cross_class.tolist()) == [0, 1, 2],
                    {"same_class_keep": same_class.tolist(),
                     "cross_class_keep": sorted(cross_class.tolist())})

    # 11) stable deterministic sort, including equal scores
    stable_decoded = DecodedHeads(
        boxes_xyxy=np.array([[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50],
                             [60, 60, 70, 70]], dtype=np.float32),
        scores=np.full((4, 20), 0.1, dtype=np.float32),
        anchor_index=np.array([10, 5, 7, 2], dtype=np.int32),
        stride=np.array([8, 8, 8, 8], dtype=np.int32),
        scale_index=np.array([0, 0, 0, 0], dtype=np.int8),
        anchor_xy=np.zeros((4, 2), dtype=np.float32),
        grids=((8, 8),), strides=(8,))
    for i, cid in enumerate((3, 1, 2, 1)):
        stable_decoded.scores[i, cid] = 0.9
    runs = [detect(stable_decoded, small_manifest, 0.5, 0.9) for _ in range(3)]
    keys = [[(d.class_id, d.anchor_index) for d in run] for run in runs]
    _selftest_check(checks, "stable_sorted_detections",
                    keys[0] == [(1, 2), (1, 5), (2, 7), (3, 10)]
                    and keys[0] == keys[1] == keys[2],
                    {"order": keys[0]})

    # 12) NHWC8_A16 unpack round-trip (including the C=20 phys_c=24 head)
    roundtrip_ok = True
    mirror = bytearray()
    for spec in manifest.heads:
        codes = rng.integers(0, 256, size=spec.shape_chw, dtype=np.uint8)
        packed = _selftest_pack(codes, spec)
        unpacked = unpack_nhwc8_a16(packed, spec)
        roundtrip_ok = roundtrip_ok and np.array_equal(codes, unpacked)
        mirror.extend(packed)
    unpacked_all = unpack_heads_from_image(bytes(mirror), manifest)
    roundtrip_ok = roundtrip_ok and all(
        unpacked_all[spec.node].shape == spec.shape_chw for spec in manifest.heads)
    _selftest_check(checks, "packed_unpack_roundtrip", roundtrip_ok,
                    {"heads": list(unpacked_all)})

    return {"schema": "v0.2-yolo-postprocess-selftest-1",
            "synthetic": True,
            "note": "synthetic manifest + random codes; this is a decoder unit "
                    "self-test, NOT evidence about the real model",
            "checks": checks, "ok": all(row["ok"] for row in checks)}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _run_cli(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Manifest-driven CPU six-head decoder (yolov8n-relu-voc20)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the synthetic decoder self-test")
    ap.add_argument("--manifest", default=None,
                    help="path to <run>/model/heads.json (or the model dir)")
    ap.add_argument("--heads-dir", default=None,
                    help="directory holding <node>.npy logical u8 heads")
    ap.add_argument("--conf-thres", type=float, default=None,
                    help="confidence threshold (reference default 0.5)")
    ap.add_argument("--iou-thres", type=float, default=None,
                    help="NMS IoU threshold (reference default 0.3)")
    ap.add_argument("--max-det", type=int, default=None)
    ap.add_argument("--out", default=None, help="write detections JSON here")
    args = ap.parse_args(argv)

    if args.self_test:
        result = self_test()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1

    if not (args.manifest and args.heads_dir and args.conf_thres is not None
            and args.iou_thres is not None):
        ap.error("--manifest, --heads-dir, --conf-thres and --iou-thres are required "
                 "unless --self-test is given")
    manifest = load_heads_manifest(args.manifest)
    heads_dir = Path(args.heads_dir)
    heads = {spec.node: np.load(str(heads_dir / (spec.node + ".npy")))
             for spec in manifest.heads}
    detections, decoded = postprocess(heads, manifest, args.conf_thres, args.iou_thres,
                                      max_det=args.max_det)
    payload = {"schema": "v0.2-yolo-detections-1",
               "manifest": {"path": manifest.source_path,
                            "sha256": manifest.source_sha256,
                            "nodes": list(manifest.nodes)},
               "thresholds": {"conf_thres": args.conf_thres,
                              "iou_thres": args.iou_thres},
               "anchors": decoded.num_anchors,
               "count": len(detections),
               "detections": detections_to_json(detections)}
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print("wrote %s (%d detections)" % (out, len(detections)))
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
