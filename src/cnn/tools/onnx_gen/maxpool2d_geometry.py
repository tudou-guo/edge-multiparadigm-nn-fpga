"""Versioned geometry and descriptor contract for the generic INT8 MaxPool.

This module is deliberately independent of backend selection.  A descriptor
is not a claim that a particular generated NPU contains the generic engine.
"""
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from .graph_ir import CompileError, E_CFG, E_LIMIT, E_UNSUPPORTED_ATTR


OPCODE_MAXPOOL2D = 0x001D
FORMAT_VERSION = 1
PAYLOAD_WORDS = 16
CAPABILITY_V1 = {
    "version": FORMAT_VERSION,
    "max_kernel": 7,
    "max_stride": 8,
    "max_dilation": 4,
    "max_padding": 24,
    "max_hw": 640,
    "max_channels": 512,
    "max_row_bytes": 65520,
    "lanes": 8,
    # V1's largest possible 32-row ring plus its input and output rows.
    "max_working_set_bytes": 34 * 65520,
}


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must be an integer" % name,
                           attribute=name)
    return value


def _vector(raw: Mapping[str, Any], name: str, count: int,
            default: Optional[Sequence[int]], low: int, high: int) -> Tuple[int, ...]:
    value = raw.get(name, default)
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise CompileError(E_UNSUPPORTED_ATTR, "%s must have %d entries" % (name, count),
                           attribute=name)
    result = tuple(_integer(v, name) for v in value)
    if any(v < low or v > high for v in result):
        raise CompileError(E_UNSUPPORTED_ATTR, "%s entries must be in %d..%d" %
                           (name, low, high), attribute=name)
    return result


def normalize_attributes(raw: Mapping[str, Any]) -> dict:
    """Normalize ONNX attributes; do not apply PyTorch's stride default."""
    cap = CAPABILITY_V1
    auto_pad = raw.get("auto_pad", "NOTSET")
    if isinstance(auto_pad, bytes):
        auto_pad = auto_pad.decode("ascii")
    if auto_pad not in ("NOTSET", "VALID"):
        raise CompileError(E_UNSUPPORTED_ATTR, "unsupported MaxPool auto_pad %r" % auto_pad,
                           attribute="auto_pad")
    kernel = _vector(raw, "kernel_shape", 2, None, 1, cap["max_kernel"])
    stride = _vector(raw, "strides", 2, (1, 1), 1, cap["max_stride"])
    dilation = _vector(raw, "dilations", 2, (1, 1), 1, cap["max_dilation"])
    pads = _vector(raw, "pads", 4, (0, 0, 0, 0), 0, cap["max_padding"])
    if auto_pad == "VALID":
        pads = (0, 0, 0, 0)
    for name in ("ceil_mode", "storage_order"):
        if _integer(raw.get(name, 0), name) != 0:
            raise CompileError(E_UNSUPPORTED_ATTR, "MaxPool %s must be 0" % name,
                               attribute=name)
    return {"auto_pad": "NOTSET",
            "kernel_shape": list(kernel), "strides": list(stride),
            "pads": list(pads), "dilations": list(dilation),
            "ceil_mode": 0, "storage_order": 0}


def _align(value: int, n: int) -> int:
    return ((value + n - 1) // n) * n


@dataclass(frozen=True)
class Geometry:
    hin: int
    win: int
    channels: int
    cout_alloc: int
    hout: int
    wout: int
    kh: int
    kw: int
    sh: int
    sw: int
    dh: int
    dw: int
    pt: int
    pl: int
    pb: int
    pr: int
    eh: int
    ew: int
    input_row_bytes: int
    output_row_bytes: int

    @property
    def input_span_bytes(self):
        return self.hin * self.input_row_bytes

    @property
    def output_span_bytes(self):
        return self.hout * self.output_row_bytes


def geometry(shape: Sequence[int], attrs: Mapping[str, Any]) -> Geometry:
    """Validate NCHW shape and return all per-record dimensions and strides."""
    if not isinstance(shape, (tuple, list)) or len(shape) != 4:
        raise CompileError(E_CFG, "MaxPool input shape must be NCHW")
    n, c, h, w = (_integer(v, "shape") for v in shape)
    cap = CAPABILITY_V1
    if n != 1 or not (1 <= c <= cap["max_channels"]) or not (
            1 <= h <= cap["max_hw"] and 1 <= w <= cap["max_hw"]):
        raise CompileError(E_LIMIT, "MaxPool input shape is outside V1 capacity")
    a = normalize_attributes(attrs)
    kh, kw = a["kernel_shape"]
    sh, sw = a["strides"]
    dh, dw = a["dilations"]
    pt, pl, pb, pr = a["pads"]
    eh, ew = dh * (kh - 1) + 1, dw * (kw - 1) + 1
    nh, nw = h + pt + pb - eh, w + pl + pr - ew
    if nh < 0 or nw < 0:
        raise CompileError(E_CFG, "MaxPool effective window exceeds padded input")
    oh, ow = nh // sh + 1, nw // sw + 1
    if not (1 <= oh <= cap["max_hw"] and 1 <= ow <= cap["max_hw"]):
        raise CompileError(E_LIMIT, "MaxPool output shape is outside V1 capacity")
    ca = _align(c, 8)
    ri, ro = _align(w * ca, 16), _align(ow * ca, 16)
    if ri > cap["max_row_bytes"] or ro > cap["max_row_bytes"]:
        raise CompileError(E_LIMIT, "MaxPool row command exceeds V1 capacity")
    return Geometry(h, w, c, ca, oh, ow, kh, kw, sh, sw, dh, dw,
                    pt, pl, pb, pr, eh, ew, ri, ro)


def capacity(geometries: Sequence[Geometry], max_working_bytes: Optional[int] = None) -> dict:
    """Calculate physical row and ring capacity from actual generic records."""
    if not geometries:
        raise CompileError(E_CFG, "at least one geometry is required")
    max_in = max(g.input_row_bytes for g in geometries)
    max_out = max(g.output_row_bytes for g in geometries)
    max_eh = max(g.eh for g in geometries)
    rows = 1 << (max_eh - 1).bit_length()
    working = max_in + (rows + 1) * max_out
    budget = CAPABILITY_V1["max_working_set_bytes"] if max_working_bytes is None else max_working_bytes
    if working > budget:
        raise CompileError(E_LIMIT, "MaxPool working set exceeds configured budget")
    return {"max_in_row_bytes": max_in, "max_out_row_bytes": max_out,
            "max_eh": max_eh, "cache_rows": rows, "logical_storage_bytes": working}


def select_backend(graph, node, policy="auto") -> str:
    """Select only proven legacy intersections; pure 2x2 defaults to generic."""
    if policy not in ("auto", "generic"):
        raise CompileError(E_CFG, "MaxPool backend must be auto or generic")
    if "legacy_layer_index" in node.metadata:
        return "legacy2x2"
    if policy == "generic":
        return "generic"
    g = geometry(graph.tensors[node.inputs[0]].shape, node.attrs)
    x, y = graph.tensors[node.inputs[0]], graph.tensors[node.outputs[0]]
    qx, qy = x.quantization or {}, y.quantization or {}
    zx = qx.get("zero_point_s8", qx.get("zero_point", 0))
    zy = qy.get("zero_point_s8", qy.get("zero_point", 0))
    groups = g.cout_alloc // 8
    if ((g.kh, g.kw, g.sh, g.sw, g.dh, g.dw, g.pt, g.pl, g.pb, g.pr) ==
            (5, 5, 1, 1, 1, 1, 2, 2, 2, 2) and
            (groups == 1 or groups % 2 == 0) and g.hin >= 3 and g.win >= 4 and
            x.dtype == "int8" and y.dtype == "int8" and
            qx.get("materialized", True) and qy.get("materialized", True) and
            qx.get("scale") is not None and qx.get("scale") == qy.get("scale") and
            int(zx) == int(zy) == 0):
        return "pool5"
    # CLAMP_MIN_EN=1 in generated builds; no input-code non-negativity proof
    # is available here, so the legacy 2x2 kernel is not a pure-MaxPool path.
    return "generic"


def encode_payload(g: Geometry, input_addr: int, output_addr: int) -> list:
    """Encode the dedicated 16-word descriptor; check before packing."""
    canonical = geometry((1, g.channels, g.hin, g.win), {
        "kernel_shape": (g.kh, g.kw), "strides": (g.sh, g.sw),
        "dilations": (g.dh, g.dw), "pads": (g.pt, g.pl, g.pb, g.pr),
    })
    if canonical != g:
        raise CompileError(E_CFG, "MaxPool geometry is not canonical")
    for address, name in ((input_addr, "input"), (output_addr, "output")):
        if isinstance(address, bool) or not isinstance(address, int) or not (0 <= address < 2**32):
            raise CompileError(E_CFG, "%s address must be a 32-bit integer" % name)
        if address % 16:
            raise CompileError(E_CFG, "%s address must be 16-byte aligned" % name)
    input_end = input_addr + g.input_span_bytes
    output_end = output_addr + g.output_span_bytes
    if input_end > 2**32 or output_end > 2**32:
        raise CompileError(E_CFG, "MaxPool address span overflows 32 bits")
    if input_addr < output_end and output_addr < input_end:
        raise CompileError(E_CFG, "MaxPool input/output spans overlap")
    def pair(low, high):
        if not (0 <= low <= 0xffff and 0 <= high <= 0xffff):
            raise CompileError(E_CFG, "MaxPool descriptor field exceeds 16 bits")
        return low | (high << 16)
    return [FORMAT_VERSION, input_addr, output_addr, pair(g.hin, g.win),
            pair(g.channels, g.cout_alloc), pair(g.hout, g.wout), pair(g.kh, g.kw),
            pair(g.sh, g.sw), pair(g.dh, g.dw), pair(g.pt, g.pl), pair(g.pb, g.pr),
            g.input_row_bytes, g.output_row_bytes, g.input_span_bytes,
            g.output_span_bytes, 0]


def decode_payload(words: Sequence[int]) -> tuple:
    """Decode and revalidate every descriptor field without silent truncation."""
    if len(words) != PAYLOAD_WORDS or any(isinstance(w, bool) or
            not isinstance(w, int) or not 0 <= w < 2**32 for w in words):
        raise CompileError(E_CFG, "MaxPool descriptor must have sixteen 32-bit words")
    if words[0] != FORMAT_VERSION or words[15] != 0:
        raise CompileError(E_CFG, "unsupported MaxPool version or reserved bits")
    def pair(i):
        return words[i] & 0xffff, words[i] >> 16
    h, w = pair(3)
    c, _ = pair(4)
    g = geometry((1, c, h, w), {"kernel_shape": pair(6), "strides": pair(7),
                                "dilations": pair(8), "pads": pair(9) + pair(10)})
    if encode_payload(g, words[1], words[2]) != list(words):
        raise CompileError(E_CFG, "MaxPool descriptor is inconsistent")
    return g, words[1], words[2]
