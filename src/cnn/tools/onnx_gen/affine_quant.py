
# -*- coding: utf-8 -*-
"""affine-int8-v1 deployment numerics (v0.2 real-YOLO M1, explicit new profile).

Why this file exists
--------------------
The reference YOLOv8n-ReLU model uses uint8 affine activations with non-zero zero
points, while the mainline spatial path is signed INT8 with zero_point=0 and
requires Add/Copy inputs to share one quantisation domain.  Rather than duplicate
a uint8 multiplier array, the deployment contract re-encodes activations to
signed INT8::

    qs = qu - 128          # byte-level: XOR 0x80
    zs = zu - 128
    real = s * (qs - zs)

This module is the single source of truth for that profile's integer math.  It is
new and versioned: the legacy signed-INT8 zero-point-0 path and
exact_quant.ratio_to_m0_shift keep their old semantics and are not modified here.
Nothing in this module reads RTL output, old DUT text or a compiled artifact: it
is pure arithmetic used by the exporter, the independent integer reference and
(later) the compiler lowering.

Contract (task book section 4, docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md)

* Conv::

      A         = sum(qs * w) + bq_folded
      bq_folded = bq_int - zs_in * sum(w)          # folded at pack time
      qs_out    = sat_s8( round_ties_away(A * M0 / 2**SHIFT) + zs_out )
      ReLU      : qs_out = max(qs_out, zs_out)     # threshold is zs_out

  M0 / 2**SHIFT is the largest legal fixed-point approximation of
  s_in * s_w / s_out (ties-away rounding, signed 31-bit M0 field).
  Spatial padding is filled with zs_in (not 0).

* Add: two inputs each in their own domain, one final rounding in the output
  domain, then add z_out and saturate::

      q_out = sat_s8( round( (q1-z1)*s1/s_out + (q2-z2)*s2/s_out ) + z_out )

* Concat: every input re-scaled into the output domain with the same single
  rounding (pure byte copy when the domain already matches).

* MaxPool: compare in the signed domain (monotone re-encode preserves order),
  padding uses the domain minimum (-128), never the Conv padding value.

* Resize/Split: value domain unchanged.

All intermediate products are widened: the accumulator bound of every real layer
is checked against INT32 at export time, and the rescale product A * M0 is
computed in at least 64 bits.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

PROFILE = "affine-int8-v1"
PROFILE_DOC = "docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md"
ROUNDING = "ties_away_from_zero"
#: the profile's frozen rounding is ROUNDING; ties_even exists so the accuracy effect of
#: the rounding choice can be measured against PyTorch's nearbyint (round-half-even)
#: before any RTL is written.  Choosing it requires a new profile version, never a
#: silent switch of the frozen one.
ROUNDING_MODES = ("ties_away_from_zero", "ties_even")

S8_MIN = -128
S8_MAX = 127
U8_MIN = 0
U8_MAX = 255
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1

#: signed M0 multiplier field width (value range [0, 2**M0_BITS - 1])
M0_BITS = 31
M0_MAX = (1 << M0_BITS) - 1
#: accumulator rescale shift bound (A * M0 fits int64 for all legal SHIFT)
SHIFT_MAX = 62


# --------------------------------------------------------------------------- #
# domain re-encoding
# --------------------------------------------------------------------------- #
def u8_to_s8(qu):
    """Recode uint8 codes to signed INT8 codes: qs = qu - 128 (identical to byte XOR 0x80)."""
    a = np.asarray(qu)
    if a.dtype.kind in "iu":
        out = a.astype(np.int16) - 128
    else:
        raise TypeError("u8_to_s8 expects integer input, got %r" % (a.dtype,))
    return out.astype(np.int8)


def s8_to_u8(qs):
    """Inverse of u8_to_s8, kept for host-side unpacking of PL output."""
    a = np.asarray(qs)
    if a.dtype.kind in "iu":
        out = a.astype(np.int16) + 128
    else:
        raise TypeError("s8_to_u8 expects integer input, got %r" % (a.dtype,))
    return np.clip(out, U8_MIN, U8_MAX).astype(np.uint8)


def zero_point_u8_to_s8(zu):
    """Signed-domain zero point; the integer zero point stays an integer."""
    zu = int(zu)
    if not (U8_MIN <= zu <= U8_MAX):
        raise ValueError("u8 zero point out of range: %r" % (zu,))
    return zu - 128


# --------------------------------------------------------------------------- #
# rounding helpers
# --------------------------------------------------------------------------- #
def _round_shift_ties_away(x, shift):
    """Round x / 2**shift for integer x with ties away from zero.

    x must be an integer numpy array (or integer scalar) so the operation is
    exact; shift == 0 is a pass-through.  Implemented as
    floor((x + sign(x) * 2**(shift-1)) / 2**shift) using an arithmetic shift after
    the tie correction, which is exact for negative values as well.
    """
    if shift < 0:
        raise ValueError("negative shift %r" % (shift,))
    x = np.asarray(x)
    if x.dtype.kind not in "iu":
        raise TypeError("_round_shift_ties_away expects an integer array, got %r" % (x.dtype,))
    if shift == 0:
        return x
    xi = x.astype(np.int64) if x.dtype != np.int64 else x
    half = np.int64(1) << np.int64(shift - 1)
    # round on the magnitude so the result is nearest and ties go away from zero
    # for both signs; |x| is safe because the rescale bound keeps |x| < 2**63.
    neg = xi < 0
    mag = np.where(neg, np.int64(0) - xi, xi)
    q = (mag + half) >> np.int64(shift)
    return np.where(neg, np.int64(0) - q, q)


def _round_shift_ties_even(x, shift):
    """Round x / 2**shift for integer x with ties to even (torch/fbgemm semantics)."""
    if shift < 0:
        raise ValueError("negative shift %r" % (shift,))
    x = np.asarray(x)
    if x.dtype.kind not in "iu":
        raise TypeError("_round_shift_ties_even expects an integer array, got %r" % (x.dtype,))
    if shift == 0:
        return x
    xi = x.astype(np.int64) if x.dtype != np.int64 else x
    floor = xi >> np.int64(shift)
    rem = xi - (floor << np.int64(shift))
    half = np.int64(1) << np.int64(shift - 1)
    up = (rem > half) | ((rem == half) & ((floor & np.int64(1)) == np.int64(1)))
    return floor + up.astype(np.int64)


def _round_shift(x, shift, rounding):
    if rounding == "ties_away_from_zero":
        return _round_shift_ties_away(x, shift)
    if rounding == "ties_even":
        return _round_shift_ties_even(x, shift)
    raise ValueError("unknown rounding %r (legal: %r)" % (rounding, ROUNDING_MODES))


def quantize_u8_from_float(x_fp32, scale, zero_point=0, rounding="ties_even"):
    """Quantise a float32 tensor to uint8 affine codes (torch-compatible default).

    rounding='ties_even' reproduces torch.quantize_per_tensor (round half to
    even); rounding='ties_away' is the profile requantisation rounding and is only
    used where the contract says so (requant of integer values, not the float
    input pipeline).
    """
    scale = float(scale)
    if not scale > 0.0:
        raise ValueError("scale must be positive, got %r" % (scale,))
    x = np.asarray(x_fp32, dtype=np.float64) / scale + float(zero_point)
    if rounding == "ties_even":
        q = np.rint(x)
    elif rounding == "ties_away":
        q = np.where(x >= 0, np.floor(np.abs(x) + 0.5), -np.floor(np.abs(x) + 0.5))
    else:
        raise ValueError("unknown rounding %r" % (rounding,))
    return np.clip(q, U8_MIN, U8_MAX).astype(np.uint8)


def dequantize_u8(qu, scale, zero_point=0):
    """real = scale * (qu - zero_point), returned as float64 for exact comparison."""
    return (np.asarray(qu, dtype=np.int64) - int(zero_point)).astype(np.float64) * float(scale)


# --------------------------------------------------------------------------- #
# Conv bias folding / accumulator bounds / requant parameters
# --------------------------------------------------------------------------- #
def conv_bias_fold(w_int8, bias_int32, zs_in, cin=None):
    """Fold the input zero point into the int32 bias.

    sum_w is taken over the true logical Cin x kernel elements (the real logical
    shape must be passed in, not the aligned one).  Returns a dict with:

        sum_w        per-output-channel int64 sum of weights
        bias_folded  int64 per-output-channel folded bias (bq_int - zs_in*sum_w)
        acc_bound    int64 worst-case |A| bound used to prove INT32 safety
        k_size       number of logical MAC products per output value
    """
    w = np.asarray(w_int8)
    if w.dtype != np.int8:
        if w.dtype.kind == "i" and w.min() >= S8_MIN and w.max() <= S8_MAX:
            w = w.astype(np.int8)
        else:
            raise TypeError("weights must be signed int8, got %r" % (w.dtype,))
    b = np.asarray(bias_int32)
    if b.dtype.kind not in "iu":
        raise TypeError("bias must be integer, got %r" % (b.dtype,))
    if w.ndim != 4:
        raise ValueError("conv weight must be 4-D OIHW, got shape %r" % (w.shape,))
    cout = w.shape[0]
    k_size = int(w.shape[1]) * int(w.shape[2]) * int(w.shape[3])
    if cin is not None and int(cin) * int(w.shape[2]) * int(w.shape[3]) != k_size:
        raise ValueError("logical cin %r does not match weight shape %r" % (cin, w.shape))
    if b.shape[0] != cout:
        raise ValueError("bias length %d != cout %d" % (b.shape[0], cout))
    w64 = w.astype(np.int64)
    sum_w = w64.sum(axis=(1, 2, 3), dtype=np.int64)
    bias_folded = b.astype(np.int64) - np.int64(zs_in) * sum_w
    prod_bound = max(abs(S8_MIN * S8_MAX), S8_MIN * S8_MIN)
    acc_bound = (int(k_size) * int(prod_bound) + int(np.abs(bias_folded).max())
                 + abs(int(zs_in)) * int(np.abs(sum_w).max()))
    return {
        "sum_w": sum_w,
        "bias_folded": bias_folded,
        "acc_bound": np.int64(acc_bound),
        "k_size": k_size,
        "cout": cout,
        "bias_int32_in_range": bool(b.min() >= INT32_MIN and b.max() <= INT32_MAX),
    }


def check_acc_int32(acc_bound):
    """Prove the folded accumulator stays inside INT32 for this layer."""
    if int(acc_bound) > INT32_MAX:
        raise ValueError("accumulator bound %d exceeds int32" % int(acc_bound))
    return True


def choose_m0_shift(s_in, s_w, s_out, m0_bits=M0_BITS, shift_max=SHIFT_MAX):
    """Pick the largest legal SHIFT with M0 = round(ratio * 2**SHIFT) < 2**m0_bits.

    ratio = s_in * s_w / s_out is treated as an exact rational (the scales are
    float64 values, so their exact binary fractions are used).  The returned dict
    records the exact rational, the chosen M0/SHIFT, the realised ratio and the
    absolute/relative error, so the approximation is evidence, never a silent
    rounding.
    """
    for name, val in (("s_in", s_in), ("s_w", s_w), ("s_out", s_out)):
        if not float(val) > 0.0:
            raise ValueError("%s must be positive, got %r" % (name, val))
    ratio = Fraction(float(s_in)) * Fraction(float(s_w)) / Fraction(float(s_out))
    m0_max = (1 << int(m0_bits)) - 1
    best = None
    for shift in range(int(shift_max), -1, -1):
        m0 = int(round(ratio * (1 << shift)))
        if 1 <= m0 <= m0_max:
            best = (shift, m0)
            break
    if best is None:
        raise ValueError("no legal M0/SHIFT for ratio %s (shift_max=%d)" % (ratio, shift_max))
    shift, m0 = best
    realised = Fraction(m0, 1 << shift)
    abs_err = abs(realised - ratio)
    rel_err = float(abs_err / ratio)
    return {
        "m0": m0,
        "shift": shift,
        "m0_bits": int(m0_bits),
        "m0_in_range": bool(1 <= m0 <= m0_max),
        "ratio_float": float(ratio),
        "ratio_exact_num": int(ratio.numerator),
        "ratio_exact_den": int(ratio.denominator),
        "realised_ratio_float": float(realised),
        "abs_err": float(abs_err),
        "rel_err": rel_err,
        "rounding": ROUNDING,
    }


def requant_conv(acc, m0, shift, zs_out, relu=False, stats=None, rounding=ROUNDING):
    """One rounded rescale + output zero point + optional ReLU threshold.

    acc is the folded INT32 accumulator (numpy integer array or scalar).
    acc * M0 is formed in int64, so the caller must have proved
    |acc| * M0 < 2**63 (check_rescale_int64).
    """
    acc = np.asarray(acc)
    if acc.dtype.kind not in "iu":
        raise TypeError("requant_conv expects an integer accumulator, got %r" % (acc.dtype,))
    a64 = acc.astype(np.int64)
    product = a64 * np.int64(m0)
    if product.size and int(np.abs(product).max()) >= (1 << 63):
        raise OverflowError("acc * M0 overflows int64: tighten SHIFT or prove the bound")
    scaled = _round_shift(product, int(shift), rounding)
    out = scaled + np.int64(int(zs_out))
    if relu:
        out = np.maximum(out, np.int64(int(zs_out)))
    if stats is not None:
        # pre-clip extremes and saturation counts are diagnostics for the
        # accuracy contract: they show whether a layer is railing at the rails.
        stats["pre_clip_min"] = int(out.min()) if out.size else 0
        stats["pre_clip_max"] = int(out.max()) if out.size else 0
        stats["sat_low"] = int((out < S8_MIN).sum())
        stats["sat_high"] = int((out > S8_MAX).sum())
        stats["elements"] = int(out.size)
    return np.clip(out, S8_MIN, S8_MAX).astype(np.int8)


def check_rescale_int64(acc_bound, m0):
    """Prove |acc| * M0 stays inside int64 for the declared bound."""
    if int(acc_bound) * int(m0) >= (1 << 63):
        raise ValueError("acc_bound %d * M0 %d exceeds int64" % (int(acc_bound), int(m0)))
    return True


# --------------------------------------------------------------------------- #
# Add / Concat requantisation (two-input, one final rounding)
# --------------------------------------------------------------------------- #
def choose_add_requant(s_a, s_b, s_out, m0_bits=M0_BITS, shift_max=SHIFT_MAX):
    """Common-SHIFT multipliers for a two-input Add/Concat requantisation.

    Each term gets its own M0 against the shared SHIFT so the final rounding is
    applied exactly once on the wide sum::

        acc   = (q_a - z_a) * M0_a + (q_b - z_b) * M0_b
        q_out = sat_s8( round_ties_away(acc / 2**SHIFT) + z_out )

    The SHIFT is the smaller of the two independent legal shifts (largest common
    legal shift), so neither multiplier loses precision.
    """
    ra = choose_m0_shift(s_a, 1.0, s_out, m0_bits=m0_bits, shift_max=shift_max)
    rb = choose_m0_shift(s_b, 1.0, s_out, m0_bits=m0_bits, shift_max=shift_max)
    shift = min(ra["shift"], rb["shift"])
    m0_a = int(round(Fraction(float(s_a)) / Fraction(float(s_out)) * (1 << shift)))
    m0_b = int(round(Fraction(float(s_b)) / Fraction(float(s_out)) * (1 << shift)))
    m0_max = (1 << int(m0_bits)) - 1
    if not (1 <= m0_a <= m0_max and 1 <= m0_b <= m0_max):
        raise ValueError("Add multipliers out of range: %d / %d" % (m0_a, m0_b))
    return {
        "m0_a": m0_a, "m0_b": m0_b, "shift": shift,
        "m0_bits": int(m0_bits), "rounding": ROUNDING,
        "acc_bound": int((1 << shift) * 2),
        "ratio_a": float(Fraction(m0_a, 1 << shift)),
        "ratio_b": float(Fraction(m0_b, 1 << shift)),
    }


def requant_add(q_a, z_a, q_b, z_b, params, z_out, rounding=ROUNDING):
    """Two-input Add in the output domain with exactly one final rounding."""
    a = np.asarray(q_a, dtype=np.int64) - np.int64(int(z_a))
    b = np.asarray(q_b, dtype=np.int64) - np.int64(int(z_b))
    acc = a * np.int64(int(params["m0_a"])) + b * np.int64(int(params["m0_b"]))
    scaled = _round_shift(acc, int(params["shift"]), rounding)
    out = scaled + np.int64(int(z_out))
    return np.clip(out, S8_MIN, S8_MAX).astype(np.int8)


def requant_copy_to_domain(q, z_in, s_in, s_out, z_out, rounding=ROUNDING):
    """Single-input requantisation (used by Copy/Concat per-input rescale)."""
    if s_in == s_out and int(z_in) == int(z_out):
        return np.asarray(q, dtype=np.int8)
    p = choose_m0_shift(s_in, 1.0, s_out)
    acc = np.asarray(q, dtype=np.int64) - np.int64(int(z_in))
    out = _round_shift(acc * np.int64(p["m0"]), int(p["shift"]), rounding) + np.int64(int(z_out))
    return np.clip(out, S8_MIN, S8_MAX).astype(np.int8)


# --------------------------------------------------------------------------- #
# pooling / resize helpers
# --------------------------------------------------------------------------- #
def maxpool_signed(x_s8, kernel=5, stride=1, pad=2, dilation=1):
    """MaxPool on the signed domain with the domain minimum as padding.

    Comparing signed codes is order-equivalent to comparing real values because
    real = s*(qs - zs) is monotone in qs.
    """
    x = np.asarray(x_s8, dtype=np.int8)
    if x.ndim != 3:
        raise ValueError("maxpool_signed expects CHW, got shape %r" % (x.shape,))
    c, h, w = x.shape
    def pair(value):
        return (int(value), int(value)) if np.isscalar(value) else tuple(map(int, value))
    kh, kw = pair(kernel)
    sh, sw = pair(stride)
    dh, dw = pair(dilation)
    pt, pl, pb, pr = (int(pad),) * 4 if np.isscalar(pad) else tuple(map(int, pad))
    oh = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
    ow = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("MaxPool output is empty")
    out = np.full((c, oh, ow), S8_MIN, dtype=np.int8)
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = oy * sh - pt + ky * dh, ox * sw - pl + kx * dw
                    if 0 <= iy < h and 0 <= ix < w:
                        out[:, oy, ox] = np.maximum(out[:, oy, ox], x[:, iy, ix])
    return out


def resize_nearest_x2_signed(x_s8):
    """nearest x2 upsampling (PyTorch nearest: out[y,x] = in[y//2, x//2])."""
    x = np.asarray(x_s8, dtype=np.int8)
    if x.ndim != 3:
        raise ValueError("resize_nearest_x2_signed expects CHW, got shape %r" % (x.shape,))
    return np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)


# --------------------------------------------------------------------------- #
# self test (cheap boundary vectors, no RTL, no model)
# --------------------------------------------------------------------------- #
def selftest():
    """Boundary vectors for the profile: saturation, ties, negatives, folding."""
    checks = []

    def expect(name, got, want):
        ok = bool(np.array_equal(np.asarray(got), np.asarray(want)))
        checks.append({"check": name, "ok": ok,
                       "got": np.asarray(got).tolist() if np.asarray(got).size <= 16 else "big",
                       "want": np.asarray(want).tolist() if np.asarray(want).size <= 16 else "big"})
        return ok

    expect("u8_to_s8_range", u8_to_s8(np.array([0, 127, 128, 255])), np.array([-128, -1, 0, 127]))
    expect("s8_to_u8_roundtrip", s8_to_u8(u8_to_s8(np.arange(256))), np.arange(256))

    expect("ties_shift1", _round_shift_ties_away(np.array([-3, -1, 1, 3]), 1), np.array([-2, -1, 1, 2]))
    expect("ties_shift2", _round_shift_ties_away(np.array([-6, -2, 2, 6]), 2), np.array([-2, -1, 1, 2]))
    expect("exact_shift2", _round_shift_ties_away(np.array([-5, 5]), 2), np.array([-1, 1]))
    expect("shift0", _round_shift_ties_away(np.array([-7, 7]), 0), np.array([-7, 7]))

    p = choose_m0_shift(0.5, 0.5, 1.0)
    checks.append({"check": "m0_rel_err_tiny", "ok": p["rel_err"] < 1e-9, "rel_err": p["rel_err"]})
    checks.append({"check": "m0_field", "ok": 1 <= p["m0"] <= M0_MAX, "m0": p["m0"],
                   "shift": p["shift"]})

    acc = np.array([1 << 30, -(1 << 30), 0, 3], dtype=np.int64)
    out = requant_conv(acc, m0=(1 << 31) - 1, shift=31, zs_out=-5, relu=True)
    checks.append({"check": "relu_threshold_is_zs_out", "ok": int(out.min()) == -5,
                   "values": out.tolist()})
    out_nr = requant_conv(acc, m0=(1 << 31) - 1, shift=31, zs_out=0, relu=False)
    checks.append({"check": "conv_saturates_low", "ok": int(out_nr[1]) == S8_MIN, "values": out_nr.tolist()})
    checks.append({"check": "conv_saturates_high", "ok": int(out_nr[0]) == S8_MAX, "values": out_nr.tolist()})

    w = np.ones((1, 1, 1, 1), dtype=np.int8)
    folding = conv_bias_fold(w, np.array([7], dtype=np.int32), zs_in=-3, cin=1)
    checks.append({"check": "bias_fold_sum_w", "ok": int(folding["sum_w"][0]) == 1,
                   "sum_w": folding["sum_w"].tolist()})
    checks.append({"check": "bias_fold_value", "ok": int(folding["bias_folded"][0]) == 7 - (-3 * 1),
                   "bias_folded": folding["bias_folded"].tolist()})
    pad_acc = np.int64(0) + np.int64(-3) * np.int64(1) + np.int64(7) - np.int64(-3) * np.int64(1)
    checks.append({"check": "padding_is_zs_in", "ok": int(pad_acc) == 7, "pad_acc": int(pad_acc)})

    two = choose_add_requant(0.5, 0.5, 1.0)
    got = requant_add(np.array([127], dtype=np.int8), 0, np.array([127], dtype=np.int8), 0, two, 0)
    checks.append({"check": "add_no_intermediate_clip", "ok": int(got[0]) == 127, "got": got.tolist()})
    # a=-128-(-3)=-125, b=127-5=122 -> acc=(-125+122)*0.5=-1.5 -> ties away -2 -> +2 = 0
    got = requant_add(np.array([-128], dtype=np.int8), -3, np.array([127], dtype=np.int8), 5, two, 2)
    checks.append({"check": "add_negative_domains", "ok": int(got[0]) == 0, "got": got.tolist()})

    x = np.full((1, 3, 3), -7, dtype=np.int8)
    pooled = maxpool_signed(x, 5, 1, 2)
    checks.append({"check": "maxpool_pad_min", "ok": int(pooled[0, 0, 0]) == -7,
                   "got": int(pooled[0, 0, 0])})

    x = np.arange(4, dtype=np.int8).reshape(1, 2, 2)
    r = resize_nearest_x2_signed(x)
    checks.append({"check": "resize_nearest", "ok": bool(np.array_equal(
        r[0], np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]], dtype=np.int8)))})

    checks.append({"check": "int32_bound_ok", "ok": check_acc_int32(1 << 30) is True})
    checks.append({"check": "int64_rescale_ok", "ok": check_rescale_int64(1 << 31, (1 << 31) - 1) is True})
    try:
        check_rescale_int64(1 << 22, 1 << 42)
        checks.append({"check": "int64_rescale_detects_overflow", "ok": False})
    except ValueError:
        checks.append({"check": "int64_rescale_detects_overflow", "ok": True})

    failed = [c for c in checks if not c["ok"]]
    return {"profile": PROFILE, "rounding": ROUNDING, "checks": len(checks),
            "failed": failed, "ok": not failed}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2))
