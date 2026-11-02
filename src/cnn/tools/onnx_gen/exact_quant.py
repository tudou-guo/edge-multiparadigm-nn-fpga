"""Exact rational helpers for P3 Mul/BN integer semantics.

Every number that participates in a P3 integer-equivalence proof is converted
through its exact binary rational value (Fraction.from_float), never through an
epsilon.  A ratio is accepted only when it is exactly M0 / 2**SHIFT with
0 < M0 <= 0xFFFFFF and 0 <= SHIFT <= 31.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, Dict, Tuple

MAX_M0_P3 = 0xFFFFFF
MAX_SHIFT_P3 = 31


class ExactQuantError(ValueError):
    """Raised when an exact numeric contract cannot be satisfied."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def exact(value: Any) -> Fraction:
    """Exact rational value of a declared float/int."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise ExactQuantError("E_QUANT_MISMATCH", "boolean is not a numeric contract value")
    if isinstance(value, int):
        return Fraction(value)
    number = float(value)
    if not math.isfinite(number):
        raise ExactQuantError("E_QUANT_MISMATCH", "non-finite numeric contract value %r" % value)
    return Fraction.from_float(number)


def round_away(value: Fraction) -> int:
    """Round a rational to the nearest integer, ties away from zero."""
    if value >= 0:
        return int((value + Fraction(1, 2)).__floor__())
    return -int(((-value) + Fraction(1, 2)).__floor__())


def ratio_to_m0_shift(ratio: Fraction, *, max_m0: int = MAX_M0_P3,
                      max_shift: int = MAX_SHIFT_P3) -> Tuple[int, int]:
    """Exact representation of ratio as M0 / 2**SHIFT.

    The search prefers the smallest shift that fits, so the value is always the
    exact ratio; there is no rounding anywhere in this function.
    """
    if ratio <= 0:
        raise ExactQuantError("E_QUANT_MISMATCH", "requant ratio %s is not positive" % ratio)
    for shift in range(0, max_shift + 1):
        scaled = ratio * (1 << shift)
        if scaled.denominator == 1 and 1 <= scaled.numerator <= max_m0:
            return int(scaled.numerator), shift
    raise ExactQuantError(
        "E_QUANT_MISMATCH",
        "ratio %.17g has no exact M0/2^SHIFT form with M0<=0x%X and SHIFT<=%d" %
        (float(ratio), max_m0, max_shift))


def m0_shift_value(m0: int, shift: int) -> Fraction:
    return Fraction(int(m0), 1 << int(shift))


def quantize_product(x: int, y: int, m0: int, shift: int) -> int:
    """Exactly the RTL mul lane: sat8(round_away(x*y*M0/2^shift))."""
    product = int(x) * int(y) * int(m0)
    value = round_away(Fraction(product, 1 << int(shift)))
    return max(-128, min(127, value))


def sat8(value: int) -> int:
    return max(-128, min(127, int(value)))


def bn_channel_terms(gamma: Any, beta: Any, mean: Any, var: Any, epsilon: Any
                     ) -> Dict[str, Any]:
    """Return exact/analytic BN folding terms for one output channel."""
    eps = exact(epsilon)
    if eps <= 0:
        raise ExactQuantError("E_UNSUPPORTED_ATTR", "epsilon must be finite and positive")
    g = exact(gamma)
    b = exact(beta)
    m = exact(mean)
    v = exact(var)
    sigma_sq = v + eps
    if sigma_sq <= 0:
        raise ExactQuantError("E_UNSUPPORTED_ATTR", "var + epsilon must be positive")
    # sigma and 1/sigma are irrational in general; the *float* BN parameters
    # therefore define A = gamma / sqrt(var+eps) as a float, and the fold uses
    # the exact float value of that quotient.  The pre-check below rejects
    # degenerate or non-finite values before that float is formed.
    sigma = math.sqrt(float(sigma_sq))
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ExactQuantError("E_UNSUPPORTED_ATTR", "sqrt(var+epsilon) is not finite")
    a_float = float(g) / sigma
    b_float = float(b) - float(g) * float(m) / sigma
    if not math.isfinite(a_float) or not math.isfinite(b_float):
        raise ExactQuantError("E_UNSUPPORTED_ATTR", "BN affine terms are not finite")
    return {"gamma": g, "beta": b, "mean": m, "var": v, "epsilon": eps,
            "sigma": sigma, "a_float": a_float, "b_float": b_float}


def bn_fold_channel(*, gamma: Any, beta: Any, mean: Any, var: Any, epsilon: Any,
                    input_scale: Any, weight_scale: Any, output_scale: Any,
                    bias_int: int) -> Dict[str, Any]:
    """Exact fold for one channel or raise ExactQuantError with the reason.

    accumulator domain scale  s_acc = input_scale * weight_scale
    BN terms                  A = gamma / sqrt(var+eps),  B = beta - A*mean

    The original integer semantics evaluated by the independent reference is
        y = A * (acc * s_acc) + B,   out = sat8(round_away(y / s_out))
    where acc is the raw INT32 Conv accumulator *including the original bias*.
    The folded Conv computes
        out = sat8(round_away((acc + delta) * M0 / 2**SHIFT))
    so exact equality for every acc requires

        A * s_acc / s_out == M0 / 2**SHIFT        (exact ratio)
        delta * A * s_acc / s_out == B / s_out    (exact bias delta)
      => delta = B / (A * s_acc)                  (exact integer)

    The delta is in pre-A accumulator units; folding it into the INT32 bias is
    only possible when it is an exact integer.
    """
    terms = bn_channel_terms(gamma, beta, mean, var, epsilon)
    a_float = terms["a_float"]
    b_float = terms["b_float"]
    if a_float <= 0.0:
        raise ExactQuantError(
            "E_UNSUPPORTED_ATTR",
            "folded scale gamma/sqrt(var+eps)=%.17g is not positive" % a_float)
    s_in = exact(input_scale)
    s_w = exact(weight_scale)
    s_out = exact(output_scale)
    if s_in <= 0 or s_w <= 0 or s_out <= 0:
        raise ExactQuantError("E_QUANT_MISMATCH", "quantization scales must be positive")
    s_acc = s_in * s_w
    ratio = exact(a_float) * s_acc / s_out
    m0, shift = ratio_to_m0_shift(ratio)
    delta = exact(b_float) / (exact(a_float) * s_acc)
    if delta.denominator != 1:
        raise ExactQuantError(
            "E_UNSUPPORTED_ATTR",
            "BN additive term is not an exact integer in the pre-scale accumulator "
            "domain: B/(A*s_acc) = %s" % delta)
    delta_int = int(delta)
    new_bias = int(bias_int) + delta_int
    if not (-(2 ** 31) <= new_bias <= 2 ** 31 - 1):
        raise ExactQuantError("E_CFG", "folded INT32 bias overflows: %d" % new_bias)
    return {"a_float": a_float, "b_float": b_float, "s_acc": s_acc,
            "ratio": ratio, "M0": m0, "SHIFT": shift,
            "bias_delta": delta_int, "bias_int": new_bias,
            "sigma": terms["sigma"]}
