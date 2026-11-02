"""E1 LUT activation table generator (S2).

Generator only. Independent oracle lives in activation_oracle.py and must not
be imported here. Critical near-half values use Decimal high-precision
intervals with adaptive precision; fast-float results at those points are
corrected, not trusted.
"""
from __future__ import annotations
from decimal import Decimal, getcontext, ROUND_HALF_UP
from fractions import Fraction
import math

GENERATOR_VERSION = "lut-gen-1"

def _exact_frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite scale/alpha")
    return Fraction.from_float(f)

def round_away_frac(x: Fraction) -> int:
    if x >= 0:
        return int((x + Fraction(1, 2)).__floor__())
    return -int(((-x) + Fraction(1, 2)).__floor__())

def sat8(v: int) -> int:
    return max(-128, min(127, int(v)))

def _check_scales(s_in, s_out):
    si = _exact_frac(s_in)
    so = _exact_frac(s_out)
    if si <= 0 or so <= 0:
        raise ValueError("scales must be positive")
    # range 2^-16 .. 16
    lo = Fraction(1, 65536)
    hi = Fraction(16, 1)
    if not (lo <= si <= hi and lo <= so <= hi):
        raise ValueError("scale out of range 2^-16..16: %s %s" % (si, so))
    return si, so

def _check_alpha(alpha) -> Fraction:
    a = _exact_frac(alpha)
    if not (Fraction(0) <= a <= Fraction(1)):
        raise ValueError("alpha out of range 0..1")
    return a

def leaky_exact(q: int, s_in, s_out, alpha) -> int:
    si, so = _check_scales(s_in, s_out)
    a = _check_alpha(alpha)
    x = Fraction(int(q)) * si
    y = x if int(q) >= 0 else a * x
    return sat8(round_away_frac(y / so))

def _sigmoid_decimal(xd: Decimal) -> Decimal:
    # stable: x>=0 -> 1/(1+exp(-x)) else exp(x)/(1+exp(x))
    if xd >= 0:
        e = (-xd).exp()
        return Decimal(1) / (Decimal(1) + e)
    else:
        e = xd.exp()
        return e / (Decimal(1) + e)

def _tanh_decimal(xd: Decimal) -> Decimal:
    if xd >= 0:
        # 1 - 2/(exp(2x)+1)
        e2 = (xd * 2).exp()
        return Decimal(1) - Decimal(2) / (e2 + Decimal(1))
    else:
        e2 = (-xd * 2).exp()
        return Decimal(2) / (e2 + Decimal(1)) - Decimal(1)

def _dec_interval_single(mode: str, q: int, si: Fraction, so: Fraction, alpha: Fraction | None, prec: int):
    # returns (low_frac_bound_as_Decimal_ratio, high, rounded_or_None, straddle_bool)
    # compute interval [v-2ulp, v+2ulp] with ulp=10^-(prec-5)
    getcontext().prec = prec + 10
    xd = Decimal(si.numerator) / Decimal(si.denominator) * Decimal(int(q))
    if mode == "Sigmoid":
        fd = _sigmoid_decimal(xd)
    elif mode == "Tanh":
        fd = _tanh_decimal(xd)
    elif mode == "SiLU":
        fd = xd * _sigmoid_decimal(xd)
    else:
        raise ValueError(mode)
    sod = Decimal(so.numerator) / Decimal(so.denominator)
    r = fd / sod
    ulp = Decimal(1).scaleb(-(prec - 5))
    # guard: Decimal exp correctly rounded to prec -> true within 1 ulp; use 2 ulp interval
    lo = r - ulp * (abs(r).copy_abs() + Decimal(1))
    hi = r + ulp * (abs(r).copy_abs() + Decimal(1))
    # find candidate integers: interval must lie strictly inside (k-0.5, k+0.5) OR be provable exact half
    # check exact-half special cases first
    if int(q) == 0:
        if mode == "Sigmoid":
            # true 0.5 exactly
            true_r = Decimal(1) / Decimal(2) / sod
            # exact rational 1/(2*so); round away
            # compute exact via Fraction: 1/2 / so
            exact = Fraction(1, 2) / so
            return (lo, hi, sat8(round_away_frac(exact)), False)
        if mode in ("SiLU", "Tanh"):
            return (lo, hi, sat8(0), False)
    # general: try each k overlapped
    # low/high as float bounds for bucket search: use Decimal compare to half boundaries
    # candidate = round_away of midpoint
    mid = (lo + hi) / 2
    # determine k by rounding mid away (Decimal half away? use Fraction approx)
    # convert mid to Fraction approximately with limited denominator? Instead search integer k in [floor(lo)-1, ceil(hi)+1]
    import math as _m
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except Exception:
        lo_f, hi_f = -1e300, 1e300
    if not (_m.isfinite(lo_f) and _m.isfinite(hi_f)):
        return (lo, hi, None, True)
    k_lo = _m.floor(lo_f - 0.5)
    k_hi = _m.ceil(hi_f + 0.5)
    cands = []
    for k in range(k_lo - 1, k_hi + 2):
        low_b = Decimal(k) - Decimal("0.5")
        high_b = Decimal(k) + Decimal("0.5")
        # ties-away: value exactly on boundary belongs to outward bucket:
        # for k>=0, upper boundary k+0.5 belongs to k+1? Actually round_away(k+0.5)=k+1 for k>=0.
        # So bucket for k is (k-0.5, k+0.5) plus lower boundary if k<=0? Let's define precisely:
        # round_away(v)=k iff v in (k-0.5, k+0.5) for k!=0? Check: k=1: (0.5,1.5) but 0.5->1 inclusive lower? 0.5 rounds to 1, so lower inclusive. 1.5->2 exclusive upper.
        # k=0: (-0.5,0.5) but -0.5->-1, 0.5->1, so both exclusive.
        # k=-1: (-1.5,-0.5) upper exclusive? -0.5->-1 inclusive upper.
        # General: lower inclusive iff k>0, upper inclusive iff k<0, zero exclusive both.
        if k > 0:
            inside = (lo >= low_b) and (hi < high_b or (hi == high_b and False))
            # allow hi==high_b? That would be exactly k+0.5 which rounds to k+1, not k. So must be strictly < high_b.
            # lo==low_b allowed (k-0.5 rounds to k for k>0).
            if inside:
                # also need lo>low_b? No, lo==low_b ok. hi must be < high_b.
                cands.append(k)
        elif k < 0:
            inside = (lo > low_b) and (hi <= high_b)
            if inside:
                cands.append(k)
        else:  # k==0
            inside = (lo > low_b) and (hi < high_b)
            if inside:
                cands.append(k)
    if len(cands) == 1:
        return (lo, hi, sat8(cands[0]), False)
    return (lo, hi, None, True)

def exp_single(mode: str, q: int, s_in, s_out, alpha=None) -> int:
    si, so = _check_scales(s_in, s_out)
    if mode == "LeakyRelu":
        if alpha is None:
            raise ValueError("Leaky needs alpha")
        return leaky_exact(int(q), si, so, _check_alpha(alpha))
    # fast analytic large-|x| path with rigorous e>2 bounds (exact Fractions, no exp)
    # e>2 => for t>0: 0<exp(-t)<2^-t. All bounds below are exact rationals.
    try:
        xr = Fraction(int(q)) * si  # exact x_real
        if mode == "Sigmoid":
            if xr > 0:
                # (1-2^-xr)/so < r < 1/so ; xr may be fractional -> need integer exponent? xr is Fraction, not int.
                # Use ceil: 2^-xr <= 2^-floor(xr). Since xr>0, floor>=0 for xr>=1 else bound 1/2?
                # For 0<xr<1, use trivial (0.5,1) bounds? Fall through to Decimal.
                if xr >= 1 and xr.denominator == 1:
                    xi = int(xr)
                    lo_b = (Fraction(1) - Fraction(1, 1 << xi)) / so
                    hi_b = Fraction(1) / so
                    # check single bucket
                    for k in range(-130, 131):
                        if k > 0:
                            if lo_b >= Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2) and lo_b < hi_b:
                                # also need lo_b inside bucket: lo_b >= k-0.5
                                # ensure entire (lo_b,hi_b) inside (k-0.5,k+0.5) with lower inclusive
                                if lo_b >= Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2) and not (hi_b == Fraction(k) + Fraction(1,2)):
                                    # need hi strictly < upper
                                    if hi_b <= Fraction(k) + Fraction(1,2):
                                        return sat8(k)
                        elif k < 0:
                            if lo_b > Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b > Fraction(-1,2) and hi_b <= Fraction(1,2):
                                return sat8(0)
                elif xr >= 20:
                    # fractional large xr: use floor bound 2^-floor(xr)
                    import math as _mm
                    xi = _mm.floor(float(xr)) if float(xr) < 1e15 else int(xr)
                    lo_b = (Fraction(1) - Fraction(1, 1 << int(xi))) / so
                    hi_b = Fraction(1) / so
                    for k in range(-130, 131):
                        if k > 0:
                            if lo_b >= Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        elif k < 0:
                            if lo_b > Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b > Fraction(-1,2) and hi_b <= Fraction(1,2):
                                return sat8(0)
            elif xr < 0:
                if -xr >= 1 and (-xr).denominator == 1:
                    xi = int(-xr)
                    hi_b = Fraction(1, 1 << xi) / so
                    lo_b = Fraction(0)
                    for k in range(-130, 131):
                        if k > 0:
                            if lo_b >= Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        elif k < 0:
                            if lo_b > Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b > Fraction(-1,2) and hi_b <= Fraction(1,2):
                                return sat8(0)
        elif mode == "Tanh":
            if xr > 0 and xr >= 1:
                import math as _mm2
                try:
                    xf = float(xr)
                except Exception:
                    xf = 1e18
                xi = int(xr) if xr.denominator == 1 else _mm2.floor(xf) if xf < 1e15 else 0
                if xi >= 1:
                    tiny = Fraction(1, 1 << (2 * int(xi))) * 2  # 2^(1-2xi)
                    lo_b = (Fraction(1) - tiny) / so
                    hi_b = Fraction(1) / so
                    for k in range(-130, 131):
                        if k > 0:
                            if lo_b >= Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        elif k < 0:
                            if lo_b > Fraction(k) - Fraction(1,2) and hi_b <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b > Fraction(-1,2) and hi_b <= Fraction(1,2):
                                return sat8(0)
            elif xr < 0 and (-xr) >= 1:
                import math as _mm3
                try:
                    xf2 = float(-xr)
                except Exception:
                    xf2 = 1e18
                from fractions import Fraction as _F
                xi2 = int(-xr) if (-xr).denominator == 1 else _mm3.floor(xf2) if xf2 < 1e15 else 0
                if xi2 >= 1:
                    tiny2 = Fraction(1, 1 << (2 * int(xi2))) * 2
                    lo_b2 = Fraction(-1) / so
                    hi_b2 = (Fraction(-1) + tiny2) / so
                    for k in range(-130, 131):
                        if k > 0:
                            if lo_b2 >= Fraction(k) - Fraction(1,2) and hi_b2 <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        elif k < 0:
                            if lo_b2 > Fraction(k) - Fraction(1,2) and hi_b2 <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b2 > Fraction(-1,2) and hi_b2 <= Fraction(1,2):
                                return sat8(0)
        elif mode == "SiLU":
            if xr > 0 and xr >= 20:
                import math as _mm4
                try:
                    xff = float(xr)
                except Exception:
                    xff = 1e18
                xi3 = int(xr) if xr.denominator == 1 else _mm4.floor(xff) if xff < 1e15 else 0
                if xi3 >= 20:
                    lo_b3 = (xr - xr * Fraction(1, 1 << int(xi3))) / so
                    hi_b3 = xr / so
                    for k in range(-2000, 2000):
                        if k > 0:
                            if lo_b3 >= Fraction(k) - Fraction(1,2) and hi_b3 <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        elif k < 0:
                            if lo_b3 > Fraction(k) - Fraction(1,2) and hi_b3 <= Fraction(k) + Fraction(1,2):
                                return sat8(k)
                        else:
                            if lo_b3 > Fraction(-1,2) and hi_b3 <= Fraction(1,2):
                                return sat8(0)
            elif xr < 0 and (-xr) >= 20:
                # xr negative large: SiLU = xr*sigmoid(xr), |r| <= |xr|*2^-|xr|/so tiny
                import math as _mm5
                try:
                    xn = float(-xr)
                except Exception:
                    xn = 1e18
                xi4 = int(-xr) if (-xr).denominator == 1 else _mm5.floor(xn) if xn < 1e15 else 0
                if xi4 >= 20:
                    mag = (-xr) * Fraction(1, 1 << int(xi4)) / so
                    lo_b4 = -mag
                    hi_b4 = Fraction(0)
                    # interval (lo,0): must lie in single bucket; since mag tiny (<0.5) => bucket 0
                    if lo_b4 > Fraction(-1,2) and hi_b4 <= Fraction(1,2):
                        return sat8(0)
    except Exception:
        pass
    # adaptive precisions (decimal digits ~ bits: 50~166b, 100~332b, 200~665b, 400~1330b, 600~1995b)
    for prec in (50, 100, 200, 400, 600):
        lo, hi, ans, straddle = _dec_interval_single(mode, int(q), si, so, None, prec)
        if not straddle:
            return ans
    raise ValueError("unprovable rounding for %s q=%s s_in=%s s_out=%s (interval straddles at max prec)" % (mode, q, si, so))

def generate_table(mode: str, s_in, s_out, alpha=None) -> list:
    si, so = _check_scales(s_in, s_out)
    if mode == "LeakyRelu":
        a = _check_alpha(alpha if alpha is not None else 0.01)
        return [leaky_exact(q, si, so, a) for q in range(-128, 128)]
    else:
        return [exp_single(mode, q, si, so) for q in range(-128, 128)]

def table_sha256(table: list) -> str:
    import hashlib
    return hashlib.sha256(bytes(int(v) & 0xFF for v in table)).hexdigest()
