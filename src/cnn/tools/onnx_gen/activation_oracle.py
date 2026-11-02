"""Independent oracle for LUT activation tables (S2).

Must NOT import activation_lut. Uses mpmath with adaptive dps and explicit
ulp intervals. Leaky path uses exact Fractions independently.
"""
from __future__ import annotations
from fractions import Fraction
import math
try:
    import mpmath
except ImportError:
    mpmath = None

ORACLE_VERSION = "lut-oracle-1"

def _frac(v) -> Fraction:
    if isinstance(v, Fraction):
        return v
    if isinstance(v, int):
        return Fraction(v)
    f = float(v)
    if not math.isfinite(f):
        raise ValueError("non-finite")
    return Fraction.from_float(f)

def _round_away(x: Fraction) -> int:
    if x >= 0:
        return int((x + Fraction(1, 2)).__floor__())
    return -int(((-x) + Fraction(1, 2)).__floor__())

def _sat8(v: int) -> int:
    return max(-128, min(127, int(v)))

def _scales(s_in, s_out):
    si, so = _frac(s_in), _frac(s_out)
    if si <= 0 or so <= 0:
        raise ValueError("scales positive")
    lo, hi = Fraction(1, 65536), Fraction(16, 1)
    if not (lo <= si <= hi and lo <= so <= hi):
        raise ValueError("scale range")
    return si, so

def leaky_oracle(q: int, s_in, s_out, alpha) -> int:
    si, so = _scales(s_in, s_out)
    a = _frac(alpha)
    if not (Fraction(0) <= a <= Fraction(1)):
        raise ValueError("alpha range")
    x = Fraction(int(q)) * si
    y = x if int(q) >= 0 else a * x
    return _sat8(_round_away(y / so))

def _mp_interval(mode: str, q: int, si: Fraction, so: Fraction, dps: int):
    if mpmath is None:
        raise ValueError("mpmath missing")
    mpmath.mp.dps = dps + 10
    # x as mpf from exact fraction (numerator/denominator with high prec)
    x = mpmath.mpf(si.numerator) / mpmath.mpf(si.denominator) * mpmath.mpf(int(q))
    if mode == "Sigmoid":
        if x >= 0:
            f = mpmath.mpf(1) / (mpmath.mpf(1) + mpmath.exp(-x))
        else:
            e = mpmath.exp(x)
            f = e / (mpmath.mpf(1) + e)
    elif mode == "Tanh":
        f = mpmath.tanh(x)
    elif mode == "SiLU":
        if x >= 0:
            s = mpmath.mpf(1) / (mpmath.mpf(1) + mpmath.exp(-x))
        else:
            e = mpmath.exp(x)
            s = e / (mpmath.mpf(1) + e)
        f = x * s
    else:
        raise ValueError(mode)
    so_m = mpmath.mpf(so.numerator) / mpmath.mpf(so.denominator)
    r = f / so_m
    # ulp bound: 4*10^-(dps) relative+absolute
    ulp = mpmath.mpf(10) ** (-(dps - 5))
    rad = ulp * (abs(r) + mpmath.mpf(1)) * mpmath.mpf(4)
    lo = r - rad
    hi = r + rad
    return lo, hi

def _bucket_unique(lo, hi):
    # mpmath mpf comparisons; find single k with interval inside correct bucket (ties-away)
    # search k in range
    import math as _m
    try:
        lof, hif = float(lo), float(hi)
    except Exception:
        return None
    if not (_m.isfinite(lof) and _m.isfinite(hif)):
        return None
    klo = _m.floor(lof - 1) - 1
    khi = _m.ceil(hif + 1) + 1
    cands = []
    for k in range(klo, khi + 1):
        lb = k - 0.5
        hb = k + 0.5
        if k > 0:
            if lo >= lb and hi < hb:
                cands.append(k)
        elif k < 0:
            if lo > lb and hi <= hb:
                cands.append(k)
        else:
            if lo > lb and hi < hb:
                cands.append(k)
    if len(cands) == 1:
        return _sat8(cands[0])
    return None

def exp_oracle_single(mode: str, q: int, s_in, s_out) -> int:
    si, so = _scales(s_in, s_out)
    if int(q) == 0:
        if mode == "Sigmoid":
            return _sat8(_round_away(Fraction(1, 2) / so))
        if mode in ("SiLU", "Tanh"):
            return _sat8(0)
    # analytic large-|x| fast path (e>2 exact Fraction bounds, independent of mpmath)
    try:
        xr = Fraction(int(q)) * si
        if mode == "Sigmoid" and xr > 0 and xr >= 1:
            import math as _m
            try:
                xf = float(xr)
            except Exception:
                xf = 1e18
            xi = int(xr) if xr.denominator == 1 else _m.floor(xf) if xf < 1e15 else 0
            if xi >= 1:
                lo_b = (Fraction(1) - Fraction(1, 1 << int(xi))) / so
                hi_b = Fraction(1) / so
                for k in range(-130, 131):
                    if k > 0:
                        if lo_b >= Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    elif k < 0:
                        if lo_b > Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    else:
                        if lo_b > Fraction(-1, 2) and hi_b <= Fraction(1, 2):
                            return _sat8(0)
        elif mode == "Sigmoid" and xr < 0 and (-xr) >= 1:
            import math as _m2
            xi = int(-xr) if (-xr).denominator == 1 else 0
            if xi >= 1:
                hi_b = Fraction(1, 1 << int(xi)) / so
                if Fraction(0) > Fraction(-1, 2) and hi_b <= Fraction(1, 2):
                    return _sat8(0)
        elif mode == "Tanh" and xr > 0 and xr >= 1:
            import math as _m3
            try:
                xf3 = float(xr)
            except Exception:
                xf3 = 1e18
            xi3 = int(xr) if xr.denominator == 1 else _m3.floor(xf3) if xf3 < 1e15 else 0
            if xi3 >= 1:
                tiny = Fraction(1, 1 << (2 * int(xi3))) * 2
                lo_b = (Fraction(1) - tiny) / so
                hi_b = Fraction(1) / so
                for k in range(-130, 131):
                    if k > 0:
                        if lo_b >= Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    elif k < 0:
                        if lo_b > Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    else:
                        if lo_b > Fraction(-1, 2) and hi_b <= Fraction(1, 2):
                            return _sat8(0)
        elif mode == "Tanh" and xr < 0 and (-xr) >= 1:
            import math as _m4
            xi4 = int(-xr) if (-xr).denominator == 1 else 0
            if xi4 >= 1:
                tiny4 = Fraction(1, 1 << (2 * int(xi4))) * 2
                lo_b = Fraction(-1) / so
                hi_b = (Fraction(-1) + tiny4) / so
                for k in range(-130, 131):
                    if k > 0:
                        if lo_b >= Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    elif k < 0:
                        if lo_b > Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    else:
                        if lo_b > Fraction(-1, 2) and hi_b <= Fraction(1, 2):
                            return _sat8(0)
        elif mode == "SiLU" and xr > 0 and xr >= 20:
            import math as _m5
            try:
                xff = float(xr)
            except Exception:
                xff = 1e18
            xi5 = int(xr) if xr.denominator == 1 else _m5.floor(xff) if xff < 1e15 else 0
            if xi5 >= 20:
                lo_b = (xr - xr * Fraction(1, 1 << int(xi5))) / so
                hi_b = xr / so
                for k in range(-2000, 2000):
                    if k > 0:
                        if lo_b >= Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    elif k < 0:
                        if lo_b > Fraction(k) - Fraction(1, 2) and hi_b <= Fraction(k) + Fraction(1, 2):
                            return _sat8(k)
                    else:
                        if lo_b > Fraction(-1, 2) and hi_b <= Fraction(1, 2):
                            return _sat8(0)
        elif mode == "SiLU" and xr < 0 and (-xr) >= 20:
            import math as _m6
            xi6 = int(-xr) if (-xr).denominator == 1 else 0
            if xi6 >= 20:
                mag = (-xr) * Fraction(1, 1 << int(xi6)) / so
                if -mag > Fraction(-1, 2):
                    return _sat8(0)
    except Exception:
        pass
    # dps schedule ~ bits: 30~100b, 50~166b, 100~332b, 200~665b, 400~1330b, 460~1530b, 620~2060b
    for dps in (30, 50, 100, 200, 400, 460, 620):
        lo, hi = _mp_interval(mode, int(q), si, so, dps)
        ans = _bucket_unique(lo, hi)
        if ans is not None:
            return ans
    raise ValueError("oracle unprovable %s q=%s si=%s so=%s" % (mode, q, si, so))

def oracle_table(mode: str, s_in, s_out, alpha=None) -> list:
    si, so = _scales(s_in, s_out)
    if mode == "LeakyRelu":
        from fractions import Fraction as _F
        a = _frac(alpha if alpha is not None else 0.01)
        return [leaky_oracle(q, si, so, a) for q in range(-128, 128)]
    return [exp_oracle_single(mode, q, si, so) for q in range(-128, 128)]
