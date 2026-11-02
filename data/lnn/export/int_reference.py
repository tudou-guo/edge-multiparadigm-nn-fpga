#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bit-exact pure-Python integer reference for the CfC demo.

Implements docs/CFC_CONTRACT.md section 1.2 / 1.3 exactly:

  * x, h, y, weights : int16, Q12   (real = int / 4096)
  * biases           : int32, Q24   (real = int / 2**24)
  * accumulator      : exact Python int (logically signed 48 bit, >= 40 bit)
  * activations      : 1025-entry int16 LUTs over [-8, 8], step 1/64
  * gate             : g = sigmoid(lin_a * DELTA_T_INT + (lin_b << 12)), i.e.
                       b = W_b z + b_b is kept, Q24 exact then rounded to Q12
                       (semantics "B1"; see gate comment in run_step)

No third-party imports (no numpy, no torch), so it can be imported on the PYNQ
board or by any test script.  Parameters are loaded from the hex files sitting
next to this file:

    export/weights.hex, export/biases.hex,
    export/tanh_lut.hex, export/sigmoid_lut.hex

Main entry points
-----------------
run_step(x_int, h_int, delta_t_int=None, params=None, debug=False) -> dict
run_sequence(x_int, h_int=None, delta_t_int=None, clear_each_step=False,
             collect_stats=False, params=None) -> dict
load(export_dir=None) / reload(export_dir=None)

run_sequence() returns a dict with keys:
    y         : list of int, y[t] for t = 0 .. n-1            (int16 Q12)
    h_steps   : list of [h0,h1,h2,h3], state committed after step t
    h_final   : [h0,h1,h2,h3]
    n_steps   : n
    delta_t_int : delta_t_int
    stats     : only when collect_stats=True
"""

import math
import os

DATA_FRAC_BITS = 12
DATA_SCALE = 1 << DATA_FRAC_BITS
BIAS_FRAC_BITS = 24
BIAS_SCALE = 1 << BIAS_FRAC_BITS
DELTA_T_INT = 655
N_STATE = 4
N_IN = 1
N_WEIGHTS = 84
N_BIASES = 17
LUT_SIZE = 1025
N_LUT = LUT_SIZE

W1_BASE = 0
W2_BASE = 20
WA_BASE = 40
WB_BASE = 60
WO_BASE = 80

B1_BASE = 0
B2_BASE = 4
BA_BASE = 8
BB_BASE = 12
BO_INDEX = 16

_EXPORT_DIR = os.path.dirname(os.path.abspath(__file__))


def round_q12_sat16(v):
    if v >= 0:
        q = (v + 2048) >> 12
    else:
        q = -(((-v) + 2048) >> 12)
    return max(-32768, min(32767, q))


def lut_index(v):
    idx = (v + 32768 + 32) >> 6
    return max(0, min(1024, idx))


def round_half_away(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def sat16(v):
    return max(-32768, min(32767, v))


class Params(object):
    __slots__ = ("weights", "biases", "tanh_lut", "sigmoid_lut", "export_dir")

    def __init__(self, weights, biases, tanh_lut, sigmoid_lut, export_dir):
        self.weights = weights
        self.biases = biases
        self.tanh_lut = tanh_lut
        self.sigmoid_lut = sigmoid_lut
        self.export_dir = export_dir

    def __repr__(self):
        return "<CfC int params: %d weights, %d biases, %d+%d LUT, from %s>" % (
            len(self.weights), len(self.biases), len(self.tanh_lut),
            len(self.sigmoid_lut), self.export_dir)


def _read_hex_file(path, count, bits):
    """Read count two's-complement hex words of the given bit width."""
    vals = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("#"):
                continue
            vals.append(int(s, 16))
    if len(vals) != count:
        raise ValueError("%s: expected %d lines, got %d" % (path, count, len(vals)))
    half = 1 << (bits - 1)
    full = 1 << bits
    return [v - full if v >= half else v for v in vals]


_params = None


def load(export_dir=None):
    """Load weights / biases / LUTs from export_dir (default: this file's dir)."""
    global _params
    d = os.path.abspath(export_dir) if export_dir else _EXPORT_DIR
    p = Params(
        _read_hex_file(os.path.join(d, "weights.hex"), N_WEIGHTS, 16),
        _read_hex_file(os.path.join(d, "biases.hex"), N_BIASES, 32),
        _read_hex_file(os.path.join(d, "tanh_lut.hex"), LUT_SIZE, 16),
        _read_hex_file(os.path.join(d, "sigmoid_lut.hex"), LUT_SIZE, 16),
        d,
    )
    _params = p
    return p


def reload(export_dir=None):
    """Force a re-read of the hex files (useful after regenerating them)."""
    return load(export_dir)


def get_params():
    global _params
    if _params is None:
        load()
    return _params


def set_params(weights, biases, tanh_lut, sigmoid_lut):
    """Install parameter lists directly (used by generators / tests)."""
    global _params
    _params = Params(list(weights), list(biases), list(tanh_lut),
                     list(sigmoid_lut), "<memory>")
    return _params


def run_step(x_int, h_int, delta_t_int=None, params=None, debug=False):
    p = params if params is not None else get_params()
    dt = DELTA_T_INT if delta_t_int is None else int(delta_t_int)
    x = int(x_int)
    hs = [int(v) for v in h_int]
    if len(hs) != N_STATE:
        raise ValueError("h_int must have %d entries, got %d" % (N_STATE, len(hs)))
    if not -32768 <= x <= 32767:
        raise ValueError("x_int out of int16 range: %r" % (x,))
    for v in hs:
        if not -32768 <= v <= 32767:
            raise ValueError("h_int entry out of int16 range: %r" % (v,))
    if not 1 <= dt <= 32767:
        raise ValueError("delta_t_int must be in 1..32767, got %r" % (dt,))

    W = p.weights
    B = p.biases
    z = [x, hs[0], hs[1], hs[2], hs[3]]

    # step 1: four linear branches, bias(Q24) + sum W(Q12)*z(Q12) -> Q24
    accs = []
    lin = []
    for br in range(4):
        wbase = br * 20
        bbase = br * 4
        branch_acc = []
        for o in range(4):
            acc = B[bbase + o]
            base = wbase + o * 5
            acc += W[base] * z[0]
            acc += W[base + 1] * z[1]
            acc += W[base + 2] * z[2]
            acc += W[base + 3] * z[3]
            acc += W[base + 4] * z[4]
            branch_acc.append(acc)
        accs.append(branch_acc)
        lin.append([round_q12_sat16(v) for v in branch_acc])
    lin1, lin2, lin_a, lin_b = lin
    acc1, acc2, acc_a, acc_b = accs

    # step 2: tanh lookups
    f1 = [p.tanh_lut[lut_index(v)] for v in lin1]
    f2 = [p.tanh_lut[lut_index(v)] for v in lin2]

    # step 3: gate pre-activation -> round to Q12
    # Contract 1.1 / requirement 2.1 define b = W_b z_t + b_b, whose Q12 value is
    # lin_b from step 1, so the exact Q24 gate pre-activation is
    #     lin_a * DELTA_T_INT + (lin_b << 12)
    # which is rounded once to int16 Q12 (no intermediate truncation of the
    # product/sum).  This is resolution "B1" agreed with the Lead; the literal
    # text of contract 1.3 step 3 wrote the bb bias directly, which would drop
    # the W_b z term entirely and contradict 1.1 / requirement 2.1.
    gate_raw = [lin_a[o] * dt + (lin_b[o] << 12) for o in range(N_STATE)]
    gate_pre = [round_q12_sat16(v) for v in gate_raw]

    # step 4: sigmoid lookup
    g = [p.sigmoid_lut[lut_index(v)] for v in gate_pre]

    # step 5: state fusion, exact Q24 before rounding
    h_new = [round_q12_sat16(f1[o] * (DATA_SCALE - g[o]) + f2[o] * g[o])
             for o in range(N_STATE)]

    # step 6: readout y = bo[0] + sum Wo[i]*h_new[i] (Q24) -> Q12
    y_acc = B[BO_INDEX]
    for i in range(N_STATE):
        y_acc += W[WO_BASE + i] * h_new[i]
    y = round_q12_sat16(y_acc)

    res = {"y": y, "h_next": h_new}
    if debug:
        res["debug"] = {
            "z": z,
            "acc1": acc1, "acc2": acc2, "acc_a": acc_a, "acc_b": acc_b,
            "lin1": lin1, "lin2": lin2, "lin_a": lin_a, "lin_b": lin_b,
            "f1": f1, "f2": f2,
            "gate_raw": gate_raw, "gate_pre": gate_pre, "g": g,
            "y_acc": y_acc,
        }
    return res


def run_sequence(x_int, h_int=None, delta_t_int=None, clear_each_step=False,
                 collect_stats=False, params=None):
    """Run a whole sequence with the integer model.

    clear_each_step=True implements MODE[1]=1: h_old = 0 at every step while
    the computed new state is still committed (see contract 3.4).
    """
    p = params if params is not None else get_params()
    dt = DELTA_T_INT if delta_t_int is None else int(delta_t_int)
    if not 1 <= dt <= 32767:
        raise ValueError("delta_t_int must be in 1..32767, got %r" % (dt,))
    xs = [int(v) for v in x_int]
    for v in xs:
        if not -32768 <= v <= 32767:
            raise ValueError("x_int out of int16 range: %r" % (v,))
    if h_int is None:
        h = [0, 0, 0, 0]
    else:
        h = [int(v) for v in h_int]
        if len(h) != N_STATE:
            raise ValueError("h_int must have %d entries" % N_STATE)

    ys = []
    h_steps = []
    stats = None
    if collect_stats:
        stats = {
            "max_abs_acc_1": 0, "max_abs_acc_2": 0,
            "max_abs_acc_a": 0, "max_abs_acc_b": 0, "max_abs_acc_y": 0,
            "max_abs_lin1": 0, "max_abs_lin2": 0,
            "max_abs_lin_a": 0, "max_abs_lin_b": 0,
            "max_abs_gate_pre": 0, "max_abs_h": 0, "max_abs_y": 0,
            "max_abs_x": max([abs(v) for v in xs]) if xs else 0,
            "sat_count": 0, "lut_index_clipped": 0,
            "y_min": 32767, "y_max": -32768,
        }

    for t, xv in enumerate(xs):
        h_old = [0, 0, 0, 0] if clear_each_step else h
        r = run_step(xv, h_old, dt, params=p, debug=collect_stats)
        h = r["h_next"]
        ys.append(r["y"])
        h_steps.append(list(h))
        if collect_stats:
            d = r["debug"]
            for key, dbg_key in (("max_abs_acc_1", "acc1"), ("max_abs_acc_2", "acc2"),
                                 ("max_abs_acc_a", "acc_a"), ("max_abs_acc_b", "acc_b")):
                stats[key] = max(stats[key], max(abs(v) for v in d[dbg_key]))
            stats["max_abs_acc_y"] = max(stats["max_abs_acc_y"], abs(d["y_acc"]))
            for key, dbg_key in (("max_abs_lin1", "lin1"), ("max_abs_lin2", "lin2"),
                                 ("max_abs_lin_a", "lin_a"), ("max_abs_lin_b", "lin_b"),
                                 ("max_abs_gate_pre", "gate_pre")):
                stats[key] = max(stats[key], max(abs(v) for v in d[dbg_key]))
            stats["max_abs_h"] = max(stats["max_abs_h"], max(abs(v) for v in h))
            stats["max_abs_y"] = max(stats["max_abs_y"], abs(r["y"]))
            stats["y_min"] = min(stats["y_min"], r["y"])
            stats["y_max"] = max(stats["y_max"], r["y"])
            for arr in (d["acc1"], d["acc2"], d["acc_a"], d["acc_b"]):
                for v in arr:
                    if v > 32767 * 4096 + 2048 or v < -32768 * 4096 - 2048:
                        stats["sat_count"] += 1
            if d["y_acc"] > 32767 * 4096 + 2048 or d["y_acc"] < -32768 * 4096 - 2048:
                stats["sat_count"] += 1
            for v in d["gate_raw"]:
                if v > 32767 * 4096 + 2048 or v < -32768 * 4096 - 2048:
                    stats["sat_count"] += 1
            for arr in (d["lin1"], d["lin2"], d["gate_pre"]):
                for v in arr:
                    raw_idx = (v + 32768 + 32) >> 6
                    if raw_idx < 0 or raw_idx > 1024:
                        stats["lut_index_clipped"] += 1

    return {
        "y": ys,
        "h_steps": h_steps,
        "h_final": list(h),
        "n_steps": len(xs),
        "delta_t_int": dt,
        "clear_each_step": bool(clear_each_step),
        "stats": stats,
    } if collect_stats else {
        "y": ys,
        "h_steps": h_steps,
        "h_final": list(h),
        "n_steps": len(xs),
        "delta_t_int": dt,
        "clear_each_step": bool(clear_each_step),
    }


step = run_step
sequence = run_sequence


if __name__ == "__main__":
    p = get_params()
    print(p)
    out = run_sequence([0] * 64, [0, 0, 0, 0], DELTA_T_INT)
    print("zero-input y[:8]:", out["y"][:8])
    print("h_final:", out["h_final"])
