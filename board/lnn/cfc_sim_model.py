#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline AXI-Lite register model for the CfC PYNQ demo (NOT HARDWARE).

Purpose
-------
There is no Zynq/ZU15EG board attached to the development machine, so the PYNQ
driver (lnn_cfc_run.py) is validated against this Python model instead.  The
model implements:

  * docs/CFC_CONTRACT.md section 1.3 - the exact integer CfC cell semantics
    (int16 Q12 weights/data/h, int32 Q24 biases, 48-bit logical accumulators,
    ROUND_Q12_SAT16 = round-half-away-from-zero, LUT index rule, LUT contents);
  * docs/CFC_CONTRACT.md section 3.4 - the frozen CFC1 register map and its
    behaviour (START/CLEAR_STATE/CLEAR_STATUS, busy/done/error, config snapshot,
    MODE bits, run-cycle counter, timeout, sign-extended data windows).

Every artefact produced through this model must be reported with
source = "simulation-model" and the note "未在真实板卡上运行".
This module must never be presented as board evidence.

The module also generates a deterministic, contract-conforming *fixture* so the
driver/notebook plumbing can be exercised before the real export/ tree or the
real bitstream exist.  Fixture outputs are computed by this module itself, so a
fixture run validates plumbing (register sequence, bit packing, comparison
logic) only - not the trained-model numbers.  When a real export/ is available,
the model loads those files and the fixture is not used.

CLI
---
  python cfc_sim_model.py --make-fixture DIR       # write the fixture tree
  python cfc_sim_model.py --selfcheck [DIR]        # contract self-tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

# ------------------------------------------------------------------ constants
# Frozen by docs/CFC_CONTRACT.md - do not change without a contract revision.
CFC_ID = 0x43464331
CFC_VERSION = 0x00010000
CSR_BASE = 0x0480000000
CSR_SIZE = 0x1000
CLOCK_HZ = 300000000
N_STEPS_MAX = 64
N_INPUT = 64
N_OUTPUT = 64
N_H4 = 4
DELTA_T_INT_DEFAULT = 655
DELTA_T_FLOAT = 0.16
TIMEOUT_CYCLES_DEFAULT = 1000000
Q12 = 4096
N_WEIGHTS = 84
N_BIASES = 17
N_LUT = 1025
WARMUP_SKIP = 5

REG = {
    "ID": 0x000,
    "VERSION": 0x004,
    "CONTROL": 0x008,
    "STATUS": 0x00C,
    "ERROR_CODE": 0x010,
    "N_STEPS": 0x014,
    "DELTA_T": 0x018,
    "MODE": 0x01C,
    "RUN_CYCLES": 0x020,
    "STEP_COUNT": 0x024,
    "TIMEOUT_CYCLES": 0x028,
    "CLOCK_HZ": 0x02C,
    "H_FINAL": 0x080,
    "INPUT": 0x100,
    "OUTPUT": 0x200,
}

CTRL_START = 0x1
CTRL_CLEAR_STATE = 0x2
CTRL_CLEAR_STATUS = 0x4

STATUS_BUSY = 0x1
STATUS_DONE = 0x2
STATUS_ERROR = 0x4

ERROR_CODES = {0: "NONE", 1: "CONFIG", 2: "BUSY_START", 3: "TIMEOUT"}

DISCLAIMER = (
    "Python register model (simulation-model): NOT a hardware measurement. "
    "未在真实板卡上运行。"
)


# ------------------------------------------------------------- integer helpers
def sat16(v):
    if v < -32768:
        return -32768
    if v > 32767:
        return 32767
    return int(v)


def sat32(v):
    if v < -2147483648:
        return -2147483648
    if v > 2147483647:
        return 2147483647
    return int(v)


def round_q12_sat16(v):
    """Contract section 1.3 ROUND_Q12_SAT16 (round half away from zero)."""
    v = int(v)
    if v >= 0:
        q = (v + 2048) >> 12
    else:
        q = -(((-v) + 2048) >> 12)
    return sat16(q)


def lut_index(v):
    """Contract section 1.3 LUT address rule for an int16 Q12 value."""
    idx = (int(v) + 32768 + 32) >> 6
    if idx < 0:
        return 0
    if idx > 1024:
        return 1024
    return idx


def round_half_away(x):
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def to_s16(word):
    word = int(word) & 0xFFFF
    return word - 0x10000 if word >= 0x8000 else word


def to_s32(word):
    word = int(word) & 0xFFFFFFFF
    return word - 0x100000000 if word >= 0x80000000 else word


def sign_extend16_word(v):
    """16-bit signed value -> 32-bit word with sign extension in bits 31:16."""
    v = int(v) & 0xFFFF
    return (v | 0xFFFF0000) if (v & 0x8000) else v


def make_contract_luts():
    """Return (tanh_lut, sigmoid_lut) exactly as contract section 1.3."""
    tanh_lut = []
    sigmoid_lut = []
    for i in range(N_LUT):
        x = (i - 512) / 64.0
        tanh_lut.append(sat16(round_half_away(math.tanh(x) * Q12)))
        sigmoid_lut.append(sat16(round_half_away((1.0 / (1.0 + math.exp(-x))) * Q12)))
    return tanh_lut, sigmoid_lut


def read_hex_words(path, bits=16):
    """Read a hex file (one value per line, optional // comments)."""
    vals = []
    with open(str(path), "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.split("//")[0].strip()
            if not s:
                continue
            s = s.replace("_", "").replace("0x", "").replace("0X", "")
            v = int(s, 16)
            vals.append(to_s16(v) if bits == 16 else to_s32(v))
    return vals


def write_hex_words(path, values, bits=16):
    width = 4 if bits == 16 else 8
    lines = []
    for v in values:
        u = int(v) & (0xFFFF if bits == 16 else 0xFFFFFFFF)
        lines.append(("%0" + str(width) + "X") % u)
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


# --------------------------------------------------------- integer CfC reference
class CfcIntegerReference(object):
    """Bit-exact integer reference for the CfC cell (contract section 1.3).

    Used by the simulation model to produce expected outputs when no golden is
    available, and by the driver to build expected values for the per-step-clear
    mode (the contract's export/ only carries default-mode golden vectors).
    """

    def __init__(self, weights, biases, tanh_lut, sigmoid_lut):
        if len(weights) != N_WEIGHTS:
            raise ValueError("weights: got %d, expected %d" % (len(weights), N_WEIGHTS))
        if len(biases) != N_BIASES:
            raise ValueError("biases: got %d, expected %d" % (len(biases), N_BIASES))
        if len(tanh_lut) != N_LUT or len(sigmoid_lut) != N_LUT:
            raise ValueError("LUT length must be %d" % N_LUT)
        self.w = [to_s16(v) for v in weights]
        self.b = [to_s32(v) for v in biases]
        self.tanh_lut = [to_s16(v) for v in tanh_lut]
        self.sigmoid_lut = [to_s16(v) for v in sigmoid_lut]

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dir(cls, expect_dir):
        """Load export/{weights,biases,tanh_lut,sigmoid_lut}.hex from a directory."""
        d = Path(str(expect_dir))
        need = ["weights.hex", "biases.hex", "tanh_lut.hex", "sigmoid_lut.hex"]
        missing = [n for n in need if not (d / n).is_file()]
        if missing:
            raise FileNotFoundError(
                "missing reference files in %s: %s" % (d, ", ".join(missing)))
        return cls(
            read_hex_words(d / "weights.hex", 16),
            read_hex_words(d / "biases.hex", 32),
            read_hex_words(d / "tanh_lut.hex", 16),
            read_hex_words(d / "sigmoid_lut.hex", 16),
        )

    # -- one step -----------------------------------------------------------
    def run_step(self, x_int, h_old, delta_t_int):
        z = [to_s16(x_int)] + [to_s16(v) for v in h_old]

        def branch(wbase, bbase):
            out = []
            for o in range(4):
                acc = self.b[bbase + o]
                base = wbase + o * 5
                for i in range(5):
                    acc += self.w[base + i] * z[i]
                out.append(acc)
            return out

        lin1 = [round_q12_sat16(v) for v in branch(0, 0)]
        lin2 = [round_q12_sat16(v) for v in branch(20, 4)]
        lina = [round_q12_sat16(v) for v in branch(40, 8)]
        linb = [round_q12_sat16(v) for v in branch(60, 12)]

        f1 = [self.tanh_lut[lut_index(v)] for v in lin1]
        f2 = [self.tanh_lut[lut_index(v)] for v in lin2]
        # Contract V1.1 section 1.3 step 3:
        #   gate_pre = ROUND_Q12_SAT16(lin_a*DELTA_T_INT + (lin_b << 12))
        # lin_a and lin_b are the rounded/saturated int16 Q12 values from step 1;
        # (lin_b << 12) is the exact Q12 -> Q24 extension of b_t = W_b z + b_b.
        gate_pre = [round_q12_sat16(lina[o] * int(delta_t_int) + (linb[o] << 12))
                    for o in range(4)]
        g = [self.sigmoid_lut[lut_index(v)] for v in gate_pre]
        h_new = []
        for o in range(4):
            acc = f1[o] * (4096 - g[o]) + f2[o] * g[o]
            h_new.append(round_q12_sat16(acc))
        acc_y = self.b[16]
        for i in range(4):
            acc_y += self.w[80 + i] * h_new[i]
        y = round_q12_sat16(acc_y)
        return y, h_new, {
            "lin1": lin1, "lin2": lin2, "lina": lina, "linb": linb,
            "f1": f1, "f2": f2, "gate_pre": gate_pre, "g": g,
        }

    # -- sequence -----------------------------------------------------------
    def run_sequence(self, x_int, h_int, delta_t_int, clear_each_step=False):
        h = [to_s16(v) for v in h_int]
        ys = []
        h_steps = []
        for x in x_int:
            h_old = [0, 0, 0, 0] if clear_each_step else h
            y, h, _ = self.run_step(x, h_old, delta_t_int)
            ys.append(y)
            h_steps.append(list(h))
        return {
            "y": ys,
            "h": list(h),
            "h_steps": h_steps,
            "delta_t_int": int(delta_t_int),
            "clear_each_step": bool(clear_each_step),
            "n_steps": len(ys),
        }


# --------------------------------------------------------------- fixture maker
def _json_num_list(vals, ndigits=6):
    out = []
    for v in vals:
        if v is None:
            out.append(None)
        else:
            out.append(round(float(v), ndigits))
    return out


def make_fixture(dest):
    """Create a deterministic contract-conforming fake export tree under dest."""
    dest = Path(str(dest))
    gen = dest / "gen"
    gen.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261031)

    def wint():
        return sat16(round_half_away(rng.uniform(-0.85, 0.85) * Q12))

    def bint():
        return sat32(round_half_away(rng.uniform(-0.4, 0.4) * (1 << 24)))

    weights = [wint() for _ in range(N_WEIGHTS)]
    biases = [bint() for _ in range(N_BIASES)]
    tanh_lut, sigmoid_lut = make_contract_luts()

    write_hex_words(dest / "weights.hex", weights, 16)
    write_hex_words(dest / "biases.hex", biases, 32)
    write_hex_words(dest / "tanh_lut.hex", tanh_lut, 16)
    write_hex_words(dest / "sigmoid_lut.hex", sigmoid_lut, 16)

    # deterministic 64-step sine, period 16 samples, phase 0 (fixture sample)
    phase = 0.0
    x_float = [0.8 * math.sin(2.0 * math.pi * t / 16.0 + phase) for t in range(N_STEPS_MAX)]
    x_int = [sat16(round_half_away(v * Q12)) for v in x_float]
    h_init = [0, 0, 0, 0]
    ref = CfcIntegerReference(weights, biases, tanh_lut, sigmoid_lut)
    sim = ref.run_sequence(x_int, h_init, DELTA_T_INT_DEFAULT, clear_each_step=False)
    y_int = sim["y"]
    h_flat = [v for step in sim["h_steps"] for v in step]

    write_hex_words(gen / "x_input.hex", x_int, 16)
    write_hex_words(gen / "h_init.hex", h_init, 16)
    write_hex_words(gen / "y_expected.hex", y_int, 16)
    write_hex_words(gen / "h_expected.hex", h_flat, 16)

    target = [None] * 4 + x_float[: len(x_float) - 4]  # input delayed by 4 steps
    sample = {
        "id": "fixture_s0",
        "fixture": True,
        "note": ("generated by cfc_sim_model.make_fixture; NOT the trained model. "
                 "float_y is set to the task target because the fixture has no trained model."),
        "phase": phase,
        "seed": 20261031,
        "dt_float": DELTA_T_FLOAT,
        "delta_t_int": DELTA_T_INT_DEFAULT,
        "n_steps": N_STEPS_MAX,
        "warmup_skip": WARMUP_SKIP,
        "float_x": _json_num_list(x_float),
        "float_y": _json_num_list(target),
        "target": _json_num_list(target),
        "int_x": x_int,
        "int_y": y_int,
        "h_init": h_init,
        "h_steps": sim["h_steps"],
    }
    with open(str(gen / "sample_fixture_s0.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(sample, fh, indent=1)
        fh.write("\n")

    files = {}
    for rel in ["weights.hex", "biases.hex", "tanh_lut.hex", "sigmoid_lut.hex",
                "gen/x_input.hex", "gen/h_init.hex", "gen/y_expected.hex",
                "gen/h_expected.hex", "gen/sample_fixture_s0.json"]:
        h = hashlib.sha256((dest / rel).read_bytes()).hexdigest()
        files[rel] = h
    manifest = {
        "schema": "cfc-export-manifest-fixture-1",
        "fixture": True,
        "note": "SYNTHETIC fixture generated by board/cfc_sim_model.py - NOT model-engineer output",
        "model": {"input_size": 1, "hidden_size": 4, "output_size": 1, "mode": "CfC default, no backbone"},
        "id": "0x%08X" % CFC_ID,
        "version": "0x%08X" % CFC_VERSION,
        "csr_base": "0x%X" % CSR_BASE,
        "clock_hz": CLOCK_HZ,
        "delta_t_int": DELTA_T_INT_DEFAULT,
        "n_steps": N_STEPS_MAX,
        "files": files,
    }
    with open(str(dest / "manifest.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return {"dir": str(dest), "sample": "fixture_s0", "fixture": True}


# ---------------------------------------------------------------- register model
class SimMmio(object):
    """Strict behavioural model of the CFC1 AXI-Lite window (contract 3.4).

    Exposes read(offset) / write(offset, value) with 32-bit words relative to
    CSR_BASE, exactly like pynq.MMIO.  kind is "simulation-model" so the
    driver/summary can never mistake it for hardware.

    Timing is simulated: the device advances its internal cycle counter on every
    MMIO access, and the modelled run completes once RUN_CYCLES of simulated
    time have elapsed.  This lets the driver's poll/wall-clock-timeout logic be
    exercised deterministically.  The cycle numbers are model assumptions, not
    measurements.
    """

    kind = "simulation-model"

    def __init__(self, expect_dir=None, clock_hz=CLOCK_HZ,
                 cycles_base=30, cycles_per_step=45, read_cost=20, write_cost=1,
                 wrong_id=False, wrong_version=False, wrong_clock=False,
                 fault=None, fixture_info=None):
        self.clock_hz = int(clock_hz)
        self.cycle_model = {"base": int(cycles_base), "per_step": int(cycles_per_step),
                            "read_cost": int(read_cost), "write_cost": int(write_cost)}
        self.read_cost = int(read_cost)
        self.write_cost = int(write_cost)
        self.cycles_base = int(cycles_base)
        self.cycles_per_step = int(cycles_per_step)
        self.wrong_id = bool(wrong_id)
        self.wrong_version = bool(wrong_version)
        self.wrong_clock = bool(wrong_clock)
        self.fault = fault
        self.fixture_info = fixture_info
        self.ref = CfcIntegerReference.from_dir(expect_dir)
        self.access_log = []
        self._now = 0
        self._reset()

    # -- state --------------------------------------------------------------
    def _reset(self):
        self.n_steps = N_STEPS_MAX
        self.delta_t = DELTA_T_INT_DEFAULT
        self.mode = 1
        self.timeout_cycles = TIMEOUT_CYCLES_DEFAULT
        self.busy = False
        self.done = False
        self.error = False
        self.error_code = 0
        self.run_cycles = 0
        self.step_count = 0
        self.h = [0, 0, 0, 0]
        self.din = [0] * N_INPUT
        self.dout = [0] * N_INPUT
        self._pending = None
        self._done_at = None
        self._timeout_pending = False
        self._snapshot = None
        self._now = 0

    # -- helpers ------------------------------------------------------------
    def _advance(self, cycles):
        self._now += int(cycles)

    def _log(self, op, offset, value, extra=None):
        rec = {"t": self._now, "op": op, "offset": int(offset) & 0xFFF,
               "value": int(value) & 0xFFFFFFFF}
        if extra:
            rec.update(extra)
        self.access_log.append(rec)

    def _maybe_complete(self):
        if self.busy and self._done_at is not None and self._now >= self._done_at:
            self._complete()

    def _complete(self):
        self.busy = False
        self._done_at = None
        if self._timeout_pending:
            self._timeout_pending = False
            self.error = True
            self.error_code = 3
            self.done = False
            self.step_count = 0
            self.run_cycles = 0
            self._pending = None
            self._snapshot = None
            return
        self.done = True
        self.error = False
        self.error_code = 0
        res = self._pending or {}
        self.step_count = int(res.get("n_steps", 0))
        self.run_cycles = int(res.get("run_cycles", 0))
        self.dout = list(res.get("y", [0] * N_INPUT))
        self.h = list(res.get("h", [0, 0, 0, 0]))
        self._pending = None
        self._snapshot = None

    # -- AXI-Lite word access ----------------------------------------------
    def read(self, offset):
        self._advance(self.read_cost)
        self._maybe_complete()
        w = self._decode_read(int(offset) & 0xFFF)
        self._log("read", offset, w)
        return int(w) & 0xFFFFFFFF

    def write(self, offset, value):
        self._advance(self.write_cost)
        self._maybe_complete()
        off = int(offset) & 0xFFF
        val = int(value) & 0xFFFFFFFF
        extra = self._decode_write(off, val)
        self._log("write", off, val, extra)
        return None

    # -- decode -------------------------------------------------------------
    def _decode_read(self, off):
        if off == REG["ID"]:
            return 0xDEADBEEF if self.wrong_id else CFC_ID
        if off == REG["VERSION"]:
            return 0x00000000 if self.wrong_version else CFC_VERSION
        if off == REG["CONTROL"]:
            return 0
        if off == REG["STATUS"]:
            st = 0
            if self.busy:
                st |= STATUS_BUSY
            if self.done:
                st |= STATUS_DONE
            if self.error:
                st |= STATUS_ERROR
            return st
        if off == REG["ERROR_CODE"]:
            return self.error_code & 0xFFFFFFFF
        if off == REG["N_STEPS"]:
            return self.n_steps & 0xFFFFFFFF
        if off == REG["DELTA_T"]:
            return self.delta_t & 0xFFFFFFFF
        if off == REG["MODE"]:
            return self.mode & 0xFFFFFFFF
        if off == REG["RUN_CYCLES"]:
            return self.run_cycles & 0xFFFFFFFF
        if off == REG["STEP_COUNT"]:
            return self.step_count & 0xFFFFFFFF
        if off == REG["TIMEOUT_CYCLES"]:
            return self.timeout_cycles & 0xFFFFFFFF
        if off == REG["CLOCK_HZ"]:
            return (123456789 if self.wrong_clock else self.clock_hz) & 0xFFFFFFFF
        if REG["H_FINAL"] <= off < REG["H_FINAL"] + 4 * N_H4:
            idx = (off - REG["H_FINAL"]) // 4
            if (off - REG["H_FINAL"]) % 4 != 0:
                return 0
            return sign_extend16_word(self.h[idx])
        if REG["INPUT"] <= off < REG["INPUT"] + 4 * N_INPUT:
            if (off - REG["INPUT"]) % 4 != 0:
                return 0
            idx = (off - REG["INPUT"]) // 4
            return sign_extend16_word(self.din[idx])
        if REG["OUTPUT"] <= off < REG["OUTPUT"] + 4 * N_OUTPUT:
            if (off - REG["OUTPUT"]) % 4 != 0:
                return 0
            idx = (off - REG["OUTPUT"]) // 4
            return sign_extend16_word(self.dout[idx])
        return 0

    def _decode_write(self, off, val):
        # Busy: input/config writes are accepted on the bus but discarded
        # (contract 3.4 "输入/配置写入不改变当前任务").
        if off == REG["CONTROL"]:
            return self._control_write(val)
        if off in (REG["N_STEPS"], REG["DELTA_T"], REG["MODE"], REG["TIMEOUT_CYCLES"]):
            if self.busy:
                return {"discarded": "busy"}
            if off == REG["N_STEPS"]:
                self.n_steps = val & 0xFFFFFFFF
            elif off == REG["DELTA_T"]:
                self.delta_t = val & 0xFFFF
            elif off == REG["MODE"]:
                self.mode = val & 0x7
            else:
                self.timeout_cycles = val & 0xFFFFFFFF
            return None
        if REG["INPUT"] <= off < REG["INPUT"] + 4 * N_INPUT:
            if (off - REG["INPUT"]) % 4 != 0:
                return {"ignored": "unaligned"}
            if self.busy:
                return {"discarded": "busy"}
            idx = (off - REG["INPUT"]) // 4
            self.din[idx] = to_s16(val & 0xFFFF)
            return None
        # ID/VERSION/STATUS/ERROR_CODE/RUN_CYCLES/STEP_COUNT/CLOCK_HZ/H_FINAL/OUTPUT
        # are read-only; writes are ignored (contract 3.4 "写入忽略").
        return {"ignored": "ro-or-unmapped"}

    def _control_write(self, val):
        if val & CTRL_START:
            self._start()
        if val & CTRL_CLEAR_STATE:
            if self.busy:
                return {"clear_state": "ignored-busy"}
            self.h = [0, 0, 0, 0]
            return {"clear_state": "done"}
        if val & CTRL_CLEAR_STATUS:
            self.done = False
            self.error = False
            self.error_code = 0
            return {"clear_status": "done"}
        return None

    def _start(self):
        if self.busy:
            self.error_code = 2
            self.error = True
            return
        n = int(self.n_steps)
        dt = int(self.delta_t)
        if n < 1 or n > N_STEPS_MAX:
            self.error_code = 1
            self.error = True
            return
        if dt < 1 or dt > 32767:
            self.error_code = 1
            self.error = True
            return
        snap = {"n_steps": n, "delta_t": dt, "mode": int(self.mode),
                "h_in": list(self.h), "x": list(self.din)}
        self._snapshot = snap
        self.busy = True
        self.done = False
        self.error = False
        self.error_code = 0
        self.run_cycles = 0
        self.step_count = 0

        h = [0, 0, 0, 0] if (snap["mode"] & 0x1) else [to_s16(v) for v in snap["h_in"]]
        y_out = [0] * N_INPUT
        for t in range(n):
            h_old = [0, 0, 0, 0] if (snap["mode"] & 0x2) else h
            y, h, _ = self.ref.run_step(snap["x"][t], h_old, snap["delta_t"])
            y_out[t] = y
        run_cycles = self.cycles_base + self.cycles_per_step * n
        self._pending = {"y": y_out, "h": h, "n_steps": n, "run_cycles": run_cycles}

        budget = int(self.timeout_cycles)
        if budget < 1:
            budget = 1
        if self.fault == "timeout" or run_cycles > budget:
            self._timeout_pending = True
            self._done_at = self._now + budget
        elif self.fault == "stuck":
            self._timeout_pending = False
            self._done_at = self._now + 10 ** 12
        else:
            self._timeout_pending = False
            self._done_at = self._now + run_cycles

    # -- diagnostics --------------------------------------------------------
    def snapshot(self):
        return {
            "id": "0x%08X" % (0xDEADBEEF if self.wrong_id else CFC_ID),
            "version": "0x%08X" % (0x00000000 if self.wrong_version else CFC_VERSION),
            "clock_hz": 123456789 if self.wrong_clock else self.clock_hz,
            "status": ("busy" if self.busy else ("done" if self.done else ("error" if self.error else "idle"))),
            "error_code": self.error_code,
            "n_steps": self.n_steps,
            "delta_t": self.delta_t,
            "mode": self.mode,
            "run_cycles": self.run_cycles,
            "step_count": self.step_count,
            "h": list(self.h),
            "sim_cycles": self._now,
            "access_log_len": len(self.access_log),
        }

    def info(self):
        d = {
            "backend": "python-sim-model",
            "module": "cfc_sim_model.SimMmio",
            "not_hardware": True,
            "disclaimer": DISCLAIMER,
            "cycle_model": dict(self.cycle_model),
            "clock_hz": self.clock_hz,
            "fault": self.fault,
        }
        if self.fixture_info:
            d["fixture"] = self.fixture_info
        return d


# ------------------------------------------------------------------- self-test
def _selfcheck(expect_dir=None):
    errors = []

    def check(name, cond, detail=""):
        print(("OK  " if cond else "FAIL") + " " + name + ((" | " + detail) if detail and not cond else ""))
        if not cond:
            errors.append(name + ": " + detail)

    tanh_lut, sig_lut = make_contract_luts()
    check("tanh_lut[512] == 0", tanh_lut[512] == 0, str(tanh_lut[512]))
    check("tanh_lut[0] clamped", tanh_lut[0] == sat16(round_half_away(math.tanh(-8.0) * 4096)), str(tanh_lut[0]))
    check("tanh_lut[1024] clamped", tanh_lut[1024] == sat16(round_half_away(math.tanh(8.0) * 4096)), str(tanh_lut[1024]))
    check("sigmoid_lut[512] == 2048", sig_lut[512] == 2048, str(sig_lut[512]))
    check("sigmoid_lut[0] in range", 0 < sig_lut[0] < 20, str(sig_lut[0]))

    check("round_q12_sat16(2048)==1", round_q12_sat16(2048) == 1)
    check("round_q12_sat16(-2048)==-1", round_q12_sat16(-2048) == -1)
    check("round_q12_sat16(2047)==0", round_q12_sat16(2047) == 0)
    check("round_q12_sat16 sat16 high", round_q12_sat16(1 << 40) == 32767)
    check("round_q12_sat16 sat16 low", round_q12_sat16(-(1 << 40)) == -32768)
    check("lut_index(-32768)==0", lut_index(-32768) == 0)
    check("lut_index(32767)==1024", lut_index(32767) == 1024)
    check("lut_index(0)==512", lut_index(0) == 512, str(lut_index(0)))

    # V1.1 gate regression: gate_pre must include the full b_t branch (W_b z + b_b),
    # not just the b_b bias as in the withdrawn V1.0 formula.
    #   Wa[0][x]=4096 (1.0), Wb[0][x]=2048 (0.5), x=4096 (1.0), dt=655
    #   lin_a[0]=4096 -> 4096*655 = 2682880 ; lin_b[0]=2048 -> 2048<<12 = 8388608
    #   gate_pre[0] = round((2682880+8388608)/4096) = 2703
    # (the V1.0 formula without the W_b term would give 655)
    toy_w = [0] * N_WEIGHTS
    toy_b = [0] * N_BIASES
    toy_w[40 + 0 * 5 + 0] = 4096   # Wa[0][x]
    toy_w[60 + 0 * 5 + 0] = 2048   # Wb[0][x]
    toy_ref = CfcIntegerReference(toy_w, toy_b, tanh_lut, sig_lut)
    _, _, toy_dbg = toy_ref.run_step(4096, [0, 0, 0, 0], DELTA_T_INT_DEFAULT)
    check("gate_pre includes W_b branch (contract V1.1)", toy_dbg["gate_pre"][0] == 2703,
          "got %s expect 2703" % (toy_dbg["gate_pre"][0],))
    check("lin_b computed from Wb", toy_dbg["linb"][0] == 2048, str(toy_dbg["linb"][0]))

    if expect_dir:
        try:
            ref = CfcIntegerReference.from_dir(expect_dir)
            d = Path(str(expect_dir))
            tanh_hex = read_hex_words(d / "tanh_lut.hex", 16)
            sig_hex = read_hex_words(d / "sigmoid_lut.hex", 16)
            check("export tanh_lut matches contract formula", tanh_hex == tanh_lut)
            check("export sigmoid_lut matches contract formula", sig_hex == sig_lut)
            gen = d / "gen"
            if (gen / "y_expected.hex").is_file() and (gen / "x_input.hex").is_file():
                xs = read_hex_words(gen / "x_input.hex", 16)
                h0 = read_hex_words(gen / "h_init.hex", 16) if (gen / "h_init.hex").is_file() else [0, 0, 0, 0]
                y_exp = read_hex_words(gen / "y_expected.hex", 16)
                got = ref.run_sequence(xs, h0, DELTA_T_INT_DEFAULT, clear_each_step=False)
                bad = [i for i in range(min(len(y_exp), len(got["y"]))) if y_exp[i] != got["y"][i]]
                check("export y_expected matches integer reference", not bad,
                      "first mismatch idx %s" % (bad[:5],))
                if (gen / "h_expected.hex").is_file():
                    h_exp = read_hex_words(gen / "h_expected.hex", 16)
                    flat = [v for step in got["h_steps"] for v in step]
                    badh = [i for i in range(min(len(h_exp), len(flat))) if h_exp[i] != flat[i]]
                    check("export h_expected matches integer reference", not badh,
                          "first mismatch idx %s" % (badh[:5],))
            else:
                print("SKIP export golden vectors (not present)")
        except Exception as exc:  # pragma: no cover
            check("export cross-check", False, repr(exc))

    print("== cfc_sim_model selfcheck: %s ==" % ("PASS" if not errors else "FAIL (%d)" % len(errors)))
    return 0 if not errors else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline CFC1 register model / fixture tool (NOT HARDWARE)")
    ap.add_argument("--make-fixture", metavar="DIR", default=None,
                    help="write a deterministic contract-conforming fixture export tree to DIR")
    ap.add_argument("--selfcheck", nargs="?", const="", default=None, metavar="EXPORT_DIR",
                    help="run model self-tests; optionally cross-check a real export dir")
    args = ap.parse_args(argv)
    if args.make_fixture:
        info = make_fixture(args.make_fixture)
        print(json.dumps(info, indent=2))
        return 0
    if args.selfcheck is not None:
        return _selfcheck(args.selfcheck or None)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
