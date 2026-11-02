#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PYNQ board driver for the LNN CfC demo (CFC1 AXI-Lite peripheral).

Frozen interface: docs/CFC_CONTRACT.md section 3.4 (requirements document
section 4).  The CSR window is the existing platform M00_AXI_0 window at
physical address 0x0480000000 (4 KiB); the address is 64-bit and is never
truncated to 32 bits here.

What this driver does
---------------------
  * loads the matching PYNQ overlay (lnn_cfc.bit + lnn_cfc.hwh, same stem) and
    maps the CSR window with pynq.MMIO;
  * verifies CFC1 ID (0x43464331), VERSION (0x00010000), CLOCK_HZ (300 MHz) and
    cross-checks export/cfc_spec.py + export/manifest.json when available;
  * writes up to 64 int16 Q12 input samples (low 16 bits, signed) and the
    N_STEPS / DELTA_T / MODE configuration, issues START (W1P), then polls
    STATUS.DONE/ERROR with an independent wall-clock deadline;
  * reads OUTPUT / H_FINAL / STEP_COUNT / RUN_CYCLES, compares every output
    point and the final hidden state bit-exactly against the export golden,
    computes the task MSE (FPGA output vs the delayed-input task target,
    skipping the first 5 warm-up steps);
  * writes summary.json with sample id, match counts, final state, task MSE,
    run cycles, hardware time (RUN_CYCLES / CLOCK_HZ), end-to-end wall time and
    the SHA256 of the bit / hwh / manifest.
  * continues with the required state tests: repeat run, 64 = 32 + 32 with
    retained state, and the per-step-clear mode.

Evidence honesty
----------------
source is "hardware" only when the real PYNQ overlay + MMIO backend is used.
Without a board the driver runs against board/cfc_sim_model.py (a strict Python
behavioural model) and reports source "simulation-model" with the note
"未在真实板卡上运行".  A simulation-model run must never be quoted as board
evidence.

CLI (run from the package directory on the board):
  python lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export \
      --sample sample_s0 --out-dir results/s0_normal --checks
  python lnn_cfc_run.py --list-samples --expect-dir export
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

try:  # numpy is used for .npy artefacts and the float-reference fallback only
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# ------------------------------------------------------------------ constants
# Frozen by docs/CFC_CONTRACT.md section 0/1/3.4 + requirements section 4.
CSR_BASE = 0x0480000000
CSR_SIZE = 0x1000
CFC_ID = 0x43464331
CFC_VERSION = 0x00010000
CLOCK_HZ_EXPECTED = 300000000
N_STEPS_MAX = 64
N_INPUT_WORDS = 64
N_OUTPUT_WORDS = 64
N_HIDDEN = 4
DELTA_T_INT_DEFAULT = 655
DELTA_T_FLOAT = 0.16
TIMEOUT_CYCLES_DEFAULT = 1000000
Q_SCALE = 4096
WARMUP_SKIP_DEFAULT = 5
DEFAULT_TIMEOUT_S = 10.0
POLL_PERIOD_S = 0.0002

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

MODE_MAP = {
    "normal": 0b01,          # clear state before run, keep state across steps
    "retain": 0b00,          # keep prior state before run
    "clear-each-step": 0b11,  # clear before run + use h_old=0 every step
    "clear-per-step-only": 0b10,
}
MODE_NAMES = {1: "normal", 0: "retain", 3: "clear-each-step", 2: "clear-per-step-only"}

DISCLAIMER = (
    "simulation-model: Python register model output, NOT a hardware measurement. "
    "未在真实板卡上运行。"
)

if CSR_BASE != 0x0480000000 or CSR_BASE <= 0xFFFFFFFF:
    raise RuntimeError("CFC CSR base must stay the 64-bit 0x0480000000 address")

# Monotonic high-resolution clock for the end-to-end / wait / readback
# measurements (requirement section 3.2 "板端单调时钟").  perf_counter is
# monotonic; on the board (Linux) it has ns resolution, and on Windows dev
# machines it avoids the coarse GetTickCount64 granularity of _clock().
_clock = time.perf_counter


class CfcBoardError(RuntimeError):
    """Driver error with machine-readable diagnostics for the notebook."""

    def __init__(self, message, diagnostics=None):
        super(CfcBoardError, self).__init__(message)
        self.diagnostics = dict(diagnostics or {})


# --------------------------------------------------------------------- helpers
def now_utc_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sat16(v):
    if v < -32768:
        return -32768
    if v > 32767:
        return 32767
    return int(v)


def to_s16(word):
    word = int(word) & 0xFFFF
    return word - 0x10000 if word >= 0x8000 else word


def to_word16(v):
    """int16 value -> 32-bit MMIO word; bits 31:16 are ignored by the hardware."""
    return int(v) & 0xFFFF


def sign_extend16_word(v):
    v = int(v) & 0xFFFF
    return (v | 0xFFFF0000) if (v & 0x8000) else v


def read_hex_words(path, bits=16):
    vals = []
    with open(str(path), "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.split("//")[0].strip()
            if not s:
                continue
            s = s.replace("_", "").replace("0x", "").replace("0X", "")
            v = int(s, 16)
            vals.append(to_s16(v) if bits == 16 else (v - 0x100000000 if v >= 0x80000000 else v))
    return vals


def write_json(path, obj):
    path = Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return str(path)


def _deep_find(obj, keys, max_depth=4):
    """Search nested dicts for the first key in keys; returns (found, value)."""
    if max_depth < 0 or not isinstance(obj, dict):
        return False, None
    for k in keys:
        if k in obj:
            return True, obj[k]
    for v in obj.values():
        if isinstance(v, dict):
            f, val = _deep_find(v, keys, max_depth - 1)
            if f:
                return True, val
    return False, None


def _parse_int(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        s = v.strip().replace("_", "")
        try:
            return int(s, 0)
        except ValueError:
            return int(s, 16)
    return None


def _parse_verilog_int(text):
    """Parse a SystemVerilog integer literal such as 84, 16'd655, 16'h028F."""
    s = str(text).strip().rstrip(";")
    if "'" in s:
        width, rest = s.split("'", 1)
        rest = rest.replace("_", "")
        if rest and rest[0].lower() in "dhbo":
            base_ch = rest[0].lower()
            digits = rest[1:]
        else:
            base_ch = "d"
            digits = rest
        base = {"d": 10, "h": 16, "b": 2, "o": 8}.get(base_ch, 10)
        return int(digits, base)
    return int(s, 0)


def check_rom_params(expect_dir):
    """Check export/gen/cfc_rom_params.svh (the RTL include) against the contract.

    The RTL and the board driver must agree on CFC_N_W/CFC_N_B/CFC_N_LUT/
    CFC_DT_INT/CFC_N_STEPS_MAX; a mismatch means the bitstream was built from
    different generated parameters.
    """
    path = Path(str(expect_dir)) / "gen" / "cfc_rom_params.svh"
    out = {"found": path.is_file(), "path": str(path), "checks": {}, "ok": True}
    if not out["found"]:
        return out
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        out["ok"] = False
        out["error"] = repr(exc)
        return out
    defines = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] == "`define":
            try:
                defines[parts[1]] = _parse_verilog_int(parts[2])
            except Exception:
                continue
    expected = [("CFC_N_W", 84), ("CFC_N_B", 17), ("CFC_N_LUT", 1025),
                ("CFC_DT_INT", DELTA_T_INT_DEFAULT),
                ("CFC_N_STEPS_MAX", N_STEPS_MAX)]
    for name, frozen in expected:
        if name not in defines:
            out["checks"][name] = {"found": False, "ok": None}
            continue
        ok = defines[name] == frozen
        out["checks"][name] = {"found": True, "value": defines[name],
                               "frozen": frozen, "ok": ok}
        if not ok:
            out["ok"] = False
    return out


def _to_int_list(v, n=None):
    if v is None:
        return None
    out = [int(x) for x in v]
    if n is not None:
        out = out[:n]
    return out


def _to_float_list(v, n=None):
    if v is None:
        return None
    out = [None if x is None else float(x) for x in v]
    if n is not None:
        out = out[:n]
    return out


def _normalize_h_steps(v, n_h=4):
    """Normalise per-step hidden states: nested [[h0..h3],...] or flat 256 list."""
    if v is None or not isinstance(v, list):
        return None
    if v and isinstance(v[0], (list, tuple)):
        return [[int(x) for x in row[:n_h]] for row in v]
    try:
        flat = [int(x) for x in v]
    except Exception:
        return None
    if len(flat) % n_h != 0:
        return None
    return [flat[i:i + n_h] for i in range(0, len(flat), n_h)]


# ----------------------------------------------------------------- path lookup
def discover_paths(bit=None, hwh=None, xsa=None, expect_dir=None):
    """Resolve bit/hwh/xsa/export locations for repo and package layouts."""
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    env = os.environ

    def first_file(cands):
        for c in cands:
            if c is None:
                continue
            p = Path(str(c))
            if p.is_file():
                return p.resolve()
        return None

    def first_dir(cands):
        for c in cands:
            if c is None:
                continue
            p = Path(str(c))
            if p.is_dir():
                return p.resolve()
        return None

    bit_p = first_file([
        bit, env.get("CFC_BIT"),
        cwd / "lnn_cfc.bit", here / "lnn_cfc.bit",
        here.parent / "lnn_cfc.bit", here.parent / "board" / "lnn_cfc.bit",
        cwd / "board" / "lnn_cfc.bit",
    ])
    hwh_cands = [hwh, env.get("CFC_HWH"),
                 here / "lnn_cfc.hwh", here.parent / "lnn_cfc.hwh",
                 here.parent / "board" / "lnn_cfc.hwh"]
    if bit_p is not None:
        hwh_cands.append(bit_p.with_suffix(".hwh"))
    hwh_p = first_file(hwh_cands)
    xsa_p = first_file([
        xsa, env.get("CFC_XSA"),
        cwd / "lnn_cfc.xsa", here / "lnn_cfc.xsa",
        here.parent / "lnn_cfc.xsa", here.parent / "board" / "lnn_cfc.xsa",
        cwd / "board" / "lnn_cfc.xsa",
    ])
    if expect_dir is not None and not Path(str(expect_dir)).is_dir():
        raise CfcBoardError(
            "expect directory not found: %s" % expect_dir,
            {"expect_dir": str(expect_dir)})

    # A usable export/ directory holds the integer reference plus the weight/LUT
    # hex files (and normally gen/ with the golden vectors).  Score the candidates
    # so the same driver works from the repo layout (D:/DL_LNN/export), from the
    # delivered package (<pkg>/export), and from a board copy where the driver and
    # bitstream live in one directory and the export tree in <dir>/export.
    def export_score(p):
        score = 0
        for name in ("weights.hex", "biases.hex", "int_reference.py", "manifest.json"):
            if (p / name).is_file():
                score += 1
        if (p / "gen").is_dir():
            score += 1
        return score

    exp_cands = [
        expect_dir, env.get("CFC_EXPECT_DIR"),
        here / "export", here / "expected", here / "export_data",
        here / "package" / "export", here / "package",
        here.parent / "export", here.parent / "expected",
        cwd / "export", cwd / "package" / "export", cwd.parent / "export",
    ]
    # one level of look-around below the driver dir / cwd: covers
    # <root>/board/export, <root>/board/package/export, <driver>/pack*/export, ...
    for base in (here, cwd):
        try:
            children = sorted((x for x in base.iterdir() if x.is_dir()), key=str)
        except OSError:
            children = []
        for child in children:
            exp_cands.append(child / "export")
            exp_cands.append(child)
    exp_p = None
    best_score = 0
    searched = []
    for cand in exp_cands:
        if cand is None:
            continue
        p = Path(str(cand))
        if not p.is_dir():
            searched.append("%s (missing)" % p)
            continue
        s = export_score(p)
        searched.append("%s (score %d)" % (p.resolve(), s))
        if s >= 4 and s > best_score:
            best_score = s
            exp_p = p.resolve()
    return {
        "driver_dir": str(here),
        "cwd": str(cwd),
        "bit": str(bit_p) if bit_p else None,
        "hwh": str(hwh_p) if hwh_p else None,
        "xsa": str(xsa_p) if xsa_p else None,
        "expect_dir": str(exp_p) if exp_p else None,
        "expect_dir_searched": searched,
    }


def discover_samples(expect_dir):
    """Return the sorted sample ids available under export/gen (sample_*.json)."""
    d = Path(str(expect_dir))
    found = []
    for base in (d / "gen", d):
        if base.is_dir():
            for p in sorted(base.glob("sample_*.json")):
                if p.stem not in found:
                    found.append(p.stem)
    return found


def load_gen_vectors(expect_dir):
    gen = Path(str(expect_dir)) / "gen"
    out = {}
    for key, name in (("x_init", "x_input.hex"), ("h_init", "h_init.hex"),
                      ("y_expected", "y_expected.hex"), ("h_expected", "h_expected.hex")):
        p = gen / name
        out[key] = read_hex_words(p, 16) if p.is_file() else None
    return out


# --------------------------------------------------------- cfc_spec / manifest
def _load_module_from_file(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def cross_check_spec(expect_dir):
    """Cross-check frozen constants against export/cfc_spec.py when present."""
    d = Path(str(expect_dir))
    cand = [d.parent / "cfc_spec.py", d / "cfc_spec.py"]
    path = next((p for p in cand if p.is_file()), None)
    result = {"found": path is not None, "path": str(path) if path else None,
              "checks": {}, "ok": True}
    if path is None:
        return result

    def attr(mod, names):
        for n in names:
            if hasattr(mod, n):
                return True, getattr(mod, n)
        return False, None

    try:
        mod = _load_module_from_file(path, "cfc_spec_crosscheck")
    except Exception as exc:
        result["ok"] = False
        result["error"] = repr(exc)
        return result
    if mod is None:
        result["ok"] = False
        result["error"] = "import failed"
        return result

    pairs = [
        ("id", ["CFC_ID", "ID", "CFC_ID_VALUE"], CFC_ID),
        ("version", ["CFC_VERSION", "VERSION"], CFC_VERSION),
        ("csr_base", ["CSR_BASE", "BASE"], CSR_BASE),
        ("delta_t_int", ["DELTA_T_INT", "DELTA_T"], DELTA_T_INT_DEFAULT),
        ("n_steps", ["N_STEPS", "N_STEPS_MAX", "CFC_N_STEPS_MAX"], N_STEPS_MAX),
        ("clock_hz", ["CLOCK_HZ", "CLK_HZ", "APP_CLOCK_HZ"], CLOCK_HZ_EXPECTED),
    ]
    for field, names, frozen in pairs:
        found, val = attr(mod, names)
        if not found:
            result["checks"][field] = {"found": False, "ok": None}
            continue
        try:
            val = _parse_int(val)
        except Exception:
            pass
        ok = (val == frozen)
        result["checks"][field] = {"found": True, "spec": val, "frozen": frozen, "ok": ok}
        if not ok:
            result["ok"] = False

    # register offsets: dict-style (REG/CSR) or REG_<NAME> constants
    reg_checks = {}
    found, regs = attr(mod, ["REG", "CSR", "REGISTERS", "REG_OFFSETS"])
    if found and isinstance(regs, dict):
        lower = {str(k).lower(): k for k in regs}
        for name, off in REG.items():
            key = name if name in regs else lower.get(name.lower())
            if key is None:
                continue
            spec_off = _parse_int(regs[key])
            ok = (spec_off == off)
            reg_checks[name] = {"spec": spec_off, "frozen": off, "ok": ok}
            if not ok:
                result["ok"] = False
    else:
        for name, off in REG.items():
            fnd, val = attr(mod, ["REG_" + name, name])
            if not fnd:
                continue
            spec_off = _parse_int(val)
            ok = (spec_off == off)
            reg_checks[name] = {"spec": spec_off, "frozen": off, "ok": ok}
            if not ok:
                result["ok"] = False
    result["register_checks"] = reg_checks
    result["registers_checked"] = len(reg_checks)

    # numeric format / parameter layout constants, when exposed
    extra = [
        ("n_weights", ["N_WEIGHTS"], 84),
        ("n_biases", ["N_BIASES"], 17),
        ("lut_size", ["LUT_SIZE"], 1025),
        ("n_input_max", ["N_INPUT_MAX"], 64),
        ("n_output_max", ["N_OUTPUT_MAX"], 64),
        ("w1_base", ["W1_BASE"], 0),
        ("w2_base", ["W2_BASE"], 20),
        ("wa_base", ["WA_BASE"], 40),
        ("wb_base", ["WB_BASE"], 60),
        ("wo_base", ["WO_BASE"], 80),
        ("b1_base", ["B1_BASE"], 0),
        ("b2_base", ["B2_BASE"], 4),
        ("ba_base", ["BA_BASE"], 8),
        ("bb_base", ["BB_BASE"], 12),
        ("bo_index", ["BO_INDEX"], 16),
        ("data_scale", ["DATA_SCALE"], 4096),
        ("bias_scale", ["BIAS_SCALE"], 16777216),
    ]
    extra_checks = {}
    for field, names, frozen in extra:
        fnd, val = attr(mod, names)
        if not fnd:
            continue
        try:
            val_i = _parse_int(val)
        except Exception:
            val_i = val
        ok = (val_i == frozen)
        extra_checks[field] = {"spec": val_i, "frozen": frozen, "ok": ok}
        if not ok:
            result["ok"] = False
    result["extra_checks"] = extra_checks
    return result


def load_manifest(expect_dir):
    d = Path(str(expect_dir))
    for cand in (d / "manifest.json", d.parent / "manifest.json"):
        if cand.is_file():
            try:
                with open(str(cand), "r", encoding="utf-8") as fh:
                    return {"path": str(cand.resolve()),
                            "sha256": sha256_file(cand),
                            "data": json.load(fh)}
            except Exception as exc:
                raise CfcBoardError("cannot parse export manifest: %s" % cand,
                                    {"path": str(cand), "error": repr(exc)})
    return None


def verify_manifest(manifest, expect_dir):
    """Check manifest config keys and file hashes (tolerant to layout)."""
    out = {"present": manifest is not None, "path": None, "sha256": None,
           "config": {}, "files": {"total": 0, "verified": 0, "mismatch": [], "missing": []},
           "ok": True, "warnings": []}
    if manifest is None:
        out["warnings"].append("manifest.json not found")
        return out
    out["path"] = manifest["path"]
    out["sha256"] = manifest["sha256"]
    data = manifest["data"]

    # prefer the dedicated hardware/numeric/task subtrees so that unrelated
    # "version" style fields (spec_version, onnx opset, ...) cannot be mistaken
    # for the CFC1 register values
    sections = {}
    for sec_name in ("hardware", "csr", "numeric", "numeric_format", "task",
                     "activation", "activation_lut"):
        sec = data.get(sec_name)
        if isinstance(sec, dict):
            sections[sec_name] = sec

    def cfg_find(keys, preferred):
        for sec_name in preferred:
            if sec_name in sections:
                f, v = _deep_find(sections[sec_name], keys)
                if f:
                    return True, v
        return _deep_find(data, keys)

    cfg_pairs = [
        ("id", ["id", "CFC_ID", "cfc_id"], CFC_ID, ["hardware", "csr"]),
        ("version", ["version", "CFC_VERSION", "cfc_version"], CFC_VERSION,
         ["hardware", "csr"]),
        ("csr_base", ["csr_base", "CSR_BASE", "base", "cfg_csr_base"], CSR_BASE,
         ["hardware", "csr"]),
        ("clock_hz", ["clock_hz", "clk_hz", "app_clock_hz", "CLOCK_HZ"],
         CLOCK_HZ_EXPECTED, ["hardware", "csr"]),
        ("delta_t_int", ["delta_t_int", "delta_t_int_default", "DELTA_T_INT"],
         DELTA_T_INT_DEFAULT, ["hardware", "numeric", "numeric_format"]),
        ("n_steps", ["n_steps_max", "n_steps", "n_steps_default", "N_STEPS",
                     "cfg_n_steps"], N_STEPS_MAX, ["hardware", "task"]),
    ]
    for field, keys, frozen, preferred in cfg_pairs:
        found, val = cfg_find(keys, preferred)
        if not found:
            out["config"][field] = {"found": False, "ok": None}
            continue
        try:
            val_i = _parse_int(val)
        except Exception:
            val_i = None
        ok = (val_i == frozen)
        out["config"][field] = {"found": True, "manifest": val, "frozen": frozen, "ok": ok}
        if not ok:
            out["ok"] = False

    # contract version recorded in the manifest (expected V1.1 after the gate fix)
    cv = None
    raw_cv = data.get("contract_version")
    if isinstance(raw_cv, str) and raw_cv.strip():
        cv = raw_cv.strip()
        if not cv.upper().startswith("V"):
            cv = "V" + cv
    if cv is None:
        try:
            m = re.search(r"CFC_CONTRACT\.md\s*\(?V([0-9]+(?:\.[0-9]+)*)",
                          json.dumps(data, ensure_ascii=False))
        except Exception:
            m = None
        if m:
            cv = "V" + m.group(1)
    if cv:
        out["contract_version"] = cv
        out["contract_version_ok"] = (cv == "V1.1")
        if not out["contract_version_ok"]:
            out["warnings"].append(
                "manifest contract version is %s, frozen contract is V1.1" % cv)

    files = None
    found, val = _deep_find(data, ["files", "hashes", "sha256", "file_hashes"])
    if found and isinstance(val, dict):
        files = val
    if files:
        d = Path(str(expect_dir))
        roots = [d, d.parent, d.parent.parent]
        for rel, hv in files.items():
            rel_s = str(rel).replace("\\", "/")
            if isinstance(hv, dict):
                hv = hv.get("sha256") or hv.get("hash") or hv.get("value")
            if hv is None:
                continue
            target = None
            for root in roots:
                cand = root / rel_s
                if cand.is_file():
                    target = cand
                    break
            if target is None:
                out["files"]["missing"].append(rel_s)
                continue
            out["files"]["total"] += 1
            got = sha256_file(target)
            if got.lower() == str(hv).lower():
                out["files"]["verified"] += 1
            else:
                out["files"]["mismatch"].append({"file": rel_s, "manifest": hv, "actual": got})
        if out["files"]["mismatch"]:
            out["ok"] = False
    else:
        out["warnings"].append("manifest has no file hash map")
    return out


# ------------------------------------------------------------------- backends
def pynq_available():
    try:
        import pynq  # noqa: F401
        return True
    except Exception:
        return False


class PynqMmioBackend(object):
    """Real PYNQ overlay + MMIO backend (kind == "hardware")."""

    kind = "hardware"

    def __init__(self, mmio):
        self.mmio = mmio

    @staticmethod
    def load_overlay(bit_path):
        from pynq import Overlay
        bit = Path(str(bit_path)).resolve()
        hwh = bit.with_suffix(".hwh")
        if not bit.is_file():
            raise CfcBoardError("bitstream not found: %s" % bit, {"bit": str(bit)})
        if not hwh.is_file():
            raise CfcBoardError(
                "hardware handoff (.hwh) with the same stem is required next to the bitstream: %s" % hwh,
                {"bit": str(bit), "hwh": str(hwh)})
        try:
            return Overlay(str(bit))
        except Exception as exc:
            raise CfcBoardError("PYNQ Overlay() failed for %s: %s" % (bit, exc),
                                {"bit": str(bit), "hwh": str(hwh), "error": repr(exc)})

    @staticmethod
    def open_mmio(csr_base=CSR_BASE, csr_size=CSR_SIZE):
        from pynq import MMIO
        base = int(csr_base)  # Python ints keep the full 64-bit address
        if base <= 0xFFFFFFFF:
            raise CfcBoardError("CSR base truncated to 32 bits", {"csr_base": base})
        return MMIO(base, int(csr_size))

    def read(self, offset):
        return int(self.mmio.read(int(offset))) & 0xFFFFFFFF

    def write(self, offset, value):
        self.mmio.write(int(offset), int(value) & 0xFFFFFFFF)

    def info(self):
        return {"backend": "pynq-mmiio", "not_hardware": False}


# ---------------------------------------------------------------- BoardSession
class BoardSession(object):
    """One bring-up session: overlay/model + expected values + run helpers."""

    def __init__(self, bit=None, hwh=None, xsa=None, expect_dir=None,
                 backend="auto", allow_clock_mismatch=False,
                 allow_fixture=False, verbose=True, sim_options=None):
        self.verbose = bool(verbose)
        self.backend_arg = str(backend or "auto").lower()
        self.allow_clock_mismatch = bool(allow_clock_mismatch)
        self.allow_fixture = bool(allow_fixture)
        self.sim_options = dict(sim_options or {})
        self.paths = discover_paths(bit=bit, hwh=hwh, xsa=xsa, expect_dir=expect_dir)
        self.dev = None
        self.kind = None
        self.report = None
        self.ref = None
        self.simmod = None
        self.manifest = None
        self.manifest_checks = None
        self.sample_cache = {}
        self._gen = None
        self._ready = False

    # -- internal ----------------------------------------------------------
    def log(self, msg):
        if self.verbose:
            print(msg)

    def _load_sim_module(self):
        if self.simmod is not None:
            return self.simmod
        path = Path(__file__).resolve().with_name("cfc_sim_model.py")
        if not path.is_file():
            return None
        try:
            self.simmod = _load_module_from_file(path, "cfc_sim_model_runtime")
            return self.simmod
        except Exception:
            return None

    def _load_reference(self):
        exp = self.paths["expect_dir"]
        if exp is None:
            searched = self.paths.get("expect_dir_searched") or []
            raise CfcBoardError(
                "export/ directory (weights, LUTs, integer golden) was not found. "
                "Copy the delivered export/ directory next to this driver, e.g. "
                "scp -r board/package/export xilinx@<board>:<driver_dir>/export ; "
                "or pass --expect-dir / set CFC_EXPECT_DIR to the directory that "
                "contains weights.hex + int_reference.py. Looked for a valid export "
                "tree in: " + ", ".join((searched or ["<none>"])[:10])
                + (", ..." if len(searched) > 10 else ""),
                {"paths": self.paths,
                 "hint": "copy <delivery>/export next to lnn_cfc_run.py, or set "
                         "CFC_EXPECT_DIR=<delivery>/export"})
        self.expect_dir = Path(exp).resolve()
        mod = self._load_sim_module()
        if mod is None:
            raise CfcBoardError(
                "board/cfc_sim_model.py is required next to the driver (integer reference)",
                {"driver_dir": self.paths["driver_dir"]})
        try:
            self.ref = mod.CfcIntegerReference.from_dir(self.expect_dir)
        except Exception as exc:
            raise CfcBoardError("cannot load integer reference from export/: %s" % exc,
                                {"expect_dir": str(self.expect_dir), "error": repr(exc)})
        self._gen = load_gen_vectors(self.expect_dir)
        self.manifest = load_manifest(self.expect_dir)
        self.manifest_checks = verify_manifest(self.manifest, self.expect_dir)

    def _fixture_mode(self):
        if self.manifest is None:
            return False
        try:
            return bool(self.manifest["data"].get("fixture"))
        except Exception:
            return False

    def _export_int_reference_check(self):
        """Cross-check export/int_reference.py (model workflow) against our reference."""
        candidates = [self.expect_dir / "int_reference.py",
                      self.expect_dir.parent / "int_reference.py"]
        path = next((p for p in candidates if p.is_file()), None)
        out = {"found": path is not None,
               "path": str(path) if path else None,
               "ok": None}
        if path is None:
            return out
        try:
            mod = _load_module_from_file(path, "export_int_reference_crosscheck")
            reuse = getattr(mod, "reload", None)
            if callable(reuse):
                try:
                    reuse(str(self.expect_dir))
                except TypeError:
                    reuse()
            gen = self._gen or {}
            x = gen.get("x_init")
            if x is None:
                out["note"] = "no export/gen/x_input.hex to compare"
                return out
            h0 = gen.get("h_init") or [0, 0, 0, 0]
            got = mod.run_sequence(x, h0, DELTA_T_INT_DEFAULT)
            mine = self.ref.run_sequence(x, h0, DELTA_T_INT_DEFAULT, clear_each_step=False)
            y_mis = 0
            for a, b in zip(got.get("y") or [], mine["y"]):
                if int(a) != int(b):
                    y_mis += 1
            got_h = []
            for step in (got.get("h_steps") or []):
                got_h.extend([int(v) for v in step])
            my_h = []
            for step in mine["h_steps"]:
                my_h.extend([int(v) for v in step])
            h_mis = 0
            for a, b in zip(got_h, my_h):
                if a != b:
                    h_mis += 1
            out.update({"ok": (y_mis == 0 and h_mis == 0),
                        "y_mismatch": y_mis, "h_mismatch": h_mis,
                        "n_steps": len(x)})
        except Exception as exc:
            out["ok"] = False
            out["error"] = repr(exc)
        return out

    def _open_backend(self):
        paths = self.paths
        want = self.backend_arg
        if want == "auto":
            want = "hardware" if (pynq_available() and paths["bit"]) else "simulation-model"
        if want == "hardware":
            if not paths["bit"]:
                raise CfcBoardError("hardware backend requested but no .bit found",
                                    {"paths": paths})
            if pynq_available() is False:
                raise CfcBoardError(
                    "hardware backend requested but the pynq package is not importable",
                    {"python": sys.version})
            if paths["hwh"] is None or Path(paths["bit"]).with_suffix(".hwh") != Path(paths["hwh"]):
                # allow explicit hwh path only if stems match
                if paths["hwh"] is None:
                    raise CfcBoardError(
                        "overlay needs lnn_cfc.bit + lnn_cfc.hwh with the same stem",
                        {"bit": paths["bit"], "hwh": None})
                if Path(paths["hwh"]).stem != Path(paths["bit"]).stem:
                    raise CfcBoardError(
                        "bit and hwh must share the same basename",
                        {"bit": paths["bit"], "hwh": paths["hwh"]})
            PynqMmioBackend.load_overlay(paths["bit"])
            self.dev = PynqMmioBackend(PynqMmioBackend.open_mmio())
            self.kind = "hardware"
        elif want == "simulation-model":
            mod = self._load_sim_module()
            if mod is None:
                raise CfcBoardError("board/cfc_sim_model.py not found next to driver",
                                    {"driver_dir": paths["driver_dir"]})
            fixture_info = None
            if self._fixture_mode():
                fixture_info = {"fixture": True,
                                "note": "期望值来自合成 fixture（cfg cfc_sim_model.make_fixture），不是训练模型"}
            self.dev = mod.SimMmio(expect_dir=self.expect_dir,
                                   clock_hz=CLOCK_HZ_EXPECTED,
                                   fixture_info=fixture_info,
                                   **self.sim_options)
            self.kind = "simulation-model"
        else:
            raise CfcBoardError("unknown backend: %s" % want,
                                {"backend": want, "allowed": ["auto", "hardware", "simulation-model"]})

    def _read(self, name):
        try:
            return int(self.dev.read(REG[name])) & 0xFFFFFFFF
        except Exception as exc:
            raise CfcBoardError("MMIO read failed at %s (0x%03X): %s" % (name, REG[name], exc),
                                self.diagnostics())

    def _write(self, name, value):
        try:
            self.dev.write(REG[name], int(value) & 0xFFFFFFFF)
        except Exception as exc:
            raise CfcBoardError("MMIO write failed at %s (0x%03X): %s" % (name, REG[name], exc),
                                self.diagnostics())

    # -- bring-up ----------------------------------------------------------
    def bringup(self):
        self._load_reference()
        fixture = self._fixture_mode()
        if fixture and not self.allow_fixture:
            raise CfcBoardError(
                "export/ contains a SYNTHETIC fixture (cfc_sim_model.make_fixture), not the "
                "trained model. Pass --allow-fixture only for plumbing tests.",
                {"expect_dir": str(self.expect_dir),
                 "manifest": str(self.manifest["path"]) if self.manifest else None})
        self._open_backend()

        dev_id = self._read("ID") & 0xFFFFFFFF
        version = self._read("VERSION") & 0xFFFFFFFF
        clock_hz = self._read("CLOCK_HZ") & 0xFFFFFFFF
        n_steps_default = self._read("N_STEPS")
        delta_t_default = self._read("DELTA_T") & 0xFFFF
        mode_default = self._read("MODE") & 0x7
        timeout_default = self._read("TIMEOUT_CYCLES")

        errors = []
        id_ok = (dev_id == CFC_ID)
        version_ok = (version == CFC_VERSION)
        clock_ok = (clock_hz == CLOCK_HZ_EXPECTED)
        if not id_ok:
            errors.append("ID mismatch: read 0x%08X expect 0x%08X" % (dev_id, CFC_ID))
        if not version_ok:
            errors.append("VERSION mismatch: read 0x%08X expect 0x%08X" % (version, CFC_VERSION))
        if not clock_ok and not self.allow_clock_mismatch:
            errors.append("CLOCK_HZ mismatch: read %d expect %d" % (clock_hz, CLOCK_HZ_EXPECTED))

        defaults = {
            "n_steps": {"read": n_steps_default, "expect": N_STEPS_MAX,
                        "ok": n_steps_default == N_STEPS_MAX},
            "delta_t": {"read": delta_t_default, "expect": DELTA_T_INT_DEFAULT,
                        "ok": delta_t_default == DELTA_T_INT_DEFAULT},
            "mode": {"read": mode_default, "expect": 1, "ok": mode_default == 1},
            "timeout_cycles": {"read": timeout_default, "expect": TIMEOUT_CYCLES_DEFAULT,
                               "ok": timeout_default == TIMEOUT_CYCLES_DEFAULT},
        }

        spec = cross_check_spec(self.expect_dir)
        if spec.get("found") and not spec.get("ok"):
            errors.append("export/cfc_spec.py disagrees with the frozen contract constants")
        rom_params = check_rom_params(self.expect_dir)
        if rom_params.get("found") and not rom_params.get("ok"):
            errors.append("export/gen/cfc_rom_params.svh (RTL include) disagrees "
                          "with the frozen contract constants")
        if not self.manifest_checks.get("ok", True):
            errors.append("export/manifest.json config/hash check failed")

        # Reference self-check: the contract integer reference re-implemented in
        # cfc_sim_model must reproduce export/gen y_expected + h_expected bit-exactly.
        ref_check = {"ok": None, "y_mismatch": None, "h_mismatch": None,
                     "source": "cfc_sim_model.CfcIntegerReference"}
        gen = self._gen or {}
        if gen.get("x_init") is not None and gen.get("y_expected") is not None:
            h0 = gen.get("h_init") or [0, 0, 0, 0]
            got = self.ref.run_sequence(gen["x_init"], h0, DELTA_T_INT_DEFAULT,
                                        clear_each_step=False)
            y_mis = sum(1 for a, b in zip(got["y"], gen["y_expected"]) if a != b)
            flat = [v for step in got["h_steps"] for v in step]
            h_exp = gen.get("h_expected") or []
            h_mis = sum(1 for a, b in zip(flat, h_exp) if a != b) if h_exp else None
            ref_check.update({"ok": (y_mis == 0 and (h_mis in (0, None))),
                              "y_mismatch": y_mis, "h_mismatch": h_mis,
                              "n_steps": len(gen["x_init"])})
            if not ref_check["ok"]:
                errors.append("integer reference disagrees with export/gen golden vectors "
                              "(stale export? contract V1.1 requires regenerated golden)")

        # Cross-check the model workflow's export/int_reference.py implementation
        exp_ref_check = self._export_int_reference_check()
        if exp_ref_check.get("found") and exp_ref_check.get("ok") is False:
            errors.append("export/int_reference.py disagrees with the board integer reference "
                          "(contract V1.1 semantics)")

        hashes = {}
        for key in ("bit", "hwh", "xsa"):
            p = self.paths.get(key)
            hashes[key + "_sha256"] = sha256_file(p) if p else None

        overlay_info = {
            "bit": self.paths.get("bit"),
            "hwh": self.paths.get("hwh"),
            "xsa": self.paths.get("xsa"),
            "bit_sha256": hashes.get("bit_sha256"),
            "hwh_sha256": hashes.get("hwh_sha256"),
            "xsa_sha256": hashes.get("xsa_sha256"),
            "hwh_same_stem": (self.paths.get("bit") is not None and self.paths.get("hwh") is not None
                              and Path(self.paths["bit"]).stem == Path(self.paths["hwh"]).stem),
            "overlay_loaded": self.kind == "hardware",
        }
        dev_info = {
            "backend": self.kind,
            "csr_base": "0x%X" % CSR_BASE,
            "csr_base_is_64bit": CSR_BASE > 0xFFFFFFFF,
            "csr_size": CSR_SIZE,
            "id": "0x%08X" % dev_id,
            "id_ok": id_ok,
            "version": "0x%08X" % version,
            "version_ok": version_ok,
            "clock_hz": clock_hz,
            "clock_hz_ok": clock_ok,
            "n_steps_default": n_steps_default,
            "delta_t_default": delta_t_default,
            "mode_default": mode_default,
            "timeout_cycles_default": timeout_default,
            "defaults": defaults,
            "sim_model": self.dev.info() if hasattr(self.dev, "info") else None,
        }
        report = {
            "schema": "lnn-cfc-board-bringup-1",
            "generated_utc": now_utc_iso(),
            "source": self.kind,
            "is_hardware_measurement": self.kind == "hardware",
            "disclaimer": None if self.kind == "hardware" else DISCLAIMER,
            "overlay": overlay_info,
            "device": dev_info,
            "expect_dir": str(self.expect_dir),
            "fixture": fixture,
            "manifest": self.manifest_checks,
            "spec": spec,
            "rtl_rom_params": rom_params,
            "reference_selfcheck": ref_check,
            "export_int_reference": exp_ref_check,
            "errors": errors,
            "warnings": list(self.manifest_checks.get("warnings") or []),
            "ok": len(errors) == 0,
        }
        if not clock_ok and self.allow_clock_mismatch:
            report["warnings"].append("CLOCK_HZ mismatch tolerated by --allow-clock-mismatch")
        if defaults["n_steps"]["ok"] is False:
            report["warnings"].append("N_STEPS reset value is not 64")
        if defaults["delta_t"]["ok"] is False:
            report["warnings"].append("DELTA_T reset value is not 655")
        if defaults["mode"]["ok"] is False:
            report["warnings"].append("MODE reset value is not 1")
        if defaults["timeout_cycles"]["ok"] is False:
            report["warnings"].append("TIMEOUT_CYCLES reset value is not 1000000")
        self.report = report
        if errors:
            raise CfcBoardError("bring-up validation failed: " + "; ".join(errors),
                                self.diagnostics(report))
        self._ready = True
        return report

    def diagnostics(self, extra=None):
        diag = {
            "source": self.kind,
            "paths": dict(self.paths),
            "expect_dir": getattr(self, "expect_dir", None),
        }
        if self.report:
            diag["device"] = self.report.get("device")
            diag["overlay"] = self.report.get("overlay")
        if self.dev is not None and hasattr(self.dev, "snapshot"):
            try:
                diag["sim_state"] = self.dev.snapshot()
            except Exception:
                pass
        if self.dev is not None and self.kind == "hardware":
            try:
                diag["registers"] = {name: ("0x%08X" % (self.dev.read(off) & 0xFFFFFFFF))
                                     for name, off in REG.items()}
            except Exception:
                pass
        if extra:
            diag.update(extra)
        return diag

    # -- samples -----------------------------------------------------------
    def _find_sample_file(self, sample_id):
        d = self.expect_dir
        candidates = []
        for base in (d / "gen", d):
            if base.is_dir():
                candidates.extend(sorted(base.glob("sample_*.json")))
        if sample_id:
            want = str(sample_id)
            want_stem = want if want.startswith("sample_") else "sample_" + want
            for p in candidates:
                if p.stem == want or p.stem == want_stem:
                    return p
            for p in candidates:
                if want in p.stem or want_stem in p.stem:
                    return p
            # match the "id" field inside the json
            for p in candidates:
                try:
                    with open(str(p), "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                    fid = doc.get("id")
                    if fid is not None and str(fid) == want:
                        return p
                except Exception:
                    pass
            return None
        # default sample: prefer the one consistent with gen y_expected
        if candidates:
            gen = self._gen or {}
            y_exp = gen.get("y_expected")
            for p in candidates:
                try:
                    with open(str(p), "r", encoding="utf-8") as fh:
                        doc = json.load(fh)
                    y = None
                    for key in ("int_y", "y_int", "output_int"):
                        if isinstance(doc.get(key), list):
                            y = [int(v) for v in doc[key]]
                            break
                    if y is not None and y_exp is not None and y[:len(y_exp)] == y_exp[:len(y)]:
                        return p
                except Exception:
                    pass
            return candidates[0]
        return None

    @staticmethod
    def _unwrap_sample(doc):
        if not isinstance(doc, dict):
            return {}
        for key in ("int_x", "x_int", "input_int", "x"):
            if isinstance(doc.get(key), list):
                return doc
        for key, val in doc.items():
            if isinstance(val, dict):
                for k2 in ("int_x", "x_int", "input_int", "x"):
                    if isinstance(val.get(k2), list):
                        return val
        return doc

    def load_sample(self, sample_id=None):
        """Load one fixed test sample + golden vectors from export/.

        Understands the model-engineer schema (export/gen/sample_<id>.json:
        sample_id, phase_rad/deg, seed, delta_t_float/int, n_steps, delay_steps,
        eval_skip_steps, h_init_int, x_float/y_float, x_int/y_int,
        h_int_steps (flat 256), y_target_float/int, mse_* metrics) and tolerates
        older/simpler key names.
        """
        if not self._ready:
            raise CfcBoardError("bringup() must run before load_sample()")
        if sample_id is None and self.sample_cache:
            return next(iter(self.sample_cache.values()))
        path = self._find_sample_file(sample_id)
        gen = self._gen or {}
        fixture = self._fixture_mode()

        def pick(doc, keys):
            for k in keys:
                if doc.get(k) is not None:
                    return doc[k]
            return None

        doc = {}
        source_files = []
        if path is not None:
            with open(str(path), "r", encoding="utf-8") as fh:
                doc = self._unwrap_sample(json.load(fh))
            source_files.append(str(path))

        int_x = _to_int_list(pick(doc, ["int_x", "x_int", "input_int", "x_q12"]))
        int_y = _to_int_list(pick(doc, ["int_y", "y_int", "output_int", "y_q12"]))
        float_x = _to_float_list(pick(doc, ["float_x", "x_float", "input_float"]))
        float_y = _to_float_list(pick(doc, ["float_y", "y_float", "output_float"]))
        h_init = _to_int_list(pick(doc, ["h_init", "h_init_int", "h0", "initial_state"]), N_HIDDEN)
        h_steps = _normalize_h_steps(pick(doc, ["h_steps", "h_int_steps", "h_seq",
                                                "h_sequence", "h_history", "h"]), N_HIDDEN)
        phase = pick(doc, ["phase", "phase_rad", "phase_deg"])
        phase_deg = pick(doc, ["phase_deg"])
        seed = doc.get("seed")
        dt_int = pick(doc, ["delta_t_int", "dt_int", "DELTA_T_INT"])
        dt_float = pick(doc, ["dt_float", "delta_t_float", "delta_t", "DELTA_T"])
        warmup_skip = pick(doc, ["warmup_skip", "eval_skip_steps", "skip", "warmup"])
        delay_steps = pick(doc, ["delay_steps", "delay", "delay_samples"])
        delay_steps = int(delay_steps) if delay_steps is not None else 4
        target = _to_float_list(pick(doc, ["target", "y_target", "target_float",
                                           "target_y", "y_target_float"]))
        metrics = {}
        for key in ("mse_task_float", "mse_task_int", "mse_int_vs_float",
                    "max_abs_int_vs_float"):
            if doc.get(key) is not None:
                try:
                    metrics[key] = float(doc[key])
                except Exception:
                    pass

        consistency = {}
        if int_x is None:
            int_x = gen.get("x_init")
            consistency["int_x_from"] = "export/gen/x_input.hex"
        if int_x is None:
            raise CfcBoardError(
                "no sample input available (export/gen/x_input.hex or sample_*.json)",
                {"expect_dir": str(self.expect_dir)})
        n = len(int_x)

        golden_y = int_y
        if golden_y is None:
            golden_y = gen.get("y_expected")
            consistency["int_y_from"] = "export/gen/y_expected.hex"
        if golden_y is None:
            raise CfcBoardError(
                "no golden output available (sample int_y or export/gen/y_expected.hex)")
        if int_y is not None and gen.get("y_expected") is not None:
            m = sum(1 for a, b in zip(int_y, gen["y_expected"]) if a != b)
            consistency["sample_int_y_vs_gen"] = {"mismatch": m, "ok": m == 0}
            if m:
                consistency["sample_int_y_vs_gen"]["note"] = (
                    "sample y_int differs from gen/y_expected.hex; sample y_int used as golden")

        h_exp_final = None
        if h_steps:
            h_exp_final = [int(v) for v in h_steps[-1][:N_HIDDEN]]
        if h_exp_final is None and gen.get("h_expected"):
            h_exp_final = gen["h_expected"][-N_HIDDEN:]
            consistency["h_final_from"] = "export/gen/h_expected.hex"
        if h_init is None:
            h_init = gen.get("h_init") or [0, 0, 0, 0]
        if float_y is None:
            float_y = self._float_ref_npz(path.stem if path else None, n)
            if float_y is not None:
                consistency["float_y_from"] = "export/float_ref.npz"
        if target is None and float_x is not None:
            d = min(delay_steps, len(float_x))
            target = [0.0] * d + float_x[:len(float_x) - d]
            consistency["target_from"] = (
                "float_x delayed by %d steps, zero pre-roll (task definition)" % delay_steps)

        sample = {
            "sample_id": (doc.get("sample_id") or doc.get("name")
                          or (path.stem if path else "default")),
            "n_steps": n,
            "x_int": int_x,
            "y_int_expected": golden_y,
            "h_expected_final": h_exp_final,
            "h_steps": h_steps,
            "h_init": h_init,
            "float_x": float_x,
            "float_y": float_y,
            "target_float": target,
            "delay_steps": delay_steps,
            "delta_t_int": int(dt_int) if dt_int is not None else DELTA_T_INT_DEFAULT,
            "delta_t_float": float(dt_float) if dt_float is not None else DELTA_T_FLOAT,
            "phase": phase,
            "phase_deg": phase_deg,
            "seed": seed,
            "warmup_skip": int(warmup_skip) if warmup_skip is not None else WARMUP_SKIP_DEFAULT,
            "export_metrics": metrics,
            "fixture": fixture,
            "source_files": source_files,
            "consistency": consistency,
        }
        self.sample_cache[sample["sample_id"]] = sample
        return sample

    def _float_ref_npz(self, stem, n):
        p = self.expect_dir / "float_ref.npz"
        if np is None or not p.is_file():
            return None
        try:
            with np.load(str(p), allow_pickle=False) as data:
                idx = 0
                if stem:
                    digits = "".join(ch for ch in str(stem) if ch.isdigit())
                    if digits:
                        idx = int(digits)
                for key in sorted(data.keys()):
                    arr = data[key]
                    if (key.lower().find("y") >= 0 and arr.ndim == 2
                            and arr.shape[1] >= n and key.lower().find("h") < 0):
                        row = int(idx) % arr.shape[0]
                        return [float(v) for v in arr[row][:n]]
        except Exception:
            return None
        return None

    # -- expected vectors ---------------------------------------------------
    def expected_for(self, sample, mode, n_steps, offset=0):
        """Golden vectors for a run of n_steps starting at sequence offset.

        Full-length default-mode runs use the exported golden vectors (the
        authoritative acceptance data).  Shorter runs, offset runs, retained
        state and the per-step-clear mode are built with the contract integer
        reference in cfc_sim_model, because the export only carries the
        64-step default-mode vectors.
        """
        n = int(n_steps)
        x = sample["x_int"][offset:offset + n]
        dt = sample["delta_t_int"]
        full_len = sample["n_steps"]
        need_reference = (mode == "clear-each-step" or offset != 0
                          or n != full_len or mode == "retain")
        seq = None
        if need_reference:
            seq = self.ref.run_sequence(sample["x_int"], sample["h_init"], dt,
                                        clear_each_step=(mode == "clear-each-step"))
        if (mode in ("normal", "retain") and offset == 0
                and n <= len(sample["y_int_expected"])):
            y = list(sample["y_int_expected"][:n])
            source = "export golden (gen/y_expected.hex / sample int_y)"
        elif seq is not None:
            y = list(seq["y"][offset:offset + n])
            source = "cfc_sim_model.CfcIntegerReference mode=%s offset=%d" % (mode, offset)
        else:
            y = list(sample["y_int_expected"][offset:offset + n])
            source = "export golden"
        if seq is not None:
            idx = offset + n - 1
            h_final = list(seq["h_steps"][idx]) if idx < len(seq["h_steps"]) else list(seq["h"])
            if source.startswith("export"):
                source += " + reference h_final"
        else:
            h_final = list(sample["h_expected_final"])
        return {"y": y, "h_final": h_final, "source": source, "x": x}

    # -- run ---------------------------------------------------------------
    def run(self, sample=None, mode="normal", n_steps=None, delta_t_int=None,
            timeout_s=None, poll_s=None, expect_override=None, label=None,
            check_writeback=True, input_offset=0):
        if not self._ready:
            raise CfcBoardError("bringup() must run before run()")
        if sample is None:
            sample = self.load_sample(None)
        if mode not in MODE_MAP:
            raise CfcBoardError("unknown mode %r" % (mode,),
                                {"allowed": sorted(MODE_MAP.keys())})
        mode_bits = MODE_MAP[mode]
        n = int(n_steps if n_steps is not None else sample["n_steps"])
        if n < 1 or n > N_STEPS_MAX:
            raise CfcBoardError("n_steps out of 1..64: %d" % n)
        dt = int(delta_t_int if delta_t_int is not None else sample["delta_t_int"])
        timeout_s = float(timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S)
        if poll_s is None:
            # The offline register model advances simulated time on every MMIO
            # read, so polling needs no sleep; real hardware polling is throttled.
            poll_s = 0.0 if self.kind == "simulation-model" else POLL_PERIOD_S
        poll_s = float(poll_s)
        input_offset = int(input_offset)
        x = list(sample["x_int"][input_offset:input_offset + n])
        if len(x) < n:
            raise CfcBoardError("input_offset=%d leaves only %d of %d samples"
                                % (input_offset, len(x), n))
        if expect_override is None:
            expect = self.expected_for(sample, mode, n, input_offset)
        else:
            expect = expect_override
        label = label or ("%s:%s" % (sample["sample_id"], mode))

        result = {
            "label": label,
            "source": self.kind,
            "sample_id": sample["sample_id"],
            "mode": mode,
            "mode_bits": mode_bits,
            "n_steps": n,
            "input_offset": input_offset,
            "delta_t_int": dt,
            "timeout_s": timeout_s,
            "status": None,
            "error_code": None,
            "error_name": None,
            "status_word": None,
            "busy_seen": False,
            "polls": 0,
            "write_time_s": None,
            "wait_time_s": None,
            "readback_time_s": None,
            "e2e_time_s": None,
            "run_cycles": None,
            "step_count": None,
            "hardware_time_s": None,
            "clock_hz": (self.report.get("device") or {}).get("clock_hz"),
            "outputs_int": None,
            "h_final_int": None,
            "input_writeback_ok": None,
            "input_writeback_mismatches": [],
            "compare": None,
            "pass": False,
            "diagnostics": None,
        }
        t_begin = _clock()
        try:
            # 1) configuration (host writes while idle, contract section 4)
            self._write("N_STEPS", n)
            self._write("DELTA_T", dt & 0xFFFF)
            self._write("MODE", mode_bits)
            self._write("TIMEOUT_CYCLES", TIMEOUT_CYCLES_DEFAULT)
            # 2) input window: low 16 bits, signed (upper 16 bits ignored)
            for i in range(n):
                try:
                    self.dev.write(REG["INPUT"] + 4 * i, to_word16(x[i]))
                except Exception as exc:
                    raise CfcBoardError(
                        "MMIO write failed at INPUT[%d] (0x%03X): %s"
                        % (i, REG["INPUT"] + 4 * i, exc), self.diagnostics())
            # 3) clear sticky DONE/ERROR before starting
            self._write("CONTROL", CTRL_CLEAR_STATUS)

            # 4) read back the input window (bit packing / AXI window check)
            bad = []
            if check_writeback:
                for i in range(n):
                    word = int(self.dev.read(REG["INPUT"] + 4 * i)) & 0xFFFFFFFF
                    if to_s16(word & 0xFFFF) != int(x[i]):
                        bad.append({"index": i, "written": int(x[i]),
                                    "low16": to_s16(word & 0xFFFF)})
            result["input_writeback_ok"] = (not bad) if check_writeback else None
            result["input_writeback_mismatches"] = bad[:8]
            result["write_time_s"] = _clock() - t_begin

            # 5) START (write-one-pulse)
            t_wait0 = _clock()
            self._write("CONTROL", CTRL_START)
            deadline = t_wait0 + timeout_s
            while True:
                st = int(self._read("STATUS")) & 0xFFFFFFFF
                result["polls"] += 1
                if st & STATUS_BUSY:
                    result["busy_seen"] = True
                if st & STATUS_ERROR:
                    result["status"] = "error"
                    break
                if st & STATUS_DONE:
                    result["status"] = "done"
                    break
                if _clock() >= deadline:
                    result["status"] = "wall-timeout"
                    break
                remain = deadline - _clock()
                if remain > 0 and poll_s > 0:
                    time.sleep(min(poll_s, remain))
            result["wait_time_s"] = _clock() - t_wait0
            result["status_word"] = int(self._read("STATUS")) & 0xFFFFFFFF

            if result["status"] == "error":
                code = int(self._read("ERROR_CODE")) & 0xF
                result["error_code"] = code
                result["error_name"] = ERROR_CODES.get(code, "UNKNOWN")
                result["step_count"] = int(self._read("STEP_COUNT")) & 0xFFFFFFFF
                result["run_cycles"] = int(self._read("RUN_CYCLES")) & 0xFFFFFFFF
                result["readback_time_s"] = 0.0
                result["e2e_time_s"] = _clock() - t_begin
                result["diagnostics"] = self.diagnostics()
                # clear the sticky status so a fixed configuration can retry
                try:
                    self._write("CONTROL", CTRL_CLEAR_STATUS)
                except Exception:
                    pass
                return result
            if result["status"] == "wall-timeout":
                result["step_count"] = int(self._read("STEP_COUNT")) & 0xFFFFFFFF
                result["run_cycles"] = int(self._read("RUN_CYCLES")) & 0xFFFFFFFF
                result["e2e_time_s"] = _clock() - t_begin
                result["diagnostics"] = self.diagnostics(
                    {"wall_timeout_s": timeout_s, "polls": result["polls"]})
                return result

            # 6) read back results
            t_rb = _clock()
            self._write("CONTROL", 0)  # W1P: no side effects, keeps the bus quiet
            outputs = [to_s16(int(self.dev.read(REG["OUTPUT"] + 4 * i)) & 0xFFFF) for i in range(n)]
            h_final = [to_s16(int(self.dev.read(REG["H_FINAL"] + 4 * i)) & 0xFFFF)
                       for i in range(N_HIDDEN)]
            step_count = int(self._read("STEP_COUNT")) & 0xFFFFFFFF
            run_cycles = int(self._read("RUN_CYCLES")) & 0xFFFFFFFF
            final_status = int(self._read("STATUS")) & 0xFFFFFFFF
            final_error = int(self._read("ERROR_CODE")) & 0xF
            result["readback_time_s"] = _clock() - t_rb
            result["e2e_time_s"] = _clock() - t_begin
            result["outputs_int"] = outputs
            result["h_final_int"] = h_final
            result["step_count"] = step_count
            result["run_cycles"] = run_cycles
            result["status_word"] = final_status
            result["error_code"] = final_error
            result["error_name"] = ERROR_CODES.get(final_error, "UNKNOWN")
            clock = (self.report.get("device") or {}).get("clock_hz") or 0
            result["hardware_time_s"] = (run_cycles / float(clock)) if clock > 0 else None
            result["compare"] = self._compare(result, expect, n)
            result["pass"] = bool(
                result["status"] == "done" and final_error == 0
                and result["compare"]["outputs_bit_exact"]
                and result["compare"]["final_state_match"]
                and result["compare"]["step_count_ok"])
            return result
        except CfcBoardError:
            raise
        except Exception as exc:
            raise CfcBoardError("run failed: %s" % exc,
                                self.diagnostics({"traceback": traceback.format_exc()}))

    def _compare(self, result, expect, n):
        got = list(result["outputs_int"])
        exp = list(expect["y"])[:n]
        mism = [i for i in range(n) if got[i] != exp[i]]
        h_got = list(result["h_final_int"])
        h_exp = list(expect["h_final"])
        h_ok = h_got == h_exp
        step_ok = result["step_count"] == n
        sample = self.sample_cache.get(result["sample_id"]) or {}
        warm = int(sample.get("warmup_skip", WARMUP_SKIP_DEFAULT))
        off = int(result.get("input_offset") or 0)
        real = [v / float(Q_SCALE) for v in got]
        target = sample.get("target_float")
        task_pairs = []
        if target:
            for i in range(min(warm, n), n):
                idx = off + i
                if idx < len(target) and target[idx] is not None:
                    task_pairs.append((real[i], float(target[idx])))
        task_mse = (sum((a - b) ** 2 for a, b in task_pairs) / len(task_pairs)) if task_pairs else None
        fixed_int_mse = sum((got[i] - exp[i]) ** 2 for i in range(n)) / float(n)
        float_ref = sample.get("float_y")
        float_ref_mse = None
        if float_ref and target:
            pairs = []
            for i in range(min(warm, n), n):
                idx = off + i
                if idx < len(float_ref) and idx < len(target) and float_ref[idx] is not None and target[idx] is not None:
                    pairs.append((float(float_ref[idx]), float(target[idx])))
            if pairs:
                float_ref_mse = sum((a - b) ** 2 for a, b in pairs) / len(pairs)
        # cross-check against the task MSE recorded by the model workflow in the
        # sample json (only meaningful for the full default-mode sequence)
        export_metrics = sample.get("export_metrics") or {}
        task_mse_export = export_metrics.get("mse_task_int")
        full_run = (off == 0 and n == sample.get("n_steps"))
        task_mse_vs_export = None
        if full_run and task_mse is not None and task_mse_export is not None:
            task_mse_vs_export = abs(float(task_mse) - float(task_mse_export))
        return {
            "output_match_count": n - len(mism),
            "output_total": n,
            "output_mismatch_count": len(mism),
            "outputs_bit_exact": len(mism) == 0,
            "mismatch_indices": mism[:16],
            "outputs_actual": got,
            "outputs_expected": exp,
            "final_state": h_got,
            "final_state_expected": h_exp,
            "final_state_match": h_ok,
            "step_count_ok": step_ok,
            "expected_source": expect.get("source"),
            "task_mse": task_mse,
            "task_mse_pairs": len(task_pairs),
            "warmup_skip": warm,
            "fixed_point_mse_int": fixed_int_mse,
            "fixed_point_mse_real": fixed_int_mse / float(Q_SCALE * Q_SCALE),
            "float_reference_mse": float_ref_mse,
            "task_mse_export": task_mse_export,
            "task_mse_export_abs_diff": task_mse_vs_export,
            "task_mse_export_ok": (task_mse_vs_export is not None
                                   and task_mse_vs_export < 1e-9),
        }

    # -- state tests -------------------------------------------------------
    def run_repeat(self, sample=None, timeout_s=None):
        sample = sample or self.load_sample(None)
        r1 = self.run(sample, mode="normal", timeout_s=timeout_s, label="repeat-run1")
        r2 = self.run(sample, mode="normal", timeout_s=timeout_s, label="repeat-run2")
        ok = (r1["status"] == "done" and r2["status"] == "done"
              and r1["outputs_int"] == r2["outputs_int"]
              and r1["h_final_int"] == r2["h_final_int"])
        return {"pass": bool(ok and r1["pass"] and r2["pass"]),
                "run1": self._compact(r1), "run2": self._compact(r2),
                "outputs_equal": r1["outputs_int"] == r2["outputs_int"],
                "final_state_equal": r1["h_final_int"] == r2["h_final_int"]}

    def run_segments(self, sample=None, first=32, timeout_s=None):
        sample = sample or self.load_sample(None)
        n = sample["n_steps"]
        first = int(first)
        if first <= 0 or first >= n:
            raise CfcBoardError("segment split out of range: first=%d n=%d" % (first, n))
        second = n - first
        exp1 = self.expected_for(sample, "normal", first, 0)
        exp2 = self.expected_for(sample, "normal", second, first)
        s1 = self.run(sample, mode="normal", n_steps=first, expect_override=exp1,
                      timeout_s=timeout_s, label="segment1-clear")
        s2 = self.run(sample, mode="retain", n_steps=second, expect_override=exp2,
                      input_offset=first, timeout_s=timeout_s, label="segment2-retain")
        combined = (s1["outputs_int"] or []) + (s2["outputs_int"] or [])
        full = self.run(sample, mode="normal", n_steps=n, timeout_s=timeout_s,
                        label="segment-full")
        full_out = full["outputs_int"] or []
        ok = (s1["status"] == "done" and s2["status"] == "done" and full["status"] == "done"
              and combined == full_out and s2["h_final_int"] == full["h_final_int"]
              and s1["pass"] and s2["pass"] and full["pass"])
        return {"pass": bool(ok), "first": first, "second": second,
                "combined_equals_full": combined == full_out,
                "final_state_equals_full": s2["h_final_int"] == full["h_final_int"],
                "segment1": self._compact(s1), "segment2": self._compact(s2),
                "full": self._compact(full)}

    def run_clear_each_step(self, sample=None, timeout_s=None):
        sample = sample or self.load_sample(None)
        exp = self.expected_for(sample, "clear-each-step", sample["n_steps"], 0)
        r = self.run(sample, mode="clear-each-step", expect_override=exp,
                     timeout_s=timeout_s, label="clear-each-step")
        return {"pass": bool(r["pass"] and r["compare"]["outputs_bit_exact"]
                             and r["compare"]["final_state_match"]),
                "result": self._compact(r),
                "task_mse": r["compare"]["task_mse"],
                "expected_source": exp["source"]}

    def run_checks(self, sample=None, timeout_s=None, include=None):
        sample = sample or self.load_sample(None)
        include = include or ["repeat", "segments", "clear-each-step"]
        out = {"sample_id": sample["sample_id"], "checks": {},
               "source": self.kind, "pass": True}
        def guarded(name, fn):
            try:
                return fn()
            except CfcBoardError as exc:
                out["checks"][name] = {"pass": False, "error": str(exc),
                                       "diagnostics": exc.diagnostics}
                return None
            except Exception as exc:
                out["checks"][name] = {"pass": False, "error": repr(exc)}
                return None

        try:
            if "repeat" in include:
                out["checks"]["repeat"] = guarded(
                    "repeat", lambda: self.run_repeat(sample, timeout_s=timeout_s))
            if "segments" in include:
                out["checks"]["segments"] = guarded(
                    "segments", lambda: self.run_segments(sample, timeout_s=timeout_s))
            if "clear-each-step" in include:
                out["checks"]["clear_each_step"] = guarded(
                    "clear-each-step",
                    lambda: self.run_clear_each_step(sample, timeout_s=timeout_s))
        except Exception as exc:
            out["checks"]["exception"] = {"message": repr(exc)}
        for key, val in out["checks"].items():
            if key == "exception" or (isinstance(val, dict) and "error" in val):
                out["pass"] = False
            elif isinstance(val, dict) and val.get("pass") is False:
                out["pass"] = False
        return out

    @staticmethod
    def _compact(result):
        if result is None:
            return None
        cmp = result.get("compare") or {}
        return {
            "label": result.get("label"),
            "mode": result.get("mode"),
            "n_steps": result.get("n_steps"),
            "status": result.get("status"),
            "error_code": result.get("error_code"),
            "pass": result.get("pass"),
            "run_cycles": result.get("run_cycles"),
            "hardware_time_s": result.get("hardware_time_s"),
            "e2e_time_s": result.get("e2e_time_s"),
            "output_match_count": cmp.get("output_match_count"),
            "output_total": cmp.get("output_total"),
            "final_state_match": cmp.get("final_state_match"),
            "task_mse": cmp.get("task_mse"),
            "fixed_point_mse_int": cmp.get("fixed_point_mse_int"),
            "outputs_int": result.get("outputs_int"),
            "h_final_int": result.get("h_final_int"),
        }

    # -- summary / artefacts ----------------------------------------------
    def summary(self, result):
        rep = self.report or {}
        overlay = rep.get("overlay") or {}
        dev = rep.get("device") or {}
        cmd = result.get("compare") or {}
        notes = []
        if self.kind != "hardware":
            notes.append("未在真实板卡上运行：source=simulation-model，数值来自 Python 寄存器模型，"
                         "不能作为板端实测证据。")
        if rep.get("fixture"):
            notes.append("期望值来自合成 fixture（不是训练模型）。")
        summary = {
            "schema": "lnn-cfc-board-run-1",
            "generated_utc": now_utc_iso(),
            "source": self.kind,
            "is_hardware_measurement": self.kind == "hardware",
            "disclaimer": None if self.kind == "hardware" else DISCLAIMER,
            "sample_id": result.get("sample_id"),
            "mode": result.get("mode"),
            "mode_bits": result.get("mode_bits"),
            "n_steps": result.get("n_steps"),
            "delta_t_int": result.get("delta_t_int"),
            "status": result.get("status"),
            "error_code": result.get("error_code"),
            "error_name": result.get("error_name"),
            "output_match_count": cmd.get("output_match_count"),
            "output_total": cmd.get("output_total"),
            "output_mismatch_count": cmd.get("output_mismatch_count"),
            "outputs_bit_exact": cmd.get("outputs_bit_exact"),
            "mismatch_indices": cmd.get("mismatch_indices"),
            "final_state": result.get("h_final_int"),
            "final_state_expected": cmd.get("final_state_expected"),
            "final_state_match": cmd.get("final_state_match"),
            "step_count": result.get("step_count"),
            "step_count_ok": cmd.get("step_count_ok"),
            "task_mse": cmd.get("task_mse"),
            "task_mse_pairs": cmd.get("task_mse_pairs"),
            "task_mse_warmup_skip": cmd.get("warmup_skip"),
            "fixed_point_mse_int": cmd.get("fixed_point_mse_int"),
            "fixed_point_mse_real": cmd.get("fixed_point_mse_real"),
            "float_reference_mse": cmd.get("float_reference_mse"),
            "task_mse_export": cmd.get("task_mse_export"),
            "task_mse_export_abs_diff": cmd.get("task_mse_export_abs_diff"),
            "task_mse_export_ok": cmd.get("task_mse_export_ok"),
            "expected_source": cmd.get("expected_source"),
            "run_cycles": result.get("run_cycles"),
            "run_cycles_source": "hardware" if self.kind == "hardware" else "simulated-model",
            "clock_hz": result.get("clock_hz"),
            "hardware_time_s": result.get("hardware_time_s"),
            "e2e_time_s": result.get("e2e_time_s"),
            "write_time_s": result.get("write_time_s"),
            "wait_time_s": result.get("wait_time_s"),
            "readback_time_s": result.get("readback_time_s"),
            "polls": result.get("polls"),
            "busy_seen": result.get("busy_seen"),
            "status_word": result.get("status_word"),
            "input_writeback_ok": result.get("input_writeback_ok"),
            "csr_base": "0x%X" % CSR_BASE,
            "csr_base_is_64bit": CSR_BASE > 0xFFFFFFFF,
            "bit_sha256": overlay.get("bit_sha256"),
            "hwh_sha256": overlay.get("hwh_sha256"),
            "xsa_sha256": overlay.get("xsa_sha256"),
            "manifest_sha256": (rep.get("manifest") or {}).get("sha256"),
            "bit": overlay.get("bit"),
            "hwh": overlay.get("hwh"),
            "xsa": overlay.get("xsa"),
            "hwh_same_stem": overlay.get("hwh_same_stem"),
            "device": dev,
            "manifest_checks": rep.get("manifest"),
            "spec_checks": rep.get("spec"),
            "rtl_rom_params": rep.get("rtl_rom_params"),
            "reference_selfcheck": rep.get("reference_selfcheck"),
            "export_int_reference": rep.get("export_int_reference"),
            "expect_dir": rep.get("expect_dir"),
            "fixture": rep.get("fixture"),
            "pass": bool(result.get("pass")),
            "notes": notes,
        }
        return summary

    def save_result(self, result, out_dir=None, name="summary", extra=None):
        if out_dir is None:
            runid = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            tag = "%s_%s_%s" % (result.get("sample_id", "sample"),
                                result.get("mode", "mode"), self.kind)
            out_dir = Path(self.paths["driver_dir"]) / "results" / (runid + "_" + tag)
        out = Path(str(out_dir))
        out.mkdir(parents=True, exist_ok=True)
        summary = self.summary(result)
        if extra:
            summary.update(extra)
        paths = {"dir": str(out.resolve())}
        paths["summary"] = write_json(out / (name + ".json"), summary)
        outputs_doc = {
            "schema": "lnn-cfc-board-outputs-1",
            "source": self.kind,
            "sample_id": result.get("sample_id"),
            "mode": result.get("mode"),
            "n_steps": result.get("n_steps"),
            "outputs_int": result.get("outputs_int"),
            "outputs_real": ([v / float(Q_SCALE) for v in result["outputs_int"]]
                             if result.get("outputs_int") else None),
            "h_final_int": result.get("h_final_int"),
            "run_cycles": result.get("run_cycles"),
            "status": result.get("status"),
        }
        paths["outputs"] = write_json(out / "outputs.json", outputs_doc)
        if np is not None and result.get("outputs_int") is not None:
            npy = out / "outputs.npy"
            np.save(str(npy), np.asarray(result["outputs_int"], dtype=np.int16))
            paths["outputs_npy"] = str(npy)
        if result.get("compare"):
            paths["compare"] = write_json(out / "compare.json", result["compare"])
        return paths

    def save_checks(self, checks, out_dir=None, name="state_checks"):
        if out_dir is None:
            out_dir = Path(self.paths["driver_dir"]) / "results" / "checks"
        out = Path(str(out_dir))
        out.mkdir(parents=True, exist_ok=True)
        return write_json(out / (name + ".json"), checks)


# ------------------------------------------------------------------ CLI report
def _print_bringup(report):
    print("=" * 72)
    print("CFC1 bring-up  source=%s  hardware_measurement=%s"
          % (report["source"], report["is_hardware_measurement"]))
    if report["source"] != "hardware":
        print("WARNING: " + str(report.get("disclaimer")))
    ov = report["overlay"]
    print("  bit   : %s" % (ov.get("bit") or "(not found)"))
    print("  hwh   : %s  same_stem=%s" % (ov.get("hwh") or "(not found)", ov.get("hwh_same_stem")))
    print("  xsa   : %s" % (ov.get("xsa") or "(not found)"))
    print("  bit_sha256 : %s" % ov.get("bit_sha256"))
    print("  hwh_sha256 : %s" % ov.get("hwh_sha256"))
    dev = report["device"]
    print("  CSR base : %s (64-bit=%s)" % (dev["csr_base"], dev["csr_base_is_64bit"]))
    print("  ID       : %s  ok=%s" % (dev["id"], dev["id_ok"]))
    print("  VERSION  : %s  ok=%s" % (dev["version"], dev["version_ok"]))
    print("  CLOCK_HZ : %s  ok=%s" % (dev["clock_hz"], dev["clock_hz_ok"]))
    mf = report["manifest"] or {}
    print("  manifest : %s" % (mf.get("path") or "(not found)"))
    print("  manifest_sha256 : %s" % (mf.get("sha256") or "(n/a)"))
    if mf.get("files"):
        f = mf["files"]
        print("  manifest files: total=%d verified=%d mismatch=%d missing=%d"
              % (f.get("total", 0), f.get("verified", 0),
                 len(f.get("mismatch", [])), len(f.get("missing", []))))
    spec = report["spec"] or {}
    print("  cfc_spec: found=%s ok=%s" % (spec.get("found"), spec.get("ok")))
    rp = report.get("rtl_rom_params") or {}
    print("  rtl rom params: found=%s ok=%s" % (rp.get("found"), rp.get("ok")))
    rc = report["reference_selfcheck"]
    print("  reference selfcheck: ok=%s y_mismatch=%s h_mismatch=%s"
          % (rc.get("ok"), rc.get("y_mismatch"), rc.get("h_mismatch")))
    for w in report.get("warnings") or []:
        print("  WARN: " + w)
    print("=" * 72)


def _print_result(result):
    cmp = result.get("compare") or {}
    print("-" * 72)
    print("run %-28s mode=%-16s n_steps=%s" %
          (result.get("label"), result.get("mode"), result.get("n_steps")))
    print("  status=%s error_code=%s step_count=%s run_cycles=%s"
          % (result.get("status"), result.get("error_code"),
             result.get("step_count"), result.get("run_cycles")))
    if result.get("hardware_time_s") is not None:
        print("  hardware_time = %.6f s (cycles/%s Hz)  e2e = %.6f s"
              % (result["hardware_time_s"], result.get("clock_hz"), result.get("e2e_time_s") or 0))
    if cmp:
        print("  outputs matched %s/%s  bit_exact=%s  final_state_match=%s"
              % (cmp.get("output_match_count"), cmp.get("output_total"),
                 cmp.get("outputs_bit_exact"), cmp.get("final_state_match")))
        if cmp.get("task_mse") is not None:
            print("  task_mse = %.8f (skip first %s steps; pairs=%s)"
                  % (cmp["task_mse"], cmp.get("warmup_skip"), cmp.get("task_mse_pairs")))
        print("  fixed_point_mse_int = %s  (real %s)"
              % (cmp.get("fixed_point_mse_int"), cmp.get("fixed_point_mse_real")))
        if cmp.get("mismatch_indices"):
            print("  first mismatch indices: %s" % (cmp["mismatch_indices"],))
    print("  RESULT: %s" % ("PASS" if result.get("pass") else "FAIL"))


def _print_checks(checks):
    print("-" * 72)
    print("state checks (source=%s)" % checks.get("source"))
    for name, val in (checks.get("checks") or {}).items():
        if name == "exception":
            print("  %-20s EXCEPTION: %s" % (name, val.get("message")))
            continue
        verdict = "PASS" if val.get("pass") else "FAIL"
        extra = ""
        if name == "repeat":
            extra = " outputs_equal=%s state_equal=%s" % (
                val.get("outputs_equal"), val.get("final_state_equal"))
        elif name == "segments":
            extra = " combined_equals_full=%s state_equal=%s first=%s second=%s" % (
                val.get("combined_equals_full"), val.get("final_state_equals_full"),
                val.get("first"), val.get("second"))
        elif name == "clear_each_step":
            extra = " task_mse=%s" % (val.get("task_mse"),)
        print("  %-20s %s%s" % (name, verdict, extra))
    print("  checks RESULT: %s" % ("PASS" if checks.get("pass") else "FAIL"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="PYNQ board driver for the LNN CfC demo (CFC1 AXI-Lite, CSR base 0x0480000000)")
    ap.add_argument("--bit", default=None, help="lnn_cfc.bit (hwh with same stem required on board)")
    ap.add_argument("--hwh", default=None, help="optional explicit hwh path")
    ap.add_argument("--xsa", default=None, help="optional XSA path (recorded in summary)")
    ap.add_argument("--expect-dir", default=None,
                    help="export/ directory (weights, LUTs, golden vectors, manifest)")
    ap.add_argument("--sample", default=None, help="sample id (e.g. sample_s0); default = export default sample")
    ap.add_argument("--mode", default="normal", choices=sorted(MODE_MAP.keys()))
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                    help="independent wall-clock timeout for the DONE/ERROR poll (seconds)")
    ap.add_argument("--out-dir", default=None, help="directory for summary.json/outputs")
    ap.add_argument("--backend", default="auto", choices=["auto", "hardware", "simulation-model"])
    ap.add_argument("--allow-clock-mismatch", action="store_true")
    ap.add_argument("--allow-fixture", action="store_true",
                    help="allow the synthetic fixture export tree (plumbing tests only)")
    ap.add_argument("--checks", action="store_true", help="also run repeat / 32+32 / per-step-clear checks")
    ap.add_argument("--list-samples", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:
            pass

    if args.list_samples:
        paths = discover_paths(bit=args.bit, xsa=args.xsa, expect_dir=args.expect_dir)
        if not paths["expect_dir"]:
            print("export/ not found")
            return 2
        print("export dir:", paths["expect_dir"])
        for name in discover_samples(paths["expect_dir"]):
            print("  sample:", name)
        return 0

    try:
        session = BoardSession(bit=args.bit, hwh=args.hwh, xsa=args.xsa,
                               expect_dir=args.expect_dir, backend=args.backend,
                               allow_clock_mismatch=args.allow_clock_mismatch,
                               allow_fixture=args.allow_fixture,
                               verbose=not args.quiet)
        report = session.bringup()
        _print_bringup(report)
        sample = session.load_sample(args.sample)
        print("sample=%s n_steps=%s delta_t_int=%s warmup_skip=%s fixture=%s"
              % (sample["sample_id"], sample["n_steps"], sample["delta_t_int"],
                 sample["warmup_skip"], sample["fixture"]))
        result = session.run(sample, mode=args.mode, n_steps=args.n_steps,
                             timeout_s=args.timeout)
        _print_result(result)
        paths = session.save_result(result, out_dir=args.out_dir)
        print("wrote:", paths["summary"])
        ok = bool(result["pass"])
        if args.checks:
            checks = session.run_checks(sample, timeout_s=args.timeout)
            _print_checks(checks)
            cpath = session.save_checks(checks, out_dir=args.out_dir)
            print("wrote:", cpath)
            ok = ok and bool(checks.get("pass"))
        return 0 if ok else 1
    except CfcBoardError as exc:
        print("BOARD_ERROR: %s" % exc, file=sys.stderr)
        if exc.diagnostics:
            print(json.dumps(exc.diagnostics, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print("UNEXPECTED_ERROR: %s" % exc, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
