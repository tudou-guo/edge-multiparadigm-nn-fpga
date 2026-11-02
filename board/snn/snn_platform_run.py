"""PYNQ runtime for the Spikformer NLP streaming platform (ZU15EG / RIGUKE).

Hardware contract (design_1 + top.v in D:/spk_platform):
  * SNN AXI-Lite CSR window : 0x4_8000_0000  (4 KiB, this file's CSR map)
  * PL DDR4 CPU window      : 0x4_0000_0000  (1 GiB, offsets are the same 32-bit
                              offsets programmed into the DMA command registers)
  * 3 read DMA channels stream one sample: tables -> tokens -> weights
  * the write DMA channel streams the 128 x u32 head sums to OUT_BASE

Bring-up sequence:
  1) Overlay(bit) -> read ID/VERSION via MMIO.
  2) write the sample stimulus into PL DDR4 (tables, tokens, weights).
  3) program IN/W/OUT bases + lengths, start, poll done, read sums.
  4) compare bit-exactly with the replay golden vectors.

NOTE: PYNQ only needs platform_top.bit + platform_top.hwh (same stem).
The PL DDR4 CPU window is a PS HPM0 mapping, so plain MMIO works for the
stimulus; no PS-side DMA allocation is required for bring-up.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

CTRL_BASE = 0x4_8000_0000
DDR_BASE = 0x4_0000_0000
DDR_SIZE = 0x0400_0000          # 64 MiB mapped window is enough for bring-up

IN_OFF = 0x0100_0000            # tables, tokens follow TAB_BYTES later
W_OFF = 0x0200_0000
OUT_OFF = 0x0300_0000

TAB_BYTES = 516096              # 64,512 x 64-bit table words
TOK_BYTES = 64                  # 32 x 16-bit tokens (at IN_OFF + TAB_BYTES)
W_BYTES = 402496                # 2 blocks x 25,156 x 64-bit words
OUT_BYTES = 512                 # 128 x u32 sums

CSR = {
    "ID": 0x000, "VERSION": 0x004, "CONTROL": 0x008, "STATUS": 0x00C,
    "ERROR": 0x010, "IN_LO": 0x014, "IN_HI": 0x018,
    "W_LO": 0x01C, "W_HI": 0x020, "OUT_LO": 0x024, "OUT_HI": 0x028,
    "TAB_BYTES": 0x02C, "TOK_BYTES": 0x030, "W_BYTES": 0x034,
    "TIMEOUT": 0x038, "RUN_CYCLES": 0x03C, "TAG": 0x040,
    "SUMS": 0x100,
}
OP_ID = 0x534E4E31
OP_VERSION = 0x00020000
ERR = {0: "NONE", 1: "TIMEOUT", 2: "CONFIG", 3: "OUT_TIMEOUT"}
CTRL_START, CTRL_SOFT_RESET, CTRL_CLR_DONE, CTRL_CLR_ERR, CTRL_ACK = 1, 2, 4, 8, 16


def _read_hex_words(path, width_bytes):
    vals = []
    with open(path, "r", encoding="ascii") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            vals.append(int(line, 16))
    if width_bytes == 8:
        dt = np.uint64
    elif width_bytes == 4:
        dt = np.uint32
    else:
        dt = np.uint16
    return np.asarray(vals, dtype=dt)


class SnnPlatform:
    def __init__(self, ctrl_base=CTRL_BASE, ddr_base=DDR_BASE, ddr_size=DDR_SIZE,
                 clock_hz=200_000_000):
        from pynq import MMIO
        self.c = MMIO(int(ctrl_base), 0x1000)
        self.d = MMIO(int(ddr_base), int(ddr_size))
        self.clock_hz = int(clock_hz)

    # ---------------------------------------------------------------- CSR
    def rd(self, off):
        return int(self.c.read(int(off))) & 0xFFFFFFFF

    def wr(self, off, val):
        self.c.write(int(off), int(val) & 0xFFFFFFFF)

    def id_check(self):
        ident = self.rd(CSR["ID"])
        ver = self.rd(CSR["VERSION"])
        if ident != OP_ID:
            raise RuntimeError("bad SNN ID 0x%08X (expected 0x%08X)" % (ident, OP_ID))
        if ver != OP_VERSION:
            raise RuntimeError("bad SNN version 0x%08X (expected 0x%08X)" % (ver, OP_VERSION))
        return ident, ver

    def program(self, in_off=IN_OFF, w_off=W_OFF, out_off=OUT_OFF,
                tab_bytes=TAB_BYTES, tok_bytes=TOK_BYTES, w_bytes=W_BYTES,
                timeout_cycles=12_000_000, tag=1):
        for name, val in (("IN", in_off), ("W", w_off), ("OUT", out_off)):
            if val >= (1 << 32):
                raise ValueError("%s_OFF 0x%X does not fit a 32-bit PL-DDR4 offset" % (name, val))
            if val & 0xF:
                raise ValueError("%s_OFF 0x%X must be 16-byte aligned" % (name, val))
        self.wr(CSR["IN_LO"], in_off)
        self.wr(CSR["IN_HI"], 0)
        self.wr(CSR["W_LO"], w_off)
        self.wr(CSR["W_HI"], 0)
        self.wr(CSR["OUT_LO"], out_off)
        self.wr(CSR["OUT_HI"], 0)
        self.wr(CSR["TAB_BYTES"], tab_bytes)
        self.wr(CSR["TOK_BYTES"], tok_bytes)
        self.wr(CSR["W_BYTES"], w_bytes)
        self.wr(CSR["TIMEOUT"], timeout_cycles)
        self.wr(CSR["TAG"], tag)

    # ---------------------------------------------------------------- DDR
    def write_ddr(self, off, words):
        arr = np.ascontiguousarray(words)
        if arr.dtype == np.uint64:
            # 64-bit words are stored as little-endian 32-bit pairs: low word first
            arr = arr.view(np.uint32)
        elif arr.dtype == np.uint16:
            # two 16-bit tokens per 32-bit MMIO word
            arr = np.frombuffer(arr.tobytes(), dtype=np.uint32)
        else:
            arr = arr.astype(np.uint32)
        # PYNQ MMIO.write only accepts int or bytes; stream the packed words as bytes
        self.d.write(int(off), arr.tobytes())

    def read_ddr(self, off, nwords):
        raw = self.d.read(int(off), int(nwords) * 4)
        if isinstance(raw, int):
            raw = raw.to_bytes(4, "little")
        return np.frombuffer(raw, dtype=np.uint32)

    def load_sample(self, vecdir, sample=0):
        """Pack the replay golden vectors for one sample into PL DDR4."""
        vecdir = Path(vecdir)
        etab = _read_hex_words(vecdir / "etab.hex", 8)
        ptab = _read_hex_words(vecdir / "ptab.hex", 8)
        toks = _read_hex_words(vecdir / ("toks_s%d.hex" % sample), 2)
        stream = []
        for b in (0, 1):
            stream.append(_read_hex_words(vecdir / ("stream_b%d_s%d.hex" % (b, sample)), 8))
        weights = np.concatenate(stream)
        tab = np.concatenate([etab, ptab])
        assert tab.size * 8 == TAB_BYTES, "table words %d" % tab.size
        assert toks.size * 2 == TOK_BYTES, "token words %d" % toks.size
        assert weights.size * 8 == W_BYTES, "weight words %d" % weights.size
        self.write_ddr(IN_OFF, tab)
        self.write_ddr(IN_OFF + TAB_BYTES, toks)
        self.write_ddr(W_OFF, weights)
        return tab, toks, weights

    # ---------------------------------------------------------------- run
    def start_and_wait(self, timeout_s=60.0, poll_s=0.002):
        # clear sticky done/error so a previous sample cannot satisfy the poll
        self.wr(CSR["CONTROL"], CTRL_CLR_DONE | CTRL_CLR_ERR)
        self.wr(CSR["CONTROL"], 0)
        self.wr(CSR["CONTROL"], CTRL_START)
        t0 = time.time()
        while True:
            st = self.rd(CSR["STATUS"])
            if st & 0x4:
                code = self.rd(CSR["ERROR"]) & 0xF
                self.wr(CSR["CONTROL"], CTRL_CLR_ERR)
                raise RuntimeError("PL error: %s (status=0x%X)" % (ERR.get(code, code), st))
            if st & 0x2:
                break
            if time.time() - t0 > timeout_s:
                self.wr(CSR["CONTROL"], CTRL_SOFT_RESET)
                raise TimeoutError("PL sample timeout after %.1fs" % timeout_s)
            time.sleep(poll_s)
        cycles = self.rd(CSR["RUN_CYCLES"])
        self.wr(CSR["CONTROL"], CTRL_ACK)
        return cycles

    def read_sums(self):
        words = np.asarray([self.rd(CSR["SUMS"] + 4 * i) for i in range(128)],
                          dtype=np.uint32)
        return words

    def run_sample(self, vecdir, sample=0, timeout_s=60.0):
        self.load_sample(vecdir, sample)
        self.program()
        cycles = self.start_and_wait(timeout_s=timeout_s)
        sums = self.read_sums()
        return cycles, sums


def _compare(sums, golden):
    golden = np.asarray(golden, dtype=np.uint32)
    ok = sums.size == golden.size and np.array_equal(sums, golden)
    bad = -1 if ok else int(np.argmax(sums != golden))
    return ok, bad


def main():
    ap = argparse.ArgumentParser(description="SNN platform board runner (PYNQ)")
    ap.add_argument("--bit", default="snn_platform.bit")
    ap.add_argument("--vecdir", required=True, help="replay vector dir (etab/ptab/stream/exp_sums)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--json-out", default="snn_board_result.json")
    args = ap.parse_args()

    from pynq import Overlay
    bit = Path(args.bit).resolve()
    if not bit.is_file():
        raise SystemExit("bitstream not found: %s" % bit)
    hwh = bit.with_suffix(".hwh")
    if not hwh.is_file():
        raise SystemExit("hardware handoff not found next to bitstream: %s" % hwh)
    print("load overlay:", bit)
    Overlay(str(bit))
    dev = SnnPlatform()
    ident, ver = dev.id_check()
    print("SNN id=0x%08X version=0x%08X" % (ident, ver))

    vecdir = Path(args.vecdir).resolve()
    results = []
    for s in range(args.samples):
        t0 = time.time()
        cycles, sums = dev.run_sample(vecdir, s, timeout_s=args.timeout)
        wall = time.time() - t0
        exp = _read_hex_words(vecdir / ("exp_sums_s%d.hex" % s), 4).astype(np.uint32)
        ok, bad = _compare(sums, exp)
        results.append({
            "sample": s, "cycles": int(cycles),
            "board_ms": round(wall * 1e3, 3),
            "model_ms": round(cycles / 200e6 * 1e3, 3),
            "sums_sha256": __import__("hashlib").sha256(sums.tobytes()).hexdigest(),
            "match": bool(ok), "first_mismatch": bad,
        })
        print("sample %d: match=%s cycles=%d board_ms=%.3f model_ms=%.3f"
              % (s, ok, cycles, wall * 1e3, cycles / 2e8 * 1e3))
    doc = {"schema": "spikformer-snn-platform-board-1",
           "bit": str(bit), "vecdir": str(vecdir),
           "id": "0x%08X" % ident, "version": "0x%08X" % ver,
           "samples": results,
           "all_match": all(r["match"] for r in results)}
    Path(args.json_out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print("wrote", args.json_out)
    raise SystemExit(0 if doc["all_match"] else 1)


if __name__ == "__main__":
    main()
