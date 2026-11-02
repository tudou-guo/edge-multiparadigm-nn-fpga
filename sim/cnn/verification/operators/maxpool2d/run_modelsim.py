"""Generate synthetic MaxPool vectors and run the standalone ModelSim test."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.onnx_gen.maxpool2d_geometry import encode_payload, geometry
from tools.onnx_gen.source_library import rtl_sources
from verification.operators.maxpool2d.reference import maxpool2d, pack_nhwc8_a16

HERE = Path(__file__).resolve().parent
SEED = 20261029


def make_cases():
    # h,w,c,kernel,strides,pads,dilations, data pattern
    cases = [
        (1, 1, 1, (1, 1), (1, 1), (0, 0, 0, 0), (1, 1), "negative"),
        (2, 3, 7, (1, 2), (1, 1), (1, 1, 0, 0), (1, 2), "negative"),
        (4, 6, 8, (2, 2), (2, 2), (0, 0, 0, 0), (1, 1), "negative"),
        (7, 5, 9, (3, 5), (2, 1), (2, 1, 1, 3), (2, 1), "random"),
        (5, 7, 16, (5, 3), (1, 2), (3, 0, 2, 4), (1, 2), "random"),
        (6, 5, 17, (2, 3), (3, 2), (1, 2, 0, 1), (2, 1), "random"),
        (3, 3, 24, (1, 7), (1, 2), (0, 6, 0, 6), (1, 2), "random"),
        (9, 3, 31, (7, 1), (2, 1), (3, 1, 4, 0), (1, 1), "random"),
        (2, 2, 8, (7, 5), (4, 3), (10, 8, 9, 7), (3, 4), "minimum"),
        (3, 4, 8, (3, 3), (2, 3), (24, 23, 22, 24), (4, 4), "maximum"),
        (8, 9, 24, (3, 2), (8, 7), (1, 0, 0, 2), (4, 3), "random"),
        (2, 3, 512, (1, 1), (1, 2), (0, 0, 0, 1), (1, 1), "random"),
        (5, 5, 8, (3, 3), (1, 1), (0, 0, 0, 0), (2, 2), "sparse"),
        (10, 8, 17, (3, 3), (1, 1), (1, 1, 1, 1), (1, 1), "random"),
        (40, 3, 8, (3, 1), (1, 1), (1, 0, 1, 0), (1, 1), "random"),
        (640, 1, 8, (7, 1), (8, 1), (12, 0, 12, 0), (4, 1), "random"),
        (1, 640, 8, (1, 1), (1, 8), (0, 0, 0, 0), (1, 1), "random"),
    ]
    rng = random.Random(SEED)
    while len(cases) < 54:
        h, w, c = rng.randrange(1, 9), rng.randrange(1, 9), rng.choice([1, 7, 8, 9, 16, 17, 24, 31])
        kernel = (rng.randrange(1, 8), rng.randrange(1, 8))
        stride = (rng.randrange(1, 9), rng.randrange(1, 9))
        dilation = (rng.randrange(1, 5), rng.randrange(1, 5))
        pads = tuple(rng.randrange(0, 9) for _ in range(4))
        try:
            g = geometry((1, c, h, w), {"kernel_shape": kernel, "strides": stride,
                                         "pads": pads, "dilations": dilation})
        except ValueError:
            continue
        if g.hout <= 12 and g.wout <= 12:
            cases.append((h, w, c, kernel, stride, pads, dilation, "random"))
    return cases


def write_words(path, rows):
    data = rows.reshape(-1).tobytes()
    with path.open("w", encoding="ascii") as f:
        for offset in range(0, len(data), 16):
            f.write("%032x\n" % int.from_bytes(data[offset:offset + 16], "little"))


def prepare(out):
    np_rng = np.random.default_rng(SEED)
    cases = make_cases()
    records = []
    descriptions = []
    for index, (h, w, c, kernel, stride, pads, dilation, pattern) in enumerate(cases):
        attrs = {"kernel_shape": kernel, "strides": stride, "pads": pads, "dilations": dilation}
        g = geometry((1, c, h, w), attrs)
        if pattern == "negative":
            x = np_rng.integers(-128, 0, size=(c, h, w), dtype=np.int8)
        elif pattern == "minimum":
            x = np.full((c, h, w), -128, dtype=np.int8)
        elif pattern == "maximum":
            x = np.full((c, h, w), 127, dtype=np.int8)
        elif pattern == "sparse":
            x = np.full((c, h, w), -80, dtype=np.int8)
            x[:, 1, 1] = 127  # skipped by dilation for the first output
        else:
            x = np_rng.integers(-128, 128, size=(c, h, w), dtype=np.int8)
        y = maxpool2d(x, kernel, stride, pads, dilation)
        assert y.shape == (c, g.hout, g.wout)
        write_words(out / ("in_%d.hex" % index), pack_nhwc8_a16(x))
        write_words(out / ("expected_%d.hex" % index), pack_nhwc8_a16(y))
        records.append(encode_payload(g, 0x1000, 0x10000))
        descriptions.append({"case": index, "shape": [h, w, c], "attrs": attrs,
                             "output_shape": [g.hout, g.wout], "pattern": pattern})
    with (out / "cases.txt").open("w", encoding="ascii") as f:
        f.write("%d\n" % len(records))
        for row in records:
            f.write(" ".join("%08x" % word for word in row) + "\n")
    (out / "cases.json").write_text(json.dumps(descriptions, indent=2), encoding="utf-8")
    # Three independent generic records on one physical router. Distinct regions
    # exercise the sequencer's runtime relocation and repeated-core launch.
    rom = [0xffffffff, 3 * 17 + 1]
    input_lines, expected_lines = [], []
    total_out_beats = 0
    for j, index in enumerate((3, 5, 13)):
        payload = list(records[index])
        payload[1] = 0x10001000 + j * 0x2000
        payload[2] = 0x10010000 + j * 0x2000
        rom.extend([0x001d0010] + payload)
        input_lines.append("@%x" % ((payload[1] - 0x10000000) >> 4))
        input_lines.extend((out / ("in_%d.hex" % index)).read_text().splitlines())
        expected_lines.append("@%x" % ((payload[2] - 0x10000000) >> 4))
        expected_lines.extend((out / ("expected_%d.hex" % index)).read_text().splitlines())
        total_out_beats += payload[14] >> 4
    rom.append(0xffffffff)
    (out / "integration_instr.hex").write_text("\n".join("%08x" % v for v in rom) + "\n")
    (out / "integration_input.hex").write_text("\n".join(input_lines) + "\n")
    (out / "integration_expected.hex").write_text("\n".join(expected_lines) + "\n")
    (out / "integration_count.txt").write_text(str(total_out_beats) + "\n")
    legacy_x = np_rng.integers(-128, 31, size=(8, 4, 4), dtype=np.int8)
    legacy_x[0] = np_rng.integers(-128, 0, size=(4, 4), dtype=np.int8)
    legacy_y = np.maximum(maxpool2d(legacy_x, (2, 2), (2, 2)), 0).astype(np.int8)
    write_words(out / "legacy_in_raw.hex", pack_nhwc8_a16(legacy_x))
    write_words(out / "legacy_out_raw.hex", pack_nhwc8_a16(legacy_y))
    legacy_rom = [0xffffffff, 53]
    input_lines, expected_lines = [], []
    total_out_beats = 0
    for j, index in enumerate((3, None, 5)):
        in_addr, out_addr = 0x10001000 + j * 0x2000, 0x10010000 + j * 0x2000
        input_lines.append("@%x" % ((in_addr - 0x10000000) >> 4))
        expected_lines.append("@%x" % ((out_addr - 0x10000000) >> 4))
        if index is None:
            payload = [in_addr, 0, 0, 0, out_addr, 0x00040004, 0x00080008,
                       0x00000404, 0, 0, 32, 16, 128, 0, 0, 0, 32]
            legacy_payload = list(payload)
            legacy_rom.extend([0x00020011] + payload)
            input_lines.extend((out / "legacy_in_raw.hex").read_text().splitlines())
            expected_lines.extend((out / "legacy_out_raw.hex").read_text().splitlines())
            total_out_beats += 2
        else:
            payload = list(records[index])
            payload[1], payload[2] = in_addr, out_addr
            legacy_rom.extend([0x001d0010] + payload)
            input_lines.extend((out / ("in_%d.hex" % index)).read_text().splitlines())
            expected_lines.extend((out / ("expected_%d.hex" % index)).read_text().splitlines())
            total_out_beats += payload[14] >> 4
    legacy_rom.append(0xffffffff)
    (out / "legacy_instr.hex").write_text("\n".join("%08x" % v for v in legacy_rom) + "\n")
    (out / "legacy_input.hex").write_text("\n".join(input_lines) + "\n")
    (out / "legacy_expected.hex").write_text("\n".join(expected_lines) + "\n")
    (out / "legacy_count.txt").write_text(str(total_out_beats) + "\n")
    only = [0xffffffff, 19, 0x00020011] + legacy_payload + [0xffffffff]
    (out / "legacy_only_instr.hex").write_text("\n".join("%08x" % v for v in only) + "\n")
    return len(records)


def run_command(cmd, cwd, log):
    result = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=180)
    with log.open("a", encoding="utf-8") as f:
        f.write("$ " + " ".join(str(v) for v in cmd) + "\n" + result.stdout + "\n")
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError("command exited %d: %s" % (result.returncode, cmd[0]))
    return result.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "sim_out" / "maxpool2d_modelsim"))
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    log.write_text("", encoding="utf-8")
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in (
        "verification.operators.maxpool2d.test_geometry", "verification.operators.maxpool2d.test_pipeline"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1
    count = prepare(out)
    work = out / "work"
    if not work.exists():
        run_command(["vlib", "work"], out, log)
    rtl = ROOT / "rtl" / "ops" / "maxpool2d"
    run_command(["vlog", "-sv", str(rtl / "maxpool2d_row_store.sv"),
                 str(rtl / "maxpool2d_reduce8.sv"), str(rtl / "maxpool2d_core.sv"),
                 str(HERE / "tb_maxpool2d.sv")], out, log)
    output = run_command(["vsim", "-c", "-do",
                          "onbreak {quit -code 1 -f}; onerror {quit -code 1 -f}; run -all; quit -code 0 -f",
                          "work.tb_maxpool2d", "+seed=%d" % SEED], out, log)
    if "TEST_PASS cases=%d" % count not in output or "** Error" in output or "** Fatal" in output:
        raise RuntimeError("ModelSim transcript did not meet the functional PASS gate")
    sources = [str(ROOT / p) for p in rtl_sources() if p.endswith((".v", ".sv"))]
    run_command(["vlog", "-sv", "+incdir+" + str(ROOT / "rtl/common")] + sources +
                [str(HERE / "tb_npu_pool2d.sv")], out, log)
    output = run_command(["vsim", "-c", "-do",
                          "onbreak {quit -code 1 -f}; onerror {quit -code 1 -f}; run -all; quit -code 0 -f",
                          "work.tb_npu_pool2d", "+seed=%d" % SEED], out, log)
    if ("ROUTER_TEST_PASS records=3" not in output or "LEGACY_MIX_TEST_PASS records=3" not in output
            or "LEGACY_ONLY_TEST_PASS records=1" not in output
            or "** Error" in output or "** Fatal" in output):
        raise RuntimeError("ModelSim router transcript did not meet the functional PASS gate")
    from verification.operators.maxpool2d.run_compiled_graphs import main as run_compiled_graphs
    if run_compiled_graphs():
        raise RuntimeError("compiled graph simulations did not pass")
    print("Artifacts:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
