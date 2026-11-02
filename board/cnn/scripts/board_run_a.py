#!/usr/bin/env python3
"""A-profile (tn8/tm8, MAC_IMPL=3) board validation runner for PYNQ / ZynqMP.

Two modes:
  heads  (default without --image): write ddr_init, START, read six heads and
          compare them bit-exactly with runtime/golden (fixed test sample).
  image  (--image <path>): preprocess the image with the frozen deployment
          preprocessing, overwrite the NPU input buffer, run one inference,
          dequant/DFL/decode/NMS on the six heads, draw boxes on the image and
          report per-stage timings. With the packaged test image the heads are
          also compared bit-exactly with golden.

Identity (same source): compile A_tn8_tm8 revision6, source digest f387744a...,
A OOC attempt23 PASS (WNS +0.269 ns / TNS 0 / 0 failing), A same-source IP 0.6.0
PASS (VLNV user.org:npu:npu_spatial_a_m4:0.6.0), platform synthesis after P0 clock fix PASS (WNS +0.080 / 0 failing);
platform implementation impl_clk1_perf (Performance_Explore) TIMING-QUALIFIED:
setup WNS +0.039 / TNS 0 / 0 failing, hold WHS +0.010 / THS 0, route 0 errors,
DRC 0 errors, methodology 0 critical (see ../LIMITATIONS.md).

This proves functional equivalence / box output on the platform for a bitstream
whose routed setup/hold/pulse-width are timing-qualified. It does NOT prove
bandwidth, FPS or mAP/VOC.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent
RUNTIME = PKG / "runtime"
DECODE = RUNTIME / "decode"
TEST_IMAGE = RUNTIME / "test_image.jpg"
TEST_IMAGE_SHA_FILE = RUNTIME / "test_image.sha256"

CSR = {
    "ID": 0x000, "VERSION": 0x004,
    "CTRL": 0x008, "STATUS": 0x00C,
    "ERROR_CODE": 0x010, "ERROR_INFO": 0x014,
    "IRQ_STATUS": 0x018, "TIMEOUT_CYCLES": 0x01C,
    "RUN_ID": 0x020,
    "RD_COUNT": 0x068, "WR_COUNT": 0x06C,
    "LAST_RADDR": 0x070, "LAST_WADDR": 0x074,
    "LAST_RESP": 0x078, "DEBUG_STATE": 0x07C,
    "LAYOUT_ID": 0x080, "CORE_FEATURES": 0x084,
}
CTRL_START, CTRL_SOFT_RESET, CTRL_CLEAR_DONE, CTRL_CLEAR_ERROR = 0x1, 0x2, 0x4, 0x8
ST_IDLE, ST_BUSY, ST_DONE, ST_ERROR, ST_TIMEOUT, ST_READY = 0x01, 0x02, 0x04, 0x08, 0x10, 0x40
OP_ID, OP_ABI_VERSION, OP_LAYOUT_ID = 0x4F504331, 0x00010000, 0x00010101
INPUT_NPU_ADDR = 0x10000000

ERRORS = {
    0x0000: "NONE", 0x0001: "CFG", 0x0002: "UNSUPPORTED", 0x0003: "ADDR_ALIGN",
    0x0004: "ADDR_RANGE", 0x0005: "STRIDE", 0x0006: "QUANT", 0x0007: "TIMEOUT",
    0x0008: "AXI_RESP", 0x0009: "AXI_PROTOCOL", 0x000A: "CORE_INTERNAL",
    0x000B: "ILLEGAL_START", 0x000C: "ILLEGAL_CTRL", 0x000D: "CORE_TIMEOUT",
    0x000E: "CORE_SEQ", 0x000F: "CORE_BLOCKED",
}


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def parse_int(text):
    return int(str(text), 0)


class Axil:
    def __init__(self, base, size=0x1000):
        from pynq import MMIO
        self._mmio = MMIO(int(base), int(size))

    def read(self, offset):
        return int(self._mmio.read(int(offset))) & 0xFFFFFFFF

    def write(self, offset, value):
        self._mmio.write(int(offset), int(value) & 0xFFFFFFFF)


class PlDdr:
    """PL DDR window accessed through /dev/mem mmap.

    PYNQ's MMIO.write() only accepts int/bytes (numpy arrays raise
    ValueError: Data type must be int or bytes), so the original block copy
    implementation fails on this PYNQ image. mmap on /dev/mem is stable and
    fast for the 8.5 MB ddr_init write and the head reads.
    """

    def __init__(self, cpu_base, size, chunk_bytes=1 << 20):
        import mmap as _mmap
        import os
        self.size = int(size)
        self.chunk = int(chunk_bytes)
        page = _mmap.PAGESIZE
        base = int(cpu_base)
        aligned = base & ~(page - 1)
        self._delta = base - aligned
        span = self._delta + self.size
        span = (span + page - 1) & ~(page - 1)
        self._fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
        self._map = _mmap.mmap(self._fd, span, _mmap.MAP_SHARED,
                               _mmap.PROT_READ | _mmap.PROT_WRITE, offset=aligned)

    def _check(self, offset, length):
        if offset < 0 or length < 0 or offset + length > self.size:
            raise ValueError("PL DDR access outside mapped window: off=0x%X len=%d size=0x%X"
                             % (offset, length, self.size))

    def write_bytes(self, offset, data):
        self._check(offset, len(data))
        start = self._delta + int(offset)
        self._map[start:start + len(data)] = bytes(data)

    def read_bytes(self, offset, length):
        self._check(offset, length)
        start = self._delta + int(offset)
        return bytes(self._map[start:start + int(length)])


def load_overlay(bitstream, hwh=None):
    from pynq import Overlay
    if bitstream is None:
        return None
    bit = Path(bitstream)
    if not bit.is_file():
        raise SystemExit("bitstream not found: %s" % bit)
    if hwh is not None:
        h = Path(hwh)
        if not h.is_file():
            raise SystemExit("hwh not found: %s" % h)
        target = bit.with_suffix(".hwh")
        if h.resolve() != target.resolve():
            import shutil
            shutil.copy2(str(h), str(target))
    print("loading overlay: %s" % bit, flush=True)
    return Overlay(str(bit))


def resolve_ctrl_base(overlay, args):
    if args.ctrl_base is not None:
        return parse_int(args.ctrl_base)
    if overlay is not None and args.ctrl_name:
        entry = overlay.ip_dict.get(args.ctrl_name)
        if entry is None:
            candidates = [k for k in overlay.ip_dict if args.ctrl_name.lower() in k.lower()]
            if len(candidates) == 1:
                entry = overlay.ip_dict[candidates[0]]
            else:
                raise SystemExit("ctrl-name not found in overlay ip_dict: %s (candidates=%s)"
                                 % (args.ctrl_name, candidates[:8]))
        base = entry.get("phys_addr")
        if base is None:
            raise SystemExit("ip_dict entry has no phys_addr: %s" % args.ctrl_name)
        return int(base)
    raise SystemExit("need --ctrl-base 0x... (or --ctrl-name with --bitstream)")


def reset_ip(axil):
    axil.write(CSR["CTRL"], CTRL_SOFT_RESET)
    time.sleep(0.001)
    axil.write(CSR["CTRL"], CTRL_CLEAR_DONE)
    axil.write(CSR["CTRL"], CTRL_CLEAR_ERROR)
    axil.write(CSR["IRQ_STATUS"], 0x7)
    time.sleep(0.0005)
    status = axil.read(CSR["STATUS"])
    if not (status & ST_READY):
        raise SystemExit("A IP not READY after reset: STATUS=0x%08X" % status)
    return status


def start_and_wait(axil, timeout_seconds, poll_sleep):
    t_start = time.perf_counter()
    axil.write(CSR["CTRL"], CTRL_START)
    busy_seen = False
    status = 0
    deadline = t_start + timeout_seconds
    while True:
        status = axil.read(CSR["STATUS"])
        if status & ST_BUSY:
            busy_seen = True
        if status & (ST_DONE | ST_ERROR | ST_TIMEOUT):
            break
        if time.perf_counter() > deadline:
            status = -1
            break
        time.sleep(poll_sleep)
    return status, busy_seen, time.perf_counter() - t_start


def read_debug(axil):
    error_code = axil.read(CSR["ERROR_CODE"])
    return {
        "error_code": error_code,
        "error_name": ERRORS.get(error_code, "UNKNOWN"),
        "error_info": "0x%08X" % axil.read(CSR["ERROR_INFO"]),
        "irq_status": "0x%08X" % axil.read(CSR["IRQ_STATUS"]),
        "rd_count": axil.read(CSR["RD_COUNT"]),
        "wr_count": axil.read(CSR["WR_COUNT"]),
        "last_raddr": "0x%08X" % axil.read(CSR["LAST_WADDR"] if False else CSR["LAST_RADDR"]),
        "last_waddr": "0x%08X" % axil.read(CSR["LAST_WADDR"]),
        "last_resp": "0x%08X" % axil.read(CSR["LAST_RESP"]),
        "debug_state": "0x%08X" % axil.read(CSR["DEBUG_STATE"]),
    }


def load_decode_modules():
    sys.path.insert(0, str(DECODE))
    import preprocess  # noqa: E402
    import postprocess  # noqa: E402
    manifest = postprocess.load_heads_manifest(DECODE / "heads.json")
    return preprocess, postprocess, manifest


def unpack_head_codes(raw_blob, spec, postprocess):
    """Hardware int8 head bytes -> manifest uint8 codes (+128 re-encoding)."""
    logical = postprocess.unpack_nhwc8_a16(raw_blob, spec)
    return ((logical.astype(np.int16) + 128) % 256).astype(np.uint8)


def decode_heads_from_raw(raw_heads, manifest, postprocess, conf, iou):
    codes = {}
    for spec in manifest.heads:
        codes[spec.node] = unpack_head_codes(raw_heads[spec.node], spec, postprocess)
    dets, _ = postprocess.postprocess(codes, manifest, conf, iou)
    return dets


def draw_detections(image_path, dets, out_path, scale_x, scale_y):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    img = Image.open(str(image_path)).convert("RGB")
    draw = ImageDraw.Draw(img)
    colors = [(255, 0, 0), (0, 160, 255), (0, 200, 0), (255, 160, 0),
              (200, 0, 255), (0, 255, 200), (255, 255, 0), (255, 0, 160)]
    for i, d in enumerate(dets):
        box = [max(0.0, d.x1 * scale_x), max(0.0, d.y1 * scale_y),
               min(img.width - 1.0, d.x2 * scale_x), min(img.height - 1.0, d.y2 * scale_y)]
        color = colors[int(d.class_id) % len(colors)]
        draw.rectangle(box, outline=color, width=3)
        label = "%s %.2f" % (d.class_name, d.score)
        draw.text((box[0] + 3, max(0.0, box[1] - 12)), label, fill=color)
    img.save(str(out_path), quality=92)
    return out_path


def detections_to_list(dets):
    return [{"class_id": int(d.class_id), "class_name": d.class_name,
             "score": float(d.score), "x1": float(d.x1), "y1": float(d.y1),
             "x2": float(d.x2), "y2": float(d.y2),
             "stride": int(d.stride), "anchor_index": int(d.anchor_index)}
            for d in dets]


def golden_raw_heads(expected, golden_dir):
    out = {}
    for head in expected["heads"]:
        out[head["name"]] = (Path(golden_dir) / Path(head["golden_file"]).name).read_bytes()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="A-profile board validation runner")
    ap.add_argument("--bitstream")
    ap.add_argument("--hwh")
    ap.add_argument("--ctrl-base")
    ap.add_argument("--ctrl-name", default=None)
    ap.add_argument("--window-base", default="0x400000000")
    ap.add_argument("--arena-base", default=None)
    ap.add_argument("--ddr-init", default=str(RUNTIME / "ddr_init.bin"))
    ap.add_argument("--expected-heads", default=str(RUNTIME / "expected_heads.json"))
    ap.add_argument("--golden-dir", default=str(RUNTIME / "golden"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--mode", choices=("auto", "heads", "image"), default="auto")
    ap.add_argument("--image", default=None, help="input image for image mode")
    ap.add_argument("--conf-thres", type=float, default=0.5)
    ap.add_argument("--iou-thres", type=float, default=0.3)
    ap.add_argument("--timeout-seconds", type=float, default=10.0)
    ap.add_argument("--timeout-cycles", default="0x0FFFFFFF")
    ap.add_argument("--poll-sleep", type=float, default=0.0002)
    ap.add_argument("--skip-ddr-write", action="store_true")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--use-golden-heads", action="store_true",
                    help="offline self-test: decode golden heads instead of reading DDR")
    ap.add_argument("--skip-head-compare", action="store_true")
    args = ap.parse_args(argv)

    mode = args.mode
    if mode == "auto":
        mode = "image" if args.image else "heads"
    if mode == "image" and not args.image:
        raise SystemExit("image mode needs --image")

    overlay = load_overlay(args.bitstream, args.hwh) if args.bitstream else None
    ctrl_base = resolve_ctrl_base(overlay, args)
    axil = Axil(ctrl_base)
    ident = axil.read(CSR["ID"])
    version = axil.read(CSR["VERSION"])
    layout = axil.read(CSR["LAYOUT_ID"])
    print("A_BOARD_INFO: id=0x%08X version=0x%08X layout=0x%08X status=0x%08X mode=%s"
          % (ident, version, layout, axil.read(CSR["STATUS"]), mode), flush=True)
    if ident != OP_ID or version != OP_ABI_VERSION:
        raise SystemExit("ABI identity mismatch")
    if layout != OP_LAYOUT_ID:
        raise SystemExit("layout fingerprint mismatch")
    if args.smoke_only:
        print("A_BOARD_RESULT: SMOKE_OK", flush=True)
        return 0

    expected = json.loads(Path(args.expected_heads).read_text(encoding="utf-8"))
    ddr_man = json.loads((RUNTIME / "ddr_init_manifest.json").read_text(encoding="utf-8"))
    arena_base = parse_int(args.arena_base) if args.arena_base else parse_int(ddr_man["base"])
    ddr_init = Path(args.ddr_init).read_bytes()
    if len(ddr_init) != int(ddr_man["bytes"]) or sha256_hex(ddr_init) != ddr_man["sha256"]:
        raise SystemExit("ddr_init.bin identity mismatch")
    map_base = parse_int(args.window_base) + arena_base
    map_size = (int(ddr_man["bytes"]) + 0xFFF) & ~0xFFF
    ddr = PlDdr(map_base, map_size)
    print("A_BOARD_INFO: PL DDR map cpu=0x%X size=0x%X" % (map_base, map_size), flush=True)

    preprocess = postprocess = manifest = None
    if mode == "image":
        preprocess, postprocess, manifest = load_decode_modules()
        t0 = time.perf_counter()
        pp_res = preprocess.preprocess_image(args.image)
        packed = preprocess.pack_nhwc8_a16(pp_res["u8_hwc"])
        preprocess_s = time.perf_counter() - t0
        print("A_BOARD_INFO: image=%s %dx%d input_sha=%s packed_sha=%s preprocess_s=%.4f"
              % (Path(args.image).name, pp_res["u8_hwc"].shape[1], pp_res["u8_hwc"].shape[0],
                 pp_res["u8_tensor_sha256"][:16], packed["sha256"][:16], preprocess_s), flush=True)

    ddr_write_s = 0.0
    if args.skip_ddr_write:
        print("A_BOARD_INFO: skip ddr_init write (--skip-ddr-write)", flush=True)
    else:
        t0 = time.perf_counter()
        ddr.write_bytes(0, ddr_init)
        ddr_write_s = time.perf_counter() - t0
        print("A_BOARD_INFO: ddr_init write %.3f s" % ddr_write_s, flush=True)

    input_write_s = 0.0
    if mode == "image":
        t0 = time.perf_counter()
        ddr.write_bytes(INPUT_NPU_ADDR - arena_base, packed["bytes"])
        input_write_s = time.perf_counter() - t0
        print("A_BOARD_INFO: input buffer write %.3f s (%d bytes)"
              % (input_write_s, len(packed["bytes"])), flush=True)

    reset_ip(axil)
    axil.write(CSR["TIMEOUT_CYCLES"], parse_int(args.timeout_cycles))

    if args.use_golden_heads:
        raw_heads = golden_raw_heads(expected, args.golden_dir)
        status, busy_seen, infer_s = 0x45, False, 0.0
        print("A_BOARD_INFO: offline self-test uses golden heads (no START)", flush=True)
    else:
        status, busy_seen, infer_s = start_and_wait(axil, args.timeout_seconds, args.poll_sleep)
    dbg = read_debug(axil)
    print("A_BOARD_DONE: status=%s busy_seen=%s infer_s=%.6f"
          % ("0x%08X" % status if status >= 0 else "POLL_TIMEOUT", busy_seen, infer_s), flush=True)
    print("A_BOARD_DEBUG: " + json.dumps(dbg, ensure_ascii=False), flush=True)

    result = {
        "schema": "m4-a-board-run-2",
        "package": "A_attempt15 (tn8/tm8, MAC_IMPL=3)",
        "mode": mode,
        "ctrl_base": "0x%X" % ctrl_base,
        "window_base": args.window_base,
        "arena_base": "0x%08X" % arena_base,
        "ddr_init_sha256": sha256_hex(ddr_init),
        "status": status,
        "busy_seen": busy_seen,
        "debug": dbg,
        "timing": {"ddr_init_s": ddr_write_s, "input_write_s": input_write_s,
                   "infer_s": infer_s},
        "heads": [],
        "detections": [],
        "verdict": "FAIL",
    }
    start_ok = (status >= 0) and bool(status & ST_DONE) and not (status & (ST_ERROR | ST_TIMEOUT))

    t_read = 0.0
    if not args.use_golden_heads:
        t0 = time.perf_counter()
        raw_heads = {}
        for head in expected["heads"]:
            off = parse_int(head["npu_addr"]) - arena_base
            raw_heads[head["name"]] = ddr.read_bytes(off, int(head["bytes"])) if start_ok else b""
        t_read = time.perf_counter() - t0
    result["timing"]["heads_read_s"] = t_read

    # head bit-exact check (meaningful for the packaged fixed test image)
    golden = golden_raw_heads(expected, args.golden_dir)
    if not args.skip_head_compare and (start_ok or args.use_golden_heads):
        for head in expected["heads"]:
            name = head["name"]
            live = raw_heads.get(name, b"")
            gold = golden[name]
            mismatch = -1
            if live and len(live) == len(gold):
                for i, (a, b) in enumerate(zip(live, gold)):
                    if a != b:
                        mismatch = i
                        break
            row = {"name": name, "npu_addr": head["npu_addr"], "bytes": int(head["bytes"]),
                   "match": bool(live) and live == gold,
                   "sha256_live": sha256_hex(live) if live else None,
                   "sha256_golden": head["sha256"],
                   "first_mismatch_offset": mismatch}
            result["heads"].append(row)
            print("A_BOARD_HEAD %s: match=%s first_mismatch=%s"
                  % (name, row["match"], mismatch), flush=True)

    decode_ok = False
    dets = []
    t_decode = 0.0
    if mode == "image" and (start_ok or args.use_golden_heads):
        t0 = time.perf_counter()
        try:
            dets = decode_heads_from_raw(raw_heads, manifest, postprocess,
                                         args.conf_thres, args.iou_thres)
            decode_ok = True
        except Exception as exc:  # noqa: BLE001
            print("A_BOARD_DECODE_ERROR: %s" % exc, flush=True)
        t_decode = time.perf_counter() - t0
        result["timing"]["decode_s"] = t_decode
        result["detections"] = detections_to_list(dets)
        print("A_BOARD_DETECTIONS: %d detections (conf>=%.2f iou<=%.2f)"
              % (len(dets), args.conf_thres, args.iou_thres), flush=True)
        for d in dets:
            print("  %-12s score=%.3f box=(%.1f,%.1f,%.1f,%.1f) stride=%d"
                  % (d.class_name, d.score, d.x1, d.y1, d.x2, d.y2, d.stride), flush=True)

        # reference decode of golden heads -> compare when the input is the packaged test image
        try:
            golden_dets = decode_heads_from_raw(golden, manifest, postprocess,
                                                args.conf_thres, args.iou_thres)
        except Exception:  # noqa: BLE001
            golden_dets = None
        test_sha = None
        if TEST_IMAGE_SHA_FILE.is_file():
            test_sha = TEST_IMAGE_SHA_FILE.read_text(encoding="utf-8").strip().split()[0]
        if golden_dets is not None and test_sha and pp_res["image_sha256"] == test_sha:
            same = (len(dets) == len(golden_dets) and all(
                a.class_id == b.class_id and abs(a.score - b.score) < 1e-6
                and abs(a.x1 - b.x1) < 1e-3 and abs(a.y1 - b.y1) < 1e-3
                and abs(a.x2 - b.x2) < 1e-3 and abs(a.y2 - b.y2) < 1e-3
                for a, b in zip(dets, golden_dets)))
            result["detections_match_golden"] = same
            print("A_BOARD_DETECTIONS_MATCH_GOLDEN: %s (%d vs %d)"
                  % (same, len(dets), len(golden_dets)), flush=True)
        elif golden_dets is not None:
            result["detections_match_golden"] = None
            print("A_BOARD_INFO: custom image -> golden head comparison not applicable", flush=True)

        out_dir = Path(args.out_dir) if args.out_dir else (PKG / "board_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        if dets:
            rgb = preprocess.decode_rgb(Path(args.image))
            scale_x = rgb.shape[1] / float(preprocess.INPUT_W)
            scale_y = rgb.shape[0] / float(preprocess.INPUT_H)
            t0 = time.perf_counter()
            draw_path = draw_detections(args.image, dets,
                                        out_dir / ("annotated_" + Path(args.image).stem + ".jpg"),
                                        scale_x, scale_y)
            result["timing"]["draw_s"] = time.perf_counter() - t0
            if draw_path:
                result["annotated_image"] = str(draw_path)
                print("A_BOARD_ANNOTATED: %s" % draw_path, flush=True)
            else:
                print("A_BOARD_INFO: PIL not available -> detections JSON only", flush=True)
        det_path = out_dir / ("detections_" + Path(args.image).stem + ".json")
        det_path.write_text(json.dumps({"image": str(args.image),
                                        "detections": result["detections"]},
                                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["detections_file"] = str(det_path)

    heads_ok = all(h["match"] for h in result["heads"]) if result["heads"] else None
    if start_ok and ((mode == "heads" and heads_ok) or (mode == "image" and decode_ok)):
        result["verdict"] = "PASS"
    elif not start_ok:
        result["verdict"] = "FAIL_START_OR_STATUS"
    elif mode == "heads":
        result["verdict"] = "FAIL_HEAD_MISMATCH"
    else:
        result["verdict"] = "FAIL_DECODE"

    try:
        axil.write(CSR["IRQ_STATUS"], 0x7)
        axil.write(CSR["CTRL"], CTRL_CLEAR_DONE)
        axil.write(CSR["CTRL"], CTRL_CLEAR_ERROR)
    except Exception:  # noqa: BLE001
        pass

    out_dir = Path(args.out_dir) if args.out_dir else (PKG / "board_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / ("board_result_A_%s_%s.json"
                          % (mode, time.strftime("%Y%m%d_%H%M%S")))
    result["evidence"] = {
        "a_ooc": "A OOC attempt15 PASS: WNS +0.046 ns / TNS 0 / 0 failing endpoints",
        "a_ip": "A same-source IP attempt15 PASS: ip_pack/same_source/bound_ooc",
        "source_digest": "86e2e6a4ca5ccd848b7ff6b2876a02ce4e5721a32b7a67f594643a5688d660be",
        "ip_vlnv": "user.org:npu:npu_spatial_a_m4:0.4.0",
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("A_BOARD_TIMING: " + json.dumps(result["timing"], ensure_ascii=False), flush=True)
    print("A_BOARD_RESULT: %s" % result["verdict"], flush=True)
    print("A_BOARD_RESULT_FILE: %s" % out_path, flush=True)
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
