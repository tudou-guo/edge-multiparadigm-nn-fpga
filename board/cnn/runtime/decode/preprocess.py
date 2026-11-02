
# -*- coding: utf-8 -*-
"""Deployment preprocessing for the real YOLOv8n-ReLU VOC20 model (affine-int8-v1).

This module is the single reproducible preprocessing used by the exporter, the
integer reference and the CPU decoder.  It fixes the image bytes and the uint8
tensor that the NPU receives, so every later comparison is on the same input.

Frozen contract (docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md section 3.4)
--------------------------------------------------------------------
1. decode        : JPEG -> RGB uint8 (Pillow, fallback torchvision.io)
2. resize        : bilinear, half-pixel centres, no antialiasing:
                   src = (dst + 0.5) * scale - 0.5, scale = src_size / 640,
                   integer neighbours clamped to the last valid pixel,
                   computed in float64.
3. normalise     : float32 divide by 255
4. quantise      : u8 = clip(rint(x / s_in), 0, 255) with s_in = 0.007870171219110489
                   and round-half-to-even (torch.quantize_per_tensor semantics)
5. colour        : the image is decoded and kept in RGB.  The reference calibration
                   script (quant.py) converted BGR->RGB; the reference C++ host has
                   that conversion commented out.  We standardise on RGB and record
                   the discrepancy instead of silently inheriting either behaviour.

Recorded environment fact (2026-10-26): the E:/miniconda opencv-python-headless
package in this environment cannot be imported (cv2 namespace package has no native
module; the bundled cv2.cp39-win_amd64.pyd fails with "DLL load failed").  Therefore
this module does not use cv2 for decode/resize.  The deployment preprocessing is
defined by the algorithm above, not by a library version, and the reference's
cv2-based pixel values are not claimed to be reproduced bit-for-bit - the recorded
deviation is that this model's preprocessing is our own frozen pipeline, applied
identically to the original quantised model and to the deployment integer model when
they are compared.

Packed layout (NHWC8_A16): channels padded to align8(C), each row padded to
align16(W * phys_c) bytes.  Only the input packing lives here; the head manifest is
owned by the exporter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

PROFILE = "affine-int8-v1"
INPUT_SCALE = 0.007870171219110489
INPUT_ZERO_POINT = 0
INPUT_H = 640
INPUT_W = 640
INPUT_LOGICAL_C = 3
INPUT_PHYS_C = 8            # align8(3)
INPUT_ROW_STRIDE = 5120     # align16(640 * 8)
INPUT_BYTES = INPUT_H * INPUT_ROW_STRIDE

#: fixed software sample of this task (VOC2007 val list head)
DEFAULT_IMAGE = Path("D:/yolov5/3.0/python/yolov8-pytorch-master/VOCdevkit/VOC2007/"
                     "JPEGImages/000032.jpg")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def decode_rgb(image_path):
    """Decode an image to an RGB uint8 HWC array without cv2."""
    try:
        from PIL import Image
        with Image.open(str(image_path)) as img:
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except Exception as pil_exc:  # noqa: BLE001 - fallback is deliberate
        try:
            from torchvision.io import read_image
            t = read_image(str(image_path))  # RGB uint8 CHW
            return np.transpose(t.numpy(), (1, 2, 0))
        except Exception:
            raise RuntimeError("cannot decode %s (PIL: %r)" % (image_path, pil_exc))


def resize_bilinear_half_pixel(src, out_h, out_w):
    """Bilinear resize with half-pixel centres and clamped borders (float64).

    Implements the frozen geometry exactly, independent of any imaging library:
        sx = (dx + 0.5) * (in_w / out_w) - 0.5
    then bilinear interpolation of the two nearest columns/rows with the integer
    indices clamped into range.
    """
    a = np.asarray(src, dtype=np.float64)
    if a.ndim != 3:
        raise ValueError("resize expects HWC input, got %r" % (a.shape,))
    in_h, in_w = a.shape[0], a.shape[1]
    sy = (np.arange(out_h, dtype=np.float64) + 0.5) * (float(in_h) / float(out_h)) - 0.5
    sx = (np.arange(out_w, dtype=np.float64) + 0.5) * (float(in_w) / float(out_w)) - 0.5
    y0 = np.floor(sy).astype(np.int64)
    x0 = np.floor(sx).astype(np.int64)
    wy = (sy - y0)[:, None]
    wx = (sx - x0)[None, :]
    y0c = np.clip(y0, 0, in_h - 1)
    y1c = np.clip(y0 + 1, 0, in_h - 1)
    x0c = np.clip(x0, 0, in_w - 1)
    x1c = np.clip(x0 + 1, 0, in_w - 1)
    top = a[y0c][:, x0c] * (1.0 - wx)[..., None] + a[y0c][:, x1c] * wx[..., None]
    bot = a[y1c][:, x0c] * (1.0 - wx)[..., None] + a[y1c][:, x1c] * wx[..., None]
    return top * (1.0 - wy)[..., None] + bot * wy[..., None]


def quantize_u8(rgb_float32):
    """u8 = clip(rint(x / s_in), 0, 255), round-half-to-even, zero point 0."""
    q = np.rint(np.asarray(rgb_float32, dtype=np.float64) / INPUT_SCALE)
    return np.clip(q, 0, 255).astype(np.uint8)


def preprocess_image(image_path, out_h=INPUT_H, out_w=INPUT_W):
    """Full frozen preprocessing; returns the uint8 HWC tensor and identity fields."""
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError("missing sample image: %s" % path)
    rgb = decode_rgb(path)
    resized = resize_bilinear_half_pixel(rgb.astype(np.float64) / 1.0, out_h, out_w)
    normalised = (resized / 255.0).astype(np.float32)
    q = quantize_u8(normalised)
    return {
        "u8_hwc": q,
        "image_path": str(path),
        "image_bytes": int(path.stat().st_size),
        "image_sha256": sha256_bytes(path.read_bytes()),
        "u8_tensor_sha256": sha256_bytes(q.tobytes()),
        "u8_shape": list(q.shape),
        "input_scale": INPUT_SCALE,
        "input_zero_point": INPUT_ZERO_POINT,
        "preprocess": {
            "decode": "Pillow RGB uint8 (torchvision.io fallback); no cv2 in this environment",
            "resize": "bilinear half-pixel centre, clamped borders, float64; out 640x640",
            "normalize": "float32 / 255",
            "quantize": "clip(rint(x / 0.007870171219110489), 0, 255) ties_even",
            "color": "RGB (reference quant.py BGR->RGB; reference yolo.cpp conversion commented out)",
        },
    }


def to_chw(q_hwc):
    return np.transpose(np.asarray(q_hwc, dtype=np.uint8), (2, 0, 1))


def pack_nhwc8_a16(q_hwc, phys_c=INPUT_PHYS_C):
    """Pack an HWC uint8 tensor into the NPU NHWC8_A16 byte image."""
    q = np.asarray(q_hwc, dtype=np.uint8)
    h, w, c = q.shape
    if phys_c < c:
        raise ValueError("phys_c %d < logical C %d" % (phys_c, c))
    row_stride = ((w * phys_c + 15) // 16) * 16
    out = np.zeros((h, row_stride), dtype=np.uint8)
    padded = np.zeros((h, w, phys_c), dtype=np.uint8)
    # The NPU activation storage is signed INT8 two's complement (qs = qu - 128),
    # so the DDR image must store the recoded byte, not the raw uint8 code.
    padded[:, :, :c] = (np.asarray(q, dtype=np.int16) + 128).astype(np.uint8)
    out = padded.reshape(h, w * phys_c)
    if row_stride != w * phys_c:
        out = np.pad(out, ((0, 0), (0, row_stride - w * phys_c)), mode="constant")
    return {"bytes": out.tobytes(), "shape": [h, row_stride], "phys_c": phys_c,
            "row_stride": row_stride, "logical_c": c, "sha256": sha256_bytes(out.tobytes())}


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Frozen deployment preprocessing of one image")
    ap.add_argument("--image", default=str(DEFAULT_IMAGE))
    ap.add_argument("--out-dir", default=None, help="write u8 npy + packed bin + json here")
    args = ap.parse_args(argv)
    result = preprocess_image(args.image)
    q = result.pop("u8_hwc")
    packed = pack_nhwc8_a16(q)
    result["packed"] = {k: v for k, v in packed.items() if k != "bytes"}
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(str(out / "input_u8_chw.npy"), to_chw(q)[None, ...])
        (out / "input_nhwc8_a16.bin").write_bytes(packed["bytes"])
        result["written"] = {"u8_npy": str(out / "input_u8_chw.npy"),
                             "packed_bin": str(out / "input_nhwc8_a16.bin")}
    result["packed"]["first_16_bytes"] = list(packed["bytes"][:16])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
