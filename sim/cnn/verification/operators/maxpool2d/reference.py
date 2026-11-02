"""Independent, direct-index INT8 oracle for operator verification."""
import numpy as np


def maxpool2d(x, kernel, strides=(1, 1), pads=(0, 0, 0, 0), dilations=(1, 1)):
    x = np.asarray(x, dtype=np.int8)
    if x.ndim != 3:
        raise ValueError("expected CHW")
    c, h, w = x.shape
    kh, kw = kernel
    sh, sw = strides
    pt, pl, pb, pr = pads
    dh, dw = dilations
    oh = (h + pt + pb - (dh * (kh - 1) + 1)) // sh + 1
    ow = (w + pl + pr - (dw * (kw - 1) + 1)) // sw + 1
    if oh <= 0 or ow <= 0:
        raise ValueError("empty output")
    out = np.full((c, oh, ow), -128, dtype=np.int8)
    for oy in range(oh):
        for ox in range(ow):
            for ky in range(kh):
                for kx in range(kw):
                    iy, ix = oy * sh - pt + ky * dh, ox * sw - pl + kx * dw
                    if 0 <= iy < h and 0 <= ix < w:
                        out[:, oy, ox] = np.maximum(out[:, oy, ox], x[:, iy, ix])
    return out


def pack_nhwc8_a16(x):
    x = np.asarray(x, dtype=np.int8)
    c, h, w = x.shape
    c_alloc = ((c + 7) // 8) * 8
    row_bytes = ((w * c_alloc + 15) // 16) * 16
    result = np.zeros((h, row_bytes), dtype=np.uint8)
    for y in range(h):
        for col in range(w):
            result[y, col * c_alloc:col * c_alloc + c] = x[:, y, col].view(np.uint8)
    return result
