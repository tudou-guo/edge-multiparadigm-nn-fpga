import dataclasses
import random
import unittest

import numpy as np

from tools.onnx_gen.graph_ir import CompileError
from tools.onnx_gen.maxpool2d_geometry import (
    capacity, decode_payload, encode_payload, geometry, normalize_attributes,
)
from verification.operators.maxpool2d.reference import maxpool2d, pack_nhwc8_a16


class MaxPoolContractTest(unittest.TestCase):
    def test_onnx_defaults_and_valid(self):
        attrs = normalize_attributes({"kernel_shape": [3, 5]})
        self.assertEqual(attrs["strides"], [1, 1])
        self.assertEqual(attrs["dilations"], [1, 1])
        self.assertEqual(attrs["pads"], [0, 0, 0, 0])
        self.assertEqual(normalize_attributes({"kernel_shape": [1, 1], "auto_pad": "VALID",
                                               "pads": [1, 2, 3, 4]})["pads"], [0, 0, 0, 0])

    def test_independent_reference_and_packing(self):
        x = np.array([[[-2, -3, -4], [-5, 127, -7]]], dtype=np.int8)
        y = maxpool2d(x, (1, 2), (1, 1), (1, 1, 0, 0), (1, 2))
        np.testing.assert_array_equal(y, np.array([[[-128, -128], [-3, -2], [127, -5]]], dtype=np.int8))
        packed = pack_nhwc8_a16(y)
        self.assertEqual(packed.shape, (3, 16))
        self.assertEqual(packed[0].tolist(), [128] + [0] * 7 + [128] + [0] * 7)

    def test_geometry_random_against_independent_shape(self):
        rng = random.Random(20261029)
        for _ in range(200):
            c, h, w = rng.randint(1, 31), rng.randint(1, 9), rng.randint(1, 9)
            kh, kw = rng.randint(1, 7), rng.randint(1, 7)
            sh, sw = rng.randint(1, 8), rng.randint(1, 8)
            dh, dw = rng.randint(1, 4), rng.randint(1, 4)
            pt, pl, pb, pr = (rng.randint(0, 8) for _ in range(4))
            attrs = {"kernel_shape": (kh, kw), "strides": (sh, sw),
                     "dilations": (dh, dw), "pads": (pt, pl, pb, pr)}
            expected_h = (h + pt + pb - dh * (kh - 1) - 1) // sh + 1
            expected_w = (w + pl + pr - dw * (kw - 1) - 1) // sw + 1
            if expected_h <= 0 or expected_w <= 0:
                with self.assertRaises(CompileError):
                    geometry((1, c, h, w), attrs)
                continue
            g = geometry((1, c, h, w), attrs)
            self.assertEqual((g.hout, g.wout), (expected_h, expected_w))
            self.assertEqual(decode_payload(encode_payload(g, 0x1000, 0x10000))[0], g)

    def test_boundaries_and_invalid(self):
        for attrs in ({}, {"kernel_shape": [0, 1]}, {"kernel_shape": [8, 1]},
                      {"kernel_shape": [1, 1], "strides": [1, 9]},
                      {"kernel_shape": [1, 1], "dilations": [5, 1]},
                      {"kernel_shape": [1, 1], "pads": [0, 0, 0, 25]},
                      {"kernel_shape": [1, 1], "ceil_mode": 1},
                      {"kernel_shape": [1, 1], "storage_order": 1},
                      {"kernel_shape": [1, 1], "auto_pad": "SAME_UPPER"}):
            with self.subTest(attrs=attrs), self.assertRaises(CompileError):
                normalize_attributes(attrs)
        g = geometry((1, 8, 16, 16), {"kernel_shape": [7, 5]})
        self.assertEqual((g.hout, g.wout), (10, 12))
        self.assertEqual(capacity([g])["logical_storage_bytes"], 992)
        edge = geometry((1, 104, 1, 630), {"kernel_shape": [1, 1]})
        self.assertEqual(edge.input_row_bytes, 65520)
        with self.assertRaises(CompileError):
            geometry((1, 104, 1, 631), {"kernel_shape": [1, 1]})
        with self.assertRaises(CompileError):
            capacity([g], 991)
        for i, value in ((0, 2), (3, 0), (4, 0), (11, 16), (15, 1)):
            bad = encode_payload(g, 0x1000, 0x10000)
            bad[i] = value
            with self.subTest(word=i), self.assertRaises(CompileError):
                decode_payload(bad)
        for a, b in ((1, 0x10000), (0xfffffff0, 0x10000), (0x1000, 0x1000)):
            with self.subTest(address=(a, b)), self.assertRaises(CompileError):
                encode_payload(g, a, b)
        with self.assertRaises(CompileError):
            encode_payload(dataclasses.replace(g, wout=3), 0x1000, 0x10000)


if __name__ == "__main__":
    unittest.main()
