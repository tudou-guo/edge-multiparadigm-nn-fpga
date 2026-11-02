"""Synthetic ONNX/QDQ compiler-to-descriptor and pruning checks."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tools.onnx_gen.affine_frontend import write_input_manifest
from tools.onnx_gen.compile_pipeline import compile_graph
from tools.onnx_gen.graph_ir import CompileError
from tools.onnx_gen.maxpool2d_geometry import decode_payload
from tools.onnx_gen.quant_contract import SCHEMA

ATTRS = {"kernel_shape": [3, 5], "strides": [2, 1],
         "pads": [2, 1, 1, 3], "dilations": [2, 1]}
IN_SHAPE, OUT_SHAPE = [1, 9, 7, 5], [1, 9, 3, 5]


def _save_model(path, qdq=False, mismatch=False, opset=13):
    if qdq:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, IN_SHAPE)
        y = helper.make_tensor_value_info("oq", TensorProto.UINT8, OUT_SHAPE)
        sc = numpy_helper.from_array(np.array(.25, dtype=np.float32), "sc")
        zp = numpy_helper.from_array(np.array(107, dtype=np.uint8), "zp")
        zpo = numpy_helper.from_array(np.array(108 if mismatch else 107, dtype=np.uint8), "zpo")
        nodes = [helper.make_node("QuantizeLinear", ["x", "sc", "zp"], ["q"]),
                 helper.make_node("DequantizeLinear", ["q", "sc", "zp"], ["dq"]),
                 helper.make_node("MaxPool", ["dq"], ["p"], **ATTRS),
                 helper.make_node("QuantizeLinear", ["p", "sc", "zpo"], ["oq"])]
        inits = [sc, zp, zpo]
    else:
        x = helper.make_tensor_value_info("x", TensorProto.INT8, IN_SHAPE)
        y = helper.make_tensor_value_info("y", TensorProto.INT8, OUT_SHAPE)
        nodes = [helper.make_node("MaxPool", ["x"], ["y"], **ATTRS)]
        inits = []
    model = helper.make_model(helper.make_graph(nodes, "pool", [x], [y], inits),
                              opset_imports=[helper.make_opsetid("", opset)])
    onnx.save(model, path)
    return model


class PipelineTest(unittest.TestCase):
    def _check_artifacts(self, out):
        stream = json.loads((out / "generated/instruction_stream.json").read_text())
        manifest = json.loads((out / "generated/rtl_manifest.json").read_text())
        self.assertEqual(len(stream["records"]), 1)
        record = stream["records"][0]
        self.assertEqual(record["opcode"], 0x001d)
        self.assertEqual(record["payload_words"], 16)
        g, _, _ = decode_payload(record["payload"])
        self.assertEqual((g.kh, g.kw, g.sh, g.sw, g.pt, g.pl, g.pb, g.pr, g.dh, g.dw),
                         (3, 5, 2, 1, 2, 1, 1, 3, 2, 1))
        self.assertTrue(manifest["maxpool2d"]["instantiated"])
        self.assertEqual(manifest["maxpool2d"]["capacity"]["cache_rows"], 8)
        self.assertFalse(manifest["physical_instances"]["pool5"]["instantiated"])
        self.assertFalse(manifest["physical_instances"]["maxpool2"]["instantiated"])
        sources = (out / "generated/sources.f").read_text()
        self.assertIn("rtl/ops/maxpool2d/maxpool2d_core.sv", sources)
        self.assertNotIn("rtl/ops/pool5/maxpool5x5_core.v", sources)
        self.assertNotIn("rtl/ops/pool2x2/maxpool2x2_core.v", sources)

    def test_qdq_nonzero_zero_point_and_pruning(self):
        with tempfile.TemporaryDirectory(prefix="pool_qdq_") as tmp:
            root = Path(tmp)
            model = root / "pool.onnx"
            _save_model(model, qdq=True)
            out = root / "out"
            input_u8 = np.random.default_rng(123).integers(0, 256, IN_SHAPE, dtype=np.uint8)
            manifest = compile_graph(model, None, None, out, numeric_profile="affine-int8-v1",
                                     engine_pruning="graph", input_u8=input_u8)
            self.assertEqual(manifest["status"], "pass")
            self._check_artifacts(out)

    def test_standard_onnx_nonzero_zero_point_and_pruning(self):
        with tempfile.TemporaryDirectory(prefix="pool_onnx_") as tmp:
            root = Path(tmp)
            model = root / "pool.onnx"
            _save_model(model)
            qfile = root / "quant.json"
            qfile.write_text(json.dumps({"schema": SCHEMA, "tensors": {
                "x": {"scale": 0.25, "zero_point": -21},
                "y": {"scale": 0.25, "zero_point": -21}}, "convs": {}}))
            inp = root / "input.json"
            write_input_manifest(inp, {"x": np.full(IN_SHAPE, -50, dtype=np.int8)})
            out = root / "out"
            manifest = compile_graph(model, qfile, inp, out, engine_pruning="graph")
            self.assertEqual(manifest["status"], "pass")
            self._check_artifacts(out)

    def test_qdq_mismatched_zero_point_rejected(self):
        with tempfile.TemporaryDirectory(prefix="pool_qdq_mismatch_") as tmp:
            root = Path(tmp)
            model = root / "pool.onnx"
            _save_model(model, qdq=True, mismatch=True)
            with self.assertRaises((CompileError, ValueError)) as caught:
                compile_graph(model, None, None, root / "out", numeric_profile="affine-int8-v1",
                              input_u8=np.zeros(IN_SHAPE, dtype=np.uint8))
            self.assertIn("QUANT", getattr(caught.exception, "code", str(caught.exception)))

    def test_opset11_int8_is_rejected_by_onnx_checker(self):
        from tools.onnx_gen.onnx_frontend import load_onnx
        with tempfile.TemporaryDirectory(prefix="pool_opset_") as tmp:
            path = Path(tmp) / "pool.onnx"
            _save_model(path, opset=11)
            with self.assertRaises(CompileError):
                load_onnx(path)

    def test_spatial_build_accepts_generic_tail_channels(self):
        with tempfile.TemporaryDirectory(prefix="pool_spatial_") as tmp:
            root = Path(tmp)
            shape, result_shape = [1, 31, 7, 5], [1, 31, 3, 5]
            x = helper.make_tensor_value_info("x", TensorProto.INT8, shape)
            y = helper.make_tensor_value_info("y", TensorProto.INT8, result_shape)
            model = helper.make_model(helper.make_graph(
                [helper.make_node("MaxPool", ["x"], ["y"], **ATTRS)], "pool", [x], [y]),
                opset_imports=[helper.make_opsetid("", 13)])
            path = root / "pool.onnx"
            onnx.save(model, path)
            qfile = root / "quant.json"
            qfile.write_text(json.dumps({"schema": SCHEMA, "tensors": {
                "x": {"scale": .25, "zero_point": -21},
                "y": {"scale": .25, "zero_point": -21}}, "convs": {}}))
            inp = root / "input.json"
            write_input_manifest(inp, {"x": np.full(shape, -50, dtype=np.int8)})
            out = root / "out"
            result = compile_graph(path, qfile, inp, out, engine_pruning="graph",
                                   max_feature_width="auto", max_feature_height="auto")
            self.assertEqual(result["status"], "pass")
            rows = json.loads((out / "generated/rtl_manifest.json").read_text())
            self.assertTrue(rows["physical_instances"]["maxpool2d"]["instantiated"])
            self.assertEqual(rows["maxpool2d"]["capacity"]["max_in_row_bytes"], 160)

    def test_pool5_auto_and_forced_generic(self):
        with tempfile.TemporaryDirectory(prefix="pool_auto_") as tmp:
            root = Path(tmp)
            model = root / "pool.onnx"
            x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 8, 4, 5])
            y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 8, 4, 5])
            node = helper.make_node("MaxPool", ["x"], ["y"], kernel_shape=[5, 5],
                                    strides=[1, 1], pads=[2, 2, 2, 2])
            onnx.save(helper.make_model(helper.make_graph([node], "pool5", [x], [y]),
                                        opset_imports=[helper.make_opsetid("", 13)]), model)
            qfile = root / "quant.json"
            qfile.write_text(json.dumps({"schema": SCHEMA, "tensors": {
                "x": {"scale": .25, "zero_point": 0},
                "y": {"scale": .25, "zero_point": 0}}, "convs": {}}))
            inp = root / "input.json"
            write_input_manifest(inp, {"x": np.full([1, 8, 4, 5], -50, dtype=np.int8)})
            for policy, expected in (("auto", 0x1a), ("generic", 0x1d)):
                out = root / policy
                compile_graph(model, qfile, inp, out, engine_pruning="graph", maxpool_backend=policy)
                stream = json.loads((out / "generated/instruction_stream.json").read_text())
                self.assertEqual(stream["records"][0]["opcode"], expected)
                manifest = json.loads((out / "generated/rtl_manifest.json").read_text())
                self.assertEqual(manifest["physical_instances"]["pool5"]["instantiated"], policy == "auto")
                self.assertEqual(manifest["physical_instances"]["maxpool2d"]["instantiated"], policy == "generic")
                sources = (out / "generated/sources.f").read_text()
                self.assertEqual("rtl/ops/maxpool2d/maxpool2d_core.sv" in sources, policy == "generic")


if __name__ == "__main__":
    unittest.main()
