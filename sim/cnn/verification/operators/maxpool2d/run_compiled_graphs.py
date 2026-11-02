"""Compile synthetic QDQ chains and replay generated top-level RTL in ModelSim."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.onnx_gen.compile_pipeline import compile_graph
from tools.onnx_gen.affine_frontend import write_input_manifest
from tools.onnx_gen.quant_contract import SCHEMA as QUANT_SCHEMA
from verification.operators.maxpool2d.test_pipeline import _save_model, IN_SHAPE


def make_model(path, zero_point):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 9, 40, 5])
    y = helper.make_tensor_value_info("out", TensorProto.UINT8, [1, 9, 11, 4])
    scale = numpy_helper.from_array(np.array(.25, dtype=np.float32), "scale")
    zp = numpy_helper.from_array(np.array(zero_point, dtype=np.uint8), "zp")
    attrs = [
        {"kernel_shape": [3, 5], "strides": [2, 1], "pads": [2, 1, 1, 3], "dilations": [2, 1]},
        {"kernel_shape": [5, 5], "strides": [1, 1], "pads": [2, 2, 2, 2]},
        {"kernel_shape": [1, 2], "strides": [2, 1], "pads": [0, 1, 1, 0], "dilations": [1, 2]},
    ]
    nodes = [helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["q0"]),
             helper.make_node("DequantizeLinear", ["q0", "scale", "zp"], ["d0"])]
    for index, parameters in enumerate(attrs):
        q = "out" if index == len(attrs) - 1 else "q%d" % (index + 1)
        nodes.append(helper.make_node("MaxPool", ["d%d" % index], ["p%d" % index], **parameters))
        nodes.append(helper.make_node("QuantizeLinear", ["p%d" % index, "scale", "zp"], [q]))
        if index < len(attrs) - 1:
            nodes.append(helper.make_node("DequantizeLinear", [q, "scale", "zp"], ["d%d" % (index + 1)]))
    graph = helper.make_graph(nodes, "synthetic_maxpool_chain", [x], [y], [scale, zp])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def make_pool5_edge(path, height, width, channels):
    shape = [1, channels, height, width]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("out", TensorProto.UINT8, shape)
    scale = numpy_helper.from_array(np.array(.25, dtype=np.float32), "scale")
    zp = numpy_helper.from_array(np.array(128, dtype=np.uint8), "zp")
    nodes = [helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["q"]),
             helper.make_node("DequantizeLinear", ["q", "scale", "zp"], ["dq"]),
             helper.make_node("MaxPool", ["dq"], ["p"], kernel_shape=[5, 5],
                              strides=[1, 1], pads=[2, 2, 2, 2]),
             helper.make_node("QuantizeLinear", ["p", "scale", "zp"], ["out"])]
    model = helper.make_model(helper.make_graph(nodes, "pool5_edge", [x], [y], [scale, zp]),
                              opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, path)


def main():
    out_parent = ROOT / "sim_out" / "maxpool2d_compiled_graphs"
    out_parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="run_", dir=str(out_parent)))
    rng = np.random.default_rng(20261029)
    input_u8 = rng.integers(0, 256, [1, 9, 40, 5], dtype=np.uint8)
    results = {}
    for name, zp, policy, expected_opcodes in (
            ("nonzero_generic", 107, "auto", [0x1d, 0x1d, 0x1d]),
            ("mixed_auto", 128, "auto", [0x1d, 0x1a, 0x1d]),
            ("forced_generic", 128, "generic", [0x1d, 0x1d, 0x1d])):
        model_path = out / (name + ".onnx")
        make_model(model_path, zp)
        compile_dir, result_dir = out / (name + "_compile"), out / (name + "_result")
        compile_graph(model_path, None, None, compile_dir, numeric_profile="affine-int8-v1",
                      engine_pruning="graph", maxpool_backend=policy, input_u8=input_u8)
        stream = json.loads((compile_dir / "generated/instruction_stream.json").read_text())
        opcodes = [r["opcode"] for r in stream["records"]]
        if opcodes != expected_opcodes:
            raise RuntimeError("unexpected %s opcodes: %r" % (name, opcodes))
        cmd = ["powershell", "-NoProfile", "-File", str(ROOT / "verification/sim/run_p3_graph_shell.ps1"),
               "-CompileDir", str(compile_dir), "-ResultDir", str(result_dir), "-ProjectRoot", str(ROOT),
               "-Simulator", "modelsim", "-SimDefine", "TB_MAC_IMPL_2,TB_LEAN_FUNC", "-Seed", "20261029"]
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=180)
        print(r.stdout, end="")
        if r.returncode:
            raise RuntimeError("%s full-graph simulation failed: %s" % (name, result_dir / "simulator.log"))
        transcript = (result_dir / "simulator.log").read_text(encoding="utf-8-sig")
        if "P3_RESULT: PASS snapshots=4 mismatch=0" not in transcript or "** Error" in transcript or "** Fatal" in transcript:
            raise RuntimeError("%s transcript did not pass all gates" % name)
        results[name] = {"opcodes": opcodes, "compile_dir": str(compile_dir),
                         "result_dir": str(result_dir),
                         "warnings": sum("** Warning:" in line for line in transcript.splitlines())}
    mixed = (Path(results["mixed_auto"]["result_dir"]) / "snapshots/final_maxpool_8.bin").read_bytes()
    generic = (Path(results["forced_generic"]["result_dir"]) / "snapshots/final_maxpool_8.bin").read_bytes()
    if mixed != generic:
        raise RuntimeError("auto and forced-generic final output differ")
    for height, width, channels in ((1, 1, 1), (2, 3, 9), (3, 1, 8),
                                    (3, 4, 8), (3, 4, 9), (3, 4, 31)):
        name = "edge_h%d_w%d_c%d" % (height, width, channels)
        model_path = out / (name + ".onnx")
        make_pool5_edge(model_path, height, width, channels)
        compile_dir, result_dir = out / (name + "_compile"), out / (name + "_result")
        edge_input = rng.integers(0, 256, [1, channels, height, width], dtype=np.uint8)
        compile_graph(model_path, None, None, compile_dir, numeric_profile="affine-int8-v1",
                      engine_pruning="graph", input_u8=edge_input)
        stream = json.loads((compile_dir / "generated/instruction_stream.json").read_text())
        opcode = stream["records"][0]["opcode"]
        expected_opcode = 0x1d if height < 3 or width < 4 else 0x1a
        if opcode != expected_opcode:
            raise RuntimeError("unexpected edge backend for %s: %x" % (name, opcode))
        cmd = ["powershell", "-NoProfile", "-File", str(ROOT / "verification/sim/run_p3_graph_shell.ps1"),
               "-CompileDir", str(compile_dir), "-ResultDir", str(result_dir), "-ProjectRoot", str(ROOT),
               "-Simulator", "modelsim", "-SimDefine", "TB_MAC_IMPL_2,TB_LEAN_FUNC", "-Seed", "20261029"]
        r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, timeout=180)
        print(r.stdout, end="")
        if r.returncode:
            raise RuntimeError("%s full-graph simulation failed: %s" % (name, result_dir / "simulator.log"))
        transcript = (result_dir / "simulator.log").read_text(encoding="utf-8-sig")
        if "P3_RESULT: PASS snapshots=2 mismatch=0" not in transcript or "** Error" in transcript or "** Fatal" in transcript:
            raise RuntimeError("%s transcript did not pass all gates" % name)
        results[name] = {"opcodes": [opcode], "compile_dir": str(compile_dir),
                         "result_dir": str(result_dir),
                         "warnings": sum("** Warning:" in line for line in transcript.splitlines())}
    # Exercise the standard INT8 ONNX path as well as the QDQ path above.
    name = "standard_onnx"
    model_path, qfile, input_file = out / "standard.onnx", out / "standard_quant.json", out / "standard_input.json"
    _save_model(model_path)
    qfile.write_text(json.dumps({"schema": QUANT_SCHEMA, "tensors": {
        "x": {"scale": .25, "zero_point": -21}, "y": {"scale": .25, "zero_point": -21}}, "convs": {}}))
    data = rng.integers(-128, 128, IN_SHAPE, dtype=np.int8)
    write_input_manifest(input_file, {"x": data})
    compile_dir, result_dir = out / (name + "_compile"), out / (name + "_result")
    compile_graph(model_path, qfile, input_file, compile_dir, engine_pruning="graph")
    cmd = ["powershell", "-NoProfile", "-File", str(ROOT / "verification/sim/run_p3_graph_shell.ps1"),
           "-CompileDir", str(compile_dir), "-ResultDir", str(result_dir), "-ProjectRoot", str(ROOT),
           "-Simulator", "modelsim", "-SimDefine", "TB_MAC_IMPL_2,TB_LEAN_FUNC", "-Seed", "20261029"]
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       text=True, timeout=180)
    print(r.stdout, end="")
    transcript_path = result_dir / "simulator.log"
    if r.returncode:
        raise RuntimeError("standard ONNX full-graph simulation failed: %s" % transcript_path)
    transcript = transcript_path.read_text(encoding="utf-8-sig")
    if "P3_RESULT: PASS snapshots=2 mismatch=0" not in transcript or "** Error" in transcript or "** Fatal" in transcript:
        raise RuntimeError("standard ONNX transcript did not pass all gates")
    results[name] = {"opcodes": [0x1d], "compile_dir": str(compile_dir),
                     "result_dir": str(result_dir),
                     "warnings": sum("** Warning:" in line for line in transcript.splitlines())}
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print("COMPILED_GRAPH_PASS scenarios=10 cross_backend_bytes=%d artifacts=%s" % (len(mixed), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
