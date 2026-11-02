#!/usr/bin/env python3
"""E1 ONNX-subset frontend and planning CLI.

Commands:
  registry              dump the operator capability registry
  parse-onnx            ONNX -> deterministic graph_ir.json + compile report
  legacy-ir             legacy ops.json -> graph IR + exact schema round-trip
  legacy-equivalence    run npu_gen on original and round-tripped ops.json
  plan                  canonical IR -> schedule/liveness/layout/allocation
  codegen               W2 plan -> instruction stream v2 + ROM image
  compile               full ONNX/quant/parameters/IR/plan/codegen pipeline + IP package
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onnx_gen import engine_profiles, targets  # noqa: E402
from tools.onnx_gen import reference_cache as reference_cache_module  # noqa: E402
from tools.onnx_gen.graph_ir import CompileError, Graph  # noqa: E402
from tools.onnx_gen.legacy_adapter import (  # noqa: E402
    load_legacy_ops,
    parse_legacy_document,
    to_legacy_ops,
)
from tools.onnx_gen.onnx_frontend import load_onnx  # noqa: E402
from tools.onnx_gen.op_registry import registry_document  # noqa: E402
from tools.onnx_gen.graph_planner import build_graph_plan  # noqa: E402
from tools.onnx_gen.layout import build_layout_plan  # noqa: E402
from tools.onnx_gen.memory_planner import (  # noqa: E402
    DEFAULT_ALIGNMENT,
    DEFAULT_BASE,
    build_memory_plan,
    check_legacy_instruction_addresses,
)
from tools.onnx_gen.instr_codegen import generate_stream, write_stream  # noqa: E402


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_report(kind: str) -> Dict[str, Any]:
    return {
        "schema": "e1-compile-report-1",
        "kind": kind,
        "status": "fail",
        "gates": {
            "onnx_dependency": "pass",
            "ir_parse": "not_run",
            "shape_inference": "not_run",
            "topological_sort": "not_run",
            "schedule": "not_run",
            "liveness": "not_run",
            "layout_planning": "not_run",
            "legacy_address_compat": "not_run",
            "memory_planning": "not_run",
            "codegen": "not_run",
            "instruction_stream_v2": "not_run",
            "instruction_equivalence": "not_run",
            "rtl": "not_run",
            "board": "not_run",
        },
    }


def _dependency_summary() -> Dict[str, Any]:
    try:
        import onnx
        return {
            "available": True,
            "version": onnx.__version__,
            "max_opset": onnx.defs.onnx_opset_version(),
            "ir_version": onnx.IR_VERSION,
            "python": sys.executable,
        }
    except ImportError as exc:
        return {"available": False, "error": str(exc), "python": sys.executable}


def command_registry(args) -> int:
    _write_json(args.out, registry_document())
    print(json.dumps({"status": "pass", "out": str(args.out)}, indent=2))
    return 0


def command_parse_onnx(args) -> int:
    report = _base_report("parse_onnx")
    report["onnx_dependency"] = _dependency_summary()
    try:
        graph = load_onnx(args.model, name=args.name)
        document = graph.document()
        _write_json(args.out, document)
        report.update({
            "status": "pass",
            "model": str(args.model),
            "model_sha256": _sha256_file(args.model),
            "graph_ir": str(args.out),
            "graph_sha256": graph.digest(),
            "graph_name": graph.name,
            "num_nodes": len(graph.nodes),
            "num_tensors": len(graph.tensors),
            "num_initializers": len(graph.initializers),
            "nodes": [node.op_type for node in graph.nodes],
            "inputs": list(graph.inputs),
            "outputs": list(graph.outputs),
        })
        report["gates"].update({
            "ir_parse": "pass",
            "shape_inference": "pass",
            "topological_sort": "pass",
        })
    except CompileError as exc:
        report["error"] = exc.to_dict()
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"],
                      "graph_ir": str(args.out) if report["status"] == "pass" else None,
                      "report": str(args.report),
                      "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def command_legacy_ir(args) -> int:
    report = _base_report("legacy_ir")
    try:
        graph = load_legacy_ops(args.ops)
        roundtrip = to_legacy_ops(graph)
        original = json.loads(args.ops.read_text(encoding="utf-8-sig"))
        _write_json(args.out, graph.document())
        if args.roundtrip_out:
            _write_json(args.roundtrip_out, roundtrip)
        exact = roundtrip == original
        storage = {}
        for name in sorted(graph.tensors):
            tensor = graph.tensors[name]
            if tensor.storage is not None:
                storage[name] = dict(tensor.storage)
        report.update({
            "status": "pass" if exact else "fail",
            "ops_json": str(args.ops),
            "ops_sha256": _sha256_file(args.ops),
            "graph_ir": str(args.out),
            "graph_sha256": graph.digest(),
            "roundtrip_ops": str(args.roundtrip_out) if args.roundtrip_out else None,
            "roundtrip_exact": exact,
            "num_nodes": len(graph.nodes),
            "num_tensors": len(graph.tensors),
            "nodes": [node.op_type for node in graph.nodes],
            "activation_storage_bytes": {
                name: value["bytes"] for name, value in storage.items()
                if value.get("layout") == "NHWC8_A16"
            },
        })
        report["gates"].update({
            "ir_parse": "pass",
            "shape_inference": "pass",
            "topological_sort": "pass",
            "legacy_roundtrip": "pass" if exact else "fail",
        })
    except CompileError as exc:
        report["error"] = exc.to_dict()
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"],
                      "graph_ir": str(args.out) if report["status"] == "pass" else None,
                      "report": str(args.report),
                      "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def _run_npu_gen(ops_path: Path, out_dir: Path, run_id: str, log_path: Path) -> None:
    if out_dir.exists():
        raise RuntimeError("refusing to reuse npu_gen output directory %s" % out_dir)
    out_dir.mkdir(parents=True)
    command = [sys.executable, "-B", str(ROOT / "tools/npu_gen/npu_gen.py"),
               "--ops", str(ops_path), "--out-dir", str(out_dir), "--run-id", run_id]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError("npu_gen failed; see %s" % log_path)


def _words_sha256(stream: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(stream["words"]).encode("utf-8")).hexdigest()


def _compare_generated(left: Path, right: Path, reference: Optional[Path] = None) -> Dict[str, Any]:
    left_stream = json.loads((left / "instruction_stream.json").read_text(encoding="utf-8"))
    right_stream = json.loads((right / "instruction_stream.json").read_text(encoding="utf-8"))
    left_mem = left / left_stream["mem_file"]
    right_mem = right / right_stream["mem_file"]
    result: Dict[str, Any] = {
        "instruction_words_equal": left_stream["words"] == right_stream["words"],
        "mem_sha256_equal": left_stream["mem_sha256"] == right_stream["mem_sha256"],
        "words_sha256_equal": _words_sha256(left_stream) == _words_sha256(right_stream),
        "left_words_sha256": _words_sha256(left_stream),
        "right_words_sha256": _words_sha256(right_stream),
        "left_mem_sha256": _sha256_file(left_mem),
        "right_mem_sha256": _sha256_file(right_mem),
    }
    result["status"] = "pass" if all(result[key] for key in
                                     ("instruction_words_equal", "mem_sha256_equal",
                                      "words_sha256_equal")) else "fail"
    if reference is not None:
        ref_stream_path = reference / "instruction_stream.json"
        ref_stream = json.loads(ref_stream_path.read_text(encoding="utf-8"))
        ref_mem = reference / ref_stream["mem_file"]
        result["reference"] = str(reference)
        result["reference_words_equal"] = left_stream["words"] == ref_stream["words"]
        result["reference_words_sha256_equal"] = _words_sha256(left_stream) == _words_sha256(ref_stream)
        result["reference_mem_equal"] = _sha256_file(ref_mem) == _sha256_file(left_mem)
        if not (result["reference_words_equal"] and result["reference_words_sha256_equal"] and
                result["reference_mem_equal"]):
            result["status"] = "fail"
    return result


def command_legacy_equivalence(args) -> int:
    report = _base_report("legacy_equivalence")
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    try:
        graph = load_legacy_ops(args.ops)
        roundtrip = to_legacy_ops(graph)
        original = json.loads(args.ops.read_text(encoding="utf-8-sig"))
        ir_path = work / "legacy_graph_ir.json"
        roundtrip_ops = work / "legacy_ops_roundtrip.json"
        _write_json(ir_path, graph.document())
        _write_json(roundtrip_ops, roundtrip)
        exact = roundtrip == original
        if not exact:
            raise CompileError("E_LEGACY_ROUNDTRIP", "legacy ops.json round-trip differs")
        original_dir = work / "generated_original"
        roundtrip_dir = work / "generated_roundtrip"
        _run_npu_gen(args.ops, original_dir, "e1_legacy_original",
                     work / "npu_gen_original.log")
        _run_npu_gen(roundtrip_ops, roundtrip_dir, "e1_legacy_roundtrip",
                     work / "npu_gen_roundtrip.log")
        comparison = _compare_generated(original_dir, roundtrip_dir, args.reference_generated)
        report.update({
            "status": comparison["status"],
            "ops_json": str(args.ops),
            "graph_ir": str(ir_path),
            "graph_sha256": graph.digest(),
            "roundtrip_ops": str(roundtrip_ops),
            "roundtrip_exact": exact,
            "original_generated": str(original_dir),
            "roundtrip_generated": str(roundtrip_dir),
            "comparison": comparison,
        })
        report["gates"].update({
            "ir_parse": "pass",
            "shape_inference": "pass",
            "topological_sort": "pass",
            "legacy_roundtrip": "pass",
            "instruction_equivalence": comparison["status"],
        })
    except (CompileError, RuntimeError, OSError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, CompileError):
            report["error"] = exc.to_dict()
        else:
            report["error"] = {"code": "E_RUNTIME", "message": str(exc)}
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def command_plan(args) -> int:
    report = _base_report("plan")
    try:
        graph_doc = json.loads(args.graph.read_text(encoding="utf-8"))
        graph = Graph.from_dict(graph_doc)
        graph_plan = build_graph_plan(graph)
        layout_plan = build_layout_plan(graph)
        memory_plan = build_memory_plan(
            graph, graph_plan, layout_plan,
            profile=args.profile.replace("-", "_"),
            base=int(args.arena_base, 16),
            alignment=int(args.alignment),
        )
        legacy_check = None
        if args.reference_instruction_stream is not None:
            stream = json.loads(args.reference_instruction_stream.read_text(encoding="utf-8"))
            legacy_check = check_legacy_instruction_addresses(graph, memory_plan, stream)
            memory_plan["checks"]["legacy_address_compat"] = legacy_check
            memory_plan["gates"]["legacy_address_compat"] = legacy_check["status"]
        manifest = {
            "schema": "e1-w2-plan-manifest-1",
            "graph_name": graph.name,
            "graph_digest": graph.digest(),
            "profile": args.profile.replace("-", "_"),
            "graph_plan": graph_plan,
            "layout_plan": layout_plan,
            "memory_plan": memory_plan,
            "gates": {
                "schedule": graph_plan["gates"]["schedule"],
                "liveness": graph_plan["gates"]["liveness"],
                "layout_planning": layout_plan["gates"]["layout_planning"],
                "alias_planning": layout_plan["gates"]["alias_planning"],
                "memory_planning": memory_plan["gates"]["memory_planning"],
                "legacy_address_compat": memory_plan["gates"]["legacy_address_compat"],
                "codegen": "not_run",
                "rtl": "not_run",
                "board": "not_run",
            },
        }
        _write_json(args.out, manifest)
        report.update({
            "status": "pass" if (memory_plan["status"] == "pass" and
                                 memory_plan["gates"]["legacy_address_compat"] != "fail") else "fail",
            "graph_ir": str(args.graph),
            "graph_sha256": graph.digest(),
            "plan_manifest": str(args.out),
            "plan_sha256": _sha256_file(args.out),
            "profile": manifest["profile"],
            "schedule_steps": len(graph_plan["schedule"]),
            "buffers": memory_plan["stats"]["buffers"],
            "reuse_count": memory_plan["stats"]["reuse_count"],
            "arena": memory_plan["arena"],
            "layout_stats": layout_plan["stats"],
            "memory_checks": memory_plan["checks"],
            "legacy_address_check": legacy_check,
        })
        report["gates"].update({
            "ir_parse": "pass",
            "shape_inference": "pass",
            "topological_sort": "pass",
            "schedule": graph_plan["gates"]["schedule"],
            "liveness": graph_plan["gates"]["liveness"],
            "layout_planning": layout_plan["gates"]["layout_planning"],
            "memory_planning": memory_plan["gates"]["memory_planning"],
            "legacy_address_compat": memory_plan["gates"]["legacy_address_compat"],
        })
    except CompileError as exc:
        report["error"] = exc.to_dict()
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"],
                      "plan_manifest": str(args.out) if report["status"] == "pass" else None,
                      "report": str(args.report),
                      "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def command_codegen(args) -> int:
    report = _base_report("codegen")
    try:
        graph_doc = json.loads(args.graph.read_text(encoding="utf-8"))
        graph = Graph.from_dict(graph_doc)
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        stream = generate_stream(graph, plan, run_id=args.name)
        artifacts = write_stream(stream, args.out_dir, name=args.name)
        reference_check = "not_run"
        if args.reference_stream is not None:
            reference = json.loads(args.reference_stream.read_text(encoding="utf-8"))
            reference_check = "pass" if stream["words"] == reference["words"] else "fail"
        report.update({
            "status": "pass" if reference_check != "fail" else "fail",
            "graph_ir": str(args.graph),
            "plan_manifest": str(args.plan),
            "graph_sha256": graph.digest(),
            "instruction_stream": artifacts["instruction_stream"],
            "mem": artifacts["mem"],
            "mem_sha256": artifacts["mem_sha256"],
            "record_count": len(stream["records"]),
            "records": [{"op": row["op"], "opcode": "0x%04X" % row["opcode"],
                         "payload_words": row["payload_words"]} for row in stream["records"]],
            "quant_allocations": stream["quant_allocations"],
            "reference_stream_equivalence": reference_check,
        })
        report["gates"].update({
            "ir_parse": "pass",
            "schedule": "pass",
            "liveness": "pass",
            "layout_planning": "pass",
            "memory_planning": "pass",
            "instruction_stream_v2": "pass",
            "instruction_equivalence": reference_check,
        })
    except CompileError as exc:
        report["error"] = exc.to_dict()
    _write_json(args.report, report)
    print(json.dumps({"status": report["status"], "report": str(args.report),
                      "instruction_stream": report.get("instruction_stream"),
                      "error": report.get("error")}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "pass" else 2


def _request_document(args) -> Dict[str, Any]:
    return {
        "compute_profile": args.compute_profile,
        "engine_profile": None if args.engine_profile is None
                          else str(args.engine_profile),
        "target": args.target,
        "dense_overrides": {key: value for key, value in
                            (("tn", args.tn), ("tm", args.tm),
                             ("mac_impl", args.mac_impl))
                            if value is not None},
        "depthwise_overrides": {key: value for key, value in
                                (("pc", args.dw_pc),
                                 ("tap_parallel", args.dw_tap_parallel),
                                 ("rq_lanes", args.dw_rq_lanes))
                                if value is not None},
        "memory_profile": args.profile,
        "numeric_profile": getattr(args, "numeric_profile", None),
        "fragment_nodes": getattr(args, "fragment_nodes", None),
        "fragment_input_rows": getattr(args, "fragment_input_rows", None),
        "max_feature_width": getattr(args, "max_feature_width", None),
        "max_feature_height": getattr(args, "max_feature_height", None),
        "sim_ddr_limit_mib": getattr(args, "sim_ddr_limit_mib", None),
    }


def command_compile(args) -> int:
    from tools.onnx_gen.compile_pipeline import compile_graph
    report: Dict[str, Any] = {
        "schema": "e1-compile-cli-1",
        "command": "compile",
        "requested": _request_document(args),
        "out_dir": str(args.out_dir),
    }
    reference_cache = reference_cache_module.cache_from_args(args)
    report["reference_cache_request"] = {
        "enabled": reference_cache is not None,
        "dir": (None if reference_cache is None
                else reference_cache.relative_directory()),
        "mode": (None if reference_cache is None else reference_cache.mode),
    }
    engine_pruning = getattr(args, "engine_pruning", None)
    if engine_pruning is None:
        engine_pruning = "graph" if args.numeric_profile else "none"
    report["engine_pruning"] = engine_pruning
    input_u8 = None
    if getattr(args, "input_u8", None) is not None:
        import numpy as _np
        input_u8 = _np.load(str(args.input_u8))
    fragment_nodes = None
    if getattr(args, "fragment_nodes", None):
        fragment_nodes = [part.strip() for part in str(args.fragment_nodes).split(",")
                          if part.strip()]
    if args.numeric_profile is not None and input_u8 is None:
        report.update({"status": "fail", "compile_graph": "rejected",
                       "error": {"code": "E_CFG",
                                 "message": "affine numeric profile requires --input-u8"}})
        if args.report is not None:
            _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    try:
        manifest = compile_graph(
            args.model, args.quant_contract, args.inputs, args.out_dir,
            numeric_profile=args.numeric_profile, fragment_nodes=fragment_nodes,
            fragment_input_rows=getattr(args, "fragment_input_rows", None),
            input_u8=input_u8,
            engine_pruning=engine_pruning,
            maxpool_backend=args.maxpool_backend,
            profile=args.profile, mac_impl=args.mac_impl,
            tn=args.tn, tm=args.tm, name=args.name, run_id=args.run_id,
            repo_root=args.source_root,
            activation_contract_path=args.activation_contract,
            compute_profile=args.compute_profile,
            engine_profile=args.engine_profile, target=args.target,
            dw_pc=args.dw_pc, dw_tap_parallel=args.dw_tap_parallel,
            dw_rq_lanes=args.dw_rq_lanes,
            max_feature_width=args.max_feature_width,
            max_feature_height=args.max_feature_height,
            sim_ddr_limit_mib=args.sim_ddr_limit_mib,
            reference_cache=reference_cache,
        )
    except CompileError as exc:
        # Structured rejection before artifacts: the compile directory must not
        # contain a half-written build.
        report.update({"status": "fail", "compile_graph": "rejected",
                       "error": exc.to_dict()})
        if args.report is not None:
            _write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2
    ip_row = None
    if args.pack_ip:
        try:
            # Keep the direct compile CLI and the model-export wrapper on the
            # same source-bound IP packer.  The compile directory itself is a
            # self-contained export root for this lower-level entry point.
            from tools.onnx_gen.export_model import _pack_ip, _sanitize_name
            logical_name = _sanitize_name(args.name or args.model.stem)
            ip_row = _pack_ip(
                args.out_dir, args.out_dir, logical_name, args.ip_version,
                targets.resolve(args.target).part, args.vivado,
                _sha256_file(args.model))
        except CompileError as exc:
            report.update({"status": "fail", "compile_graph": "pass",
                           "ip_pack": "fail", "error": exc.to_dict()})
            if args.report is not None:
                _write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 4
        except Exception as exc:
            report.update({"status": "fail", "compile_graph": "pass",
                           "ip_pack": "fail",
                           "error": {"code": "E_CP_EXTERNAL", "message": str(exc)}})
            if args.report is not None:
                _write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 4
    report.update({
        "status": manifest["status"],
        "compile_graph": "pass",
        "ip_pack": "pass" if ip_row is not None else "not_run",
        "ip": ip_row,
        "manifest": str(Path(args.out_dir) / "compile_manifest.json"),
        "records": len(manifest["records"]),
        "graph_digest": manifest["lowered_graph_digest"],
        "engine_profile": manifest.get("engine_profile"),
        "target": manifest.get("target"),
        "dense": manifest.get("dense"),
        "depthwise": manifest.get("depthwise"),
        "generated_parameters": manifest.get("generated_parameters"),
        "spatial_capacity": manifest.get("spatial_capacity"),
        "engine_capacity": manifest.get("engine_capacity"),
        "memory_limits": manifest.get("memory_limits"),
        "simulation_budget": manifest.get("simulation_budget"),
        "reference_cache": manifest.get("reference_cache"),
        "gates": manifest.get("gates"),
    })
    if args.report is not None:
        _write_json(args.report, report)
    print(json.dumps({key: report[key] for key in
                      ("status", "manifest", "records", "graph_digest", "ip",
                       "engine_profile", "target", "depthwise",
                       "generated_parameters")}, ensure_ascii=False, indent=2))
    return 0


def command_engine_profiles(args) -> int:
    print(json.dumps({"schema": engine_profiles.SCHEMA,
                      "default": engine_profiles.DEFAULT_PROFILE,
                      "profiles": engine_profiles.list_profiles()},
                     ensure_ascii=False, indent=2))
    return 0


def command_targets(args) -> int:
    print(json.dumps({"schema": targets.SCHEMA,
                      "default": targets.DEFAULT_TARGET,
                      "targets": targets.list_targets()},
                     ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    registry = sub.add_parser("registry")
    registry.add_argument("--out", type=Path, required=True)
    registry.set_defaults(func=command_registry)

    parse_onnx = sub.add_parser("parse-onnx")
    parse_onnx.add_argument("--model", type=Path, required=True)
    parse_onnx.add_argument("--out", type=Path, required=True)
    parse_onnx.add_argument("--report", type=Path, required=True)
    parse_onnx.add_argument("--name")
    parse_onnx.set_defaults(func=command_parse_onnx)

    legacy = sub.add_parser("legacy-ir")
    legacy.add_argument("--ops", type=Path, required=True)
    legacy.add_argument("--out", type=Path, required=True)
    legacy.add_argument("--roundtrip-out", type=Path)
    legacy.add_argument("--report", type=Path, required=True)
    legacy.set_defaults(func=command_legacy_ir)

    eq = sub.add_parser("legacy-equivalence")
    eq.add_argument("--ops", type=Path, required=True)
    eq.add_argument("--work-dir", type=Path, required=True)
    eq.add_argument("--report", type=Path, required=True)
    eq.add_argument("--reference-generated", type=Path)
    eq.set_defaults(func=command_legacy_equivalence)

    plan = sub.add_parser("plan")
    plan.add_argument("--graph", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--report", type=Path, required=True)
    plan.add_argument("--profile", choices=("first-fit", "no-reuse", "legacy-compat"), default="first-fit")
    plan.add_argument("--arena-base", default="0x%08X" % DEFAULT_BASE)
    plan.add_argument("--alignment", type=int, default=DEFAULT_ALIGNMENT)
    plan.add_argument("--reference-instruction-stream", type=Path)
    plan.set_defaults(func=command_plan)

    codegen = sub.add_parser("codegen")
    codegen.add_argument("--graph", type=Path, required=True)
    codegen.add_argument("--plan", type=Path, required=True)
    codegen.add_argument("--out-dir", type=Path, required=True)
    codegen.add_argument("--name", default="e1_graph")
    codegen.add_argument("--report", type=Path, required=True)
    codegen.add_argument("--reference-stream", type=Path)
    codegen.set_defaults(func=command_codegen)

    compile_cmd = sub.add_parser("compile")
    compile_cmd.add_argument("--model", type=Path, required=True)
    compile_cmd.add_argument("--quant-contract", type=Path, default=None)
    compile_cmd.add_argument("--inputs", type=Path, default=None)
    # v0.2 affine-int8-v1 public path: an explicit numeric profile selects the
    # QDQ frontend (affine_frontend) and derives the contract from the model.
    compile_cmd.add_argument("--numeric-profile", default=None,
                             choices=(None, "affine-int8-v1"),
                             help="explicit numeric profile; affine-int8-v1 compiles a "
                                  "standard QDQ ONNX through the mainline pipeline")
    compile_cmd.add_argument("--fragment-nodes", default=None,
                             help="comma-separated deployment node ids to compile as a "
                                  "short real fragment (affine profile only)")
    compile_cmd.add_argument("--fragment-input-rows", type=int, default=None,
                             help="prefix height of a fragment boundary input")
    compile_cmd.add_argument("--input-u8", type=Path, default=None,
                             help="uint8 NCHW/NHWC npy sample for the affine reference "
                                  "and the DDR input image")
    compile_cmd.add_argument("--source-root", type=Path,
                             default=Path(__file__).resolve().parents[2],
                             help="RTL source root (default: project mainline)")
    compile_cmd.add_argument("--activation-contract", type=Path,
                             help="explicit LUT activation deployment contract")
    compile_cmd.add_argument("--out-dir", type=Path, required=True)
    compile_cmd.add_argument("--profile", choices=("no_reuse", "first_fit"), default="first_fit")
    # Dense tier: legacy flags stay, but their defaults are NOT passed as
    # explicit values, so an explicit engine profile can never be silently
    # overridden by a parser default.
    compile_cmd.add_argument("--mac-impl", type=int, default=None)
    compile_cmd.add_argument("--tn", type=int, default=None)
    compile_cmd.add_argument("--tm", type=int, default=None)
    compile_cmd.add_argument("--compute-profile", choices=("tn4", "tn8", "tn16"),
                             default=None,
                             help="dense-only compatibility tier (scope=dense_conv)")
    # Per-engine configuration (new in this batch).
    compile_cmd.add_argument("--engine-pruning", choices=("none", "graph"), default=None,
                             help="physical-instance pruning: 'graph' elaborates only the "
                                  "cores the lowered opcodes need (default: graph for "
                                  "affine numeric profiles, none otherwise); 'none' keeps "
                                  "the historical full-function netlist")
    compile_cmd.add_argument("--maxpool-backend", choices=("auto", "generic"), default="auto",
                             help="choose safe optimized MaxPool where available, or force generic")
    compile_cmd.add_argument("--engine-profile", default=None,
                             help="engine profile JSON path or builtin id "
                                  "(e1-engine-profile-1); default dw_c8_t1_q1")
    compile_cmd.add_argument("--target", default=None,
                             help="target registry id, e.g. project2_zu15eg_200mhz")
    compile_cmd.add_argument("--dw-pc", type=int, default=None,
                             help="convenience DW_PC override (legal: 4, 8, 16)")
    compile_cmd.add_argument("--dw-tap-parallel", type=int, default=None,
                             help="convenience DW_TAP_PARALLEL override "
                                  "(implemented: 1)")
    compile_cmd.add_argument("--dw-rq-lanes", type=int, default=None,
                             help="convenience DW_RQ_LANES override (legal: 1, 2)")
    compile_cmd.add_argument("--max-feature-width", default=None,
                             help="v0.2 feature-map width capacity: 'auto' or int (1..640)")
    compile_cmd.add_argument("--max-feature-height", default=None,
                             help="v0.2 feature-map height capacity: 'auto' or int (1..640)")
    compile_cmd.add_argument("--sim-ddr-limit-mib", type=int, default=None,
                             help="simulation/DDR-image budget in MiB (default 32 in "
                                  "spatial mode)")
    compile_cmd.add_argument("--name")
    compile_cmd.add_argument("--run-id", default="e1p2")
    compile_pack = compile_cmd.add_mutually_exclusive_group()
    compile_pack.add_argument("--pack-ip", dest="pack_ip", action="store_true",
                              help="compatibility alias; IP packaging is the default")
    compile_pack.add_argument("--no-pack-ip", dest="pack_ip", action="store_false",
                              help="skip Vivado packaging and emit compile/RTL artifacts only")
    compile_cmd.set_defaults(pack_ip=True)
    compile_cmd.add_argument("--ip-version", default="0.1",
                             help="Vivado IP VLNV version (default 0.1)")
    compile_cmd.add_argument("--vivado", default="E:/Vitis/Vivado/2021.1/bin/vivado.bat",
                             help="Vivado batch executable used for IP packaging")
    reference_cache_module.add_cli_arguments(compile_cmd)
    compile_cmd.add_argument("--report", type=Path,
                             help="optional structured acceptance/rejection report")
    compile_cmd.set_defaults(func=command_compile)

    profiles_cmd = sub.add_parser("engine-profiles")
    profiles_cmd.set_defaults(func=command_engine_profiles)

    targets_cmd = sub.add_parser("targets")
    targets_cmd.set_defaults(func=command_targets)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompileError, OSError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False),
              file=sys.stderr)
        raise SystemExit(1)
