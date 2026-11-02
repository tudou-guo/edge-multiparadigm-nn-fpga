"""User-facing compute-profile export CLI (thin wrapper around compile_pipeline).

This is the "user-understandable" layer: it only selects an elaboration-time
compute tier (tn4 / tn8 / tn16), reserves a fresh output root, calls the same
p2 compile engine, automatically packs the generated IP source repository,
and records the honest gate state.  Use --no-pack-ip for RTL-only output. It
never duplicates the compile pipeline and never claims
that compile success, IP packaging or a historical 8x8 physical PASS is a
hardware qualification of the new candidate.

Exit values (technical plan section 4.4):
  0 = requested action completed (the export may still be unverified)
  2 = CLI syntax / CompileError configuration rejection
  3 = resource preflight blocked
  4 = external tool (Vivado/pack) failure
  5 = identity / qualification verification failure
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import compute_profiles
from . import reference_cache as reference_cache_module
from .compute_profiles import DEFAULT_PUBLIC, INTERNAL_NAMES, PUBLIC_NAMES, list_profiles, resolve
from .graph_ir import (
    CompileError,
    E_CP_IDENTITY,
    E_CP_NOT_QUALIFIED,
    E_CP_OUTPUT,
    E_CP_RESOURCE,
)
from .compile_pipeline import SCHEMA as COMPILE_SCHEMA, compile_graph

SCHEMA = "e1-compute-profile-export-1"
ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_RESOURCE = 3
EXIT_EXTERNAL = 4
EXIT_IDENTITY = 5

MIN_FREE_BYTES = 5 * 1024 ** 3
PROTECTED_PREFIXES = (
    "baselines", "releases", "examples", "pynq", "maintenance", "docs",
    "rtl", "tools", "verification", "build/mnist_project2_board",
)
# Run-directory name prefixes whose own export/ root is a legitimate fresh
# output (they are created by the owning driver immediately before export).
ALLOWED_NEW_RUN_PREFIXES = ("compute_profile_", "e1_final_acceptance_",
                            "lut_activation_", "library_consolidation_",
                            "v02_spatial_", "spatial_",
                            # v0.2 second-batch fix run (fast development /
                            # B2 closure): fresh export roots under this run are
                            # legitimate outputs, old runs stay protected.
                            "v02_b2_fix_")
DEFAULT_VIVADO = "E:/Vitis/Vivado/2021.1/bin/vivado.bat"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _err(code: str, message: str, detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"status": "fail", "error": {"code": code, "message": message,
                                        "detail": dict(detail or {})}}


def _exit_code(code: str) -> int:
    if code in (E_CP_IDENTITY, E_CP_NOT_QUALIFIED):
        return EXIT_IDENTITY
    if code == E_CP_RESOURCE:
        return EXIT_RESOURCE
    return EXIT_CONFIG


def _resolve_out(out: str) -> Path:
    path = Path(out)
    if not path.is_absolute():
        path = (Path.cwd() / path)
    return Path(os.path.realpath(str(path)))


def _guard_out(out: Path) -> None:
    """Reject protected trees, existing assets and symlink/junction escapes.

    The out root must be a fresh, empty directory.  It may live outside the
    project, or under build/onnx_e1/ in a *new* run directory; it may never be
    inside an already delivered asset tree or an old run.
    """
    project_root = Path(os.path.realpath(str(PROJECT_ROOT)))
    try:
        relative = out.relative_to(project_root)
    except ValueError:
        relative = None
    if relative is not None:
        parts = relative.as_posix().split("/")
        head = "/".join(parts[:3]) if len(parts) >= 3 else relative.as_posix()
        for prefix in PROTECTED_PREFIXES:
            if relative.as_posix() == prefix or relative.as_posix().startswith(prefix + "/"):
                raise CompileError(E_CP_OUTPUT,
                                   "output root is inside the protected tree %r" % prefix,
                                   detail={"out": str(out), "prefix": prefix})
        if parts[0] == "build" and len(parts) >= 3 and parts[1] == "onnx_e1":
            run_dir = project_root / parts[0] / parts[1] / parts[2]
            if run_dir.exists() and not any(
                    parts[2].startswith(prefix) for prefix in ALLOWED_NEW_RUN_PREFIXES):
                raise CompileError(E_CP_OUTPUT,
                                   "output root is inside an existing (old) run directory",
                                   detail={"out": str(out), "run": str(run_dir)})
    if out.exists():
        if not out.is_dir():
            raise CompileError(E_CP_OUTPUT, "output root exists and is not a directory",
                               detail={"out": str(out)})
        if any(out.iterdir()):
            raise CompileError(E_CP_OUTPUT,
                               "output root must be a fresh empty directory",
                               detail={"out": str(out)})


def _guard_outputs(out: Path, lock_path: Path) -> None:
    if lock_path.exists():
        raise CompileError(E_CP_OUTPUT,
                           "another export already owns this output root",
                           detail={"lock": str(lock_path)})


def _cli_spatial_axis(value):
    """CLI form of --max-feature-width/height: 'auto' or a positive integer.

    The compiler API keeps its strict type contract (bool/float/numeric string
    rejected as E_SPATIAL_AXIS/E_SPATIAL_SCHEMA), but the public CLI receives
    every value as a string, so a plain decimal integer is converted here before
    the strict parser sees it. Anything else is passed through unchanged and the
    versioned parser raises its structured error.
    """
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    return value


def _cli_int(value):
    if value is None:
        return None
    text = str(value).strip()
    if re.fullmatch(r"[+-]?[0-9]+", text):
        return int(text)
    return value


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_")
    if not cleaned:
        raise CompileError(E_CP_OUTPUT, "logical name is empty after sanitizing")
    if not cleaned[0].isalpha():
        cleaned = "m_" + cleaned
    return cleaned


def _resource_gate() -> Dict[str, Any]:
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free = int(usage.free)
    if free < MIN_FREE_BYTES:
        raise CompileError(E_CP_RESOURCE, "free disk space is below the 5 GiB export gate",
                           detail={"free_bytes": free, "required": MIN_FREE_BYTES})
    return {"free_bytes": free, "required": MIN_FREE_BYTES, "pass": True}


def _require_qualified(path: str, resolved, model_sha: str, contract_sha: str,
                       input_sha: str) -> Dict[str, Any]:
    """Accept only a complete identity-bound qualification record."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompileError(E_CP_IDENTITY, "qualification record unreadable: %s" % exc,
                           detail={"path": path})
    profile = document.get("compute_profile") or {}
    if document.get("schema") != "e1-compute-profile-qualification-1":
        raise CompileError(E_CP_IDENTITY, "qualification schema mismatch",
                           detail={"path": path, "schema": document.get("schema")})
    if profile.get("name") != resolved.name or profile.get("definition_sha256") != resolved.definition_sha256:
        raise CompileError(E_CP_NOT_QUALIFIED,
                           "qualification does not bind this exact compute profile definition",
                           detail={"path": path, "recorded": profile.get("name"),
                                   "requested": resolved.name})
    identity = document.get("identity") or {}
    expected = {"model_sha256": model_sha, "quant_contract_sha256": contract_sha,
                "input_manifest_sha256": input_sha}
    mismatched = {key: {"recorded": identity.get(key), "actual": value}
                  for key, value in expected.items() if identity.get(key) != value}
    if mismatched:
        raise CompileError(E_CP_IDENTITY,
                           "qualification identity does not match the requested inputs",
                           detail={"mismatched": mismatched})
    gates = document.get("gates") or {}
    required = ("functional", "ooc", "platform_5ns", "three_piece", "identity")
    not_pass = {key: gates.get(key) for key in required if gates.get(key) != "pass"}
    if not_pass:
        raise CompileError(E_CP_NOT_QUALIFIED,
                           "--require-qualified needs every gate to be pass",
                           detail={"gates": not_pass})
    return document


def _run(cmd: List[str], log_path: Path, cwd: Optional[Path] = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.write("STARTED: " + started + "\n")
        process = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                                 stdout=log, stderr=subprocess.STDOUT)
        log.write("\nEXIT_CODE: %d\n" % process.returncode)
        log.write("FINISHED: " + _now() + "\n")
    return int(process.returncode)


def _pack_ip(compile_dir: Path, out: Path, ip_name: str, version: str,
             part: str, vivado: str, model_sha: str) -> Dict[str, Any]:
    from .source_library import verify_generated
    verify_generated(compile_dir / "generated")
    rtl_manifest = json.loads((compile_dir / "generated" / "rtl_manifest.json").read_text(encoding="utf-8"))
    source_root = Path(rtl_manifest["source_root"]).resolve()
    ip_repo = out / "ip_repo"
    ip_dir = ip_repo / ("%s_%s" % (ip_name, version))
    if ip_dir.exists():
        raise CompileError(E_CP_OUTPUT, "IP directory already exists; refusing to repack",
                           detail={"ip_dir": str(ip_dir)})
    ip_repo.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / "pack_npu_ip.log"
    script = PROJECT_ROOT / "tools/npu_gen/pack_npu_ip.tcl"
    # vivado.bat must go through cmd (CreateProcess cannot start a batch file).
    cmd = ["cmd", "/c", vivado, "-mode", "batch", "-notrace", "-source", str(script),
           "-tclargs", str(compile_dir / "generated"), str(ip_repo), ip_name,
           version, str(source_root), part]
    rc = _run(cmd, log)
    component = ip_dir / "component.xml"
    if rc != 0 or not component.exists():
        raise CompileError("E_CP_EXTERNAL", "IP packaging failed",
                           detail={"rc": rc, "log": str(log), "ip_dir": str(ip_dir)})
    source_verification = verify_generated(compile_dir / "generated", package=ip_dir)
    return {"ip_name": ip_name, "version": version, "part": part,
            "ip_dir": str(ip_dir),
            "component_xml": str(component),
            "component_xml_sha256": _sha_file(component),
            "pack_log": str(log), "pack_rc": rc, "model_sha256": model_sha,
            "source_root": str(source_root), "source_verification": source_verification}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.onnx_gen.export_model",
        description="Export a static INT8 ONNX graph as an NPU IP source package "
                    "with an explicit elaboration-time compute profile.")
    parser.add_argument("--model", help="static INT8 ONNX model path")
    parser.add_argument("--contract", help="quantization contract JSON path")
    parser.add_argument("--inputs", help="input manifest JSON path")
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="RTL source tree containing rtl/ and tools/npu_gen/ "
                             "(default: this project's mainline; recorded and hashed)")
    parser.add_argument("--activation-contract", type=Path,
                        help="explicit e1-lut-activation-contract-1 for LUT deployment/SiLU fusion; "
                             "Conv weights/bias must already be INT8/INT32")
    parser.add_argument("--out", help="fresh, empty export root directory")
    parser.add_argument("--compute-profile", default=DEFAULT_PUBLIC,
                        choices=list(PUBLIC_NAMES + INTERNAL_NAMES),
                        help="elaboration-time compute tier (default tn8)")
    parser.add_argument("--max-feature-width", default=None,
                        help="v0.2 build capacity for the feature-map width: 'auto' or a "
                             "positive integer (1..640 in this batch). Omitted keeps the "
                             "legacy compiler behaviour")
    parser.add_argument("--max-feature-height", default=None,
                        help="v0.2 build capacity for the feature-map height: 'auto' or a "
                             "positive integer (1..640). Resolved independently of width")
    parser.add_argument("--sim-ddr-limit-mib", type=int, default=None,
                        help="simulation/DDR-image budget in MiB for a spatial build "
                             "(default 32 when any spatial option is supplied)")
    parser.add_argument("--engine-pruning", choices=("none", "graph"), default="graph",
                        help="physical-instance pruning (default graph: elaborate only the "
                             "cores the lowered opcodes need; 'none' = historical full netlist)")
    parser.add_argument("--maxpool-backend", choices=("auto", "generic"), default="auto")
    parser.add_argument("--memory-profile", default="no_reuse",
                        choices=["no_reuse", "first_fit"],
                        help="existing memory-planning profile (default no_reuse)")
    parser.add_argument("--name", help="deterministic logical model name "
                                       "(same across tiers; not part of the instruction identity)")
    parser.add_argument("--run-id", default="cp_export",
                        help="deterministic run id (same across tiers)")
    parser.add_argument("--ip-version", default="0.1", help="IP VLNV version (default 0.1)")
    pack_group = parser.add_mutually_exclusive_group()
    pack_group.add_argument("--pack-ip", dest="pack_ip", action="store_true",
                            help="compatibility alias; IP packaging is the default")
    pack_group.add_argument("--no-pack-ip", dest="pack_ip", action="store_false",
                            help="skip Vivado packaging and emit compile/RTL artifacts only")
    parser.set_defaults(pack_ip=True)
    parser.add_argument("--vivado", default=DEFAULT_VIVADO,
                        help="Vivado 2021.1 batch executable")
    parser.add_argument("--list-compute-profiles", action="store_true",
                        help="list the registered compute profiles and their honest status")
    parser.add_argument("--require-qualified", default=None,
                        help="optional qualification JSON; the export refuses to run unless "
                             "the record binds this exact profile/definition/inputs and all "
                             "gates are pass")
    reference_cache_module.add_cli_arguments(parser)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_compute_profiles:
        print(json.dumps({"status": "pass", "schema": compute_profiles.SCHEMA,
                          "profiles": list_profiles(include_internal=True),
                          "uncertainty": "tn4/tn16 (and the new tn8 candidate) are "
                                         "unverified: functional/OOC/platform_5ns/"
                                         "three_piece are not_run until really executed"},
                         ensure_ascii=False, indent=2))
        return EXIT_OK
    missing = [flag for flag, value in (("--model", args.model), ("--contract", args.contract),
                                        ("--inputs", args.inputs), ("--out", args.out))
               if not value]
    if missing:
        print(json.dumps(_err("E_CP_SCHEMA", "missing required arguments",
                              {"missing": missing}), ensure_ascii=False))
        return EXIT_CONFIG
    for flag, value in (("--model", args.model), ("--contract", args.contract),
                        ("--inputs", args.inputs),
                        ("--activation-contract", args.activation_contract)):
        if value is None:
            continue
        if not Path(value).is_file():
            print(json.dumps(_err("E_CP_SCHEMA", "input file missing",
                                  {"flag": flag, "path": value}), ensure_ascii=False))
            return EXIT_CONFIG
    try:
        resolved = resolve(args.compute_profile)
    except CompileError as exc:
        print(json.dumps(_err(exc.code, exc.message, exc.detail), ensure_ascii=False))
        return _exit_code(exc.code)
    model_sha = _sha_file(Path(args.model))
    contract_sha = _sha_file(Path(args.contract))
    input_sha = _sha_file(Path(args.inputs))
    qualification = None
    if args.require_qualified:
        try:
            qualification = _require_qualified(args.require_qualified, resolved,
                                               model_sha, contract_sha, input_sha)
        except CompileError as exc:
            print(json.dumps(_err(exc.code, exc.message, exc.detail), ensure_ascii=False))
            return _exit_code(exc.code)
    try:
        out = _resolve_out(args.out)
        _guard_out(out)
        lock_path = out.parent / (out.name + ".lock")
        _guard_outputs(out, lock_path)
        resource = _resource_gate()
    except CompileError as exc:
        print(json.dumps(_err(exc.code, exc.message, exc.detail), ensure_ascii=False))
        return _exit_code(exc.code)

    logical_name = _sanitize_name(args.name or Path(args.model).stem)
    ip_name = "%s_%s" % (logical_name, resolved.name)
    out.mkdir(parents=True, exist_ok=False)
    command_digest = hashlib.sha256(json.dumps(
        {"argv": [str(args.model), str(args.contract), str(args.inputs),
                  resolved.name, args.memory_profile, logical_name, args.run_id,
                  str(args.source_root.resolve()), str(args.activation_contract),
                  args.max_feature_width, args.max_feature_height,
                  args.sim_ddr_limit_mib,
                  bool(args.pack_ip), args.ip_version, args.vivado, resolved.part,
                  bool(getattr(args, "reference_cache", False)),
                  getattr(args, "reference_cache_dir", None),
                  bool(getattr(args, "reference_cache_read_only", False))]},
        sort_keys=True).encode("utf-8")).hexdigest()
    lock = {
        "schema": "e1-compute-profile-export-lock-1",
        "owner": "export_model",
        "pid": os.getpid(),
        "created_local": _now(),
        "command": "export_model",
        "command_digest": command_digest,
        "out": str(out),
        "compute_profile": resolved.name,
        "memory_profile": args.memory_profile,
        "status": "running",
    }
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reference_cache = reference_cache_module.cache_from_args(args)
    started = _now()
    try:
        manifest = compile_graph(Path(args.model), Path(args.contract), Path(args.inputs),
                                 out / "compile", profile=args.memory_profile,
                                 compute_profile=resolved.name, name=logical_name,
                                 run_id=args.run_id, schema=COMPILE_SCHEMA,
                                 repo_root=args.source_root.resolve(),
                                 activation_contract_path=args.activation_contract,
                                 max_feature_width=_cli_spatial_axis(args.max_feature_width),
                                 max_feature_height=_cli_spatial_axis(args.max_feature_height),
                                 sim_ddr_limit_mib=_cli_int(args.sim_ddr_limit_mib),
                                 engine_pruning=args.engine_pruning,
                                 maxpool_backend=args.maxpool_backend,
                                 reference_cache=reference_cache)
        pack_row = None
        if args.pack_ip:
            pack_row = _pack_ip(out / "compile", out, ip_name, args.ip_version,
                                resolved.part, args.vivado, model_sha)
        export = {
            "schema": SCHEMA,
            "status": "unverified",
            "status_reason": "software compile succeeded; functional/OOC/platform 5ns/"
                             "three-piece gates are not_run for this candidate",
            "created_local": started,
            "finished_local": _now(),
            "model": {"path": str(Path(args.model).resolve()), "sha256": model_sha},
            "quant_contract": {"path": str(Path(args.contract).resolve()),
                               "sha256": contract_sha},
            "input_manifest": {"path": str(Path(args.inputs).resolve()),
                               "sha256": input_sha},
            "name": logical_name,
            "run_id": args.run_id,
            "memory_profile": args.memory_profile,
            "compute_profile": manifest["compute_profile"],
            "spatial_capacity": manifest.get("spatial_capacity"),
            "engine_capacity": manifest.get("engine_capacity"),
            "memory_limits": manifest.get("memory_limits"),
            "compile_manifest": "compile/compile_manifest.json",
            "compile_manifest_sha256": _sha_file(out / "compile" / "compile_manifest.json"),
            "reference_cache": manifest.get("reference_cache"),
            "generated_rtl_manifest": "compile/generated/rtl_manifest.json",
            "source_root": manifest.get("source_root"),
            "source_digest": manifest.get("source_digest"),
            "activation_contract": None if args.activation_contract is None else {
                "path": str(args.activation_contract.resolve()),
                "sha256": _sha_file(args.activation_contract)},
            "ip": pack_row,
            "qualification_ref": args.require_qualified,
            "qualification": None if qualification is None else {
                "path": args.require_qualified},
            "gates": {"compile": "pass", "rtl": "not_run", "functional": "not_run",
                      "ooc": "not_run", "platform_5ns": "not_run",
                      "three_piece": "not_run", "board": "not_run"},
            "warnings": [
                "tn4/tn16 (and the new tn8 candidate) are unverified until each gate is "
                "really executed on this exact source/model identity",
                "IP packaging and compile status=pass are not OOC, timing or board PASS",
                "32x32 is not supported and is rejected before RTL elaboration",
            ],
            "resource_preflight": resource,
            "lock": str(lock_path),
        }
        _json(out / "export_manifest.json", export)
        lock.update({"status": "completed", "finished_local": _now()})
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(json.dumps({"status": "pass", "out": str(out),
                          "compute_profile": resolved.name,
                          "tn": resolved.tn, "tm": resolved.tm,
                          "dsp_array_estimate": resolved.dsp_array_estimate,
                          "status_note": "unverified (functional/OOC/physical not_run)",
                          "ip": None if pack_row is None else pack_row["ip_dir"]},
                         ensure_ascii=False))
        return EXIT_OK
    except CompileError as exc:
        failure = {
            "schema": "e1-compute-profile-export-failure-1",
            "status": "fail",
            "created_local": started,
            "finished_local": _now(),
            "compute_profile": resolved.name,
            "memory_profile": args.memory_profile,
            "error": exc.to_dict(),
            "note": "failed export roots are not reusable: create a new --out",
        }
        _json(out / "failure.json", failure)
        lock.update({"status": "failed", "finished_local": _now(),
                     "error": exc.to_dict()})
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        print(json.dumps(_err(exc.code, exc.message, exc.detail), ensure_ascii=False))
        return _exit_code(exc.code)
    except Exception as exc:  # external tooling / unexpected
        failure = {"schema": "e1-compute-profile-export-failure-1", "status": "fail",
                   "created_local": started, "finished_local": _now(),
                   "error": {"code": "E_CP_EXTERNAL", "message": str(exc)}}
        _json(out / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False))
        return EXIT_EXTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
