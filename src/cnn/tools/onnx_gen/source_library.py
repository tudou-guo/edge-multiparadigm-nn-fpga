"""Resolve and verify the shared RTL library used by compile, simulation and IP.

The editable catalog is rtl/operator_library.json. Frozen legacy source trees
remain usable with --source-root; their bytes, never a historical PASS label,
are bound into each new compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .graph_ir import CompileError, E_CFG

ROOT = Path(__file__).resolve().parents[2]
CATALOG = "rtl/operator_library.json"
OPTIONAL_LEGACY_DIRS = ("rtl/ops/activation", "rtl/ops/depthwise", "rtl/ops/maxpool2d")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _path(root, relative):
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise CompileError(E_CFG, "library path escapes source root: %s" % relative)
    if not path.is_file():
        raise CompileError(E_CFG, "library source missing: %s" % path)
    return path


def load_library(source_root=None):
    root = Path(source_root or ROOT).resolve()
    catalog_path = root / CATALOG
    legacy = not catalog_path.is_file()
    if legacy and root == ROOT:
        raise CompileError(E_CFG, "mainline catalog missing: %s" % catalog_path)
    # Old sealed snapshots predate the catalog. Only their absent new-engine
    # directories are optional; every selected file is subsequently required.
    if legacy:
        catalog_path = ROOT / CATALOG
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    if catalog.get("schema") != "e1-operator-library-1":
        raise CompileError(E_CFG, "unsupported RTL library catalog schema")
    sources = list(catalog["sources"])
    if legacy:
        sources = [p for p in sources if not any(
            p.startswith(d + "/") and not (root / d).is_dir()
            for d in OPTIONAL_LEGACY_DIRS)]
    if len(sources) != len(set(sources)):
        raise CompileError(E_CFG, "duplicate source in RTL library catalog")
    required = sources + [catalog["template"]] + catalog["vendor_ip"]
    for relative in required:
        _path(root, relative)
    # Missing catalog entries must not silently drop a newly added operator.
    discovered = {p.relative_to(root).as_posix() for p in (root / "rtl/ops").rglob("*")
                  if p.is_file() and p.suffix in (".v", ".sv", ".vh")}
    missing = sorted(discovered - set(sources) - {row["path"] for row in catalog.get("auxiliary_sources", [])})
    if missing:
        raise CompileError(E_CFG, "operator RTL is not registered", detail={"files": missing})
    return {"root": root, "catalog": catalog, "catalog_path": catalog_path,
            "legacy_snapshot": legacy, "sources": sources}


def rtl_sources(source_root=None):
    return load_library(source_root)["sources"]


def source_identity(source_root=None):
    library = load_library(source_root)
    root, catalog = library["root"], library["catalog"]
    rows = [{"path": p, "sha256": sha(root / p)} for p in library["sources"]]
    template = {"path": catalog["template"], "sha256": sha(root / catalog["template"])}
    vendor = [{"path": p, "sha256": sha(root / p)} for p in catalog["vendor_ip"]]
    mapping = {row["path"]: row["sha256"] for row in rows + [template] + vendor}
    catalog_row = {"path": str(library["catalog_path"]),
                   "sha256": sha(library["catalog_path"]),
                   "legacy_snapshot": library["legacy_snapshot"]}
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    return {"source_root": root.as_posix(), "rtl_sources": library["sources"],
            "source_files": rows, "template": template, "vendor_ip": vendor,
            "library_catalog": catalog_row, "source_digest": digest}


def write_binding(generated, identity, generated_files):
    """Plain data for Tcl; never executable Tcl or shell code."""
    generated = Path(generated)
    rows = ["source\t%s\t%s" % (r["sha256"], r["path"])
            for r in identity["source_files"] + [identity["template"]] + identity["vendor_ip"]]
    for path in generated_files:
        path = Path(path)
        rows.append("generated\t%s\t%s" % (sha(path), path.name))
    if any("\n" in row or "\r" in row for row in rows):
        raise CompileError(E_CFG, "newline in library source path")
    (generated / "source_root.txt").write_text(identity["source_root"] + "\n", encoding="utf-8")
    (generated / "source_hashes.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def verify_generated(generated, package=None):
    generated = Path(generated).resolve()
    manifest = json.loads((generated / "rtl_manifest.json").read_text(encoding="utf-8-sig"))
    if not manifest.get("source_root") or not manifest.get("source_files"):
        raise CompileError(E_CFG, "compile has no bound source identity; recompile in a fresh directory")
    root = Path(manifest["source_root"]).resolve()
    rows = manifest["source_files"] + [manifest["template"]] + manifest["vendor_ip"]
    mapping = {}
    for row in rows:
        path = _path(root, row["path"])
        if sha(path) != row["sha256"]:
            raise CompileError(E_CFG, "bound RTL source changed: %s" % path)
        mapping[row["path"]] = row["sha256"]
    digest = hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()
    if digest != manifest["source_digest"]:
        raise CompileError(E_CFG, "RTL source digest is inconsistent")
    for filename, expected in manifest["files_sha256"].items():
        path = Path(filename)
        if not path.is_file() or sha(path) != expected:
            raise CompileError(E_CFG, "generated artifact changed: %s" % path)
    if Path((generated / "source_root.txt").read_text(encoding="utf-8").strip()).resolve() != root:
        raise CompileError(E_CFG, "Tcl source root differs from compiled source root")
    lines = (generated / "source_hashes.tsv").read_text(encoding="utf-8").splitlines()
    expected_lines = {"source\t%s\t%s" % (r["sha256"], r["path"]) for r in rows}
    expected_lines.update("generated\t%s\t%s" % (h, Path(p).name)
                          for p, h in manifest["files_sha256"].items()
                          if Path(p).name not in ("source_root.txt", "source_hashes.tsv"))
    if set(lines) != expected_lines:
        raise CompileError(E_CFG, "Tcl source hashes differ from compile manifest")
    package_count = 0
    optional_not_packaged = []
    if package is not None:
        package = Path(package)
        # v0.3 area pruning: sources.f (and therefore the packaged IP) contains
        # only the physical engines the graph needs. The pruned paths are listed
        # by rtl_manifest.sources_f_pruned; they are intentionally NOT packaged.
        pruned_paths = set(manifest.get("sources_f_pruned") or [])
        package_rows = [r for r in manifest["source_files"] + manifest["vendor_ip"]
                        if "sim_netlist" not in r["path"]
                        and r["path"] not in pruned_paths]
        for row in package_rows:
            name = Path(row["path"]).name
            if name.endswith(".v"):
                name = Path(name).stem + ".sv"
            matches = [p for p in package.rglob(name)
                       if not any(part.startswith(".pack_tmp_") for part in p.parts)]
            # Legacy multiplier black-box stub is not referenced by the MAC3
            # top. Vivado omits it during dependency-based IP packaging.
            if not matches and row["path"] == "rtl/ops/conv/conv_mul_dsp.v":
                optional_not_packaged.append(row["path"])
                continue
            if len(matches) != 1 or sha(matches[0]) != row["sha256"]:
                raise CompileError(E_CFG, "packaged source mismatch: %s" % row["path"])
            package_count += 1
        for key in ("npu_top", "mem"):
            path = Path(manifest[key])
            matches = [p for p in package.rglob(path.name)
                       if not any(part.startswith(".pack_tmp_") for part in p.parts)]
            if len(matches) != 1 or sha(matches[0]) != sha(path):
                raise CompileError(E_CFG, "packaged generated file mismatch: %s" % path.name)
            package_count += 1
    return {"status": "pass", "source_root": root.as_posix(), "source_digest": digest,
            "source_files": len(rows), "package_files": package_count,
            "optional_not_packaged": optional_not_packaged}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    listing.add_argument("--source-root", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--generated", type=Path, required=True)
    verify.add_argument("--package", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            lib = load_library(args.source_root)
            engines = [engine for engine in lib["catalog"]["engines"] if any(
                path.startswith(engine["directory"] + "/") for path in lib["sources"])]
            report = dict(status="pass", **source_identity(args.source_root), engines=engines)
        else:
            report = verify_generated(args.generated, args.package)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (CompileError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "fail", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
