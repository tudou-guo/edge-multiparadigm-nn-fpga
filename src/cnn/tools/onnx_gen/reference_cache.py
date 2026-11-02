"""Explicit, disable-by-default dependency cache for the P2 semantic reference.

Task book section 7: compile_graph computes the source-graph and lowered-graph
reference with graph_reference.evaluate_graph and then packs golden/parameter
blobs, plans addresses/instructions and generates the RTL.  A pure RTL fix (for
example the Copy D1 single-driver fix) does not change those reference values,
so they can be reused when every numeric dependency is identical.

What is cached -- layer 1 only, the "semantic reference":

* the source-graph and lowered-graph evaluate_graph outputs (per node-output
  tensor: shape/dtype/values plus the value sha256), and
* the source/lowered cross-check result that compile_pipeline performs on them.

Nothing else is cached.  Golden bins, parameter blobs, memory/address planning,
the instruction stream, the DDR image, the generated top/ROM, the RTL manifest
and the compile manifest are regenerated on every compile.  A reference hit
therefore can never hand old packing/layout/address bytes to a build.

Cache key (never filename/mtime/shape only, never the whole RTL source digest):

* ONNX model bytes and external weight files,
* loaded initializer identity (dtype/shape/per-tensor sha256),
* input manifest bytes plus the actual input file bytes,
* quant contract bytes and (when supplied) activation contract bytes,
* canonical source and lowered graph digests,
* sha256 of the numeric tool modules (frontend / lowering / reference / oracle),
* interpreter and schema environment.

Entry layout (non-pickle, deterministic):

  <dir>/entries/<kk>/<key>/entry.json         manifest of payload files
                          entry.sha256        sha256 of entry.json
                          source_values.bin
                          lowered_values.bin

Publishing is atomic: the entry is written under <dir>/tmp/... and renamed into
place, so concurrent compiles of the same key publish exactly one complete entry.
A hit verifies entry.json, its sha256 sidecar and every payload file before use.
A corrupt/incomplete/mismatching entry is quarantined under <dir>/quarantine/
(evidence is kept) and recomputed; the miss reason is recorded in the compile
directory and in the compile manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_ir import CompileError, E_CFG
from .graph_reference import (TensorValue, attach_lut_scales, evaluate_graph,
                              evaluate_stats)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_KEY = "e1-reference-cache-key-1"
SCHEMA_ENTRY = "e1-reference-cache-entry-1"
SCHEMA_USE = "e1-reference-cache-use-1"
DEFAULT_RELATIVE_DIR = "build/onnx_e1/_cache/reference_v1"

# Numeric dependency closure: touching any of these files must invalidate the
# reference even when the canonical graph IR digest happens to stay identical
# (for example a reference rounding fix).  The cache module itself is included.
NUMERIC_DEPENDENCY_FILES = (
    "tools/onnx_gen/graph_reference.py",
    "tools/onnx_gen/exact_quant.py",
    "tools/onnx_gen/graph_ir.py",
    "tools/onnx_gen/onnx_frontend.py",
    "tools/onnx_gen/shape_infer.py",
    "tools/onnx_gen/op_registry.py",
    "tools/onnx_gen/lowering.py",
    "tools/onnx_gen/quant_contract.py",
    "tools/onnx_gen/activation_oracle.py",
    "tools/onnx_gen/activation_lut.py",
    "tools/onnx_gen/parameter_pack.py",
    "tools/onnx_gen/reference_cache.py",
)

_ENCODINGS = {"int8": 1, "int32": 4}


class CacheReadError(Exception):
    """An entry looked valid in the directory but could not be materialised."""


class _UnsupportedDtype(Exception):
    pass


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _now_local() -> str:
    return datetime.now().astimezone().isoformat()


def _round_seconds(value: float) -> float:
    return round(float(value), 4)


def _signed8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _value_sha256(values: Sequence[int]) -> str:
    """Exactly the sha256 graph_reference.evaluate_graph records per output."""
    return hashlib.sha256(bytes((int(value) & 0xFF) for value in values)).hexdigest()


def _encode_tensor(value: TensorValue) -> Tuple[bytes, str]:
    dtype = str(value.dtype)
    if dtype == "int8":
        return bytes((int(item) & 0xFF) for item in value.values), "int8"
    if dtype == "int32":
        return struct.pack("<%di" % len(value.values),
                           *[int(item) for item in value.values]), "int32"
    raise _UnsupportedDtype(dtype)


def _decode_tensor(blob: bytes, dtype: str, encoding: str) -> List[int]:
    if encoding == "int8" and dtype == "int8":
        return [_signed8(item) for item in blob]
    if encoding == "int32" and dtype == "int32":
        count = len(blob) // 4
        if count * 4 != len(blob):
            raise CacheReadError("int32 payload has a non-multiple-of-4 length")
        return list(struct.unpack("<%di" % count, blob))
    raise CacheReadError("unsupported tensor encoding %r for dtype %r" % (encoding, dtype))


def _preload_values(graph, initializers: Dict[str, Any],
                    inputs: Dict[str, Any]) -> Dict[str, TensorValue]:
    """Recreate evaluate_graph's initial values dict exactly.

    Only initializers that are graph tensors and the declared graph inputs are
    preloaded; node outputs are then filled from the cache.  This mirrors
    graph_reference.evaluate_graph, so the source/lowered comparison in
    compile_pipeline sees the same dict on a cache hit as on a cold compute.
    """
    values: Dict[str, TensorValue] = {}
    for name, meta in initializers.items():
        if name in graph.tensors:
            values[name] = TensorValue(tuple(int(v) for v in meta["shape"]),
                                       str(meta["dtype"]),
                                       [int(v) for v in meta["values"]])
    for name, meta in inputs.items():
        values[name] = TensorValue(tuple(int(v) for v in meta["shape"]), "int8",
                                   [_signed8(int(v)) for v in meta["values"]])
    return values


def _external_weight_files(model_path: Path) -> Dict[str, Any]:
    """Hash ONNX external-data weight files (they are numeric key material)."""
    try:
        import onnx
    except Exception as exc:
        return {"status": "unreadable", "reason": "onnx import failed: %s" % exc}
    try:
        model = onnx.load_model(str(model_path), load_external_data=False)
    except Exception as exc:
        return {"status": "unreadable", "reason": "%s: %s" % (type(exc).__name__, exc)}
    root = Path(model_path).resolve().parent
    rows: List[Dict[str, Any]] = []
    seen = set()
    for tensor in list(model.graph.initializer):
        if int(getattr(tensor, "data_location", 0)) != int(onnx.TensorProto.EXTERNAL):
            continue
        for field in tensor.external_data:
            if str(field.key) != "location":
                continue
            location = str(field.value)
            if location in seen:
                continue
            seen.add(location)
            path = (root / location).resolve()
            if not path.is_file():
                return {"status": "missing",
                        "reason": "external weight file missing: %s" % location,
                        "files": rows}
            rows.append({"location": location, "bytes": int(path.stat().st_size),
                         "sha256": _sha256_file(path)})
    return {"status": "ok", "files": rows,
            "digest": _sha256_bytes(_canonical(rows))}


def _graph_key_rows(graph) -> Dict[str, Any]:
    return {"digest": graph.digest(), "nodes": len(graph.nodes),
            "outputs": list(graph.outputs)}


def _sha256_row_values(shas: Sequence[str]) -> Dict[str, Any]:
    rows = list(shas)
    return {"rows": len(rows), "digest": _sha256_bytes(_canonical(rows))}
class ReferenceSession:
    """One reference resolution for one compile.

    source_values / lowered_values behave like the dicts returned by
    graph_reference.evaluate_graph: {"values": {...}, "node_outputs": {...}}.
    commit(compare) is called by compile_pipeline after its unchanged source/lowered
    cross-check; a cold compute publishes the entry only after that check passed.
    """

    def __init__(self, cache, key: Optional[Dict[str, Any]],
                 cache_state: Dict[str, Any], stage: Dict[str, float],
                 source_values: Dict[str, Any], lowered_values: Dict[str, Any],
                 from_cache: bool, compare_from_cache: Optional[Dict[str, Any]],
                 provenance: Dict[str, Any], mode: str):
        self.cache = cache
        self.key = key
        self.cache_state = cache_state
        self.stage = stage
        self.source_values = source_values
        self.lowered_values = lowered_values
        self.from_cache = bool(from_cache)
        self.compare_from_cache = compare_from_cache
        self.provenance = dict(provenance)
        self.mode = mode
        self.committed = False
        self.compare: Optional[Dict[str, Any]] = None
        self.commit_record: Dict[str, Any] = {}

    def commit(self, compare: Dict[str, Any]) -> None:
        if self.committed:
            raise RuntimeError("reference session committed twice")
        self.committed = True
        self.compare = compare
        if self.from_cache:
            if self.compare_from_cache is None:
                raise CompileError(E_CFG,
                                   "reference cache entry has no recorded "
                                   "source/lowered check")
            if (self.compare_from_cache.get("common_mismatches")
                    != compare.get("common_mismatches")
                    or self.compare_from_cache.get("output_mismatches")
                    != compare.get("output_mismatches")):
                raise CompileError(
                    E_CFG,
                    "reference cache replay disagrees with the recorded "
                    "source/lowered check; refusing to continue with a possibly "
                    "corrupt entry (rerun with --reference-cache off, or delete "
                    "the entry)",
                    detail={"entry": self.cache_state.get("entry")})
            return
        if self.cache is None or self.key is None or not self.key.get("cacheable"):
            return
        started = time.perf_counter()
        try:
            record = self.cache.store(
                self.key, self.source_values, self.lowered_values, compare,
                provenance=self.provenance,
                graph_digests=self.provenance.get("graph_digests") or {})
        except Exception as exc:
            record = {"published": False,
                      "reason": "store_error:%s: %s" % (type(exc).__name__, exc)}
        self.stage["store_seconds"] = _round_seconds(time.perf_counter() - started)
        record["seconds"] = self.stage["store_seconds"]
        self.commit_record = record
        if record.get("published"):
            self.cache_state["origin"] = "computed_cached"
            self.cache_state["entry"] = record.get("entry")
        elif record.get("concurrent"):
            self.cache_state["origin"] = "computed_concurrent_entry"
            self.cache_state["entry"] = record.get("entry")
    def summary(self) -> Dict[str, Any]:
        state = dict(self.cache_state)
        passed = (self.compare is not None
                  and not self.compare.get("common_mismatches")
                  and not self.compare.get("output_mismatches"))
        return {
            "schema": SCHEMA_USE,
            "enabled": bool(state.get("enabled")),
            "mode": self.mode,
            "dir": state.get("dir"),
            "hit": bool(state.get("hit")),
            "origin": state.get("origin"),
            "key": state.get("key"),
            "key_schema": SCHEMA_KEY,
            "key_dependencies": state.get("key_dependencies"),
            "entry": state.get("entry"),
            "verified_files": int(state.get("verified_files") or 0),
            "recompute_calls": int(state.get("recompute_calls") or 0),
            "evaluate_graph_calls": int(state.get("evaluate_graph_calls") or 0),
            "miss_reason": state.get("miss_reason"),
            "source_lowered_compare": "pass" if passed else "not_run",
            "stage": dict(self.stage),
            "store": dict(self.commit_record) if self.commit_record else None,
        }

    def document(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_USE,
            "use": self.summary(),
            "key_document": (None if self.key is None else self.key.get("document")),
            "stage_seconds": dict(self.stage),
        }


class ReferenceCache:
    """Filesystem reference cache rooted at a directory."""

    def __init__(self, directory, *, mode: str = "read-write", origin: str = "cli"):
        if mode not in ("read-write", "read-only"):
            raise ValueError("cache mode must be read-write or read-only")
        self.directory = Path(directory).resolve()
        self.mode = mode
        self.origin = origin
        self.force_refresh = False

    def relative_directory(self) -> str:
        try:
            return self.directory.relative_to(ROOT).as_posix()
        except ValueError:
            return str(self.directory)

    def _entry_path(self, key_sha: str) -> Path:
        return self.directory / "entries" / key_sha[:2] / key_sha

    def _relative(self, path) -> str:
        try:
            return Path(path).resolve().relative_to(self.directory).as_posix()
        except ValueError:
            return str(path)

    # -- key ---------------------------------------------------------------
    def key_for(self, *, model_path: Path, contract_path: Path,
                input_manifest_path: Path,
                activation_contract_path: Optional[Path],
                inputs: Dict[str, Any], initializers: Dict[str, Any],
                source, lowered) -> Dict[str, Any]:
        document = self.build_key_document(
            model_path=Path(model_path), contract_path=Path(contract_path),
            input_manifest_path=Path(input_manifest_path),
            activation_contract_path=(None if activation_contract_path is None
                                      else Path(activation_contract_path)),
            inputs=inputs, initializers=initializers, source=source, lowered=lowered)
        sha = _sha256_bytes(_canonical(document))
        external = ((document.get("dependencies") or {}).get("onnx") or {}
                    ).get("external_data") or {}
        cacheable = external.get("status") in (None, "ok")
        return {"document": document, "sha256": sha, "cacheable": bool(cacheable),
                "reason": (None if cacheable
                           else "dependency_unreadable:%s"
                                % (external.get("reason") or external.get("status")))}

    def build_key_document(self, *, model_path: Path, contract_path: Path,
                           input_manifest_path: Path,
                           activation_contract_path: Optional[Path],
                           inputs: Dict[str, Any], initializers: Dict[str, Any],
                           source, lowered) -> Dict[str, Any]:
        dependencies: Dict[str, Any] = {}
        model_path = Path(model_path)
        dependencies["onnx"] = {
            "sha256": _sha256_file(model_path),
            "bytes": int(model_path.stat().st_size),
            "external_data": _external_weight_files(model_path),
        }
        initializer_rows = [
            {"name": str(name), "dtype": str(row.get("dtype")),
             "shape": [int(v) for v in (row.get("shape") or [])],
             "sha256": str(row.get("sha256"))}
            for name, row in sorted(initializers.items())
        ]
        dependencies["initializers"] = {
            "count": len(initializer_rows),
            "digest": _sha256_bytes(_canonical(initializer_rows)),
        }
        contract_path = Path(contract_path)
        dependencies["quant_contract"] = {
            "sha256": _sha256_file(contract_path),
            "bytes": int(contract_path.stat().st_size)}
        if activation_contract_path is None:
            dependencies["activation_contract"] = None
        else:
            activation_contract_path = Path(activation_contract_path)
            dependencies["activation_contract"] = {
                "sha256": _sha256_file(activation_contract_path),
                "bytes": int(activation_contract_path.stat().st_size)}
        input_rows = []
        for name, row in sorted(inputs.items()):
            input_rows.append({"name": str(name),
                               "shape": [int(v) for v in row["shape"]],
                               "format": str(row.get("format")),
                               "sha256": str(row.get("sha256")),
                               "bytes": int(row.get("bytes") or 0)})
        input_manifest_path = Path(input_manifest_path)
        dependencies["inputs"] = {
            "manifest_sha256": _sha256_file(input_manifest_path),
            "manifest_bytes": int(input_manifest_path.stat().st_size),
            "rows": input_rows}
        dependencies["graphs"] = {"source": _graph_key_rows(source),
                                  "lowered": _graph_key_rows(lowered)}
        code_rows = []
        for relative in NUMERIC_DEPENDENCY_FILES:
            path = ROOT / relative
            code_rows.append({"path": relative,
                              "sha256": (_sha256_file(path)
                                         if path.is_file() else None)})
        dependencies["code"] = {"files": code_rows,
                                "digest": _sha256_bytes(_canonical(code_rows))}
        dependencies["environment"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "byteorder": sys.byteorder,
            "entry_schema": SCHEMA_ENTRY,
            "key_schema": SCHEMA_KEY}
        return {"schema": SCHEMA_KEY, "dependencies": dependencies}

    # -- lookup / verification --------------------------------------------
    def lookup(self, key: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        key_sha = key["sha256"]
        if getattr(self, "force_refresh", False):
            return {"hit": False, "entry": None,
                    "entry_dir": self._entry_path(key_sha),
                    "entry_relative": None, "verified_files": 0,
                    "reason": "refresh_requested",
                    "seconds": _round_seconds(time.perf_counter() - started)}
        entry_dir = self._entry_path(key_sha)
        entry, reason = self._verify_entry(key_sha)
        if entry is not None:
            return {"hit": True, "entry": entry, "entry_dir": entry_dir,
                    "entry_relative": self._relative(entry_dir),
                    "verified_files": len(entry.get("files") or []),
                    "reason": None,
                    "seconds": _round_seconds(time.perf_counter() - started)}
        if entry_dir.is_dir():
            if self.mode == "read-write":
                self._quarantine(entry_dir, reason or "invalid")
            else:
                reason = "%s(read_only)" % (reason or "invalid")
        return {"hit": False, "entry": None, "entry_dir": entry_dir,
                "entry_relative": None, "verified_files": 0,
                "reason": reason or "corrupt_entry",
                "seconds": _round_seconds(time.perf_counter() - started)}

    def _verify_entry(self, key_sha: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        entry_dir = self._entry_path(key_sha)
        if not entry_dir.is_dir():
            return None, "absent"
        entry_file = entry_dir / "entry.json"
        sha_file = entry_dir / "entry.sha256"
        try:
            raw = entry_file.read_bytes()
        except OSError:
            return None, "missing_entry_json"
        try:
            expected = sha_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None, "missing_entry_sha256"
        if _sha256_bytes(raw) != expected:
            return None, "entry_json_sha_mismatch"
        try:
            entry = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None, "entry_json_invalid"
        if entry.get("schema") != SCHEMA_ENTRY:
            return None, "entry_schema_mismatch"
        key = entry.get("key") or {}
        if key.get("sha256") != key_sha:
            return None, "key_mismatch"
        for row in entry.get("files") or []:
            relative = str(row.get("path"))
            path = entry_dir / relative
            if not path.is_file():
                return None, "missing_file:%s" % relative
            try:
                data = path.read_bytes()
            except OSError:
                return None, "unreadable_file:%s" % relative
            if (_sha256_bytes(data) != str(row.get("sha256"))
                    or len(data) != int(row.get("bytes") or -1)):
                return None, "corrupt_file:%s" % relative
        return entry, None

    def quarantine_entry(self, key_sha: str, reason: str) -> Optional[Path]:
        """Move an unusable entry aside (evidence kept) so a recompute can publish."""
        entry_dir = self._entry_path(key_sha)
        if not entry_dir.is_dir() or self.mode != "read-write":
            return None
        return self._quarantine(entry_dir, "entry_" + str(reason))

    def _quarantine(self, entry_dir, reason: str) -> Optional[Path]:
        stamp = "%s_%d_%s" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid(),
                              _sha256_bytes(str(reason).encode("utf-8"))[:8])
        target = self.directory / "quarantine" / (Path(entry_dir).name + "." + stamp)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(entry_dir, target)
            return target
        except OSError:
            return None

    # -- payload -----------------------------------------------------------
    def load_payload(self, entry: Dict[str, Any], entry_dir, *, source, lowered,
                     initializers: Dict[str, Any],
                     inputs: Dict[str, Any]) -> Dict[str, Any]:
        graphs = entry.get("graphs") or {}
        if (graphs.get("source") or {}).get("digest") != source.digest():
            raise CacheReadError("cached source graph digest differs")
        if (graphs.get("lowered") or {}).get("digest") != lowered.digest():
            raise CacheReadError("cached lowered graph digest differs")
        file_rows = {str(row["path"]): row for row in (entry.get("files") or [])}
        payload: Dict[str, bytes] = {}
        for name in ("source_values.bin", "lowered_values.bin"):
            row = file_rows.get(name)
            if row is None:
                raise CacheReadError("payload file %s is missing from the entry" % name)
            try:
                data = Path(entry_dir, name).read_bytes()
            except OSError as exc:
                raise CacheReadError("payload file %s unreadable: %s" % (name, exc))
            if _sha256_bytes(data) != str(row.get("sha256")):
                raise CacheReadError("payload file %s sha256 mismatch" % name)
            payload[name] = data
        result = {}
        for role, graph in (("source", source), ("lowered", lowered)):
            # Same reference-only metadata attachment a cold evaluate_graph does,
            # so the canonical graph digests match between cold and hot builds.
            attach_lut_scales(graph)
            values = _preload_values(graph, initializers, inputs)
            node_outputs: Dict[str, List[Dict[str, Any]]] = {}
            blob = payload["%s_values.bin" % role]
            rows = (graphs.get(role) or {}).get("rows")
            if not isinstance(rows, list):
                raise CacheReadError("cached %s rows are missing" % role)
            cursor = 0
            for node in graph.nodes:
                node_rows = []
                for output_name in node.outputs:
                    if cursor >= len(rows):
                        raise CacheReadError("%s payload has too few rows" % role)
                    row = rows[cursor]
                    cursor += 1
                    if row.get("node") != node.id or row.get("name") != output_name:
                        raise CacheReadError("%s row order mismatch at node %s"
                                             % (role, node.id))
                    offset = int(row.get("offset") or 0)
                    length = int(row.get("bytes") or 0)
                    segment = blob[offset:offset + length]
                    if len(segment) != length:
                        raise CacheReadError("%s row for %s escapes the payload"
                                             % (role, output_name))
                    dtype = str(row.get("dtype"))
                    values_list = _decode_tensor(segment, dtype,
                                                 str(row.get("encoding")))
                    if _value_sha256(values_list) != str(row.get("sha256_values")):
                        raise CacheReadError("%s values sha256 mismatch for %s"
                                             % (role, output_name))
                    shape = tuple(int(v) for v in row.get("shape") or [])
                    values[output_name] = TensorValue(shape, dtype, values_list)
                    node_rows.append({"name": output_name, "shape": list(shape),
                                      "dtype": dtype, "values": list(values_list),
                                      "sha256_values": str(row.get("sha256_values"))})
                node_outputs[node.id] = node_rows
            if cursor != len(rows):
                raise CacheReadError("%s payload has extra rows" % role)
            result[role] = {"values": values, "node_outputs": node_outputs}
        return {"source_values": result["source"],
                "lowered_values": result["lowered"]}

    def store(self, key: Dict[str, Any], source_values: Dict[str, Any],
              lowered_values: Dict[str, Any], compare: Dict[str, Any],
              *, provenance: Dict[str, Any],
              graph_digests: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        if self.mode != "read-write":
            return {"published": False, "reason": "read_only",
                    "seconds": _round_seconds(time.perf_counter() - started)}
        payload, graphs, values_identity = self._encode_payload(source_values,
                                                                lowered_values)
        for role, digest in graph_digests.items():
            if role in graphs:
                graphs[role]["digest"] = str(digest)
        key_sha = key["sha256"]
        entry = {
            "schema": SCHEMA_ENTRY,
            "created_local": _now_local(),
            "created_unix": int(time.time()),
            "key": {"schema": SCHEMA_KEY, "sha256": key_sha,
                    "dependencies": key["document"]["dependencies"]},
            "graphs": graphs,
            "compare": {
                "status": "pass",
                "common_mismatches": list(compare.get("common_mismatches") or []),
                "output_mismatches": list(compare.get("output_mismatches") or [])},
            "values": values_identity,
            "provenance": dict(provenance),
            "files": [],
        }
        files: Dict[str, bytes] = {}
        for name in sorted(payload):
            data = payload[name]
            files[name] = data
            entry["files"].append({"path": name, "bytes": len(data),
                                   "sha256": _sha256_bytes(data)})
        tmp_parent = self.directory / "tmp"
        final = self._entry_path(key_sha)
        tmp = tmp_parent / ("%d_%s" % (os.getpid(), uuid.uuid4().hex))
        tmp_parent.mkdir(parents=True, exist_ok=True)
        result: Dict[str, Any]
        try:
            tmp.mkdir(parents=True)
            for name in sorted(files):
                (tmp / name).write_bytes(files[name])
            raw = _canonical(entry) + b"\n"
            (tmp / "entry.json").write_bytes(raw)
            (tmp / "entry.sha256").write_text(_sha256_bytes(raw) + "\n",
                                              encoding="ascii")
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.rename(tmp, final)
                result = {"published": True, "reason": None,
                          "entry": self._relative(final)}
            except OSError:
                existing, reason = self._verify_entry(key_sha)
                if existing is not None:
                    result = {"published": False, "concurrent": True,
                              "reason": "concurrent_publish",
                              "entry": self._relative(final)}
                else:
                    if Path(final).is_dir():
                        self._quarantine(final, reason or "invalid_before_publish")
                    os.rename(tmp, final)
                    result = {"published": True,
                              "reason": "replaced_invalid_entry",
                              "entry": self._relative(final)}
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
        result["seconds"] = _round_seconds(time.perf_counter() - started)
        return result

    def _encode_payload(self, source_values: Dict[str, Any],
                        lowered_values: Dict[str, Any]
                        ) -> Tuple[Dict[str, bytes], Dict[str, Any], Dict[str, Any]]:
        payload: Dict[str, bytes] = {}
        graphs: Dict[str, Any] = {}
        values_identity: Dict[str, Any] = {}
        for role, result in (("source", source_values), ("lowered", lowered_values)):
            values = result["values"]
            node_outputs = result["node_outputs"]
            blob = bytearray()
            rows: List[Dict[str, Any]] = []
            for node_id in node_outputs:
                for row in node_outputs[node_id]:
                    name = row["name"]
                    tensor = values[name]
                    data, encoding = _encode_tensor(tensor)
                    rows.append({"node": node_id, "name": name,
                                 "shape": [int(v) for v in tensor.shape],
                                 "dtype": str(tensor.dtype), "encoding": encoding,
                                 "offset": len(blob), "bytes": len(data),
                                 "sha256_values": _value_sha256(tensor.values),
                                 "value_sha256": _sha256_bytes(data)})
                    blob.extend(data)
            payload["%s_values.bin" % role] = bytes(blob)
            graphs[role] = {"rows": rows, "row_count": len(rows),
                            "values_bytes": len(blob), "digest": None}
            values_identity[role] = _sha256_row_values(
                row["sha256_values"] for row in rows)
        return payload, graphs, values_identity


def _compact_dependencies(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Small, manifest-friendly projection of the full key document."""
    deps = ((document or {}).get("dependencies") or {})
    onnx = deps.get("onnx") or {}
    external = onnx.get("external_data") or {}
    graphs = deps.get("graphs") or {}
    return {
        "model_sha256": onnx.get("sha256"),
        "model_bytes": onnx.get("bytes"),
        "external_weights_digest": external.get("digest"),
        "initializers_digest": (deps.get("initializers") or {}).get("digest"),
        "quant_contract_sha256": (deps.get("quant_contract") or {}).get("sha256"),
        "activation_contract_sha256":
            (deps.get("activation_contract") or {}).get("sha256")
            if deps.get("activation_contract") else None,
        "input_manifest_sha256":
            (deps.get("inputs") or {}).get("manifest_sha256"),
        "input_files": [
            {"name": row.get("name"), "sha256": row.get("sha256")}
            for row in (deps.get("inputs") or {}).get("rows") or []],
        "source_graph_digest": (graphs.get("source") or {}).get("digest"),
        "lowered_graph_digest": (graphs.get("lowered") or {}).get("digest"),
        "code_digest": (deps.get("code") or {}).get("digest"),
        "environment": deps.get("environment"),
    }


def open_reference(cache: Optional[ReferenceCache], *, model_path: Path,
                   contract_path: Path, input_manifest_path: Path,
                   activation_contract_path: Optional[Path], source, lowered,
                   initializers: Dict[str, Any], inputs: Dict[str, Any],
                   run_id: str = "", graph_name: str = "") -> ReferenceSession:
    """Resolve the semantic reference for one compile, using the cache if allowed.

    A cache problem never fails the compile: the values are recomputed and the
    miss reason is recorded.  evaluate_graph calls are counted through
    graph_reference.evaluate_stats so a hot hit can be proven to have made zero
    calls (recompute_calls == 0).
    """
    stats_before = evaluate_stats()
    # Canonical digests at the cache boundary (before evaluate_graph runs and
    # before later planning stages attach storage metadata): the key, the entry
    # and the hot-hit verification must all use this same pipeline position.
    boundary_digests = {"source": source.digest(), "lowered": lowered.digest()}
    stage: Dict[str, float] = {}
    state: Dict[str, Any] = {
        "enabled": cache is not None,
        "hit": False,
        "key": None,
        "key_dependencies": None,
        "entry": None,
        "origin": "disabled" if cache is None else "computed",
        "verified_files": 0,
        "miss_reason": None,
        "dir": None,
        "recompute_calls": 0,
        "evaluate_graph_calls": 0,
    }
    key: Optional[Dict[str, Any]] = None
    hit_attempted = False
    source_values: Optional[Dict[str, Any]] = None
    lowered_values: Optional[Dict[str, Any]] = None
    compare_from_cache: Optional[Dict[str, Any]] = None
    mode = "disabled" if cache is None else cache.mode
    if cache is not None:
        state["dir"] = cache.relative_directory()
        probe_started = time.perf_counter()
        try:
            started = time.perf_counter()
            key = cache.key_for(
                model_path=model_path, contract_path=contract_path,
                input_manifest_path=input_manifest_path,
                activation_contract_path=activation_contract_path,
                inputs=inputs, initializers=initializers,
                source=source, lowered=lowered)
            stage["key_seconds"] = _round_seconds(time.perf_counter() - started)
            state["key"] = key["sha256"]
            state["key_dependencies"] = _compact_dependencies(key["document"])
            if not key.get("cacheable"):
                state["miss_reason"] = str(key.get("reason") or "not_cacheable")
            else:
                started = time.perf_counter()
                lookup = cache.lookup(key)
                stage["lookup_seconds"] = _round_seconds(
                    time.perf_counter() - started)
                if lookup["hit"]:
                    hit_attempted = True
                    started = time.perf_counter()
                    payload = cache.load_payload(
                        lookup["entry"], lookup["entry_dir"], source=source,
                        lowered=lowered, initializers=initializers, inputs=inputs)
                    stage["materialize_seconds"] = _round_seconds(
                        time.perf_counter() - started)
                    source_values = payload["source_values"]
                    lowered_values = payload["lowered_values"]
                    compare_from_cache = dict(lookup["entry"].get("compare") or {})
                    state["hit"] = True
                    state["origin"] = "cache"
                    state["entry"] = lookup["entry_relative"]
                    state["verified_files"] = int(lookup["verified_files"])
                else:
                    state["origin"] = "computed"
                    state["miss_reason"] = str(lookup.get("reason") or "miss")
        except Exception as exc:
            source_values = None
            lowered_values = None
            compare_from_cache = None
            state["hit"] = False
            state["origin"] = "computed"
            state["entry"] = None
            state["verified_files"] = 0
            state["miss_reason"] = "cache_error:%s: %s" % (type(exc).__name__, exc)
            if hit_attempted and key is not None:
                # The entry passed the file-hash check but could not be
                # materialised (for example a stale digest boundary).  Keep it
                # as quarantine evidence and let the recompute republish.
                try:
                    cache.quarantine_entry(key["sha256"], "materialize_error")
                except Exception:
                    pass
        stage["probe_seconds"] = _round_seconds(time.perf_counter() - probe_started)
    if source_values is None or lowered_values is None:
        started = time.perf_counter()
        source_values = evaluate_graph(source, initializers, inputs)
        lowered_values = evaluate_graph(lowered, initializers, inputs)
        stage["compute_seconds"] = _round_seconds(time.perf_counter() - started)
    stats_after = evaluate_stats()
    state["recompute_calls"] = int(
        stats_after["evaluate_graph_calls"] - stats_before["evaluate_graph_calls"])
    state["evaluate_graph_calls"] = int(stats_after["evaluate_graph_calls"])
    provenance = {
        "run_id": str(run_id),
        "graph_name": str(graph_name),
        "graph_digests": boundary_digests,
        "writer": "tools/onnx_gen/reference_cache.py",
    }
    return ReferenceSession(cache=cache, key=key, cache_state=state, stage=stage,
                            source_values=source_values,
                            lowered_values=lowered_values,
                            from_cache=bool(state["hit"]),
                            compare_from_cache=compare_from_cache,
                            provenance=provenance, mode=mode)


def default_directory() -> Path:
    return ROOT / DEFAULT_RELATIVE_DIR


def make_cache(enabled: bool, directory=None, read_only: bool = False,
               origin: str = "cli") -> Optional[ReferenceCache]:
    if not enabled:
        return None
    path = Path(directory).expanduser() if directory else default_directory()
    if not path.is_absolute():
        path = (Path.cwd() / path)
    return ReferenceCache(path, mode="read-only" if read_only else "read-write",
                          origin=origin)


def add_cli_arguments(parser) -> None:
    """Register the explicit cache switches on a public compile/export parser."""
    group = parser.add_argument_group("reference dependency cache (task book 7)")
    group.add_argument(
        "--reference-cache", action="store_true", default=False,
        help="enable the explicit semantic-reference dependency cache. Default: "
             "off, which keeps the historical compile behaviour and manifest "
             "byte-for-byte unchanged")
    group.add_argument(
        "--reference-cache-dir", default=None,
        help="cache directory (default: %s; the directory is created on the "
             "first cold compute and only ever contains cache entries)" % (
                 DEFAULT_RELATIVE_DIR,))
    group.add_argument(
        "--reference-cache-read-only", action="store_true", default=False,
        help="only read verified cache entries; never publish new ones "
             "(CI/pinned-cache use)")
    group.add_argument(
        "--reference-cache-refresh", action="store_true", default=False,
        help="ignore an existing entry for this compile and recompute it "
             "(useful to re-derive evidence after a corruption check)")


def cache_from_args(args) -> Optional[ReferenceCache]:
    """Build a cache object from argparse namespace created by add_cli_arguments."""
    cache = make_cache(bool(getattr(args, "reference_cache", False)),
                       getattr(args, "reference_cache_dir", None),
                       bool(getattr(args, "reference_cache_read_only", False)))
    if cache is not None and bool(getattr(args, "reference_cache_refresh", False)):
        cache.force_refresh = True
    return cache
