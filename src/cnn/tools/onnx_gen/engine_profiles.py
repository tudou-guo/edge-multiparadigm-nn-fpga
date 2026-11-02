"""Versioned per-engine configuration profiles (schema 'e1-engine-profile-1').

A profile is the *complete* elaboration-time engine selection for one compile:

    dense      : {"tn", "tm", "mac_impl"}      -- Dense Conv array tier
    depthwise  : {"pc", "tap_parallel", "rq_lanes"}
    activation : {"kind"}                      -- fixed LUT implementation

The module is pure Python: no graph mutation, no disk writes except reading an
explicit profile file.  It is the single rule source shared by
tools/onnx_gen/onnx_gen.py (compile CLI), compile_pipeline.compile_graph and
verilog_codegen.write_graph_rtl.

Legal implemented DW hardware matrix (frozen by the Lead, do not change here):

    DW_PC           in {4, 8, 16}
    DW_TAP_PARALLEL in {1}        (3 is assessed/deferred -> structurally rejected)
    DW_RQ_LANES     in {1, 2}     with PC=4 -> RQ=1, PC=8 -> RQ=1,
                                  PC=16 -> RQ=1 or 2
    default profile: dw_c8_t1_q1 = (8, 1, 1)

Unknown fields, wrong types, illegal combinations and conflicting explicit
selections are rejected with stable CompileError codes; there is no silent
fallback to the default profile and no silent adjustment of one DW field to fit
another.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .graph_ir import CompileError, canonical_json

SCHEMA = "e1-engine-profile-1"
DEFAULT_PROFILE = "dw_c8_t1_q1"

# Stable rejection codes (E_CFG-style family).
E_EP_SCHEMA = "E_EP_SCHEMA"
E_EP_IO = "E_EP_IO"
E_EP_UNKNOWN = "E_EP_UNKNOWN"
E_EP_DENSE = "E_EP_DENSE"
E_EP_DW_PC = "E_EP_DW_PC"
E_EP_DW_TAP = "E_EP_DW_TAP"
E_EP_DW_TAP_NOT_IMPLEMENTED = "E_EP_DW_TAP_NOT_IMPLEMENTED"
E_EP_DW_RQ = "E_EP_DW_RQ"
E_EP_DW_COMBO = "E_EP_DW_COMBO"
E_EP_ACTIVATION = "E_EP_ACTIVATION"
E_EP_CONFLICT = "E_EP_CONFLICT"

PROFILE_SCHEMA_FIELDS = frozenset({"schema", "name", "dense", "depthwise",
                                   "activation"})
DENSE_FIELDS = frozenset({"tn", "tm", "mac_impl"})
DW_FIELDS = frozenset({"pc", "tap_parallel", "rq_lanes"})
ACTIVATION_FIELDS = frozenset({"kind"})

DW_PC_LEGAL: Tuple[int, ...] = (4, 8, 16)
DW_TAP_IMPLEMENTED: Tuple[int, ...] = (1,)
DW_TAP_ASSESSED_NOT_IMPLEMENTED: Tuple[int, ...] = (3,)
DW_RQ_LEGAL: Tuple[int, ...] = (1, 2)
DW_TAP_COMBINATIONS = {4: (1,), 8: (1,), 16: (1, 2)}
ACTIVATION_KINDS = ("lut8",)
DENSE_TILE_MIN, DENSE_TILE_MAX = 1, 16
DENSE_MAC_IMPL = (1, 3)

DEFAULT_DENSE = {"tn": 8, "tm": 8, "mac_impl": 3}
DEFAULT_ACTIVATION = {"kind": "lut8"}

# Repo-maintained legal profiles.  The ids are the manifest-level identity and
# are shared with the frozen run contract contracts/dw_profiles.json.
BUILTIN_PROFILES: Dict[str, Dict[str, Any]] = {
    "dw_c4_t1_q1": {"dense": {"tn": 8, "tm": 8, "mac_impl": 3},
                    "depthwise": {"pc": 4, "tap_parallel": 1, "rq_lanes": 1},
                    "activation": {"kind": "lut8"}},
    "dw_c8_t1_q1": {"dense": {"tn": 8, "tm": 8, "mac_impl": 3},
                    "depthwise": {"pc": 8, "tap_parallel": 1, "rq_lanes": 1},
                    "activation": {"kind": "lut8"}},
    "dw_c16_t1_q1": {"dense": {"tn": 8, "tm": 8, "mac_impl": 3},
                     "depthwise": {"pc": 16, "tap_parallel": 1, "rq_lanes": 1},
                     "activation": {"kind": "lut8"}},
    "dw_c16_t1_q2": {"dense": {"tn": 8, "tm": 8, "mac_impl": 3},
                     "depthwise": {"pc": 16, "tap_parallel": 1, "rq_lanes": 2},
                     "activation": {"kind": "lut8"}},
}


def _strict_int(value: Any, what: str, code: str = E_EP_SCHEMA) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(code, "%s must be an integer (bool/float/string rejected)"
                           % what, detail={"field": what, "value": repr(value)})
    return int(value)


def _strict_str(value: Any, what: str, code: str = E_EP_SCHEMA) -> str:
    if not isinstance(value, str) or not value:
        raise CompileError(code, "%s must be a non-empty string" % what,
                           detail={"field": what, "value": repr(value)})
    return value


def _require_fields(document: Mapping[str, Any], fields: frozenset, what: str) -> None:
    missing = sorted(fields - set(document))
    unknown = sorted(set(document) - fields)
    if missing or unknown:
        raise CompileError(E_EP_SCHEMA, "%s fields invalid" % what,
                           detail={"missing": missing, "unknown": unknown})


def validate_dense(value: Any) -> Dict[str, int]:
    """Validate/normalize the dense section; never coerces a value."""
    if not isinstance(value, Mapping):
        raise CompileError(E_EP_SCHEMA, "dense must be an object",
                           detail={"type": type(value).__name__})
    _require_fields(value, DENSE_FIELDS, "dense")
    out = {"tn": _strict_int(value["tn"], "dense.tn", E_EP_DENSE),
           "tm": _strict_int(value["tm"], "dense.tm", E_EP_DENSE),
           "mac_impl": _strict_int(value["mac_impl"], "dense.mac_impl",
                                   E_EP_DENSE)}
    for axis in ("tn", "tm"):
        if out[axis] < DENSE_TILE_MIN or out[axis] > DENSE_TILE_MAX:
            raise CompileError(E_EP_DENSE,
                               "dense.%s=%d is outside the supported %d..%d range"
                               % (axis, out[axis], DENSE_TILE_MIN, DENSE_TILE_MAX),
                               detail={"axis": axis, "value": out[axis],
                                       "min": DENSE_TILE_MIN,
                                       "max": DENSE_TILE_MAX})
    if out["mac_impl"] not in DENSE_MAC_IMPL:
        raise CompileError(E_EP_DENSE,
                           "dense.mac_impl must be 1 (behavioral) or 3 (DSP)",
                           detail={"value": out["mac_impl"],
                                   "allowed": list(DENSE_MAC_IMPL)})
    return out


def validate_depthwise(value: Any) -> Dict[str, int]:
    """Validate the DW hardware triple, including the frozen legal matrix."""
    if not isinstance(value, Mapping):
        raise CompileError(E_EP_SCHEMA, "depthwise must be an object",
                           detail={"type": type(value).__name__})
    _require_fields(value, DW_FIELDS, "depthwise")
    pc = _strict_int(value["pc"], "depthwise.pc", E_EP_DW_PC)
    tap = _strict_int(value["tap_parallel"], "depthwise.tap_parallel")
    rq = _strict_int(value["rq_lanes"], "depthwise.rq_lanes", E_EP_DW_RQ)
    if pc not in DW_PC_LEGAL:
        raise CompileError(E_EP_DW_PC,
                           "DW_PC=%d is not implemented; legal values are %s"
                           % (pc, list(DW_PC_LEGAL)),
                           detail={"field": "depthwise.pc", "value": pc,
                                   "legal": list(DW_PC_LEGAL)})
    if tap in DW_TAP_ASSESSED_NOT_IMPLEMENTED:
        raise CompileError(
            E_EP_DW_TAP_NOT_IMPLEMENTED,
            "DW_TAP_PARALLEL=%d was assessed in this batch and is deliberately "
            "not implemented (exploration_incomplete, no third implementation/"
            "synthesis group); builds must request DW_TAP_PARALLEL=1" % tap,
            detail={"field": "depthwise.tap_parallel", "value": tap,
                    "implemented": list(DW_TAP_IMPLEMENTED),
                    "assessed_not_implemented": list(DW_TAP_ASSESSED_NOT_IMPLEMENTED)})
    if tap not in DW_TAP_IMPLEMENTED:
        raise CompileError(E_EP_DW_TAP,
                           "DW_TAP_PARALLEL=%d is not implemented; implemented "
                           "values are %s" % (tap, list(DW_TAP_IMPLEMENTED)),
                           detail={"field": "depthwise.tap_parallel", "value": tap,
                                   "implemented": list(DW_TAP_IMPLEMENTED)})
    if rq not in DW_RQ_LEGAL:
        raise CompileError(E_EP_DW_RQ,
                           "DW_RQ_LANES=%d is not implemented; legal values are %s"
                           % (rq, list(DW_RQ_LEGAL)),
                           detail={"field": "depthwise.rq_lanes", "value": rq,
                                   "legal": list(DW_RQ_LEGAL)})
    allowed = DW_TAP_COMBINATIONS[pc]
    if rq not in allowed:
        raise CompileError(
            E_EP_DW_COMBO,
            "DW_PC=%d with DW_RQ_LANES=%d is not a legal combination; "
            "DW_PC=%d allows RQ lanes %s" % (pc, rq, pc, list(allowed)),
            detail={"dw_pc": pc, "dw_rq_lanes": rq, "allowed_rq_lanes": list(allowed),
                    "combination_rule": "PC=4 -> RQ=1; PC=8 -> RQ=1; "
                                        "PC=16 -> RQ=1 or 2"})
    return {"pc": pc, "tap_parallel": tap, "rq_lanes": rq}


def validate_activation(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise CompileError(E_EP_SCHEMA, "activation must be an object",
                           detail={"type": type(value).__name__})
    _require_fields(value, ACTIVATION_FIELDS, "activation")
    kind = _strict_str(value["kind"], "activation.kind", E_EP_ACTIVATION)
    if kind not in ACTIVATION_KINDS:
        raise CompileError(E_EP_ACTIVATION,
                           "activation.kind=%r is not implemented; the mainline "
                           "activation engine is a fixed %s lookup implementation"
                           % (kind, ACTIVATION_KINDS[0]),
                           detail={"value": kind, "implemented": list(ACTIVATION_KINDS)})
    return {"kind": kind}


def validate_document(document: Any) -> Dict[str, Any]:
    """Validate one complete profile document; returns a normalized deep copy."""
    if not isinstance(document, Mapping):
        raise CompileError(E_EP_SCHEMA, "engine profile must be a JSON object",
                           detail={"type": type(document).__name__})
    _require_fields(document, PROFILE_SCHEMA_FIELDS, "engine profile")
    if document["schema"] != SCHEMA:
        raise CompileError(E_EP_SCHEMA, "engine profile schema must be %r" % SCHEMA,
                           detail={"schema": document["schema"], "expected": SCHEMA})
    name = _strict_str(document["name"], "name", E_EP_SCHEMA)
    return {"schema": SCHEMA,
            "name": name,
            "dense": validate_dense(document["dense"]),
            "depthwise": validate_depthwise(document["depthwise"]),
            "activation": validate_activation(document["activation"])}


def default_document(dense: Optional[Mapping[str, int]] = None,
                     depthwise: Optional[Mapping[str, int]] = None) -> Dict[str, Any]:
    """Complete default profile document (dw_c8_t1_q1) with optional overrides."""
    builtin = BUILTIN_PROFILES[DEFAULT_PROFILE]
    return validate_document({
        "schema": SCHEMA,
        "name": DEFAULT_PROFILE,
        "dense": dict(dense if dense is not None else builtin["dense"]),
        "depthwise": dict(depthwise if depthwise is not None
                          else builtin["depthwise"]),
        "activation": dict(builtin["activation"]),
    })


def _read_document(source: Union[str, Path]) -> Tuple[Dict[str, Any], str, Optional[str]]:
    """Load a profile file or builtin id; returns (document, ref, file_sha256)."""
    path = Path(source)
    raw = None
    try:
        raw = path.read_bytes()
    except OSError:
        raw = None
    if raw is None:
        text = str(source)
        if text in BUILTIN_PROFILES:
            document = {"schema": SCHEMA, "name": text,
                        "dense": dict(BUILTIN_PROFILES[text]["dense"]),
                        "depthwise": dict(BUILTIN_PROFILES[text]["depthwise"]),
                        "activation": dict(BUILTIN_PROFILES[text]["activation"])}
            return document, text, None
        if "/" not in text and "\\" not in text and not text.lower().endswith(".json"):
            raise CompileError(E_EP_UNKNOWN,
                               "unknown engine profile %r; builtin profiles are %s"
                               % (text, sorted(BUILTIN_PROFILES)),
                               detail={"requested": text,
                                       "builtin": sorted(BUILTIN_PROFILES)})
        raise CompileError(E_EP_IO, "engine profile file is unreadable",
                           detail={"path": text})
    sha = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompileError(E_EP_SCHEMA, "engine profile JSON is invalid: %s" % exc,
                           detail={"path": str(source)})
    return document, str(path).replace("\\", "/"), sha


def load(source: Any) -> Dict[str, Any]:
    """Load + validate one profile from a path, builtin id, dict or resolved value."""
    if isinstance(source, ResolvedEngineProfile):
        return validate_document(source.document())
    if isinstance(source, Mapping):
        return validate_document(source)
    if isinstance(source, (str, Path)):
        document, _ref, _sha = _read_document(source)
        return validate_document(document)
    raise CompileError(E_EP_SCHEMA,
                       "engine profile must be a JSON path, builtin id, dict or "
                       "resolved profile",
                       detail={"type": type(source).__name__})


@dataclass(frozen=True)
class ResolvedEngineProfile:
    """Immutable, fully validated engine configuration."""

    name: str
    dense: Dict[str, int]
    depthwise: Dict[str, int]
    activation: Dict[str, str]
    source_kind: str          # default | flags | builtin | file | dict | resolved
    source_ref: Optional[str]
    source_sha256: Optional[str]
    overrides: Dict[str, Any]

    @property
    def explicit_request(self) -> bool:
        return self.source_kind != "default"

    def document(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "name": self.name,
                "dense": dict(self.dense),
                "depthwise": dict(self.depthwise),
                "activation": dict(self.activation)}

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.document()).encode("utf-8")).hexdigest()

    def manifest_document(self) -> Dict[str, Any]:
        """The engine_profile object written into compile/rtl manifests."""
        requested: Dict[str, Any] = {"kind": self.source_kind,
                                      "ref": self.source_ref,
                                      "overrides": dict(self.overrides)}
        if self.source_sha256 is not None:
            requested["sha256"] = self.source_sha256
        return {
            "schema": SCHEMA,
            "requested": requested,
            "resolved": self.document(),
            "sha256": self.sha256(),
            "provenance": {
                "default_profile": DEFAULT_PROFILE,
                "legal_depthwise": {
                    "pc": list(DW_PC_LEGAL),
                    "tap_parallel_implemented": list(DW_TAP_IMPLEMENTED),
                    "tap_parallel_assessed_not_implemented":
                        list(DW_TAP_ASSESSED_NOT_IMPLEMENTED),
                    "rq_lanes": list(DW_RQ_LEGAL),
                    "combination_rule": "PC=4 -> RQ=1; PC=8 -> RQ=1; "
                                        "PC=16 -> RQ=1 or 2",
                },
            },
        }


def _builtin_match(dense: Mapping[str, int],
                   depthwise: Mapping[str, int],
                   activation: Mapping[str, str]) -> Optional[str]:
    for name, row in BUILTIN_PROFILES.items():
        if (dict(row["dense"]) == dict(dense)
                and dict(row["depthwise"]) == dict(depthwise)
                and dict(row["activation"]) == dict(activation)):
            return name
    return None


def _normalize_overrides(**kwargs) -> Dict[Tuple[str, str], Any]:
    field_map = {
        "dense_tn": ("dense", "tn"), "dense_tm": ("dense", "tm"),
        "dense_mac_impl": ("dense", "mac_impl"),
        "dw_pc": ("depthwise", "pc"),
        "dw_tap_parallel": ("depthwise", "tap_parallel"),
        "dw_rq_lanes": ("depthwise", "rq_lanes"),
    }
    return {field_map[key]: value for key, value in kwargs.items()
            if value is not None}


def _load_with_provenance(source: Any):
    """Return (document, source_kind, ref, sha256) for one explicit request."""
    if isinstance(source, ResolvedEngineProfile):
        # Preserve the original request provenance instead of relabelling it.
        return (source.document(), source.source_kind, source.source_ref,
                source.source_sha256)
    if isinstance(source, Mapping):
        return validate_document(source), "dict", None, None
    if isinstance(source, (str, Path)):
        document, ref, sha = _read_document(source)
        kind = "builtin" if ref in BUILTIN_PROFILES and sha is None else "file"
        return validate_document(document), kind, ref, sha
    raise CompileError(E_EP_SCHEMA,
                       "engine profile must be a JSON path, builtin id, dict or "
                       "resolved profile",
                       detail={"type": type(source).__name__})


def resolve(requested: Any = None, *,
            base_dense: Optional[Mapping[str, int]] = None,
            explicit_dense: Iterable[str] = (),
            dw_pc: Any = None, dw_tap_parallel: Any = None,
            dw_rq_lanes: Any = None) -> ResolvedEngineProfile:
    """Resolve defaults, an explicit profile and explicit convenience overrides.

    Rules (frozen technical plan section 5.1):

    * no explicit source  -> the mainline default profile dw_c8_t1_q1, but
      base_dense (the dense tier already resolved by the compute-profile
      machinery) is preserved, and explicit dw_* convenience values replace the
      default triple; the result is validated as a whole.
    * one explicit source -> the profile document; explicit convenience values
      must AGREE with it, otherwise E_EP_CONFLICT (never a silent override).
    * two explicit sources that disagree -> E_EP_CONFLICT (both for dense tiers
      and DW fields).
    * unknown / illegal values -> the stable E_EP_* codes above; no fallback.
    """
    overrides = _normalize_overrides(dw_pc=dw_pc,
                                     dw_tap_parallel=dw_tap_parallel,
                                     dw_rq_lanes=dw_rq_lanes)
    explicit_dense = set(explicit_dense or ())
    unknown_dense = explicit_dense - DENSE_FIELDS
    if unknown_dense:
        raise CompileError(E_EP_SCHEMA, "unknown explicit dense fields",
                           detail={"fields": sorted(unknown_dense)})
    # Validate the convenience values early (numeric types only); the
    # combination check still happens after the final triple is known.
    for (group, field), value in overrides.items():
        _strict_int(value, "%s.%s" % (group, field))

    if requested is not None:
        document, source_kind, source_ref, source_sha = _load_with_provenance(requested)
        conflicts: Dict[str, Any] = {}
        for field in sorted(explicit_dense):
            if base_dense is not None and field in base_dense:
                explicit_value = int(base_dense[field])
                if explicit_value != int(document["dense"][field]):
                    conflicts["dense." + field] = {
                        "requested": explicit_value,
                        "profile": int(document["dense"][field])}
        for (group, field), value in overrides.items():
            if int(value) != int(document[group][field]):
                conflicts["%s.%s" % (group, field)] = {
                    "requested": int(value),
                    "profile": int(document[group][field])}
        if conflicts:
            raise CompileError(E_EP_CONFLICT,
                               "explicit engine selection conflicts with the "
                               "engine profile; nothing was silently overridden",
                               detail={"profile": document["name"],
                                       "conflicts": conflicts})
        resolved = validate_document(document)
        return ResolvedEngineProfile(
            name=resolved["name"], dense=dict(resolved["dense"]),
            depthwise=dict(resolved["depthwise"]),
            activation=dict(resolved["activation"]),
            source_kind=source_kind, source_ref=source_ref,
            source_sha256=source_sha,
            overrides={("%s.%s" % key): int(value)
                       for key, value in overrides.items()})

    # No explicit profile: start from the default profile, keep the dense tier
    # that the compute-profile/legacy path already resolved.
    resolved = default_document(dense=base_dense)
    for (group, field), value in overrides.items():
        resolved[group][field] = int(value)
    resolved = validate_document(resolved)
    name = _builtin_match(resolved["dense"], resolved["depthwise"],
                          resolved["activation"])
    source_kind = "flags" if overrides else "default"
    if name is None:
        name = "custom:" + hashlib.sha256(
            canonical_json({"dense": resolved["dense"],
                            "depthwise": resolved["depthwise"],
                            "activation": resolved["activation"]}).encode()
        ).hexdigest()[:12]
    return ResolvedEngineProfile(
        name=name, dense=dict(resolved["dense"]),
        depthwise=dict(resolved["depthwise"]),
        activation=dict(resolved["activation"]),
        source_kind=source_kind, source_ref=None, source_sha256=None,
        overrides={("%s.%s" % key): int(value)
                   for key, value in overrides.items()})


def builtin_names() -> Tuple[str, ...]:
    return tuple(sorted(BUILTIN_PROFILES))


def list_profiles() -> Sequence[Dict[str, Any]]:
    """Definition summary for CLI listing and tests."""
    rows = []
    for name in builtin_names():
        resolved = resolve(name)
        rows.append({"id": name,
                     "dense": dict(resolved.dense),
                     "depthwise": dict(resolved.depthwise),
                     "activation": dict(resolved.activation),
                     "sha256": resolved.sha256(),
                     "default": name == DEFAULT_PROFILE})
    return rows
