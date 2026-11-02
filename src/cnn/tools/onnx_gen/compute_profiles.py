"""Compute-profile (elaboration-time Tn/Tm tier) registry and resolver.

Schema: 'e1-compute-profile-1' (new in the 2026-10-23 compute-profile work).
Definitions live in 'tools/onnx_gen/profiles/{tn4,tn8,tn16}.json' and, for
internal mixed tiles, 'tools/onnx_gen/profiles/internal/*.json'.

This module is pure Python: it never imports Vivado, never writes to disk and
never mutates the graph or memory plan.  It is the single rule source shared by
compile_pipeline / p3_compile / verilog_codegen, the export_model CLI and the
compute-profile checkers.

Fixed public tiers (user-facing, immutable for this round):

    tn4  = 4x4   (array estimate 9*4*ceil(4/2) = 72 DSP)
    tn8  = 8x8   (288 DSP)   -- default; historically the only 5 ns tier
    tn16 = 16x16 (1152 DSP)

Only MAC_IMPL=3 (DSP backend, DDR weights WEIGHT_MODE=0) may scale tiers.
32x32 and any tile axis outside 1..16 are rejected here, before the RTL lane
clamps in conv3x3_calc*/conv3x3_core could silently accept them.  The memory
planning profile (first_fit / no_reuse) is a separate, unchanged concept.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graph_ir import (
    CompileError,
    E_CP_BACKEND,
    E_CP_CAPACITY,
    E_CP_CONFLICT,
    E_CP_GEOMETRY,
    E_CP_IDENTITY,
    E_CP_SCHEMA,
    E_CP_TARGET,
    E_CP_TILE_GROUP,
    E_CP_TILE_SET,
    E_CP_UNKNOWN,
    E_CP_UNSUPPORTED_TILE,
    align_up,
)

SCHEMA = "e1-compute-profile-1"
PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
INTERNAL_DIR = PROFILE_DIR / "internal"

PUBLIC_NAMES: Tuple[str, ...] = ("tn4", "tn8", "tn16")
INTERNAL_NAMES: Tuple[str, ...] = ("tn4_tm8", "tn8_tm4", "tn4_tm16", "tn16_tm4",
                                   "tn8_tm16", "tn16_tm8")
DEFAULT_PUBLIC = "tn8"

PART = "xczu15eg-ffvb1156-2-i"
CLOCK_PERIOD_NS = 5.0
DSP_TOTAL = 3528
LUT_TOTAL = 341280
FF_TOTAL = 682560
LUTRAM_TOTAL = 184320
BRAM36_EQUIV_TOTAL = 744
URAM_TOTAL = 112

CAPACITY_LIMIT = {"max_w": 32, "max_cin": 16, "max_cout": 16, "max_ch": 16}
TILE_SET = (4, 8, 16)
MAX_TILE = 16

DEFINITION_FIELDS = {"schema", "name", "part", "clock_period_ns", "tn", "tm",
                     "mac_impl", "weight_mode", "capacity_limit", "budget",
                     "default_status", "historical_evidence", "internal"}
BUDGET_FIELDS = {"dsp_array_estimate", "dsp_total_max", "lut_max", "ff_max",
                 "lutram_max", "bram36_equiv_max", "uram_max"}
CAPACITY_FIELDS = {"max_w", "max_cin", "max_cout", "max_ch"}
DEVICE_BUDGET = {"dsp_total_max": DSP_TOTAL, "lut_max": LUT_TOTAL,
                 "ff_max": FF_TOTAL, "lutram_max": LUTRAM_TOTAL,
                 "bram36_equiv_max": BRAM36_EQUIV_TOTAL, "uram_max": URAM_TOTAL}


def _strict_int(value: Any, what: str, code: str = E_CP_SCHEMA) -> int:
    """Reject bool/float/str before any comparison (bool is an int subclass)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(code, "%s must be an integer (bool/float/numeric "
                                 "string are rejected)" % what,
                           detail={"field": what, "value": repr(value)})
    return int(value)


def _strict_number(value: Any, what: str, code: str = E_CP_SCHEMA) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(code, "%s must be a number" % what,
                           detail={"field": what, "value": repr(value)})
    return float(value)


def dsp_array_estimate(tn: int, tm: int) -> int:
    """Array-level DSP count of conv3x3_calc_balanced (9*Tn*ceil(Tm/2))."""
    return 9 * int(tn) * ((int(tm) + 1) // 2)


def single_weight_face(tn: int, tm: int, mac_impl: int) -> bool:
    """Mirror of conv3x3_core.v: SINGLE_WEIGHT_FACE (fast single-face path)."""
    return (int(mac_impl) != 1 and int(tn) <= 8 and int(tm) <= 8 and
            8 % int(tn) == 0 and 8 % int(tm) == 0)


def runtime_tile_skip(max_cin: int, max_cout: int, max_w: int, tn: int, tm: int,
                      mac_impl: int, weight_mode: int) -> bool:
    """Mirror of conv3x3_core.v: OPT_RUNTIME_TILE_SKIP applicability."""
    return (int(max_cin) == 16 and int(max_cout) == 16 and int(max_w) == 32 and
            int(tn) == 8 and int(tm) == 8 and int(mac_impl) == 3 and
            int(weight_mode) == 0)


def row_beats(channels: int, width: int) -> int:
    """align16(align8(C) * W) / 16 -- the ABI row capacity in 16-byte beats."""
    alloc8 = ((int(channels) + 7) // 8) * 8
    return align_up(alloc8 * int(width), 16) // 16


@dataclass(frozen=True)
class ResolvedComputeProfile:
    """Immutable, fully validated compute-profile configuration."""

    name: str
    definition_path: str
    definition_sha256: str
    tn: int
    tm: int
    mac_impl: int
    weight_mode: int
    part: str
    clock_period_ns: float
    capacity_limit: Dict[str, int]
    budget: Dict[str, int]
    default_status: str
    historical_evidence: Tuple[Dict[str, Any], ...]
    internal: bool

    @property
    def dsp_array_estimate(self) -> int:
        return dsp_array_estimate(self.tn, self.tm)

    @property
    def single_weight_face(self) -> bool:
        return single_weight_face(self.tn, self.tm, self.mac_impl)

    def definition_document(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "name": self.name,
            "part": self.part,
            "clock_period_ns": self.clock_period_ns,
            "tn": self.tn,
            "tm": self.tm,
            "mac_impl": self.mac_impl,
            "weight_mode": self.weight_mode,
            "capacity_limit": dict(self.capacity_limit),
            "budget": dict(self.budget),
            "default_status": self.default_status,
            "historical_evidence": [dict(row) for row in self.historical_evidence],
            "internal": self.internal,
        }


def _load_bytes(name: str) -> Tuple[Path, bytes]:
    if name in PUBLIC_NAMES:
        path = PROFILE_DIR / (name + ".json")
    elif name in INTERNAL_NAMES:
        path = INTERNAL_DIR / (name + ".json")
    else:
        raise CompileError(E_CP_UNKNOWN, "unknown compute profile %r" % (name,),
                           detail={"requested": name,
                                   "public": list(PUBLIC_NAMES),
                                   "internal": list(INTERNAL_NAMES)})
    try:
        return path, path.read_bytes()
    except OSError as exc:
        raise CompileError(E_CP_SCHEMA, "compute profile definition unreadable: %s" % exc,
                           detail={"name": name, "path": str(path)})


def load_definition(name: str) -> ResolvedComputeProfile:
    """Load + fully validate one repo-maintained definition file."""
    if not isinstance(name, str) or not name:
        raise CompileError(E_CP_SCHEMA, "compute profile name must be a non-empty string")
    path, raw = _load_bytes(name)
    sha = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise CompileError(E_CP_SCHEMA, "compute profile JSON is invalid: %s" % exc,
                           detail={"name": name, "path": str(path)})
    if not isinstance(document, dict):
        raise CompileError(E_CP_SCHEMA, "compute profile must be a JSON object",
                           detail={"name": name})
    missing = sorted(DEFINITION_FIELDS - set(document))
    unknown = sorted(set(document) - DEFINITION_FIELDS)
    if missing or unknown:
        raise CompileError(E_CP_SCHEMA, "compute profile fields invalid",
                           detail={"name": name, "missing": missing,
                                   "unknown": unknown})
    if document["schema"] != SCHEMA:
        raise CompileError(E_CP_SCHEMA, "compute profile schema must be %r" % SCHEMA,
                           detail={"name": name, "schema": document["schema"]})
    if document["name"] != name:
        raise CompileError(E_CP_SCHEMA, "definition name does not match its registry name",
                           detail={"name": name, "definition_name": document["name"]})
    internal = document["internal"]
    if not isinstance(internal, bool):
        raise CompileError(E_CP_SCHEMA, "internal must be a boolean",
                           detail={"name": name})
    if internal != (name in INTERNAL_NAMES):
        raise CompileError(E_CP_SCHEMA, "internal flag does not match registry location",
                           detail={"name": name, "internal": internal})
    part = document["part"]
    if not isinstance(part, str) or part != PART:
        raise CompileError(E_CP_TARGET, "compute profile targets an unsupported part",
                           detail={"name": name, "part": part, "allowed": PART})
    clock = _strict_number(document["clock_period_ns"], "clock_period_ns")
    if clock != CLOCK_PERIOD_NS:
        raise CompileError(E_CP_TARGET, "compute profile targets an unsupported clock",
                           detail={"name": name, "clock_period_ns": clock,
                                   "allowed": CLOCK_PERIOD_NS})
    mac_impl = _strict_int(document["mac_impl"], "mac_impl")
    if mac_impl != 3:
        raise CompileError(E_CP_BACKEND, "compute profiles require MAC_IMPL=3",
                           detail={"name": name, "mac_impl": mac_impl})
    weight_mode = _strict_int(document["weight_mode"], "weight_mode")
    if weight_mode != 0:
        raise CompileError(E_CP_BACKEND, "compute profiles require WEIGHT_MODE=0 "
                                         "(DDR weights)",
                           detail={"name": name, "weight_mode": weight_mode})
    tn = _strict_int(document["tn"], "tn")
    tm = _strict_int(document["tm"], "tm")
    for axis, value in (("tn", tn), ("tm", tm)):
        if value <= 0 or value > MAX_TILE:
            raise CompileError(E_CP_UNSUPPORTED_TILE,
                               "%s=%d is outside the supported 1..%d range"
                               % (axis, value, MAX_TILE),
                               detail={"name": name, axis: value})
        if value not in TILE_SET:
            raise CompileError(E_CP_TILE_SET,
                               "%s=%d is not in the registered tile set %s"
                               % (axis, value, list(TILE_SET)),
                               detail={"name": name, axis: value,
                                       "tile_set": list(TILE_SET)})
    capacity = document["capacity_limit"]
    if not isinstance(capacity, dict):
        raise CompileError(E_CP_SCHEMA, "capacity_limit must be an object",
                           detail={"name": name})
    if set(capacity) != CAPACITY_FIELDS:
        raise CompileError(E_CP_SCHEMA, "capacity_limit fields invalid",
                           detail={"name": name, "fields": sorted(capacity)})
    limit: Dict[str, int] = {}
    for key in ("max_w", "max_cin", "max_cout", "max_ch"):
        value = _strict_int(capacity[key], "capacity_limit." + key)
        ceiling = CAPACITY_LIMIT[key]
        if key == "max_ch" and value not in (8, 16):
            raise CompileError(E_CP_SCHEMA, "capacity_limit.max_ch must be 8 or 16",
                               detail={"name": name, "max_ch": value})
        if value < 1 or value > ceiling:
            raise CompileError(E_CP_SCHEMA, "capacity_limit.%s out of range" % key,
                               detail={"name": name, key: value, "max": ceiling})
        limit[key] = value
    if limit["max_ch"] < max(limit["max_cin"], limit["max_cout"]):
        raise CompileError(E_CP_SCHEMA, "capacity_limit.max_ch must cover cin/cout",
                           detail={"name": name, "capacity_limit": limit})
    budget = document["budget"]
    if not isinstance(budget, dict) or set(budget) != BUDGET_FIELDS:
        raise CompileError(E_CP_SCHEMA, "budget fields invalid",
                           detail={"name": name,
                                   "fields": sorted(budget) if isinstance(budget, dict) else None})
    checked_budget: Dict[str, int] = {}
    for key in sorted(BUDGET_FIELDS):
        value = _strict_int(budget[key], "budget." + key)
        if value <= 0:
            raise CompileError(E_CP_SCHEMA, "budget.%s must be positive" % key,
                               detail={"name": name, key: value})
        checked_budget[key] = value
    expected_dsp = dsp_array_estimate(tn, tm)
    if checked_budget["dsp_array_estimate"] != expected_dsp:
        raise CompileError(E_CP_SCHEMA,
                           "budget.dsp_array_estimate does not match 9*Tn*ceil(Tm/2)",
                           detail={"name": name,
                                   "declared": checked_budget["dsp_array_estimate"],
                                   "expected": expected_dsp})
    for key, wanted in DEVICE_BUDGET.items():
        if checked_budget[key] != wanted:
            raise CompileError(E_CP_SCHEMA,
                               "budget.%s does not match the device limit" % key,
                               detail={"name": name, key: checked_budget[key],
                                       "expected": wanted})
    status = document["default_status"]
    if not isinstance(status, str) or status != "unverified":
        raise CompileError(E_CP_SCHEMA,
                           "repo definitions must stay default_status=unverified; "
                           "physical qualification is recorded in the run, not in the "
                           "definition file",
                           detail={"name": name, "default_status": status})
    evidence = document["historical_evidence"]
    if not isinstance(evidence, list):
        raise CompileError(E_CP_SCHEMA, "historical_evidence must be a list",
                           detail={"name": name})
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(evidence):
        if (not isinstance(row, dict) or not isinstance(row.get("ref"), str)
                or not isinstance(row.get("scope"), str)):
            raise CompileError(E_CP_SCHEMA,
                               "historical_evidence[%d] needs string ref/scope" % index,
                               detail={"name": name, "index": index})
        rows.append({"ref": row["ref"], "scope": row["scope"]})
    return ResolvedComputeProfile(
        name=name, definition_path=str(path).replace("\\", "/"),
        definition_sha256=sha, tn=tn, tm=tm, mac_impl=mac_impl,
        weight_mode=weight_mode, part=part, clock_period_ns=clock,
        capacity_limit=limit, budget=checked_budget, default_status=status,
        historical_evidence=tuple(rows), internal=internal)


def is_tile_token(name: str) -> bool:
    """True for the explicitly unsupported 32x32 spellings."""
    lowered = name.strip().lower()
    return lowered in ("tn32", "tn_32", "tn32x32", "32x32", "tile32")


def resolve(compute_profile: Any, *, tn: Any = None, tm: Any = None,
            mac_impl: Any = None,
            weight_mode: Any = None) -> Optional[ResolvedComputeProfile]:
    """Resolve an explicit compute profile and check explicit legacy parameters.

    Returns None exactly when compute_profile is None (legacy branch: the
    caller keeps the historical 8/8/3 defaults).  tn/tm/mac_impl/weight_mode
    are None when not supplied.

    Error precedence (technical plan section 4.4):
    type/schema -> name -> definition target/backend -> explicit legacy backend
    -> explicit parameter conflict -> tile numeric range -> registered tile set
    -> front-end/capacity/geometry (later, validate_build) -> group width.
    """
    if compute_profile is None:
        return None
    # 1) type/schema before any name lookup.
    for field_name, value in (("tn", tn), ("tm", tm), ("mac_impl", mac_impl),
                              ("weight_mode", weight_mode)):
        if value is not None:
            _strict_int(value, field_name)
    if isinstance(compute_profile, ResolvedComputeProfile):
        resolved = load_definition(compute_profile.name)
        if resolved.definition_sha256 != compute_profile.definition_sha256:
            raise CompileError(E_CP_IDENTITY,
                               "resolved compute-profile definition changed on disk",
                               detail={"name": compute_profile.name,
                                       "expected_sha256": compute_profile.definition_sha256,
                                       "actual_sha256": resolved.definition_sha256})
    elif isinstance(compute_profile, str):
        if not compute_profile:
            raise CompileError(E_CP_UNKNOWN, "compute profile name must not be empty")
        if is_tile_token(compute_profile):
            raise CompileError(E_CP_UNSUPPORTED_TILE,
                               "%s/32x32 is not supported: the ABI tile ceiling is 16x16"
                               % compute_profile,
                               detail={"requested": compute_profile})
        resolved = load_definition(compute_profile)
    else:
        raise CompileError(E_CP_SCHEMA,
                           "compute_profile must be a name or a resolved profile",
                           detail={"type": type(compute_profile).__name__})
    # 2) definition target/backend already validated in load_definition.
    # 3) explicit legacy backend must not select a non-scaling backend.
    if mac_impl is not None and int(mac_impl) != 3:
        raise CompileError(E_CP_BACKEND, "compute profiles only support MAC_IMPL=3",
                           detail={"name": resolved.name, "mac_impl": int(mac_impl)})
    if weight_mode is not None and int(weight_mode) != 0:
        raise CompileError(E_CP_BACKEND, "compute profiles only support WEIGHT_MODE=0",
                           detail={"name": resolved.name,
                                   "weight_mode": int(weight_mode)})
    # 4) explicit legacy parameters must agree, never be silently overridden.
    conflicts = {}
    for field_name, value, actual in (("tn", tn, resolved.tn),
                                      ("tm", tm, resolved.tm),
                                      ("mac_impl", mac_impl, resolved.mac_impl),
                                      ("weight_mode", weight_mode,
                                       resolved.weight_mode)):
        if value is not None and int(value) != int(actual):
            conflicts[field_name] = {"requested": int(value),
                                     "profile": int(actual)}
    if conflicts:
        raise CompileError(E_CP_CONFLICT,
                           "explicit parameters conflict with compute profile %r"
                           % resolved.name,
                           detail={"name": resolved.name, "conflicts": conflicts})
    return resolved


def validate_legacy_tile(tn: Any, tm: Any) -> Tuple[int, int]:
    """Universal safety guard for every ONNX entry point (old calls included).

    Historically tn/tm were clamped inside conv3x3_calc*; a raw 32 silently
    produced a smaller array.  New builds must reject instead of relying on the
    RTL clamp.  Only the numeric range is tightened here; the historical
    accepted-value semantics (any 1..16) are unchanged for legacy callers.
    """
    out = []
    for axis, value in (("tn", tn), ("tm", tm)):
        value = _strict_int(value, axis)
        if value <= 0 or value > MAX_TILE:
            raise CompileError(E_CP_UNSUPPORTED_TILE,
                               "%s=%d is outside the supported 1..%d range"
                               % (axis, value, MAX_TILE),
                               detail={axis: value, "max": MAX_TILE})
        out.append(value)
    return out[0], out[1]


def _build_capacity(build_capacity: Dict[str, Any]) -> Dict[str, int]:
    if not isinstance(build_capacity, dict):
        raise CompileError(E_CP_SCHEMA, "build capacity must be an object")
    missing = sorted(CAPACITY_FIELDS - set(build_capacity))
    if missing:
        raise CompileError(E_CP_SCHEMA, "build capacity is missing fields",
                           detail={"missing": missing})
    out: Dict[str, int] = {}
    for key in ("max_w", "max_cin", "max_cout", "max_ch"):
        out[key] = _strict_int(build_capacity[key], "build_capacity." + key)
    return out


def validate_build(resolved: ResolvedComputeProfile,
                   build_capacity: Dict[str, Any],
                   capacity_limit_override: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """Capacity / geometry / group-width checks for an explicit compute profile.

    build_capacity is the actual MAX_CIN/MAX_COUT/MAX_CH/MAX_W derived from the
    lowered graph (verilog_codegen.derive_capacities), never the profile limit.
    The profile bounds it, except for a v0.2 spatial build which passes the
    resolved spatial axis as a versioned override (recorded in the manifest).
    """
    capacity = _build_capacity(build_capacity)
    limit = dict(resolved.capacity_limit)
    if capacity_limit_override:
        for key, value in capacity_limit_override.items():
            if key not in CAPACITY_FIELDS:
                raise CompileError(E_CP_SCHEMA,
                                   "unknown capacity_limit override field %r" % (key,))
            limit[key] = _strict_int(value, "capacity_limit_override." + key)
    for key, label in (("max_cin", "MAX_CIN"), ("max_cout", "MAX_COUT"),
                       ("max_ch", "MAX_CH"), ("max_w", "MAX_W")):
        value = capacity[key]
        if value < 1:
            raise CompileError(E_CP_CAPACITY, "%s=%d must be positive" % (label, value),
                               detail={"name": resolved.name, key: value})
        if value > limit[key]:
            raise CompileError(E_CP_CAPACITY,
                               "%s=%d exceeds compute profile %r limit %d"
                               % (label, value, resolved.name, limit[key]),
                               detail={"name": resolved.name, key: value,
                                       "limit": limit[key]})
    if capacity["max_ch"] not in (8, 16):
        raise CompileError(E_CP_CAPACITY, "MAX_CH must be 8 or 16",
                           detail={"name": resolved.name,
                                   "max_ch": capacity["max_ch"]})
    in_beats = row_beats(capacity["max_cin"], capacity["max_w"])
    out_beats = row_beats(capacity["max_cout"], capacity["max_w"])
    if in_beats < 2 or out_beats < 2:
        raise CompileError(E_CP_GEOMETRY,
                           "derived row capacity is below 2 AXI beats "
                           "(in=%d, out=%d); this build geometry is not supported"
                           % (in_beats, out_beats),
                           detail={"name": resolved.name,
                                   "in_row_beats": in_beats,
                                   "out_row_beats": out_beats,
                                   "build_capacity": capacity})
    gi = 8 if capacity["max_cin"] <= 8 else 16
    go = 8 if capacity["max_cout"] <= 8 else 16
    if resolved.tn > gi or gi % resolved.tn:
        raise CompileError(E_CP_TILE_GROUP,
                           "Tn=%d does not divide the input group width %d "
                           "(MAX_CIN=%d)" % (resolved.tn, gi, capacity["max_cin"]),
                           detail={"name": resolved.name, "tn": resolved.tn,
                                   "GI": gi, "max_cin": capacity["max_cin"]})
    if resolved.tm > go or go % resolved.tm:
        raise CompileError(E_CP_TILE_GROUP,
                           "Tm=%d does not divide the output group width %d "
                           "(MAX_COUT=%d)" % (resolved.tm, go, capacity["max_cout"]),
                           detail={"name": resolved.name, "tm": resolved.tm,
                                   "GO": go, "max_cout": capacity["max_cout"]})
    return {
        "build_capacity": capacity,
        "group_width": {"in": gi, "out": go},
        "row_beats": {"in": in_beats, "out": out_beats},
        "single_weight_face": resolved.single_weight_face,
        "runtime_tile_skip": runtime_tile_skip(capacity["max_cin"],
                                               capacity["max_cout"],
                                               capacity["max_w"], resolved.tn,
                                               resolved.tm, resolved.mac_impl,
                                               resolved.weight_mode),
    }


def manifest_document(resolved: ResolvedComputeProfile,
                      derived: Dict[str, Any],
                      capacity_limit_override: Optional[Dict[str, int]] = None
                      ) -> Dict[str, Any]:
    """The compute_profile object written into compile/rtl manifests."""
    limit = dict(resolved.capacity_limit)
    if capacity_limit_override:
        limit.update({key: int(value) for key, value in capacity_limit_override.items()})
    document = {
        "name": resolved.name,
        "definition_sha256": resolved.definition_sha256,
        "tn": resolved.tn,
        "tm": resolved.tm,
        "mac_impl": resolved.mac_impl,
        "weight_mode": resolved.weight_mode,
        "capacity_limit": limit,
        "build_capacity": dict(derived["build_capacity"]),
        "GI": derived["group_width"]["in"],
        "GO": derived["group_width"]["out"],
        "row_beats": dict(derived["row_beats"]),
        "single_weight_face": bool(derived["single_weight_face"]),
        "runtime_tile_skip": bool(derived["runtime_tile_skip"]),
        "dsp_array_estimate": resolved.dsp_array_estimate,
        "part": resolved.part,
        "clock_period_ns": resolved.clock_period_ns,
        "status": "unverified",
        "qualification_ref": None,
    }
    if capacity_limit_override:
        # Additive, spatial-build only: a legacy document keeps its exact key set
        # so every pre-v0.2 manifest consumer/test is unaffected.
        document["capacity_limit_source"] = "explicit spatial override (v0.2)"
    return document


def list_profiles(include_internal: bool = False) -> List[Dict[str, Any]]:
    """Definition summary for export_model --list-compute-profiles."""
    names = list(PUBLIC_NAMES) + (list(INTERNAL_NAMES) if include_internal else [])
    rows = []
    for name in names:
        resolved = load_definition(name)
        rows.append({
            "name": resolved.name,
            "internal": resolved.internal,
            "tn": resolved.tn,
            "tm": resolved.tm,
            "mac_impl": resolved.mac_impl,
            "weight_mode": resolved.weight_mode,
            "dsp_array_estimate": resolved.dsp_array_estimate,
            "single_weight_face": resolved.single_weight_face,
            "part": resolved.part,
            "clock_period_ns": resolved.clock_period_ns,
            "capacity_limit": dict(resolved.capacity_limit),
            "status": "unverified",
            "historical_evidence": [dict(row) for row in resolved.historical_evidence],
            "definition_sha256": resolved.definition_sha256,
            "gates": {"functional": "not_run", "ooc": "not_run",
                      "platform_5ns": "not_run", "three_piece": "not_run"},
        })
    return rows


def public_names() -> Tuple[str, ...]:
    return PUBLIC_NAMES
