"""Versioned target-platform descriptions (schema 'e1-target-1').

The first registered implementation target is 'project2_zu15eg_200mhz':

    part            : xczu15eg-ffvb1156-2-i
    platform        : project_2
    clock           : 5.000 ns (200 MHz)
    AXI data width  : 128 bit
    resource budget : xczu15eg part-family totals
    memory model    : nominal RTL bench contract; board DDR bandwidth unknown

Unknown target ids are rejected with a structured CompileError; there is no
silent fallback to the current part.  The part string is checked against a fixed
exact-match allowlist so "changing a string" can never pretend to add a board.

This module is pure Python and does not touch the graph, RTL or disk.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from . import compute_profiles
from .graph_ir import CompileError, canonical_json

SCHEMA = "e1-target-1"
DEFAULT_TARGET = "project2_zu15eg_200mhz"

E_TGT_SCHEMA = "E_TGT_SCHEMA"
E_TGT_UNKNOWN = "E_TGT_UNKNOWN"
E_TGT_PART = "E_TGT_PART"
E_TGT_CLOCK = "E_TGT_CLOCK"
E_TGT_AXI = "E_TGT_AXI"
E_TGT_BUDGET = "E_TGT_BUDGET"
E_TGT_MEMORY = "E_TGT_MEMORY"

# Exact-match allowlist: only parts with existing project evidence.
PART_ALLOWLIST: Tuple[str, ...] = ("xczu15eg-ffvb1156-2-i",)
TARGET_FIELDS = frozenset({"schema", "id", "part", "platform", "clock_ns",
                           "axi_data_width", "part_family", "resource_budget",
                           "memory_service_model", "evidence"})
BUDGET_FIELDS = frozenset({"lut", "ff", "lutram", "bram36_equiv", "dsp", "uram"})
MEMORY_FIELDS = frozenset({"mode", "board_ddr_bandwidth", "contract_ref", "note"})
PLATFORM_FIELDS = frozenset({"name", "project", "status_scope"})
EVIDENCE_FIELDS = frozenset({"ref", "scope"})
MEMORY_MODE = "nominal_bench_contract"

TARGET_REGISTRY: Dict[str, Dict[str, Any]] = {
    DEFAULT_TARGET: {
        "schema": SCHEMA,
        "id": DEFAULT_TARGET,
        "part": "xczu15eg-ffvb1156-2-i",
        "platform": {
            "name": "project_2",
            "project": "e1_project2",
            "status_scope": "target identity only; physical/board qualification "
                            "is recorded in the owning run, never inferred here",
        },
        "clock_ns": 5.0,
        "axi_data_width": 128,
        "part_family": "xczu15eg",
        "resource_budget": {
            "lut": compute_profiles.LUT_TOTAL,
            "ff": compute_profiles.FF_TOTAL,
            "lutram": compute_profiles.LUTRAM_TOTAL,
            "bram36_equiv": compute_profiles.BRAM36_EQUIV_TOTAL,
            "dsp": compute_profiles.DSP_TOTAL,
            "uram": compute_profiles.URAM_TOTAL,
        },
        "memory_service_model": {
            "mode": MEMORY_MODE,
            "board_ddr_bandwidth": "unknown",
            "contract_ref": "build/onnx_e1/depthwise_performance_20261024_201800_a1"
                            "/contracts/benchmark_contract.json"
                            "#/memory_service_model_nominal",
            "note": "RTL service model frozen in the benchmark contract; NOT a "
                    "measured board DDR bandwidth and never used to predict "
                    "board FPS",
        },
        "evidence": [
            {"ref": "build/onnx_e1/depthwise_performance_20261024_201800_a1/"
                    "contracts/benchmark_contract.json",
             "scope": "part, 5 ns clock, 128-bit AXI, nominal memory service model"},
            {"ref": "docs/E1_DEPTHWISE_PERFORMANCE_AND_PROFILES_PLAN_20261024.md"
                    " section 5.1",
             "scope": "target description fields and unknown-board-bandwidth rule"},
            {"ref": "build/onnx_e1/lut_activation_20261024_145300_6206",
             "scope": "historical project_2 route-only 5 ns evidence for the same "
                      "part/clock; not re-run and not extended by this batch"},
        ],
    },
}


def _strict_int(value: Any, what: str, code: str = E_TGT_SCHEMA) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompileError(code, "%s must be an integer" % what,
                           detail={"field": what, "value": repr(value)})
    return int(value)


def _strict_number(value: Any, what: str, code: str = E_TGT_SCHEMA) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(code, "%s must be a number" % what,
                           detail={"field": what, "value": repr(value)})
    return float(value)


def _strict_str(value: Any, what: str, code: str = E_TGT_SCHEMA) -> str:
    if not isinstance(value, str) or not value:
        raise CompileError(code, "%s must be a non-empty string" % what,
                           detail={"field": what, "value": repr(value)})
    return value


def _require_fields(document: Mapping[str, Any], fields: frozenset, what: str) -> None:
    missing = sorted(fields - set(document))
    unknown = sorted(set(document) - fields)
    if missing or unknown:
        raise CompileError(E_TGT_SCHEMA, "%s fields invalid" % what,
                           detail={"missing": missing, "unknown": unknown})


def validate_part(part: Any) -> str:
    """Exact-match part allowlist validation (no normalization, no aliases)."""
    part = _strict_str(part, "part", E_TGT_PART)
    if part not in PART_ALLOWLIST:
        raise CompileError(E_TGT_PART,
                           "target part %r is not in the fixed allowlist %s; a "
                           "string change does not add board support"
                           % (part, list(PART_ALLOWLIST)),
                           detail={"part": part, "allowlist": list(PART_ALLOWLIST)})
    return part


def validate_definition(document: Any) -> Dict[str, Any]:
    """Validate one complete target definition; returns a normalized copy."""
    if not isinstance(document, Mapping):
        raise CompileError(E_TGT_SCHEMA, "target definition must be a JSON object",
                           detail={"type": type(document).__name__})
    _require_fields(document, TARGET_FIELDS, "target")
    if document["schema"] != SCHEMA:
        raise CompileError(E_TGT_SCHEMA, "target schema must be %r" % SCHEMA,
                           detail={"schema": document["schema"], "expected": SCHEMA})
    target_id = _strict_str(document["id"], "id")
    part = validate_part(document["part"])
    platform = document["platform"]
    if not isinstance(platform, Mapping):
        raise CompileError(E_TGT_SCHEMA, "platform must be an object")
    _require_fields(platform, PLATFORM_FIELDS, "platform")
    platform_out = {key: _strict_str(platform[key], "platform." + key)
                    for key in sorted(PLATFORM_FIELDS)}
    clock = _strict_number(document["clock_ns"], "clock_ns", E_TGT_CLOCK)
    if clock <= 0.0:
        raise CompileError(E_TGT_CLOCK, "clock_ns must be positive",
                           detail={"clock_ns": clock})
    axi = _strict_int(document["axi_data_width"], "axi_data_width", E_TGT_AXI)
    if axi not in (64, 128, 256, 512):
        raise CompileError(E_TGT_AXI,
                           "axi_data_width=%d is not a supported AXI data width"
                           % axi, detail={"axi_data_width": axi,
                                          "supported": [64, 128, 256, 512]})
    family = _strict_str(document["part_family"], "part_family")
    budget = document["resource_budget"]
    if not isinstance(budget, Mapping):
        raise CompileError(E_TGT_BUDGET, "resource_budget must be an object")
    _require_fields(budget, BUDGET_FIELDS, "resource_budget")
    budget_out: Dict[str, int] = {}
    for key in sorted(BUDGET_FIELDS):
        value = _strict_int(budget[key], "resource_budget." + key, E_TGT_BUDGET)
        if value <= 0:
            raise CompileError(E_TGT_BUDGET,
                               "resource_budget.%s must be positive" % key,
                               detail={"field": key, "value": value})
        budget_out[key] = value
    memory = document["memory_service_model"]
    if not isinstance(memory, Mapping):
        raise CompileError(E_TGT_MEMORY, "memory_service_model must be an object")
    _require_fields(memory, MEMORY_FIELDS, "memory_service_model")
    memory_out = {"mode": _strict_str(memory["mode"], "memory_service_model.mode",
                                      E_TGT_MEMORY),
                  "board_ddr_bandwidth": memory["board_ddr_bandwidth"],
                  "contract_ref": memory["contract_ref"],
                  "note": memory["note"]}
    if memory_out["mode"] != MEMORY_MODE:
        raise CompileError(E_TGT_MEMORY,
                           "memory_service_model.mode must be %r" % MEMORY_MODE,
                           detail={"mode": memory_out["mode"],
                                   "implemented": [MEMORY_MODE]})
    bandwidth = memory_out["board_ddr_bandwidth"]
    if bandwidth is not None:
        if isinstance(bandwidth, bool) or not isinstance(bandwidth, (int, float, str)):
            raise CompileError(E_TGT_MEMORY,
                               "board_ddr_bandwidth must be null or a measured value",
                               detail={"value": repr(bandwidth)})
    for key in ("contract_ref", "note"):
        memory_out[key] = _strict_str(memory_out[key],
                                      "memory_service_model." + key, E_TGT_MEMORY)
    evidence = document["evidence"]
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise CompileError(E_TGT_SCHEMA, "evidence must be a list")
    evidence_out = []
    for index, row in enumerate(evidence):
        if not isinstance(row, Mapping):
            raise CompileError(E_TGT_SCHEMA,
                               "evidence[%d] must be an object" % index)
        _require_fields(row, EVIDENCE_FIELDS, "evidence[%d]" % index)
        evidence_out.append({"ref": _strict_str(row["ref"], "evidence.ref"),
                             "scope": _strict_str(row["scope"], "evidence.scope")})
    return {"schema": SCHEMA, "id": target_id, "part": part,
            "platform": platform_out, "clock_ns": clock, "axi_data_width": axi,
            "part_family": family, "resource_budget": budget_out,
            "memory_service_model": memory_out, "evidence": evidence_out}


def load_definition(target_id: Any) -> Dict[str, Any]:
    """Validate one registry definition by id (unknown ids are rejected)."""
    target_id = _strict_str(target_id, "target id", E_TGT_UNKNOWN)
    if target_id not in TARGET_REGISTRY:
        raise CompileError(E_TGT_UNKNOWN,
                           "unknown target %r; registered targets are %s"
                           % (target_id, sorted(TARGET_REGISTRY)),
                           detail={"requested": target_id,
                                   "registered": sorted(TARGET_REGISTRY)})
    document = validate_definition(TARGET_REGISTRY[target_id])
    if document["id"] != target_id:
        raise CompileError(E_TGT_SCHEMA,
                           "target definition id does not match its registry key",
                           detail={"key": target_id, "id": document["id"]})
    return document


@dataclass(frozen=True)
class ResolvedTarget:
    """Immutable, fully validated target selection."""

    id: str
    part: str
    platform: Dict[str, str]
    clock_ns: float
    axi_data_width: int
    part_family: str
    resource_budget: Dict[str, int]
    memory_service_model: Dict[str, Any]
    evidence: Tuple[Dict[str, str], ...]
    source_kind: str          # default | id

    @property
    def clock_mhz(self) -> float:
        return 1000.0 / float(self.clock_ns)

    @property
    def explicit_request(self) -> bool:
        return self.source_kind != "default"

    def document(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "id": self.id, "part": self.part,
                "platform": dict(self.platform), "clock_ns": self.clock_ns,
                "axi_data_width": self.axi_data_width,
                "part_family": self.part_family,
                "resource_budget": dict(self.resource_budget),
                "memory_service_model": dict(self.memory_service_model),
                "evidence": [dict(row) for row in self.evidence]}

    def hash(self) -> str:
        return hashlib.sha256(canonical_json(self.document()).encode("utf-8")).hexdigest()

    def manifest_document(self) -> Dict[str, Any]:
        """The target object written into compile/rtl manifests."""
        return {"schema": SCHEMA,
                "requested": {"kind": self.source_kind, "id": self.id},
                "resolved": {"id": self.id, "part": self.part,
                             "platform": dict(self.platform),
                             "clock_ns": self.clock_ns,
                             "clock_mhz": 1000.0 / float(self.clock_ns),
                             "axi_data_width": self.axi_data_width,
                             "part_family": self.part_family,
                             "resource_budget": dict(self.resource_budget),
                             "memory_service_model":
                                 dict(self.memory_service_model)},
                "hash": self.hash(),
                "evidence": [dict(row) for row in self.evidence],
                "qualification": "target identity only; rtl/ooc/platform/board "
                                 "are separate findings recorded in the run"}


def resolve(target_id: Any = None) -> ResolvedTarget:
    """Resolve the default or one registered target id; never fall back silently."""
    if target_id is None:
        document = load_definition(DEFAULT_TARGET)
        kind = "default"
    else:
        document = load_definition(target_id)
        kind = "id"
    return ResolvedTarget(
        id=document["id"], part=document["part"],
        platform=dict(document["platform"]), clock_ns=document["clock_ns"],
        axi_data_width=document["axi_data_width"],
        part_family=document["part_family"],
        resource_budget=dict(document["resource_budget"]),
        memory_service_model=dict(document["memory_service_model"]),
        evidence=tuple(dict(row) for row in document["evidence"]),
        source_kind=kind)


def registry_ids() -> Tuple[str, ...]:
    return tuple(sorted(TARGET_REGISTRY))


def list_targets() -> Sequence[Dict[str, Any]]:
    """Definition summary for CLI listing and tests."""
    rows = []
    for target_id in registry_ids():
        resolved = resolve(target_id)
        rows.append({"id": resolved.id, "part": resolved.part,
                     "platform": resolved.platform["name"],
                     "clock_ns": resolved.clock_ns,
                     "axi_data_width": resolved.axi_data_width,
                     "hash": resolved.hash(),
                     "default": target_id == DEFAULT_TARGET})
    return rows
