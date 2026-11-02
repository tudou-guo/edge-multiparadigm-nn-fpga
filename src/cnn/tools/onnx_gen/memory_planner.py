"""Deterministic 4KB memory planning for E1 W2.

The default profile uses first-fit reuse over non-overlapping live ranges.  The
legacy-compat profile reproduces the frozen npu_gen address sequence for the
existing four-layer MNIST chain, so W2 does not perturb R1/R2 compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .graph_ir import (
    CompileError,
    E_CFG,
    Graph,
    align_up,
)

MEMORY_SCHEMA = "e1-memory-plan-1"
DEFAULT_ALIGNMENT = 4096
DEFAULT_BASE = 0x10000000


def _hex(value: int) -> str:
    return "0x%08X" % (int(value) & 0xFFFFFFFF)


def _live_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return int(a["start"]) <= int(b["end"]) and int(b["start"]) <= int(a["end"])


def _addr_overlap(offset: int, size: int, other_offset: int, other_size: int) -> bool:
    return offset < other_offset + other_size and other_offset < offset + size


def _binding_from_buffer(buffer: Dict[str, Any], *, alias_of: Optional[str] = None,
                         view_offset_bytes: int = 0,
                         view_bytes: Optional[int] = None) -> Dict[str, Any]:
    if alias_of is None:
        return {
            "managed_by": "pl",
            "buffer": buffer["tensor"],
            "alias_of": None,
            "bytes": int(buffer["raw_bytes"]),
            "allocated_bytes": int(buffer["allocated_bytes"]),
            "offset": int(buffer["offset"]),
            "address": buffer["address"],
            "storage_id": int(buffer["storage_id"]),
            "view_offset_bytes": 0,
            "view_bytes": int(buffer["raw_bytes"]),
        }
    root_offset = int(buffer["offset"])
    root_address = int(str(buffer["address"]), 16)
    root_bytes = int(buffer["raw_bytes"])
    size = root_bytes if view_bytes is None else int(view_bytes)
    offset = root_offset + int(view_offset_bytes)
    is_view = (int(view_offset_bytes) != 0 or size != root_bytes)
    return {
        "managed_by": "view" if is_view else "alias",
        "buffer": buffer["tensor"],
        "alias_of": alias_of,
        "bytes": size,
        "allocated_bytes": 0 if is_view else int(buffer["allocated_bytes"]),
        "offset": offset,
        "address": _hex(root_address + int(view_offset_bytes)),
        "storage_id": int(buffer["storage_id"]),
        "view_offset_bytes": int(view_offset_bytes),
        "view_bytes": size,
    }


def _finalize_buffers(buffers: List[Dict[str, Any]], *, base: int, alignment: int) -> None:
    buffers.sort(key=lambda row: (int(row["offset"]), row["tensor"]))
    for index, row in enumerate(buffers):
        row["storage_id"] = index
        row["address"] = _hex(base + int(row["offset"]))


def _validate(buffer_map: Dict[str, Dict[str, Any]], tensor_bindings: Dict[str, Dict[str, Any]],
              *, alignment: int) -> Dict[str, Any]:
    alignment_errors = []
    overlap_errors = []
    alias_errors = []
    view_errors = []
    rows = list(buffer_map.values())
    for row in rows:
        if int(row["offset"]) % alignment or int(row["allocated_bytes"]) % alignment:
            alignment_errors.append(row["tensor"])
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            live_a = left["live_range"]
            live_b = right["live_range"]
            if _live_overlap(live_a, live_b) and _addr_overlap(
                    int(left["offset"]), int(left["allocated_bytes"]),
                    int(right["offset"]), int(right["allocated_bytes"])):
                overlap_errors.append({
                    "left": left["tensor"], "right": right["tensor"],
                    "left_live": [live_a["start"], live_a["end"]],
                    "right_live": [live_b["start"], live_b["end"]],
                    "left_offset": left["offset"], "right_offset": right["offset"],
                })
    for name, binding in tensor_bindings.items():
        alias = binding.get("alias_of")
        if not alias:
            continue
        root = tensor_bindings.get(alias)
        if root is None or binding.get("storage_id") != root.get("storage_id"):
            alias_errors.append(name)
            continue
        if binding.get("managed_by") == "view":
            view_offset = int(binding.get("view_offset_bytes", 0))
            view_bytes = int(binding.get("bytes", 0))
            root_address = int(str(root.get("address")), 16)
            root_allocated = int(root.get("allocated_bytes", 0))
            if (view_offset < 0 or view_bytes <= 0 or
                    view_offset + view_bytes > root_allocated or
                    int(str(binding.get("address")), 16) != root_address + view_offset):
                view_errors.append(name)
        elif binding.get("address") != root.get("address"):
            alias_errors.append(name)
    return {
        "alignment": "pass" if not alignment_errors else "fail",
        "live_overlap": "pass" if not overlap_errors else "fail",
        "alias_binding": "pass" if not alias_errors else "fail",
        "view_binding": "pass" if not view_errors else "fail",
        "alignment_errors": alignment_errors,
        "live_overlap_errors": overlap_errors,
        "alias_binding_errors": alias_errors,
        "view_binding_errors": view_errors,
    }


def _validate_copy_buffers(graph: Graph,
                           tensor_bindings: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    for node in graph.nodes:
        if not node.metadata.get("copy_required"):
            continue
        out_binding = tensor_bindings.get(node.outputs[0])
        if out_binding is None or not out_binding.get("address"):
            errors.append({"node": node.id, "error": "missing copy output binding"})
            continue
        out_start = int(str(out_binding["address"]), 16)
        out_end = out_start + int(out_binding.get("bytes", 0))
        for input_name in node.inputs:
            tensor = graph.tensors.get(input_name)
            if tensor is None or tensor.kind == "initializer":
                continue
            in_binding = tensor_bindings.get(input_name)
            if in_binding is None or not in_binding.get("address"):
                errors.append({"node": node.id, "input": input_name,
                               "error": "missing copy input binding"})
                continue
            in_start = int(str(in_binding["address"]), 16)
            in_end = in_start + int(in_binding.get("bytes", 0))
            if out_start < in_end and in_start < out_end:
                errors.append({"node": node.id, "input": input_name,
                               "error": "copy output overlaps input",
                               "input_range": [in_start, in_end],
                               "output_range": [out_start, out_end]})
    return {"copy_buffers": "pass" if not errors else "fail",
            "copy_buffer_errors": errors}


def _first_fit_plan(graph: Graph, graph_plan: Dict[str, Any], layout_plan: Dict[str, Any],
                    *, base: int, alignment: int, allow_reuse: bool = True) -> Dict[str, Any]:
    if base % alignment:
        raise CompileError(E_CFG, "arena base must be aligned to %d" % alignment)
    live_ranges = graph_plan["live_ranges"]
    alias_groups = layout_plan["alias_groups"]

    candidates: List[Dict[str, Any]] = []
    for name, row in sorted(layout_plan["tensors"].items()):
        if row["managed_by"] != "pl":
            continue
        storage = row["storage"]
        size = int(storage.get("bytes", 0))
        if size <= 0:
            continue
        members = [name] + list(alias_groups.get(name, []))
        live_members = [live_ranges[m] for m in members if m in live_ranges]
        if not live_members:
            continue
        if all(item.get("dead") for item in live_members):
            continue
        start = min(int(item["start"]) for item in live_members)
        end = max(int(item["end"]) for item in live_members)
        candidates.append({
            "tensor": name,
            "kind": graph.tensors[name].kind,
            "shape": list(graph.tensors[name].shape),
            "dtype": graph.tensors[name].dtype,
            "layout": storage.get("layout"),
            "raw_bytes": size,
            "allocated_bytes": align_up(size, alignment),
            "alignment": alignment,
            "live_range": {"start": start, "end": end, "end_exclusive": end + 1},
            "alias_members": sorted(members[1:]),
            "reuse_of": None,
            "offset": 0,
            "address": _hex(base),
            "storage_id": -1,
        })
    candidates.sort(key=lambda row: (row["live_range"]["start"], -row["allocated_bytes"], row["tensor"]))

    placed: List[Dict[str, Any]] = []
    next_page = 0
    for candidate in candidates:
        size = int(candidate["allocated_bytes"])
        if allow_reuse:
            page = 0
            while True:
                offset = page * alignment
                conflict = any(
                    _live_overlap(candidate["live_range"], other["live_range"]) and
                    _addr_overlap(offset, size, int(other["offset"]), int(other["allocated_bytes"]))
                    for other in placed
                )
                if not conflict:
                    break
                page += 1
        else:
            offset = next_page * alignment
            next_page += (size + alignment - 1) // alignment
        candidate["offset"] = offset
        reuse = [] if not allow_reuse else sorted(
            (other for other in placed if _addr_overlap(
                offset, size, int(other["offset"]), int(other["allocated_bytes"]))),
            key=lambda other: (int(other["offset"]), other["tensor"]),
        )
        if reuse:
            candidate["reuse_of"] = reuse[0]["tensor"]
        placed.append(candidate)

    _finalize_buffers(placed, base=base, alignment=alignment)
    buffer_map = {row["tensor"]: row for row in placed}
    tensor_bindings: Dict[str, Dict[str, Any]] = {}
    for name, row in sorted(layout_plan["tensors"].items()):
        managed = row["managed_by"]
        if managed in ("host", "constant"):
            tensor_bindings[name] = {
                "managed_by": managed, "buffer": None, "alias_of": None,
                "bytes": 0, "allocated_bytes": 0, "offset": None,
                "address": None, "storage_id": None,
            }
        elif managed in ("alias", "view"):
            root = row["alias_of"]
            binding = _binding_from_buffer(
                buffer_map[root], alias_of=root,
                view_offset_bytes=int(row.get("view_offset_bytes", 0)),
                view_bytes=int(row.get("view_bytes", row["storage"].get("bytes", 0))))
            tensor_bindings[name] = binding
        elif name in buffer_map:
            tensor_bindings[name] = _binding_from_buffer(buffer_map[name])
        else:
            tensor_bindings[name] = {
                "managed_by": "dead", "buffer": None, "alias_of": None,
                "bytes": int(row["storage"].get("bytes", 0)), "allocated_bytes": 0,
                "offset": None, "address": None, "storage_id": None,
            }

    checks = _validate(buffer_map, tensor_bindings, alignment=alignment)
    checks.update(_validate_copy_buffers(graph, tensor_bindings))
    used_bytes = max((int(row["offset"]) + int(row["allocated_bytes"]) for row in placed), default=0)
    reuse_count = sum(1 for row in placed if row["reuse_of"])
    status = "pass" if all(checks[key] == "pass"
                           for key in ("alignment", "live_overlap", "alias_binding",
                                       "view_binding", "copy_buffers")) else "fail"
    return {
        "schema": MEMORY_SCHEMA,
        "graph_digest": graph.digest(),
        "profile": "first_fit",
        "arena": {"base": _hex(base), "alignment": alignment,
                  "used_bytes": used_bytes, "used_pages": used_bytes // alignment},
        "buffers": buffer_map,
        "tensor_bindings": tensor_bindings,
        "alias_bindings": {name: binding for name, binding in tensor_bindings.items()
                           if binding.get("alias_of")},
        "host_boundaries": layout_plan.get("host_boundaries", []),
        "checks": checks,
        "status": status,
        "stats": {
            "buffers": len(placed),
            "raw_bytes": sum(int(row["raw_bytes"]) for row in placed),
            "allocated_bytes": sum(int(row["allocated_bytes"]) for row in placed),
            "reuse_count": reuse_count,
            "alias_tensors": sum(1 for row in layout_plan["tensors"].values()
                                 if row["managed_by"] == "alias"),
            "view_tensors": sum(1 for row in layout_plan["tensors"].values()
                                if row["managed_by"] == "view"),
            "host_tensors": sum(1 for row in layout_plan["tensors"].values()
                                if row["managed_by"] == "host"),
            "constant_tensors": sum(1 for row in layout_plan["tensors"].values()
                                    if row["managed_by"] == "constant"),
        },
        "gates": {"memory_planning": status, "view_copy_planning":
                  layout_plan.get("gates", {}).get("view_copy_planning", "not_run"),
                  "legacy_address_compat": "not_run",
                  "codegen": "not_run", "rtl": "not_run", "board": "not_run"},
    }


def _legacy_plan(graph: Graph, graph_plan: Dict[str, Any], layout_plan: Dict[str, Any],
                 *, base: int, alignment: int) -> Dict[str, Any]:
    if base % alignment:
        raise CompileError(E_CFG, "arena base must be aligned to %d" % alignment)
    if graph.source.get("format") != "legacy_ops_json":
        raise CompileError(E_CFG, "legacy-compat profile requires an IR from legacy ops.json")
    live_ranges = graph_plan["live_ranges"]
    if len(graph.inputs) != 1:
        raise CompileError(E_CFG, "legacy-compat profile requires one graph input")

    placed: List[Dict[str, Any]] = []
    bindings: Dict[str, Dict[str, Any]] = {}
    cursor = 0

    def alloc(name: str, kind: str, storage: Dict[str, Any], live: Dict[str, Any],
              legacy_role: str) -> None:
        nonlocal cursor
        size = int(storage.get("bytes", 0))
        if size <= 0:
            raise CompileError(E_CFG, "legacy allocation %s has zero/unknown bytes" % name)
        allocated = align_up(size, alignment)
        row = {
            "tensor": name,
            "kind": kind,
            "shape": live.get("shape"),
            "dtype": live.get("dtype"),
            "layout": storage.get("layout"),
            "raw_bytes": size,
            "allocated_bytes": allocated,
            "alignment": alignment,
            "live_range": {"start": int(live["start"]), "end": int(live["end"]),
                           "end_exclusive": int(live["end"]) + 1},
            "alias_members": [],
            "reuse_of": None,
            "offset": cursor,
            "address": _hex(base + cursor),
            "storage_id": -1,
            "legacy_role": legacy_role,
        }
        cursor += allocated
        placed.append(row)
        bindings[name] = _binding_from_buffer({**row, "storage_id": -1})
        bindings[name]["legacy_role"] = legacy_role

    input_name = graph.inputs[0]
    input_row = layout_plan["tensors"][input_name]
    alloc(input_name, "activation", input_row["storage"], live_ranges[input_name], "input")

    expected_input = input_name
    for node in graph.nodes:
        if node.inputs[0] != expected_input:
            raise CompileError(E_CFG,
                               "legacy-compat profile requires a linear chain at node %s" % node.id,
                               node_id=node.id)
        if node.op_type == "Conv":
            weight_name, bias_name = node.inputs[1], node.inputs[2]
            alloc(weight_name, "weight", layout_plan["tensors"][weight_name]["storage"],
                  live_ranges[weight_name], "weight")
            alloc(bias_name, "bias", layout_plan["tensors"][bias_name]["storage"],
                  live_ranges[bias_name], "bias")
            layer_index = int(node.metadata.get("legacy_layer_index", node.order))
            quant_name = "layer%02d.quant" % layer_index
            quant_storage = dict(node.metadata.get("quant_storage", {}))
            quant_live = {"start": 0, "end": node.order, "shape": [1], "dtype": "uint8"}
            alloc(quant_name, "quant", quant_storage, quant_live, "quant")
        out_name = node.outputs[0]
        alloc(out_name, "activation", layout_plan["tensors"][out_name]["storage"],
              live_ranges[out_name], "output")
        expected_input = out_name

    _finalize_buffers(placed, base=base, alignment=alignment)
    buffer_map = {row["tensor"]: row for row in placed}
    for name, binding in list(bindings.items()):
        row = buffer_map[name]
        bindings[name] = _binding_from_buffer(row)
        bindings[name]["legacy_role"] = row["legacy_role"]
    for name, row in sorted(layout_plan["tensors"].items()):
        if name not in bindings:
            bindings[name] = {
                "managed_by": row["managed_by"], "buffer": None, "alias_of": None,
                "bytes": 0, "allocated_bytes": 0, "offset": None,
                "address": None, "storage_id": None,
            }

    checks = _validate(buffer_map, bindings, alignment=alignment)
    checks.update(_validate_copy_buffers(graph, bindings))
    used_bytes = cursor
    status = "pass" if all(checks[key] == "pass"
                           for key in ("alignment", "live_overlap", "alias_binding",
                                       "view_binding", "copy_buffers")) else "fail"
    return {
        "schema": MEMORY_SCHEMA,
        "graph_digest": graph.digest(),
        "profile": "legacy_compat",
        "arena": {"base": _hex(base), "alignment": alignment,
                  "used_bytes": used_bytes, "used_pages": used_bytes // alignment},
        "buffers": buffer_map,
        "tensor_bindings": bindings,
        "alias_bindings": {},
        "host_boundaries": [],
        "checks": checks,
        "status": status,
        "stats": {
            "buffers": len(placed),
            "raw_bytes": sum(int(row["raw_bytes"]) for row in placed),
            "allocated_bytes": sum(int(row["allocated_bytes"]) for row in placed),
            "reuse_count": 0,
            "alias_tensors": 0,
            "view_tensors": 0,
            "host_tensors": 0,
            "constant_tensors": 0,
        },
        "gates": {"memory_planning": status,
                  "view_copy_planning": "pass",
                  "legacy_address_compat": "not_run",
                  "codegen": "not_run", "rtl": "not_run", "board": "not_run"},
    }


def build_memory_plan(graph: Graph, graph_plan: Dict[str, Any], layout_plan: Dict[str, Any],
                      *, profile: str = "first_fit",
                      base: int = DEFAULT_BASE,
                      alignment: int = DEFAULT_ALIGNMENT) -> Dict[str, Any]:
    if profile == "first_fit":
        result = _first_fit_plan(graph, graph_plan, layout_plan, base=base, alignment=alignment)
    elif profile == "no_reuse":
        result = _first_fit_plan(graph, graph_plan, layout_plan, base=base, alignment=alignment,
                                 allow_reuse=False)
    elif profile == "legacy_compat":
        result = _legacy_plan(graph, graph_plan, layout_plan, base=base, alignment=alignment)
    else:
        raise CompileError(E_CFG, "unknown memory profile %r" % profile)
    result["profile"] = profile
    return result


def check_legacy_instruction_addresses(graph: Graph, memory_plan: Dict[str, Any],
                                       stream: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a legacy-compat allocation against npu_gen payload addresses."""
    records = stream.get("records")
    if not isinstance(records, list):
        return {"status": "fail", "mismatches": [{"error": "missing records"}],
                "checked_records": 0}
    if len(records) != len(graph.nodes):
        return {"status": "fail", "checked_records": len(records),
                "mismatches": [{"error": "record count %d != node count %d" %
                                           (len(records), len(graph.nodes))}]}
    bindings = memory_plan["tensor_bindings"]
    mismatches = []
    for node, record in zip(graph.nodes, records):
        payload = record.get("payload", [])
        if len(payload) < 5:
            mismatches.append({"node": node.id, "error": "payload shorter than 5"})
            continue
        expected = [bindings[node.inputs[0]]["address"], None, None, None,
                    bindings[node.outputs[0]]["address"]]
        if node.op_type == "Conv":
            layer_index = int(node.metadata.get("legacy_layer_index", node.order))
            quant_name = "layer%02d.quant" % layer_index
            expected[1] = bindings[node.inputs[1]]["address"]
            expected[2] = bindings[node.inputs[2]]["address"]
            expected[3] = bindings[quant_name]["address"]
        else:
            expected[1] = "0x00000000"
            expected[2] = "0x00000000"
            expected[3] = "0x00000000"
        actual = [_hex(payload[i]) for i in range(5)]
        if actual != expected:
            mismatches.append({"node": node.id, "op_type": node.op_type,
                               "expected": expected, "actual": actual})
    return {
        "status": "pass" if not mismatches else "fail",
        "checked_records": len(records),
        "mismatches": mismatches,
    }
