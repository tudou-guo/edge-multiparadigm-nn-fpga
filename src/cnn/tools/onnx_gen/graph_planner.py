"""Deterministic DAG scheduling and live-range extraction for E1.

This pass does not allocate memory and never reorders independent nodes.  It
turns the canonical graph into a serial schedule (the current hardware runs one
core at a time), computes producer/consumer tables and makes the release point
of multi-consumer tensors explicit.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph_ir import E_GRAPH, CompileError, Graph

PLAN_SCHEMA = "e1-graph-plan-1"


def _node_kind(node) -> str:
    if node.placement == "ps":
        return "host"
    if node.placement == "metadata" or node.metadata_only:
        return "metadata"
    return "compute"


def build_graph_plan(graph: Graph) -> Dict[str, Any]:
    graph.finalize()  # deterministic revalidation; also refreshes consumers
    terminal = len(graph.nodes)
    order_by_id = {node.id: node.order for node in graph.nodes}

    consumer_table: Dict[str, List[Dict[str, Any]]] = {}
    for name, tensor in sorted(graph.tensors.items()):
        rows = []
        for consumer in tensor.consumers:
            rows.append({
                "node": consumer["node"],
                "input_index": int(consumer["input_index"]),
                "order": int(order_by_id[consumer["node"]]),
            })
        rows.sort(key=lambda row: (row["order"], row["input_index"], row["node"]))
        consumer_table[name] = rows

    live_ranges: Dict[str, Dict[str, Any]] = {}
    for name, tensor in sorted(graph.tensors.items()):
        consumers = consumer_table[name]
        producer_order = order_by_id.get(tensor.producer) if tensor.producer else None
        start = int(producer_order) if producer_order is not None else 0
        last_consumer = max((row["order"] for row in consumers), default=None)
        is_output = name in graph.outputs
        if is_output:
            end = terminal
            dead = False
        elif last_consumer is not None:
            end = int(last_consumer)
            dead = False
        elif producer_order is not None:
            end = int(producer_order)
            dead = True
        else:
            end = 0
            dead = True
        first_use = start if producer_order is not None else (
            min((row["order"] for row in consumers), default=None)
        )
        live_ranges[name] = {
            "tensor": name,
            "kind": tensor.kind,
            "start": start,
            "end": end,
            "end_exclusive": end + 1,
            "first_use": first_use,
            "last_consumer": last_consumer,
            "producer": tensor.producer,
            "consumers": consumers,
            "is_graph_input": name in graph.inputs,
            "is_initializer": tensor.kind == "initializer",
            "is_graph_output": is_output,
            "dead": dead,
            "shape": list(tensor.shape) if tensor.shape is not None else None,
            "dtype": tensor.dtype,
            "bytes_hint": (int(tensor.storage["bytes"]) if tensor.storage is not None and "bytes" in tensor.storage
                           else tensor.logical_bytes),
        }

    schedule: List[Dict[str, Any]] = []
    max_live = 0
    total_live = 0
    for node in graph.nodes:
        order = node.order
        before = sorted(name for name, info in live_ranges.items()
                        if info["start"] <= order <= info["end"])
        after = sorted(name for name, info in live_ranges.items()
                       if info["start"] <= order + 1 <= info["end"])
        released = sorted(set(before) - set(after))
        produced = [name for name in node.outputs if live_ranges[name]["start"] == order]
        max_live = max(max_live, len(after))
        total_live += len(after)
        schedule.append({
            "order": order,
            "node_id": node.id,
            "name": node.name,
            "op_type": node.op_type,
            "kind": _node_kind(node),
            "placement": node.placement,
            "metadata_only": bool(node.metadata_only),
            "inputs": list(node.inputs),
            "outputs": list(node.outputs),
            "live_before": before,
            "live_after": after,
            "released_after": released,
            "produced": produced,
        })

    aliases = []
    for node in graph.nodes:
        if node.metadata_only:
            if not node.inputs or len(node.outputs) != 1:
                raise CompileError(E_GRAPH,
                                   "metadata node %s must have a data input and one output" % node.id,
                                   node_id=node.id)
            aliases.append({"node": node.id, "order": node.order,
                            "input": node.inputs[0], "output": node.outputs[0],
                            "aux_inputs": list(node.inputs[1:])})

    return {
        "schema": PLAN_SCHEMA,
        "graph_name": graph.name,
        "graph_digest": graph.digest(),
        "serial_policy": "stable_topological_order_no_independent_reordering",
        "terminal_order": terminal,
        "schedule": schedule,
        "consumer_table": consumer_table,
        "live_ranges": live_ranges,
        "metadata_aliases": aliases,
        "stats": {
            "nodes": len(graph.nodes),
            "compute_nodes": sum(1 for row in schedule if row["kind"] == "compute"),
            "metadata_nodes": sum(1 for row in schedule if row["kind"] == "metadata"),
            "host_nodes": sum(1 for row in schedule if row["kind"] == "host"),
            "tensors": len(graph.tensors),
            "dead_tensors": sorted(name for name, info in live_ranges.items() if info["dead"]),
            "max_live_tensors": max_live,
            "total_live_tensor_steps": total_live,
        },
        "gates": {"schedule": "pass", "liveness": "pass", "memory_planning": "not_run"},
    }
