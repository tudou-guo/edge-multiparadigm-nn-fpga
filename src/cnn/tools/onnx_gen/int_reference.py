
# -*- coding: utf-8 -*-
"""Independent integer reference for the v0.2 real-YOLO affine-int8-v1 model.

This module is the mathematical source of truth for the deployment integer
contract.  It reads ONLY the exported deployment assets::

    <model-dir>/graph.json        topology, shapes, per-node domains
    <model-dir>/quant_table.json  per-conv weight scales / packing evidence
    <model-dir>/weights.npz       int8 weights, int32 bias, int64 folded bias

It never reads RTL output, never reads the reference project's saved DUT text and
never calls a compiled artifact as its maths source.  It recomputes the bias
folding and the M0/SHIFT requantisation parameters from the declared scales and
weights (cross_check_pack proves the exporter's packed values agree with that
recomputation), and it executes the graph in the signed re-encoded domain defined
by docs/V0_2_REAL_YOLO_NUMERIC_CONTRACT.md:

    qs = qu - 128, zs = zu - 128, real = s*(qs - zs)

Performance notes (task book section 9: no per-MAC Python loops, no whole-image
im2col monster)::

* the convolution accumulator is formed with a float64 GEMM over an im2col view
  that is materialised in output-row blocks.  Every operand is an exactly
  representable integer and every partial sum stays below 2**53, so the float64
  result is exactly the integer accumulator; prove_exact_gemm() checks that
  against int64 tensordot on real layer data.
* intermediates are kept as C,H,W int8 arrays (the largest real one is
  128x20x20), so memory stays bounded and the whole 640 graph runs in seconds.

Usage::

    python -B -m tools.onnx_gen.int_reference --model-dir <run>/model \
        --input-u8 <sample_u8.npy> --out-dir <run>/reports/model/int_ref
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np

from tools.onnx_gen.affine_quant import (
    PROFILE,
    S8_MAX,
    S8_MIN,
    check_acc_int32,
    check_rescale_int64,
    choose_add_requant,
    choose_m0_shift,
    conv_bias_fold,
    maxpool_signed,
    requant_add,
    requant_conv,
    resize_nearest_x2_signed,
    s8_to_u8,
    u8_to_s8,
)

#: maximum number of float64 im2col elements materialised per GEMM block
BLOCK_ELEMENTS = 1 << 22  # 4M elements = 32 MiB


class AssetError(RuntimeError):
    """Raised when the deployment assets are inconsistent with the contract."""


class RssSampler(object):
    """Background RSS high-water sampler (peak memory evidence, best effort)."""

    def __init__(self, period=0.02):
        self.period = float(period)
        self.peak = 0
        self.start = 0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:
            self._proc = None

    def _rss(self):
        if self._proc is None:
            return 0
        try:
            return int(self._proc.memory_info().rss)
        except Exception:
            return 0

    def _loop(self):
        while not self._stop.wait(self.period):
            self.peak = max(self.peak, self._rss())

    def __enter__(self):
        self.start = self._rss()
        self.peak = self.start
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak = max(self.peak, self._rss())
        return False

    def summary(self):
        return {"rss_start_bytes": int(self.start), "rss_peak_bytes": int(self.peak),
                "rss_peak_delta_bytes": int(max(0, self.peak - self.start)),
                "sampler_period_s": self.period}


# --------------------------------------------------------------------------- #
# asset loading
# --------------------------------------------------------------------------- #
def load_assets(model_dir):
    model_dir = Path(model_dir)
    graph_path = model_dir / "graph.json"
    quant_path = model_dir / "quant_table.json"
    weights_path = model_dir / "weights.npz"
    for p in (graph_path, quant_path, weights_path):
        if not p.is_file():
            raise AssetError("missing deployment asset: %s" % p)
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    weights = dict(np.load(str(weights_path), allow_pickle=False))
    if graph.get("profile") != PROFILE:
        raise AssetError("graph profile %r != %r" % (graph.get("profile"), PROFILE))
    return graph, quant, weights


def _hex_sha(arr):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def cross_check_pack(graph, quant, weights):
    """Recompute bias folding and M0/SHIFT from scales/weights; compare the pack.

    The integer reference must not trust packed constants: this function proves
    the exporter's bq_folded, sum_w, m0 and shift are exactly the contract values.
    """
    problems = []
    checked = 0
    conv_nodes = [n for n in graph["nodes"] if n["op"] == "Conv"]
    for node in conv_nodes:
        nid = node["id"]
        w = weights.get(nid + ".w")
        b = weights.get(nid + ".b")
        bq = weights.get(nid + ".bq")
        if w is None or b is None or bq is None:
            problems.append({"node": nid, "issue": "missing packed arrays"})
            continue
        qnode = node["quant"]
        zs_in = int(qnode["in_zero_points"][0]) - 128
        folding = conv_bias_fold(w, b, zs_in, cin=int(node["attrs"]["cin"]))
        check_acc_int32(int(folding["acc_bound"]))
        if not np.array_equal(folding["bias_folded"], bq.astype(np.int64)):
            bad = int((folding["bias_folded"] != bq.astype(np.int64)).sum())
            problems.append({"node": nid, "issue": "bias_folded mismatch", "count": bad})
        entry = quant.get("conv", {}).get(nid, {})
        m0s = choose_m0_shift(float(qnode["in_scales"][0]), float(qnode["w_scale"]),
                              float(qnode["out_scale"]))
        check_rescale_int64(int(folding["acc_bound"]), int(m0s["m0"]))
        packed = entry.get("m0_shift", {})
        if int(packed.get("m0", -1)) != int(m0s["m0"]) or int(packed.get("shift", -1)) != int(m0s["shift"]):
            problems.append({"node": nid, "issue": "m0/shift mismatch",
                             "packed": [packed.get("m0"), packed.get("shift")],
                             "recomputed": [m0s["m0"], m0s["shift"]]})
        if int(entry.get("acc_bound", -1)) != int(folding["acc_bound"]):
            problems.append({"node": nid, "issue": "acc_bound mismatch",
                             "packed": entry.get("acc_bound"),
                             "recomputed": int(folding["acc_bound"])})
        if "sum_w" in entry:
            packed_sum = np.asarray(entry["sum_w"], dtype=np.int64)
            if not np.array_equal(packed_sum, folding["sum_w"]):
                problems.append({"node": nid, "issue": "sum_w mismatch"})
        checked += 1
    return {"schema": "v0.2-yolo-int-ref-pack-cross-check-1", "conv_checked": checked,
            "problems": problems, "ok": not problems}


# --------------------------------------------------------------------------- #
# exact integer GEMM helpers
# --------------------------------------------------------------------------- #
def _im2col_rows(xp, kh, kw, stride, i0, i1, ow):
    """Materialise im2col columns for output rows [i0, i1) of one padded plane set.

    The strided view must start at the *first requested output row* (i0*stride in
    the padded plane); without that base offset every block after the first
    re-read rows 0.. and silently corrupts the accumulator of every downstream
    node.  This matters for the lowered 9-tap graph, where a 1x1 Conv becomes a
    centre-tap 3x3 Conv: k*ow grows 9x and BLOCK_ELEMENTS splits the output rows
    into more than one block for real layers (e.g. conv3: 91-row blocks over 160
    rows), so the bug was reachable in the full-model reference even though the
    source 1x1 form stayed inside a single block.
    """
    c = xp.shape[0]
    base = xp[:, int(i0) * int(stride):, :]
    shape = (c, kh, kw, i1 - i0, ow)
    strides = (base.strides[0], base.strides[1], base.strides[2],
               base.strides[1] * stride, base.strides[2] * stride)
    view = np.lib.stride_tricks.as_strided(base, shape=shape, strides=strides,
                                           writeable=False)
    return np.ascontiguousarray(view).reshape(c * kh * kw, (i1 - i0) * ow)


def conv_accumulate(x_s8, w_s8, stride, pad, zpad):
    """Exact integer convolution accumulator (does not requantise).

    Returns int64 C,H,W accumulator of sum(qs*w) only (the folded bias and the
    zero-point reactive term are added by the caller through conv_bias_fold).
    """
    x = np.asarray(x_s8, dtype=np.int8)
    w = np.asarray(w_s8, dtype=np.int8)
    if x.ndim != 3 or w.ndim != 4:
        raise ValueError("conv_accumulate expects CHW input and OIHW weight, got %r %r"
                         % (x.shape, w.shape))
    c, h, wd = x.shape
    o, cin, kh, kw = w.shape
    if cin != c:
        raise ValueError("weight Cin %d != input C %d" % (cin, c))
    xp = np.full((c, h + 2 * pad, wd + 2 * pad), int(zpad), dtype=np.int8)
    xp[:, pad:pad + h, pad:pad + wd] = x
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (wd + 2 * pad - kw) // stride + 1
    w2 = w.reshape(o, c * kh * kw).astype(np.float64)
    out = np.empty((o, oh, ow), dtype=np.int64)
    k = c * kh * kw
    rows_per_block = max(1, int(BLOCK_ELEMENTS // max(1, k * ow)))
    for i0 in range(0, oh, rows_per_block):
        i1 = min(oh, i0 + rows_per_block)
        cols = _im2col_rows(xp, kh, kw, stride, i0, i1, ow).astype(np.float64)
        acc = w2 @ cols  # exact: integer operands, partial sums < 2**53
        out[:, i0:i1, :] = acc.reshape(o, i1 - i0, ow).astype(np.int64)
    return out


def prove_exact_gemm(x_s8, w_s8, stride, pad, zpad, sample=3):
    """Prove float64 GEMM equals int64 tensordot on real layer data.

    Compares a few output rows computed by the production conv_accumulate against
    a slow-but-obviously-exact int64 path (one output pixel at a time on the
    sampled rows).  Returns a dict with the compared element count and max |diff|.
    """
    x = np.asarray(x_s8, dtype=np.int8)
    w = np.asarray(w_s8, dtype=np.int8)
    c, h, wd = x.shape
    o, cin, kh, kw = w.shape
    acc = conv_accumulate(x, w, stride, pad, zpad)
    oh, ow = acc.shape[1], acc.shape[2]
    rows = sorted(set([0, oh // 2, oh - 1][:max(1, min(sample, oh))]))
    xp = np.full((c, h + 2 * pad, wd + 2 * pad), int(zpad), dtype=np.int64)
    xp[:, pad:pad + h, pad:pad + wd] = x
    diffs = []
    checked = 0
    for i in rows:
        for j in range(ow):
            patch = xp[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
            expect = np.tensordot(w.astype(np.int64), patch, axes=([1, 2, 3], [0, 1, 2]))
            got = acc[:, i, j]
            diffs.append(int(np.abs(expect - got).max()))
            checked += o
    return {"rows": rows, "elements_checked": checked, "max_abs_diff": int(max(diffs) if diffs else 0),
            "exact": bool(max(diffs or [0]) == 0)}


# --------------------------------------------------------------------------- #
# graph execution
# --------------------------------------------------------------------------- #
class IntegerReference(object):
    """Execute the deployment graph in the signed re-encoded integer domain."""

    def __init__(self, graph, quant, weights, log=None, rounding="ties_away_from_zero"):
        self.graph = graph
        self.quant = quant
        self.weights = weights
        self.log = log if log is not None else (lambda *a: None)
        # the frozen profile rounding is ties_away_from_zero; ties_even exists only to
        # measure the accuracy effect of the rounding choice (never silently switched)
        self.rounding = rounding
        self._requant_cache = {}

    # -- per-node requantisation parameters (computed from declared scales) --
    def conv_params(self, node):
        nid = node["id"]
        key = ("conv", nid)
        if key not in self._requant_cache:
            q = node["quant"]
            zs_in = int(q["in_zero_points"][0]) - 128
            zs_out = int(q["out_zero_points"][0]) - 128
            w = self.weights[nid + ".w"]
            b = self.weights[nid + ".b"]
            folding = conv_bias_fold(w, b, zs_in, cin=int(node["attrs"]["cin"]))
            m0s = choose_m0_shift(float(q["in_scales"][0]), float(q["w_scale"]),
                                  float(q["out_scale"]))
            self._requant_cache[key] = {"folding": folding, "m0": int(m0s["m0"]),
                                        "shift": int(m0s["shift"]), "zs_in": zs_in,
                                        "zs_out": zs_out, "relu": bool(node["attrs"].get("relu", False)),
                                        "m0_shift": m0s}
        return self._requant_cache[key]

    def run(self, input_u8=None, collect_stats=True, initial=None):
        """Execute the graph; returns heads plus optional per-node diagnostics.

        input_u8: uint8 array with shape (1,3,H,W), (3,H,W) or (H,W,3).
        initial:  optional dict tensor_name -> signed int8 CHW array that seeds the
                  tensors already computed (used by isolated per-node short tests and
                  by the row-slice checks of later phases); when given, input_u8 is
                  not required.
        """
        t_start = time.time()
        if initial is not None:
            # per-node / row-slice short tests seed the already-computed tensors and
            # execute only the remaining nodes of this (usually single-node) graph
            tensors = {name: np.ascontiguousarray(np.asarray(arr, dtype=np.int8))
                       for name, arr in initial.items()}
            stats = {}
            timings = {}
            return self._run_from(tensors, stats, timings, collect_stats, t_start)
        x_u8 = np.asarray(input_u8)
        if x_u8.dtype != np.uint8:
            raise ValueError("input must be uint8, got %r" % (x_u8.dtype,))
        declared = tuple(int(v) for v in (self.graph.get("input") or {}).get("shape", []))
        expected = declared[1:] if len(declared) == 4 and declared[0] == 1 else None
        was_nchw = x_u8.ndim == 4
        if was_nchw:
            if x_u8.shape[0] != 1:
                raise ValueError("batch must be 1, got %r" % (x_u8.shape,))
            x_u8 = x_u8[0]
        if (not was_nchw and x_u8.ndim == 3 and expected is not None and
                tuple(x_u8.shape) != expected and tuple(x_u8.shape) == (expected[1], expected[2], expected[0])):
            x_u8 = np.transpose(x_u8, (2, 0, 1))
        elif (not was_nchw and expected is None and x_u8.ndim == 3 and
              x_u8.shape[-1] == 3 and x_u8.shape[0] != 3):
            x_u8 = np.transpose(x_u8, (2, 0, 1))
        if x_u8.ndim != 3:
            raise ValueError("cannot interpret input shape %r" % (x_u8.shape,))
        if expected is not None and tuple(x_u8.shape) != expected:
            raise ValueError("input CHW shape %r != graph input %r" % (x_u8.shape, expected))
        tensors = {}
        stats = {}
        input_node = self.graph["input"]
        zs_in = int(input_node["zero_point"]) - 128
        tensors["__input__"] = u8_to_s8(x_u8)
        stats["__input__"] = {"shape": list(x_u8.shape), "zero_point_s8": zs_in,
                              "scale": float(input_node["scale"])}
        timings = {}
        return self._run_from(tensors, stats, timings, collect_stats, t_start)

    def _run_from(self, tensors, stats, timings, collect_stats, t_start):
        """Execute the remaining nodes on the seeded tensor map, then collect the heads."""
        for node in self.graph["nodes"]:
            t0 = time.time()
            nid = node["id"]
            op = node["op"]
            q = node["quant"]
            ins = [tensors[name] for name in node["inputs"]]
            zs_out = int(q["out_zero_points"][0]) - 128
            if op == "Conv":
                p = self.conv_params(node)
                attrs = node["attrs"]
                acc = conv_accumulate(ins[0], self.weights[nid + ".w"],
                                      int(attrs["stride"]), int(attrs["pad"]), p["zs_in"])
                acc += p["folding"]["bias_folded"].astype(np.int64).reshape(-1, 1, 1)
                st = {} if collect_stats else None
                out = requant_conv(acc, p["m0"], p["shift"], p["zs_out"],
                                   relu=p["relu"], stats=st, rounding=self.rounding)
                if st:
                    st.update({"acc_min": int(acc.min()), "acc_max": int(acc.max()),
                               "acc_bound_proved": int(p["folding"]["acc_bound"]),
                               "m0": p["m0"], "shift": p["shift"]})
                    stats[nid] = st
            elif op == "Add":
                if len(ins) != 2:
                    raise AssetError("Add %s expects 2 inputs, got %d" % (nid, len(ins)))
                params = choose_add_requant(float(q["in_scales"][0]), float(q["in_scales"][1]),
                                           float(q["out_scale"]))
                out = requant_add(ins[0], int(q["in_zero_points"][0]) - 128,
                                 ins[1], int(q["in_zero_points"][1]) - 128,
                                 params, zs_out, rounding=self.rounding)
                if collect_stats:
                    stats[nid] = {"params": params, "shape": list(out.shape),
                                  "sat_low": int((out == S8_MIN).sum()),
                                  "sat_high": int((out == S8_MAX).sum())}
            elif op == "Concat":
                parts = []
                for i, t in enumerate(ins):
                    s_in = float(q["in_scales"][i])
                    s_out = float(q["out_scale"])
                    z_in = int(q["in_zero_points"][i]) - 128
                    if s_in == s_out and z_in == zs_out:
                        parts.append(t)
                    else:
                        from tools.onnx_gen.affine_quant import requant_copy_to_domain
                        parts.append(requant_copy_to_domain(t, z_in, s_in, s_out, zs_out,
                                                           rounding=self.rounding))
                out = np.concatenate(parts, axis=0)
                if collect_stats:
                    stats[nid] = {"inputs": len(parts), "shape": list(out.shape),
                                  "same_domain_inputs": int(sum(
                                      1 for i in range(len(ins))
                                      if float(q["in_scales"][i]) == float(q["out_scale"])
                                      and int(q["in_zero_points"][i]) - 128 == zs_out))}
            elif op == "Split":
                outputs = node["attrs"].get("outputs") or []
                offset = 0
                sizes = [int(s) for s in node["attrs"]["sizes"]]
                for i, size in enumerate(sizes):
                    name = outputs[i]["name"] if i < len(outputs) else "%s_%d" % (nid, i)
                    tensors[name] = ins[0][offset:offset + size]
                    if collect_stats:
                        stats[name] = {"shape": list(tensors[name].shape),
                                       "channel_offset": offset, "domain_of": node["inputs"][0]}
                    offset += size
                timings[nid] = time.time() - t0
                continue
            elif op == "MaxPool":
                attrs = node["attrs"]
                out = maxpool_signed(ins[0], attrs.get("kernel_shape", attrs.get("kernel", 5)),
                                     attrs.get("strides", attrs.get("stride", 1)),
                                     attrs.get("pads", attrs.get("pad", 2)),
                                     attrs.get("dilations", 1))
                if collect_stats:
                    stats[nid] = {"shape": list(out.shape), "pad_value": S8_MIN}
            elif op == "Resize":
                out = resize_nearest_x2_signed(ins[0])
                if collect_stats:
                    stats[nid] = {"shape": list(out.shape), "mode": "nearest"}
            elif op in ("Relu", "Clip"):
                out = np.maximum(ins[0], np.int8(zs_out))
                if collect_stats:
                    stats[nid] = {"shape": list(out.shape), "threshold_s8": zs_out}
            else:
                raise AssetError("unsupported op %r in node %s" % (op, nid))
            tensors[nid] = np.ascontiguousarray(out, dtype=np.int8)
            timings[nid] = time.time() - t0
            self.log("  %-10s %-8s %s %s" % (nid, op, list(tensors[nid].shape),
                                             "%.3fs" % timings[nid]))
        heads = {}
        node_by_id = {node["id"]: node for node in self.graph["nodes"]}
        for name in self.graph["outputs"]:
            if name not in tensors:
                raise AssetError("graph output %r was not produced" % name)
            t = tensors[name]
            # Multi-output nodes (Split) expose outputs whose names are not node
            # ids; their domain lives in the reference quant table instead.
            if name in node_by_id:
                quant = node_by_id[name]["quant"]
                scale = float(quant["out_scale"])
                zero_point = int(quant["out_zero_points"][0])
            else:
                row = (self.quant.get("tensors") or {}).get(name)
                if not row:
                    raise AssetError("graph output %r has no node or quant entry" % name)
                scale = float(row["out_scale"])
                zero_point = int(row["out_zero_point"])
            heads[name] = {"s8": t, "u8": s8_to_u8(t),
                           "scale": scale, "zero_point": zero_point,
                           "shape": list(t.shape)}
        elapsed = time.time() - t_start
        return {"heads": heads, "tensors": tensors, "stats": stats,
                "timings": timings, "elapsed_s": elapsed}

    def _node_by_id(self, nid):
        for node in self.graph["nodes"]:
            if node["id"] == nid:
                return node
        raise AssetError("unknown node %r" % (nid,))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_input_u8(path, shape=None):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p))
    else:
        raw = p.read_bytes()
        arr = np.frombuffer(raw, dtype=np.uint8)
        if shape is None:
            raise ValueError("binary input needs --input-shape")
        arr = arr.reshape(tuple(int(x) for x in shape.split(",")))
    return arr


def run_reference(model_dir, input_u8, out_dir=None, dump_npz=None, log=print):
    graph, quant, weights = load_assets(model_dir)
    ref = IntegerReference(graph, quant, weights, log=log)
    with RssSampler() as sampler:
        result = ref.run(input_u8)
    report = {
        "schema": "v0.2-yolo-int-reference-run-1",
        "profile": PROFILE,
        "model_dir": str(Path(model_dir)),
        "rounding": "ties_away_from_zero",
        "elapsed_s": round(result["elapsed_s"], 3),
        "memory": sampler.summary(),
        "heads": {name: {"shape": h["shape"], "scale": h["scale"], "zero_point": h["zero_point"],
                         "s8_sha256": _hex_sha(h["s8"]), "u8_sha256": _hex_sha(h["u8"]),
                         "s8_min": int(h["s8"].min()), "s8_max": int(h["s8"].max())}
                  for name, h in result["heads"].items()},
        "node_stats": result["stats"],
        "node_timings_s": {k: round(v, 4) for k, v in result["timings"].items()},
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "int_reference_run.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if dump_npz is None:
            dump_npz = out / "heads_s8.npz"
        np.savez_compressed(str(dump_npz), **{name: h["s8"] for name, h in result["heads"].items()})
        report["dumped_heads_npz"] = str(dump_npz)
    return report, result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Independent integer reference for the "
                                             "v0.2 real-YOLO affine-int8-v1 deployment assets.")
    ap.add_argument("--model-dir", required=True, help="directory with graph.json, "
                    "quant_table.json, weights.npz")
    ap.add_argument("--input-u8", required=True, help="uint8 input tensor (.npy) or raw bytes")
    ap.add_argument("--input-shape", default=None, help="shape for raw byte input, e.g. 1,3,640,640")
    ap.add_argument("--out-dir", default=None, help="write int_reference_run.json + heads_s8.npz here")
    ap.add_argument("--cross-check-pack", action="store_true",
                    help="recompute bias fold / M0 / SHIFT and compare with the packed assets")
    ap.add_argument("--prove-exact-gemm", type=int, default=0, metavar="N",
                    help="prove float64 GEMM exactness on N sampled conv layers (int64 cross-check)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    log = (lambda *a: None) if args.quiet else print
    graph, quant, weights = load_assets(args.model_dir)
    out = {"schema": "v0.2-yolo-int-reference-cli-1", "model_dir": args.model_dir}
    if args.cross_check_pack:
        cc = cross_check_pack(graph, quant, weights)
        out["cross_check_pack"] = cc
        log("cross_check_pack: %d conv checked, %d problems" % (cc["conv_checked"], len(cc["problems"])))
        if not cc["ok"]:
            for p in cc["problems"][:10]:
                log("  problem:", p)
    if args.prove_exact_gemm:
        convs = [n for n in graph["nodes"] if n["op"] == "Conv"]
        order = sorted(range(len(convs)), key=lambda i: -int(convs[i]["attrs"]["cin"]))
        proofs = []
        rng = np.random.RandomState(0)
        for idx in order[:args.prove_exact_gemm]:
            node = convs[idx]
            cin = int(node["attrs"]["cin"])
            h = min(8, int(node["out_shape"][1]) + 2)
            w_ = min(8, int(node["out_shape"][2]) + 2)
            fake = rng.randint(-128, 128, size=(cin, h, w_)).astype(np.int8)
            proof = prove_exact_gemm(fake, weights[node["id"] + ".w"],
                                     int(node["attrs"]["stride"]), int(node["attrs"]["pad"]),
                                     int(node["quant"]["in_zero_points"][0]) - 128)
            proof["node"] = node["id"]
            proofs.append(proof)
            log("exact gemm %s: %s" % (node["id"], proof))
        out["exact_gemm_proofs"] = proofs
        out["all_exact"] = all(p["exact"] for p in proofs)
    input_u8 = _load_input_u8(args.input_u8, args.input_shape)
    report, _ = run_reference(args.model_dir, input_u8, out_dir=args.out_dir,
                              log=log)
    out["run"] = {k: report[k] for k in ("elapsed_s", "memory", "heads")}
    if args.out_dir:
        (Path(args.out_dir) / "int_reference_cli.json").write_text(
            json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
