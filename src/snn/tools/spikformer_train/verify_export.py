"""Export-fidelity evidence: torch quantised integer-equivalent forward vs integer_ref
on the FINAL real checkpoint (and a fixed pair of corpus samples).

Runs as TWO processes because the torch and numpy OpenMP runtimes clash on this host when
integer matmuls run after torch in one process (keep the golden integer path clean):

  part torch  (E:/miniconda/envs/pytorch_clean/python.exe):
      load train/best_state_<tag>.pt, bind the FINAL exported int32 thresholds from
      train/ckpt_real_<tag>.npz, run the trace, dump all per-layer integer arrays + logits.
  part numpy  (E:/miniconda/python.exe):
      load ckpt_real_<tag>.npz via integer_ref (strict), run the same samples, compare every
      layer with integer_ref.first_mismatch, check PL sums + logits/argmax, write the report.

PASS here means: the exported integer checkpoint is exactly what the trained quantised
torch forward computes (no export drift), on real corpus samples under the final thresholds.

Usage:
  python -B tools/spikformer_train/verify_export.py --part torch --tag real50 \
      --config examples/spikformer/config/nlp_sentiment_b1.json \
      --run-dir build/spikformer/nlp_48h_20261029_1915
  python -B tools/spikformer_train/verify_export.py --part numpy --tag real50 \
      --config examples/spikformer/config/nlp_sentiment_b1.json \
      --run-dir build/spikformer/nlp_48h_20261029_1915
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def flat_from_trace(tr):
    out = {
        "embed_spikes": np.rint(tr["embed_spikes"].detach().cpu().numpy()).astype(np.int64),
        "embed_delta": np.rint(tr["embed"]["delta"].detach().cpu().numpy()).astype(np.int64),
        "embed_mem": np.rint(tr["embed"]["mem"].detach().cpu().numpy()).astype(np.int64),
        "final_residual": np.rint(tr["final_residual"].detach().cpu().numpy()).astype(np.int64),
        "pl_sums": np.rint(tr["pl_sums"].detach().cpu().numpy()).astype(np.int64),
        "pooled": np.asarray(tr["pooled"].detach().cpu().numpy(), np.float32),
        "logits": np.asarray(tr["logits"].detach().cpu().numpy(), np.float32),
    }
    for b, btr in enumerate(tr["blocks"]):
        for k, v in btr.items():
            out["block%d.%s" % (b, k)] = np.rint(
                v.detach().cpu().numpy()).astype(np.int64)
    return out


def unflat_to_trace(flat):
    tr = {"embed": {"delta": flat["embed_delta"], "mem": flat["embed_mem"]},
          "embed_spikes": flat["embed_spikes"],
          "final_residual": flat["final_residual"], "pl_sums": flat["pl_sums"],
          "logits": flat["logits"], "blocks": []}
    b = 0
    while "block%d.in" % b in flat:
        d = {}
        for key in flat.files:
            pref = "block%d." % b
            if key.startswith(pref):
                d[key[len(pref):]] = np.asarray(flat[key])
        tr["blocks"].append(d)
        b += 1
    return tr


def part_torch(a):
    import torch
    from tools.backends.spikformer import config as C
    from tools.spikformer_train import train_nlp as TR

    torch.backends.cuda.matmul.allow_tf32 = False
    full = C.load(a.config)
    dims = {"E": full["embed"], "H": full["heads"], "HD": full["head_dim"],
            "N": full["seq_len"], "T": full["time_steps"], "D": full["depth"],
            "MLP": full["mlp_hidden"], "C": full["num_classes"],
            "vocab_size": full["vocab_size"]}
    run = pathlib.Path(a.run_dir)
    ckpt = run / "train" / ("ckpt_real_%s.npz" % a.tag)
    state = run / "train" / ("best_state_%s.pt" % a.tag)
    z = np.load(str(ckpt))
    thr = {"e": int(z["thr_e"][0])}
    for b in range(dims["D"]):
        for nm in ("q", "k", "v", "proj", "fc1", "fc2"):
            thr["block%d.%s" % (b, nm)] = int(z["block%d.%s_thr" % (b, nm)][0])
    net = TR.Net(dims)
    ck = torch.load(str(state), map_location="cpu", weights_only=False)
    net.load_state_dict(ck["state_dict"])
    net.set_thresholds(thr)
    net.eval()
    cor = np.load(str(run / "corpus" / "corpus_encoded.npz"))
    ids = np.asarray(cor["train_ids"][:a.n_samples], dtype=np.int64)
    print("part torch: state=%s epoch=%s thresholds=%s ids=%s"
          % (state, ck.get("epoch"), json.dumps(thr), ids.tolist()))
    with torch.no_grad():
        logits, tr = net(torch.from_numpy(ids), trace=True)
    flat = flat_from_trace(tr)
    flat["ids"] = ids
    out = run / "reports" / ("train_realweight_torch_%s.npz" % a.tag)
    np.savez(str(out), **flat)
    print("part torch: wrote", out, "keys", len(flat))
    return 0


def part_numpy(a):
    from tools.backends.spikformer import integer_ref as R

    run = pathlib.Path(a.run_dir)
    ckpt = run / "train" / ("ckpt_real_%s.npz" % a.tag)
    tf = run / "reports" / ("train_realweight_torch_%s.npz" % a.tag)
    if not tf.exists():
        raise SystemExit("missing torch trace dump: %s (run --part torch first)" % tf)
    flat = np.load(str(tf))
    ids = np.asarray(flat["ids"], dtype=np.int64)
    ref_params = R.load_checkpoint(str(ckpt))
    cfg = ref_params["cfg"]
    got = R.run_sample(ref_params, ids, cfg=cfg, trace=True)
    torch_tr = unflat_to_trace(flat)
    # integer datapath layers must match EXACTLY; the FP32 head logits are compared with
    # tolerance below (torch/numpy GEMM ordering may differ in the last ulp).
    int_keys = [k for k in R.trace_keys(got) if k != "logits"]
    mism = R.first_mismatch(torch_tr, got, keys=int_keys)
    logits_t = np.asarray(flat["logits"], np.float32)
    logits_r = np.asarray(got["logits"], np.float32)
    pl_equal = bool(np.array_equal(np.asarray(flat["pl_sums"]), got["pl_sums"]))
    logits_close = bool(np.allclose(logits_t, logits_r, rtol=1e-5, atol=1e-5))
    cls_equal = bool(np.array_equal(np.argmax(logits_t, 1), np.argmax(logits_r, 1)))
    n_layers = len([k for k in R.trace_keys(got) if k != "logits"])
    ok = mism is None and pl_equal and logits_close and cls_equal
    report = {
        "schema": "spikformer-nlp-realweight-fidelity-1",
        "tag": a.tag,
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256(ckpt),
        "torch_trace": str(tf),
        "torch_trace_sha256": sha256(tf),
        "n_samples": int(len(ids)),
        "sample_ids": ids.tolist(),
        "layers_compared": n_layers,
        "first_mismatch": mism,
        "pl_sums_equal": pl_equal,
        "logits_close": logits_close,
        "argmax_equal": cls_equal,
        "torch_logits": logits_t.tolist(),
        "integer_ref_logits": logits_r.tolist(),
        "thresholds_checkpoint": {k: int(v) for k, v in
                                  _threshold_dict(ref_params, cfg).items()},
        "PASS": bool(ok),
    }
    rep = run / "reports" / ("train_realweight_fidelity_%s.json" % a.tag)
    rep.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("PASS", "n_samples", "layers_compared",
                                            "first_mismatch", "pl_sums_equal",
                                            "logits_close", "argmax_equal")}, indent=1))
    print("wrote", rep)
    return 0 if ok else 1


def _threshold_dict(params, cfg):
    d = {"embed": int(params["thr_e"])}
    for b in range(cfg["D"]):
        for nm in ("q", "k", "v", "proj", "fc1", "fc2"):
            d["block%d.%s" % (b, nm)] = int(params["blocks"][b][nm + "_thr"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("torch", "numpy"), required=True)
    ap.add_argument("--tag", default="real50")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-samples", type=int, default=2)
    a = ap.parse_args()
    return part_torch(a) if a.part == "torch" else part_numpy(a)


if __name__ == "__main__":
    sys.exit(main())
