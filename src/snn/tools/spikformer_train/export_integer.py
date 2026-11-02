"""Integer-domain calibration + final frozen export + integer accuracy + strict compile import.

Runs in a PURE-NumPy process (no torch): stage 1 of the export is
  python -B tools/spikformer_train/train_nlp.py --stage export --state ... --tag ...
which only dumps the candidate frozen npz from the torch best_state. This script then
  1. loads the candidate via integer_ref.load_checkpoint (strict dtype/shape checks),
  2. runs the focused integer-domain calibration (2 rounds, ONE rule per LIF: median of
     positive membrane values from exact integer_ref traces; thr>=1; dead LIF => error;
     warns if thr > max positive delta),
  3. writes the final checkpoint npz exactly per contract
     export_abi.checkpoint_npz_required_exact (+ embed_scale for scale provenance),
  4. reloads it through integer_ref (strict) and reports accuracy on dev + the fixed
     holdout (integer model, not torch; test.tsv is unlabelled and never used),
  5. writes vocab/label-map/split provenance and runs the strict importer
     weights.import_checkpoint + compile.run --weights to prove the param package builds.

Usage:
  E:/miniconda/python.exe -B tools/spikformer_train/export_integer.py --candidate \
      build/spikformer/nlp_48h_20261029_1915/train/ckpt_candidate_real50.npz \
      --tag real50 --run-dir build/spikformer/nlp_48h_20261029_1915 \
      --config examples/spikformer/config/nlp_sentiment_b1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.backends.spikformer import config as C            # noqa: E402
from tools.backends.spikformer import integer_ref as R      # noqa: E402
from tools.spikformer_train import integer_calib as ICAL    # noqa: E402

LIN = ("q", "k", "v", "proj", "fc1", "fc2")


def sha256(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def write_final_npz(params, meta, cfg, out_path):
    """Write the frozen checkpoint npz per contract checkpoint_npz_required_exact."""
    arrs = {
        "etab": np.asarray(params["etab"], np.int8),
        "ptab": np.asarray(params["ptab"], np.int8),
        "thr_e": np.array([int(params["thr_e"])], np.int32),
        "head_w": np.asarray(params["head_w"], np.float32),
        "head_b": np.asarray(params["head_b"], np.float32),
        "meta_bn_state": np.array("absent"),
        "meta_model_kind": np.array(meta["model_kind"]),
        "meta_corpus_sha256": np.array(meta["corpus_sha256"]),
        "meta_vocab_sha256": np.array(meta["vocab_sha256"]),
        "meta_cfg": np.array(json.dumps({k: int(cfg[k]) for k in
                                        ("E", "H", "HD", "N", "T", "D", "MLP", "C",
                                         "vocab_size")})),
        "embed_scale": np.array([float(params.get("embed_scale", 1.0))], np.float32),
    }
    for b, blk in enumerate(params["blocks"]):
        p = "block%d." % b
        for nm in LIN:
            arrs[p + nm + "_w"] = np.asarray(blk[nm + "_w"], np.int8)
            arrs[p + nm + "_b"] = np.asarray(blk[nm + "_b"], np.int32)
            arrs[p + nm + "_scale"] = np.array([float(blk[nm + "_scale"])], np.float32)
            arrs[p + nm + "_thr"] = np.array([int(blk[nm + "_thr"])], np.int32)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **arrs)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--tag", default="real50")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--calib-samples", type=int, default=64)
    ap.add_argument("--calib-rounds", type=int, default=2)
    ap.add_argument("--compile-out", default=None)
    ap.add_argument("--skip-compile", action="store_true")
    ap.add_argument("--train-eval-samples", type=int, default=512)
    a = ap.parse_args()
    t0 = time.time()
    run = pathlib.Path(a.run_dir)
    train_dir = run / "train"
    rep_dir = run / "reports"
    corpus_dir = run / "corpus"
    rep_dir.mkdir(parents=True, exist_ok=True)
    full = C.load(a.config)
    dims = {"E": full["embed"], "H": full["heads"], "HD": full["head_dim"],
            "N": full["seq_len"], "T": full["time_steps"], "D": full["depth"],
            "MLP": full["mlp_hidden"], "C": full["num_classes"],
            "vocab_size": full["vocab_size"]}

    corpus_npz = corpus_dir / "corpus_encoded.npz"
    cor = np.load(str(corpus_npz))
    corpus_meta = json.loads((corpus_dir / "corpus_meta.json").read_text(encoding="utf-8"))
    sidecar = json.loads((train_dir / ("export_candidate_%s.json" % a.tag))
                         .read_text(encoding="utf-8"))
    meta = dict(sidecar["meta"])
    meta["corpus_sha256"] = corpus_meta["archive"]["sha256"]
    meta["vocab_sha256"] = corpus_meta["vocab"]["sha256"]
    meta.setdefault("model_kind", "real_corpus")
    if meta["model_kind"] != "real_corpus":
        raise SystemExit("final export requires meta_model_kind=real_corpus, got %r"
                         % meta["model_kind"])
    print("candidate:", a.candidate, "sha256:", sha256(a.candidate))
    print("corpus:", corpus_meta["archive"]["sha256"], "vocab:",
          corpus_meta["vocab"]["sha256"], "vocab_size:",
          corpus_meta["vocab"]["size_actual"])

    params = R.load_checkpoint(a.candidate, cfg=dims)
    print("candidate loaded: blocks=%d etab=%s ptab=%s head_w=%s"
          % (len(params["blocks"]), params["etab"].shape, params["ptab"].shape,
             params["head_w"].shape))

    calib_ids = np.asarray(cor["train_ids"][:a.calib_samples])
    before = {"thr_e": int(params["thr_e"])}
    for b in range(dims["D"]):
        for nm in LIN:
            before["block%d.%s" % (b, nm)] = int(params["blocks"][b][nm + "_thr"])
    cal = ICAL.calibrate_thresholds(params, calib_ids, dims, rounds=a.calib_rounds)
    after = cal["thresholds"]
    print("calibrated thresholds:", json.dumps(after))
    print("activity (64-sample calib):", json.dumps(cal["final_activity"]))

    out_npz = train_dir / ("ckpt_real_%s.npz" % a.tag)
    write_final_npz(params, meta, dims, out_npz)
    ckpt_sha = sha256(out_npz)
    print("wrote FINAL integer checkpoint", out_npz, "sha256", ckpt_sha)

    ip = R.load_checkpoint(str(out_npz), cfg=dims)
    ev = {}
    t_eval = time.time()
    for split in ("dev", "holdout"):
        ev[split] = R.evaluate(ip, np.asarray(cor[split + "_ids"]),
                               np.asarray(cor[split + "_y"]), cfg=dims)
    n_sub = min(a.train_eval_samples, len(cor["train_ids"]))
    ev["train_subset"] = R.evaluate(ip, np.asarray(cor["train_ids"][:n_sub]),
                                    np.asarray(cor["train_y"][:n_sub]), cfg=dims)
    eval_sec = round(time.time() - t_eval, 2)
    acc_dev, acc_hold = ev["dev"]["acc"], ev["holdout"]["acc"]
    maj_dev, maj_hold = ev["dev"]["majority_acc"], ev["holdout"]["majority_acc"]
    established = bool(acc_dev > maj_dev and acc_hold > maj_hold)
    print("INTEGER accuracy: dev=%.4f (majority %.4f, n=%d) holdout=%.4f (majority %.4f, n=%d)"
          % (acc_dev, maj_dev, ev["dev"]["n"], acc_hold, maj_hold, ev["holdout"]["n"]))
    print("dev confusion:", ev["dev"]["confusion"], "class_counts:",
          ev["dev"]["class_counts"], "eval_sec:", eval_sec)
    print("accuracy_established:", established)

    label_map = {
        "schema": "spikformer-nlp-labelmap-1",
        "num_classes": 2,
        "labels": {"0": "negative", "1": "positive"},
        "source": "ChnSentiCorp label column (0=negative, 1=positive sentiment)",
        "note": "public test.tsv is unlabelled; no test accuracy is reported",
    }
    (corpus_dir / "label_map.json").write_text(json.dumps(label_map, indent=1),
                                               encoding="utf-8")
    splits = {
        "schema": "spikformer-nlp-splits-1",
        "corpus_npz_sha256": sha256(corpus_npz),
        "corpus_meta_sha256": sha256(corpus_dir / "corpus_meta.json"),
        "vocab_sha256": corpus_meta["vocab"]["sha256"],
        "vocab_size_actual": corpus_meta["vocab"]["size_actual"],
        "archive_sha256": corpus_meta["archive"]["sha256"],
        "seq_len": dims["N"],
        "splits": {k: {"n": corpus_meta["splits"][k]["n"],
                       "label_counts": corpus_meta["splits"][k].get("label_counts")}
                   for k in ("train", "holdout", "dev", "test")},
        "holdout_train_index_sha256":
            corpus_meta["splits"]["holdout"]["holdout_train_index_sha256"],
        "holdout_seed": corpus_meta["splits"]["holdout"]["seed"],
        "test_labelled": False,
    }
    (corpus_dir / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
    vocab_path = corpus_dir / "vocab.json"
    extra = {
        "vocab": {"path": str(vocab_path), "sha256": sha256(vocab_path),
                  "size_actual": corpus_meta["vocab"]["size_actual"],
                  "size_requested": corpus_meta["vocab"]["size_requested"],
                  "pad_id": 0, "unk_id": 1},
        "label_map": {"path": str(corpus_dir / "label_map.json"),
                      "sha256": sha256(corpus_dir / "label_map.json")},
        "splits": {"path": str(corpus_dir / "splits.json"),
                   "sha256": sha256(corpus_dir / "splits.json"),
                   "holdout_train_index_sha256":
                       splits["holdout_train_index_sha256"]},
        "calibration": {"samples": int(len(calib_ids)), "rounds": a.calib_rounds,
                        "rule": cal["rule"], "thresholds_before": before,
                        "thresholds_after": after,
                        "activity": cal["final_activity"],
                        "min_activity": cal["min_activity"],
                        "warn": cal["warn"]},
        "training": {"state": sidecar.get("state"),
                     "state_epoch": sidecar.get("state_epoch"),
                     "thresholds_train_time": sidecar.get("thresholds_train_time")},
    }

    # companion asset binding next to the checkpoint, auto-discovered by compile.run
    # (so the official thin entry -- python -m tools.toolchain compile --weights ... --
    # produces the same complete manifest).
    assets_path = out_npz.with_name(out_npz.stem + ".assets.json")
    assets_path.write_text(json.dumps(extra, indent=1))
    print("wrote companion assets", assets_path, "sha256", sha256(assets_path))

    compile_info = {"status": "not_run"}
    if not a.skip_compile:
        from tools.backends.spikformer import compile as COMP
        cdir = pathlib.Path(a.compile_out or (train_dir / "nlp_real_compile"))
        if cdir.exists():
            raise SystemExit("compile out dir already exists: %s" % cdir)
        man = COMP.run(a.config, cdir, weights=str(out_npz), assets_extra=extra)
        compile_info = {"status": "pass", "out_dir": str(cdir),
                        "manifest": str(cdir / "manifest.json"),
                        "manifest_sha256": man["manifest_sha256"],
                        "model_kind": man["model_kind"],
                        "n_params_blobs": len(man["params"]),
                        "numeric_contract": str(cdir / "numeric_contract.json"),
                        "thresholds_bound": man["numeric"]["thresholds"]}
        print("compile import PASS:", compile_info["manifest"],
              "model_kind:", man["model_kind"])

    report = {
        "stage": "export_integer_final",
        "tag": a.tag,
        "dims": dims,
        "checkpoint": str(out_npz),
        "checkpoint_sha256": ckpt_sha,
        "candidate": a.candidate,
        "candidate_sha256": sha256(a.candidate),
        "corpus_npz_sha256": sha256(corpus_npz),
        "vocab_sha256": corpus_meta["vocab"]["sha256"],
        "meta_model_kind": meta["model_kind"],
        "calibration": {"samples": int(len(calib_ids)), "rounds": a.calib_rounds,
                        "rule": cal["rule"], "thresholds_before": before,
                        "thresholds_after": after,
                        "activity": cal["final_activity"],
                        "min_activity": cal["min_activity"],
                        "dead_lifs": cal["dead_lifs"], "warn": cal["warn"]},
        "integer_ref_thresholds_verified": bool(min(
            [after["thr_e"]] + [after["block%d" % b][nm]
                                for b in range(dims["D"]) for nm in LIN]) >= 1),
        "eval_integer_ref": {k: {kk: vv for kk, vv in v.items() if kk != "pred"}
                             for k, v in ev.items()},
        "confusion": {k: ev[k]["confusion"] for k in ("dev", "holdout", "train_subset")},
        "class_counts": {k: ev[k]["class_counts"] for k in
                          ("dev", "holdout", "train_subset")},
        "majority_baseline": {"dev": maj_dev, "holdout": maj_hold},
        "accuracy_established": established,
        "accuracy_established_rule":
            "dev_acc > dev_majority AND holdout_acc > holdout_majority",
        "test_accuracy": "not_applicable (public test.tsv unlabelled; never used)",
        "eval_sec": eval_sec,
        "compile": compile_info,
        "assets": {"vocab": extra["vocab"], "label_map": extra["label_map"],
                   "splits": extra["splits"]},
        "wall_s": round(time.time() - t0, 2),
    }
    rep = rep_dir / ("export_final_%s.json" % a.tag)
    rep.write_text(json.dumps(report, indent=1))
    (rep_dir / ("export_final_%s.log" % a.tag)).write_text(json.dumps(report, indent=1))
    summary = {"checkpoint": str(out_npz), "checkpoint_sha256": ckpt_sha,
               "dev_acc": acc_dev, "dev_majority": maj_dev,
               "holdout_acc": acc_hold, "holdout_majority": maj_hold,
               "confusion_dev": ev["dev"]["confusion"],
               "accuracy_established": established,
               "compile": compile_info, "report": str(rep)}
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
