"""Deterministic DEVELOPMENT-FIXTURE checkpoint for the 48h run.

Explicitly NOT a real model: meta_model_kind=development_fixture, meta corpus
hash 'fixture'. Only used to probe the pipeline (compile/import/replay) before
the real trained checkpoint exists; gates must never claim accuracy from it.

Keys follow contractor export_abi.checkpoint_npz_required_exact so that
integer_ref.load_checkpoint and the updated weights.py import work unchanged.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np

LINEARS = ('q', 'k', 'v', 'proj', 'fc1', 'fc2')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', required=True)
    ap.add_argument('--thr', type=int, default=8)
    a = ap.parse_args(argv)
    cfg = json.loads(pathlib.Path(a.config).read_text(encoding='utf-8-sig'))
    E, H, HD = cfg['embed'], cfg['heads'], cfg['head_dim']
    MLP, D, T = cfg['mlp_hidden'], cfg['depth'], cfg['time_steps']
    V, N, C = cfg['vocab_size'], cfg['seq_len'], cfg['num_classes']
    rng = np.random.RandomState(a.seed)
    out = {}
    out['etab'] = rng.randint(-127, 128, size=(V, E)).astype(np.int8)
    out['ptab'] = rng.randint(-127, 128, size=(N, E)).astype(np.int8)
    out['thr_e'] = np.array([max(1, a.thr)], np.int32)
    for b in range(D):
        p = 'block%d.' % b
        for nm in LINEARS:
            shp = {'fc1': (MLP, E), 'fc2': (E, MLP)}.get(nm, (E, E))
            out[p + nm + '_w'] = rng.randint(-127, 128, size=shp).astype(np.int8)
            out[p + nm + '_b'] = rng.randint(-64, 65, size=(shp[0],)).astype(np.int32)
            out[p + nm + '_scale'] = np.array([float(rng.uniform(0.5, 2.0))], np.float32)
            out[p + nm + '_thr'] = np.array([max(1, a.thr + int(rng.randint(0, 8)))], np.int32)
    out['head_w'] = (rng.randn(C, E) * 0.02).astype(np.float32)
    out['head_b'] = np.zeros(C, np.float32)
    dims = {'E': E, 'H': H, 'HD': HD, 'N': N, 'T': T, 'D': D, 'MLP': MLP,
            'C': C, 'vocab_size': V}
    out['meta_cfg'] = np.array(json.dumps(dims, sort_keys=True))
    out['meta_bn_state'] = np.array('absent')
    out['meta_model_kind'] = np.array('development_fixture')
    out['meta_corpus_sha256'] = np.array('fixture')
    out['meta_vocab_sha256'] = np.array('fixture')
    dst = pathlib.Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(dst), **out)
    print('wrote %s (DEVELOPMENT FIXTURE, not a real model; %d arrays, seed=%d)'
          % (dst, len(out), a.seed))


if __name__ == '__main__':
    main()
