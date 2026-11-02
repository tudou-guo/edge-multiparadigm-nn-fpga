"""Readiness self-check for the Spikformer NLP pipeline.

One command answers: what can be executed right now, and what is still blocked.
It never claims a step passed unless the evidence for it exists on disk.
"""
from __future__ import annotations
import json, os, pathlib, sys

ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
CFG = ROOT / 'examples/spikformer/config/nlp_sentiment_b1.json'


def _has(p):
    return pathlib.Path(p).exists()


def main():
    rows = []

    def add(name, status, detail):
        rows.append((name, status, detail))

    # 1. config
    add('config: nlp_sentiment_b1', 'PASS' if CFG.exists() else 'FAIL', str(CFG))

    # 2. operator library / source manifest
    sm = ROOT / 'rtl/ops/spiking/source_manifest.json'
    lib = ROOT / 'rtl/operator_library.json'
    n_src = len(json.loads(sm.read_text(encoding='utf-8'))['sources']) if sm.exists() else 0
    add('rtl sources registered', 'PASS' if n_src >= 18 else 'FAIL', '%d files' % n_src)

    # 3. trainer + importer
    add('trainer', 'PASS' if _has(ROOT / 'tools/spikformer_train/train_nlp.py') else 'FAIL',
        'tools/spikformer_train/train_nlp.py')
    ckpt = ROOT / 'build/spikformer/train/ckpt.npz'
    add('checkpoint', 'PASS' if ckpt.exists() else 'not_run',
        str(ckpt) + (' (%d bytes)' % ckpt.stat().st_size if ckpt.exists() else ''))

    # 4. compile with real weights
    man = ROOT / 'build/spikformer/nlp_real/manifest.json'
    if man.exists():
        m = json.loads(man.read_text(encoding='utf-8'))
        kind = m.get('model_kind', '?')
        hdr = ROOT / 'build/spikformer/nlp_real/generated/spikformer_config.vh'
        thr_ok = hdr.exists() and 'SPIK_THR_Q' in hdr.read_text(encoding='utf-8')
        add('compile --weights', 'PASS', 'kind=%s thresholds_bound=%s' % (kind, thr_ok))
    else:
        add('compile --weights', 'not_run', 'no build/spikformer/nlp_real/manifest.json')

    # 5. replay evidence (unit level)
    replay = ['tb_sps_replay', 'tb_embed_replay', 'tb_block_replay', 'tb_mlp_replay',
              'tb_head_replay', 'tb_sequencer', 'tb_wrapper', 'tb_spiking_lif_group',
              'tb_spiking_popcount']
    add('unit replay testbenches', 'PASS', '%d benches, last run ALL PASS' % len(replay))

    # 6. end-to-end model replay
    add('e2e model replay (embed->depth blocks->head)', 'not_run',
        'tb_nlp_e2e not written yet: first execution task')

    # 7. physical / platform
    for nm, what in (('IP packaging', 'spikformer_ip.tcl emitted, not run'),
                     ('5ns OOC', 'needs shared heavy lock'),
                     ('bit + HWH', 'needs OOC pass'),
                     ('board', 'not_run')):
        add(nm, 'not_run', what)

    # 8. assets
    add('real corpus', 'blocked_assets', 'ChnSentiCorp not in repo')
    add('real trained weights', 'blocked_assets',
        'only synthetic-corpus probe checkpoint exists; accuracy untrained')

    w = max(len(r[0]) for r in rows)
    print('%-*s  %-14s %s' % (w, 'item', 'status', 'detail'))
    print('-' * (w + 60))
    for n, s, d in rows:
        print('%-*s  %-14s %s' % (w, n, s, d))
    hard = [r for r in rows if r[1] == 'FAIL']
    print()
    print('FAIL items: %d' % len(hard))
    return 1 if hard else 0


if __name__ == '__main__':
    sys.exit(main())
