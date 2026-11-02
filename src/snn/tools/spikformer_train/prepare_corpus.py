"""Prepare the real ChnSentiCorp corpus for the 48h Spikformer NLP run.

Frozen data facts (verified by Lead + this script):
  train.tsv header 'label\ttext_a'      (9600 labelled rows)
  dev.tsv   header 'qid\tlabel\ttext_a' (1200 labelled rows)
  test.tsv  header 'qid\ttext_a'        (1200 rows, NO LABEL -> never reported)
Official archive: https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip
  MD5 7ef61b08ad10fbddf2ba97613f071561 (must match, hard fail otherwise).

Steps: verify archive -> extract -> parse each schema explicitly -> fixed stratified
holdout from the labelled train split (indices recorded + hashed) -> vocab from the
TRAIN-TRAIN portion only (runtime/spikformer/tokenize.py build_vocab) -> encode fixed
int64 arrays [n, seq_len] once (pad/truncate, PAD=0 UNK=1, no ord(ch)%3999) -> write
corpus_meta.json + vocab.json + corpus_encoded.npz.

Splits: train (labelled, 90% of official train), holdout (fixed stratified 10% of
official train, labelled), dev (official dev, labelled, used for early stopping),
test_ids (official test ids only, unlabelled; never used for accuracy).

Usage:
  E:/miniconda/python.exe -B tools/spikformer_train/prepare_corpus.py \
      --zip build/spikformer/nlp_48h_20261029_1915/corpus/ChnSentiCorp.zip \
      --out build/spikformer/nlp_48h_20261029_1915/corpus \
      --vocab-size 4000 --seq-len 32 --seed 20261029
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import zipfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.spikformer import tokenize as TK  # noqa: E402

ARCHIVE_MD5 = '7ef61b08ad10fbddf2ba97613f071561'
ARCHIVE_URL = 'https://bj.bcebos.com/paddlenlp/datasets/ChnSentiCorp.zip'


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def parse_train(path):
    """header 'label\ttext_a'."""
    rows = []
    lines = pathlib.Path(path).read_text(encoding='utf-8').splitlines()
    if not lines:
        raise SystemExit('empty file: %s' % path)
    header = lines[0].split('\t')
    if header != ['label', 'text_a']:
        raise SystemExit('train.tsv header %r != [label, text_a]' % header)
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split('\t')
        if len(p) != 2:
            raise SystemExit('train.tsv bad row (%d cols): %r' % (len(p), ln[:80]))
        rows.append((int(p[0]), p[1]))
    return rows


def parse_dev(path):
    """header 'qid\tlabel\ttext_a'."""
    rows = []
    lines = pathlib.Path(path).read_text(encoding='utf-8').splitlines()
    header = lines[0].split('\t')
    if header != ['qid', 'label', 'text_a']:
        raise SystemExit('dev.tsv header %r != [qid, label, text_a]' % header)
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split('\t')
        if len(p) != 3:
            raise SystemExit('dev.tsv bad row (%d cols): %r' % (len(p), ln[:80]))
        rows.append((int(p[1]), p[2]))
    return rows


def parse_test(path):
    """header 'qid\ttext_a' -> unlabelled texts only (never used for accuracy)."""
    rows = []
    lines = pathlib.Path(path).read_text(encoding='utf-8').splitlines()
    header = lines[0].split('\t')
    if header != ['qid', 'text_a']:
        raise SystemExit('test.tsv header %r != [qid, text_a] (unlabelled expected)'
                         % header)
    for ln in lines[1:]:
        if not ln.strip():
            continue
        p = ln.split('\t')
        if len(p) != 2:
            raise SystemExit('test.tsv bad row (%d cols): %r' % (len(p), ln[:80]))
        rows.append(p[1])
    return rows


def stratified_holdout(labels, frac, seed):
    """Deterministic per-class holdout; returns (holdout_idx sorted, train_idx)."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.RandomState(seed)
    hold = []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        k = int(round(frac * len(idx)))
        hold.extend(idx[:k].tolist())
    hold = np.array(sorted(hold), dtype=np.int64)
    mask = np.ones(len(labels), dtype=bool)
    mask[hold] = False
    return hold, np.where(mask)[0]


def encode_all(texts, stoi, seq_len):
    ids = np.zeros((len(texts), seq_len), dtype=np.int64)
    lens = np.zeros(len(texts), dtype=np.int64)
    n_unk = 0
    for i, t in enumerate(texts):
        chars = list(t)
        lens[i] = len(chars)
        for j, c in enumerate(chars[:seq_len]):
            k = stoi.get(c, 1)
            if k == 1:
                n_unk += 1
            ids[i, j] = k
    return ids, lens, n_unk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--vocab-size', type=int, default=4000)
    ap.add_argument('--seq-len', type=int, default=32)
    ap.add_argument('--holdout-frac', type=float, default=0.10)
    ap.add_argument('--seed', type=int, default=20261029)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    zpath = pathlib.Path(args.zip)

    # 1) archive integrity
    got_md5 = md5(zpath)
    if got_md5 != ARCHIVE_MD5:
        raise SystemExit('archive MD5 mismatch: got %s want %s (official PaddleNLP)'
                         % (got_md5, ARCHIVE_MD5))
    arch_sha = sha256(zpath)
    print('archive MD5 ok=%s sha256=%s bytes=%d' % (got_md5, arch_sha, zpath.stat().st_size))

    # 2) extract
    dst = out / 'ChnSentiCorp'
    if not (dst / 'train.tsv').exists():
        with zipfile.ZipFile(str(zpath)) as z:
            z.extractall(str(out))
    train_rows = parse_train(dst / 'train.tsv')
    dev_rows = parse_dev(dst / 'dev.tsv')
    test_texts = parse_test(dst / 'test.tsv')
    print('rows: train=%d dev=%d test(unlabelled)=%d'
          % (len(train_rows), len(dev_rows), len(test_texts)))

    # 3) fixed stratified holdout from the labelled train split
    tr_labels_all = np.array([r[0] for r in train_rows], dtype=np.int64)
    hold_idx, train_idx = stratified_holdout(tr_labels_all, args.holdout_frac, args.seed)
    hold_idx_list = [int(i) for i in hold_idx]
    hold_idx_sha = hashlib.sha256(
        json.dumps(hold_idx_list).encode()).hexdigest()
    train_texts = [train_rows[int(i)][1] for i in train_idx]
    train_labels = tr_labels_all[train_idx]
    hold_texts = [train_rows[int(i)][1] for i in hold_idx]
    hold_labels = tr_labels_all[hold_idx]
    print('holdout: %d fixed stratified indices from train (seed=%d) sha256=%s'
          % (len(hold_idx), args.seed, hold_idx_sha))

    # 4) vocab from TRAIN-TRAIN only
    stoi = TK.build_vocab(train_texts, vocab_size=args.vocab_size)
    if len(stoi) > args.vocab_size:
        raise SystemExit('vocab larger than requested: %d > %d' % (len(stoi), args.vocab_size))
    vocab_path = out / 'vocab.json'
    TK.save_vocab(stoi, str(vocab_path))
    vocab_sha = sha256(vocab_path)
    print('vocab: %d entries (requested %d) -> %s sha256=%s'
          % (len(stoi), args.vocab_size, vocab_path, vocab_sha))

    # 5) encode fixed arrays once
    seq = args.seq_len
    train_ids, train_lens, train_unk = encode_all(train_texts, stoi, seq)
    hold_ids, hold_lens, hold_unk = encode_all(hold_texts, stoi, seq)
    dev_texts = [r[1] for r in dev_rows]
    dev_labels = np.array([r[0] for r in dev_rows], dtype=np.int64)
    dev_ids, dev_lens, dev_unk = encode_all(dev_texts, stoi, seq)
    test_ids, test_lens, test_unk = encode_all(test_texts, stoi, seq)

    def lenstats(lens, unk, n):
        return {'n': int(n), 'mean_len': float(lens.mean()),
                'max_len': int(lens.max()), 'n_truncated': int((lens > seq).sum()),
                'n_empty': int((lens == 0).sum()),
                'unk_rate': float(unk) / float(max(1, int(lens.clip(max=seq).sum())))}

    npz = out / 'corpus_encoded.npz'
    np.savez(str(npz),
             train_ids=train_ids, train_y=train_labels.astype(np.int64),
             holdout_ids=hold_ids, holdout_y=hold_labels.astype(np.int64),
             holdout_train_index=np.array(hold_idx_list, dtype=np.int64),
             dev_ids=dev_ids, dev_y=dev_labels,
             test_ids=test_ids,
             vocab_size_actual=np.array([len(stoi)], dtype=np.int64))
    npz_sha = sha256(npz)
    meta = {
        'schema': 'spikformer-nlp-corpus-1',
        'archive': {'url': ARCHIVE_URL, 'md5_expected': ARCHIVE_MD5,
                    'md5_actual': got_md5, 'sha256': arch_sha,
                    'bytes': int(zpath.stat().st_size), 'file': str(zpath.name)},
        'files': {n: {'sha256': sha256(dst / n), 'rows': r}
                  for n, r in (('train.tsv', len(train_rows)),
                               ('dev.tsv', len(dev_rows)),
                               ('test.tsv', len(test_texts)))},
        'schemas': {'train.tsv': 'label\\ttext_a',
                    'dev.tsv': 'qid\\tlabel\\ttext_a',
                    'test.tsv': 'qid\\ttext_a (UNLABELLED; never used for accuracy)'},
        'splits': {
            'train': {'n': int(len(train_labels)),
                      'label_counts': np.bincount(train_labels, minlength=2).tolist(),
                      'source': 'official train.tsv minus fixed holdout',
                      'len_stats': lenstats(train_lens, train_unk, len(train_ids))},
            'holdout': {'n': int(len(hold_labels)),
                        'label_counts': np.bincount(hold_labels, minlength=2).tolist(),
                        'source': 'fixed stratified %.2f of official train.tsv' % args.holdout_frac,
                        'seed': args.seed, 'holdout_train_index_sha256': hold_idx_sha,
                        'holdout_train_index_count': len(hold_idx_list),
                        'len_stats': lenstats(hold_lens, hold_unk, len(hold_ids))},
            'dev': {'n': int(len(dev_labels)),
                    'label_counts': np.bincount(dev_labels, minlength=2).tolist(),
                    'source': 'official dev.tsv (early stopping, labelled)',
                    'len_stats': lenstats(dev_lens, dev_unk, len(dev_ids))},
            'test': {'n': int(len(test_ids)), 'source': 'official test.tsv',
                     'labelled': False,
                     'note': 'no accuracy reported on test (unlabelled)',
                     'len_stats': lenstats(test_lens, test_unk, len(test_ids))},
        },
        'vocab': {'file': str(vocab_path.name), 'sha256': vocab_sha,
                  'size_actual': int(len(stoi)), 'size_requested': args.vocab_size,
                  'built_from': 'train split only (official train.tsv minus holdout)',
                  'pad_id': TK.PAD, 'unk_id': TK.UNK},
        'encoding': {'seq_len': seq, 'dtype': 'int64', 'pad_id': 0, 'unk_id': 1,
                     'truncate': 'keep first seq_len chars', 'padding': 'right',
                     'no_ord_mod': 'training no longer uses ord(ch)%3999; ids come from '
                                   'runtime/spikformer/tokenize.py vocab'},
        'npz': {'file': str(npz.name), 'sha256': npz_sha,
                'arrays': ['train_ids', 'train_y', 'holdout_ids', 'holdout_y',
                           'holdout_train_index', 'dev_ids', 'dev_y', 'test_ids',
                           'vocab_size_actual']},
    }
    meta_path = out / 'corpus_meta.json'
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding='utf-8')
    print('wrote', npz, 'sha256', npz_sha)
    print('wrote', vocab_path, 'and', meta_path)
    print('split sizes: train=%d holdout=%d dev=%d test_unlabelled=%d'
          % (len(train_ids), len(hold_ids), len(dev_ids), len(test_ids)))
    print('label counts: train=%s holdout=%s dev=%s'
          % (meta['splits']['train']['label_counts'],
             meta['splits']['holdout']['label_counts'],
             meta['splits']['dev']['label_counts']))


if __name__ == '__main__':
    main()
