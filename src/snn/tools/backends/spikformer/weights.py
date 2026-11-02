"""Weight import/export. Deterministic, from-empty-file, no append.
BN state explicit: already_folded | requires_fold | absent.

Two paths:
  * fixture path uses a seeded RNG (manifest model_kind=development_fixture);
  * real checkpoint path reads the frozen npz produced by the trainer
    (contract export_abi.checkpoint_npz_required_exact: etab/ptab/thr_e shape(1,)/
    block{b}.{q,k,v,proj,fc1,fc2}_{w int8, b int32, scale float32 shape(1,),
    thr int32 shape(1,)}/head_w/head_b/meta_*). Keys are validated strictly:
    dtype, shape, threshold>=1, int8 range [-127,127], finite scales/head (NO silent
    casts, NO strict=False, NO defaults for data blobs).

    RTL blob namespace is unchanged: etab -> embed_w, ptab -> pos_w. Per-block
    thresholds+scales and the FP32 head params are emitted into params/ as real
    blobs and bound into the manifest by compile.py (contract param_blob_mapping).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct

import numpy as np

BN_STATES = ('already_folded', 'requires_fold', 'absent')
LIN_NAMES = ('q', 'k', 'v', 'proj', 'fc1', 'fc2')
NPZ_REQUIRED = ('etab', 'ptab', 'thr_e', 'head_w', 'head_b',
                'meta_bn_state', 'meta_model_kind', 'meta_corpus_sha256',
                'meta_vocab_sha256', 'meta_cfg')


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _emit(out, name, arr):
    """Write one blob as .bin + .coe, always from an empty file (deterministic)."""
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in (np.int8, np.int32, np.float32):
        raise TypeError('unsupported blob dtype %s for %r (want int8/int32/float32)'
                        % (arr.dtype, name))
    fp = out / (name + '.bin')
    if fp.exists():
        fp.unlink()  # from-empty: never append
    arr.tofile(str(fp))
    coe = out / (name + '.coe')
    if coe.exists():
        coe.unlink()
    words = arr.reshape(-1)
    with open(coe, 'w') as f:
        f.write('; deterministic export, %d elements, dtype=%s\n' % (words.size, arr.dtype))
        f.write('memory_initialization_radix=16;\n')
        f.write('memory_initialization_vector=\n')
        if arr.dtype == np.int8:
            vals = ['%02x' % (int(v) & 0xFF) for v in words]
        elif arr.dtype == np.int32:
            vals = ['%08x' % (int(v) & 0xFFFFFFFF) for v in words]
        else:  # float32: raw IEEE-754 bit pattern, same bytes as the .bin
            vals = ['%08x' % int(v) for v in words.view(np.uint32)]
        f.write(',\n'.join(vals) + ';\n')
    return {'bin': fp.name, 'coe': coe.name, 'sha256': _sha(str(fp)),
            'shape': list(arr.shape), 'dtype': str(arr.dtype)}


def _required_keys(cfg):
    """Canonical blob names for this config (same namespace as the fixture)."""
    E, MLP, D = cfg['embed'], cfg['mlp_hidden'], cfg['depth']
    keys = ['sps_conv1_w', 'sps_conv1_b', 'sps_conv2_w', 'sps_conv2_b']
    if cfg.get('family', '') in ('spikformer-nlp-sentiment', 'spikformer-nlp-intent',
                                 'spikformer-nlp-match'):
        keys += ['embed_w', 'pos_w']
    for b in range(D):
        p = 'block%d.' % b
        keys += [p + k for k in ('q_w', 'k_w', 'v_w', 'proj_w', 'fc1_w', 'fc2_w',
                                 'q_b', 'k_b', 'v_b', 'proj_b', 'fc1_b', 'fc2_b')]
    return keys


def _shapes(cfg):
    E, MLP, D = cfg['embed'], cfg['mlp_hidden'], cfg['depth']
    sps = cfg['derived'].get('sps')
    if sps:
        c1, c2, c0 = sps['c1'], sps['c2'], sps['c0']
        sh = {'sps_conv1_w': (c1, c0, 3, 3), 'sps_conv1_b': (c1,),
              'sps_conv2_w': (c2, c1, 3, 3), 'sps_conv2_b': (c2,)}
    else:
        sh = {'sps_conv1_w': (48, 3, 3, 3), 'sps_conv1_b': (48,)}
    if cfg.get('family', '') in ('spikformer-nlp-sentiment', 'spikformer-nlp-intent',
                                 'spikformer-nlp-match'):
        sh['embed_w'] = (cfg['vocab_size'], E)
        sh['pos_w'] = (cfg['seq_len'], E)
    for b in range(D):
        p = 'block%d.' % b
        sh[p + 'q_w'] = (E, E); sh[p + 'k_w'] = (E, E)
        sh[p + 'v_w'] = (E, E); sh[p + 'proj_w'] = (E, E)
        sh[p + 'fc1_w'] = (MLP, E); sh[p + 'fc2_w'] = (E, MLP)
        for k in ('q_b', 'k_b', 'v_b', 'proj_b', 'fc2_b'):
            sh[p + k] = (E,)
        sh[p + 'fc1_b'] = (MLP,)
    return sh


def fixture_params(cfg, seed, out_dir):
    rng = np.random.RandomState(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sh = _shapes(cfg)
    blobs = {}
    for name, shape in sh.items():
        if name.endswith('_w'):
            blobs[name] = rng.randint(-128, 128, size=shape, dtype=np.int16).astype(np.int8)
        else:
            blobs[name] = rng.randint(-2048, 2048, size=shape).astype(np.int32)
    files = {}
    for name, arr in blobs.items():
        files[name] = _emit(out, name, arr)
    return files


# ---------------------------------------------------------------------------
# strict real-checkpoint import (contract export_abi.checkpoint_npz_required_exact)
# ---------------------------------------------------------------------------
def _get(z, key, shape, dtype, what):
    if key not in z.files:
        raise RuntimeError('checkpoint missing required key %r (%s)' % (key, what))
    a = np.asarray(z[key])
    if a.dtype != np.dtype(dtype):
        raise TypeError('key %r must be %s, got %s (no silent casts)'
                        % (key, np.dtype(dtype).name, a.dtype))
    if tuple(a.shape) != tuple(shape):
        raise RuntimeError('key %r shape %s != %s' % (key, tuple(a.shape), tuple(shape)))
    return a


def _scalar_i32(z, key, lo=None):
    a = _get(z, key, (1,), np.int32, 'scalar int32')
    v = int(a[0])
    if lo is not None and v < lo:
        raise RuntimeError('key %r = %d < %d' % (key, v, lo))
    return v


def _scalar_f32(z, key):
    a = _get(z, key, (1,), np.float32, 'scalar float32 scale')
    v = float(a[0])
    if not np.isfinite(v) or v <= 0.0:
        raise RuntimeError('key %r scale must be finite and > 0, got %r' % (key, v))
    return v


def import_checkpoint(cfg, ckpt_path, out_dir=None):
    """Import a trained frozen npz into params/*.bin+coe (strict; no silent casts).

    Returns (files, assets) where assets carries everything the manifest must bind:
    model kind, corpus/vocab hashes, per-tensor scales/shapes/ranges, all thresholds,
    head W/b sha256, structure fields.
    """
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zp = pathlib.Path(ckpt_path)
    z = np.load(str(zp))
    keys = set(z.files)
    missing = [k for k in NPZ_REQUIRED if k not in keys]
    if missing:
        raise RuntimeError('checkpoint missing %d required keys: %s'
                           % (len(missing), missing))
    bn_state = str(np.asarray(z['meta_bn_state']).item())
    if bn_state not in BN_STATES:
        raise RuntimeError('meta_bn_state must be one of %s, got %r' % (BN_STATES, bn_state))
    if bn_state == 'requires_fold':
        raise RuntimeError('checkpoint declares BN requires_fold: fold BN before export')
    model_kind = str(np.asarray(z['meta_model_kind']).item())
    if model_kind not in ('real_corpus', 'development_fixture'):
        raise RuntimeError('meta_model_kind must be real_corpus|development_fixture, got %r'
                           % model_kind)
    meta_cfg = json.loads(str(np.asarray(z['meta_cfg']).item()))
    for k in ('E', 'H', 'HD', 'N', 'T', 'D', 'MLP', 'C', 'vocab_size'):
        if int(meta_cfg[k]) != int(cfg['embed'] if k == 'E' else
                                   cfg['heads'] if k == 'H' else
                                   cfg['head_dim'] if k == 'HD' else
                                   cfg['seq_len'] if k == 'N' else
                                   cfg['time_steps'] if k == 'T' else
                                   cfg['depth'] if k == 'D' else
                                   cfg['mlp_hidden'] if k == 'MLP' else
                                   cfg['num_classes'] if k == 'C' else
                                   cfg['vocab_size']):
            raise RuntimeError('meta_cfg[%r]=%r != config %r' % (k, meta_cfg[k], cfg))
    E, MLP, D, C = cfg['embed'], cfg['mlp_hidden'], cfg['depth'], cfg['num_classes']
    V, N = cfg['vocab_size'], cfg['seq_len']

    etab = _get(z, 'etab', (V, E), np.int8, 'int8 embedding table')
    ptab = _get(z, 'ptab', (N, E), np.int8, 'int8 position table')
    thr_e = _scalar_i32(z, 'thr_e', lo=1)
    head_w = _get(z, 'head_w', (C, E), np.float32, 'FP32 head weight')
    head_b = _get(z, 'head_b', (C,), np.float32, 'FP32 head bias')
    if not np.all(np.isfinite(head_w)) or not np.all(np.isfinite(head_b)):
        raise RuntimeError('head params contain non-finite values')
    for nm, a in (('etab', etab), ('ptab', ptab)):
        if a.size and (int(a.min()) < -127 or int(a.max()) > 127):
            raise RuntimeError('%s out of frozen int8 range [-127,127]' % nm)

    blocks = []
    for b in range(D):
        p = 'block%d.' % b
        blk = {}
        for nm in LIN_NAMES:
            o, i = (MLP, E) if nm == 'fc1' else ((E, MLP) if nm == 'fc2' else (E, E))
            w = _get(z, p + nm + '_w', (o, i), np.int8, 'int8 weight')
            bias = _get(z, p + nm + '_b', (o,), np.int32, 'int32 bias')
            scale = _scalar_f32(z, p + nm + '_scale')
            thr = _scalar_i32(z, p + nm + '_thr', lo=1)
            if w.size and (int(w.min()) < -127 or int(w.max()) > 127):
                raise RuntimeError('%s%s_w out of frozen int8 range [-127,127]' % (p, nm))
            blk[nm] = {'w': w, 'b': bias, 'scale': scale, 'thr': thr}
        blocks.append(blk)

    files = {}
    files['embed_w'] = _emit(out, 'embed_w', etab)
    files['pos_w'] = _emit(out, 'pos_w', ptab)
    files['thr_e'] = _emit(out, 'thr_e', np.array([thr_e], np.int32))
    if 'embed_scale' in keys:
        embed_scale = _scalar_f32(z, 'embed_scale')
        embed_scale_src = 'npz'
    else:
        # documented derivation (never silently defaulted): shared etab/ptab symmetric scale
        m = max(int(np.abs(etab.astype(np.int32)).max()), int(np.abs(ptab.astype(np.int32)).max()))
        embed_scale = (float(m) / 127.0) if m > 0 else 1.0
        embed_scale_src = 'derived_maxabs_over_127'
    files['embed_scale'] = _emit(out, 'embed_scale', np.array([embed_scale], np.float32))
    scales = {'embed': embed_scale}
    thresholds = {'thr_e': thr_e, 'blocks': []}
    for b, blk in enumerate(blocks):
        p = 'block%d.' % b
        for nm in LIN_NAMES:
            files[p + nm + '_w'] = _emit(out, p + nm + '_w', blk[nm]['w'])
            files[p + nm + '_b'] = _emit(out, p + nm + '_b', blk[nm]['b'])
            files[p + nm + '_scale'] = _emit(out, p + nm + '_scale',
                                             np.array([blk[nm]['scale']], np.float32))
            files[p + nm + '_thr'] = _emit(out, p + nm + '_thr',
                                           np.array([blk[nm]['thr']], np.int32))
        scales['block%d' % b] = {nm: blk[nm]['scale'] for nm in LIN_NAMES}
        thresholds['blocks'].append({nm: blk[nm]['thr'] for nm in LIN_NAMES})
    files['head_w'] = _emit(out, 'head_w', head_w)
    files['head_b'] = _emit(out, 'head_b', head_b)

    assets = {
        'model_kind': model_kind,
        'bn_state': bn_state,
        'checkpoint': str(zp),
        'checkpoint_sha256': _sha(zp),
        'meta_cfg': meta_cfg,
        'corpus_sha256': str(np.asarray(z['meta_corpus_sha256']).item()),
        'vocab_sha256': str(np.asarray(z['meta_vocab_sha256']).item()),
        'embed_scale': embed_scale,
        'embed_scale_source': embed_scale_src,
        'scales': scales,
        'thresholds': thresholds,
        'head': {'norm': 'ps_div', 'reduction': cfg['head']['reduction'],
                 'w_sha256': _sha(out / 'head_w.bin'),
                 'b_sha256': _sha(out / 'head_b.bin'),
                 'shape': [list(head_w.shape), list(head_b.shape)]},
        'tensor_ranges': _ranges(etab, ptab, blocks),
        'structure': {
            'E': E, 'HEADS': cfg['heads'], 'HEAD_DIM': cfg['head_dim'],
            'HIDDEN': MLP, 'T': cfg['time_steps'], 'TOKENS': N, 'DEPTH': D,
            'TL': 8, 'TC': 8, 'RW': 4, 'PSUM_SLICE': 32,
            'SUM_W': int(cfg['derived']['sum_w']),
            'lif_add_bits': 33, 'accum_bits': 32,
        },
    }
    return files, assets


def _ranges(etab, ptab, blocks):
    def rng(a):
        a = np.asarray(a, dtype=np.int32)
        return {'min': int(a.min()) if a.size else 0,
                'max': int(a.max()) if a.size else 0, 'shape': list(a.shape)}
    r = {'etab': rng(etab), 'ptab': rng(ptab)}
    for b, blk in enumerate(blocks):
        for nm, v in blk.items():
            r['block%d.%s_w' % (b, nm)] = rng(v['w'])
            r['block%d.%s_b' % (b, nm)] = rng(v['b'])
    return r
