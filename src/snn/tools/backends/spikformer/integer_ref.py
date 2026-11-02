"""Independent NumPy integer reference for the Spikformer NLP 48h integer variant.

This module is the GOLDEN TRUTH for RTL replay. It implements the frozen numeric
contract in build/spikformer/nlp_48h_20261029_1915/contract.json with integer
arithmetic only on the datapath (the sole FP32 stage is the PS head, as frozen):

  embed:  delta[n][e] = etab[id[n]][e] + ptab[pos=n][e]   (same delta over T)
          LIF(thr_e) over t -> spike[B,T,N,E] in {0,1}
  block:  Q/K/V = linear(input_spikes) + int32 bias -> LIF(q/k/v)
          per head, per t: score[i][j] = popcount(Q[i] & K[j]) over head_dim bits
                           (bitwise AND; NO /head_dim; max 32)
          AV[i][a] = sum_j score[i][j] * V[j][a]   (unnormalized; max 32*32=1024)
          proj: AV -> linear+int32 bias -> LIF(proj) -> residual add
          MLP : fc1(residual) -> linear+int32 bias -> LIF(fc1)
                fc2(fc1_spike) -> linear+int32 bias -> LIF(fc2) -> residual add
  head:   PL per-channel sum over T*N (uint32; integer, exact)
          PS x/(T*N) in FP32 -> FP32 Linear + bias -> argmax

LIF (frozen): pre/delta are signed integers; (pre+delta) is sign-extended to 33 bit
BEFORE the add (we use int64, exact), mem = floor((pre+delta)/2) == arithmetic >>1,
spike iff valid and mem >= 0 and mem >= thr, hard reset mem=0 on spike, pre=0 at
t=0 of every token/channel, thr >= 1 (int32).

Accumulator width: every linear stage is a signed 32-bit accumulator in the RTL.
Here we accumulate in int64 and assert the frozen int32 range at each stage, so an
overflow would be reported instead of silently wrapping.

Public API (stable; replay imports this module):

  cfg = default_cfg()                        # contract target dims
  cfg = cfg_from_contract(path)              # read model block from contract.json
  params = load_checkpoint(npz_path, cfg=None)
  out = run_sample(params, ids, cfg=None, trace=False)
        ids: [N] or [B,N] int token ids
        out keys: 'embed_spikes' [B,T,N,E]; 'blocks' list of per-block dicts;
                  'pl_sums' [B,E] uint32; 'pooled' [B,E] float32;
                  'logits' [B,C] float32; 'cls' [B] int
  logits, cls = forward(params, ids, cfg=None)
  res = evaluate(params, ids, labels, cfg=None)   # accuracy/confusion/majority
  first_mismatch(ref_trace, got_trace)             # debug localisation helper
  pack_* helpers for the frozen 64/128-bit ABI word formats.

Integer datapath note: all activations/spikes/accumulators in the returned trace are
exact integers (numpy integer dtypes); the FP32 head is computed from the exact
integer PL sums exactly like runtime/spikformer/head.py (ps_div).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# contract constants (build/spikformer/nlp_48h_20261029_1915/contract.json)
# ---------------------------------------------------------------------------
RW_BITS = 4          # residual allocation bits (uint4)
RES_BITS_MIN = 3     # minimum required
SUM_W = 6            # QK sum width; all-ones score 32 fits
AV_BITS = 11         # unnormalized AV max 1024
AV_MAX = 1024
TAU = 2              # LIF tau; mem = floor((pre+delta)/2)
LIF_ADD_BITS = 33    # pre+delta sign-extended to 33 bit before add
THR_MIN = 1
PAD_ID = 0
UNK_ID = 1

DEFAULT_CFG = {
    'E': 128, 'H': 4, 'HD': 32, 'N': 32, 'T': 4, 'D': 2,
    'MLP': 512, 'C': 2, 'vocab_size': 4000,
}

_LINEAR_KEYS = ('q', 'k', 'v', 'proj', 'fc1', 'fc2')


def default_cfg():
    """Contract target configuration (E128/H4/HD32/seq32/MLP512/depth2/T4)."""
    return dict(DEFAULT_CFG)


def cfg_from_contract(path):
    """Read the frozen model dims from contract.json."""
    raw = json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
    m = raw['model']
    cfg = {
        'E': int(m['E']), 'H': int(m['heads']), 'HD': int(m['head_dim']),
        'N': int(m['seq_len']), 'T': int(m['time_steps']), 'D': int(m['depth']),
        'MLP': int(m['mlp_hidden']), 'C': int(m['num_classes']),
        'vocab_size': int(m['vocab_size']),
    }
    _check_cfg(cfg)
    return cfg


def _check_cfg(cfg):
    E, H, HD, N, T, D = cfg['E'], cfg['H'], cfg['HD'], cfg['N'], cfg['T'], cfg['D']
    if E != H * HD:
        raise ValueError('cfg: E != H*HD (%d != %d*%d)' % (E, H, HD))
    if HD > 32:
        raise ValueError('cfg: head_dim must be <= 32 (popcount over 32-bit word)')
    if T < 1 or N < 1 or D < 1:
        raise ValueError('cfg: T/N/D must be positive')
    if cfg['MLP'] < 1 or cfg['C'] < 2:
        raise ValueError('cfg: bad MLP/C')


# ---------------------------------------------------------------------------
# integer primitives
# ---------------------------------------------------------------------------
def arith_shr1(x):
    """floor(x/2) for signed integers == arithmetic right shift by 1."""
    return np.right_shift(np.asarray(x, dtype=np.int64), 1)


def lif_over_time(deltas, thr, valid=None, axis=0):
    """Run the frozen LIF recurrence over the given time axis (default 0).

    deltas: integer array whose `axis` dimension is time (callers pass [B,T,...]
    with axis=1).
    thr:    int32 threshold, must be >= 1.
    valid:  optional bool array with the same shape; spike additionally requires valid.

    Returns (spikes, mem) with spikes uint8 in {0,1} and mem int64 (pre-shift value
    mem = floor((pre+delta)/2)). Hard reset to 0 on a spike; pre=0 at t=0.
    The pre+delta sum is exact int64 here (RTL keeps it in a 33-bit adder) and is
    range-checked against the 33-bit signed domain, never silently wrapped.
    """
    deltas = np.moveaxis(np.asarray(deltas, dtype=np.int64), axis, 0)
    thr = int(thr)
    if thr < THR_MIN:
        raise ValueError('LIF threshold must be >= 1, got %r' % thr)
    if deltas.ndim < 1:
        raise ValueError('deltas must have a time axis')
    if valid is not None:
        valid = np.moveaxis(np.asarray(valid, dtype=bool), axis, 0)
    T = deltas.shape[0]
    pre = np.zeros(deltas.shape[1:], dtype=np.int64)
    spikes = np.zeros(deltas.shape, dtype=np.uint8)
    mems = np.empty(deltas.shape, dtype=np.int64)
    for t in range(T):
        s = pre + deltas[t]                    # exact int64: 33-bit sign-extended add
        _assert_i33(s, 'LIF pre+delta')
        mem = arith_shr1(s)                    # mem = floor(s/2)
        spk = (mem >= 0) & (mem >= thr)
        if valid is not None:
            spk = spk & valid[t]
        spikes[t] = spk.astype(np.uint8)
        mems[t] = mem
        pre = np.where(spk, np.int64(0), mem)
    return np.moveaxis(spikes, 0, axis), np.moveaxis(mems, 0, axis)


def linear_int(x, w, b):
    """pre = x @ w.T + b with signed-int32 accumulators (int64 used + range assert).

    x: [..., IN] integers; w: [OUT, IN] signed int8; b: [OUT] signed int32.
    """
    x = np.asarray(x, dtype=np.int64)
    w = np.asarray(w, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    out = x @ w.T + b
    _assert_i32(out, 'linear accumulator')
    return out


def _assert_i32(arr, what):
    """Frozen contract: every linear stage accumulator is signed 32-bit."""
    if np.size(arr) == 0:
        return
    mx = int(np.max(arr))
    mn = int(np.min(arr))
    if mx > 2147483647 or mn < -2147483648:
        raise OverflowError('%s exceeds signed int32 range: [%d, %d]' % (what, mn, mx))


def _assert_i33(arr, what):
    """Frozen contract: LIF pre+delta is a signed 33-bit adder (no 32-bit wrap)."""
    if np.size(arr) == 0:
        return
    mx = int(np.max(arr))
    mn = int(np.min(arr))
    if mx > 4294967295 or mn < -4294967296:
        raise OverflowError('%s exceeds signed int33 range: [%d, %d]' % (what, mn, mx))


_POP8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)


def popcount32(x):
    """Bitwise popcount of uint32 values (per-T QK popcount)."""
    x = np.asarray(x, dtype=np.uint32)
    pc = (_POP8[x & 0xFF] + _POP8[(x >> 8) & 0xFF]
          + _POP8[(x >> 16) & 0xFF] + _POP8[(x >> 24) & 0xFF])
    return pc.astype(np.uint8)


def _pack_head_bits(spikes, HD):
    """[..., H, HD] 0/1 -> [..., H] uint32 bit words (bit d = channel h*HD+d)."""
    shifts = np.arange(HD, dtype=np.uint64)
    arr = np.asarray(spikes, dtype=np.uint64)
    words = np.zeros(arr.shape[:-1], dtype=np.uint64)
    for d in range(HD):
        words |= (arr[..., d] & 1) << shifts[d]
    return words.astype(np.uint32)


def qk_scores(q_spikes, k_spikes, H, HD):
    """Per-head per-t QK bit-AND popcount, NO division by head_dim.

    q_spikes/k_spikes: [B,T,N,E] 0/1 with channel = h*HD+d.
    Returns [B,T,N,N,H] uint8: score[b,t,i,j,h] = popcount(q[b,t,i,:] & k[b,t,j,:])_h.
    """
    q = np.asarray(q_spikes)
    k = np.asarray(k_spikes)
    B, T, N, E = q.shape
    if E != H * HD:
        raise ValueError('qk_scores: E != H*HD')
    qw = _pack_head_bits(q.reshape(B, T, N, H, HD), HD)   # [B,T,N,H]
    kw = _pack_head_bits(k.reshape(B, T, N, H, HD), HD)
    anded = qw[:, :, :, None, :] & kw[:, :, None, :, :]   # [B,T,N,N,H]
    return popcount32(anded)


def av_unnormalized(scores, v_spikes, H, HD):
    """AV[b,t,i,a] = sum_j score[b,t,i,j,h] * V[b,t,j,a] (unnormalized).

    scores: [B,T,N,N,H] uint8 (from qk_scores); v_spikes: [B,T,N,E] 0/1.
    Returns int64 [B,T,N,E] with channel = h*HD+d. Max 32*32=1024.
    """
    scores = np.asarray(scores, dtype=np.int64)
    v = np.asarray(v_spikes, dtype=np.int64)
    B, T, N, _ = v.shape
    vh = v.reshape(B, T, N, H, HD)
    avh = np.einsum('btijh,btjhd->btihd', scores, vh)
    av = avh.reshape(B, T, N, H * HD)
    _assert_i32(av, 'AV')
    mx = int(np.max(av)) if av.size else 0
    if mx > AV_MAX:
        raise OverflowError('AV exceeds frozen max %d: %d' % (AV_MAX, mx))
    return av


# ---------------------------------------------------------------------------
# parameter container / loading
# ---------------------------------------------------------------------------
def _as_i8(a, name):
    a = np.asarray(a)
    if a.dtype != np.int8:
        raise TypeError('%s must be int8, got %s' % (name, a.dtype))
    if a.size and (int(a.min()) < -128 or int(a.max()) > 127):
        raise ValueError('%s out of int8 range' % name)
    return a


def _as_i32(a, name):
    a = np.asarray(a)
    if a.dtype != np.int32:
        raise TypeError('%s must be int32, got %s' % (name, a.dtype))
    return a


def load_checkpoint(path, cfg=None):
    """Load a trained .npz into the integer parameter container.

    Required keys follow contract export_abi (block{b}.{q,k,v,proj,fc1,fc2}_w int8,
    _b int32, _scale float32, block{b}.*_thr int32, thr_e int32, head_w/head_b
    float32, meta_*). Dtypes and shapes are validated strictly (no silent casts).
    """
    z = np.load(str(path))
    keys = set(z.files)
    if cfg is None:
        cfg = _cfg_from_meta(z)
    _check_cfg(cfg)
    E, H, HD, N, T, D = cfg['E'], cfg['H'], cfg['HD'], cfg['N'], cfg['T'], cfg['D']
    MLP, C, V = cfg['MLP'], cfg['C'], cfg['vocab_size']

    def need(name, shape, kind):
        if name not in keys:
            raise RuntimeError('checkpoint missing required key %r' % name)
        arr = z[name]
        if tuple(arr.shape) != tuple(shape):
            raise RuntimeError('key %r shape %s != %s' % (name, tuple(arr.shape), tuple(shape)))
        return _as_i8(arr, name) if kind == 'i8' else _as_i32(arr, name)

    params = {'cfg': cfg}
    params['etab'] = need('etab', (V, E), 'i8')
    params['ptab'] = need('ptab', (N, E), 'i8')
    params['thr_e'] = int(need('thr_e', (1,), 'i32')[0])
    if 'embed_scale' in keys:
        es = np.asarray(z['embed_scale'])
        if es.dtype != np.float32 or tuple(es.shape) != (1,):
            raise RuntimeError('embed_scale must be float32 shape (1,)')
        params['embed_scale'] = float(es[0])
    params['head_w'] = np.asarray(z['head_w'], dtype=np.float32) if 'head_w' in keys else None
    params['head_b'] = np.asarray(z['head_b'], dtype=np.float32) if 'head_b' in keys else None
    if params['head_w'] is None or params['head_b'] is None:
        raise RuntimeError('checkpoint missing head_w/head_b (FP32 head)')
    if tuple(params['head_w'].shape) != (C, E):
        raise RuntimeError('head_w shape %s != (%d,%d)' % (params['head_w'].shape, C, E))
    if tuple(params['head_b'].shape) != (C,):
        raise RuntimeError('head_b shape %s != (%d,)' % (params['head_b'].shape, C))

    blocks = []
    for b in range(D):
        p = 'block%d.' % b
        blk = {}
        for nm in _LINEAR_KEYS:
            out_dim, in_dim = (MLP, E) if nm == 'fc1' else ((E, MLP) if nm == 'fc2' else (E, E))
            blk[nm + '_w'] = need(p + nm + '_w', (out_dim, in_dim), 'i8')
            blk[nm + '_b'] = need(p + nm + '_b', (out_dim,), 'i32')
            blk[nm + '_thr'] = int(need(p + nm + '_thr', (1,), 'i32')[0])
            if p + nm + '_scale' in keys:
                sc = np.asarray(z[p + nm + '_scale'])
                if sc.dtype != np.float32 or tuple(sc.shape) != (1,):
                    raise RuntimeError('%s_scale must be float32 shape (1,), got %s %s'
                                       % (p + nm, sc.dtype, sc.shape))
                blk[nm + '_scale'] = float(sc[0])
        blocks.append(blk)
    params['blocks'] = blocks
    if 'meta_model_kind' in keys:
        params['model_kind'] = str(z['meta_model_kind'])
    return params


def _cfg_from_meta(z):
    """Best-effort cfg recovery from checkpoint meta keys (shape-derived)."""
    keys = set(z.files)
    if 'meta_cfg' in keys:
        return json.loads(str(z['meta_cfg']))
    raise RuntimeError('cfg not given and checkpoint has no meta_cfg; '
                       'pass cfg=default_cfg() explicitly')


# ---------------------------------------------------------------------------
# forward chain
# ---------------------------------------------------------------------------
def embed_forward(params, ids, cfg, trace=False):
    cfg = cfg or params['cfg']
    ids = np.asarray(ids, dtype=np.int64)
    squeeze = ids.ndim == 1
    if squeeze:
        ids = ids[None, :]
    B, N = ids.shape
    T, E = cfg['T'], cfg['E']
    etab = np.asarray(params['etab'], dtype=np.int64)
    ptab = np.asarray(params['ptab'], dtype=np.int64)
    if B > 0 and int(ids.max()) >= etab.shape[0]:
        raise ValueError('token id out of vocab range: %d >= %d' % (int(ids.max()), etab.shape[0]))
    d0 = etab[ids] + ptab[None, :, :]                      # [B,N,E] exact integer
    deltas = np.repeat(d0[:, None, :, :], T, axis=1)       # [B,T,N,E] same over t
    spikes, mems = lif_over_time(deltas, int(params['thr_e']), axis=1)
    out = {'spikes': spikes}
    if trace:
        out['delta'] = d0
        out['mem'] = mems
    return out


def block_forward(params_block, x_spikes, cfg, trace=False):
    """One Spikformer NLP integer block. x_spikes: [B,T,N,E] uint 0..(1+2b)."""
    E, H, HD, N, T = cfg['E'], cfg['H'], cfg['HD'], cfg['N'], cfg['T']
    x = np.asarray(x_spikes, dtype=np.int64)
    B = x.shape[0]
    if x.shape != (B, T, N, E):
        raise ValueError('block input shape %s != (%d,%d,%d,%d)' % (x.shape, B, T, N, E))
    tr = {'in': x} if trace else None

    # --- Q/K/V linear + LIF ---
    pre_q = linear_int(x, params_block['q_w'], params_block['q_b'])
    pre_k = linear_int(x, params_block['k_w'], params_block['k_b'])
    pre_v = linear_int(x, params_block['v_w'], params_block['v_b'])
    q_s, q_m = lif_over_time(pre_q, params_block['q_thr'], axis=1)
    k_s, k_m = lif_over_time(pre_k, params_block['k_thr'], axis=1)
    v_s, v_m = lif_over_time(pre_v, params_block['v_thr'], axis=1)
    if trace:
        tr.update(q_delta=pre_q, q_spikes=q_s, q_mem=q_m,
                  k_delta=pre_k, k_spikes=k_s, k_mem=k_m,
                  v_delta=pre_v, v_spikes=v_s, v_mem=v_m)

    # --- attention: per-head per-t QK popcount (no /head_dim), unnormalized AV ---
    scores = qk_scores(q_s, k_s, H, HD)          # [B,T,N,N,H]
    av = av_unnormalized(scores, v_s, H, HD)     # [B,T,N,E] integer, max 1024
    if trace:
        tr['scores'] = scores
        tr['av'] = av

    # --- proj -> LIF -> residual ---
    pre_p = linear_int(av, params_block['proj_w'], params_block['proj_b'])
    p_s, p_m = lif_over_time(pre_p, params_block['proj_thr'], axis=1)
    res1 = x + p_s.astype(np.int64)
    if trace:
        tr.update(proj_delta=pre_p, proj_spikes=p_s, proj_mem=p_m, res1=res1)

    # --- MLP: fc1 -> LIF -> fc2 -> LIF -> residual ---
    pre_1 = linear_int(res1, params_block['fc1_w'], params_block['fc1_b'])
    f1_s, f1_m = lif_over_time(pre_1, params_block['fc1_thr'], axis=1)
    pre_2 = linear_int(f1_s, params_block['fc2_w'], params_block['fc2_b'])
    f2_s, f2_m = lif_over_time(pre_2, params_block['fc2_thr'], axis=1)
    out = res1 + f2_s.astype(np.int64)
    if trace:
        tr.update(fc1_delta=pre_1, fc1_spikes=f1_s, fc1_mem=f1_m,
                  fc2_delta=pre_2, fc2_spikes=f2_s, fc2_mem=f2_m, out=out)
    return out, tr


def run_sample(params, ids, cfg=None, trace=False):
    """Full integer forward for token id array(s). Returns dict with exact traces."""
    cfg = cfg or params['cfg']
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim == 1:
        ids = ids[None, :]
    B, N = ids.shape
    if N != cfg['N']:
        raise ValueError('ids seq len %d != cfg N %d (fixed length, mask=none)' % (N, cfg['N']))
    out = {'ids': ids}
    emb = embed_forward(params, ids, cfg, trace=trace)
    out['embed_spikes'] = emb['spikes']
    if trace:
        out['embed'] = emb
    x = emb['spikes'].astype(np.int64)
    blocks = []
    for b, blk_params in enumerate(params['blocks']):
        x, tr = block_forward(blk_params, x, cfg, trace=trace)
        if trace:
            blocks.append(tr)
    out['blocks'] = blocks
    out['final_residual'] = x
    # PL: per-channel integer sum over T*N -> uint32 (exact integer)
    pl = x.sum(axis=(1, 2))                     # [B,E] int64
    if np.any(pl < 0) or np.any(pl > 0xFFFFFFFF):
        raise OverflowError('PL sums outside uint32 range')
    out['pl_sums'] = pl.astype(np.uint32)
    # PS: x/(T*N) in FP32, then FP32 Linear + bias (same ops as runtime/spikformer/head.py)
    pooled = out['pl_sums'].astype(np.float32) / np.float32(cfg['T'] * cfg['N'])
    head_w = np.asarray(params['head_w'], dtype=np.float32)
    head_b = np.asarray(params['head_b'], dtype=np.float32)
    logits = pooled @ head_w.T + head_b          # [B,C] float32
    out['pooled'] = pooled.astype(np.float32)
    out['logits'] = logits.astype(np.float32)
    out['cls'] = np.argmax(logits, axis=1).astype(np.int64)
    return out


def forward(params, ids, cfg=None):
    r = run_sample(params, ids, cfg=cfg, trace=False)
    return r['logits'], r['cls']


def evaluate(params, ids, labels, cfg=None, batch_size=64):
    """Integer-domain accuracy / confusion matrix / majority baseline."""
    cfg = cfg or params['cfg']
    ids = np.asarray(ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    C = cfg['C']
    preds = []
    for i in range(0, len(ids), batch_size):
        _, cls = forward(params, ids[i:i + batch_size], cfg=cfg)
        preds.append(cls)
    pred = np.concatenate(preds) if preds else np.zeros(0, np.int64)
    conf = np.zeros((C, C), dtype=np.int64)
    for y, p in zip(labels, pred):
        conf[int(y), int(p)] += 1
    counts = np.bincount(labels, minlength=C)
    majority = int(np.argmax(counts))
    maj_acc = float(counts[majority]) / float(len(labels)) if len(labels) else 0.0
    acc = float(np.mean(pred == labels)) if len(labels) else 0.0
    return {'acc': acc, 'n': int(len(labels)), 'confusion': conf.tolist(),
            'class_counts': counts.tolist(), 'majority_class': majority,
            'majority_acc': maj_acc, 'pred': pred}


# ---------------------------------------------------------------------------
# debug: first-mismatch localisation between two traces
# ---------------------------------------------------------------------------
def trace_keys(trace):
    """Ordered list of comparable layer keys for a run_sample(trace=True) dict."""
    keys = ['embed_spikes', 'embed.delta', 'embed.mem']
    nb = len(trace.get('blocks') or [])
    for b in range(nb):
        keys += ['block%d.in' % b, 'block%d.q_delta' % b, 'block%d.q_spikes' % b,
                 'block%d.q_mem' % b, 'block%d.k_delta' % b, 'block%d.k_spikes' % b,
                 'block%d.v_delta' % b, 'block%d.v_spikes' % b,
                 'block%d.scores' % b, 'block%d.av' % b,
                 'block%d.proj_delta' % b, 'block%d.proj_spikes' % b,
                 'block%d.proj_mem' % b, 'block%d.res1' % b,
                 'block%d.fc1_delta' % b, 'block%d.fc1_spikes' % b,
                 'block%d.fc1_mem' % b, 'block%d.fc2_delta' % b,
                 'block%d.fc2_spikes' % b, 'block%d.fc2_mem' % b,
                 'block%d.out' % b]
    keys += ['final_residual', 'pl_sums', 'logits']
    return keys


def _get_key(trace, key):
    if key == 'embed_spikes':
        return trace.get('embed_spikes')
    if key.startswith('embed.'):
        return (trace.get('embed') or {}).get(key.split('.', 1)[1])
    if key == 'final_residual' or key == 'pl_sums' or key == 'logits':
        return trace.get(key)
    # blockX.field
    blk, field = key.split('.', 1)
    b = int(blk[5:])
    blocks = trace.get('blocks') or []
    if b >= len(blocks):
        return None
    return blocks[b].get(field)


def first_mismatch(ref_trace, got_trace, keys=None):
    """Return the first differing layer/index between two traces (or None).

    Both arguments are dicts shaped like run_sample(trace=True). 'got_trace' may be
    an RTL-collected dict with the same keys. Returns
    {'layer', 'index', 'ref', 'got', 'shape'} for first-mismatch debugging.
    """
    for key in (keys or trace_keys(ref_trace)):
        a = _get_key(ref_trace, key)
        if a is None:
            continue
        b = _get_key(got_trace, key)
        if b is None:
            return {'layer': key, 'index': None, 'ref': 'present', 'got': 'missing',
                    'shape': np.asarray(a).shape}
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            return {'layer': key, 'index': None, 'ref': a.shape, 'got': b.shape,
                    'shape': a.shape}
        if a.size == 0:
            continue
        neq = np.argwhere(a.reshape(-1) != b.reshape(-1))
        if len(neq):
            i = int(neq[0][0])
            idx = tuple(int(v) for v in np.unravel_index(i, a.shape))
            return {'layer': key, 'index': idx, 'ref': int(a.reshape(-1)[i]),
                    'got': int(b.reshape(-1)[i]), 'shape': a.shape}
    return None


# ---------------------------------------------------------------------------
# frozen ABI word packing helpers (for replay TB vector generation)
# ---------------------------------------------------------------------------
def pack_int8_beats(arr):
    """row-major signed int8 -> list of 64-bit words, byte i in bits [8*i +: 8]."""
    flat = np.asarray(arr, dtype=np.int64).reshape(-1)
    if flat.size % 8:
        raise ValueError('int8 stream length %d not a multiple of 8' % flat.size)
    out = []
    for i in range(0, flat.size, 8):
        w = 0
        for j in range(8):
            w |= (int(flat[i + j]) & 0xFF) << (8 * j)
        out.append(w)
    return out


def pack_int32_beats(b):
    """signed int32 -> 64-bit {b[2m+1][31:0], b[2m][31:0]} (contract bias beats)."""
    flat = np.asarray(b, dtype=np.int64).reshape(-1)
    if flat.size % 2:
        raise ValueError('bias stream length %d not a multiple of 2' % flat.size)
    out = []
    for m in range(flat.size // 2):
        lo = int(flat[2 * m]) & 0xFFFFFFFF
        hi = int(flat[2 * m + 1]) & 0xFFFFFFFF
        out.append(lo | (hi << 32))
    return out


def pack_block_io_beats(x, rw=RW_BITS, tl=8):
    """Pack a per-token/group beat stream: [B,T,N,E] integer -> list of 128-bit ints.

    Layout frozen in contract.hardware_abi: TL lanes * T timeslices * RW bits,
    lane-major; bit offset of lane a, timeslice tt is (a*T + tt)*RW.
    Beat order = token-major, then channel group of TL lanes.
    """
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim == 3:
        arr = arr[None, ...]
    B, T, N, E = arr.shape
    if E % tl:
        raise ValueError('E %d not divisible by TL %d' % (E, tl))
    beats = []
    for b in range(B):
        for n in range(N):
            for g in range(E // tl):
                w = 0
                for l in range(tl):
                    a = g * tl + l
                    for tt in range(T):
                        v = int(arr[b, tt, n, a]) & ((1 << rw) - 1)
                        w |= v << ((l * T + tt) * rw)
                beats.append(w)
    return beats


def pack_uint32_words(vals):
    """uint32 array -> list of ints (o_sums word stream)."""
    return [int(v) & 0xFFFFFFFF for v in np.asarray(vals).reshape(-1)]


__all__ = [
    'RW_BITS', 'RES_BITS_MIN', 'SUM_W', 'AV_BITS', 'AV_MAX', 'TAU', 'THR_MIN',
    'PAD_ID', 'UNK_ID', 'default_cfg', 'cfg_from_contract', 'arith_shr1',
    'lif_over_time', 'linear_int', 'popcount32', 'qk_scores', 'av_unnormalized',
    'load_checkpoint', 'embed_forward', 'block_forward', 'run_sample', 'forward',
    'evaluate', 'trace_keys', 'first_mismatch', 'pack_int8_beats',
    'pack_int32_beats', 'pack_block_io_beats', 'pack_uint32_words',
]