"""Common golden/packing for the Spikformer NLP 48h replay cases.

Contract (frozen, authoritative):
  build/spikformer/nlp_48h_20261029_1915/contract.json
Golden truth (owned by trainer line):
  tools/backends/spikformer/integer_ref.py  -- imported and cross-checked when
  available; the NumPy implementation below is the contract-faithful generator.

Numeric contract implemented here (all integers, int64 numpy internals):
  embed : delta[n,c,t] = etab[tok[n],c] + ptab[n,c]  (constant over t)
          pre=0; per t: mem=(pre+delta)>>1 (arith floor); spike=(mem>=0 and mem>=thr_e);
          hard reset to 0 on spike; state resets at t=0 of every token/channel.
  block : r = block input residual (unsigned RW bit, target 0..5 after depth2)
          Q/K/V: psum[n,o,t] = bias[o] + sum_i W[o,i]*r[n,i,t]; LIF over t per token/channel
          score[h,n,j,t] = sum_{m<HD} Q[n,h*HD+m,t] & K[j,h*HD+m,t]   (bit AND, no /HD)
          AV[n,c,t] = sum_j score[h,n,j,t]*V[j,c,t], c in head h   (unnormalized)
          proj: psum over AV -> LIF(thr_proj);  ares = r + proj_spike
          fc1 : psum over ares -> LIF(thr_fc1) -> hid
          fc2 : psum over hid -> LIF(thr_fc2);  out = ares + fc2_spike
  head  : PL sums[c] = sum_{n,t} out_last[n,c,t] (uint32, no divide in HW)
          PS: x = sums/(T*N) float32 -> logits = head_w @ x + head_b -> argmax

Beat ABI (contract hardware_abi):
  weight beat 64b = 8 consecutive signed int8 row-major [out][in]
  bias beat 64b   = {bias[2m+1][31:0], bias[2m][31:0]} signed int32
  block I/O beat  = TL lanes x T timeslices x RW bits, lane-major
                    bits[(a*T+tt+1)*RW-1 -: RW]
  embed beat      = LANES lanes x T spikes, lane-major bits[a*T+tt]
  descriptor beats: {thr_q,thr_k}, {thr_v,thr_proj}, {thr_fc1,thr_fc2},
                    {32'h0, thr_e}  (thr_e meaningful only block 0)
"""
from __future__ import annotations

import datetime
import json
import os
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
RUN_DIR = os.path.join(REPO, 'build', 'spikformer', 'nlp_48h_20261029_1915')
REPLAY_DIR = os.path.join(RUN_DIR, 'replay')
REPORTS_DIR = os.path.join(RUN_DIR, 'reports')
CONTRACT_REL = 'build/spikformer/nlp_48h_20261029_1915/contract.json'

# Frozen case configs (must stay in sync with the TB parameters):
#   tb_nlp_small.v     : E32/H1/T2/seq4/depth2/V64
#   tb_nlp_multihead.v : block-level E64/H2/T2/seq4/depth1
#   tb_nlp_e2e.v       : target E128/H4/T4/seq32/depth2/V4000
CASE_CFG = {
    'small': dict(name='small', E=32, HEADS=1, HEAD_DIM=32, HIDDEN=64, T=2,
                  TOKENS=4, DEPTH=2, TL=8, TC=8, RW=4, PSUM_SLICE=32, SUM_W=6,
                  LANES=8, VTAB=64, OW=32, NUM_CLASSES=2),
    'multihead': dict(name='multihead', E=64, HEADS=2, HEAD_DIM=32, HIDDEN=64, T=2,
                      TOKENS=4, DEPTH=1, TL=8, TC=8, RW=4, PSUM_SLICE=32, SUM_W=6,
                      LANES=8, VTAB=64, OW=32, NUM_CLASSES=2),
    'e2e': dict(name='e2e', E=128, HEADS=4, HEAD_DIM=32, HIDDEN=512, T=4,
                TOKENS=32, DEPTH=2, TL=8, TC=8, RW=4, PSUM_SLICE=32, SUM_W=6,
                LANES=8, VTAB=4000, OW=32, NUM_CLASSES=2),
}

# Seed bases: fixture params are regenerated deterministically; the accepted
# seed is stored in golden.json.
SEED_BASE = {'small': 3001, 'multihead': 4001, 'e2e': 5001}
MAX_SEED_TRIES = 60


# --------------------------------------------------------------------------
# numeric kernel
# --------------------------------------------------------------------------
def arith_shr1(x):
    """Arithmetic floor shift right by one (contract mem=(pre+delta)>>1)."""
    return np.asarray(x, dtype=np.int64) >> 1


def lif_chain(deltas, thr):
    """LIF over the last (time) axis. deltas (...,T) int; returns spikes (...,T)."""
    d = np.asarray(deltas, dtype=np.int64)
    assert d.ndim >= 1
    pre = np.zeros(d.shape[:-1], dtype=np.int64)
    out = np.zeros(d.shape, dtype=np.int64)
    thr = int(thr)
    for t in range(d.shape[-1]):
        s = pre + d[..., t]
        mem = s >> 1                      # arithmetic shift == floor division
        sp = ((mem >= 0) & (mem >= thr)).astype(np.int64)
        pre = np.where(sp != 0, 0, mem)   # hard reset to 0 on spike
        out[..., t] = sp
    return out


def linear_psum(W, b, inp):
    """psum[n,o,t] = b[o] + sum_i W[o,i]*inp[n,i,t]. inp (N,IN,T) int."""
    W = np.asarray(W, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    x = np.asarray(inp, dtype=np.int64)
    return np.einsum('nit,oi->not', x, W) + b[None, :, None]


def linear_lif(W, b, inp, thr):
    """Linear layer + LIF over t. Returns (spikes (N,OUT,T), psum (N,OUT,T))."""
    ps = linear_psum(W, b, inp)
    return lif_chain(ps, thr), ps


def attention_scores_av(cfg, q, k, v):
    """Per-head QK AND-popcount scores (no /HD) and unnormalized AV."""
    H, HD, N, T = cfg['HEADS'], cfg['HEAD_DIM'], cfg['TOKENS'], cfg['T']
    E = cfg['E']
    assert E == H * HD
    scr = np.zeros((H, N, N, T), dtype=np.int64)
    av = np.zeros((N, E, T), dtype=np.int64)
    for h in range(H):
        sl = slice(h * HD, (h + 1) * HD)
        qh = q[:, sl, :]
        kh = k[:, sl, :]
        vh = v[:, sl, :]
        for t in range(T):
            a = qh[:, :, t]
            bb = kh[:, :, t]
            # score[n,j] = popcount over HD of (Q[n,:,t] & K[j,:,t])
            scr[h, :, :, t] = np.einsum('nm,jm->nj', a, bb)
            av[:, sl, t] = np.einsum('nj,jc->nc', scr[h, :, :, t], vh[:, :, t])
    return scr, av


def forward_block(cfg, pb, r):
    """One transformer block forward. r (N,E,T) unsigned residual -> dict of traces."""
    q, pq = linear_lif(pb['q_w'], pb['q_b'], r, pb['thr_q'])
    k, pk = linear_lif(pb['k_w'], pb['k_b'], r, pb['thr_k'])
    v, pv = linear_lif(pb['v_w'], pb['v_b'], r, pb['thr_v'])
    scr, av = attention_scores_av(cfg, q, k, v)
    pj, pproj = linear_lif(pb['proj_w'], pb['proj_b'], av, pb['thr_proj'])
    ares = r + pj
    hid, pfc1 = linear_lif(pb['fc1_w'], pb['fc1_b'], ares, pb['thr_fc1'])
    f2, pfc2 = linear_lif(pb['fc2_w'], pb['fc2_b'], hid, pb['thr_fc2'])
    out = ares + f2
    return dict(inp=r, q=q, k=k, v=v, scr=scr, av=av, proj=pj, ares=ares,
                hid=hid, fc2=f2, out=out,
                psum_q=pq, psum_k=pk, psum_v=pv, psum_proj=pproj,
                psum_fc1=pfc1, psum_fc2=pfc2,
                thr=dict(q=pb['thr_q'], k=pb['thr_k'], v=pb['thr_v'],
                         proj=pb['thr_proj'], fc1=pb['thr_fc1'], fc2=pb['thr_fc2']))


def forward_blocks(cfg, params, r0):
    """Stack of DEPTH blocks. Returns dict(blocks=[...], out, sums, ps_x, logits, pred)."""
    r = np.asarray(r0, dtype=np.int64)
    blocks = []
    for b in range(cfg['DEPTH']):
        bd = forward_block(cfg, params['blocks'][b], r)
        blocks.append(bd)
        r = bd['out']
    sums = r.sum(axis=(0, 2)).astype(np.int64)          # PL: sum over n,t
    head = head_ps(cfg, params, sums)
    return dict(blocks=blocks, out=r, sums=sums, **head)


def head_ps(cfg, params, sums):
    """PS side: x = sums/(T*N) FP32 -> logits = head_w @ x + head_b -> argmax."""
    tn = np.float32(cfg['T'] * cfg['TOKENS'])
    x = (np.asarray(sums, dtype=np.int64).astype(np.float32) / tn).astype(np.float32)
    W = np.asarray(params['head_w'], dtype=np.float32)
    b = np.asarray(params['head_b'], dtype=np.float32)
    logits = (W @ x + b).astype(np.float32)
    pred = int(np.argmax(logits))
    return dict(ps_x=x, logits=logits, pred=pred)


def embed_spikes(cfg, params, tokens):
    """etab[id]+ptab[pos] -> LIF(thr_e) over T. tokens (N,) int."""
    ep = params['embed']
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    tok = np.asarray(tokens, dtype=np.int64)
    assert tok.shape == (N,)
    assert tok.min() >= 0 and tok.max() < cfg['VTAB']
    etab = np.asarray(ep['etab'], dtype=np.int64)
    ptab = np.asarray(ep['ptab'], dtype=np.int64)
    delta2 = etab[tok, :] + ptab                       # (N,E)
    delta = np.repeat(delta2[:, :, None], T, axis=2)    # (N,E,T), constant over t
    return lif_chain(delta, ep['thr_e'])


# --------------------------------------------------------------------------
# fixture parameter generation (development fixture, deterministic)
# --------------------------------------------------------------------------
def pick_thr(psums, rate=0.35):
    """Pick an integer LIF threshold so that roughly (1-rate) of mem steps are below it."""
    mem = (np.asarray(psums, dtype=np.int64) >> 1).ravel()
    if mem.size == 0:
        return 1
    q = np.percentile(mem, 100.0 * (1.0 - rate))
    return max(1, int(np.ceil(q)))


def _randint8(rng, shape):
    return rng.randint(-127, 128, size=shape).astype(np.int8)


def _rand_i32(rng, shape, lim=2048):
    return rng.randint(-lim, lim + 1, size=shape).astype(np.int32)


def make_embed_params(cfg, seed, tokens_list):
    rng = np.random.RandomState(seed)
    V, E, N, T = cfg['VTAB'], cfg['E'], cfg['TOKENS'], cfg['T']
    etab = rng.randint(-127, 128, size=(V, E)).astype(np.int8)
    ptab = rng.randint(-64, 65, size=(N, E)).astype(np.int8)
    # calibrate thr_e on sample 0's deltas
    tok0 = np.asarray(tokens_list[0], dtype=np.int64)
    delta = etab[tok0].astype(np.int64) + ptab.astype(np.int64)   # (N,E)
    thr_e = pick_thr(np.repeat(delta[:, :, None], T, axis=2), 0.35)
    return dict(etab=etab, ptab=ptab, thr_e=thr_e)


def make_block_params(cfg, seed, r0_list, diagonal_heads=False):
    """Generate DEPTH blocks with LIF thresholds calibrated so every stage spikes."""
    E, HIDDEN, DEPTH, HD = cfg['E'], cfg['HIDDEN'], cfg['DEPTH'], cfg['HEAD_DIM']
    H = cfg['HEADS']
    blocks = []
    prev = [np.asarray(r0, dtype=np.int64) for r0 in r0_list]
    for b in range(DEPTH):
        rng = np.random.RandomState(seed + 7919 * b + 13)
        pb = {}
        mats = {}
        for name, shape in (('q_w', (E, E)), ('k_w', (E, E)), ('v_w', (E, E)),
                            ('proj_w', (E, E)), ('fc1_w', (HIDDEN, E)),
                            ('fc2_w', (E, HIDDEN))):
            mats[name] = _randint8(rng, shape)
        if diagonal_heads:
            # Q/K/V: head-h output rows read only head-h input lanes so the two
            # heads carry clearly different spike patterns (cross-head leakage
            # then changes the result measurably).
            for name in ('q_w', 'k_w', 'v_w'):
                W = mats[name]
                Wt = W.copy()
                for h in range(H):
                    sl = slice(h * HD, (h + 1) * HD)
                    for h2 in range(H):
                        if h2 == h:
                            continue
                        Wt[sl, slice(h2 * HD, (h2 + 1) * HD)] = 0
                mats[name] = Wt
        for name, W in mats.items():
            pb[name] = W
        for name, shape in (('q_b', (E,)), ('k_b', (E,)), ('v_b', (E,)),
                            ('proj_b', (E,)), ('fc1_b', (HIDDEN,)), ('fc2_b', (E,))):
            pb[name] = _rand_i32(rng, shape)
        # ---- calibrate thresholds on sample 0 of the current block input ----
        r = prev[0]
        pq = linear_psum(pb['q_w'], pb['q_b'], r)
        pk = linear_psum(pb['k_w'], pb['k_b'], r)
        pv = linear_psum(pb['v_w'], pb['v_b'], r)
        pb['thr_q'] = pick_thr(pq, 0.35)
        pb['thr_k'] = pick_thr(pk, 0.35)
        pb['thr_v'] = pick_thr(pv, 0.35)
        q = lif_chain(pq, pb['thr_q'])
        k = lif_chain(pk, pb['thr_k'])
        v = lif_chain(pv, pb['thr_v'])
        _scr, av = attention_scores_av(cfg, q, k, v)
        pproj = linear_psum(pb['proj_w'], pb['proj_b'], av)
        pb['thr_proj'] = pick_thr(pproj, 0.35)
        pj = lif_chain(pproj, pb['thr_proj'])
        ares = r + pj
        pfc1 = linear_psum(pb['fc1_w'], pb['fc1_b'], ares)
        pb['thr_fc1'] = pick_thr(pfc1, 0.35)
        hid = lif_chain(pfc1, pb['thr_fc1'])
        pfc2 = linear_psum(pb['fc2_w'], pb['fc2_b'], hid)
        pb['thr_fc2'] = pick_thr(pfc2, 0.35)
        blocks.append(pb)
        # advance this block input for all samples (uses calibrated thresholds)
        new_prev = []
        for r_in in prev:
            new_prev.append(forward_block(cfg, pb, r_in)['out'])
        prev = new_prev
    # make sure the two blocks do not share the same threshold vector
    if DEPTH >= 2:
        t0 = [blocks[0][k] for k in ('thr_q', 'thr_k', 'thr_v', 'thr_proj', 'thr_fc1', 'thr_fc2')]
        t1 = [blocks[1][k] for k in ('thr_q', 'thr_k', 'thr_v', 'thr_proj', 'thr_fc1', 'thr_fc2')]
        if t0 == t1:
            blocks[1]['thr_proj'] = blocks[1]['thr_proj'] + 1
    return blocks


def make_head_params(cfg, seed):
    rng = np.random.RandomState(seed + 104729)
    E, NC = cfg['E'], cfg['NUM_CLASSES']
    return dict(head_w=rng.uniform(-0.5, 0.5, size=(NC, E)).astype(np.float32),
                head_b=rng.uniform(-0.5, 0.5, size=(NC,)).astype(np.float32))


# --------------------------------------------------------------------------
# tokens / crafted block inputs
# --------------------------------------------------------------------------
def make_tokens(cfg, kind):
    N, V = cfg['TOKENS'], cfg['VTAB']
    if kind == 'small':
        # 4-token fixed length: PAD=0; clearly different sentences.
        s0 = [7, 21, 0, 0]
        s1 = [33, 5, 12, 0]
        assert max(s0 + s1) < V
        return [np.asarray(s0, dtype=np.int64), np.asarray(s1, dtype=np.int64)]
    if kind == 'e2e':
        rng = np.random.RandomState(20261029)
        s0 = [0] * N
        s1 = [0] * N
        # 12 real tokens (padded to 32) / 25 real tokens (padded to 32)
        for i in range(12):
            s0[i] = int(rng.randint(2, V))
        for i in range(25):
            s1[i] = int(rng.randint(2, V))
        return [np.asarray(s0, dtype=np.int64), np.asarray(s1, dtype=np.int64)]
    raise ValueError('no tokens for kind %r' % kind)


def make_multihead_inputs(cfg):
    """Two block-input spike patterns with complementary head activity.

    run A: head0 lanes spike when (n+t) even, head1 lanes when (n+t) odd.
    run B: the two head halves swapped -> any head swap/leak in the datapath
           changes the output (both runs compared beat-exact).
    """
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    HD = cfg['HEAD_DIM']
    H = cfg['HEADS']
    assert H == 2 and E == H * HD
    a = np.zeros((N, E, T), dtype=np.int64)
    for n in range(N):
        for t in range(T):
            for h in range(H):
                v = 1 if ((n + t + h) % 2 == 0) else 0
                a[n, h * HD:(h + 1) * HD, t] = v
    b = np.zeros_like(a)
    for h in range(H):
        b[:, h * HD:(h + 1) * HD, :] = a[:, ((h + 1) % H) * HD:((h + 1) % H + 1) * HD, :]
    return [a, b]


# --------------------------------------------------------------------------
# activity / correctness checks on a generated fixture
# --------------------------------------------------------------------------
def _spk_stat(x):
    x = np.asarray(x)
    tot = int(x.size)
    on = int((x != 0).sum())
    return on, tot


def activity_check(cfg, params, goldens):
    """Require every LIF placement to fire and not be constant, plus diversity checks."""
    fails = []
    info = {}
    # embed spikes
    for gi, g in enumerate(goldens):
        if 'embed_spikes' in g:
            on, tot = _spk_stat(g['embed_spikes'])
            info['s%d.embed.on' % gi] = on
            info['s%d.embed.tot' % gi] = tot
            if on == 0 or on == tot:
                fails.append('sample%d embed constant (%d/%d)' % (gi, on, tot))
    # per block stage spikes
    for name in ('q', 'k', 'v', 'proj', 'hid', 'fc2'):
        on = 0
        tot = 0
        for gi, g in enumerate(goldens):
            for b in range(cfg['DEPTH']):
                o, tt = _spk_stat(g['blocks'][b][name])
                on += o
                tot += tt
        info['block.%s.on' % name] = on
        info['block.%s.tot' % name] = tot
        if on == 0 or on == tot:
            fails.append('block %s stage constant (%d/%d)' % (name, on, tot))
    # bounds / residual ranges
    max_res = 0
    max_av = 0
    max_scr = 0
    for gi, g in enumerate(goldens):
        for b in range(cfg['DEPTH']):
            bd = g['blocks'][b]
            max_res = max(max_res, int(bd['ares'].max()), int(bd['out'].max()))
            max_av = max(max_av, int(bd['av'].max()))
            max_scr = max(max_scr, int(bd['scr'].max()))
    info['max_residual'] = max_res
    info['max_av'] = max_av
    info['max_score'] = max_scr
    bound = 1 + 2 * cfg['DEPTH']
    if max_res > bound:
        fails.append('residual %d > bound %d' % (max_res, bound))
    if max_res > (1 << cfg['RW']) - 1:
        fails.append('residual %d does not fit RW=%d' % (max_res, cfg['RW']))
    if max_av > (1 << 10):
        fails.append('AV %d > contract max 1024' % max_av)
    if max_scr > cfg['HEAD_DIM']:
        fails.append('score %d > HEAD_DIM %d' % (max_scr, cfg['HEAD_DIM']))
    # per-block thresholds / weights must differ (multi-block fixtures)
    if cfg['DEPTH'] >= 2:
        same_w = all(np.array_equal(params['blocks'][0][k], params['blocks'][1][k])
                     for k in ('q_w', 'k_w', 'v_w', 'proj_w', 'fc1_w', 'fc2_w'))
        same_t = all(params['blocks'][0][k] == params['blocks'][1][k]
                     for k in ('thr_q', 'thr_k', 'thr_v', 'thr_proj', 'thr_fc1', 'thr_fc2'))
        if same_w:
            fails.append('block0/block1 weights identical')
        if same_t:
            fails.append('block0/block1 thresholds identical')
    # final outputs must differ between samples / runs
    if len(goldens) >= 2:
        o0 = goldens[0]['out']
        o1 = goldens[1]['out']
        if np.array_equal(o0, o1):
            fails.append('two samples/runs produce identical final outputs')
    # multi-head: per-head Q/K/V/score patterns must differ
    if cfg['HEADS'] > 1:
        bd = goldens[0]['blocks'][0]
        HD = cfg['HEAD_DIM']
        for nm in ('q', 'k', 'v'):
            x = bd[nm]
            h0 = x[:, 0:HD, :]
            h1 = x[:, HD:2 * HD, :]
            if np.array_equal(h0, h1):
                fails.append('multihead %s head0==head1 (no distinct patterns)' % nm)
        scr = bd['scr']
        for h in range(1, cfg['HEADS']):
            if np.array_equal(scr[0], scr[h]):
                fails.append('multihead scores head0==head%d' % h)
        info['head0_spikes'] = int(bd['q'][:, 0:HD, :].sum())
        info['head1_spikes'] = int(bd['q'][:, HD:2 * HD, :].sum())
    return dict(ok=(len(fails) == 0), fails=fails, info=info)


# --------------------------------------------------------------------------
# packing (contract ABI)
# --------------------------------------------------------------------------
def _words_from_int8_flat(flat):
    flat = np.asarray(flat, dtype=np.int64).ravel()
    assert flat.size % 8 == 0, 'int8 beat packing needs a multiple of 8 values'
    b = (flat & 0xFF).astype(np.uint64).reshape(-1, 8)
    shifts = (np.arange(8, dtype=np.uint64) * 8)
    return (b << shifts).sum(axis=1, dtype=np.uint64).tolist()


def words_int8_mat(M):
    return _words_from_int8_flat(np.asarray(M).ravel())


def words_i32_pairs(b):
    b = np.asarray(b, dtype=np.int64).ravel()
    assert b.size % 2 == 0
    u = (b & 0xFFFFFFFF)
    return ((u[0::2] | (u[1::2] << 32)).astype(np.uint64)).tolist()


def pack_block_beats(vals, cfg):
    """(N,E,T) unsigned residual -> N*G block I/O beats (TL lanes x T x RW)."""
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    TL, RW = cfg['TL'], cfg['RW']
    G = E // TL
    vals = np.asarray(vals, dtype=np.int64)
    assert vals.shape == (N, E, T)
    out = []
    for n in range(N):
        for g in range(G):
            lane = vals[n, g * TL:(g + 1) * TL, :].ravel()
            w = 0
            for i in range(TL * T):
                w |= int(lane[i]) << (i * RW)
            out.append(w)
    return out


def pack_embed_beats(vals, cfg):
    """(N,E,T) 0/1 spikes -> N*E/LANES embed beats (LANES lanes x T)."""
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    LANES = cfg['LANES']
    G = E // LANES
    vals = np.asarray(vals, dtype=np.int64)
    assert vals.shape == (N, E, T)
    out = []
    for n in range(N):
        for g in range(G):
            lane = vals[n, g * LANES:(g + 1) * LANES, :].ravel()
            w = 0
            for i in range(LANES * T):
                w |= int(lane[i]) << i
            out.append(w)
    return out


def desc_words(pb, cfg, is_block0, thr_e):
    """Contract descriptor beats, Verilog concat order (left operand = high half):
       beat0 = {thr_q[31:0], thr_k[31:0]}   -> thr_q in [63:32], thr_k in [31:0]
       beat1 = {thr_v[31:0], thr_proj[31:0]}
       beat2 = {thr_fc1[31:0], thr_fc2[31:0]}
       beat3 = {32'h0, thr_e[31:0]}         (meaningful for block 0 only)"""
    q, k, v = int(pb['thr_q']), int(pb['thr_k']), int(pb['thr_v'])
    pj, f1, f2 = int(pb['thr_proj']), int(pb['thr_fc1']), int(pb['thr_fc2'])
    w0 = ((q & 0xFFFFFFFF) << 32) | (k & 0xFFFFFFFF)
    w1 = ((v & 0xFFFFFFFF) << 32) | (pj & 0xFFFFFFFF)
    w2 = ((f1 & 0xFFFFFFFF) << 32) | (f2 & 0xFFFFFFFF)
    w3 = (int(thr_e) & 0xFFFFFFFF) if is_block0 else 0
    return [w0, w1, w2, w3]


def block_weight_words(pb, cfg):
    """Contract block_weight_stream_order (no descriptors)."""
    words = []
    for nm in ('q_w', 'k_w', 'v_w', 'proj_w'):
        words += words_int8_mat(pb[nm])
    for nm in ('q_b', 'k_b', 'v_b', 'proj_b'):
        words += words_i32_pairs(pb[nm])
    words += words_int8_mat(pb['fc1_w'])
    words += words_i32_pairs(pb['fc1_b'])
    words += words_int8_mat(pb['fc2_w'])
    words += words_i32_pairs(pb['fc2_b'])
    return words


def wbeats(cfg):
    """Weight beats per block (without descriptors) exactly per contract."""
    E, HIDDEN = cfg['E'], cfg['HIDDEN']
    return (4 * E * E // 8) + (4 * E // 2) + (HIDDEN * E // 8) + (HIDDEN // 2) \
        + (E * HIDDEN // 8) + (E // 2)


def stream_words(pb, cfg, is_block0, thr_e):
    return desc_words(pb, cfg, is_block0, thr_e) + block_weight_words(pb, cfg)


# --------------------------------------------------------------------------
# file dumping
# --------------------------------------------------------------------------
def write_hex(path, words, bits):
    """Write one hex word per line; bits = word width in bits."""
    digits = (bits + 3) // 4
    mask = (1 << bits) - 1
    with open(path, 'w', encoding='ascii', newline='\n') as f:
        for w in words:
            f.write(('%0*x\n' % (digits, int(w) & mask)))


def _f32_bits_hex(v):
    return struct.pack('<f', float(np.float32(v))).hex()


def iref_cfg(cfg):
    """Config dict in the frozen integer_ref.py schema."""
    return dict(E=cfg['E'], H=cfg['HEADS'], HD=cfg['HEAD_DIM'], N=cfg['TOKENS'],
                T=cfg['T'], D=cfg['DEPTH'], MLP=cfg['HIDDEN'],
                C=cfg['NUM_CLASSES'], vocab_size=cfg['VTAB'])


def dump_params_npz(path, cfg, params, kind):
    """Write checkpoint keys exactly as tools/backends/spikformer/integer_ref.py expects."""
    d = {}
    ep = params.get('embed')
    if ep is not None:
        d['etab'] = np.asarray(ep['etab'], dtype=np.int8)
        d['ptab'] = np.asarray(ep['ptab'], dtype=np.int8)
        d['thr_e'] = np.asarray([ep['thr_e']], dtype=np.int32)
    for b, pb in enumerate(params['blocks']):
        p = 'block%d.' % b
        for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
            W = np.asarray(pb[nm + '_w'], dtype=np.int8)
            d[p + nm + '_w'] = W
            d[p + nm + '_b'] = np.asarray(pb[nm + '_b'], dtype=np.int32)
            mx = int(np.abs(W.astype(np.int64)).max()) if W.size else 0
            d[p + nm + '_scale'] = np.asarray([mx / 127.0 if mx else 1.0], dtype=np.float32)
            d[p + nm + '_thr'] = np.asarray([pb['thr_' + nm]], dtype=np.int32)
    d['head_w'] = np.asarray(params['head_w'], dtype=np.float32)
    d['head_b'] = np.asarray(params['head_b'], dtype=np.float32)
    d['meta_model_kind'] = np.asarray('development_fixture')
    d['meta_bn_state'] = np.asarray('absent')
    d['meta_case'] = np.asarray(kind)
    d['meta_seed'] = np.asarray([int(params['seed'])], dtype=np.int32)
    d['meta_cfg'] = np.asarray(json.dumps(iref_cfg(cfg)))
    np.savez(path, **d)


def dump_top_case(kind, cfg, params, tokens_list, goldens, act):
    """Dump vectors for a top-level replay case (embed -> DEPTH blocks -> sums)."""
    out = os.path.join(REPLAY_DIR, 'case_' + kind)
    os.makedirs(out, exist_ok=True)
    ep = params['embed']
    thr_e = int(ep['thr_e'])
    cis = cfg['T'] * cfg['TOKENS']
    # tables
    write_hex(os.path.join(out, 'etab.hex'), words_int8_mat(ep['etab']), 64)
    write_hex(os.path.join(out, 'ptab.hex'), words_int8_mat(ep['ptab']), 64)
    files = {'etab': 'etab.hex', 'ptab': 'ptab.hex'}
    samples = []
    for s in range(len(tokens_list)):
        toks = tokens_list[s]
        write_hex(os.path.join(out, 'toks_s%d.hex' % s), [int(x) for x in toks], 16)
        files['toks_s%d' % s] = 'toks_s%d.hex' % s
        g = goldens[s]
        # embed beats (LANES*T bits) and block-in/out beats (TL*T*RW bits)
        write_hex(os.path.join(out, 'exp_emb_s%d.hex' % s),
                  pack_embed_beats(g['embed_spikes'], cfg), cfg['LANES'] * cfg['T'])
        files['exp_emb_s%d' % s] = 'exp_emb_s%d.hex' % s
        prev_in = g['embed_spikes']
        for b in range(cfg['DEPTH']):
            write_hex(os.path.join(out, 'stream_b%d_s%d.hex' % (b, s)),
                      stream_words(params['blocks'][b], cfg, b == 0, thr_e), 64)
            files['stream_b%d_s%d' % (b, s)] = 'stream_b%d_s%d.hex' % (b, s)
            write_hex(os.path.join(out, 'exp_in_b%d_s%d.hex' % (b, s)),
                      pack_block_beats(prev_in, cfg), cfg['TL'] * cfg['T'] * cfg['RW'])
            files['exp_in_b%d_s%d' % (b, s)] = 'exp_in_b%d_s%d.hex' % (b, s)
            write_hex(os.path.join(out, 'exp_out_b%d_s%d.hex' % (b, s)),
                      pack_block_beats(g['blocks'][b]['out'], cfg),
                      cfg['TL'] * cfg['T'] * cfg['RW'])
            files['exp_out_b%d_s%d' % (b, s)] = 'exp_out_b%d_s%d.hex' % (b, s)
            prev_in = g['blocks'][b]['out']
        write_hex(os.path.join(out, 'exp_sums_s%d.hex' % s),
                  [int(x) for x in g['sums']], cfg['OW'])
        files['exp_sums_s%d' % s] = 'exp_sums_s%d.hex' % s
        samples.append(dict(
            sample=s, tokens=[int(x) for x in toks],
            content_len=int(np.count_nonzero(toks)) if kind != 'small' else None,
            sums=[int(x) for x in g['sums']],
            ps_x=[float(x) for x in g['ps_x']],
            ps_x_bits=[_f32_bits_hex(x) for x in g['ps_x']],
            logits=[float(x) for x in g['logits']],
            logits_bits=[_f32_bits_hex(x) for x in g['logits']],
            pred=int(g['pred']),
            block_thr=[dict((k, int(params['blocks'][b]['thr_' + k]))
                            for k in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'))
                       for b in range(cfg['DEPTH'])],
            files=dict((k, v) for k, v in files.items() if k.endswith('_s%d' % s)),
        ))
    dump_params_npz(os.path.join(out, 'params.npz'), cfg, params, kind)
    meta = dict(
        schema='spikformer-nlp-replay-golden-1',
        case=kind, kind='top_level', cfg=cfg, seed=int(params['seed']),
        fixture='development_fixture',
        contract=CONTRACT_REL,
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
        wbeats=wbeats(cfg), table_beats=cfg['VTAB'] * cfg['E'] // 8,
        ptab_beats=cfg['TOKENS'] * cfg['E'] // 8,
        block_beats=cfg['TOKENS'] * (cfg['E'] // cfg['TL']),
        embed_beats=cfg['TOKENS'] * (cfg['E'] // cfg['LANES']),
        sums_ow=cfg['OW'], n_samples=len(tokens_list),
        head_norm='ps_div', t_times_n=cis,
        thresholds=dict(thr_e=thr_e,
                        blocks=[dict((k, int(params['blocks'][b]['thr_' + k]))
                                     for k in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'))
                                for b in range(cfg['DEPTH'])]),
        activity=act['info'],
        samples=samples,
        files=files,
        params='params.npz',
    )
    with open(os.path.join(out, 'golden.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
    return out, meta


def dump_block_case(kind, cfg, params, r0_list, goldens, act):
    """Dump vectors for a direct block-level replay case (multihead)."""
    out = os.path.join(REPLAY_DIR, 'case_' + kind)
    os.makedirs(out, exist_ok=True)
    pb = params['blocks'][0]
    thr_e = int(params['embed']['thr_e']) if params.get('embed') else 0
    write_hex(os.path.join(out, 'stream_b0.hex'),
              block_weight_words(pb, cfg), 64)
    write_hex(os.path.join(out, 'desc_b0.hex'), desc_words(pb, cfg, True, thr_e), 64)
    files = {'stream_b0': 'stream_b0.hex', 'desc_b0': 'desc_b0.hex'}
    runs = []
    for s in range(len(r0_list)):
        write_hex(os.path.join(out, 'in_b%d.hex' % s),
                  pack_block_beats(r0_list[s], cfg), cfg['TL'] * cfg['T'] * cfg['RW'])
        files['in_b%d' % s] = 'in_b%d.hex' % s
        write_hex(os.path.join(out, 'exp_out_b%d.hex' % s),
                  pack_block_beats(goldens[s]['out'], cfg),
                  cfg['TL'] * cfg['T'] * cfg['RW'])
        files['exp_out_b%d.hex' % s] = 'exp_out_b%d.hex' % s
        runs.append(dict(run=s, files={'in': 'in_b%d.hex' % s,
                                       'exp_out': 'exp_out_b%d.hex' % s},
                         block_beats=cfg['TOKENS'] * (cfg['E'] // cfg['TL'])))
    dump_params_npz(os.path.join(out, 'params.npz'), cfg, params, kind)
    meta = dict(
        schema='spikformer-nlp-replay-golden-1',
        case=kind, kind='block_level', cfg=cfg, seed=int(params['seed']),
        fixture='development_fixture', contract=CONTRACT_REL,
        generated_at=datetime.datetime.now().isoformat(timespec='seconds'),
        wbeats=wbeats(cfg), block_beats=cfg['TOKENS'] * (cfg['E'] // cfg['TL']),
        thresh=dict((k, int(pb['thr_' + k])) for k in
                    ('q', 'k', 'v', 'proj', 'fc1', 'fc2')),
        activity=act['info'], runs=runs, files=files, params='params.npz')
    with open(os.path.join(out, 'golden.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
    return out, meta


# --------------------------------------------------------------------------
# integer_ref cross-check hook (golden truth owned by trainer line)
# --------------------------------------------------------------------------
def _as_btne(x):
    """(N,E,T) local layout -> [1,T,N,E] (integer_ref layout)."""
    return np.transpose(np.asarray(x), (2, 0, 1))[None, ...]


def _first_mismatch_arrays(checks):
    """First differing (layer, index, ref, got) between two array lists."""
    for name, ref, got in checks:
        a = np.asarray(ref)
        b = np.asarray(got)
        if a.shape != b.shape:
            return dict(layer=name, kind='shape', ref=list(a.shape), got=list(b.shape))
        if a.size == 0:
            continue
        af = a.reshape(-1)
        bf = b.reshape(-1)
        neq = np.argwhere(af != bf)
        if len(neq):
            i = int(neq[0][0])
            idx = np.unravel_index(i, a.shape)
            return dict(layer=name, kind='value', index=[int(x) for x in idx],
                        ref=int(af[i]), got=int(bf[i]), shape=list(a.shape))
    return None


def _block_checks(cfg, b, tr, golden):
    """Comparison list for one block: integer_ref trace vs local golden trace."""
    mb = golden['blocks'][b]
    p = 'block%d' % b
    scr_ref = np.transpose(np.asarray(mb['scr']), (3, 1, 2, 0))[None, ...]  # [1,T,N,N,H]
    return [
        (p + '.in', tr['in'], _as_btne(mb['inp'])),
        (p + '.q_spikes', tr['q_spikes'], _as_btne(mb['q'])),
        (p + '.k_spikes', tr['k_spikes'], _as_btne(mb['k'])),
        (p + '.v_spikes', tr['v_spikes'], _as_btne(mb['v'])),
        (p + '.scores', tr['scores'], scr_ref),
        (p + '.av', tr['av'], _as_btne(mb['av'])),
        (p + '.proj_spikes', tr['proj_spikes'], _as_btne(mb['proj'])),
        (p + '.res1', tr['res1'], _as_btne(mb['ares'])),
        (p + '.fc1_spikes', tr['fc1_spikes'], _as_btne(mb['hid'])),
        (p + '.fc2_spikes', tr['fc2_spikes'], _as_btne(mb['fc2'])),
        (p + '.out', tr['out'], _as_btne(mb['out'])),
    ]


def crosscheck_integer_ref(kind, cfg, params, tokens_list, r0_list, goldens, case_dir):
    """Full-trace cross-check against the golden truth integer_ref.py.

    Integer_ref is owned by the trainer line and is the frozen golden truth for
    RTL replay; this generator must agree with it bit-exactly at every stage
    (embed spikes, per-block q/k/v/scores/av/proj/ares/fc1/fc2/out, final
    residual, PL sums, PS logits/class). The result is recorded in golden.json.
    """
    try:
        from tools.backends.spikformer import integer_ref as iref
    except Exception as exc:  # pragma: no cover - depends on trainer delivery
        return dict(status='not_available', detail=repr(exc))
    icfg = iref_cfg(cfg)
    npz = os.path.join(case_dir, 'params.npz')
    try:
        p = iref.load_checkpoint(npz, cfg=icfg)
    except Exception as exc:
        return dict(status='load_failed', detail=repr(exc), params=npz)
    results = []
    for s in range(len(goldens)):
        checks = []
        if kind == 'multihead':
            x = _as_btne(r0_list[s])
            out, tr = iref.block_forward(p['blocks'][0], x, icfg, trace=True)
            checks = _block_checks(cfg, 0, tr, goldens[s])
            checks.append(('final_residual', out, _as_btne(goldens[s]['out'])))
        else:
            ref = iref.run_sample(p, np.asarray(tokens_list[s], dtype=np.int64),
                                  cfg=icfg, trace=True)
            checks = [('embed_spikes', ref['embed_spikes'],
                       _as_btne(goldens[s]['embed_spikes']))]
            for b in range(cfg['DEPTH']):
                checks += _block_checks(cfg, b, ref['blocks'][b], goldens[s])
            checks += [
                ('final_residual', ref['final_residual'], _as_btne(goldens[s]['out'])),
                ('pl_sums', ref['pl_sums'],
                 np.asarray(goldens[s]['sums'], dtype=np.uint32)[None, :]),
                ('logits', ref['logits'],
                 np.asarray(goldens[s]['logits'], dtype=np.float32)[None, :]),
                ('cls', ref['cls'], np.asarray([goldens[s]['pred']], dtype=np.int64)),
            ]
        mm = _first_mismatch_arrays(checks)
        results.append(dict(sample=s, match=(mm is None), first_mismatch=mm))
    ok = all(r['match'] for r in results)
    import hashlib
    try:
        md5 = hashlib.md5(open(iref.__file__, 'rb').read()).hexdigest()
    except Exception:
        md5 = None
    return dict(status='match' if ok else 'mismatch',
                module='tools/backends/spikformer/integer_ref.py',
                module_md5=md5,
                comparison='integer_ref vs independent contract generator '
                           '(both directions; full per-stage traces)',
                params=npz, samples=results)






# --------------------------------------------------------------------------
# external checkpoint mode (final gate: real_weight_replay)
# --------------------------------------------------------------------------
def _ref_to_mine(ref, cfg):
    """Convert integer_ref.run_sample(trace=True) output to this module's layout."""
    def t(x):
        a = np.asarray(x)
        if a.ndim == 4:
            a = a[0]
        return np.transpose(a, (1, 2, 0)).astype(np.int64)   # [T,N,E] -> (N,E,T)
    emb = t(ref['embed_spikes'])
    blocks = []
    for b in range(cfg['DEPTH']):
        tr = ref['blocks'][b]
        scr = np.transpose(np.asarray(tr['scores'])[0], (3, 1, 2, 0)).astype(np.int64)
        blocks.append(dict(inp=t(tr['in']), q=t(tr['q_spikes']), k=t(tr['k_spikes']),
                           v=t(tr['v_spikes']), scr=scr, av=t(tr['av']),
                           proj=t(tr['proj_spikes']), ares=t(tr['res1']),
                           hid=t(tr['fc1_spikes']), fc2=t(tr['fc2_spikes']),
                           out=t(tr['out'])))
    return dict(embed_spikes=emb, blocks=blocks, out=t(ref['final_residual']),
                sums=np.asarray(ref['pl_sums'][0], dtype=np.int64),
                ps_x=np.asarray(ref['pooled'][0], dtype=np.float32),
                logits=np.asarray(ref['logits'][0], dtype=np.float32),
                pred=int(np.asarray(ref['cls'][0])))


def load_real_params(cfg, ckpt_path):
    """Load an external .npz strictly through integer_ref.load_checkpoint."""
    from tools.backends.spikformer import integer_ref as iref
    p = iref.load_checkpoint(ckpt_path, cfg=iref_cfg(cfg))
    params = dict(cfg=cfg, seed=0, kind='real', source=ckpt_path,
                  embed=dict(etab=np.asarray(p['etab'], dtype=np.int8),
                             ptab=np.asarray(p['ptab'], dtype=np.int8),
                             thr_e=int(p['thr_e'])),
                  head_w=np.asarray(p['head_w'], dtype=np.float32),
                  head_b=np.asarray(p['head_b'], dtype=np.float32))
    blocks = []
    for blk in p['blocks']:
        pb = {}
        for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
            pb[nm + '_w'] = np.asarray(blk[nm + '_w'], dtype=np.int8)
            pb[nm + '_b'] = np.asarray(blk[nm + '_b'], dtype=np.int32)
            pb['thr_' + nm] = int(blk[nm + '_thr'])
        blocks.append(pb)
    params['blocks'] = blocks
    params['model_kind'] = str(p.get('model_kind', 'real_corpus'))
    return params, p


def load_tokens_file(path, seq_len, vocab):
    """Fixed sample token ids for the real-weight replay.

    Accepted formats (all validated: exactly seq_len ids per sample, 0<=id<vocab):
      .json : {"samples": [[...], ...], "names": [...]} or [[...], ...]
      .npy  : int array [n_samples, seq_len]
      .hex/.txt : one sample per line, ids separated by comma/space (hex if 0x-prefixed)
    """
    samples = []
    names = []
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            samples = obj.get('samples') or obj.get('token_ids') or obj.get('ids')
            names = obj.get('names') or []
            if samples is None:
                raise ValueError('tokens json needs "samples"/"token_ids"/"ids"')
        else:
            samples = obj
    elif path.endswith('.npy'):
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        samples = [[int(x) for x in row] for row in arr]
    else:
        raw = open(path, 'r', encoding='utf-8').read().splitlines()
        for line in raw:
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            row = []
            for tok in line.replace(',', ' ').split():
                row.append(int(tok, 16) if tok.lower().startswith('0x') else int(tok))
            samples.append(row)
    out = []
    for i, row in enumerate(samples):
        row = [int(x) for x in row]
        if len(row) != seq_len:
            raise ValueError('tokens sample %d has %d ids, want seq_len=%d'
                             % (i, len(row), seq_len))
        bad = [x for x in row if x < 0 or x >= vocab]
        if bad:
            raise ValueError('tokens sample %d has ids outside [0,%d): %r'
                             % (i, vocab, bad[:8]))
        out.append(np.asarray(row, dtype=np.int64))
    if not out:
        raise ValueError('no samples in tokens file ' + path)
    return out, names


def _sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def generate_real(ckpt_path, tokens_path, vocab_path=None, out_kind='real',
                  max_samples=3, verbose=True):
    """Generate target-size replay vectors from an externally trained checkpoint.

    Golden truth = integer_ref.run_sample on the same strict-loaded checkpoint
    (external checkpoint mode: no fixture calibration, no seed retry).
    """
    from tools.backends.spikformer import integer_ref as iref
    cfg = dict(CASE_CFG['e2e'])
    cfg['name'] = out_kind
    params, iref_params = load_real_params(cfg, ckpt_path)
    tokens_all, names = load_tokens_file(tokens_path, cfg['TOKENS'], cfg['VTAB'])
    if len(tokens_all) > max_samples:
        tokens_all = tokens_all[:max_samples]
        if verbose:
            print('real: using first %d of the provided samples' % max_samples)
    goldens = []
    for toks in tokens_all:
        ref = iref.run_sample(iref_params, toks, cfg=iref_cfg(cfg), trace=True)
        goldens.append(_ref_to_mine(ref, cfg))
    act = activity_check(cfg, params, goldens)
    out, meta = dump_top_case(out_kind, cfg, params, tokens_all, goldens, act)
    # provenance: checkpoint + vocab + fixed samples (no fixture identity)
    meta['fixture'] = 'real_checkpoint'
    meta['model_kind'] = params.get('model_kind', 'real_corpus')
    meta['n_samples'] = len(tokens_all)
    meta['checkpoint'] = dict(path=os.path.relpath(os.path.abspath(ckpt_path), REPO),
                              sha256=_sha256_file(ckpt_path))
    if vocab_path and os.path.isfile(vocab_path):
        meta['vocab'] = dict(path=os.path.relpath(os.path.abspath(vocab_path), REPO),
                             sha256=_sha256_file(vocab_path))
    else:
        meta['vocab'] = dict(path=None, sha256=None,
                             note='no vocab file supplied on the command line')
    meta['tokens_source'] = dict(
        path=os.path.relpath(os.path.abspath(tokens_path), REPO),
        sha256=_sha256_file(tokens_path), names=[str(n) for n in names],
        samples=[[int(x) for x in t] for t in tokens_all])
    meta['golden_source'] = 'tools/backends/spikformer/integer_ref.py run_sample(trace=True)'
    import hashlib
    try:
        from tools.backends.spikformer import integer_ref as _iref
        _md5 = hashlib.md5(open(_iref.__file__, 'rb').read()).hexdigest()
    except Exception:
        _md5 = None
    meta['integer_ref_crosscheck'] = dict(
        status='source_of_truth', module_md5=_md5,
        detail='golden vectors/expected sums/logits produced directly by '
               'integer_ref.run_sample(trace=True) on the strict-loaded checkpoint; '
               'no independent generator reimplementation is involved in this mode')
    sc = selfcheck_case(out_kind, cfg, params, tokens_all, None, goldens, out)
    meta['packing_selfcheck'] = sc
    with open(os.path.join(out, 'golden.json'), 'w', encoding='ascii', newline='\n') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
    if verbose:
        print('real case generated: dir=%s samples=%d' % (os.path.relpath(out, REPO),
                                                          len(tokens_all)))
        print('  checkpoint sha256=%s' % meta['checkpoint']['sha256'])
        print('  vocab sha256=%s' % meta['vocab']['sha256'])
        print('  packing selfcheck:', 'ok' if sc['ok'] else sc['fails'][:5])
        for s, g in enumerate(goldens):
            print('  sample%d pred=%d logits=%s sums[0:4]=%s'
                  % (s, g['pred'], np.asarray(g['logits']).tolist(),
                     np.asarray(g['sums'])[:4].tolist()))
    return out, meta, sc


# --------------------------------------------------------------------------
# file-level self-check: decode dumped hex and compare against golden/model
# (never claim vectors are good just because files exist)
# --------------------------------------------------------------------------
def read_hex_file(path):
    vals = []
    with open(path, 'r', encoding='ascii') as f:
        for line in f:
            line = line.strip()
            if line:
                vals.append(int(line, 16))
    return vals


def _s32(x):
    x = int(x) & 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def decode_int8_words(words):
    out = np.zeros(len(words) * 8, dtype=np.int64)
    for i, w in enumerate(words):
        w = int(w)
        for k in range(8):
            b = (w >> (8 * k)) & 0xFF
            out[i * 8 + k] = b - 256 if b >= 128 else b
    return out


def decode_i32_pairs(words):
    out = np.zeros(len(words) * 2, dtype=np.int64)
    for i, w in enumerate(words):
        w = int(w)
        out[2 * i] = _s32(w)
        out[2 * i + 1] = _s32(w >> 32)
    return out


def decode_desc(words):
    assert len(words) == 4
    w0, w1, w2, w3 = [int(x) for x in words]
    # Verilog concat order: left operand is the high half
    return dict(q=_s32(w0 >> 32), k=_s32(w0), v=_s32(w1 >> 32), proj=_s32(w1),
                fc1=_s32(w2 >> 32), fc2=_s32(w2), thr_e=(w3 & 0xFFFFFFFF))


def decode_block_beats(words, cfg):
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    TL, RW = cfg['TL'], cfg['RW']
    G = E // TL
    assert len(words) == N * G, 'block beat count %d != %d' % (len(words), N * G)
    out = np.zeros((N, E, T), dtype=np.int64)
    mask = (1 << RW) - 1
    idx = 0
    for n in range(N):
        for g in range(G):
            w = int(words[idx])
            idx += 1
            for l in range(TL):
                a = g * TL + l
                for tt in range(T):
                    out[n, a, tt] = (w >> ((l * T + tt) * RW)) & mask
    return out


def decode_embed_beats(words, cfg):
    N, E, T = cfg['TOKENS'], cfg['E'], cfg['T']
    LANES = cfg['LANES']
    G = E // LANES
    assert len(words) == N * G, 'embed beat count %d != %d' % (len(words), N * G)
    out = np.zeros((N, E, T), dtype=np.int64)
    idx = 0
    for n in range(N):
        for g in range(G):
            w = int(words[idx])
            idx += 1
            for l in range(LANES):
                a = g * LANES + l
                for tt in range(T):
                    out[n, a, tt] = (w >> (l * T + tt)) & 1
    return out


def _decode_block_stream(words, cfg, has_desc=True):
    """Split a block stream (optional 4 descriptor beats + weight beats) into fields."""
    E, HIDDEN = cfg['E'], cfg['HIDDEN']
    want = (4 if has_desc else 0) + wbeats(cfg)
    assert len(words) == want, 'stream len %d != %d' % (len(words), want)
    d = decode_desc(words[:4]) if has_desc else {}
    off = 4 if has_desc else 0
    out = {}
    for nm in ('q_w', 'k_w', 'v_w', 'proj_w'):
        n = E * E // 8
        out[nm] = decode_int8_words(words[off:off + n]).reshape(E, E)
        off += n
    for nm in ('q_b', 'k_b', 'v_b', 'proj_b'):
        n = E // 2
        out[nm] = decode_i32_pairs(words[off:off + n])
        off += n
    n = HIDDEN * E // 8
    out['fc1_w'] = decode_int8_words(words[off:off + n]).reshape(HIDDEN, E)
    off += n
    n = HIDDEN // 2
    out['fc1_b'] = decode_i32_pairs(words[off:off + n])
    off += n
    n = E * HIDDEN // 8
    out['fc2_w'] = decode_int8_words(words[off:off + n]).reshape(E, HIDDEN)
    off += n
    n = E // 2
    out['fc2_b'] = decode_i32_pairs(words[off:off + n])
    off += n
    assert off == len(words), 'block stream trailing words: %d != %d' % (off, len(words))
    return d, out


def _cmp(name, a, b, checks, fails):
    a = np.asarray(a)
    b = np.asarray(b)
    ok = a.shape == b.shape and bool(np.array_equal(a, b))
    checks.append(name)
    if not ok:
        fails.append('%s mismatch (shape %s vs %s)' % (name, a.shape, b.shape))
    return ok


def selfcheck_case(kind, cfg, params, tokens_list, r0_list, goldens, case_dir):
    """Decode every dumped vector file and compare with the in-memory model."""
    checks = []
    fails = []
    ep = params['embed']
    if kind != 'multihead':
        _cmp('etab', decode_int8_words(read_hex_file(os.path.join(case_dir, 'etab.hex'))).reshape(
            cfg['VTAB'], cfg['E']), np.asarray(ep['etab']), checks, fails)
        _cmp('ptab', decode_int8_words(read_hex_file(os.path.join(case_dir, 'ptab.hex'))).reshape(
            cfg['TOKENS'], cfg['E']), np.asarray(ep['ptab']), checks, fails)
    thr_e = int(ep['thr_e'])
    nblk = cfg['DEPTH']
    for s in range(len(goldens)):
        g = goldens[s]
        if kind != 'multihead':
            _cmp('toks_s%d' % s, np.array(read_hex_file(os.path.join(case_dir, 'toks_s%d.hex' % s)),
                                           dtype=np.int64), tokens_list[s], checks, fails)
            _cmp('exp_emb_s%d' % s, decode_embed_beats(
                read_hex_file(os.path.join(case_dir, 'exp_emb_s%d.hex' % s)), cfg),
                g['embed_spikes'], checks, fails)
        prev = g['embed_spikes'] if kind != 'multihead' else r0_list[s]
        for b in range(nblk):
            f = os.path.join(case_dir, ('stream_b%d_s%d.hex' % (b, s))
                             if kind != 'multihead' else 'stream_b0.hex')
            d, got = _decode_block_stream(read_hex_file(f), cfg,
                                          has_desc=(kind != 'multihead'))
            pb = params['blocks'][b]
            # descriptor semantic check (signed values)
            if kind != 'multihead':
                for key in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                    if d[key] != int(pb['thr_' + key]):
                        fails.append('desc_b%d_s%d %s %d != %d'
                                     % (b, s, key, d[key], int(pb['thr_' + key])))
                if b == 0 and d['thr_e'] != thr_e:
                    fails.append('desc thr_e %d != %d' % (d['thr_e'], thr_e))
            for nm in ('q_w', 'k_w', 'v_w', 'proj_w', 'fc1_w', 'fc2_w'):
                _cmp('w_b%d_s%d.%s' % (b, s, nm), got[nm],
                     np.asarray(pb[nm]), checks, fails)
            for nm in ('q_b', 'k_b', 'v_b', 'proj_b', 'fc1_b', 'fc2_b'):
                _cmp('b_b%d_s%d.%s' % (b, s, nm), got[nm],
                     np.asarray(pb[nm]), checks, fails)
            if kind != 'multihead':
                _cmp('exp_in_b%d_s%d' % (b, s), decode_block_beats(
                    read_hex_file(os.path.join(case_dir, 'exp_in_b%d_s%d.hex' % (b, s))), cfg),
                    prev, checks, fails)
                _cmp('exp_out_b%d_s%d' % (b, s), decode_block_beats(
                    read_hex_file(os.path.join(case_dir, 'exp_out_b%d_s%d.hex' % (b, s))), cfg),
                    g['blocks'][b]['out'], checks, fails)
                prev = g['blocks'][b]['out']
        if kind == 'multihead':
            _cmp('in_b%d' % s, decode_block_beats(
                read_hex_file(os.path.join(case_dir, 'in_b%d.hex' % s)), cfg),
                r0_list[s], checks, fails)
            _cmp('exp_out_b%d' % s, decode_block_beats(
                read_hex_file(os.path.join(case_dir, 'exp_out_b%d.hex' % s)), cfg),
                g['out'], checks, fails)
            if s == 0:
                d = decode_desc(read_hex_file(os.path.join(case_dir, 'desc_b0.hex')))
                pb = params['blocks'][0]
                for key in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                    if d[key] != int(pb['thr_' + key]):
                        fails.append('multihead desc %s %d != %d'
                                     % (key, d[key], int(pb['thr_' + key])))
        if kind != 'multihead':
            _cmp('exp_sums_s%d' % s, np.array(read_hex_file(
                os.path.join(case_dir, 'exp_sums_s%d.hex' % s)), dtype=np.int64),
                np.asarray(g['sums'], dtype=np.int64), checks, fails)
            if len(read_hex_file(os.path.join(case_dir, 'exp_sums_s%d.hex' % s))) != cfg['E']:
                fails.append('exp_sums_s%d line count != E' % s)
    return dict(ok=(len(fails) == 0), n_checks=len(checks), fails=fails)


# --------------------------------------------------------------------------
# top-level generation
# --------------------------------------------------------------------------
def make_params(kind, cfg, seed):
    if kind in ('small', 'e2e'):
        tokens_list = make_tokens(cfg, kind)
        embed = make_embed_params(cfg, seed, tokens_list)
        params = dict(cfg=cfg, seed=seed, kind=kind, embed=embed)
        r0_list = [embed_spikes(cfg, params, t) for t in tokens_list]
        params['blocks'] = make_block_params(cfg, seed, r0_list)
        params.update(make_head_params(cfg, seed))
        return params, tokens_list, r0_list
    if kind == 'multihead':
        # block-level: no embed encode needed, but keep a valid placeholder embed
        # dict so dump_params_npz stays uniform.
        r0_list = make_multihead_inputs(cfg)
        embed = make_embed_params(cfg, seed, [np.zeros(cfg['TOKENS'], dtype=np.int64)])
        params = dict(cfg=cfg, seed=seed, kind=kind, embed=embed)
        params['blocks'] = make_block_params(cfg, seed, r0_list, diagonal_heads=True)
        params.update(make_head_params(cfg, seed))
        return params, None, r0_list
    raise ValueError('unknown case %r' % kind)


def build_goldens(kind, cfg, params, r0_list):
    out = []
    for r0 in r0_list:
        g = forward_blocks(cfg, params, r0)
        g['embed_spikes'] = r0
        out.append(g)
    return out


def generate(kind, verbose=True):
    cfg = dict(CASE_CFG[kind])
    tokens_list = None
    last = None
    chosen = None
    for attempt in range(MAX_SEED_TRIES):
        seed = SEED_BASE[kind] + attempt
        params, tokens, r0_list = make_params(kind, cfg, seed)
        goldens = build_goldens(kind, cfg, params, r0_list)
        act = activity_check(cfg, params, goldens)
        last = (params, tokens, r0_list, goldens, act, seed)
        if act['ok']:
            chosen = last
            break
    if chosen is None:
        params, tokens, r0_list, goldens, act, seed = last
        if verbose:
            print('WARNING: no fully active fixture seed found for %r; fails=%s'
                  % (kind, act['fails']))
        chosen = last
    else:
        seed = chosen[5]
    params, tokens, r0_list, goldens, act, seed = chosen
    if kind == 'multihead':
        out, meta = dump_block_case(kind, cfg, params, r0_list, goldens, act)
    else:
        out, meta = dump_top_case(kind, cfg, params, tokens, goldens, act)
    ref = crosscheck_integer_ref(kind, cfg, params, tokens, r0_list, goldens, out)
    sc = selfcheck_case(kind, cfg, params, tokens, r0_list, goldens, out)
    if verbose:
        print('integer_ref crosscheck:', ref.get('status'),
              '' if ref.get('status') in ('match', 'not_available')
              else json.dumps(ref.get('samples'))[:800])
        print('packing selfcheck:', 'ok' if sc['ok'] else ('FAIL ' + str(sc['fails'][:5])),
              '(%d checks)' % sc['n_checks'])
    # record the crosscheck result into golden.json (rewrite right after dump)
    jp = os.path.join(out, 'golden.json')
    with open(jp, 'r', encoding='ascii') as f:
        meta = json.load(f)
    meta['integer_ref_crosscheck'] = ref
    meta['packing_selfcheck'] = sc
    with open(jp, 'w', encoding='ascii', newline='\n') as f:
        json.dump(meta, f, indent=1)
        f.write('\n')
    if verbose:
        print('case %r: seed=%d dir=%s' % (kind, seed, os.path.relpath(out, REPO)))
        for s, g in enumerate(goldens):
            print('  sample%d sums[0:4]=%s pred=%d' % (s, g['sums'][:4].tolist(), g['pred']))
        print('  activity: %s' % json.dumps(act['info']))
        if not act['ok']:
            print('  ACTIVITY FAILS: %s' % act['fails'])
    return out, meta


if __name__ == '__main__':
    for k in ('small', 'multihead', 'e2e'):
        generate(k)
