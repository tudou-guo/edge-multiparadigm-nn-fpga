
"""Parametric block golden generator for native chain replay (P2).

Reference equations/bit order are taken from scripts/spikformer/gen_block_replay.py
(E=32/H=1) generalized to HEADS>1, TOKENS>4, HIDDEN>64. The S_LDW host stream
order/packing is read from rtl/ops/spiking/control/spiking_block_top.sv:
  wq,wk,wv,wp  : row-major [E][E], 8 int8 per 64b beat            (EEB=E*E/8)
  bq,bk,bv,bp  : E/2 beats, {signed32 odd, signed32 even}        (2x PSUM_SLICE)
  fc1          : row-major [HIDDEN][E], 8 int8 per beat          (FCD=HIDDEN*E/8)
  bf1          : HIDDEN/2 beats (2x signed32)
  fc2          : row-major [E][HIDDEN], 8 int8 per beat
  bf2          : E/2 beats
  total = 4*EEB + 4*E/2 + FCD + HIDDEN/2 + FCD + E/2  (=222912 for native)

Usage:
  python gen_block_param.py --small-selftest     # diff vs build/spikformer/block_replay
  python gen_block_param.py --e 384 --heads 12 --hd 32 --hidden 1536 --t 4 \
    --tokens 64 --tl 8 --tc 8 --rw 4 --seed 52 \
    --tok-file <sps exp.hex> --out-dir <dir> --tag img1 [--dump-tok tok.hex]
"""
import argparse, json, os, sys
import numpy as np

def lif_chain_vec(deltas, thr):
    T = deltas.shape[-1]
    pre = np.zeros(deltas.shape[:-1], dtype=np.int64)
    out = np.zeros(deltas.shape, dtype=np.int64)
    for t in range(T):
        s = pre + deltas[..., t]
        mem = np.floor_divide(s, 2)
        sp = ((mem >= 0) & (mem >= thr)).astype(np.int64)
        out[..., t] = sp
        pre = np.where(sp.astype(bool), 0, mem)
    return out

def parse_sps_tokens(path, ntokens, E, T):
    G2 = E // 8
    tok = np.zeros((ntokens, E, T), dtype=np.int64)
    lines = [l.strip() for l in open(path) if l.strip()]
    assert len(lines) == ntokens * G2, (len(lines), ntokens * G2)
    for idx, line in enumerate(lines):
        g, pos = idx % G2, idx // G2
        for a in range(8):
            bits = int(line[a], 16)
            for t in range(T):
                tok[pos][g * 8 + a][t] = (bits >> t) & 1
    return tok

def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))

def dump_wmat(W, path):
    r, c = W.shape
    flat = W.reshape(-1)
    with open(path, 'w') as f:
        for i in range(0, len(flat), 8):
            word = 0
            for k in range(8):
                word |= (int(flat[i + k]) & 0xFF) << (8 * k)
            f.write(hx(word, 16) + '\n')

def mat_beats(W):
    r, c = W.shape
    flat = W.reshape(-1)
    out = []
    for i in range(0, len(flat), 8):
        word = 0
        for k in range(8):
            word |= (int(flat[i + k]) & 0xFF) << (8 * k)
        out.append(word)
    return out

def bias_beats(B):
    out = []
    for i in range(0, len(B), 2):
        lo = int(B[i]) & 0xFFFFFFFF
        hi = int(B[i + 1]) & 0xFFFFFFFF
        out.append(lo | (hi << 32))
    return out

def gen(e, heads, hd, hidden, T, ntokens, tl, tc, rw, seed, tok, out_dir, tag, thr=60, dump_legacy=False):
    assert heads * hd == e and hd % tl == 0 and e % tl == 0 and e % tc == 0
    assert hidden % tl == 0 and hidden % tc == 0
    G = e // tl
    mask = (1 << rw) - 1
    rng = np.random.RandomState(seed)
    wq = rng.randint(-64, 65, size=(e, e)).astype(np.int64)
    wk = rng.randint(-64, 65, size=(e, e)).astype(np.int64)
    wv = rng.randint(-64, 65, size=(e, e)).astype(np.int64)
    wp = rng.randint(-64, 65, size=(e, e)).astype(np.int64)
    bq = rng.randint(-200, 201, size=(e,)).astype(np.int64)
    bk = rng.randint(-200, 201, size=(e,)).astype(np.int64)
    bv = rng.randint(-200, 201, size=(e,)).astype(np.int64)
    bp = rng.randint(-200, 201, size=(e,)).astype(np.int64)
    wfc1 = rng.randint(-64, 65, size=(hidden, e)).astype(np.int64)
    bf1 = rng.randint(-200, 201, size=(hidden,)).astype(np.int64)
    wfc2 = rng.randint(-64, 65, size=(e, hidden)).astype(np.int64)
    bf2 = rng.randint(-200, 201, size=(e,)).astype(np.int64)
    if tok is None:
        # exactly like gen_block_replay.py: tokens are the next draw after bf2
        tok = rng.randint(0, 16, size=(ntokens, e, T)).astype(np.int64)
    n = ntokens
    def linear(W, B, thrv, x):
        od = W.shape[0]
        acc = np.zeros((n, od, T), dtype=np.int64)
        for t in range(T):
            acc[:, :, t] = x[:, :, t] @ W.T + B[None, :]
        return lif_chain_vec(acc, thrv)
    Q = linear(wq, bq, thr, tok)
    K = linear(wk, bk, thr, tok)
    V = linear(wv, bv, thr, tok)
    # attention: per-head QK popcount, gated AV (one head per channel)
    scr = np.zeros((n, n, heads, T), dtype=np.int64)
    AV = np.zeros((n, e, T), dtype=np.int64)
    for t in range(T):
        for h in range(heads):
            sl = slice(h * hd, (h + 1) * hd)
            scr[:, :, h, t] = Q[:, sl, t] @ K[:, sl, t].T
            AV[:, sl, t] = scr[:, :, h, t] @ V[:, sl, t]
    projspk = linear(wp, bp, thr, AV)
    ares = (tok + projspk) & mask
    hid = linear(wfc1, bf1, thr, ares)
    sp2 = linear(wfc2, bf2, thr, hid)
    out = (ares + sp2) & mask
    sums = out.sum(axis=(0, 2)).astype(np.uint32)
    # ---- streams ----
    beats = []
    for W in (wq, wk, wv, wp):
        beats += mat_beats(W)
    for B in (bq, bk, bv, bp):
        beats += bias_beats(B)
    beats += mat_beats(wfc1)
    beats += bias_beats(bf1)
    beats += mat_beats(wfc2)
    beats += bias_beats(bf2)
    EEB = e * e // 8
    FCD = hidden * e // 8
    exp_total = 4 * EEB + 4 * (e // 2) + FCD + hidden // 2 + FCD + e // 2
    assert len(beats) == exp_total, (len(beats), exp_total)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'blk_w.hex'), 'w') as f:
        for b in beats:
            f.write(hx(b, 16) + '\n')
    # tok beats (group-major within token; matches token input ABI)
    with open(os.path.join(out_dir, 'tok.hex'), 'w') as f:
        for nn in range(n):
            for g in range(G):
                word = 0
                for a in range(tl):
                    for t in range(T):
                        word |= (int(tok[nn, g * tl + a, t]) & mask) << ((a * T + t) * rw)
                f.write(hx(word, tl * T * rw // 4) + '\n')
    beats_per = tl * T * rw // 4
    with open(os.path.join(out_dir, 'exp_out_%s.hex' % tag), 'w') as f:
        for nn in range(n):
            for g in range(G):
                word = 0
                for a in range(tl):
                    for t in range(T):
                        word |= (int(out[nn, g * tl + a, t]) & mask) << ((a * T + t) * rw)
                f.write(hx(word, beats_per) + '\n')
    with open(os.path.join(out_dir, 'exp_sums_%s.hex' % tag), 'w') as f:
        for v in sums:
            f.write('%08x\n' % (int(v) & 0xFFFFFFFF))
    if dump_legacy:
        for nm, W in (('wq', wq), ('wk', wk), ('wv', wv), ('wp', wp),
                      ('wfc1', wfc1), ('wfc2', wfc2)):
            dump_wmat(W, os.path.join(out_dir, nm + '.hex'))
        for nm, B in (('bq', bq), ('bk', bk), ('bv', bv), ('bp', bp),
                      ('bf1', bf1), ('bf2', bf2)):
            with open(os.path.join(out_dir, nm + '.hex'), 'w') as f:
                for b in bias_beats(B):
                    f.write(hx(b, 16) + '\n')
        with open(os.path.join(out_dir, 'exp.hex'), 'w') as f:
            for nn in range(n):
                for g in range(G):
                    word = 0
                    for a in range(tl):
                        for t in range(T):
                            word |= (int(out[nn, g * tl + a, t]) & mask) << ((a * T + t) * rw)
                    f.write(hx(word, beats_per) + '\n')
    meta = {
        'e': e, 'heads': heads, 'head_dim': hd, 'hidden': hidden, 'T': T,
        'tokens': ntokens, 'tl': tl, 'tc': tc, 'rw': rw, 'seed': seed,
        'thr_q': thr, 'thr_k': thr, 'thr_v': thr, 'thr_proj': thr,
        'thr_fc1': thr, 'thr_fc2': thr,
        'blk_w_beats': len(beats), 'blk_w_layout': 'wq,wk,wv,wp(EEB=E*E/8 each); bq,bk,bv,bp(E/2 each, 2x signed32 lo=even hi=odd); fc1(FCD); bf1(H/2); fc2(FCD); bf2(E/2)',
        'tok_beats': n * G, 'out_beats': n * G,
        'out_order': 'token-major then group g=0..G-1; 128b = TL lanes x T x RW nibbles at (a*T+t)',
        'sums_order': 'channel e=0..E-1, sum over tokens and T of 4b out value',
        'tag': tag,
    }
    with open(os.path.join(out_dir, 'meta_%s.json' % tag), 'w') as f:
        json.dump(meta, f, indent=1)
    print('tag=%s blk_w=%d tok=%d out=%d sums=%d' % (tag, len(beats), n * G, n * G, sums.size))
    return dict(beats=beats, out=out, sums=sums, tok=tok)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--small-selftest', action='store_true')
    ap.add_argument('--e', type=int, default=384)
    ap.add_argument('--heads', type=int, default=12)
    ap.add_argument('--hd', type=int, default=32)
    ap.add_argument('--hidden', type=int, default=1536)
    ap.add_argument('--t', type=int, default=4)
    ap.add_argument('--tokens', type=int, default=64)
    ap.add_argument('--tl', type=int, default=8)
    ap.add_argument('--tc', type=int, default=8)
    ap.add_argument('--rw', type=int, default=4)
    ap.add_argument('--seed', type=int, default=52)
    ap.add_argument('--thr', type=int, default=60)
    ap.add_argument('--tok-file', default=None)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--tag', default='img1')
    a = ap.parse_args()
    if a.small_selftest:
        import filecmp
        ref = 'D:/dl_snn/build/spikformer/block_replay/'
        out = 'D:/dl_snn/sim_work/vis_sps/gen_block_selftest/'
        # (1) RNG/weight draw order + stream packing: tokens drawn like the ref
        gen(32, 1, 32, 64, 2, 4, 8, 8, 4, 52, None, out, 'img1', dump_legacy=True)
        ok = True
        names = ['wq.hex', 'wk.hex', 'wv.hex', 'wp.hex', 'bq.hex', 'bk.hex',
                 'bv.hex', 'bp.hex', 'wfc1.hex', 'bf1.hex', 'wfc2.hex', 'bf2.hex', 'tok.hex']
        for nm in names:
            if filecmp.cmp(os.path.join(ref, nm), os.path.join(out, nm), shallow=False):
                print('SELFTEST OK  ', nm)
            else:
                print('SELFTEST DIFF', nm); ok = False
        # (2) equations: same tokens as the reference exp.hex, compare out
        #     (parse ref tok.hex: 16 beats, 128b = 8 lanes x T2 x RW4 nibbles)
        tokr = np.zeros((4, 32, 2), dtype=np.int64)
        lines = [l.strip() for l in open(ref + 'tok.hex') if l.strip()]
        for idx, line in enumerate(lines):
            nn, g = idx // 4, idx % 4
            w = int(line, 16)
            for aa in range(8):
                for tt in range(2):
                    tokr[nn][g * 8 + aa][tt] = (w >> ((aa * 2 + tt) * 4)) & 0xF
        out2 = out + 'ref_tok/'
        gen(32, 1, 32, 64, 2, 4, 8, 8, 4, 52, tokr, out2, 'img1', dump_legacy=True)
        if filecmp.cmp(ref + 'exp.hex', os.path.join(out2, 'exp.hex'), shallow=False):
            print('SELFTEST OK   exp.hex (equations, ref tokens)')
        else:
            print('SELFTEST DIFF exp.hex (equations, ref tokens)'); ok = False
        # (3) blk_w.hex stream = concatenation of the per-matrix ref files
        refbeats = []
        for nm in ('wq.hex', 'wk.hex', 'wv.hex', 'wp.hex', 'bq.hex', 'bk.hex',
                   'bv.hex', 'bp.hex', 'wfc1.hex', 'bf1.hex', 'wfc2.hex', 'bf2.hex'):
            refbeats += [l.strip() for l in open(ref + nm) if l.strip()]
        mybeats = [l.strip() for l in open(os.path.join(out, 'blk_w.hex')) if l.strip()]
        if refbeats == mybeats:
            print('SELFTEST OK   blk_w.hex (= wq..bf2 concatenation, %d beats)' % len(mybeats))
        else:
            print('SELFTEST DIFF blk_w.hex (%d vs %d beats)' % (len(mybeats), len(refbeats))); ok = False
        print('SELFTEST', 'PASS' if ok else 'FAIL')
        sys.exit(0 if ok else 1)
    tok = None
    if a.tok_file:
        tok = parse_sps_tokens(a.tok_file, a.tokens, a.e, a.t)
    assert a.out_dir
    gen(a.e, a.heads, a.hd, a.hidden, a.t, a.tokens, a.tl, a.tc, a.rw, a.seed, tok, a.out_dir, a.tag, a.thr)
