"""Parametric SPS golden (same equations as gen_sps_replay.py, vectorized fast path).
Keeps: 3x3 pad=1, signed INT8 weights/pixels, 21-bit bias pack, T-step LIF
arithmetic-shift/hard-reset, 2x2 stride-2 OR pools, token-major beats.
Adds: --out-dir (never overwrites legacy), unique-weight layout (no w_tiles
replication), meta with unique bytes/beats/reread counts.
Usage:
  python gen_sps_param.py --h 8 --w 8 --c1 16 --c2 32 --t 2 --out-dir <dir> [--seed 11]
  python gen_sps_param.py --h 32 --w 32 --c1 96 --c2 384 --t 4 --out-dir <dir>
"""
import argparse, json, os
import numpy as np

def lif_chain_vec(deltas, thr):
    # deltas: (..., T) int32, same per-taps replicated over T in golden
    # golden lif_chain feeds [acc]*T sequentially with pre=0, mem=(pre+d)//2 floor, spike iff mem>=0 and mem>=thr
    shape = deltas.shape
    T = shape[-1]
    pre = np.zeros(shape[:-1], dtype=np.int32)
    out = np.zeros(shape, dtype=np.int32)
    for t in range(T):
        d = deltas[..., t].astype(np.int32)
        s = pre + d
        # floor div by 2 matching python //2 for negatives: use floor_divide
        mem = np.floor_divide(s, 2)
        spike = ((mem >= 0) & (mem >= thr)).astype(np.int32)
        out[..., t] = spike
        pre = np.where(spike.astype(bool), 0, mem)
    return out

def gen(h, w, c0, c1, c2, T, lanes, tc2, thr1, thr2, seed, out):
    assert c1 % lanes == 0 and c2 % lanes == 0 and c1 % tc2 == 0
    g1, g2, nt2 = c1 // lanes, c2 // lanes, c1 // tc2
    HH1, WW1, HH2, WW2 = h//2, w//2, h//4, w//4
    ntok, tbeats = HH2*WW2, (HH2*WW2)*g2
    rng = np.random.RandomState(seed)
    w1 = rng.randint(-64, 65, size=(c1, c0, 9)).astype(np.int32)
    b1 = rng.randint(-200, 201, size=(c1,)).astype(np.int32)
    w2 = rng.randint(-64, 65, size=(c2, c1, 9)).astype(np.int32)
    b2 = rng.randint(-100, 101, size=(c2,)).astype(np.int32)
    pix = rng.randint(0, 128, size=(h, w, c0)).astype(np.int32)
    # ---- conv1 via im2col + matmul (exact same sums, order-independent) ----
    # pad 1
    pad = np.pad(pix, ((1,1),(1,1),(0,0)), mode='constant')
    cols = np.zeros((h*w, c0*9), dtype=np.int32)
    for oy in range(h):
        for ox in range(w):
            patch = pad[oy:oy+3, ox:ox+3, :].reshape(-1)  # row-major tap b, channel-minor? need match: flat order oc,ic,tp with tp row-major
            # patch reshape gives (row, col, ch); need (ic, tap) -> transpose
            patch3 = pad[oy:oy+3, ox:ox+3, :]  # 3,3,c0
            vec = np.zeros((c0*9,), dtype=np.int32)
            for ic in range(c0):
                for tp in range(9):
                    rr, cc = tp // 3, tp % 3
                    vec[ic*9+tp] = patch3[rr, cc, ic]
            cols[oy*w+ox, :] = vec
    w1m = np.zeros((c1, c0*9), dtype=np.int32)
    for oc in range(c1):
        for ic in range(c0):
            for tp in range(9):
                w1m[oc, ic*9+tp] = w1[oc, ic, tp]
    acc1 = cols.astype(np.int32) @ w1m.T.astype(np.int32)  # (h*w, c1)
    acc1 = acc1 + b1[None, :]
    deltas1 = np.repeat(acc1[:, :, None], T, axis=2)  # (P, c1, T)
    spk1 = lif_chain_vec(deltas1, thr1)  # (P,c1,T)
    f1 = spk1.reshape(h, w, c1, T)
    # pool1 OR -> pooled (h1,w1,c1,T)
    pooled = np.zeros((HH1, WW1, c1, T), dtype=np.int32)
    for py in range(HH1):
        for px in range(WW1):
            pooled[py, px] = f1[2*py, 2*px] | f1[2*py+1, 2*px] | f1[2*py, 2*px+1] | f1[2*py+1, 2*px+1]
    # ---- conv2 via im2col on pooled (binary) ----
    pad2 = np.pad(pooled, ((1,1),(1,1),(0,0),(0,0)), mode='constant')  # (HH1+2,WW1+2,c1,T)
    # build cols2: (h1*w1, c1*9) per t -> stack T
    cols2 = np.zeros((HH1*WW1, c1*9, T), dtype=np.int32)
    for pr in range(HH1):
        for pc in range(WW1):
            patch = pad2[pr:pr+3, pc:pc+3, :, :]  # 3,3,c1,T
            for t in range(T):
                vec = np.zeros((c1*9,), dtype=np.int32)
                for ch in range(c1):
                    for tp in range(9):
                        rr, cc = tp // 3, tp % 3
                        vec[ch*9+tp] = patch[rr, cc, ch, t]
                cols2[pr*WW1+pc, :, t] = vec
    w2m = np.zeros((c2, c1*9), dtype=np.int32)
    for oc in range(c2):
        for ch in range(c1):
            for tp in range(9):
                w2m[oc, ch*9+tp] = w2[oc, ch, tp]
    prow = np.zeros((HH1, WW1, c2, T), dtype=np.int32)
    for t in range(T):
        a = cols2[:, :, t].astype(np.int32) @ w2m.T.astype(np.int32)  # (P2, c2) binary-gated sums
        a = a + b2[None, :]
        prow[:, :, :, t] = a.reshape(HH1, WW1, c2)
    deltas2 = np.transpose(prow, (0,1,2,3)).reshape(HH1*WW1, c2, T)
    spk2 = lif_chain_vec(deltas2, thr2).reshape(HH1, WW1, c2, T)
    # pool2 OR -> tokens
    exp_beats = []
    for pos in range(HH2*WW2):
        tpc, tpr = pos % WW2, pos // WW2
        for g in range(g2):
            words = []
            for a in range(lanes):
                oc = g*lanes+a
                bits = 0
                for t in range(T):
                    v = int(spk2[2*tpr, 2*tpc, oc, t]) | int(spk2[2*tpr+1, 2*tpc, oc, t]) | int(spk2[2*tpr, 2*tpc+1, oc, t]) | int(spk2[2*tpr+1, 2*tpc+1, oc, t])
                    bits |= (v << t)
                words.append(bits)
            exp_beats.append(words)
    def hx(v, wd):
        return '%0*x' % (wd, int(v) & ((1 << (wd*4))-1))
    os.makedirs(out, exist_ok=True)
    # unique streams
    flat1 = [int(w1[oc, ic, tp]) for oc in range(c1) for ic in range(c0) for tp in range(9)]
    with open(os.path.join(out, 'w_ld1.hex'), 'w') as f:
        for i in range(0, len(flat1), 8):
            word = 0
            for k in range(8):
                word |= ((flat1[i+k] & 0xFF) << (8*k))
            f.write(hx(word, 16)+'\n')
    n = max(c1, c2)
    with open(os.path.join(out, 'b_ld.hex'), 'w') as f:
        for i in range(n):
            lo = int(b1[i]) if i < c1 else 0
            hi = int(b2[i]) if i < c2 else 0
            f.write(hx((lo & 0x1FFFFF) | ((hi & 0x1FFFFF) << 21), 16)+'\n')
    flat2u = [int(w2[oc, ch, tp]) for oc in range(c2) for ch in range(c1) for tp in range(9)]
    with open(os.path.join(out, 'w_uniq.hex'), 'w') as f:
        for i in range(0, len(flat2u), 8):
            word = 0
            for k in range(8):
                word |= ((flat2u[i+k] & 0xFF) << (8*k))
            f.write(hx(word, 16)+'\n')
    # scheduled tile stream: g outer, tile inner, bytes (a,k,b); matches
    # spiking_eyeriss_array tile load order (no spatial replication).
    g2n, nt2 = c2 // lanes, c1 // tc2
    sched = []
    for g in range(g2n):
        for tile in range(nt2):
            for a in range(lanes):
                for k in range(tc2):
                    for b in range(9):
                        sched.append(int(w2[g*lanes+a, tile*tc2+k, b]))
    with open(os.path.join(out, 'w_sched.hex'), 'w') as f:
        for i in range(0, len(sched), 8):
            word = 0
            for k in range(8):
                word |= ((sched[i+k] & 0xFF) << (8*k))
            f.write(hx(word, 16)+'\n')
    assert len(sched) == c2*c1*9
    with open(os.path.join(out, 'pix.hex'), 'w') as f:
        for yy in range(h):
            for xx in range(w):
                word = 0
                for c in range(c0):
                    word |= ((int(pix[yy, xx, c]) & 0xFF) << (8*c))
                f.write(hx(word, 6)+'\n')
    with open(os.path.join(out, 'exp.hex'), 'w') as f:
        for words in exp_beats:
            f.write(''.join(hx(wd, 1) for wd in words)+'\n')
    meta = {
        'h': h, 'w': w, 'c0': c0, 'c1': c1, 'c2': c2, 'T': T, 'lanes': lanes, 'tc2': tc2,
        'thr1': thr1, 'thr2': thr2, 'seed': seed,
        'conv1_unique_bytes': c1*c0*9, 'conv2_unique_bytes': c2*c1*9,
        'bias_unique_bytes': (c1+c2)*4, 'bias_beats_21pair': n,
        'conv1_beats_64b': (c1*c0*9+7)//8, 'conv2_beats_64b': (c2*c1*9+7)//8,
        'pix_beats': h*w, 'exp_beats': len(exp_beats),
        'legacy_w_tiles_replicated_bytes_note': 'legacy w_tiles.hex replicates unique conv2 per (pr,pc,g,tile); native uses w_uniq.hex single copy',
    }
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('beats: ld1=%d bias=%d uniq_conv2=%d pix=%d exp=%d' % ((len(flat1)+7)//8, n, (len(flat2u)+7)//8, h*w, len(exp_beats)))
    print(json.dumps(meta, indent=1))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, default=8); ap.add_argument('--w', type=int, default=8)
    ap.add_argument('--c0', type=int, default=3); ap.add_argument('--c1', type=int, default=16)
    ap.add_argument('--c2', type=int, default=32); ap.add_argument('--t', type=int, default=2)
    ap.add_argument('--lanes', type=int, default=8); ap.add_argument('--tc2', type=int, default=4)
    ap.add_argument('--thr1', type=int, default=200); ap.add_argument('--thr2', type=int, default=60)
    ap.add_argument('--seed', type=int, default=11); ap.add_argument('--out-dir', required=True)
    a = ap.parse_args()
    gen(a.h, a.w, a.c0, a.c1, a.c2, a.t, a.lanes, a.tc2, a.thr1, a.thr2, a.seed, a.out_dir)