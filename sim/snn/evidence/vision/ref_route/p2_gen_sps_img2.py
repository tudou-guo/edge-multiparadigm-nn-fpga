
"""Second image for native SPS: same weights as vectors/sps_native_tc2_8
(seed=7, first draws), new pixels from a separate RNG seed. Emits pix.hex +
exp.hex in the same formats as gen_sps_param.py. Does NOT touch weights."""
import argparse, json, os
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

def gen(h, w, c0, c1, c2, T, lanes, thr1, thr2, seed, pixseed, out):
    rng = np.random.RandomState(seed)
    w1 = rng.randint(-64, 65, size=(c1, c0, 9)).astype(np.int64)
    b1 = rng.randint(-200, 201, size=(c1,)).astype(np.int64)
    w2 = rng.randint(-64, 65, size=(c2, c1, 9)).astype(np.int64)
    b2 = rng.randint(-100, 101, size=(c2,)).astype(np.int64)
    prng = np.random.RandomState(pixseed)
    pix = prng.randint(0, 128, size=(h, w, c0)).astype(np.int64)
    HH1, WW1, HH2, WW2 = h // 2, w // 2, h // 4, w // 4
    pad = np.pad(pix, ((1, 1), (1, 1), (0, 0)), mode='constant')
    cols = np.zeros((h * w, c0 * 9), dtype=np.int64)
    for oy in range(h):
        for ox in range(w):
            p3 = pad[oy:oy + 3, ox:ox + 3, :]
            vec = np.zeros((c0 * 9,), dtype=np.int64)
            for ic in range(c0):
                for tp in range(9):
                    rr, cc = tp // 3, tp % 3
                    vec[ic * 9 + tp] = p3[rr, cc, ic]
            cols[oy * w + ox, :] = vec
    w1m = np.zeros((c1, c0 * 9), dtype=np.int64)
    for oc in range(c1):
        for ic in range(c0):
            for tp in range(9):
                w1m[oc, ic * 9 + tp] = w1[oc, ic, tp]
    acc1 = cols @ w1m.T + b1[None, :]
    spk1 = lif_chain_vec(np.repeat(acc1[:, :, None], T, axis=2), thr1).reshape(h, w, c1, T)
    pooled = np.zeros((HH1, WW1, c1, T), dtype=np.int64)
    for py in range(HH1):
        for px in range(WW1):
            pooled[py, px] = (spk1[2 * py, 2 * px] | spk1[2 * py + 1, 2 * px]
                              | spk1[2 * py, 2 * px + 1] | spk1[2 * py + 1, 2 * px + 1])
    pad2 = np.pad(pooled, ((1, 1), (1, 1), (0, 0), (0, 0)), mode='constant')
    cols2 = np.zeros((HH1 * WW1, c1 * 9, T), dtype=np.int64)
    for pr in range(HH1):
        for px in range(WW1):
            patch = pad2[pr:pr + 3, px:px + 3, :, :]
            for t in range(T):
                vec = np.zeros((c1 * 9,), dtype=np.int64)
                for ch in range(c1):
                    for tp in range(9):
                        rr, cc = tp // 3, tp % 3
                        vec[ch * 9 + tp] = patch[rr, cc, ch, t]
                cols2[pr * WW1 + px, :, t] = vec
    w2m = np.zeros((c2, c1 * 9), dtype=np.int64)
    for oc in range(c2):
        for ch in range(c1):
            for tp in range(9):
                w2m[oc, ch * 9 + tp] = w2[oc, ch, tp]
    prow = np.zeros((HH1, WW1, c2, T), dtype=np.int64)
    for t in range(T):
        a = cols2[:, :, t] @ w2m.T + b2[None, :]
        prow[:, :, :, t] = a.reshape(HH1, WW1, c2)
    spk2 = lif_chain_vec(prow.reshape(HH1 * WW1, c2, T), thr2).reshape(HH1, WW1, c2, T)
    os.makedirs(out, exist_ok=True)
    def hx(v, wd):
        return '%0*x' % (wd, int(v) & ((1 << (wd * 4)) - 1))
    with open(os.path.join(out, 'pix.hex'), 'w') as f:
        for yy in range(h):
            for xx in range(w):
                word = 0
                for c in range(c0):
                    word |= (int(pix[yy, xx, c]) & 0xFF) << (8 * c)
                f.write(hx(word, 6) + '\n')
    g2 = c2 // lanes
    with open(os.path.join(out, 'exp.hex'), 'w') as f:
        for pos in range(HH2 * WW2):
            tpc, tpr = pos % WW2, pos // WW2
            for g in range(g2):
                words = []
                for a in range(lanes):
                    oc = g * lanes + a
                    bits = 0
                    for t in range(T):
                        v = (int(spk2[2 * tpr, 2 * tpc, oc, t]) | int(spk2[2 * tpr + 1, 2 * tpc, oc, t])
                             | int(spk2[2 * tpr, 2 * tpc + 1, oc, t]) | int(spk2[2 * tpr + 1, 2 * tpc + 1, oc, t]))
                        bits |= (v << t)
                    words.append(bits)
                f.write(''.join(hx(wd, 1) for wd in words) + '\n')
    meta = {'h': h, 'w': w, 'c0': c0, 'c1': c1, 'c2': c2, 'T': T, 'lanes': lanes,
            'thr1': thr1, 'thr2': thr2, 'sps_weight_seed': seed, 'pix_seed': pixseed,
            'note': 'second image: same SPS weights as sps_native_tc2_8, different pixels',
            'pix_beats': h * w, 'exp_beats': HH2 * WW2 * g2}
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('img2: pix=%d exp=%d (pixseed=%d)' % (h * w, HH2 * WW2 * g2, pixseed))
    return spk1, spk2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--pixseed', type=int, default=99)
    a = ap.parse_args()
    gen(32, 32, 3, 96, 384, 4, 8, 200, 60, 7, a.pixseed, a.out_dir)
