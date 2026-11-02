"""Golden generator: small SPS replay (H=8,C0=3,C1=16,C2=32,T=2,LANES=8,TC2=4).
Tap convention (canonical row-major): tap b <-> window (row b//3, col b%3).
RTL >>> arithmetic shift == python floor div; pixels signed int8 (0..127 used).
Writes hex streams (weights/pixels) + expected token beats for the TB."""
import numpy as np
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
H = W = 8
C0, C1, C2, T = 3, 16, 32, 2
LANES, TC2 = 8, 4
G1, G2, NT2 = C1 // LANES, C2 // LANES, C1 // TC2
H1 = W1 = 4
H2 = W2 = 2
NTOK = 4
TBEATS = NTOK * G2
THR1, THR2 = 200, 60
rng = np.random.RandomState(11)
def s8(x):
    x = np.asarray(x, dtype=np.int32)
    return np.where(x >= 128, x - 256, x)
w1 = rng.randint(-64, 65, size=(C1, C0, 9)).astype(np.int32)
b1 = rng.randint(-200, 201, size=(C1,)).astype(np.int32)
w2 = rng.randint(-64, 65, size=(C2, C1, 9)).astype(np.int32)
b2 = rng.randint(-100, 101, size=(C2,)).astype(np.int32)
pix = rng.randint(0, 128, size=(H, W, C0)).astype(np.int32)
def lif_chain(deltas, thr):
    pre, out = 0, []
    for d in deltas:
        mem = (pre + d) // 2 if (pre + d) >= 0 else -((-(pre + d) + 1) // 2)
        if mem >= 0 and mem >= thr:
            pre, s = 0, 1
        else:
            pre, s = mem, 0
        out.append(s)
    return out
f1 = np.zeros((H, W, C1, T), dtype=np.int32)
for oy in range(H):
    for ox in range(W):
        for g in range(G1):
            for a in range(LANES):
                oc = g * LANES + a
                acc = int(b1[oc])
                for b in range(9):
                    rr, cc = b // 3, b % 3
                    for c in range(C0):
                        yy, xx = oy + rr - 1, ox + cc - 1
                        p = int(pix[yy, xx, c]) if (0 <= yy < H and 0 <= xx < W) else 0
                        acc += int(w1[oc, c, b]) * p
                sp = lif_chain([acc] * T, THR1)
                for t in range(T):
                    f1[oy, ox, oc, t] = sp[t]
pooled = np.zeros((H1, W1, C1, T), dtype=np.int32)
for py in range(H1):
    for px in range(W1):
        for ch in range(C1):
            for t in range(T):
                v = 0
                for fy in range(2):
                    for fx in range(2):
                        v |= int(f1[2 * py + fy, 2 * px + fx, ch, t])
                pooled[py, px, ch, t] = v
prow = np.zeros((H1, W1, C2, T), dtype=np.int32)
for pr in range(H1):
    for pc in range(W1):
        for g in range(G2):
            ps = {(a, t): int(b2[g * LANES + a]) for a in range(LANES) for t in range(T)}
            for tile in range(NT2):
                for a in range(LANES):
                    for t in range(T):
                        s = ps[(a, t)]
                        oc = g * LANES + a
                        for b in range(9):
                            rr, cc = b // 3, b % 3
                            wy, wx = pr + rr - 1, pc + cc - 1
                            if 0 <= wy < H1 and 0 <= wx < W1:
                                for k in range(TC2):
                                    ch = tile * TC2 + k
                                    bit = 0
                                    for fy in range(2):
                                        for fx in range(2):
                                            bit |= int(f1[2 * wy + fy, 2 * wx + fx, ch, t])
                                    if bit:
                                        s += int(w2[oc, ch, b])
                        ps[(a, t)] = s
            for a in range(LANES):
                oc = g * LANES + a
                sp = lif_chain([ps[(a, t)] for t in range(T)], THR2)
                for t in range(T):
                    prow[pr, pc, oc, t] = sp[t]
exp_beats = []
for tb in range(TBEATS):
    g, pos = tb % G2, tb // G2
    tpc, tpr = pos % W2, pos // W2
    words = []
    for a in range(LANES):
        oc = g * LANES + a
        bits = 0
        for t in range(T):
            v = 0
            for fy in range(2):
                for fx in range(2):
                    v |= int(prow[2 * tpr + fy, 2 * tpc + fx, oc, t])
            bits |= (v << t)
        words.append(bits)
    exp_beats.append(words)
def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))
out = str(ROOT / 'build/spikformer/sps_replay') + '/'
import os
os.makedirs(out, exist_ok=True)
with open(out + 'w_ld1.hex', 'w') as f:
    flat = []
    for oc in range(C1):
        for ic in range(C0):
            for tp in range(9):
                flat.append(int(w1[oc, ic, tp]))
    for i in range(0, len(flat), 8):
        word = 0
        for k in range(8):
            word |= ((flat[i + k] & 0xFF) << (8 * k))
        f.write(hx(word, 16) + chr(10))
with open(out + 'b_ld.hex', 'w') as f:
    n = max(C1, C2)
    for i in range(n):
        lo = int(b1[i]) if i < C1 else 0
        hi = int(b2[i]) if i < C2 else 0
        f.write(hx((lo & 0x1FFFFF) | ((hi & 0x1FFFFF) << 21), 16) + chr(10))
tiles = []
for pr in range(H1):
    for pc in range(W1):
        for g in range(G2):
            for tile in range(NT2):
                flat = []
                for a in range(LANES):
                    for k in range(TC2):
                        for b in range(9):
                            flat.append(int(w2[g * LANES + a, tile * TC2 + k, b]))
                tiles.append(flat)
with open(out + 'w_tiles.hex', 'w') as f:
    for flat in tiles:
        for i in range(0, len(flat), 8):
            word = 0
            for k in range(8):
                word |= ((flat[i + k] & 0xFF) << (8 * k))
            f.write(hx(word, 16) + chr(10))
with open(out + 'pix.hex', 'w') as f:
    for yy in range(H):
        for xx in range(W):
            word = 0
            for c in range(C0):
                word |= ((int(pix[yy, xx, c]) & 0xFF) << (8 * c))
            f.write(hx(word, 6) + chr(10))
with open(out + 'exp.hex', 'w') as f:
    for words in exp_beats:
        f.write(''.join(hx(w, 1) for w in words) + chr(10))
print('beats: ld1=%d bias=%d tiles=%d pix=64 exp=%d' % (54, 32, len(tiles) * 36, len(exp_beats)))
