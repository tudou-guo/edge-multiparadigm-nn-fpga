"""Golden: small MLP replay (C=32,HIDDEN=64,T=2,TL=8,TC=8,RW=4).
fc1: psum = b1 + sum_tiles in*W; LIF per lane; fc2: psum = b2 + sum_tiles gated;
out = (in + psum) mod 2^RW per slice (matches RTL truncation)."""
import numpy as np
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
C, HIDDEN, T, TL, TC, RW = 32, 64, 2, 8, 8, 4
G1, NT, G0, NT2 = HIDDEN // TL, C // TC, C // TL, HIDDEN // TC
THRH = 60
MASK = (1 << RW) - 1
rng = np.random.RandomState(23)
inp = rng.randint(0, 1 << RW, size=(C, T)).astype(np.int32)
w1 = rng.randint(-64, 65, size=(HIDDEN, C)).astype(np.int32)
b1 = rng.randint(-200, 201, size=(HIDDEN,)).astype(np.int32)
w2 = rng.randint(-64, 65, size=(C, HIDDEN)).astype(np.int32)
b2 = rng.randint(-200, 201, size=(C,)).astype(np.int32)
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
hbuf = np.zeros((HIDDEN, T), dtype=np.int32)
for g in range(G1):
    ps = {(a, t): int(b1[g * TL + a]) for a in range(TL) for t in range(T)}
    for t1 in range(NT):
        for a in range(TL):
            for t in range(T):
                s = ps[(a, t)]
                for k in range(TC):
                    s += int(inp[t1 * TC + k, t]) * int(w1[g * TL + a, t1 * TC + k])
                ps[(a, t)] = s
    for a in range(TL):
        sp = lif_chain([ps[(a, t)] for t in range(T)], THRH)
        for t in range(T):
            hbuf[g * TL + a, t] = sp[t]
exp = []
for g in range(G0):
    beat = []
    ps = {(a, t): int(b2[g * TL + a]) for a in range(TL) for t in range(T)}
    for t2 in range(NT2):
        for a in range(TL):
            for t in range(T):
                s = ps[(a, t)]
                for k in range(TC):
                    if hbuf[t2 * TC + k, t]:
                        s += int(w2[g * TL + a, t2 * TC + k])
                ps[(a, t)] = s
    for a in range(TL):
        for t in range(T):
            beat.append((int(inp[g * TL + a, t]) + ps[(a, t)]) & MASK)
    exp.append(beat)
import os
out = str(ROOT / 'build/spikformer/mlp_replay') + '/'
os.makedirs(out, exist_ok=True)
def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))
with open(out + 'inp.hex', 'w') as f:
    for t1 in range(NT):
        word = 0
        for k in range(TC):
            for t in range(T):
                word |= ((int(inp[t1 * TC + k, t]) & MASK) << ((k * T + t) * RW))
        f.write(hx(word, TC * T * RW // 4) + chr(10))
tiles1, tiles2 = [], []
for g in range(G1):
    for t1 in range(NT):
        flat = []
        for a in range(TL):
            for k in range(TC):
                flat.append(int(w1[g * TL + a, t1 * TC + k]))
        tiles1.append((flat, [int(b1[g * TL + a]) for a in range(TL)]))
for g in range(G0):
    for t2 in range(NT2):
        flat = []
        for a in range(TL):
            for k in range(TC):
                flat.append(int(w2[g * TL + a, t2 * TC + k]))
        tiles2.append((flat, [int(b2[g * TL + a]) for a in range(TL)]))
with open(out + 'wt1.hex', 'w') as f:
    for flat, bl in tiles1:
        for i in range(0, len(flat), 8):
            word = 0
            for k in range(8):
                word |= ((flat[i + k] & 0xFF) << (8 * k))
            f.write(hx(word, 16) + chr(10))
        for a in range(0, TL, 2):
            f.write(hx((bl[a] & 0xFFFFFFFF) | ((bl[a + 1] & 0xFFFFFFFF) << 32), 16) + chr(10))
with open(out + 'wt2.hex', 'w') as f:
    for flat, bl in tiles2:
        for i in range(0, len(flat), 8):
            word = 0
            for k in range(8):
                word |= ((flat[i + k] & 0xFF) << (8 * k))
            f.write(hx(word, 16) + chr(10))
        for a in range(0, TL, 2):
            f.write(hx((bl[a] & 0xFFFFFFFF) | ((bl[a + 1] & 0xFFFFFFFF) << 32), 16) + chr(10))
with open(out + 'exp.hex', 'w') as f:
    for beat in exp:
        f.write(''.join(hx(v, 1) for v in beat) + chr(10))
with open(out + 'psum.hex', 'w') as f:
    for g in range(G0):
        ps = {(a, t): int(b2[g * TL + a]) for a in range(TL) for t in range(T)}
        for t2 in range(NT2):
            for a in range(TL):
                for t in range(T):
                    s = ps[(a, t)]
                    for k in range(TC):
                        if hbuf[t2 * TC + k, t]:
                            s += int(w2[g * TL + a, t2 * TC + k])
                    ps[(a, t)] = s
        for a in range(TL):
            for t in range(T):
                f.write(hx(ps[(a, t)] & 0xFFFFFFFF, 8) + chr(10))
print('tiles1=%d tiles2=%d exp=%d' % (len(tiles1), len(tiles2), len(exp)))
