"""Golden: embed replay (V=64,N=8,E=16,T=2,LANES=8)."""
import numpy as np
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
V, N, E, T, LANES = 64, 8, 16, 2, 8
G = E // LANES
THR = 60
rng = np.random.RandomState(41)
etab = rng.randint(-127, 128, size=(V, E)).astype(np.int32)
ptab = rng.randint(-64, 65, size=(N, E)).astype(np.int32)
toks = rng.randint(0, V, size=(N,)).astype(np.int32)
toks[0] = 0
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
beats = []
for pos in range(N):
    for g in range(G):
        words = []
        for a in range(LANES):
            e = int(etab[int(toks[pos]), g * LANES + a]) + int(ptab[pos, g * LANES + a])
            words.append(lif_chain([e] * T, THR))
        beats.append(words)
import os
out = str(ROOT / 'build/spikformer/embed_replay') + '/'
os.makedirs(out, exist_ok=True)
def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))
with open(out + 'etab.hex', 'w') as f:
    flat = [int(etab[r, c]) for r in range(V) for c in range(E)]
    for i in range(0, len(flat), 8):
        f.write(hx(sum([(flat[i + k] & 0xFF) << (8 * k) for k in range(8)]), 16) + chr(10))
with open(out + 'ptab.hex', 'w') as f:
    flat = [int(ptab[r, c]) for r in range(N) for c in range(E)]
    for i in range(0, len(flat), 8):
        f.write(hx(sum([(flat[i + k] & 0xFF) << (8 * k) for k in range(8)]), 16) + chr(10))
with open(out + 'toks.hex', 'w') as f:
    for t in toks:
        f.write(hx(int(t), 4) + chr(10))
with open(out + 'exp.hex', 'w') as f:
    for words in beats:
        f.write(''.join(hx((w[0] | (w[1] << 1)) if T == 2 else w[0], 1) for w in words) + chr(10))
print('etab=%d ptab=%d toks=%d beats=%d' % (V * E // 8, N * E // 8, N, len(beats)))
