"""Golden: small head replay (CH=32,TOKENS=8,T=2,HL=8)."""
import numpy as np
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
CH, TOKENS, T, HL = 32, 8, 2, 8
G = CH // HL
rng = np.random.RandomState(31)
beats = []
for n in range(TOKENS * G):
    lanes = []
    for _ in range(HL):
        lanes.append(int(rng.randint(0, 1 << T)))
    beats.append(lanes)
sums = [0] * CH
for n in range(TOKENS * G):
    g = n % G
    for l in range(HL):
        sums[g * HL + l] += bin(beats[n][l]).count('1')
import os
out = str(ROOT / 'build/spikformer/head_replay') + '/'
os.makedirs(out, exist_ok=True)
def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))
with open(out + 'beats.hex', 'w') as f:
    for lanes in beats:
        word = 0
        for l, v in enumerate(lanes):
            word |= ((v & ((1 << T) - 1)) << (l * T))
        f.write(hx(word, (HL * T + 3) // 4) + chr(10))
with open(out + 'exp.hex', 'w') as f:
    for s in sums:
        f.write(hx(s, 8) + chr(10))
print('beats=%d sums=%d max=%d' % (len(beats), len(sums), max(sums)))
