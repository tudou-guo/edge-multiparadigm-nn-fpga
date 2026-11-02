"""Vision end-to-end golden: SPS image tokens -> block -> per-channel sums.

Sum definition (PL head): for each of the E channels, add the block output over
all tokens and all T timesteps. PS then divides by T*N and does FP32 Linear.
"""
import numpy as np, io
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
out = np.load(str(ROOT / 'build/spikformer/block_replay/out_golden.npy'))   # [NTOK][E][T]
sums = out.sum(axis=(0, 2)).astype(np.uint32)
p = str(ROOT / 'build/spikformer/vis_replay') + '/'
import pathlib; pathlib.Path(p).mkdir(parents=True, exist_ok=True)
with io.open(p + 'exp_sums.hex', 'w') as f:
    for v in sums:
        f.write('%08x\n' % (int(v) & 0xFFFFFFFF))
print('sums[:8] =', sums[:8].tolist(), '| range', int(sums.min()), int(sums.max()))
print('wrote exp_sums.hex  (%d channels)' % sums.size)
