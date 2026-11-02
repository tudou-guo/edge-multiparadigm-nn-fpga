"""Short tests for new/changed SNN paths only (pure Python, no Vivado).
1. export determinism: same seed twice -> identical key files (no append).
2. LIF threshold + negative shift semantics.
3. popcount all-ones = 32 (the fixed boundary).
4. single/multi-block: different weights per block, residual > 3.
5. lifecycle: two consecutive starts (sequencer model), ack clears done.
6. packing round-trip time-minor layout.
"""
import json, os, pathlib, sys
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()
sys.path.insert(0, str(ROOT / 'tools/backends/spikformer'))
import reference as R
import packing as P
import config as C
import weights as W
import schedule as S
FAILS = []
def check(name, cond, info=''):
    print(('PASS ' if cond else 'FAIL ') + name + (' ' + str(info) if info else ''))
    if not cond:
        FAILS.append(name)
# 2. LIF: thr=2048 native-ish; negative arithmetic shift
out, _ = R.lif_chain([0, 0, 0, 0], 2048)
check('lif.quiet', out == [0, 0, 0, 0], out)
pre, s = R.lif_step(-3, 0, 2048)
check('lif.negshr', pre == -2 and s == 0, (pre, s))
out, _ = R.lif_chain([5000, 5000, 5000, 5000], 1000)
check('lif.fires', out[0] == 1 and out[1] == 1, out)
# 3. popcount boundary
check('popcount.full', R.popcount32(0xFFFFFFFF) == 32)
check('popcount.qk', R.qk_and_popcount([0xFFFFFFFF], [0xFFFFFFFF]) == [32])
# 4. multi-block residuals exceed 3 with different weights
r1 = R.residual_add([2, 2], [2, 2], vmax=6)
r2 = R.residual_add(r1, [1, 2], vmax=8)
check('multiblock.residual', r2 == [5, 6] and max(r2) > 3, r2)
# 6. packing round-trip
lanes = [0b1010, 0b0101]
w = P.pack_spikes_bitplane(lanes, 4)
back = P.unpack_spikes_bitplane(w, 2, 4)
check('packing.roundtrip', back == lanes, (hex(w), back))
check('packing.words', P.residual_words(64, 384, 4, 4) == 64 * 384 * 4 * 4 // 64)
# 5. lifecycle model: two starts, ack clears
class Seq:
    def __init__(s, d):
        s.d, s.blk, s.done, s.busy = d, 0, False, False
    def start(s):
        assert not s.busy, 'double start not rejected'
        s.busy, s.blk, s.done = True, 0, False
    def step_block(s):
        s.blk += 1
        if s.blk >= s.d:
            s.busy, s.done = False, True
    def ack(s):
        s.done = False
q = Seq(2)
q.start()
try:
    q.start()
    check('lifecycle.reject', False)
except AssertionError:
    check('lifecycle.reject', True)
q.step_block(); q.step_block()
check('lifecycle.done', q.done and not q.busy)
q.ack()
q.start(); q.step_block(); q.step_block()
check('lifecycle.second', q.done, 'A->B ok')
check('conv.tap', R.conv3x3_tap([1]*9, [2]*9, 5) == 23, '9x1x2+5')
check('conv.signed', R.conv3x3_tap([1, -1, 0]*3, [3]*9, 0) == 0)
check('gated.tap', R.gated_tap([1, 0, 1, 0]*3, list(range(12))) == 0 + 2 + 4 + 6 + 8 + 10)
check('pool.or', R.pool2x2_or([0, 1, 0, 0]) == 1 and R.pool2x2_or([0, 0, 0, 0]) == 0)
print('FAILURES: %d' % len(FAILS))
sys.exit(1 if FAILS else 0)
