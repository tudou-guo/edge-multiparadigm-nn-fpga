"""Small-integer reference: numeric truth for short tests.
Mirrors LIF (tau=2 hard reset), QK popcount, residual adds, head sum.
Units are integer contract units (see numeric doc)."""
from __future__ import annotations
def arith_shr1(x):
    return x // 2 if x >= 0 else -((-x + 1) // 2)
def lif_step(pre, delta, thr):
    mem = arith_shr1(pre + delta)
    if mem >= 0 and mem >= thr:
        return 0, 1
    return mem, 0
def lif_chain(deltas, thr):
    pre, out = 0, []
    for d in deltas:
        pre, s = lif_step(pre, d, thr)
        out.append(s)
    return out, pre
def popcount32(x):
    assert 0 <= x < (1 << 32)
    return bin(x).count('1')
def qk_and_popcount(q, k):
    return [popcount32((qq & kk) & 0xFFFFFFFF) for qq, kk in zip(q, k)]
def residual_add(a, b, vmax=None):
    c = [x + y for x, y in zip(a, b)]
    if vmax is not None:
        assert all(0 <= v <= vmax for v in c), 'residual bound violated'
    return c
def head_sum(token_sums):
    return sum(token_sums)
def conv3x3_tap(window, weights, bias=0):
    # window/weights: 9 x Cin ints; returns int psum (stage-1 pixel x int8)
    s = bias
    for p, w in zip(window, weights):
        s += p * w
    return s
def gated_tap(spike_bits, weights):
    # spike_bits/weights: 9 x Cin bits/ints; returns int (stage-2 spike-gated)
    s = 0
    for b, w in zip(spike_bits, weights):
        if b:
            s += w
    return s
def pool2x2_or(bits):
    # 4 spike bits (or T-bit lanes OR-ed per t by caller); binary max == OR
    o = 0
    for b in bits:
        o = o | b
    return o
