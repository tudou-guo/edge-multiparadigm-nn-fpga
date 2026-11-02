"""Spike/low-bit/weight packing. Layout matches RTL time-minor convention:
lane m owns bits [m*T +: T] inside each HEAD_DIM*T word.
Residual words per token-channel: T*RES_BITS bits; 64-bit word counts are
derived, never hard-coded (upstream 3072 assumed 64b words x 2b x T=4)."""
from __future__ import annotations
import math
def residual_words(n, c, t, res_bits):
    return math.ceil(n * c * t * res_bits / 64)
def attn_word_bits(t, sum_w):
    return t * sum_w
def pack_spikes_bitplane(lanes, t):
    word = 0
    for m, bits in enumerate(lanes):
        for tt in range(t):
            if (bits >> tt) & 1:
                word |= 1 << (m * t + tt)
    return word
def unpack_spikes_bitplane(word, head_dim, t):
    return [((word >> (m * t)) & ((1 << t) - 1)) for m in range(head_dim)]
