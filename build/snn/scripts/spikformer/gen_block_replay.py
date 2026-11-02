"""Golden: block replay (E=32,H=1,HD=32,HIDDEN=64,T=2,TOKENS=4,TL=8,TC=8,RW=4).
Mirrors the current spiking_block_top contract: QKV+LIF, all-pairs QK
popcount, gated AV, proj+LIF+residual, then fc1+LIF, fc2+LIF+residual.
All arithmetic is modulo 2^RW only at residual adds; psums retain full
precision; LIF uses an arithmetic floor divide by two and hard reset."""
import io
import numpy as np
import os
import pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
E, H, HD, HIDDEN, T, NTOK, TL, TC, RW = 32, 1, 32, 64, 2, 4, 8, 8, 4
G, NT = E // TL, E // TC
NT2 = HIDDEN // TC
MASK = (1 << RW) - 1
THRQ, THRK, THRV = 60, 60, 60
THRPROJ, THRFC1, THRFC2 = 60, 60, 60
rng = np.random.RandomState(52)
wq = rng.randint(-64, 65, size=(E, E)).astype(np.int32)
wk = rng.randint(-64, 65, size=(E, E)).astype(np.int32)
wv = rng.randint(-64, 65, size=(E, E)).astype(np.int32)
wp = rng.randint(-64, 65, size=(E, E)).astype(np.int32)
bq = rng.randint(-200, 201, size=(E,)).astype(np.int32)
bk = rng.randint(-200, 201, size=(E,)).astype(np.int32)
bv = rng.randint(-200, 201, size=(E,)).astype(np.int32)
bp = rng.randint(-200, 201, size=(E,)).astype(np.int32)
wfc1 = rng.randint(-64, 65, size=(HIDDEN, E)).astype(np.int32)
bf1 = rng.randint(-200, 201, size=(HIDDEN,)).astype(np.int32)
wfc2 = rng.randint(-64, 65, size=(E, HIDDEN)).astype(np.int32)
bf2 = rng.randint(-200, 201, size=(E,)).astype(np.int32)
import os
_toksrc = os.environ.get('TOKSRC')
if _toksrc:
    # tokens come from the SPS golden (vision end-to-end): one beat per line,
    # hex chars are 1-bit spikes, beat order is token-major then channel group.
    tok = np.zeros((NTOK, E, T), dtype=np.int32)
    _lines = [l.strip() for l in io.open(_toksrc) if l.strip()]
    _G2 = E // 8
    for _tb, _line in enumerate(_lines):
        _g, _pos = _tb % _G2, _tb // _G2
        for _a in range(8):
            _bits = int(_line[_a], 16)
            for _t in range(T):
                tok[_pos][_g * 8 + _a][_t] = (_bits >> _t) & 1
else:
    tok = rng.randint(0, 16, size=(NTOK, E, T)).astype(np.int32)
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
Q = np.zeros((NTOK, E, T), dtype=np.int32)
K = np.zeros((NTOK, E, T), dtype=np.int32)
V = np.zeros((NTOK, E, T), dtype=np.int32)
for n in range(NTOK):
    for W, B, THR, O in ((wq, bq, THRQ, Q), (wk, bk, THRK, K), (wv, bv, THRV, V)):
        for e in range(E):
            acc = [int(B[e])] * T
            for t in range(T):
                for i in range(E):
                    acc[t] += int(W[e, i]) * int(tok[n, i, t])
            sp = lif_chain(acc, THR)
            for t in range(T):
                O[n, e, t] = sp[t]
scr = np.zeros((NTOK, NTOK, T), dtype=np.int32)
for i in range(NTOK):
    for j in range(NTOK):
        for t in range(T):
            s = 0
            for m in range(HD):
                s += (int(Q[i, m, t]) & 1) & (int(K[j, m, t]) & 1)
            scr[i, j, t] = s
AV = np.zeros((NTOK, E, T), dtype=np.int32)
for i in range(NTOK):
    for e in range(E):
        for t in range(T):
            s = 0
            for j in range(NTOK):
                if V[j, e, t]:
                    s += scr[i, j, t]
            AV[i, e, t] = s
proj = np.zeros((NTOK, E, T), dtype=np.int32)
ares = np.zeros((NTOK, E, T), dtype=np.int32)
for n in range(NTOK):
    for e in range(E):
        acc = [int(bp[e])] * T
        for t in range(T):
            for i in range(E):
                acc[t] += int(wp[e, i]) * int(AV[n, i, t])
        sp = lif_chain(acc, THRPROJ)
        for t in range(T):
            proj[n, e, t] = sp[t]
            ares[n, e, t] = (int(tok[n, e, t]) + sp[t]) & MASK
out = np.zeros((NTOK, E, T), dtype=np.int32)
HID = np.zeros((NTOK, HIDDEN, T), dtype=np.int32)
FC2 = np.zeros((NTOK, E, T), dtype=np.int32)
for n in range(NTOK):
    hid = HID[n]
    for e in range(HIDDEN):
        acc = [int(bf1[e])] * T
        for t in range(T):
            for i in range(E):
                acc[t] += int(wfc1[e, i]) * int(ares[n, i, t])
        sp = lif_chain(acc, THRFC1)
        for t in range(T):
            hid[e, t] = sp[t]
    for e in range(E):
        acc = [int(bf2[e])] * T
        for t in range(T):
            for i in range(HIDDEN):
                if hid[i, t]:
                    acc[t] += int(wfc2[e, i])
        sp = lif_chain(acc, THRFC2)
        for t in range(T):
            FC2[n, e, t] = sp[t]
            out[n, e, t] = (int(ares[n, e, t]) + sp[t]) & MASK
import os
o = str(ROOT / 'build/spikformer/block_replay') + '/'
os.makedirs(o, exist_ok=True)
def hx(v, w):
    return '%0*x' % (w, int(v) & ((1 << (w * 4)) - 1))
def dump_mat(M, r, c, name):
    flat = [int(M[i, j]) for i in range(r) for j in range(c)]
    with open(o + name, 'w') as f:
        for i in range(0, len(flat), 8):
            f.write(hx(sum([(flat[i + k] & 0xFF) << (8 * k) for k in range(8)]), 16) + chr(10))
def dump_bias(B, name):
    with open(o + name, 'w') as f:
        for i in range(0, len(B), 2):
            f.write(hx((int(B[i]) & 0xFFFFFFFF) | ((int(B[i + 1]) & 0xFFFFFFFF) << 32), 16) + chr(10))
dump_mat(wq, E, E, 'wq.hex')
dump_mat(wk, E, E, 'wk.hex')
dump_mat(wv, E, E, 'wv.hex')
dump_mat(wp, E, E, 'wp.hex')
dump_bias(bq, 'bq.hex')
dump_bias(bk, 'bk.hex')
dump_bias(bv, 'bv.hex')
dump_bias(bp, 'bp.hex')
dump_mat(wfc1, HIDDEN, E, 'wfc1.hex')
dump_bias(bf1, 'bf1.hex')
dump_mat(wfc2, E, HIDDEN, 'wfc2.hex')
dump_bias(bf2, 'bf2.hex')
with open(o + 'tok.hex', 'w') as f:
    for n in range(NTOK):
        for g in range(G):
            word = 0
            for a in range(TL):
                for t in range(T):
                    word |= ((int(tok[n, g * TL + a, t]) & MASK) << ((a * T + t) * RW))
            f.write(hx(word, TL * T * RW // 4) + chr(10))
with open(o + 'exp.hex', 'w') as f:
    for n in range(NTOK):
        for g in range(G):
            word = 0
            for a in range(TL):
                for t in range(T):
                    word |= ((int(out[n, g * TL + a, t]) & MASK) << ((a * T + t) * RW))
            f.write(hx(word, TL * T * RW // 4) + chr(10))
nbeats = 4 * 128 + 4 * 16 + 256 + 32 + 256 + 16
print('ld=%d tok=%d exp=%d' % (nbeats, NTOK * G, NTOK * G))

# --- intermediate dumps for stage localisation (token 0) ---
def dump2(M, name):
    with open(o + name, "w") as f:
        for e in range(E):
            for t in range(T):
                f.write(hx(M[e, t], 8) + chr(10))
    return
dump2(Q[0], "g_q.hex")
dump2(K[0], "g_k.hex")
dump2(V[0], "g_v.hex")
dump2(AV[0], "g_av.hex")
dump2(proj[0], "g_proj.hex")
dump2(ares[0], "g_ares.hex")
dump2(FC2[0], "g_fc2.hex")
with open(o + "g_hid.hex", "w") as f:
    for e in range(HIDDEN):
        for t in range(T):
            f.write(hx(HID[0, e, t], 8) + chr(10))
with open(o + "g_scr.hex", "w") as f:
    for j in range(NTOK):
        for t in range(T):
            f.write(hx(scr[0, j, t], 8) + chr(10))
print("intermediates dumped")

def dump2t(M, tok, name):
    with open(o + name, "w") as f:
        for e in range(E):
            for t in range(T):
                f.write(hx(M[tok][e, t], 8) + chr(10))
dump2t(Q, 1, "g_q1.hex")
dump2t(K, 1, "g_k1.hex")
dump2t(V, 1, "g_v1.hex")
dump2t(proj, 1, "g_proj1.hex")
dump2t(ares, 1, "g_ares1.hex")
dump2t(FC2, 1, "g_fc21.hex")
print("token1 dumped")

np.save(str(ROOT / 'build/spikformer/block_replay/out_golden.npy'), out)
print('saved out_golden.npy', out.shape)
