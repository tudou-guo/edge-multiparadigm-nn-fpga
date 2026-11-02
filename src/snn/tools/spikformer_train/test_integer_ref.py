"""M1 short math/interface tests: PyTorch quantised forward vs independent NumPy integer_ref.

Covers the frozen contract edge cases (docs/SPIKFORMER_NLP_48H_HANDOFF_20261029.md +
contract.json verification_ladder):
  1. LIF threshold boundary / floor semantics (mem == thr fires, thr-1 does not, mem<0
     never, odd negative pre+delta floors like an arithmetic shift)
  2. QK popcount = 32 (SUM_W=6 representable) and unnormalised AV max = 32*32 = 1024
  3. one-hot patterns across tokens AND across t: same-t token pairing, per-head
     isolation, Q/K role orientation (asymmetric permutation), no cross-t leakage
  4. negative accumulator: pre < 0 never fires
  5. embedding frontend (etab+ptab shared quantisation domain) and head: PL integer sum,
     PS /(T*N) FP32 Linear == runtime/spikformer/head.py bit-exactly
  6. STE forward equivalence: forward value == round(clamp(w)) exactly
  7. full-trace PyTorch quantised forward == integer_ref on small and target configs

Every check that compares the two implementations compares integer traces
(first_mismatch localisation). PASS/FAIL is reported per test; exit code != 0 on fail.
Report: build/<run>/reports/math_tests_<tag>.json

Usage:
  E:/miniconda/envs/pytorch_clean/python.exe -B tools/spikformer_train/test_integer_ref.py \
      --out build/spikformer/nlp_48h_20261029_1915/reports/math_tests_m1.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.backends.spikformer import integer_ref as R          # noqa: E402
from tools.spikformer_train import train_nlp as TR              # noqa: E402
import runtime.spikformer.head as RHEAD                        # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False


SMALL = {'E': 32, 'H': 1, 'HD': 32, 'N': 4, 'T': 2, 'D': 2, 'MLP': 16, 'C': 2,
         'vocab_size': 50}
TARGET = {'E': 128, 'H': 4, 'HD': 32, 'N': 32, 'T': 4, 'D': 2, 'MLP': 512, 'C': 2,
          'vocab_size': 64}

RESULTS = []


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({'name': name, 'status': 'pass', 'detail': _jsonable(detail)})
        print('PASS %-46s %s' % (name, json.dumps(_jsonable(detail)) if detail else ''))
        return True
    except Exception as exc:  # noqa: BLE001
        RESULTS.append({'name': name, 'status': 'fail',
                        'detail': {'error': '%s: %s' % (type(exc).__name__, exc),
                                   'trace': traceback.format_exc().splitlines()[-6:]}})
        print('FAIL %-46s %s: %s' % (name, type(exc).__name__, exc))
        return False


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def randomize_net_integers(net, seed):
    """Set all datapath params/thresholds to exact integers (random)."""
    rng = np.random.RandomState(seed)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith('weight') and 'head' not in name:
                v = rng.randint(-127, 128, size=tuple(p.shape))
                p.copy_(torch.from_numpy(v.astype(np.float32)))
            elif name.endswith('bias') and 'head' not in name:
                v = rng.randint(-60, 61, size=tuple(p.shape))
                p.copy_(torch.from_numpy(v.astype(np.float32)))
            elif 'head' in name:
                v = rng.randn(*tuple(p.shape)).astype(np.float32) * 0.1
                p.copy_(torch.from_numpy(v))
        for b in net.blocks:
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                getattr(b, 'thr_' + nm).fill_(int(rng.randint(1, 300)))
        net.thr_e.fill_(int(rng.randint(1, 300)))
    return net


def torch_trace_to_np(tr):
    def cv(t):
        return np.rint(np.asarray(t.detach().cpu().numpy())).astype(np.int64)

    out = {'embed': {'delta': cv(tr['embed']['delta']),
                     'spikes': cv(tr['embed']['spikes']),
                     'mem': cv(tr['embed']['mem'])},
           'embed_spikes': cv(tr['embed_spikes']),
           'blocks': [],
           'final_residual': cv(tr['final_residual']),
           'pl_sums': cv(tr['pl_sums']),
           'logits': np.asarray(tr['logits'].detach().cpu().numpy(), dtype=np.float32)}
    for btr in tr['blocks']:
        out['blocks'].append({k: cv(v) for k, v in btr.items()})
    return out


def assert_traces_equal(reftr, gottr, label):
    keys = [k for k in R.trace_keys(reftr) if k != 'logits']
    mm = R.first_mismatch(reftr, gottr, keys=keys)
    if mm is not None:
        raise AssertionError('%s trace mismatch %s' % (label, json.dumps(_jsonable(mm))))
    # logits: FP32 head may differ in the last ulp; class must agree
    a = np.asarray(reftr['logits'], np.float32)
    b = np.asarray(gottr['logits'], np.float32)
    if a.shape != b.shape or not np.allclose(a, b, rtol=1e-5, atol=1e-5):
        raise AssertionError('%s logits mismatch ref=%s got=%s' % (label, a, b))
    if not np.array_equal(np.argmax(a, 1), np.argmax(b, 1)):
        raise AssertionError('%s argmax mismatch' % label)
    return {'layers_compared': len(keys)}


def set_block_identity(blk, scale=2, thr=1, perm=None):
    """Set q/k/v/proj weights (integer) and thresholds explicitly."""
    E = blk.q.weight.shape[0]
    with torch.no_grad():
        for nm in ('q', 'v', 'proj', 'fc1', 'fc2'):
            pass
        I = np.eye(E, dtype=np.int64) * scale
        blk.q.weight.copy_(torch.from_numpy(I.astype(np.float32)))
        blk.q.bias.zero_()
        blk.k.weight.copy_(torch.from_numpy(I.astype(np.float32)))
        blk.k.bias.zero_()
        blk.v.weight.copy_(torch.from_numpy(I.astype(np.float32)))
        blk.v.bias.zero_()
        blk.thr_q.fill_(thr); blk.thr_k.fill_(thr); blk.thr_v.fill_(thr)


# ---------------------------------------------------------------------------
# 1. LIF threshold boundary / floor semantics
# ---------------------------------------------------------------------------
def t_lif_boundary():
    thr = 3
    # hand-computed integer LIF chain (tau=2, hard reset, floor == arithmetic >>1)
    deltas = np.array([6, 5, 4, -1, -3, 7], dtype=np.int64).reshape(6, 1)
    exp_spk, exp_mem = [], []
    pre = 0
    for d in deltas[:, 0]:
        mem = (pre + int(d)) >> 1
        spk = 1 if (mem >= 0 and mem >= thr) else 0
        exp_spk.append(spk); exp_mem.append(mem)
        pre = 0 if spk else mem
    spk, mem = R.lif_over_time(deltas, thr)
    if spk[:, 0].tolist() != exp_spk or mem[:, 0].tolist() != exp_mem:
        raise AssertionError('integer LIF chain mismatch exp=%s/%s got=%s/%s'
                             % (exp_spk, exp_mem, spk[:, 0].tolist(), mem[:, 0].tolist()))
    # threshold boundary: mem == thr fires, mem == thr-1 does not
    for d, want in ((6, 1), (5, 0), (-2, 0), (-1, 0), (-3, 0)):
        s, m = R.lif_over_time(np.array([[d]], np.int64), thr)
        memv = int(m[0, 0])
        if memv != (d >> 1):
            raise AssertionError('floor mismatch d=%d mem=%d' % (d, memv))
        if int(s[0, 0]) != want:
            raise AssertionError('spike mismatch d=%d thr=%d got=%d want=%d'
                                 % (d, thr, int(s[0, 0]), want))
    # torch LIF primitive must agree bit-exactly (integer-valued tensors)
    for d in (6, 5, 4, -1, -3, 7, -7):
        pre = torch.zeros(1)
        spk_t, mem_t = TR.lif_step(torch.tensor([float(d)]), pre, thr)
        memv = int(d) >> 1
        want = 1 if (memv >= 0 and memv >= thr) else 0
        want_mem_next = 0 if want else memv          # hard reset to 0 on spike
        if int(spk_t.item()) != want or int(mem_t.item()) != want_mem_next:
            raise AssertionError('torch LIF mismatch d=%d thr=%d spk=%d/%d mem=%d/%d'
                                 % (d, thr, int(spk_t.item()), want,
                                    int(mem_t.item()), want_mem_next))
    # thr >= 1 is enforced
    try:
        R.lif_over_time(np.zeros((2, 1), np.int64), 0)
    except ValueError:
        pass
    else:
        raise AssertionError('LIF accepted thr=0')
    return {'case': 'thr=3 mem>=thr fires, mem=thr-1 not, negative floors'}


# ---------------------------------------------------------------------------
# 2. popcount=32 and AV max=1024
# ---------------------------------------------------------------------------
def t_popcount32_avmax():
    cfg = {'E': 32, 'H': 1, 'HD': 32, 'N': 32, 'T': 1, 'D': 1, 'MLP': 16, 'C': 2,
           'vocab_size': 64}
    net = TR.Net(cfg)
    randomize_net_integers(net, 7)
    blk = net.blocks[0]
    set_block_identity(blk, scale=2, thr=1)
    # input: every channel spikes at t=0 for every token -> Q=K=V=all ones
    x = torch.ones(1, 1, 32, 32)
    _, tr = blk(x, trace=True)
    if int(tr['q_spikes'].sum()) != 32 * 32:
        raise AssertionError('all-ones Q spikes expected, got %d' % int(tr['q_spikes'].sum()))
    sc = tr['scores']
    if int(sc.max()) != 32 or int(sc.min()) != 32:
        raise AssertionError('popcount all-ones != 32 (max=%d min=%d)'
                             % (int(sc.max()), int(sc.min())))
    av = tr['av']
    if int(av.max()) != 1024:
        raise AssertionError('AV max %d != 1024' % int(av.max()))
    # integer_ref (independent implementation) must give the same
    ip = net.integer_params()
    ref = R.run_sample(ip, np.array([1] * 32), cfg=cfg, trace=True)
    assert_traces_equal(ref, torch_trace_np_full(net, np.array([1] * 32), cfg), 'popcount32')
    return {'score_max': 32, 'av_max': 1024, 'SUM_W': 6, 'popcount_bits': 32}


def torch_trace_np_full(net, ids, cfg):
    with torch.no_grad():
        _, tr = net(torch.from_numpy(np.asarray(ids, dtype=np.int64)[None, :]), trace=True)
    return torch_trace_to_np(tr)


# ---------------------------------------------------------------------------
# 3. one-hot across tokens and across t, asymmetric Q/K roles
# ---------------------------------------------------------------------------
def t_onehot_cross_token_and_t():
    cfg = dict(SMALL)   # E32/H1/HD32/N=4/T=2
    net = TR.Net(cfg)
    randomize_net_integers(net, 11)
    blk = net.blocks[0]
    E = cfg['E']
    perm = np.array([2, 3, 0, 1, 6, 7, 4, 5])   # asymmetric channel permutation
    set_block_identity(blk, scale=2, thr=1)
    with torch.no_grad():
        # k_w = 2*P with P[c,i]=1 for c=perm(i): k spike channel = perm(input channel)
        K = np.zeros((E, E), np.int64)
        for i in range(len(perm)):
            K[int(perm[i]), i] = 2
        blk.k.weight.copy_(torch.from_numpy(K.astype(np.float32)))
    # tokens: distinct one-hot channels per token and per t (t-dependent channels)
    ch = {(n, t): (n + 4 * t) for n in range(4) for t in range(2)}
    x = np.zeros((1, 2, 4, E), np.int64)
    for n in range(4):
        for t in range(2):
            x[0, t, n, ch[(n, t)]] = 1
    x_t = torch.from_numpy(x.astype(np.float32))
    _, tr = blk(x_t, trace=True)
    # independent slow-loop expected values (pure python integers)
    q_spk = np.zeros((1, 2, 4, E), np.int64)
    k_spk = np.zeros_like(q_spk)
    for t in range(2):
        for n in range(4):
            q_spk[0, t, n, ch[(n, t)]] = 1
            k_spk[0, t, n, int(perm[ch[(n, t)]])] = 1
    if not np.array_equal(np.rint(tr['q_spikes'].numpy()).astype(np.int64), q_spk):
        raise AssertionError('Q spikes (cross-t leakage?) differ from expected one-hot')
    if not np.array_equal(np.rint(tr['k_spikes'].numpy()).astype(np.int64), k_spk):
        raise AssertionError('K spikes differ from expected permuted one-hot')
    # (perm covers all channels used below; scores/AV validated by explicit loop)
    scores = np.rint(tr['scores'].numpy()).astype(np.int64)
    for t in range(2):
        for i in range(4):
            for j in range(4):
                want = 1 if ch[(i, t)] == perm[ch[(j, t)]] else 0
                if int(scores[0, t, i, j, 0]) != want:
                    raise AssertionError('score[%d,%d,%d] %d != %d (Q/K role or same-t pairing)'
                                         % (t, i, j, int(scores[0, t, i, j, 0]), want))
    exp_av = np.zeros((1, 2, 4, E), np.int64)
    for t in range(2):
        for i in range(4):
            for j in range(4):
                if ch[(i, t)] == perm[ch[(j, t)]]:
                    exp_av[0, t, i, ch[(j, t)]] += 1
    av = np.rint(tr['av'].numpy()).astype(np.int64)
    if not np.array_equal(av, exp_av):
        raise AssertionError('AV token pairing mismatch (cross-t or Q/K swap)')
    # integer_ref must agree with the torch implementation on the full trace
    ip = net.integer_params()
    ref = R.run_sample(ip, np.array([1] * cfg['N']), cfg=SMALL, trace=True)
    got = torch_trace_np_full(net, np.array([1] * cfg['N']), cfg)
    assert_traces_equal(ref, got, 'onehot')
    return {'t_dependent_scores': True, 'qk_asymmetric_permutation': perm.tolist(),
            'av_token_pairing': 'ok'}


# ---------------------------------------------------------------------------
# 4. negative accumulator never fires
# ---------------------------------------------------------------------------
def t_negative_accumulator():
    cfg = dict(SMALL)
    net = TR.Net(cfg)
    randomize_net_integers(net, 13)
    blk = net.blocks[0]
    set_block_identity(blk, scale=-2, thr=1)
    x = torch.ones(1, cfg['T'], cfg['N'], cfg['E'])
    _, tr = blk(x, trace=True)
    if int(tr['q_spikes'].sum()) != 0:
        raise AssertionError('negative accumulator fired %d times' % int(tr['q_spikes'].sum()))
    if int(tr['q_mem'].max()) >= 0:
        raise AssertionError('negative delta produced mem >= 0')
    ip = net.integer_params()
    x_np = np.ones((1, cfg['T'], cfg['N'], cfg['E']), np.int64)
    x_out, btr = R.block_forward(ip['blocks'][0], x_np, cfg, trace=True)
    if int(np.asarray(btr['q_spikes']).sum()) != 0:
        raise AssertionError('integer_ref negative accumulator fired')
    if int(np.asarray(btr['q_mem']).max()) >= 0:
        raise AssertionError('integer_ref negative mem >= 0 (max=%d)'
                             % int(np.asarray(btr['q_mem']).max()))
    if int(np.asarray(btr['q_delta']).max()) != -2:
        raise AssertionError('expected delta=-2 for all-negative accumulator')
    return {'q_spikes': 0, 'q_mem_max': int(np.asarray(btr['q_mem']).max()),
            'q_delta_max': -2}


# ---------------------------------------------------------------------------
# 5. embedding frontend + head == runtime PS head bit-exactly
# ---------------------------------------------------------------------------
def t_embed_and_head():
    cfg = dict(SMALL)
    net = TR.Net(cfg)
    randomize_net_integers(net, 17)
    net.thr_e.fill_(40)
    ids = np.array([3, 7, 11, 2], np.int64)
    with torch.no_grad():
        _, tr = net(torch.from_numpy(ids)[None, :], trace=True)
    ip = net.integer_params()
    ref = R.run_sample(ip, ids, cfg=cfg, trace=True)
    if not np.array_equal(np.rint(tr['embed']['delta'].numpy()).astype(np.int64),
                          ref['embed']['delta']):
        raise AssertionError('embed delta (etab+ptab shared domain) mismatch')
    if not np.array_equal(np.rint(tr['embed_spikes'].numpy()).astype(np.int64),
                          ref['embed_spikes']):
        raise AssertionError('embed LIF spikes mismatch')
    # PL sum is the exact integer per-channel sum over T*N
    pl = np.array(ref['pl_sums'])[0]
    want = np.asarray(ref['final_residual'][0]).sum(axis=(0, 1))
    if not np.array_equal(pl.astype(np.int64), want):
        raise AssertionError('PL sum != sum over T*N')
    # head: integer_ref vs runtime PS head, bit-exact FP32 logits
    cls_r, y_r = RHEAD.ps_head(pl, ip['head_w'], ip['head_b'], cfg['T'], cfg['N'],
                               norm='ps_div')
    if not np.array_equal(y_r, ref['logits'][0]):
        raise AssertionError('integer_ref logits != runtime/head.py ps_head (%s vs %s)'
                             % (y_r, ref['logits'][0]))
    if cls_r != int(ref['cls'][0]):
        raise AssertionError('argmax mismatch')
    # torch quantised head must pick the same class
    if int(tr['logits'].argmax(dim=1)[0]) != cls_r:
        raise AssertionError('torch quantised forward class != PS head class')
    return {'thr_e': 40, 'pl_sum_max': int(pl.max()), 'head_class': cls_r,
            'ps_head_bit_exact': True}


# ---------------------------------------------------------------------------
# 6. STE forward == round(clamp(w)) exactly
# ---------------------------------------------------------------------------
def t_ste_forward():
    cfg = dict(SMALL)
    net = TR.Net(cfg)
    randomize_net_integers(net, 19)
    # make weights non-integer: scale params by 0.37 -> forward still uses round()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if name.endswith('weight') and 'head' not in name:
                p.mul_(0.371)
    ids = np.array([1, 5, 9, 2], np.int64)
    with torch.no_grad():
        _, tr_ste = net(torch.from_numpy(ids)[None, :], trace=True)
        # same net with params replaced by the exact STE forward values
        net2 = TR.Net(cfg)
        net2.load_state_dict(net.state_dict())
        for name, p in net2.named_parameters():
            if name.endswith('weight') and 'head' not in name:
                p.copy_(torch.clamp(torch.round(p), -127, 127))
    ip = net.integer_params()
    ref = R.run_sample(ip, ids, cfg=cfg, trace=True)
    got = torch_trace_to_np(tr_ste)
    assert_traces_equal(ref, got, 'ste_forward')
    return {'ste_equals_round_clamp': True}


# ---------------------------------------------------------------------------
# 7. full trace equality on small and target configs
# ---------------------------------------------------------------------------
def _full_trace_case(cfg, seed, ids_seed):
    net = TR.Net(cfg)
    randomize_net_integers(net, seed)
    rng = np.random.RandomState(ids_seed)
    ids = rng.randint(0, cfg['vocab_size'], size=cfg['N']).astype(np.int64)
    ip = net.integer_params()
    ref = R.run_sample(ip, ids, cfg=cfg, trace=True)
    got = torch_trace_np_full(net, ids, cfg)
    return assert_traces_equal(ref, got, 'full_trace_seed%d' % seed)


def t_full_small():
    return _full_trace_case(dict(SMALL), 23, 3)


def t_full_target():
    return _full_trace_case(dict(TARGET), 29, 5)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    print('torch', torch.__version__, 'device=cpu; integer_ref', R.__file__)
    ok = True
    for name, fn in [
        ('lif_threshold_boundary', t_lif_boundary),
        ('popcount32_avmax1024', t_popcount32_avmax),
        ('onehot_cross_token_and_t', t_onehot_cross_token_and_t),
        ('negative_accumulator', t_negative_accumulator),
        ('embed_shared_domain_and_ps_head', t_embed_and_head),
        ('ste_forward_equals_round', t_ste_forward),
        ('full_trace_small_E32', t_full_small),
        ('full_trace_target_E128', t_full_target),
    ]:
        ok = check(name, fn) and ok
    n_pass = sum(1 for r in RESULTS if r['status'] == 'pass')
    summary = {'stage': 'M1_math_tests', 'PASS': bool(ok and n_pass == len(RESULTS)),
               'n_pass': n_pass, 'n_total': len(RESULTS), 'results': RESULTS,
               'torch': torch.__version__,
               'integer_ref': str(pathlib.Path(R.__file__)),
               'train_nlp': str(pathlib.Path(TR.__file__))}
    print('\nRESULT: %d/%d PASS -> %s' % (n_pass, len(RESULTS),
                                          'PASS' if summary['PASS'] else 'FAIL'))
    if args.out:
        p = pathlib.Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=1))
        print('report:', p)
    return 0 if summary['PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
