"""Spikformer NLP 48h integer-variant trainer: real corpus -> strict integer checkpoint.

Model = the frozen integer RTL datapath, trained with quantisation-aware gradients:

  embed : delta = etab[id] + ptab[pos] (shared int8 table scale), LIF(thr_e) over T
  block : Q/K/V integer linear (+int32 bias) -> LIF; per head per t QK bit-AND popcount
          (no /head_dim) -> score; AV[i][a] = sum_j score[i][j]*V[j][a] (unnormalized,
          max 32*32=1024); proj -> LIF(proj) -> residual add; fc1 -> LIF(fc1);
          fc2 -> LIF(fc2) -> residual add
  head  : PL integer per-channel sum over T*N -> PS x/(T*N) FP32 -> FP32 Linear -> argmax

Training is quantisation-aware with straight-through int8 weights (forward VALUES are
integers: q = round(clamp(w,-127,127)); gradient flows through the STE). All datapath
arithmetic in float32 is exact integer arithmetic because the frozen bounds keep every
partial sum <= 2^24 (see contract linear_bounds_checked_at_export); the LIF add is done
in float64 and rounded back to an exact integer, so the trained forward is bit-identical
to tools/backends/spikformer/integer_ref.py. That equivalence is checked by
tools/spikformer_train/test_integer_ref.py.

Surrogate gradient (frozen decision): centred at (mem - thr), tau=2 factor 1/2, hard
reset is detached (spk.detach() in the reset path). The surrogate slope is normalised by
max(1,thr) so it is scale-free w.r.t. the integer threshold (the integer network is
scale-free by construction; the exact integer model is what is exported and evaluated).

Threshold calibration: one rule per LIF, integer domain (no percentile grid):
  thr = max(1, round(median of positive pre-activation (delta) values))
collected from the exact integer traces (integer_ref for the frozen export). Warn if
thr > max positive delta (LIF can never fire). Thresholds are fixed int32 buffers, not
learned Parameters.

Stages:
  --stage train   (default) train, early-stop on dev, write best float state + report
  --stage export  load best state, integer-domain calibration via integer_ref, write the
                  frozen npz (contract export_abi.checkpoint_npz_required_exact) and the
                  accuracy report (dev + fixed holdout), no test-label use.

Usage:
  E:/miniconda/envs/pytorch_clean/python.exe -B tools/spikformer_train/train_nlp.py \
      --config examples/spikformer/config/nlp_sentiment_b1.json \
      --data build/spikformer/nlp_48h_20261029_1915/corpus/corpus_encoded.npz \
      --run-dir build/spikformer/nlp_48h_20261029_1915 \
      --stage train --epochs 5
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.backends.spikformer import config as C          # noqa: E402
from tools.backends.spikformer import integer_ref as IREF  # noqa: E402
from tools.spikformer_train import integer_calib as ICAL       # noqa: E402

SURROGATE_SLOPE = 5.0


# ---------------------------------------------------------------------------
# frozen numeric primitives (integer-domain, STE)
# ---------------------------------------------------------------------------
def quant_ste(w):
    """Straight-through int8 quantisation: forward value is round(clamp(w,-127,127))."""
    q = torch.clamp(torch.round(w), -127.0, 127.0)
    return w + (q - w).detach()


class LobotoLIF(torch.autograd.Function):
    """One frozen LIF step: mem=floor((pre+delta)/2); spike iff mem>=0 and mem>=thr.

    Inputs are integer-valued float32 tensors (q = pre-activation delta, pre membrane).
    Returns (spike 0/1, next membrane). Hard reset to 0 on spike, detached in the reset
    path; surrogate gradient centred at (mem - thr) with the tau=2 1/2 factor.
    """

    @staticmethod
    def forward(ctx, delta, pre, thr, thr_norm):
        with torch.no_grad():
            tot = pre.double() + delta.double()
            mem = torch.div(tot, 2.0, rounding_mode='floor')
            spk = ((mem >= 0.0) & (mem >= float(thr))).to(torch.float32)
            mem_next = (mem * (1.0 - spk.double())).to(torch.float32)
        ctx.save_for_backward(mem.to(torch.float32), spk)
        ctx.thr = float(thr)
        ctx.norm = float(thr_norm)
        return spk, mem_next

    @staticmethod
    def backward(ctx, g_spk, g_mem):
        mem, spk = ctx.saved_tensors
        u = (mem - ctx.thr) / ctx.norm
        sg = SURROGATE_SLOPE / (1.0 + SURROGATE_SLOPE * u.abs()) ** 2
        sg = sg / ctx.norm * 0.5                      # dspk/ddelta; tau=2 -> 1/2
        g = g_mem * 0.5 * (1.0 - spk) + g_spk * sg    # d/dpre and d/ddelta
        return g, g, None, None


def lif_step(delta, pre, thr):
    """delta/pre integer-valued float32; thr int >= 1. Returns (spike, next_pre)."""
    thr_i = max(1, int(thr))
    return LobotoLIF.apply(delta, pre, thr_i, float(thr_i))


def _mem_decision(pre, delta):
    """Contract 'mem' trace value: floor((pre+delta)/2), i.e. the decision value BEFORE
    the hard reset (float64 exact integer -> float32 exact for |mem| <= 2^24)."""
    with torch.no_grad():
        tot = pre.double() + delta.double()
        return torch.div(tot, 2.0, rounding_mode='floor').to(torch.float32)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------
class QLinear(nn.Module):
    """Integer-unit linear: forward value = int8 weight * integer activation + int32 bias."""

    def __init__(self, out_f, in_f, init_std):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * float(init_std))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):
        return F.linear(x, quant_ste(self.weight)) + quant_ste(self.bias)


class QBlock(nn.Module):
    INIT_STD = {'q': 48.0, 'k': 48.0, 'v': 48.0, 'proj': 3.0, 'fc1': 18.0, 'fc2': 24.0}

    def __init__(self, dims):
        super().__init__()
        E, MLP = dims['E'], dims['MLP']
        self.q = QLinear(E, E, self.INIT_STD['q'])
        self.k = QLinear(E, E, self.INIT_STD['k'])
        self.v = QLinear(E, E, self.INIT_STD['v'])
        self.proj = QLinear(E, E, self.INIT_STD['proj'])
        self.fc1 = QLinear(MLP, E, self.INIT_STD['fc1'])
        self.fc2 = QLinear(E, MLP, self.INIT_STD['fc2'])
        for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
            self.register_buffer('thr_' + nm, torch.tensor(1, dtype=torch.int32))

    def thresholds(self):
        return {nm: int(getattr(self, 'thr_' + nm).item())
                for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2')}

    def set_thresholds(self, d):
        for nm, v in d.items():
            getattr(self, 'thr_' + nm).fill_(max(1, int(v)))

    def forward(self, x, trace=False):
        """x: [B,T,N,E] integer-valued float32 residual (uint4 domain)."""
        B, T, N, E = x.shape
        H, HD = self.H, self.HD
        tr = {'in': x} if trace else None

        # --- Q/K/V linear + LIF over t (state per token/channel) ---
        pre_q = torch.zeros(B, N, E, dtype=x.dtype, device=x.device)
        pre_k = torch.zeros_like(pre_q)
        pre_v = torch.zeros_like(pre_q)
        qs, ks, vs = [], [], []
        qd, kd, vd = [], [], []
        qm, km, vm = [], [], []
        for t in range(T):
            dq = self.q(x[:, t]); dk = self.k(x[:, t]); dv = self.v(x[:, t])
            if trace:
                qm.append(_mem_decision(pre_q, dq))
                km.append(_mem_decision(pre_k, dk))
                vm.append(_mem_decision(pre_v, dv))
            sq, pre_q = lif_step(dq, pre_q, self.thr_q)
            sk, pre_k = lif_step(dk, pre_k, self.thr_k)
            sv, pre_v = lif_step(dv, pre_v, self.thr_v)
            qs.append(sq); ks.append(sk); vs.append(sv)
            qd.append(dq); kd.append(dk); vd.append(dv)
        q = torch.stack(qs, 1); k = torch.stack(ks, 1); v = torch.stack(vs, 1)
        if trace:
            tr['q_delta'] = torch.stack(qd, 1).detach()
            tr['k_delta'] = torch.stack(kd, 1).detach()
            tr['v_delta'] = torch.stack(vd, 1).detach()
            tr['q_spikes'] = q.detach(); tr['k_spikes'] = k.detach(); tr['v_spikes'] = v.detach()
            tr['q_mem'] = torch.stack(qm, 1).detach()
            tr['k_mem'] = torch.stack(km, 1).detach()
            tr['v_mem'] = torch.stack(vm, 1).detach()

        # --- per-head per-t QK bit-AND popcount (NO /head_dim), unnormalized AV ---
        qh = q.view(B, T, N, H, HD)
        kh = k.view(B, T, N, H, HD)
        vh = v.view(B, T, N, H, HD)
        scores = torch.einsum('btihd,btjhd->bhtij', qh, kh)     # [B,H,T,N,N]
        av = torch.einsum('bhtij,btjhd->btihd', scores, vh)    # [B,T,N,H,HD]
        av = av.reshape(B, T, N, E)
        if trace:
            tr['scores'] = scores.permute(0, 2, 3, 4, 1).detach().contiguous()  # [B,T,N,N,H]
            tr['av'] = av.detach()

        # --- proj -> LIF -> residual add ---
        pre_p = torch.zeros(B, N, E, dtype=x.dtype, device=x.device)
        ps, pd, pm = [], [], []
        for t in range(T):
            dp = self.proj(av[:, t])
            if trace:
                pm.append(_mem_decision(pre_p, dp))
            sp, pre_p = lif_step(dp, pre_p, self.thr_proj)
            ps.append(sp); pd.append(dp)
        p = torch.stack(ps, 1)
        res1 = x + p
        if trace:
            tr['proj_delta'] = torch.stack(pd, 1).detach()
            tr['proj_spikes'] = p.detach()
            tr['proj_mem'] = torch.stack(pm, 1).detach()
            tr['res1'] = res1.detach()

        # --- fc1 -> LIF -> fc2 -> LIF -> residual add ---
        pre_1 = torch.zeros(B, N, self.fc1.weight.shape[0], dtype=x.dtype, device=x.device)
        f1s, f1d, f1m = [], [], []
        for t in range(T):
            d1 = self.fc1(res1[:, t])
            if trace:
                f1m.append(_mem_decision(pre_1, d1))
            s1, pre_1 = lif_step(d1, pre_1, self.thr_fc1)
            f1s.append(s1); f1d.append(d1)
        f1 = torch.stack(f1s, 1)
        pre_2 = torch.zeros(B, N, E, dtype=x.dtype, device=x.device)
        f2s, f2d, f2m = [], [], []
        for t in range(T):
            d2 = self.fc2(f1[:, t])
            if trace:
                f2m.append(_mem_decision(pre_2, d2))
            s2, pre_2 = lif_step(d2, pre_2, self.thr_fc2)
            f2s.append(s2); f2d.append(d2)
        f2 = torch.stack(f2s, 1)
        out = res1 + f2
        if trace:
            tr['fc1_delta'] = torch.stack(f1d, 1).detach()
            tr['fc1_spikes'] = f1.detach()
            tr['fc1_mem'] = torch.stack(f1m, 1).detach()
            tr['fc2_delta'] = torch.stack(f2d, 1).detach()
            tr['fc2_spikes'] = f2.detach()
            tr['fc2_mem'] = torch.stack(f2m, 1).detach()
            tr['out'] = out.detach()
        return out, tr


class Net(nn.Module):
    """Spikformer NLP integer model (graph mirrors integer_ref.run_sample exactly)."""

    EMB_INIT_STD = 60.0
    POS_INIT_STD = 20.0

    def __init__(self, dims):
        super().__init__()
        self.dims = dict(dims)
        E, N, V = dims['E'], dims['N'], dims['vocab_size']
        self.etab = nn.Parameter(torch.randn(V, E) * self.EMB_INIT_STD)
        self.ptab = nn.Parameter(torch.randn(N, E) * self.POS_INIT_STD)
        self.register_buffer('thr_e', torch.tensor(1, dtype=torch.int32))
        self.blocks = nn.ModuleList([QBlock(dims) for _ in range(dims['D'])])
        self.head = nn.Linear(E, dims['C'], bias=True)
        for b in self.blocks:
            b.H, b.HD = dims['H'], dims['HD']
        self.set_probe_hooks()

    # trace hooks for calibration bookkeeping
    def set_probe_hooks(self):
        pass

    def thresholds(self):
        d = {'e': int(self.thr_e.item())}
        for i, b in enumerate(self.blocks):
            d.update({'block%d.%s' % (i, k): v for k, v in b.thresholds().items()})
        return d

    def set_thresholds(self, d):
        self.thr_e.fill_(max(1, int(d['e'])))
        for i, b in enumerate(self.blocks):
            b.set_thresholds({k: d['block%d.%s' % (i, k)]
                              for k in ('q', 'k', 'v', 'proj', 'fc1', 'fc2')})

    def forward(self, ids, trace=False):
        dims = self.dims
        B, N = ids.shape
        T, E = dims['T'], dims['E']
        tr = {} if trace else None
        d0 = quant_ste(self.etab)[ids] + quant_ste(self.ptab)[None, :, :]
        if trace:
            tr['embed'] = {'delta': d0.detach()}
        pre = torch.zeros(B, N, E, dtype=d0.dtype, device=ids.device)
        spikes = []
        mems = []
        for _ in range(T):
            mems.append(_mem_decision(pre, d0))
            spk, pre = lif_step(d0, pre, self.thr_e)
            spikes.append(spk)
        x = torch.stack(spikes, 1)
        if trace:
            tr['embed']['spikes'] = x.detach()
            tr['embed']['mem'] = torch.stack(mems, 1).detach()
            tr['embed_spikes'] = x.detach()
        for b in self.blocks:
            x, btr = b(x, trace=trace)
            if trace:
                tr.setdefault('blocks', []).append(btr)
        pl = x.sum(dim=(1, 2))                       # integer PL sums, exact
        pooled = pl / float(T * N)                   # PS divide, FP32
        logits = self.head(pooled)
        if trace:
            tr['final_residual'] = x.detach()
            tr['pl_sums'] = pl.detach()
            tr['pooled'] = pooled.detach()
            tr['logits'] = logits.detach()
        return logits, tr

    # ---- integer export -------------------------------------------------
    def integer_params(self, meta=None):
        """Export the exact integer parameter container (contract key names)."""
        dims = self.dims
        E, D, MLP, C, V, N = (dims['E'], dims['D'], dims['MLP'], dims['C'],
                              dims['vocab_size'], dims['N'])
        g = np.asarray  # noqa: F841
        etab = torch.round(torch.clamp(quant_ste(self.etab), -127, 127)).detach().cpu().numpy().astype(np.int8)
        ptab = torch.round(torch.clamp(quant_ste(self.ptab), -127, 127)).detach().cpu().numpy().astype(np.int8)
        params = {'cfg': dict(dims), 'etab': etab, 'ptab': ptab,
                  'thr_e': int(self.thr_e.item()),
                  'head_w': self.head.weight.detach().cpu().numpy().astype(np.float32),
                  'head_b': self.head.bias.detach().cpu().numpy().astype(np.float32),
                  'blocks': []}
        for b in self.blocks:
            blk = {}
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                lin = getattr(b, nm)
                w = torch.round(torch.clamp(quant_ste(lin.weight), -127, 127)).detach().cpu().numpy()
                blk[nm + '_w'] = w.astype(np.int8)
                blk[nm + '_b'] = torch.round(quant_ste(lin.bias)).detach().cpu().numpy().astype(np.int32)
                blk[nm + '_thr'] = int(getattr(b, 'thr_' + nm).item())
            params['blocks'].append(blk)
        return params

    def export_npz_arrays(self, meta):
        """np.savez-ready dict using the frozen checkpoint_npz_required_exact names."""
        dims = self.dims
        ip = self.integer_params()
        etab_arr = ip['etab']
        v_actual = int(meta.get('vocab_size_actual') or etab_arr.shape[0])
        if v_actual < dims['vocab_size']:
            pad = np.zeros((dims['vocab_size'], dims['E']), np.int8)
            pad[:v_actual] = etab_arr[:v_actual]
            etab_arr = pad
        arrs = {
            'etab': etab_arr, 'ptab': ip['ptab'],
            'thr_e': np.array([ip['thr_e']], dtype=np.int32),
            'head_w': ip['head_w'], 'head_b': ip['head_b'],
            'meta_bn_state': np.array('absent'),
            'meta_model_kind': np.array(meta['model_kind']),
            'meta_corpus_sha256': np.array(meta['corpus_sha256']),
            'meta_vocab_sha256': np.array(meta['vocab_sha256']),
            'meta_cfg': np.array(json.dumps({k: dims[k] for k in
                                             ('E', 'H', 'HD', 'N', 'T', 'D', 'MLP', 'C', 'vocab_size')})),
            'embed_scale': np.array([_maxabs_scale(ip['etab'], ip['ptab'])], dtype=np.float32),
        }
        for b, blk in enumerate(ip['blocks']):
            p = 'block%d.' % b
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                arrs[p + nm + '_w'] = blk[nm + '_w']
                arrs[p + nm + '_b'] = blk[nm + '_b']
                arrs[p + nm + '_scale'] = np.array([_maxabs_scale(blk[nm + '_w'])], dtype=np.float32)
                arrs[p + nm + '_thr'] = np.array([blk[nm + '_thr']], dtype=np.int32)
        return arrs


def _maxabs_scale(*arrays):
    m = 0
    for a in arrays:
        a = np.asarray(a, dtype=np.int64)
        if a.size:
            m = max(m, int(np.abs(a).max()))
    return float(m) / 127.0 if m > 0 else 1.0


# ---------------------------------------------------------------------------
# calibration (integer domain, one rule per LIF)
# ---------------------------------------------------------------------------
_positive = ICAL.positive
_calib_rule = ICAL.calib_rule


def calibrate_thresholds_from_net(net, ids, chunk=16, rounds=2):
    """Calibrate int32 threshold buffers from exact integer torch traces."""
    report = {'warn': [], 'stats': {}}
    ids = np.asarray(ids, dtype=np.int64)
    for rnd in range(rounds):
        acc = {}
        with torch.no_grad():
            for i in range(0, len(ids), chunk):
                xb = torch.from_numpy(ids[i:i + chunk]).to(next(net.parameters()).device)
                _, tr = net(xb, trace=True)
                acc.setdefault('e', []).append(
                    np.asarray(tr['embed']['mem'].cpu()).reshape(-1))
                acc.setdefault('e_d', []).append(
                    int(np.max(np.asarray(tr['embed']['delta'].cpu()))))
                for b, btr in enumerate(tr['blocks']):
                    for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                        key = 'block%d.%s' % (b, nm)
                        acc.setdefault(key, []).append(
                            np.asarray(btr[nm + '_mem'].cpu()).reshape(-1))
                        acc.setdefault(key + '_d', []).append(
                            int(np.max(np.asarray(btr[nm + '_delta'].cpu()))))
        new = {'e': 1}
        new['e'] = _calib_rule(_positive(np.concatenate(acc['e'])),
                               max(acc['e_d']), 'embed', report)
        for b in range(len(net.blocks)):
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                key = 'block%d.%s' % (b, nm)
                new[key] = _calib_rule(_positive(np.concatenate(acc[key])),
                                       max(acc[key + '_d']), key, report)
        net.set_thresholds(new)
        report['round%d' % rnd] = {
            'thresholds': new,
            'firing_rates': layer_firing_rates(net, ids[:min(len(ids), 16)]),
        }
    final_rates = report['round%d' % (rounds - 1)]['firing_rates']
    report['final_firing_rates'] = final_rates
    report['min_firing_rate'] = min(final_rates.values())
    dead = [k for k, v in final_rates.items() if v <= 0.0]
    report['dead_lifs'] = dead
    if dead:
        raise RuntimeError('calibration produced dead LIFs (zero spikes on the calibration '
                           'set): %s' % dead)
    thr_all = [new[k] for k in new if k != 'e'] + [new['e']]
    if min(thr_all) < 1:
        raise RuntimeError('calibration produced thr < 1: %s' % new)
    return report


def layer_firing_rates(net, ids, chunk=8):
    rates = {}
    ids = np.asarray(ids, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, len(ids), chunk):
            xb = torch.from_numpy(ids[i:i + chunk]).to(next(net.parameters()).device)
            _, tr = net(xb, trace=True)
            def add(k, v):
                rates.setdefault(k, []).append(float(np.asarray(v.cpu()).mean()))
            add('embed', tr['embed_spikes'])
            for b, btr in enumerate(tr['blocks']):
                for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                    add('block%d.%s' % (b, nm), btr[nm + '_spikes'])
    return {k: float(np.mean(v)) for k, v in rates.items()}


# ---------------------------------------------------------------------------
# corpus loading / evaluation
# ---------------------------------------------------------------------------
def load_data(path):
    z = np.load(str(path))
    need = ('train_ids', 'train_y', 'dev_ids', 'dev_y', 'holdout_ids', 'holdout_y')
    for k in need:
        if k not in z.files:
            raise SystemExit('corpus npz missing %r (run prepare_corpus.py)' % k)
    return {k: z[k] for k in z.files}, z


def torch_accuracy(net, ids, y, batch=128):
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(ids), batch):
            xb = torch.from_numpy(np.asarray(ids[i:i + batch], dtype=np.int64)).to(
                next(net.parameters()).device)
            logits, _ = net(xb, trace=False)
            pred = logits.argmax(dim=1).cpu().numpy()
            correct += int((pred == y[i:i + batch]).sum())
    net.train()
    return correct / float(len(ids))


class _Ids(torch.utils.data.Dataset):
    def __init__(self, ids, y):
        self.ids = ids
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(np.asarray(self.ids[i], dtype=np.int64)), int(self.y[i])


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# train stage
# ---------------------------------------------------------------------------
def _make_opt(net, args):
    """Two LR groups: datapath integer-scale weights (large magnitudes -> large LR) and
    the FP32 head (standard init scale). Not a searched hyperparameter: one fixed choice."""
    dp = [p for n, p in net.named_parameters() if not n.startswith('head.')]
    hd = [p for n, p in net.named_parameters() if n.startswith('head.')]
    return torch.optim.Adam([
        {'params': dp, 'lr': args.lr},
        {'params': hd, 'lr': args.head_lr},
    ])


def stage_train(args, dims, corpus, paths):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if (args.device != 'auto') else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('device=%s torch=%s cuda=%s' % (device, torch.__version__, torch.version.cuda))

    X = corpus['train_ids']; Y = corpus['train_y']
    dev_ids, dev_y = corpus['dev_ids'], corpus['dev_y']
    net = Net(dims).to(device)
    nparams = sum(p.numel() for p in net.parameters())
    print('params=%d model=%s' % (nparams, dims))

    # initial integer-domain threshold calibration
    t0 = time.time()
    calib_ids = X[:args.calib_samples]
    cal = calibrate_thresholds_from_net(net, calib_ids, rounds=2)
    print('init thresholds: %s' % json.dumps(net.thresholds()))
    print('init firing rates: %s' % json.dumps(cal['round1']['firing_rates']))

    opt = _make_opt(net, args)
    ds = _Ids(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True,
                                     num_workers=0, drop_last=False,
                                     generator=torch.Generator().manual_seed(args.seed))
    hist = []
    best = {'dev_acc': -1.0, 'state': None, 'epoch': -1}
    run_start = time.time()
    epoch_sec = []
    stop = False
    for ep in range(args.epochs):
        ep_start = time.time()
        net.train()
        tot_loss = 0.0
        nb = 0
        for ids_b, y_b in dl:
            ids_b = ids_b.to(device)
            y_b = y_b.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = net(ids_b, trace=False)
            loss = F.cross_entropy(logits, y_b)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()
            tot_loss += float(loss)
            nb += 1
            if not np.isfinite(tot_loss):
                raise SystemExit('non-finite loss at epoch %d step %d' % (ep, nb))
        train_loss = tot_loss / max(1, nb)
        tr_acc = torch_accuracy(net, X, Y)
        dev_acc = torch_accuracy(net, dev_ids, dev_y)
        rec = {'epoch': ep, 'train_loss': train_loss, 'train_acc': tr_acc,
               'dev_acc': dev_acc, 'sec': round(time.time() - ep_start, 2),
               'grad_norm_last': float(gn)}
        if args.recalibrate:
            cal = calibrate_thresholds_from_net(net, calib_ids, rounds=1)
            rec['thresholds'] = net.thresholds()
            rec['firing_rates'] = cal['round0']['firing_rates']
            rec['thr_warn'] = cal['warn']
        hist.append(rec)
        epoch_sec.append(rec['sec'])
        print(json.dumps(rec, sort_keys=True))
        if dev_acc > best['dev_acc']:
            best = {'dev_acc': dev_acc, 'epoch': ep,
                    'state': copy.deepcopy(net.state_dict())}
        if ep >= 2 and dev_acc <= max(h['dev_acc'] for h in hist[:-1]) and \
                ep - best['epoch'] >= args.patience:
            print('early stop: no dev improvement for %d epochs' % args.patience)
            stop = True
        wall = time.time() - run_start
        if wall > args.max_wall_s:
            print('wall-clock cap hit: %.1fs > %ss' % (wall, args.max_wall_s))
            stop = True
        if stop:
            break

    net.load_state_dict(best['state'])
    net.eval()
    final_dev = torch_accuracy(net, dev_ids, dev_y)
    final_hold = torch_accuracy(net, corpus['holdout_ids'], corpus['holdout_y'])
    print('best epoch %d dev_acc=%.4f holdout_acc=%.4f' % (best['epoch'], final_dev, final_hold))

    out_dir = paths['run'] / 'train'
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / ('best_state_%s.pt' % args.tag)
    torch.save({'state_dict': net.state_dict(), 'dims': dims, 'epoch': best['epoch'],
                'dev_acc': final_dev, 'args': vars(args)}, str(state_path))
    report = {
        'stage': 'train', 'tag': args.tag, 'dims': dims,
        'seed': args.seed, 'lr': args.lr, 'batch': args.batch, 'epochs_max': args.epochs,
        'epochs_run': len(hist), 'history': hist, 'best_epoch': best['epoch'],
        'dev_acc_torch': final_dev, 'holdout_acc_torch': final_hold,
        'train_acc_torch': torch_accuracy(net, X, Y),
        'nparams': int(nparams), 'device': str(device), 'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'calib_initial': cal, 'init_sec': round(time.time() - t0, 2),
        'wall_s_train': round(time.time() - run_start, 2),
        'mean_epoch_s': round(float(np.mean(epoch_sec)), 2) if epoch_sec else None,
        'state_path': str(state_path),
        'corpus_npz_sha256': paths['data_sha'] if 'data_sha' in paths else None,
    }
    rep_path = paths['reports'] / ('train_%s.json' % args.tag)
    rep_path.write_text(json.dumps(report, indent=1))
    print('wrote %s and %s' % (state_path, rep_path))
    return report


# ---------------------------------------------------------------------------
# smoke stage (M2): finite loss/grads, non-degenerate outputs
# ---------------------------------------------------------------------------
def stage_smoke(args, dims, corpus, paths):
    """Short training smoke on real data: finite loss/grads, live layers, no collapse."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != 'auto' else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    torch.backends.cuda.matmul.allow_tf32 = False
    X, Y = corpus['train_ids'], corpus['train_y']
    net = Net(dims).to(device)
    cal = calibrate_thresholds_from_net(net, X[:args.calib_samples], rounds=2)
    opt = _make_opt(net, args)
    checks = {'loss_finite': True, 'grads_finite': True, 'grads_nonzero': {},
              'spike_rates': cal['round1']['firing_rates'], 'warm': []}
    n_steps = args.smoke_steps
    bs = args.batch
    net.train()
    t0 = time.time()
    first_loss = None
    for step in range(n_steps):
        i = (step * bs) % max(1, len(X) - bs)
        xb = torch.from_numpy(X[i:i + bs]).to(device)
        yb = torch.from_numpy(Y[i:i + bs]).to(device)
        opt.zero_grad(set_to_none=True)
        logits, _ = net(xb, trace=False)
        loss = F.cross_entropy(logits, yb)
        if not np.isfinite(float(loss)):
            checks['loss_finite'] = False
        loss.backward()
        step_gn = {}
        for name, p in net.named_parameters():
            if p.grad is None:
                checks['grads_nonzero'][name] = 0.0
                continue
            g = float(p.grad.norm())
            if not np.isfinite(g):
                checks['grads_finite'] = False
            step_gn[name] = g
        for name in ('etab', 'blocks.0.q.weight', 'blocks.0.k.weight', 'blocks.0.v.weight',
                     'blocks.0.proj.weight', 'blocks.0.fc1.weight', 'blocks.0.fc2.weight',
                     'head.weight', 'head.bias'):
            checks['grads_nonzero'][name] = step_gn.get(name, 0.0)
        if step == 0:
            first_loss = float(loss)
        checks['warm'].append({'step': step, 'loss': float(loss),
                               'gnorm': float(torch.nn.utils.clip_grad_norm_(
                                   net.parameters(), 1e9))})
        opt.step()
    checks['sec'] = round(time.time() - t0, 2)
    # class collapse check on dev
    preds = []
    net.eval()
    with torch.no_grad():
        for i in range(0, min(128, len(corpus['dev_ids'])), 64):
            xb = torch.from_numpy(corpus['dev_ids'][i:i + 64]).to(device)
            preds.append(net(xb, trace=False)[0].argmax(1).cpu().numpy())
    preds = np.concatenate(preds)
    checks['dev_pred_distribution'] = np.bincount(preds, minlength=dims['C']).tolist()
    checks['not_collapsed'] = len(set(preds.tolist())) > 1
    checks['loss_decreased'] = (checks['warm'][-1]['loss'] < first_loss) if len(checks['warm']) > 1 else None
    ok = (checks['loss_finite'] and checks['grads_finite'] and checks['not_collapsed']
          and all(v > 0 for k, v in checks['grads_nonzero'].items() if k in
                  ('etab', 'blocks.0.q.weight', 'blocks.0.proj.weight',
                   'blocks.0.fc1.weight', 'blocks.0.fc2.weight', 'head.weight')))
    checks['PASS'] = bool(ok)
    report = {'stage': 'smoke', 'dims': dims, 'seed': args.seed, 'device': str(device),
              'checks': checks, 'calib': cal,
              'corpus_npz_sha256': paths.get('data_sha')}
    paths['reports'].mkdir(parents=True, exist_ok=True)
    rep = paths['reports'] / ('train_smoke_%s.json' % args.tag)
    rep.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: checks[k] for k in
                      ('loss_finite', 'grads_finite', 'not_collapsed', 'PASS',
                       'grads_nonzero', 'spike_rates', 'dev_pred_distribution')},
                     indent=1))
    print('smoke report:', rep)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# export stage (M4): torch process dumps the candidate frozen npz; the pure-NumPy
# process tools/spikformer_train/export_integer.py does integer-domain calibration
# (integer_ref traces), final export, integer accuracy and strict compile import.
# ---------------------------------------------------------------------------
def stage_export(args, dims, corpus, paths):
    """Stage 2a (torch process): dump the candidate frozen npz from best_state.pt.

    The integer-domain threshold calibration (integer_ref traces), the final export and
    the integer accuracy/compile verification run in a SEPARATE pure-NumPy process
    (tools/spikformer_train/export_integer.py): on this host the torch and numpy OpenMP
    runtimes clash when integer matmuls run after torch in the same process. Keeping torch
    and the golden integer path in separate processes also keeps golden evidence clean.
    """
    device = torch.device('cpu')
    net = Net(dims).to(device)
    ck = torch.load(str(args.state), map_location=device, weights_only=False)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    print('loaded state %s (epoch %s)' % (args.state, ck.get('epoch')))
    meta = {'model_kind': args.model_kind,
            'corpus_sha256': args.corpus_sha256,
            'vocab_sha256': args.vocab_sha256}
    if paths['run'] is not None:
        cmf = paths['run'] / 'corpus' / 'corpus_meta.json'
        if cmf.exists():
            cm = json.loads(cmf.read_text(encoding='utf-8'))
            meta['corpus_sha256'] = cm['archive']['sha256']
            meta['vocab_sha256'] = cm['vocab']['sha256']
            meta['vocab_size_actual'] = int(cm['vocab']['size_actual'])
    if not meta['corpus_sha256'] or not meta['vocab_sha256']:
        raise SystemExit('export needs meta_corpus_sha256 and meta_vocab_sha256 '
                         '(pass --corpus-sha256/--vocab-sha256 or a corpus_meta.json)')
    arrs = net.export_npz_arrays(meta)
    paths['train'].mkdir(parents=True, exist_ok=True)
    out_npz = paths['train'] / ('ckpt_candidate_%s.npz' % args.tag)
    np.savez(str(out_npz), **arrs)
    cand_sha = _sha256(out_npz)
    side = {'stage': 'export_candidate', 'tag': args.tag, 'candidate': str(out_npz),
            'candidate_sha256': cand_sha, 'state': str(args.state),
            'state_epoch': ck.get('epoch'), 'dims': dims, 'meta': meta,
            'thresholds_train_time': net.thresholds(),
            'corpus_npz_sha256': paths.get('data_sha')}
    side_path = paths['train'] / ('export_candidate_%s.json' % args.tag)
    side_path.write_text(json.dumps(side, indent=1))
    print('wrote candidate', out_npz, 'sha256', cand_sha)
    print('wrote candidate sidecar', side_path)
    print('next (pure numpy process): E:/miniconda/python.exe -B '
          'tools/spikformer_train/export_integer.py --candidate %s --tag %s '
          '--run-dir %s' % (out_npz, args.tag, args.run_dir))
    return side


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--stage', choices=('train', 'smoke', 'export'), default='train')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.1,
                    help='datapath integer-scale LR (weights are O(10..127) integer units)')
    ap.add_argument('--head-lr', type=float, default=2e-3, help='FP32 head LR')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--calib-samples', type=int, default=64)
    ap.add_argument('--recalibrate', type=int, default=1, help='1 = recalibrate thresholds each epoch')
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--max-wall-s', type=float, default=7200.0)
    ap.add_argument('--smoke-steps', type=int, default=30)
    ap.add_argument('--state', default=None)
    ap.add_argument('--model-kind', default='real_corpus')
    ap.add_argument('--corpus-sha256', default=None)
    ap.add_argument('--vocab-sha256', default=None)
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    full = C.load(str(cfg_path))
    dims = {'E': full['embed'], 'H': full['heads'], 'HD': full['head_dim'],
            'N': full['seq_len'], 'T': full['time_steps'], 'D': full['depth'],
            'MLP': full['mlp_hidden'], 'C': full['num_classes'],
            'vocab_size': full['vocab_size']}
    paths = {'run': pathlib.Path(args.run_dir), 'train': pathlib.Path(args.run_dir) / 'train',
             'reports': pathlib.Path(args.run_dir) / 'reports'}
    paths['data_sha'] = _sha256(args.data) if pathlib.Path(args.data).exists() else None
    corpus, _z = load_data(args.data)
    print('corpus: train=%d dev=%d holdout=%d' % (len(corpus['train_y']),
                                                  len(corpus['dev_y']),
                                                  len(corpus['holdout_y'])))
    if args.stage == 'smoke':
        return stage_smoke(args, dims, corpus, paths)
    if args.stage == 'train':
        stage_train(args, dims, corpus, paths)
        return 0
    if args.stage == 'export':
        if not args.state:
            raise SystemExit('--stage export requires --state <best_state.pt>')
        stage_export(args, dims, corpus, paths)
        return 0
    raise SystemExit('unknown stage')


if __name__ == '__main__':
    sys.exit(main())
