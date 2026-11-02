"""Contract checker: config/derived/contract/checkpoint consistency for the 48h run.

Fails loudly (JSON status=fail) on: numeric-contract drift, out-of-int32
accumulator bounds (including real bias), bad quantized weight ranges, missing
thresholds/head params, or a checkpoint that does not cover the frozen config.
Gamma: independent of the trainer internals; reads only config/contract/npz.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
sys.path.insert(0, str(ROOT))


def _need_bits(v):
    return max(1, int(math.ceil(math.log2(v + 1))))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--contract', required=True)
    ap.add_argument('--checkpoint', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    from tools.backends.spikformer import config as C
    cfg = C.load(a.config)
    d = cfg['derived']
    con = json.loads(pathlib.Path(a.contract).read_text(encoding='utf-8-sig'))
    checks = []

    def check(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})

    m = con.get('model', {})
    for k, want in (('E', cfg['embed']), ('heads', cfg['heads']),
                    ('head_dim', cfg['head_dim']), ('seq_len', d['tokens']),
                    ('mlp_hidden', cfg['mlp_hidden']), ('depth', cfg['depth']),
                    ('time_steps', cfg['time_steps']),
                    ('vocab_size', cfg.get('vocab_size', 0))):
        check('model.%s' % k, m.get(k) == want, 'contract=%r config=%r' % (m.get(k), want))
    num = con.get('numeric', {})
    check('rw_alloc', num.get('rw_alloc_bits') == d['rw'], 'contract=%r derived=%r' % (num.get('rw_alloc_bits'), d['rw']))
    check('res_bits_min', num.get('res_bits_min_required') == d['res_bits_min'],
          'contract=%r derived=%r' % (num.get('res_bits_min_required'), d['res_bits_min']))
    check('psum_slice', d['psum_slice'] == 32, 'derived=%r' % d['psum_slice'])
    check('lif_add_bits', d['lif_add_bits'] == 33, 'derived=%r' % d['lif_add_bits'])
    check('sum_w', num.get('qk', {}).get('sum_w_bits') == d['sum_w'],
          'contract=%r derived=%r' % (num.get('qk', {}).get('sum_w_bits'), d['sum_w']))
    check('av_max', num.get('av', {}).get('max') == d['av_max'],
          'contract=%r derived=%r' % (num.get('av', {}).get('max'), d['av_max']))
    check('av_bits', num.get('av', {}).get('bits_required') == d['av_bits'],
          'contract=%r derived=%r' % (num.get('av', {}).get('bits_required'), d['av_bits']))
    cb = num.get('linear_bounds_checked_at_export', {})
    db = d['accum_bounds']
    check('bound.qkv.pos', cb.get('qkv', {}).get('pos') == db['qkv']['pos'],
          'contract=%r derived=%r' % (cb.get('qkv', {}).get('pos'), db['qkv']['pos']))
    check('bound.proj.pos', cb.get('proj', {}).get('pos') == db['proj']['pos'],
          'contract=%r derived=%r' % (cb.get('proj', {}).get('pos'), db['proj']['pos']))
    check('bound.proj.neg', cb.get('proj', {}).get('neg') == db['proj']['neg'],
          'contract=%r derived=%r' % (cb.get('proj', {}).get('neg'), db['proj']['neg']))
    check('bound.fc1.pos', cb.get('fc1', {}).get('pos') == db['fc1']['pos'],
          'contract=%r derived=%r' % (cb.get('fc1', {}).get('pos'), db['fc1']['pos']))
    check('bound.fc2.pos', cb.get('fc2', {}).get('pos') == db['fc2']['pos'],
          'contract=%r derived=%r' % (cb.get('fc2', {}).get('pos'), db['fc2']['pos']))
    for name, b in db.items():
        check('int32.%s' % name, -(2**31) <= b['neg'] and b['pos'] <= 2**31 - 1, repr(b))
    st = d['stream']
    check('stream.block_w_beats', st['block_w_beats'] > 0, repr(st))
    check('stream.mlp_fc1', st['mlp_fc1_internal_beats'] > 0, repr(st))
    E, MLP, D = cfg['embed'], cfg['mlp_hidden'], cfg['depth']
    extra = []
    ck = pathlib.Path(a.checkpoint) if a.checkpoint else None
    if ck is not None and ck.is_file():
        z = np.load(str(ck))
        keys = set(z.files)
        check('ckpt.meta_bn_state', str(z['meta_bn_state']) in ('absent', 'already_folded'),
              str(z['meta_bn_state']) if 'meta_bn_state' in keys else 'missing')
        thr_need = {'thr_e'}
        for b in range(D):
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                for suffix in ('_w', '_b', '_scale', '_thr'):
                    if suffix != '_thr':
                        need_key = 'block%d.%s%s' % (b, nm, suffix)
                    else:
                        need_key = 'block%d.%s%s' % (b, nm, suffix)
                thr_need.add('block%d.%s_thr' % (b, nm))
        missing = sorted(k for k in (['thr_e'] + ['block%d.%s_thr' % (b, nm) for b in range(D)
                            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2')]) if k not in keys)
        check('ckpt.thresholds_present', not missing, 'missing=%r' % missing)
        for b in range(D):
            for nm in ('q', 'k', 'v', 'proj', 'fc1', 'fc2'):
                name = 'block%d.%s_w' % (b, nm)
                if name not in keys:
                    check('ckpt.' + name, False, 'missing')
                    continue
                w = np.asarray(z[name])
                check('ckpt.%s.dtype' % name, w.dtype == np.int8, str(w.dtype))
                check('ckpt.%s.range' % name, int(w.min()) >= -128 and int(w.max()) <= 127,
                      'min=%d max=%d' % (w.min(), w.max()))
                sv = 'block%d.%s_scale' % (b, nm)
                if sv in keys:
                    s = float(np.asarray(z[sv]).reshape(-1)[0])
                    extra.append((name, s))
                    check('ckpt.%s.scale' % sv, s > 0, repr(s))
                bias = np.asarray(z['block%d.%s_b' % (b, nm)]) if 'block%d.%s_b' % (b, nm) in keys else None
                if bias is not None:
                    check('ckpt.block%d.%s_b.dtype' % (b, nm), bias.dtype == np.int32, str(bias.dtype))
                    bmax = int(np.abs(bias.astype(np.int64)).max())
                    check('ckpt.block%d.%s_b.range' % (b, nm), bmax <= 2**31 - 1, 'maxabs=%d' % bmax)
                    layer = {'q': 'qkv', 'k': 'qkv', 'v': 'qkv', 'proj': 'proj',
                             'fc1': 'fc1', 'fc2': 'fc2'}[nm]
                    bound = max(abs(db[layer]['pos']), abs(db[layer]['neg']))
                    check('ckpt.%s.bound_plus_bias' % name, bound + bmax <= 2**31 - 1,
                          'bound=%d bias=%d' % (bound, bmax))
        check('ckpt.head_w', 'head_w' in keys, 'head_w present')
        if 'head_w' in keys:
            hw = np.asarray(z['head_w'])
            check('ckpt.head_w.shape', hw.shape == (cfg['num_classes'], E), repr(hw.shape))
            check('ckpt.head_w.dtype', hw.dtype.name.startswith('float'), str(hw.dtype))
        check('ckpt.head_b', 'head_b' in keys, 'head_b present')
        if 'head_b' in keys:
            check('ckpt.head_b.shape', np.asarray(z['head_b']).shape == (cfg['num_classes'],),
                  repr(np.asarray(z['head_b']).shape))
        if 'meta_model_kind' in keys:
            mk = str(z['meta_model_kind'])
            check('ckpt.model_kind_real', mk == 'real_corpus', 'meta_model_kind=%r' % mk)
        for k in ('meta_corpus_sha256', 'meta_vocab_sha256'):
            check('ckpt.' + k, k in keys, 'present' if k in keys else 'missing')
    else:
        check('ckpt.present', False, 'no checkpoint given/existing: %r' % (str(ck) if ck else None))
    fails = [c for c in checks if not c['ok']]
    rec = {'status': 'pass' if not fails else 'fail', 'config': a.config,
           'contract': a.contract, 'checkpoint': str(ck) if ck else None,
           'checks': checks, 'fails': [c['name'] for c in fails],
           'derived': {k: d[k] for k in ('tokens', 'sum_w', 'rw', 'res_bits_min',
                                        'psum_slice', 'lif_add_bits', 'av_max', 'av_bits')}}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({'status': rec['status'], 'fails': rec['fails'],
                      'checks': len(checks), 'out': str(out)}, indent=1))
    return 0 if not fails else 1


if __name__ == '__main__':
    sys.exit(main())
