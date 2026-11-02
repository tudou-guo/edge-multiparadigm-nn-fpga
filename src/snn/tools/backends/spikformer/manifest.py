"""SNN manifest: binds config, numeric contract, params, sources, IP, model assets."""
from __future__ import annotations
import hashlib, json, os, pathlib
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()
def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write_manifest(cfg, sched, param_files, codegen_out, out_dir, model_kind, seed=None,
                   assets=None):
    out = pathlib.Path(out_dir)
    srcs = json.loads((ROOT / 'rtl/ops/spiking/source_manifest.json').read_text())['sources']
    assets = assets or {}
    structure = {
        'E': cfg['embed'], 'HEADS': cfg['heads'], 'HEAD_DIM': cfg['head_dim'],
        'HIDDEN': cfg['mlp_hidden'], 'T': cfg['time_steps'],
        'TOKENS': cfg['derived']['tokens'], 'DEPTH': cfg['depth'],
        'BATCH': cfg['batch'], 'TL': 8, 'TC': 8, 'RW': 4, 'PSUM_SLICE': 32,
        'SUM_W': cfg['derived']['sum_w'], 'lif_add_bits': 33, 'accum_bits': 32,
    }
    manifest = {
        'backend': 'spikformer',
        'schema': 'spikformer-manifest-1',
        'model_kind': model_kind,
        'fixture_seed': seed,
        'config': {k: v for k, v in cfg.items() if k != 'derived'},
        'derived': cfg['derived'],
        'structure': structure,
        'schedule': sched,
        'params': param_files,
        'rtl_sources': srcs,
        'rtl_sha256': {s: _sha(ROOT / s) for s in srcs},
        'generated': codegen_out,
        'target': cfg['target'],
        'head': {
            'reduction': cfg['head']['reduction'],
            'norm': 'ps_div',
            'params': assets.get('head'),
        },
        'assets': assets,
        'numeric': {
            'lif': 'mem=floor((pre+delta)/2) (arith >>1); spike iff valid && mem>=0 && '
                   'mem>=thr; hard reset 0; pre=0 at t=0; thr>=1; pre+delta sign-extended '
                   'to 33 bit BEFORE add (no 32-bit wrap)',
            'qk_popcount': 'per-T bitwise AND popcount, no /head_dim, SUM_W=6, max 32',
            'av': 'unnormalized sum over 32 source tokens, uint11, max 1024',
            'residual': {'bounds': cfg['derived']['stage_bounds'],
                         'worst': cfg['derived']['worst_residual'],
                         'bits': cfg['derived']['res_bits'], 'rw_alloc': 4},
            'quant': 'signed int8 weights, per-tensor symmetric scale (maxabs/127), '
                     'int32 bias in accumulator domain, BN state explicit',
            'thresholds': (assets.get('thresholds') if assets else {}),
            'scales': (assets.get('scales') if assets else {}),
            'model_assets': {
                'model_kind': model_kind,
                'corpus_sha256': assets.get('corpus_sha256'),
                'vocab_sha256': assets.get('vocab_sha256'),
                'checkpoint': assets.get('checkpoint'),
                'checkpoint_sha256': assets.get('checkpoint_sha256'),
                'bn_state': assets.get('bn_state'),
            },
        },
    }
    blob = json.dumps(manifest, sort_keys=True).encode()
    manifest['manifest_sha256'] = hashlib.sha256(blob).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    (out / 'resolved_config.json').write_text(json.dumps(cfg, indent=1))
    numeric = {
        'lif': 'V=(delta+pre)>>>1 tau=2, spike iff valid and 0<=V and V>=THR, reset 0',
        'lif_add_bits': 33,
        'qk_popcount': 'per-T sum width clog2(head_dim+1)=6, all-ones=32 representable, '
                       'no /head_dim',
        'av': 'unnormalized sum over TOKENS of score*V_spike, uint11 max 1024',
        'residual': {'bounds': cfg['derived']['stage_bounds'],
                     'worst': cfg['derived']['worst_residual'],
                     'bits': cfg['derived']['res_bits'], 'rw_alloc': 4},
        'head': {'reduction': cfg['head']['reduction'], 'norm': 'ps_div',
                 'note': 'PL uint32 per-channel sum over T*N; PS divides then FP32 Linear'},
        'quant': 'int8 weights; per-tensor scale/threshold/bias in params+manifest; '
                 'BN state explicit (absent for this batch)',
        'structure': structure,
        'thresholds': (assets.get('thresholds') if assets else {}),
        'scales': (assets.get('scales') if assets else {}),
        'model_assets': manifest['numeric']['model_assets'],
    }
    (out / 'numeric_contract.json').write_text(json.dumps(numeric, indent=1))
    return manifest
