"""SNN compile orchestration: config -> params -> schedule -> codegen -> manifest.

With --weights (real trained checkpoint, contract export_abi.checkpoint_npz_required_exact)
the strict importer validates every blob and returns the model assets (scales, per-block
thresholds, FP32 head params, corpus/vocab hashes); compile.py binds all of them into
params/ + manifest so the package/runtime can load them. Fixture mode is unchanged and
explicitly reports model_kind=development_fixture.
"""
from __future__ import annotations
import pathlib
from . import config as C
from . import weights as W
from . import schedule as S
from . import codegen as G
from . import manifest as M
def run(config_path, out_dir, weights=None, params_dir=None, fixture_seed=None,
        assets_extra=None):
    cfg = C.load(config_path)
    out = pathlib.Path(out_dir)
    if (out / 'manifest.json').exists():
        raise RuntimeError('refusing to overwrite non-empty out_dir: ' + str(out))
    (out / 'params').mkdir(parents=True, exist_ok=True)
    assets = None
    if weights or params_dir:
        if fixture_seed is not None:
            raise RuntimeError('fixture_seed conflicts with real weights')
        if not weights:
            raise RuntimeError('params_dir reuse not implemented; pass --weights')
        files, assets = W.import_checkpoint(cfg, weights, out / 'params')
        kind, seed = 'real_corpus', None
    else:
        if fixture_seed is None:
            raise RuntimeError('no weights: pass --fixture-seed N (explicit dev mode)')
        files = W.fixture_params(cfg, fixture_seed, out / 'params')
        kind, seed = 'development_fixture', fixture_seed
    for name, meta in files.items():
        meta['nbytes'] = int(pathlib.Path(out / 'params' / meta['bin']).stat().st_size)
    # Per-block thresholds (6 LIFs each) + embedding thr_e are real params: bind them
    # into the resolved config so the descriptor stream / manifest carry the values.
    if assets is not None:
        thr = assets['thresholds']
        cfg['thresholds'] = {'thr_e': int(thr['thr_e'])}
        for b in range(cfg['depth']):
            cfg['thresholds']['block%d' % b] = {k: int(v)
                                                for k, v in dict(thr['blocks'][b]).items()}
        cfg['assets'] = assets
    else:
        cfg['thresholds'] = {}
    if assets is not None and assets_extra is None and weights:
        # companion assets written next to the final checkpoint by
        # tools/spikformer_train/export_integer.py (corpus/vocab/label-map/split hashes,
        # calibration evidence). Auto-discovered so the official thin entry
        # (python -m tools.toolchain --backend spikformer compile --weights ...) binds them too.
        comp = pathlib.Path(weights).with_name(pathlib.Path(weights).stem + '.assets.json')
        if comp.exists():
            assets_extra = __import__('json').loads(comp.read_text(encoding='utf-8'))
    if assets is not None and assets_extra:
        assets = dict(assets)
        assets.update(assets_extra)
    sched = S.build(cfg, files)
    gen = G.generate(cfg, sched, files, out)
    man = M.write_manifest(cfg, sched, files, gen, out, kind, seed, assets=assets)
    (out / 'schedule.json').write_text(__import__('json').dumps(sched, indent=1))
    return man
