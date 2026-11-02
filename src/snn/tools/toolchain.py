"""Unified entry: python -B -m tools.toolchain --backend spikformer ...

ANN branch (dl_v1) is untouched; this toolchain owns the SNN backend only.
Commands: compile | build | package. build/package consume the manifest
and never re-export weights; compile never launches Vivado.

build --stage synth|ooc goes through scripts/spikformer/run_physical.py which
verifies manifest source hashes and starts Vivado ONLY with --execute and ONLY
under the shared heavy lock (via scripts/spikformer/run_heavy.py).
"""
from __future__ import annotations
import argparse, os, pathlib, subprocess, sys

ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[1]).resolve()


def _spik_compile(a):
    from tools.backends.spikformer import compile as C
    man = C.run(a.config, a.out_dir, weights=a.weights, params_dir=a.params_dir, fixture_seed=a.fixture_seed)
    print('manifest: ' + a.out_dir + '/manifest.json kind=' + man['model_kind'])


def _spik_build(a):
    import json
    man_path = pathlib.Path(a.manifest).resolve()
    man = json.loads(man_path.read_text(encoding='utf-8-sig'))
    if a.stage in ('synth', 'ooc'):
        cmd = [sys.executable, '-B', 'scripts/spikformer/run_physical.py',
               '--manifest', str(man_path), '--stage', a.stage,
               '--lock', a.shared_heavy_lock, '--wait-sec', str(a.wait_sec)]
        if a.execute:
            cmd.append('--execute')
        print('physical driver: ' + ' '.join(cmd))
        r = subprocess.run(cmd, cwd=str(ROOT))
        return r.returncode
    if a.stage in ('ip', 'platform'):
        gen = man.get('generated') or {}
        print('stage %s: project tcl = %s' % (a.stage, gen.get('project_tcl')))
        print('ip/platform packaging entry not executed here; use '
              'scripts/spikformer/run_ip_pack.py once the OOC stage is frozen.')
        return 0
    print('unknown stage: ' + a.stage)
    return 2


def _spik_package(a):
    from tools.backends.spikformer import package as P
    P.run(a.manifest, a.out_dir)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', required=True)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('compile')
    c.add_argument('--config', required=True)
    c.add_argument('--out-dir', required=True)
    c.add_argument('--weights', default=None)
    c.add_argument('--params-dir', default=None)
    c.add_argument('--input', default=None)
    c.add_argument('--fixture-seed', type=int, default=None)
    b = sub.add_parser('build')
    b.add_argument('--manifest', required=True)
    b.add_argument('--stage', required=True)
    b.add_argument('--host-tools-root', default='D:/dl_v0.1/verification/host')
    b.add_argument('--shared-heavy-lock', required=True)
    b.add_argument('--execute', action='store_true',
                   help='actually start the heavy tool under the shared lock')
    b.add_argument('--wait-sec', type=int, default=0)
    p = sub.add_parser('package')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    if a.backend != 'spikformer':
        print('unknown backend: ' + a.backend)
        return 2
    if a.cmd == 'compile':
        return _spik_compile(a)
    if a.cmd == 'build':
        return _spik_build(a)
    if a.cmd == 'package':
        return _spik_package(a)
    return 2


if __name__ == '__main__':
    sys.exit(main())
