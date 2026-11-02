"""ModelSim driver for the Spikformer NLP replay testbenches (task-4, replay line).

Cases (frozen):
  small      : tb_nlp_small.v     E32/H1/T2/seq4/depth2      (light sim)
  multihead  : tb_nlp_multihead.v block-level E64/H2/T2/seq4/depth1 (light sim)
  e2e        : tb_nlp_e2e.v       target E128/H4/T4/seq32/depth2 (long sim -> heavy lock)

Every run compiles in a FRESH directory:
  build/spikformer/nlp_48h_20261029_1915/replay/sim_<case>_<timestamp>/
with the real ModelSim SE-64 2020.4 tools:
  D:/modelsim/win64/vlib.exe work
  D:/modelsim/win64/vlog.exe -sv -work work <rtl sources from source_manifest> <tb>
  D:/modelsim/win64/vsim.exe -c -work work <tb> -do "run -all; quit" -l vsim.log +VECDIR=<case dir>

Result: PASS/FAIL parsed from the TB's "NLP_RESULT case=... pass=..." line, JSON report at
build/spikformer/nlp_48h_20261029_1915/reports/replay_<case>.json, nonzero exit on failure.
No pass is claimed from file existence: only a parsed executed-simulation result passes.

Heavy discipline (48H handoff / Lead rule): a simulation expected to run >60 s must run
under the ONE shared heavy lock via scripts/spikformer/run_heavy.py --category long_sim.
The target e2e sim defaults to heavy (auto). Lock busy -> adapter refuses to start the
tool; this driver reports blocked and exits nonzero instead of running unsynchronised.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(REPO, 'build', 'spikformer', 'nlp_48h_20261029_1915')
REPLAY_DIR = os.path.join(RUN_DIR, 'replay')
REPORTS_DIR = os.path.join(RUN_DIR, 'reports')
MANIFEST = os.path.join(REPO, 'rtl', 'ops', 'spiking', 'source_manifest.json')
MS = 'D:/modelsim/win64'
HEAVY_LOCK = ('D:/dl_v0.1/build/onnx_e1/v02_real_yolo_20261026_2022_b1/jobs/'
              'heavy.lock.json')
RUN_HEAVY = os.path.join(REPO, 'scripts', 'spikformer', 'run_heavy.py')
GEN_SCRIPT = {'small': 'gen_nlp_small.py', 'multihead': 'gen_nlp_multihead.py',
              'e2e': 'gen_nlp_e2e.py', 'real': 'gen_nlp_e2e.py'}
TB_MODULE = {'small': 'tb_nlp_small', 'multihead': 'tb_nlp_multihead',
             'e2e': 'tb_nlp_e2e', 'real': 'tb_nlp_e2e'}
TB_FILE = {'small': 'verification/sim/spikformer/tb_nlp_small.v',
           'multihead': 'verification/sim/spikformer/tb_nlp_multihead.v',
           'e2e': 'verification/sim/spikformer/tb_nlp_e2e.v',
           'real': 'verification/sim/spikformer/tb_nlp_e2e.v'}
# SPS image chain / platform wrapper are not part of the NLP datapath; compiling
# them would only add unrelated elaboration surface.
SRC_EXCLUDE = ('/sps/', '/wrappers/')

RESULT_RE = re.compile(r'NLP_RESULT\s+case=(\S+)\s+pass=(\d+)\s+cycles=(\d+)')
FAIL_RE = re.compile(r'NLP_FAIL')
MS_VERSION_RE = re.compile(r'vsim\s+(\S+)\s+Simulator\s+(\S+)')


def _p(*parts):
    return os.path.join(*parts)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rtl_sources():
    """RTL source list for the NLP chain from the checked-in source manifest."""
    with open(MANIFEST, 'r', encoding='utf-8') as f:
        man = json.load(f)
    out = []
    for rel in man.get('sources', []):
        relu = rel.replace('\\', '/')
        if relu.endswith('.vh'):
            continue
        if any(x in relu for x in SRC_EXCLUDE):
            continue
        p = os.path.join(REPO, rel.replace('/', os.sep))
        if not os.path.isfile(p):
            raise FileNotFoundError('manifest source missing: ' + p)
        out.append(p)
    if not out:
        raise RuntimeError('empty RTL source list from manifest')
    return out


def _exec(cmd, cwd, log_path, timeout, append=False):
    mode = 'a' if append else 'w'
    t0 = time.time()
    with open(log_path, mode, encoding='utf-8', errors='replace') as lf:
        lf.write('\n$ ' + ' '.join(cmd) + '\n')
        lf.flush()
        try:
            r = subprocess_run(cmd, cwd, timeout)
            out = r.stdout.decode('utf-8', 'replace')
            lf.write(out)
            if r.stderr:
                lf.write(r.stderr.decode('utf-8', 'replace'))
            return r.returncode, out, time.time() - t0
        except TimeoutExpired as exc:
            out = (exc.stdout or b'').decode('utf-8', 'replace')
            lf.write(out)
            lf.write('\n[driver] TIMEOUT after %s s\n' % timeout)
            return 124, out, time.time() - t0


# imported lazily so --compile-only style smoke tests work even if subprocess shims
try:
    from subprocess import run as _sub_run, TimeoutExpired as TimeoutExpired
except Exception:  # pragma: no cover
    _sub_run = None
    TimeoutExpired = Exception


def subprocess_run(cmd, cwd, timeout):
    import subprocess
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, timeout=timeout)


def _opt(name):
    return os.path.join(MS, name + '.exe')


def sim_dir_name(case, tag=None):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = ('_' + tag) if tag else ''
    return _p(REPLAY_DIR, 'sim_%s_%s%s' % (case, ts, suffix))


def case_vectors(case):
    return _p(REPLAY_DIR, 'case_' + case)


def golden_hash(case):
    gp = _p(case_vectors(case), 'golden.json')
    if not os.path.isfile(gp):
        return None
    return _sha256(gp)


def regenerate(case):
    if case == 'real':
        raise SystemExit('[driver] case real needs external vectors: run '
                         'gen_nlp_e2e.py --checkpoint <npz> --tokens-file <samples> '
                         '[--vocab <vocab.json>] first')
    gen = _p(REPO, 'scripts', 'spikformer', GEN_SCRIPT[case])
    print('[driver] regenerating golden vectors:', gen)
    import subprocess
    r = subprocess.run([sys.executable, gen], cwd=REPO)
    if r.returncode != 0:
        raise SystemExit('[driver] golden generation failed rc=%d' % r.returncode)
    return golden_hash(case)


def modelsim_version():
    try:
        import subprocess
        r = subprocess.run([_opt('vsim'), '-version'], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=60)
        m = MS_VERSION_RE.search(r.stdout.decode('utf-8', 'replace'))
        if m:
            return 'vsim %s Simulator %s' % (m.group(1), m.group(2))
        return r.stdout.decode('utf-8', 'replace').strip().splitlines()[0]
    except Exception as exc:
        return 'unavailable: %r' % (exc,)


def _write_report(case, report):
    os.makedirs(REPORTS_DIR, exist_ok=True)
    rp = _p(REPORTS_DIR, 'replay_%s.json' % case)
    with open(rp, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(report, f, indent=1)
        f.write('\n')
    sd = report.get('sim_dir')
    if sd and os.path.isdir(sd):
        with open(_p(sd, 'result.json'), 'w', encoding='utf-8', newline='\n') as f:
            json.dump(report, f, indent=1)
            f.write('\n')
    return rp


def run_inner(args):
    case = args.case
    os.makedirs(REPLAY_DIR, exist_ok=True)
    sim_dir = args.sim_dir or sim_dir_name(case, args.tag)
    os.makedirs(sim_dir, exist_ok=True)
    log = _p(sim_dir, 'run.log')
    vecdir = case_vectors(case).replace('\\', '/')
    tb = TB_MODULE[case]
    rep = {
        'schema': 'spikformer-nlp-replay-report-1',
        'case': case,
        'tb': tb,
        'tb_file': TB_FILE[case],
        'vecdir': vecdir,
        'sim_dir': sim_dir,
        'log': log,
        'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'modelsim': modelsim_version(),
        'heavy_lock': {'used': bool(args.inner), 'lock': HEAVY_LOCK},
        'golden_sha256': golden_hash(case),
        'commands': [],
        'steps': [],
        'pass': False,
        'result': 'error',
    }
    if args.gen and not args.inner:
        rep['golden_sha256'] = regenerate(case)
    if not os.path.isfile(os.path.join(vecdir, 'golden.json')):
        rep['error'] = ('golden vectors missing at %s; run gen_nlp_%s.py first'
                        % (vecdir, case))
        _write_report(case, rep)
        print('[driver] ERROR:', rep['error'])
        return 2
    try:
        sources = rtl_sources()
    except Exception as exc:
        rep['error'] = 'RTL source list: %r' % (exc,)
        _write_report(case, rep)
        print('[driver] ERROR:', rep['error'])
        return 2
    tb_full = _p(REPO, TB_FILE[case])
    src_hashes = []
    for sp in sources + [tb_full]:
        try:
            src_hashes.append({'path': os.path.relpath(sp, REPO).replace('\\', '/'),
                               'sha256': _sha256(sp)})
        except Exception:
            src_hashes.append({'path': sp, 'sha256': None})
    rep['rtl_sources'] = src_hashes
    rep['rtl_source_manifest_sha256'] = _sha256(MANIFEST)
    rep['tb_sha256'] = _sha256(tb_full)
    rc_vlib, _o, _t = _exec([_opt('vlib'), 'work'], sim_dir, log, 300)
    rep['steps'].append({'step': 'vlib', 'exit': rc_vlib})
    if rc_vlib != 0:
        rep['error'] = 'vlib failed'
        _write_report(case, rep)
        print('[driver] ERROR: vlib failed, see', log)
        return 2
    vlog_cmd = [_opt('vlog'), '-sv', '-work', 'work'] + sources + [_p(REPO, TB_FILE[case])]
    rc_vlog, out, _t = _exec(vlog_cmd, sim_dir, log, 1800, append=True)
    errs = len(re.findall(r'^\*\* (Error|Fatal)', out, re.M))
    warns = len(re.findall(r'^\*\* Warning', out, re.M))
    rep['steps'].append({'step': 'vlog', 'exit': rc_vlog, 'errors': errs, 'warnings': warns})
    if rc_vlog != 0 or errs:
        rep['error'] = 'compile/elaboration failed (errors=%d)' % errs
        _write_report(case, rep)
        print('[driver] FAIL compile: errors=%d exit=%d log=%s' % (errs, rc_vlog, log))
        print(out[-4000:])
        return 1
    if args.elaborate_only:
        elab_cmd = [_opt('vsim'), '-c', '-work', 'work', tb, '-do', 'quit',
                    '-l', 'elab.log']
        rc_e, out_e, _t = _exec(elab_cmd, sim_dir, log, args.timeout, append=True)
        bad = ('Error loading design' in out_e) or len(re.findall(r'^\*\* (Error|Fatal)', out_e, re.M))
        rep['steps'].append({'step': 'elaborate', 'exit': rc_e, 'ok': not bad})
        if rc_e != 0 or bad:
            rep['error'] = 'elaboration/vopt failed'
            _write_report(case, rep)
            print('[driver] FAIL elaborate: exit=%d log=%s' % (rc_e, log))
            for ln in out_e.splitlines():
                if '** Error' in ln or 'Error loading' in ln:
                    print('  | ' + ln)
            return 1
        rep['result'] = 'elaborate_only'
        rep['pass'] = True
        rep['finished_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        _write_report(case, rep)
        print('[driver] elaborate-only OK: sources=%d warnings=%d log=%s'
              % (len(sources), warns, log))
        return 0
    if args.compile_only:
        rep['result'] = 'compile_only'
        rep['pass'] = True
        rep['finished_at'] = datetime.datetime.now().isoformat(timespec='seconds')
        _write_report(case, rep)
        print('[driver] compile-only OK: sources=%d warnings=%d log=%s'
              % (len(sources), warns, log))
        return 0
    vsim_cmd = [_opt('vsim'), '-c', '-work', 'work', tb,
                '-do', 'run -all; quit', '-l', 'vsim.log', '+VECDIR=' + vecdir]
    if args.case == 'real':
        ns = args.samples
        if ns is None:
            with open(os.path.join(vecdir, 'golden.json'), 'r', encoding='utf-8') as f:
                ns = int(json.load(f).get('n_samples', 2))
        vsim_cmd.append('+NLP_NS=%d' % int(ns))
    if args.trace:
        vsim_cmd.append('+NLP_TRACE=1')
    for pa in args.plusarg:
        vsim_cmd.append(pa if pa.startswith('+') else '+' + pa)
    t0 = time.time()
    rc_vsim, out, wall = _exec(vsim_cmd, sim_dir, log, args.timeout, append=True)
    simlog = _p(sim_dir, 'vsim.log')
    if os.path.isfile(simlog):
        with open(simlog, 'r', encoding='utf-8', errors='replace') as f:
            out = out + '\n' + f.read()
    rep['wall_sec'] = round(wall, 2)
    m = RESULT_RE.search(out)
    nfail = len(FAIL_RE.findall(out))
    rep['steps'].append({'step': 'vsim', 'exit': rc_vsim, 'wall_sec': round(wall, 2),
                         'result_line': m.group(0) if m else None,
                         'fail_lines': nfail})
    if m:
        rep['pass'] = (m.group(2) == '1') and rc_vsim == 0
        rep['cycles'] = int(m.group(3))
        rep['result'] = 'pass' if rep['pass'] else 'fail'
    else:
        rep['pass'] = False
        rep['result'] = 'no_result_line'
        rep['error'] = ('simulation produced no NLP_RESULT line (exit=%d); a missing '
                        'result is a failure, not a pass' % rc_vsim)
    rep['finished_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    # long-sim policy check: >60 s without the shared lock is not acceptable evidence
    if wall > 60 and not args.inner:
        rep['pass'] = False
        rep['result'] = 'policy_violation_long_sim_without_lock'
        rep['error'] = ('simulation took %.1f s > 60 s without the shared heavy lock; '
                        'rerun with --heavy on' % wall)
    _write_report(case, rep)
    tail = [ln for ln in out.splitlines() if ('NLP_' in ln or 'FAIL' in ln
            or 'MISMATCH' in ln or 'timeout' in ln.lower())]
    print('[driver] case=%s %s cycles=%s wall=%.1fs' %
          (case, rep['result'].upper(), rep.get('cycles'), wall))
    for ln in tail[-25:]:
        print('  | ' + ln)
    print('[driver] log:', log)
    print('[driver] report:', _p(REPORTS_DIR, 'replay_%s.json' % case))
    return 0 if rep['pass'] else 1


def delegate_heavy(args):
    """Run the actual simulation inside the ONE shared heavy lock (long sim)."""
    case = args.case
    os.makedirs(REPLAY_DIR, exist_ok=True)
    sim_dir = sim_dir_name(case, args.tag)
    os.makedirs(sim_dir, exist_ok=True)
    job_id = 'nlp-replay-%s-%s' % (case, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    outer_log = _p(sim_dir, 'heavy_outer.log')
    if args.gen:
        regenerate(case)
    cmd = [sys.executable, RUN_HEAVY,
           '--lock', HEAVY_LOCK,
           '--job-id', job_id,
           '--category', 'long_sim',
           '--wait-sec', str(args.wait_sec),
           '--',
           sys.executable, os.path.abspath(__file__),
           '--case', case, '--inner', '--sim-dir', sim_dir,
           '--timeout', str(args.timeout)]
    if args.trace:
        cmd.append('--trace')
    for pa in args.plusarg:
        cmd.extend(['--plusarg', pa])
    print('[driver] long sim -> shared heavy lock (%s)' % HEAVY_LOCK)
    print('[driver] outer command:', ' '.join(cmd))
    import subprocess
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = r.stdout.decode('utf-8', 'replace')
    with open(outer_log, 'w', encoding='utf-8', newline='\n') as f:
        f.write('$ ' + ' '.join(cmd) + '\n')
        f.write(out)
    print(out[-6000:])
    print('[driver] heavy wrapper exit=%d wall=%.1fs log=%s'
          % (r.returncode, time.time() - t0, outer_log))
    if 'refusing to start the tool without a lease' in out or 'NOT ACQUIRED' in out:
        print('[driver] BLOCKED: shared heavy lock not acquired; nothing was simulated')
        return 3
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, choices=sorted(TB_MODULE))
    ap.add_argument('--gen', action='store_true',
                    help='regenerate golden vectors before running')
    ap.add_argument('--compile-only', action='store_true',
                    help='fresh compile only (vlog, no simulation result)')
    ap.add_argument('--elaborate-only', action='store_true',
                    help='fresh compile + vopt elaboration check, then quit (no simulation)')
    ap.add_argument('--trace', action='store_true', help='pass +NLP_TRACE=1 to the TB')
    ap.add_argument('--plusarg', action='append', default=[],
                    help='extra +plusarg passed to vsim (repeatable)')
    ap.add_argument('--heavy', choices=['auto', 'on', 'off'], default='auto',
                    help='use the shared heavy lock (auto: on for e2e target sim)')
    ap.add_argument('--wait-sec', type=int, default=600,
                    help='seconds to wait for the shared heavy lock (0 = fail fast)')
    ap.add_argument('--timeout', type=int, default=3600,
                    help='external vsim process timeout in seconds')
    ap.add_argument('--tag', default=None, help='suffix for the fresh sim dir name')
    ap.add_argument('--samples', type=int, default=None,
                    help='real case: number of fixed samples to replay (1-3, default from golden)')
    ap.add_argument('--inner', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--sim-dir', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    heavy = {'auto': args.case in ('e2e', 'real'), 'on': True, 'off': False}[args.heavy]
    if heavy and not args.inner and not args.compile_only and not args.elaborate_only:
        return delegate_heavy(args)
    return run_inner(args)


if __name__ == '__main__':
    sys.exit(main())