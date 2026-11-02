"""Physical build driver for the Spikformer NLP backend (48h run).

Thin, real entry that:
  * reads the compile manifest, locates the generated Tcl/adapter/sources;
  * verifies the source hashes in the manifest still match (frozen-source check);
  * queues (default) or executes the stage through the SHARED heavy lock via
    scripts/spikformer/run_heavy.py (never a private per-run lock, never two
    heavy tasks, never kills anything);
  * records exit code, completion marker, WNS/hold, unrouted nets, latch count,
    utilization, DRC log and the full log path into a JSON report.

Usage:
  python -B scripts/spikformer/run_physical.py --manifest <run>/manifest.json \
      --stage synth|ooc [--execute] [--wait-sec N]
No Vivado is started unless --execute is given AND the lease is acquired.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()
VIVADO = pathlib.Path('E:/Vitis/Vivado/2021.1/bin/vivado.bat')
HEAVY = ROOT / 'scripts/spikformer/run_heavy.py'
DEFAULT_LOCK = pathlib.Path(
    'D:/dl_v0.1/build/onnx_e1/v02_real_yolo_20261026_2022_b1/jobs/heavy.lock.json')
PY = sys.executable


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _parse_facts(out_dir):
    facts = {}
    fp = pathlib.Path(out_dir) / 'facts.txt'
    if fp.is_file():
        for line in fp.read_text(encoding='utf-8', errors='replace').splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                facts[k.strip()] = v.strip()
    return facts


def _parse_util(path):
    util = {}
    p = pathlib.Path(path)
    if not p.is_file():
        return util
    pat = re.compile(r'^\|\s*([^|]+?)\s*\|\s*([0-9.]+)\s*\|')
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        m = pat.match(line)
        if m:
            util.setdefault(m.group(1).strip(), m.group(2).strip())
    return util


def _count_in_file(path, patterns):
    hits = {}
    p = pathlib.Path(path)
    if not p.is_file():
        return hits
    text = p.read_text(encoding='utf-8', errors='replace')
    for name, rx in patterns.items():
        hits[name] = len(re.findall(rx, text, flags=re.IGNORECASE))
    return hits


def _verify_sources(man):
    srcs = man.get('rtl_sources') or []
    hashes = man.get('rtl_sha256') or {}
    bad = []
    for s in srcs:
        want = hashes.get(s)
        if not want:
            bad.append((s, 'no recorded hash'))
            continue
        got = _sha(ROOT / s)
        if got != want:
            bad.append((s, 'sha mismatch'))
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--stage', required=True, choices=['synth', 'ooc'])
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--lock', default=str(DEFAULT_LOCK))
    ap.add_argument('--wait-sec', type=int, default=0)
    ap.add_argument('--job-id', default=None)
    ap.add_argument('--category', default=None, help='heavy category; default vivado_l0 for synth, vivado_ooc for ooc')
    ap.add_argument('--mem-mib', type=int, default=None, help='explicit free-memory reservation for the heavy lease')
    ap.add_argument('--skip-source-check', action='store_true')
    a = ap.parse_args(argv)
    man_path = pathlib.Path(a.manifest).resolve()
    man = json.loads(man_path.read_text(encoding='utf-8-sig'))
    out_dir = man_path.parent
    gen = man.get('generated') or {}
    tcl = out_dir / gen.get('ooc_tcl', 'generated/spikformer_ooc.tcl')
    if not tcl.is_file():
        print('MISSING OOC TCL: ' + str(tcl))
        return 3
    if not a.skip_source_check:
        bad = _verify_sources(man)
        if bad:
            print('SOURCE FROZEN CHECK FAILED (%d mismatches):' % len(bad))
            for s, why in bad[:10]:
                print('  %s: %s' % (s, why))
            print('regenerate the manifest/affected build instead of synthesizing stale sources')
            return 4
        print('source frozen check: %d files match manifest hashes' % len(man.get('rtl_sources') or []))
    category = a.category or ('vivado_l0' if a.stage == 'synth' else 'vivado_ooc')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    job_id = a.job_id or ('nlp48h-%s-%s' % (a.stage, stamp))
    log_dir = out_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ('%s_%s.log' % (a.stage, stamp))
    report_path = out_dir / 'reports' / ('%s_%s.json' % (a.stage, stamp))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / ('%s_run_%s' % (a.stage, stamp))
    viv_cmd = [str(VIVADO), '-mode', 'batch', '-nojournal',
               '-log', str(log_path), '-source', str(tcl),
               '-tclargs', str(run_dir), a.stage]
    adapter = out_dir / gen.get('adapter', 'generated/spikformer_nlp_target_top.sv')
    adapter_sha = _sha(adapter) if adapter.is_file() else None
    if not a.execute:
        print('QUEUED (no heavy start). Shared lock required to run:')
        print('  python -B %s --lock %s --job-id %s --category %s --wait-sec %d -- %s'
              % (HEAVY, a.lock, job_id, category, a.wait_sec, ' '.join(viv_cmd)))
        print('then re-run this driver with --execute to get the parsed report.')
        rec = {'stage': a.stage, 'executed': False, 'reason': 'queued without --execute',
               'queued_command': viv_cmd, 'lock': a.lock, 'job_id': job_id,
               'category': category, 'mem_mib': a.mem_mib,
               'tcl': str(tcl), 'adapter_sha256': adapter_sha,
               'manifest_sha256': man.get('manifest_sha256')}
        report_path.write_text(json.dumps(rec, indent=1))
        print('queued report: ' + str(report_path))
        return 0
    runner = [PY, '-B', str(HEAVY), '--lock', a.lock, '--job-id', job_id,
              '--category', category, '--wait-sec', str(a.wait_sec)]
    if a.mem_mib is not None:
        runner += ['--mem-mib', str(a.mem_mib)]
    runner += ['--'] + viv_cmd
    t0 = time.time()
    with log_path.open('w', encoding='utf-8', errors='replace') as fh:
        proc = subprocess.run(runner, stdout=fh, stderr=subprocess.STDOUT)
    wall = round(time.time() - t0, 1)
    try:
        text = log_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        text = ''
    marker = ('OOC_DONE_OK' if a.stage == 'ooc' else 'SYNTH_DONE_OK') in text
    lock_refused = 'HEAVY LOCK NOT ACQUIRED' in text
    facts = _parse_facts(run_dir) or _parse_facts(out_dir)
    util = _parse_util(run_dir / ('util_%s.rpt' % ('route' if a.stage == 'ooc' else 'synth')))
    if not util:
        util = _parse_util(out_dir / ('util_%s.rpt' % ('route' if a.stage == 'ooc' else 'synth')))
    log_hits = _count_in_file(log_path, {
        'multi_driven': r'multi-driven|multiple drivers',
        'latch_inferred': r'LATCH',
        'critical_warnings': r'CRITICAL WARNING',
        'errors': r'\bERROR:',
    })
    try:
        wns = float(facts.get('wns_setup_ns')) if 'wns_setup_ns' in facts else None
    except ValueError:
        wns = None
    try:
        wns_hold = float(facts.get('wns_hold_ns')) if 'wns_hold_ns' in facts else None
    except ValueError:
        wns_hold = None
    try:
        unrouted = int(facts.get('unrouted_nets')) if 'unrouted_nets' in facts else None
    except ValueError:
        unrouted = None
    try:
        latches = int(facts.get('latch_cells')) if 'latch_cells' in facts else None
    except ValueError:
        latches = None
    status = 'not_run'
    reason = ''
    if proc.returncode != 0 or lock_refused:
        status = 'fail'
        reason = ('lock not acquired; tool not started' if lock_refused
                  else 'driver/tool exit code %d' % proc.returncode)
    elif not marker:
        status = 'fail'
        reason = 'completion marker missing from log (tool did not finish cleanly)'
    else:
        status = 'pass'
        problems = []
        if a.stage == 'ooc':
            if wns is None or wns < 0:
                problems.append('negative/missing setup WNS: %r' % wns)
            if wns_hold is None or wns_hold < 0:
                problems.append('negative/missing hold WNS: %r' % wns_hold)
            if unrouted is None or unrouted != 0:
                problems.append('unrouted nets: %r' % unrouted)
        if latches:
            problems.append('inferred latches: %d' % latches)
        if log_hits.get('multi_driven'):
            problems.append('multi-driven messages: %d' % log_hits['multi_driven'])
        if problems:
            status = 'fail'
            reason = '; '.join(problems)
    rec = {
        'stage': a.stage, 'executed': True, 'status': status, 'reason': reason,
        'job_id': job_id, 'lock': a.lock, 'category': category, 'mem_mib': a.mem_mib, 'started': time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(t0)),
        'wall_sec': wall, 'exit_code': proc.returncode,
        'completion_marker': marker, 'marker_text': ('OOC_DONE_OK' if a.stage == 'ooc' else 'SYNTH_DONE_OK'),
        'wns_setup_ns': wns, 'wns_hold_ns': wns_hold, 'unrouted_nets': unrouted,
        'latch_cells': latches, 'log_hits': log_hits, 'facts': facts,
        'utilization': util, 'tcl': str(tcl), 'log': str(log_path),
        'adapter_sha256': adapter_sha, 'manifest_sha256': man.get('manifest_sha256'),
        'target': man.get('target'), 'run_dir': str(run_dir), 'report_paths': {
            'timing': str(run_dir / ('timing_%s.rpt' % ('route' if a.stage == 'ooc' else 'synth'))),
            'utilization': str(run_dir / ('util_%s.rpt' % ('route' if a.stage == 'ooc' else 'synth'))),
            'drc': str(run_dir / ('drc_%s.rpt' % ('route' if a.stage == 'ooc' else 'synth'))),
            'route_status': str(run_dir / 'route_status.rpt') if a.stage == 'ooc' else None,
            'pulse_width': str(run_dir / 'pulse_width.rpt') if a.stage == 'ooc' else None,
            'clocks': str(run_dir / 'clocks.rpt') if a.stage == 'ooc' else None,
            'exceptions': str(run_dir / 'exceptions.rpt') if a.stage == 'ooc' else None,
        },
    }
    report_path.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ('stage', 'executed', 'status', 'reason', 'wns_setup_ns',
                                          'wns_hold_ns', 'unrouted_nets', 'wall_sec', 'log')},
                     indent=1))
    print('report: ' + str(report_path))
    return 0 if status == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())