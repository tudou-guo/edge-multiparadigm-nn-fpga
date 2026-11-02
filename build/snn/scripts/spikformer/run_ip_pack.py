"""IP source package for the Spikformer NLP 48h run.

Same-source rule (handoff): the packaged source copy must SHA-match the sources
used by simulation (manifest rtl_sha256 + generated adapter/config/xdc/sources.f).
We never claim a routed/IP-qualified accelerator just because a memory-IP Tcl was
generated. Timing qualification is separate (OOC report); bit/HWH only if real
files exist.

Usage:
  python -B scripts/spikformer/run_ip_pack.py --manifest <run>/manifest.json \
      --out-dir <run>/ip [--ooc-report <run>/ooc/reports/ooc_*.json] [--force]
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, sys, time

ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[2]).resolve()


def _sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--ooc-report', default=None)
    ap.add_argument('--bit', default=None)
    ap.add_argument('--hwh', default=None)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args(argv)
    man_path = pathlib.Path(a.manifest).resolve()
    man = json.loads(man_path.read_text(encoding='utf-8-sig'))
    run_dir = man_path.parent
    out = pathlib.Path(a.out_dir)
    if out.exists() and any(out.iterdir()) and not a.force:
        print('refusing to overwrite non-empty IP dir: %s (use --force)' % out)
        return 3
    if out.exists() and a.force:
        shutil.rmtree(out)
    (out / 'src').mkdir(parents=True, exist_ok=True)
    (out / 'generated').mkdir(parents=True, exist_ok=True)
    (out / 'params').mkdir(parents=True, exist_ok=True)
    ledger = {}

    # 1) RTL sources exactly as bound by the manifest
    srcs = man.get('rtl_sources') or []
    recorded = man.get('rtl_sha256') or {}
    bad = []
    for rel in srcs:
        src = ROOT / rel
        if not src.is_file():
            bad.append('%s missing' % rel)
            continue
        got = _sha(src)
        if recorded.get(rel) != got:
            bad.append('%s sha mismatch (manifest %s)' % (rel, recorded.get(rel)))
        dst = out / 'src' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        ledger[rel] = {'sha256': got, 'bytes': dst.stat().st_size}
    gen = man.get('generated') or {}
    gen_files = ['header', 'adapter', 'sources_f', 'xdc', 'ooc_tcl', 'project_tcl']
    for key in gen_files:
        rel = gen.get(key)
        if not rel:
            continue
        src = run_dir / rel
        if not src.is_file():
            bad.append('generated %s missing: %s' % (key, src))
            continue
        dst = out / 'generated' / pathlib.Path(rel).name
        shutil.copyfile(src, dst)
        ledger[rel] = {'sha256': _sha(src), 'bytes': dst.stat().st_size}
    # 2) params + manifest/numeric contract
    copied = 0
    for name, meta in (man.get('params') or {}).items():
        s = run_dir / 'params' / meta['bin']
        if s.is_file():
            shutil.copyfile(s, out / 'params' / meta['bin'])
            copied += 1
            ledger['params/' + meta['bin']] = {'sha256': _sha(s), 'bytes': s.stat().st_size}
    shutil.copyfile(man_path, out / 'manifest.json')
    for extra in ('numeric_contract.json', 'resolved_config.json', 'schedule.json'):
        p = run_dir / extra
        if p.is_file():
            shutil.copyfile(p, out / extra)
            ledger[extra] = {'sha256': _sha(p), 'bytes': p.stat().st_size}
    contract = run_dir.parent / run_dir.name / 'contract.json'
    # contract.json lives at run root; run_dir == run root here
    cpath = run_dir / 'contract.json'
    if cpath.is_file():
        shutil.copyfile(cpath, out / 'contract.json')
        ledger['contract.json'] = {'sha256': _sha(cpath), 'bytes': cpath.stat().st_size}
    # 3) optional real bit/HWH (never fabricated)
    hw = {}
    for tag, p in (('bit', a.bit), ('hwh', a.hwh)):
        if p and pathlib.Path(p).is_file():
            shutil.copyfile(p, out / pathlib.Path(p).name)
            hw[tag] = {'file': pathlib.Path(p).name, 'sha256': _sha(p)}
            ledger[pathlib.Path(p).name] = {'sha256': _sha(p),
                                            'bytes': pathlib.Path(p).stat().st_size}
        else:
            hw[tag] = None
    # 4) OOC timing qualification (separate from packaging)
    ooc = None
    if a.ooc_report and pathlib.Path(a.ooc_report).is_file():
        ooc = json.loads(pathlib.Path(a.ooc_report).read_text())
        shutil.copyfile(a.ooc_report, out / 'ooc_report.json')
        ledger['ooc_report.json'] = {'sha256': _sha(a.ooc_report),
                                     'bytes': pathlib.Path(a.ooc_report).stat().st_size}
    target = man.get('target') or {}
    ip_manifest = {
        'schema': 'spikformer-nlp-ip-1',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'run_dir': str(run_dir),
        'model_kind': man.get('model_kind'),
        'manifest_sha256': man.get('manifest_sha256'),
        'source_manifest_sha256_check': {rel: ('match' if (recorded.get(rel) == ledger.get(rel, {}).get('sha256')) else 'mismatch')
                                         for rel in srcs},
        'source_hashes_match_sim': not bad,
        'source_mismatch': bad,
        'target': target,
        'part': target.get('part'), 'clock': '5.000 ns (200 MHz) s_clk',
        'reset': 's_rst active-high asynchronous reset (FSM/valid only; big RAMs not element-reset)',
        'param_interface': {
            'tables': 'i_tab_data[63:0]/i_tab_valid after i_start (etab then ptab, 64-bit words)',
            'tokens': 'i_tok_id[15:0]/i_tok_valid (N beats)',
            'weights': 'i_w_data[63:0]/i_w_valid, per block: 4 descriptor beats + 25152 weight beats',
            'outputs': 'o_sums[128] uint32 + o_sums_valid + o_done',
            'thresholds': 'runtime parameter header (descriptor beats), not compile-time constants'},
        'structure': man.get('derived'),
        'generated_files': {k: gen.get(k) for k in gen_files},
        'hardware_artifacts': hw,
        'timing_qualified': bool(ooc and ooc.get('status') == 'pass'),
        'ooc_status': (ooc or {}).get('status'),
        'assumptions': [
            'OOC 5.000 ns sign-off only (xczu15eg-ffvb1156-2-i, Vivado 2021.1, s_clk 5.000 ns, no timing exceptions).',
            'The generated spikformer.xdc models every data input with set_input_delay 0.000 and has HD.CLK_SRC unset: '
            'hold is analysed under a pessimistic port-launch model, setup under an optimistic (zero) upstream-delay model.',
            'The generated target adapter contains 1,192 DONT_TOUCH LUT1 input hold-margin cells (8 stages on 149 '
            'sampled input bits, s_rst included). They are a required part of this netlist; keep them in integration.',
            'Routed WNS binding path is the hold-margin chain itself (from-port setup slack +0.135 ns in the OOC '
            '0-delay model). A real upstream max input delay greater than 0.135 ns would violate it: re-time or reduce '
            'the chain, or add an integrated-top register slice, and re-constrain with real interface min/max before '
            'platform integration. This package is not integration-ready timing sign-off.',
            'Not integrated-platform, bit/HWH or board timing; those remain not_run.',
        ],
        'package_sha256': None,
    }
    files_ledger = '\n'.join('%s  %s' % (v['sha256'], k) for k, v in sorted(ledger.items())) + '\n'
    (out / 'SOURCE_SHA256.txt').write_text(files_ledger)
    ip_manifest['package_files'] = ledger
    digest = hashlib.sha256((json.dumps(ledger, sort_keys=True) + files_ledger).encode()).hexdigest()
    ip_manifest['package_sha256'] = digest
    (out / 'ip_manifest.json').write_text(json.dumps(ip_manifest, indent=1))
    readme = ('Spikformer NLP IP source package (48h run).\n'
              'same-source: %s (mismatches=%d)\n'
              'part=%s clock=%s\n'
              'timing_qualified=%s (ooc status=%s)\n'
              'bit=%r hwh=%r\n'
              'package_sha256=%s\n'
              '\nSCOPE / ASSUMPTIONS (read before using this package)\n'
              '- OOC 5.000 ns sign-off only (xczu15eg-ffvb1156-2-i, Vivado 2021.1, no timing exceptions).\n'
              '- Input delay model is 0.000 ns and HD.CLK_SRC is unset: hold is pessimistic, setup is optimistic.\n'
              '- Keep the 1,192 DONT_TOUCH LUT1 hold-margin cells (8 stages x 149 sampled input bits incl. s_rst).\n'
              '- Routed WNS binding path is the hold-margin chain itself (from-port setup slack +0.135 ns under the\n'
              '  0-delay model); a real upstream max input delay greater than 0.135 ns would violate it - re-constrain\n'
              '  with real interface min/max and re-time/reduce the chain or register-slice at the integrated top\n'
              '  before platform use. Not integration-ready timing sign-off.\n'
              '- platform / bit / HWH / board = not_run.\n'
              % (ip_manifest['source_hashes_match_sim'], len(bad), target.get('part'),
                 ip_manifest['clock'], ip_manifest['timing_qualified'],
                 ip_manifest['ooc_status'], hw.get('bit'), hw.get('hwh'), digest))
    (out / 'README.txt').write_text(readme)
    status = 'pass' if (not bad and copied == len(man.get('params') or {})) else 'fail'
    print(json.dumps({'status': status, 'out': str(out), 'sources': len(srcs),
                      'params_copied': copied, 'mismatch': bad,
                      'timing_qualified': ip_manifest['timing_qualified'],
                      'package_sha256': digest}, indent=1))
    return 0 if status == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())