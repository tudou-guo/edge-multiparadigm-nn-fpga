"""Pure-NumPy integer-domain threshold calibration (golden integer_ref traces).

Frozen rule (Lead): ONE rule per LIF, no percentile grid:
    thr = max(1, round(median of positive membrane values))
The membrane value is exactly the contract value compared against the threshold
(mem = floor((pre+delta)/2) BEFORE the hard reset), collected from the exact integer
traces of tools/backends/spikformer/integer_ref.py. After calibration every LIF must
fire a nonzero number of spikes on the calibration set (dead LIF => hard error), and
thr >= 1 is asserted. If thr would exceed the largest positive pre-activation (delta)
the LIF can never fire, recorded as a warning.

No torch import: this module runs in the pure-NumPy export process (torch and numpy
OpenMP runtimes clash on this host when integer matmuls run after torch).
"""
from __future__ import annotations

import numpy as np

from tools.backends.spikformer import integer_ref as IREF

LIF_NAMES = ('q', 'k', 'v', 'proj', 'fc1', 'fc2')


def positive(a):
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    return a[a > 0]


def calib_rule(pos_mem, pos_delta_max, name, report):
    """ONE rule per LIF: thr = max(1, round(median(positive mem)))."""
    if pos_mem.size == 0:
        report['warn'].append('%s: no positive mem values; thr=1' % name)
        report['stats'][name] = {'n_pos_mem': 0, 'thr': 1}
        return 1
    med = float(np.median(pos_mem))
    thr = max(1, int(round(med)))
    report['stats'][name] = {'n_pos_mem': int(pos_mem.size), 'median_pos_mem': med,
                             'max_pos_mem': int(pos_mem.max()),
                             'max_pos_delta': int(pos_delta_max), 'thr': thr}
    if thr > int(pos_delta_max):
        report['warn'].append('%s: thr=%d > max positive delta=%d (can never fire)'
                              % (name, thr, int(pos_delta_max)))
    return thr


def collect_lif_activity(params, ids, cfg, chunk=16):
    """Per-LIF firing rate over ids using exact integer_ref traces."""
    ids = np.asarray(ids, dtype=np.int64)
    rates = {}
    for i in range(0, len(ids), chunk):
        tr = IREF.run_sample(params, ids[i:i + chunk], cfg=cfg, trace=True)
        rates.setdefault('embed', []).append(float(np.mean(tr['embed_spikes'])))
        for b, btr in enumerate(tr['blocks']):
            for nm in LIF_NAMES:
                rates.setdefault('block%d.%s' % (b, nm), []).append(
                    float(np.mean(btr[nm + '_spikes'])))
    return {k: float(np.mean(v)) for k, v in rates.items()}


def calibrate_thresholds(params, ids, cfg, rounds=2, chunk=16):
    """Integer-domain threshold calibration on golden integer_ref traces.

    Mutates params thr_e and per-block thresholds in place; returns the report dict
    (thresholds per round, stats, per-LIF activity, dead-LIF check).
    """
    report = {'rule': 'median of positive mem (integer domain), clamp >= 1',
              'warn': [], 'stats': {}, 'rounds': []}
    ids = np.asarray(ids, dtype=np.int64)
    for rnd in range(rounds):
        acc = {}
        for i in range(0, len(ids), chunk):
            tr = IREF.run_sample(params, ids[i:i + chunk], cfg=cfg, trace=True)
            acc.setdefault('e', []).append(np.asarray(tr['embed']['mem']).reshape(-1))
            acc.setdefault('e_d', []).append(int(np.max(np.asarray(tr['embed']['delta']))))
            for b, btr in enumerate(tr['blocks']):
                for nm in LIF_NAMES:
                    key = 'block%d.%s' % (b, nm)
                    acc.setdefault(key, []).append(np.asarray(btr[nm + '_mem']).reshape(-1))
                    acc.setdefault(key + '_d', []).append(
                        int(np.max(np.asarray(btr[nm + '_delta']))))
        new = {'e': calib_rule(positive(np.concatenate(acc['e'])), max(acc['e_d']),
                               'embed', report)}
        for b in range(cfg['D']):
            for nm in LIF_NAMES:
                key = 'block%d.%s' % (b, nm)
                new[key] = calib_rule(positive(np.concatenate(acc[key])),
                                      max(acc[key + '_d']), key, report)
        params['thr_e'] = new['e']
        for b in range(cfg['D']):
            for nm in LIF_NAMES:
                params['blocks'][b][nm + '_thr'] = new['block%d.%s' % (b, nm)]
        report['rounds'].append({'thresholds': dict(new),
                                 'activity': collect_lif_activity(
                                     params, ids[:min(len(ids), 32)], cfg)})
    act = collect_lif_activity(params, ids, cfg)
    report['final_activity'] = act
    report['min_activity'] = min(act.values())
    report['dead_lifs'] = [k for k, v in act.items() if v <= 0.0]
    thr_all = [params['thr_e']] + [params['blocks'][b][nm + '_thr']
                                   for b in range(cfg['D']) for nm in LIF_NAMES]
    if min(thr_all) < 1:
        raise RuntimeError('calibration produced thr < 1: %r' % thr_all)
    if report['dead_lifs']:
        raise RuntimeError('calibration produced dead LIFs (zero spikes on the %d-sample '
                           'calibration set): %s' % (len(ids), report['dead_lifs']))
    report['thresholds'] = {'thr_e': int(params['thr_e'])}
    for b in range(cfg['D']):
        report['thresholds']['block%d' % b] = {nm: int(params['blocks'][b][nm + '_thr'])
                                               for nm in LIF_NAMES}
    return report
