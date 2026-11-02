"""Schedule: SPS once + per-block descriptors + memory map.
No block reuses block.0 weights: each block gets base+stride addresses.
Only the last block feeds head reduce.

48h NLP contract binding: each block descriptor carries ONLY that block's six
LIF thresholds (q,k,v,proj,fc1,fc2) plus the 4x64-bit hardware descriptor words
from contract.json hardware_abi.descriptor_beats_64b (Verilog-concat order:
beat0={thr_q,thr_k} hi/lo, beat1={thr_v,thr_proj}, beat2={thr_fc1,thr_fc2},
beat3={0,thr_e} for block 0). thr_e is a sample-level parameter echoed once.
"""
from __future__ import annotations

THR_KEYS = ('q', 'k', 'v', 'proj', 'fc1', 'fc2')


def _u32(v):
    return int(v) & 0xFFFFFFFF


def build(cfg, param_files):
    D = cfg['depth']
    T, E, MLP = cfg['time_steps'], cfg['embed'], cfg['mlp_hidden']
    tokens = cfg['derived']['tokens']
    stride = 0x100000  # per-block weight window stride (bytes), checked vs map
    thr = cfg.get('thresholds') or {}
    try:
        thr_e = int(thr['thr_e']) if 'thr_e' in thr else None
    except (TypeError, ValueError):
        raise ValueError('schedule: thr_e must be an int, got %r' % thr.get('thr_e'))
    descs = []
    addr = 0x70000000
    order = ['q_w', 'k_w', 'v_w', 'proj_w', 'fc1_w', 'fc2_w']
    for b in range(D):
        bt = dict(thr.get('block%d' % b) or {})
        missing = [k for k in THR_KEYS if k not in bt]
        if thr and missing:
            raise ValueError('schedule: block%d thresholds missing %r' % (b, missing))
        six = {k: int(bt[k]) for k in THR_KEYS} if bt else {}
        w0 = (_u32(six['q']) << 32) | _u32(six['k']) if six else 0
        w1 = (_u32(six['v']) << 32) | _u32(six['proj']) if six else 0
        w2 = (_u32(six['fc1']) << 32) | _u32(six['fc2']) if six else 0
        w3 = _u32(thr_e) if (six and b == 0 and thr_e is not None) else 0
        blk = {'block': b, 'first': b == 0, 'last': b == D - 1,
               'weights': {}, 'lif_thr': six,
               'descriptor_beats_64b': ['0x%016x' % w0, '0x%016x' % w1,
                                        '0x%016x' % w2, '0x%016x' % w3],
               'bank_sel': b % 2}
        base = addr + b * stride
        off = 0
        for k in order:
            name = 'block%d.%s' % (b, k)
            nbytes = param_files[name]['nbytes'] if name in param_files else E * E
            blk['weights'][k] = {'base': base + off, 'bytes': nbytes}
            off += nbytes
        blk['window'] = {'base': base, 'bytes': off}
        descs.append(blk)
    mem = {
        'sps_out_tokens': tokens,
        'residual_banks': 2,
        'bank_words_64b': __import__('math').ceil(
            tokens * E * T * cfg['derived']['res_bits'] / 64),
        'attn_tmp_words': tokens * cfg['derived']['attn_word'] // 8,
    }
    return {'sps_once': True, 'blocks': descs,
            'head_after_block': D - 1, 'memory_map': mem,
            'weight_stride': stride, 'thr_e': thr_e,
            'descriptor_note': '4 beats per block, Verilog-concat order per contract.hardware_abi.descriptor_beats_64b'}
