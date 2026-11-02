"""Spikformer config schema v1 + derived quantities. No silent fallbacks."""
from __future__ import annotations
import json, math
SCHEMA = 'spikformer-config-1'
REQUIRED_BASE = ['schema', 'family', 'batch',
    'time_steps', 'embed', 'heads', 'head_dim', 'mlp_hidden', 'depth',
    'num_classes', 'lif', 'input', 'checkpoint', 'head', 'target']
REQUIRED_VIS = ['image_h', 'image_w', 'channels']
REQUIRED_NLP = ['seq_len', 'vocab_size', 'frontend']
FAMILIES_NLP = ('spikformer-nlp-sentiment', 'spikformer-nlp-intent', 'spikformer-nlp-match')
HEAD_NORMS = ('sum', 'ps_div', 'folded')
# Frozen by the 48h handoff: residual allocation 4 bit (min requirement recorded),
# 32-bit accumulator, 33-bit LIF adder width, 11-bit AV domain at seq32.
RW_ALLOC = 4
PSUM_SLICE = 32
LIF_ADD_BITS = 33
def _err(msg):
    raise ValueError('config: ' + msg)
def load(path):
    cfg = json.loads(open(path, encoding='utf-8-sig').read())
    return resolve(cfg)
def _need_bits(v):
    return max(1, int(math.ceil(math.log2(v + 1))))
def resolve(cfg):
    fam = cfg.get('family', '')
    req = list(REQUIRED_BASE)
    if fam in FAMILIES_NLP:
        req += REQUIRED_NLP
    else:
        req += REQUIRED_VIS
    for k in req:
        if k not in cfg:
            _err('missing required field %r' % k)
    if cfg['schema'] != SCHEMA:
        _err('unsupported schema (want %s)' % SCHEMA)
    if cfg['batch'] != 1:
        _err('only B=1 implemented')
    T, E, H, HD, MLP, D = (cfg['time_steps'], cfg['embed'], cfg['heads'],
        cfg['head_dim'], cfg['mlp_hidden'], cfg['depth'])
    if T not in (1, 2, 4):
        _err('time_steps not in legal [1,2,4]')
    if HD != 32:
        _err('head_dim: only 32 implemented (PE16 lane pair)')
    if E != H * HD:
        _err('embed != heads*head_dim')
    if E % 16 or E % 32:
        _err('embed must be divisible by 16 and 32')
    if MLP % 16:
        _err('mlp_hidden must be multiple of 16')
    if D < 1 or int(D) != D:
        _err('depth must be positive int')
    is_nlp = fam in FAMILIES_NLP
    if is_nlp:
        n = cfg['seq_len']
        if int(n) != n or n < 1 or n > 256:
            _err('seq_len must be int in [1,256]')
        if int(cfg['vocab_size']) != cfg['vocab_size'] or cfg['vocab_size'] < 1:
            _err('vocab_size must be positive int')
        fe = cfg['frontend']
        if fe.get('kind') != 'embed_spike':
            _err('frontend.kind must be embed_spike')
        if fe.get('mask', 'none') != 'none':
            _err('only mask=none (fixed length) implemented')
        tokens = int(n)
        geom = {'seq_len': tokens, 'embed': E}
        frontend_max = 1                      # embedding LIF output is a spike
    else:
        ih, iw = cfg['image_h'], cfg['image_w']
        if ih != iw:
            _err('rectangular input rejected: square-only pool geometry')
        if ih % 4:
            _err('image must be divisible by 4 (two stride-2 pools)')
        tokens = (ih // 4) * (iw // 4)
        geom = {'h': ih // 4, 'w': iw // 4, 'c': E}
        frontend_max = 2
    # ---- SPS (vision front end) geometry ----------------------------------
    # Referenced design: conv3x3(C0->C1) + LIF + 2x maxpool(stride2)
    #                  + conv3x3(C1->C2) + LIF, C2 == embed (token width).
    # C1 == embed/4 matches the upstream Spikformer-S patch embedding.
    if not is_nlp:
        c1 = int(cfg.get('sps_c1', E // 4))
        c2 = E
        if c1 < 8 or c1 > 1024:
            _err('sps C1 out of range: %r' % c1)
        if c1 % 8:
            _err('sps C1 must be a multiple of 8 (LANES): %r' % c1)
        if c2 % 8:
            _err('sps C2(=embed) must be a multiple of 8: %r' % c2)
        if (c1 // 8) % 8 and (c1 % 8):
            _err('sps C1/LANES must tile conv2 input: %r' % c1)
        # Bias ABI (unified 2026-10-31, see VISION_REFERENCE_ROUTE_REPAIR_PLAN s3.7):
        # RTL/TB golden packs one 21-bit signed bias pair per 64-bit beat:
        #   beat[i] = {hi=bias_c2[i] 21b, lo=bias_c1[i] 21b} for i < max(c1,c2).
        # Hence bias_beats_21pair == max(c1,c2). The old (c1+c2+15)//16 value
        # (30 for native 96/384) is a different packing unit and MUST NOT be
        # used for SPS bias stream lengths. Unique bias bytes are reported
        # separately from transfer beats; spatial re-reads are counted in
        # the schedule, not in the unique-weight capacity.
        sps = {'h': ih, 'w': iw, 'c0': cfg['channels'], 'c1': c1, 'c2': c2,
               'h1': ih // 2, 'w1': iw // 2, 'h2': ih // 4, 'w2': iw // 4,
               'tokens': tokens,
               'conv1_weights': c1 * cfg['channels'] * 9,
               'conv2_weights': c2 * c1 * 9,
               'conv1_unique_bytes': c1 * cfg['channels'] * 9,
               'conv2_unique_bytes': c2 * c1 * 9,
               'bias_unique_bytes': (c1 + c2) * 4,
               'conv1_beats': (c1 * cfg['channels'] * 9 + 7) // 8,
               'conv2_beats': (c2 * c1 * 9 + 7) // 8,
               'bias_beats': max(c1, c2),
               'bias_packing': '21b-pair per 64b beat, lo=c1 hi=c2, depth=max(c1,c2)',
               'bias_beats_deprecated_16pack': (c1 + c2 + 15) // 16}
    else:
        sps = None
    worst = frontend_max + 2 * D
    res_bits_min = _need_bits(worst)
    if RW_ALLOC < res_bits_min:
        _err('RW=%d cannot hold residual worst %d (needs %d bits)'
             % (RW_ALLOC, worst, res_bits_min))
    sum_w = int(math.ceil(math.log2(HD + 1)))
    # AV upper bound: source tokens * max per-t score (all-ones popcount of head_dim)
    av_max = tokens * HD
    av_bits = _need_bits(av_max)
    # signed 32-bit accumulator bounds, worst-case weight domain [-128,127]
    pos_w, neg_w = 127, -128
    bounds = {
        'qkv':  {'pos': pos_w * E,                'neg': neg_w * E},
        'proj': {'pos': pos_w * av_max * E,       'neg': neg_w * av_max * E},
        'fc1':  {'pos': pos_w * worst * E,        'neg': neg_w * worst * E},
        'fc2':  {'pos': pos_w * MLP,              'neg': neg_w * MLP},
    }
    int32_max = (1 << 31) - 1
    for name, b in bounds.items():
        if b['pos'] > int32_max or b['neg'] < -int32_max - 1:
            _err('accumulator bound for %s exceeds signed 32 bit: %r' % (name, b))
    if 'lif' in cfg:
        if int(cfg['lif'].get('tau', 0)) != 2:
            _err('lif.tau must be 2 for this batch')
        if int(cfg['lif'].get('reset', -1)) != 0:
            _err('lif.reset must be 0 (hard reset)')
    # weight/embed stream beats (64-bit words); all must divide evenly
    raw = {
        'tab_beats': (int(cfg['vocab_size']) * E + tokens * E) // 8 if is_nlp else 0,
        'token_beats': tokens,
        'block_w_beats': (4 * E * E // 8) + (4 * E // 2)
                         + (MLP * E // 8) + (MLP // 2)
                         + (E * MLP // 8) + (E // 2),
        'block_in_beats': tokens * E // 8,
        'descriptor_beats': 4,
        'mlp_fc1_internal_beats': (MLP // 8) * (E // 8) * ((8 * 8 // 8) + (8 // 2)),
        'mlp_fc2_internal_beats_per_group': (MLP // 8) * ((8 * 8 // 8) + (8 // 2)),
    }
    derived = {
        'tokens': tokens,
        'final_fmaps_width': tokens,
        'sum_w': sum_w,
        'attn_word': T * sum_w,
        'rw': RW_ALLOC,
        'res_bits': RW_ALLOC,
        'res_bits_min': res_bits_min,
        'spike_bits': 2,
        'psum_slice': PSUM_SLICE,
        'lif_add_bits': LIF_ADD_BITS,
        'av_bits': av_bits,
        'av_max': av_max,
        'worst_residual': worst,
        'stage_bounds': ({'sps': frontend_max, 'attn_res': frontend_max + 1,
                          'mlp_res': frontend_max + 2} if not is_nlp else
                         {'embed': frontend_max, 'attn_res': frontend_max + 1,
                          'mlp_res': frontend_max + 2}),
        'accum_bounds': bounds,
        'stream': raw,
        'packing': {
            'layout': 'time-minor lane pack (lane m owns bits [m*T +: T])',
            'words_note': 'residual words = N*C*T*RW/64',
            'res_bits': RW_ALLOC,
        },
        'geometry': geom,
        'sps': sps,
    }
    for k, v in raw.items():
        if isinstance(v, int) and v < 0:
            _err('derived stream beat %s negative' % k)
    if cfg['head'].get('reduction') not in HEAD_NORMS:
        _err('head.reduction must be sum/ps_div/folded')
    if cfg['head'].get('norm') not in HEAD_NORMS:
        _err('head.norm must be sum/ps_div/folded')
    cfg = dict(cfg)
    cfg['derived'] = derived
    return cfg
