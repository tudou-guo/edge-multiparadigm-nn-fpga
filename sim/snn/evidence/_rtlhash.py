# -*- coding: utf-8 -*-
"""Regenerate evidence/nlp/rtl_hash_check_20261031.txt (derived check; temp tool)."""
import json, hashlib, os, datetime

ROOT = r'D:/DL_FINAL/SNN'
DL = r'D:/dl_snn'
frz = json.load(open(DL + '/build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json'))
rows, ndiff = [], 0
for rel, want in sorted(frz['files'].items()):
    p = os.path.join(DL, rel.replace('/', os.sep))
    got = hashlib.sha256(open(p,'rb').read()).hexdigest() if os.path.exists(p) else 'MISSING'
    ip = os.path.join(DL, 'build/spikformer/nlp_48h_20261029_1915/final/nlp_real_5stage_r4_holdfix/ip/src', rel.replace('/', os.sep))
    iph = hashlib.sha256(open(ip,'rb').read()).hexdigest() if os.path.exists(ip) else 'MISSING'
    ok = (got == want); ndiff += (0 if ok else 1)
    mt = datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime('%Y-%m-%d %H:%M') if os.path.exists(p) else '-'
    rows.append((rel, want, got, 'MATCH' if ok else 'CHANGED', mt, 'frozen' if iph == want else ('changed' if iph != 'MISSING' else 'MISSING')))

now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
lines = []
lines.append('# NLP 冻结 RTL 哈希核对（派生文件，非原始证据；%s CST 收拢时实测）' % now)
lines.append('')
lines.append('冻结记录：D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json')
lines.append('           schema=%s, frozen_at=%s' % (frz['schema'], frz['frozen_at']))
lines.append('当前主线：D:/dl_snn/rtl/ops/spiking/**（%s 时刻磁盘状态；D:/dl_snn 非 git worktree，无法用 git 归因）' % now)
lines.append('快照 rtl/：16:36 复制、17:00 core 刷新 spiking_sps_ref_ctrl.sv（74452587→728337f1）、18:30 再刷一次；以最终包为准（快照与主线的关系见 docs/PORTABILITY.md §4）。')
lines.append('')
lines.append('结论：18 个冻结文件中 %d 个与当前主线一致，%d 个已变化。' % (18-ndiff, ndiff))
lines.append('  - 发布快照 rtl/ 以复制时刻的主线内容为准（含图像线新文件，= 主线 superset 28 源 vs 冻结 19 源），原件哈希见 docs/PORTABILITY.md。')
lines.append('  - NLP 的 OOC / replay / IP 证据对应冻结时刻 RTL；ip/src 是构建时刻副本（最后一列 frozen 表示与冻结哈希一致）。')
lines.append('  - 当前主线 RTL 已被图像线改动：不能用当前主线重新综合/回放来声称与 NLP 冻结 OOC 证据同一版。')
lines.append('  - 变化文件为 sps/spiking_sps_top.sv（图像线 S_LDT 修复）与 sps/spiking_eyeriss_pe.sv（PE 单窗语义修复）；NLP 顶层不实例化 SPS，NLP 功能 RTL 16/16 未变。')
lines.append('  - 另：图像线 2026-10-31 17:22 又修改 control/spikformer_vis_native_top.sv（sha256 8ece767a…，P2 修复；不在冻结 18 文件内）；快照 rtl/ 18:30 刷新须包含。')
lines.append('')
lines.append('| # | 文件 | rtl_freeze sha256 | 当前主线 sha256 | 结果 | 当前主线 mtime | ip/src 副本 |')
lines.append('|---|---|---|---|---|---|---|')
for i, (rel, want, got, res, mt, ips) in enumerate(rows, 1):
    lines.append('| %d | %s | %s | %s | %s | %s | %s |' % (i, rel, want, got, res, mt, ips))
open(ROOT + '/evidence/nlp/rtl_hash_check_20261031.txt','w',encoding='utf-8').write('\n'.join(lines) + '\n')
print('rtl_hash_check written, ndiff=%d, frozen ip/src ok=%d' % (ndiff, sum(1 for x in rows if x[5]=='frozen')))
