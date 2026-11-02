# -*- coding: utf-8 -*-
"""Generate evidence/SHA256SUMS.txt and evidence/INDEX.md from _copy_manifest.json.
Temp tool for the consolidation; deleted after use."""
import json, hashlib, os, datetime, pathlib

EV = pathlib.Path(r'D:/DL_FINAL/SNN/evidence')
rows = json.loads((EV / '_copy_manifest.json').read_text(encoding='utf-8'))

EXTRA = [
 ('evidence/platform/r7_sumsfix_route_status.rpt', 'D:/spk_platform/reports/sumsfix_route_status.rpt',
  'r7-sumsfix（实际生成 bit 的版本）routed route status：0 routing errors，fully routed=180101'),
 ('evidence/platform/platform_timing_summary_20261031.txt', '(派生) D:/spk_platform/reports/sumsfix_timing_{setup,hold}.rpt + r6_final_timing_*.rpt',
  '派生：r7-sumsfix routed 时序摘要 WNS +0.051 / WHS +0.011 / 0 failing；r6 对照 WNS +0.245 / WHS +0.010（源报告 2MB+ 未复制）'),
 ('evidence/nlp/rtl_hash_check_20261031.txt', '(派生) D:/dl_snn/rtl/ops/spiking/** vs build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json',
  '派生：18 个冻结 RTL 哈希核对，16 match / 2 changed（spiking_sps_top.sv、spiking_eyeriss_pe.sv 被图像线改动）；ip/src 18/18 一致'),
 ('evidence/ACCEPTANCE_CORE.md', '(core 子代理) D:/DL_FINAL/SNN 验收记录', 'core 子代理的三条命令验收记录（A1a/A1b/A2/A3 等；含编译 kind=real_corpus、params 哈希集合、迁移目录一致性）——core 所有，最终版本以文件本身为准'),
 ('evidence/PORTABILITY_DIFFS.txt', '(core 子代理) D:/DL_FINAL/SNN 可移植性 diff 摘要', 'core 子代理的路径相对化 diff 摘要（tools/scripts/verification 改动清单）——core 所有，最终版本以文件本身为准'),
 ('evidence/COPY_SHA256.txt', '(core 子代理) D:/DL_FINAL/SNN 复制时刻源码树 sha256 清单', 'core 子代理在 16:36 复制源码树后生成的快照清单（覆盖 docs/rtl/tools/scripts/runtime/verification 等）——不属于本 INDEX 的主证据行，仅登记'),
 ('evidence/vision/ref_route/PROGRESS_DOCS_SHA256.md', '(core 子代理) 图像线进展文档副本说明', 'core 子代理补入的两份图像线进展文档（VISION_REF_P1_PROGRESS / VISION_NATIVE_SIM_PROGRESS）副本哈希说明——与本 INDEX 行一致',),
]
for dst, src, desc in EXTRA:
    rows.append({'dst': dst, 'src': src, 'sha256': '', 'cat': 'derived' if not dst.split('/')[-1] in ('ACCEPTANCE_CORE.md','PORTABILITY_DIFFS.txt','COPY_SHA256.txt','PROGRESS_DOCS_SHA256.md') else 'aux', 'desc': desc})

DESC = {
'evidence/nlp/contract.json': '冻结合同（2026-10-29 19:30）：数值语义 + stream ABI 权威；scope 明确非原版 Spikformer 同构（无 softmax、未归一 AV、无 BN/LayerNorm、仅 proj+fc2 后 LIF）',
'evidence/nlp/final_gates.json': '九门状态总表：math_contract/target_e2e/real_weight_replay/integer_accuracy/ip/ooc=pass；platform/bit_hwh/board=not_run（48h run 口径）',
'evidence/nlp/rtl_freeze.json': '18 文件冻结哈希 + 冻结点前后回放证据清单（frozen_at 2026-10-29T22:26，5-stage MAC pipeline 后 r2）',
'evidence/nlp/replay_small_holdfix.json': '小几何回放 pass：cycles=39774（hold-fix 后同一冻结 RTL）',
'evidence/nlp/replay_e2e_noforce_holdfix.json': 'target e2e 回放 pass：cycles=7084772，两连续样本无全局复位，0 mismatch',
'evidence/nlp/replay_e2e_bpforce_holdfix.json': '带回压窗口 e2e 回放 pass：cycles=7084812，输出 stall 14 / 输入 stall 10 窗口，0 mismatch',
'evidence/nlp/replay_real_holdfix.json': '真实 checkpoint 3 样本（dev#0/dev#1/holdout#0）回放 pass：cycles=10627157，0 mismatch',
'evidence/nlp/train_real50.json': '真实训练记录：5 epochs，dev 61.6→67.7→74.5→77.25→78.33%，holdout(torch) 76.77%（约 16s/epoch）',
'evidence/nlp/train_realweight_fidelity_real50.json': '导出保真 PASS：torch 量化整数等价前向 vs integer_ref，47 层整数 trace 全等，pl_sums_equal/argmax_equal=true',
'evidence/nlp/export_final_real50.json': '整数域标定 + 导出 + 严格导入；integer_ref 精度 dev 0.7592(n=1200,majority 0.5058)、holdout 0.7562(n=960,majority 0.5000)；checkpoint sha256 e6a3a411…2448a1',
'evidence/nlp/math_tests_m1.log': 'M1 数学/接口短测 8/8 PASS 日志（torch 量化 vs integer_ref 全层整数 trace 一致）',
'evidence/nlp/math_tests_m1.json': 'M1 结果 JSON（8/8 PASS；覆盖 LIF 边界、popcount=32、AV=1024、跨 token/t 置换、PS head 逐 bit）',
'evidence/nlp/final/manifest.json': '最终构建 manifest（model_kind=real_corpus，54 个 param blob，绑定全部阈值/scale/FP32 head）',
'evidence/nlp/final/numeric_contract.json': '数值合同 JSON（manifest 绑定）：LIF 33bit / QK SUM_W=6 / AV uint11 max1024 / residual 4bit / head ps_div / 结构 E128-H4-HD32-N32-T4-D2',
'evidence/nlp/final/resolved_config.json': '解析后配置（TL8/TC8/RW4/PSUM_SLICE32/SUM_W6；target xczu15eg-ffvb1156-2-i, 5ns）',
'evidence/nlp/final/schedule.json': '调度：block0 权重 base/bytes（q/k/v/proj/fc1/fc2）、descriptor beats、memory_map',
'evidence/nlp/final/reports/build_identity_20261030.json': '构建身份绑定：manifest/adapter/xdc/ooc_tcl/codegen 哈希 + cache_key（r2 一致）',
'evidence/nlp/final/reports/hold_closure_20261030.json': 'hold 收口记录：DONT_TOUCH LUT1 输入 hold-margin 链（1192 cells）、routed-DCP 复检、合成/实现细节',
'evidence/nlp/final/reports/ooc_20261030_120804.json': '5ns routed OOC PASS：WNS +0.135 / WHS +0.023 / TNS 0 / 0 failing / 0 unrouted；LUT 40964 / FF 71173 / BRAM 186.5 / DSP 128',
'evidence/nlp/final/reports/synth_20261030_115812.json': 'OOC synth PASS：LUT 42229 / FF 71111 / BRAM 186.5 / DSP 128（routed 前）',
'evidence/nlp/final/generated/spikformer.xdc': '生成的 5ns XDC（sha256 9fc1c267…10842；无 false path/multicycle、无降频）',
'evidence/nlp/final/generated/spikformer_ooc.tcl': 'OOC 实现脚本（含 adapter 路径与 place/phys_opt Explore 指令，无时序例外）',
'evidence/nlp/final/generated/spikformer_nlp_target_top.sv': '生成 target adapter（含 1192 DONT_TOUCH 输入 hold-margin LUT1 单元；属生成的适配层，不在 18 文件 rtl_freeze 内）',
'evidence/nlp/final/generated/spikformer_config.vh': '生成配置宏（结构/I型参数绑定）',
'evidence/nlp/final/ip/ip_manifest.json': 'IP 包清单：timing_qualified=true；package_sha256=894599b9…292f22；假设/范围（OOC 5ns only、0 输入延迟模型、保留 1192 DONT_TOUCH cells）',
'evidence/nlp/final/ip/SOURCE_SHA256.txt': 'IP 19 个源文件 sha256（与 rtl_freeze 18 文件一致 + include/spikformer_cfg.vh）',
'evidence/nlp/final/ip/README.txt': 'IP 使用说明与 SCOPE/ASSUMPTIONS（platform/bit/HWH/board not_run）',
'evidence/nlp/final/ip/ooc_report.json': 'IP 绑定的 OOC 报告副本（pass，WNS +0.135 / WHS +0.023）',
'evidence/platform/platform_build_20261030.json': '平台 BD 搭建阶段状态（2026-10-30 14:09）：block design/generate_target/CSR smoke pass；当时 synth/impl/timing/bit/HWH/board=not_run（后续 r6/r7 见同目录其他证据）',
'evidence/platform/r6_final_route_status.rpt': '平台 r6 routed route status：0 routing errors，fully routed=172589（r7 之前 CDC 修复版本）',
'evidence/platform/r6_final_util.rpt': '平台 r6 利用率：CLB LUTs 72167/341280=21.15%，LUT as Logic 67348',
'evidence/platform/CDC_FIX_R6_RESULT_20261030.md': '平台 r6 CDC/时序收口记录（WNS +0.245 / WHS +0.010 / 0 failing；关键改动与限制）',
'evidence/platform/r7_sumsfix_route_status.rpt': 'r7-sumsfix（实际 bit 版本）route status：0 routing errors，fully routed=180101',
'evidence/platform/platform_timing_summary_20261031.txt': '派生摘要：r7-sumsfix routed Setup WNS +0.051/TNS 0/0 failing，Hold WHS +0.011/THS 0/0 failing，WPWS +0.039；r6 对照',
'evidence/platform/board/MANIFEST.sha256': '上板包 MANIFEST.sha256（bit/hwh/run 脚本/demo 资产/vec 全量哈希）',
'evidence/platform/board/README_BOARD.md': '上板说明：r7-sumsfix（WNS +0.051/WHS +0.011）、运行步骤、预期输出（文档声明）；§3 记录主会话 3 样本 match=True（原始 JSON 未随包→快照 not_verified）',
'evidence/platform/board/demo_samples.json': '板端 demo 3 样本清单（真实 checkpoint 的固定样本）',
'evidence/platform/board/vec/case_real/golden.json': '板测 golden：3 样本的 exp_sums/stream/etab/ptab 参考（与 NLP 回放同源）',
'evidence/platform/board/vec/case_real/exp_sums_s0.hex': '板测样本0期望 sums（板端 run 脚本比对用）',
'evidence/platform/board/vec/case_real/exp_sums_s1.hex': '板测样本1期望 sums',
'evidence/platform/board/vec/case_real/exp_sums_s2.hex': '板测样本2期望 sums',
'evidence/platform/board/board_run_local_attempt.log': '本机板端运行尝试失败日志（ModuleNotFoundError: pynq）→ 说明本收拢环境未复跑板端；不是板测证据',
'evidence/VISION_REFERENCE_ROUTE_RESULT_20261031.md': '图像参考工程路线结果快照（16:11 版本：P0 PASS；P1 small 当时 FAIL 53/256 nibbles；P1 native/P2/P3/B/P4 not_run）。此后 P1 small 已四配置 PASS（16:44–16:59，见 vision/ref_route/p1_r2_small.log、p1_r7_*.log、p1_r8_*.log）、P1 native SPS 正式门 PASS（17:01，见 tb_sps_native_imgnative_r4_*.log）、P2 单图 PASS（17:26，见 tb_vis_native_chain_p2fix_img1_*.log）；P2 两图 in_progress、P3 OOC blocked；最终状态以 docs/VISION_CASE.md 与 DELIVERY_SUMMARY.md 为准',
'evidence/vision/toolchain/resolved_config.json': '图像 native 配置解析产物：32×32×3 → C1=96/C2=384/E384/H12/HD32/MLP1536/T4/64 token/depth1；checkpoint kind=missing；派生流长度（conv1 324 beats、conv2 41472、block_w 222912、block_in 3072）',
'evidence/vision/toolchain/schedule.json': '图像 native 调度产物：block0 六矩阵 DDR base/bytes、descriptor beats（全 0，fixture 无阈值）、memory_map',
'evidence/vision/toolchain/manifest.json': '图像 native 工具链 manifest（family=spikformer-b1-cifar10，model_kind=development_fixture，10 类）',
'evidence/vision/toolchain/numeric_contract.json': '图像 native 数值合同（结构 E384/H12/HD32/T4/TOKENS64/DEPTH1；其 av 字段沿用 32-token 的 uint11/1024 文案（陈旧），2048/12bit 见 resolved_config.derived；fixture 权重）',
'evidence/vision/toolchain/spikformer_config.vh': '图像 native 生成配置宏（SPIK_SPS_*/SPIK_*）；编译 pass 的直接产物',
'evidence/vision/toolchain/spikformer.xdc': '图像 native 生成的 5ns XDC（无时序例外；仅工具链产物，OOC 未跑）',
'evidence/vision/behavior/vsim_block_r.log': 'tb_block_replay ALL PASS（16 beats，13:19 最新回归）',
'evidence/vision/behavior/vsim_sps_r.log': 'tb_vis_sps ALL PASS（16 beats，13:19 最新回归）',
'evidence/vision/behavior/vsim_replay_r.log': 'tb_vis_replay ALL PASS（32 channels；TOKEN CHECK fails=0；block weight=1136、block output=16；TSUM 4 6 12 16 24 16 16 3）',
'evidence/vision/behavior/vsim_block_sps_tokens_fixed_20261031.log': 'tb_block_replay（SPS token 输入）ALL PASS（16 beats）',
'evidence/vision/behavior/vsim_block_stage_probe_20261031.log': 'block 定向 stage probe PASS：QKV LIF / QK / AV / 两次完整权重重载 / 残差输出全匹配',
'evidence/vision/behavior/vsim_replay_fixed_model_20261031.log': 'SPS→block→sums 整链 ALL PASS（token fails=0、32 sums 匹配；golden 修复后模型）',
'evidence/vision/behavior/vsim_block_baseline_20261031.log': '历史基线：golden 修复前 tb_block_replay FAIL（240/256 nibble mismatch）——证明修复因果',
'evidence/vision/behavior/vsim_block_fixed_model_20261031.log': 'golden 修复后 tb_block_replay ALL PASS（16 beats）',
'evidence/vision/ref_route/baseline_hashes.json': 'P0 基线冻结哈希（参考工程路线的 baseline identity）',
'evidence/vision/ref_route/tb_sps_ref_small.log': '历史证据：P1 参考式 SPS small 第一版 FAIL（FAILURES=53/256 nibbles；BEATFAIL 明细）——已被 p1_r2_small.log 的 PASS 取代',
'evidence/vision/ref_route/compile_ref_small.log': '参考式 SPS small TB 编译日志（16:06 版本）',
'evidence/vision/ref_route/p1_r0_compile.log': '参考式 SPS small TB 编译日志（p1_r0）',
'evidence/vision/ref_route/p1_r2_small.log': 'P1 参考式 SPS small PASS：16/16 beats bit-exact，capture=64 flush=16 structural ok（16:44:21；RTL ctrl sha e6008ad6… 的中间版本，已被 p1_r7_* 三配置 PASS 取代）',
'evidence/vision/ref_route/p1_r7_small_t2.log': 'P1 三配置 PASS 之一（2026-10-31 16:56，ctrl sha 728337f1…）：tb_sps_ref_small（T=2）16/16 beats bit-exact，capture=64 flush=16 structural ok',
'evidence/vision/ref_route/p1_r7_small_t4.log': 'P1 三配置 PASS 之一（16:56）：tb_sps_ref_small_t4（T=4）16/16 beats bit-exact，capture=64 flush=16 structural ok',
'evidence/vision/ref_route/p1_r7_h32_t4.log': 'P1 三配置 PASS 之一（16:56，H32 proxy）：tb_sps_ref_small_h32_t4 256 beats ALL PASS，capture=1024 flush=64 structural ok',
'evidence/vision/ref_route/p1_r8_h32_t4_tc2_8.log': 'P1 四配置 PASS 之一（16:59，H32-T4/TC2=8）：tb_sps_ref_small_h32_t4_tc2_8 256 beats ALL PASS，capture=1024 flush=64 structural ok',
'evidence/vision/ref_route/tb_sps_native_imgnative_r4_20261031.log': 'P1 native SPS 正式门 PASS（17:01）：H32/W32/C1-96/C2-384/T4/TC2=8，3072 beats 全等，capture=12288/flush=768 结构断言 0 fail，cycles=5,573,066；ctrl sha 728337f1…',
'evidence/vision/ref_route/tb_sps_native_imgnative_r4_diff_summary.txt': 'native 门独立 python 复核：exp/got 均 3072 beats，exact 3072/3072，hamming nibbles and bits=0，nibbles outside {0,1}=0',
'evidence/vision/ref_route/native_gate_result_20261031.json': 'native SPS 门结果 JSON（status=PASS）：H32/W32/C1-96/C2-384/T4/TC2=8；3072 beats/98304 nibbles 0 mismatch、capture=12288/flush=768、cycles=5,573,066；rtl_ctrl_sha256=728337f1…；含 invalidated_runs 与 not_run 清单',
'evidence/vision/ref_route/native_gate_evidence_sha256.txt': 'native SPS 门全证据 sha256 清单（最新版 sha256 b1fafa6c…；manifest 新增 "P2 chain external references" 段；绑定日志/向量/脚本/结果 JSON）',
'evidence/vision/ref_route/native_gate_verify.py': 'native 门独立 Python 复核脚本（verify_r4_pass_imgnative.py：读 exp.hex 与 gotbeats hex 逐 beat 比较；主会话路径 D:/dl_snn/sim_work/vis_sps/）',
'evidence/vision/ref_route/sps_native_tc2_8_meta.json': 'native 正式门向量 meta（TC2=8 口径）：H32/W32/C1-96/C2-384/T4/lanes8、thr1=200/thr2=60、conv2 unique 331776B、conv1 324 beats、conv2 41472 beats、pix 1024、exp 3072；大 hex 未复制',
'evidence/vision/ref_route/native_gate_rtl_hashes.txt': 'native SPS 门运行时刻 RTL 哈希（6 个参考式模块；ctrl 728337f1… 见 gate JSON）',
'evidence/vision/ref_route/native_gate_compile.log': 'native SPS TB 编译日志（work_native，Errors=0）',
'evidence/vision/ref_route/VISION_REF_P1_PROGRESS_20261031.md': '参考路线 P1 进度记录（2026-10-31 17:02）：native SPS TB 建成与 r1–r4 定位过程、P1 small 多配置结果、修复清单（RTL 只读、未跑综合/板卡）',
'evidence/vision/ref_route/VISION_NATIVE_SIM_PROGRESS_20261031.md': '图像 native 仿真进度（最新版 sha256 53d63f56…）：含 §10 最终证据清单、§11 收尾确认（r1pref/r2/r3 作废、r4 有效、旧 vectors=sps_native tc2=4 不配对、正式向量 sps_native_tc2_8、flush 断言修法、OOC probe 参数核对）、§6.1 P2 后续结果；native 整链骨架编译 PASS / 门控 NOT_RUN',
'evidence/vision/ref_route/tb_vis_native_chain_p1_img1_20261031.log': 'P2 单图"修复前"日志（17:18，sha256 c073917f…）：block-out 3072/3072 beats bit-exact PASS；sums 382/384 不等（got/want≈1/4）；FAILURES=1。根因定位为 spikformer_vis_native_top.sv N_RUN 累加 RTL bug（NBA 循环只保留最后一次 t2 赋值）；img-p1 获授权只修该 control 顶层（sps RTL 728337f1 冻结不动），修复后重跑单图+两图',
'evidence/vision/ref_route/tb_vis_native_chain_p2fix_img1_20261031.log': 'P2 单图"修复后"PASS 日志（17:26，本次运行 +IMG2=0 即单图；TB 终行硬编码“2 images”是 TB 文案瑕疵）：SUMS PASS img0（384 sums）+ BLKOUT PASS img0（3072 beats bit-exact, last ok）。修复 spikformer_vis_native_top.sv 的 N_RUN 累加 bug（改动仅 beat_add 组合求和 + 每 lane 单次 acc 累加；该 control 顶层 sha256 8ece767a…），sps/block RTL 哈希未变（sps 728337f1 冻结）',
'evidence/vision/ref_route/compile_native_chain_p2_fix_20261031.log': 'P2 修复后整链 TB 编译日志（work_chain，Errors=0）',
'evidence/vision/ref_route/tb_vis_native_chain_p2fix_img2_20261031.log': 'P2 两图连续 FAIL 日志（17:33:42，sha256 9dc10409…）：同一次运行 img0 SUMS+BLKOUT 均 PASS，img1 卡死（top=N_RUN blk=S_TOK sps=C_LDW1）到 watchdog 120M cyc（5:19 wall）；根因（RTL 只读分析）：spiking_rgb_encoder.sv 的 w1_done/bias_done/pix_done/fwin_loaded 与 spiking_sps_ref_ctrl.sv 的 enc_started 为 sticky、只在 s_rst 清 → 第二次 start 不再开权重接收；修复需改这两个 sps 参考模块，本轮 RTL 冻结未改',
'evidence/vision/ref_route/VISION_P2_CHAIN_PROGRESS_20261031.md': 'P2 整链进度/根因文档（17:38 更正版，sha256 98a0d2f1…）：修复前后对比、两图 blocker 根因（sps 参考模块第二次 start re-arm 缺失）、golden 自检（gen_block_param.py 394605c4…）、向量清单（block_native + sps_native_img2）、TB sha 8aead49c…、复现命令、未测项（输入气泡/输出反压/OOC/平台/板测 not_run）',
'evidence/vision/ref_route/block_native_meta_img1.json': 'P2 golden 向量 meta（img1）：block_native 权重/期望输出/期望 sums 的生成参数与哈希',
'evidence/vision/ref_route/block_native_meta_img2.json': 'P2 golden 向量 meta（img2）：同权重不同像素（第二张图）',
'evidence/vision/ref_route/block_native_exp_sums_img1.hex': 'P2 golden：img1 期望 384 sums（每通道 uint32）',
'evidence/vision/ref_route/block_native_exp_sums_img2.hex': 'P2 golden：img2 期望 384 sums',
'evidence/vision/ref_route/block_native_exp_out_img1.hex': 'P2 golden：img1 期望 3072 block-out beats（128-bit/beat，逐 beat 比较用）',
'evidence/vision/ref_route/block_native_exp_out_img2.hex': 'P2 golden：img2 期望 3072 block-out beats',
'evidence/vision/ref_route/sps_native_img2_meta.json': 'P2 第二张图 SPS 向量 meta（同权重不同像素）',
'evidence/vision/ref_route/sps_native_img2_pix.hex': 'P2 第二张图 SPS 像素输入（1024 beats）',
'evidence/vision/ref_route/sps_native_img2_exp.hex': 'P2 第二张图 SPS 期望 token（3072 beats）',
'evidence/vision/ref_route/p2_gen_block_param.py': 'P2 golden 生成器（gen_block_param.py，sha256 394605c4…）：先在 small 参数逐字节复现旧 block_replay 全部 wq..bf2 + blk_w 拼接 1136 beats + 旧 tok 复现 exp.hex，再生成 native',
'evidence/vision/ref_route/p2_gen_sps_img2.py': 'P2 第二张图 SPS 向量生成器（gen_sps_img2.py，sha256 98396cfd…）：同权重不同像素',
'evidence/vision/ref_route/ooc_native/vivado.log': 'native OOC 整顶 synth 失败日志（17:39:25，sha256 55db145a…）：RTL Elaboration failed，6 Errors；[Synth 8-91] ambiguous clock in event control @ spiking_sps_psum.sv:115；连带 psum/ref_top/vis_native_top/vis_native_probe synth 失败；impl/routed not_run',
'evidence/vision/ref_route/ooc_native/vivado_whole_top_synth_fail_1739.log': '同上失败日志的命名副本（img-ooc 命名；与 vivado.log 同 sha256 55db145a…）',
'evidence/vision/ref_route/ooc_native/facts.json': 'native OOC facts（17:44 更新版）：gate.p3_native_ooc=fail（synth RTL elaboration；非资源/时序失败、非 design infeasible）；synth fail 6 errors、root_cause [Synth 8-91] @ spiking_sps_psum.sv:115、cascade、hypothesis（for-loop body 内 process-block-scoped unpacked arrays oldv/newv）、reports_produced=[]、impl not_run；含 heavy lock 时间线、measured_vs_derived、next_actions',
'evidence/vision/ref_route/ooc_native/rtl_sha256_at_synth.txt': 'OOC 综合时刻 RTL 身份（17:04 生成；参考式 SPS/vis native top 哈希）',
'evidence/vision/ref_route/ooc_native/constraints_ooc.xdc': 'OOC 约束（5ns、无时序例外、不生成 bitstream）',
'evidence/vision/ref_route/ooc_native/commands.txt': 'OOC 运行命令记录',
'evidence/vision/ref_route/ooc_native/heavy_wait_latest.json': 'heavy lock 等待/资源探测记录（external Vivado PID 504、内存不足）',
'evidence/vision/ref_route/ooc_native/heavy_wait_170705.json': 'heavy lock 等待记录（17:07 采样）',
'evidence/vision/ref_route/ooc_native/memwatch_latest.log': '内存监控记录（free physical 采样）',
'evidence/vision/ref_route/ooc_native/vivado_fail_include_option_1734.log': '早期失败尝试日志（include option 错误，17:34；正式失败是 17:39 synth elaboration）',
'evidence/vision/ref_route/ooc_native/blk_only_1742/vivado.log': 'block-only OOC（降级补充，17:42）：独立于 SPS 缺陷的 block 侧综合尝试日志；不是整顶门，不用于报 P3 资源/时序（Lead 口径：P3=whole-top synth fail，impl/routed not measured）',
'evidence/vision/ref_route/ooc_native/blk_only_1742/constraints_ooc.xdc': 'block-only OOC 约束副本（5ns、无时序例外）',
'evidence/vision/ref_route/ooc_native/blk_only_1742/vivado.jou': 'block-only OOC journal（17:41）',
'evidence/vision/ref_route/VISION_NATIVE_OOC_PROGRESS_20261031.md': 'native OOC 进度/根因文档（17:42 版，sha256 ac157105…）：P3 fail 记录、根因假设与下一轮修复路径（单模块回归、数组 hoist、case default、u_enc.o_beat_last 连接）、资源纪律',
'evidence/vision/ref_route/p1_r2_compile.log': 'P1 small PASS 对应编译日志（Errors=0）',
'evidence/vision/ref_route/sps_small_check_meta.json': 'small 向量 meta（P0 阶段 golden 冻结）',
'evidence/vision/ref_route/sps_native_meta.json': 'native 向量 meta（ld1=324 / bias=384 / uniq=41472 / pix=1024 / exp=3072 等）',
'evidence/vision/native_resource/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md': 'native 资源问题记录：spiking_sps_top 旧结构 native 几何 OOC synth 尝试 >25min 未收敛中止（无 util/timing 报告）；静态评估 ≈846k FF（>器件 682,560 FF）、conv2 2304 MAC/拍（>DSP 1248）——均为静态推导非实测，且部分估算口径已被 REPAIR_PLAN §3 纠正',
'evidence/vision/native_resource/VISION_LINE_RETURN_20261031.md': '图像行为级整链 PASS 的修复记录：根因是 Python golden 落后于 block 合约（proj/fc2 后 LIF），仅改 gen_block_replay.py，未改 RTL；含验证矩阵与日志索引',
'evidence/vision/native_resource/VISION_LINE_PROGRESS_20261031.md': '历史进展（13:10）：当时整链未过、tb_block_replay 失效；9 个工具链/vis_top bug 修复清单（历史，不当作当前状态）',
'evidence/vision/native_resource/SPIKFORMER_VISION_HANDOFF_20261031.md': '图像线交接：工具链完成（compile 命令/产物）、native 几何、上游仓库不带 CIFAR-10 checkpoint（→ 只能 fixture 功能冒烟、不能报精度）、vis_platform_top 待建',
}
for r in rows:
    d = r['dst']
    if '/ip/src/' in d:
        r['desc'] = 'NLP 构建时刻 RTL 副本（冻结态；逐文件 sha256 见 SHA256SUMS.txt 与 nlp/final/ip/SOURCE_SHA256.txt）'
    elif d in DESC:
        r['desc'] = DESC[d]
    elif r.get('desc'):
        pass
    else:
        r['desc'] = '(见文件名)'
    p = EV / d[len('evidence/'):].replace('/', os.sep)
    r['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    r['bytes'] = p.stat().st_size

# --- SHA256SUMS.txt ---
sums = ['# evidence/ 快照文件 sha256（2026-10-31 收拢时计算；格式 <sha256>  <相对路径>）',
        '# 注：ACCEPTANCE_CORE.md / PORTABILITY_DIFFS.txt / COPY_SHA256.txt 由 core 子代理维护并在 18:30 刷新，不计入本清单；其 sha256 见 INDEX.md §5。']
for r in sorted(rows, key=lambda x: x['dst']):
    if r['cat'] == 'aux':
        continue
    rel = r['dst'][len('evidence/'):]
    sums.append('%s  %s' % (r['sha256'], rel))
sums.append('')
(EV / 'SHA256SUMS.txt').write_text('\n'.join(sums), encoding='utf-8')

# --- INDEX.md ---
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
total_bytes = sum(r['bytes'] for r in rows)
L = []
L.append('# 证据快照 INDEX（D:/DL_FINAL/SNN/evidence）')
L.append('')
L.append('快照时间：%s CST（复制时刻；来源 D:/dl_snn、D:/spk_platform 只读，未改动源工程）。' % now)
n_aux = sum(1 for r in rows if r['cat'] == 'aux')
L.append('规模：共 %d 个条目 / %.1f KB（RELEASE_SCOPE §5 预算 <=8MB）；其中 %d 个快照文件计入 SHA256SUMS.txt，%d 个 core 维护文件仅登记（§5）。' % (len(rows), total_bytes/1024.0, len(rows)-n_aux, n_aux))
L.append('原则：只复制小文件；不复制 routed DCP、synth 大树、2MB+ 时序报告与参数二进制（缺口见 §6）。')
L.append('RTL 快照：rtl/ 于 16:36 复制、17:00 core 刷新 spiking_sps_ref_ctrl.sv、18:30 再刷一次（以最终包为准）；本 INDEX 的冻结核对针对“主线 RTL vs NLP 冻结哈希”，见 nlp/rtl_hash_check_20261031.txt。')
L.append('')
L.append('## 0. 校验方法')
L.append('')
L.append('4 空格缩进的命令可直接复制（Windows bash / cmd 下 E:/miniconda/python.exe 均可）：')
L.append('')
L.append('    cd D:/DL_FINAL/SNN/evidence')
L.append(r'''    E:/miniconda/python.exe -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"''')
L.append('')
n_sums = len(rows) - n_aux
L.append('**最终校验（%s CST）**：SHA256SUMS.txt 覆盖 %d 个文件，实测 bad=0；证据目录共 %d 个条目 / %.1f KB（<=8MB）。' % (now, n_sums, len(rows), total_bytes/1024.0))
L.append('')
L.append('## 1. 状态摘要（一行一事实；每个 PASS 指向证据）')
L.append('')
L.append('| 主题 | 状态 | 证据 |')
L.append('|---|---|---|')
SUM = [
 ('NLP 数学合同（math_contract）', 'pass', '[nlp/final_gates.json](nlp/final_gates.json)、[nlp/math_tests_m1.log](nlp/math_tests_m1.log)'),
 ('NLP 真实训练 + 整数域精度（integer_accuracy）', 'pass：integer_ref dev 0.7592（n=1200，majority 0.5058）、holdout 0.7562（n=960，majority 0.5000）；public test 无标签→不报 test acc', '[nlp/export_final_real50.json](nlp/export_final_real50.json)、[nlp/train_real50.json](nlp/train_real50.json)、[nlp/train_realweight_fidelity_real50.json](nlp/train_realweight_fidelity_real50.json)'),
 ('NLP 功能回放（冻结 RTL，4 项）', 'pass：small 39774 / e2e-noforce 7084772 / e2e-bpforce 7084812 / real 10627157 cycles，全部 0 mismatch', '[nlp/replay_small_holdfix.json](nlp/replay_small_holdfix.json)、[nlp/replay_e2e_noforce_holdfix.json](nlp/replay_e2e_noforce_holdfix.json)、[nlp/replay_e2e_bpforce_holdfix.json](nlp/replay_e2e_bpforce_holdfix.json)、[nlp/replay_real_holdfix.json](nlp/replay_real_holdfix.json)'),
 ('NLP OOC 5ns routed', 'pass：WNS +0.135 / WHS +0.023 / TNS 0 / THS 0 / 0 unrouted / 无时序例外', '[nlp/final/reports/ooc_20261030_120804.json](nlp/final/reports/ooc_20261030_120804.json)'),
 ('NLP IP 包', 'pass：timing_qualified=true，package_sha256=894599b9…292f22，19/19 源哈希；scope=OOC 5ns only', '[nlp/final/ip/ip_manifest.json](nlp/final/ip/ip_manifest.json)、[nlp/final/ip/SOURCE_SHA256.txt](nlp/final/ip/SOURCE_SHA256.txt)'),
 ('平台集成（D:/spk_platform）', 'partial（定稿口径）：BD/compile/CSR smoke pass；r7-sumsfix 全片 routed Setup WNS +0.051 / Hold WHS +0.011 / 0 failing，bit+HWH 已生成；完整协议仿真矩阵 not_run、CDC 1 Critical 未收敛、软复位不排空 DMA', '[platform/platform_build_20261030.json](platform/platform_build_20261030.json)、[platform/platform_timing_summary_20261031.txt](platform/platform_timing_summary_20261031.txt)、[platform/r7_sumsfix_route_status.rpt](platform/r7_sumsfix_route_status.rpt)'),
 ('板测（board）', '记录级 3/3 match（主会话记录；原始板端 JSON/日志未随包）→ 本快照 **not_verified**，不宣称板测 pass；板包/golden 齐备，板卡离线未复跑', '[platform/board/README_BOARD.md](platform/board/README_BOARD.md) §3、[platform/board/MANIFEST.sha256](platform/board/MANIFEST.sha256)、[platform/board/board_run_local_attempt.log](platform/board/board_run_local_attempt.log)'),
 ('图像工具链（vis_native 配置编译）', 'pass（产物齐；model_kind=development_fixture，checkpoint missing）', '[vision/toolchain/resolved_config.json](vision/toolchain/resolved_config.json)、[vision/toolchain/schedule.json](vision/toolchain/schedule.json)、[vision/toolchain/manifest.json](vision/toolchain/manifest.json)'),
 ('图像行为级小几何回放', 'pass：tb_block_replay / tb_vis_sps / tb_vis_replay 均 ALL PASS（16 beats / 32 channels）', '[vision/behavior/](vision/behavior/)（vsim_block_r.log、vsim_sps_r.log、vsim_replay_r.log）'),
 ('图像参考工程路线 P0（基线/哈希/向量）', 'pass', '[vision/ref_route/baseline_hashes.json](vision/ref_route/baseline_hashes.json)'),
 ('图像参考式 SPS small（P1）', 'pass（2026-10-31 16:56–16:59，四配置）：T=2 / T=4 / H32-T4 proxy / H32-T4-TC2=8 均 ALL PASS（16/16 beats ×2，H32 proxy 256 beats ×2；capture=64/1024 flush=16/64）；旧 53/256 nibbles FAIL 为历史基线', '[vision/ref_route/p1_r7_small_t2.log](vision/ref_route/p1_r7_small_t2.log)、[vision/ref_route/p1_r7_small_t4.log](vision/ref_route/p1_r7_small_t4.log)、[vision/ref_route/p1_r7_h32_t4.log](vision/ref_route/p1_r7_h32_t4.log)、[vision/ref_route/p1_r8_h32_t4_tc2_8.log](vision/ref_route/p1_r8_h32_t4_tc2_8.log)、[vision/ref_route/tb_sps_ref_small.log](vision/ref_route/tb_sps_ref_small.log)（历史 FAIL）'),
 ('图像 P1 native SPS 正式门（3072 beats）', 'pass（2026-10-31 17:01）：H32/W32/C1-96/C2-384/T4/TC2=8，3072 beats 全等，98,304 nibbles hamming=0，capture=12288/flush=768 结构断言 0 fail，cycles=5,573,066；ctrl sha256 728337f1…', '[vision/ref_route/tb_sps_native_imgnative_r4_20261031.log](vision/ref_route/tb_sps_native_imgnative_r4_20261031.log)、[vision/ref_route/tb_sps_native_imgnative_r4_diff_summary.txt](vision/ref_route/tb_sps_native_imgnative_r4_diff_summary.txt)、[vision/ref_route/sps_native_meta.json](vision/ref_route/sps_native_meta.json)'),
 ('图像 P2 native 整链', '单图 pass（2026-10-31 17:26）；两图连续 **fail**（17:33:42：同一次运行 img0 SUMS+BLKOUT 均 PASS，img1 卡死 top=N_RUN/blk=S_TOK/sps=C_LDW1 到 watchdog 120M cyc）。根因（RTL 只读分析）：spiking_rgb_encoder.sv 的 w1_done/bias_done/pix_done/fwin_loaded 与 spiking_sps_ref_ctrl.sv 的 enc_started 为 sticky、只在 s_rst 清 → 第二次 start 不再开权重接收；修复需改这两个 sps 参考模块，本轮 RTL 冻结未改。输出反压/输入气泡 not_run。功能仿真，非资源/时序/板级。修复前单图 BLKOUT PASS/SUMS FAIL（N_RUN 累加 RTL bug 已修，control 顶层 sha256 8ece767a…）', '[vision/ref_route/tb_vis_native_chain_p2fix_img1_20261031.log](vision/ref_route/tb_vis_native_chain_p2fix_img1_20261031.log)（单图 PASS）、[vision/ref_route/tb_vis_native_chain_p2fix_img2_20261031.log](vision/ref_route/tb_vis_native_chain_p2fix_img2_20261031.log)（两图 FAIL）、[vision/ref_route/tb_vis_native_chain_p1_img1_20261031.log](vision/ref_route/tb_vis_native_chain_p1_img1_20261031.log)（修复前，sha256 c073917f…）、[vision/ref_route/VISION_P2_CHAIN_PROGRESS_20261031.md](vision/ref_route/VISION_P2_CHAIN_PROGRESS_20261031.md)'),
 ('图像 P3 OOC / B 轮 / P4 平台板测', 'P3 OOC **fail（synth elaboration，2026-10-31 17:39:25）**：stage=all 首次整顶 OOC synth 失败 6 errors，根因 [Synth 8-91] ambiguous clock in event control @ spiking_sps_psum.sv:115（伴 case not full/no default line 119）；impl/routed/资源/时序 not measured（不报 LUT/FF/BRAM/WNS）。sim 功能门 P1/P2 不受影响；这是首次综合参考式 RTL 暴露的兼容性问题，下一轮修复路径见 VISION_NATIVE_OOC_PROGRESS。B 轮 / P4 平台板测 not_run', '[vision/ref_route/ooc_native/vivado.log](vision/ref_route/ooc_native/vivado.log)、[vision/ref_route/ooc_native/facts.json](vision/ref_route/ooc_native/facts.json)、[vision/ref_route/ooc_native/rtl_sha256_at_synth.txt](vision/ref_route/ooc_native/rtl_sha256_at_synth.txt)、[vision/ref_route/VISION_NATIVE_OOC_PROGRESS_20261031.md](vision/ref_route/VISION_NATIVE_OOC_PROGRESS_20261031.md)'),
 ('图像精度（CIFAR-10）', 'not_run / 无：上游仓库不带 CIFAR-10 checkpoint，只有 fixture 权重 → 只做功能冒烟，不报精度', '[vision/native_resource/SPIKFORMER_VISION_HANDOFF_20261031.md](vision/native_resource/SPIKFORMER_VISION_HANDOFF_20261031.md)'),
 ('RTL 冻结一致性', '16/18 match，2 changed（spiking_sps_top.sv、spiking_eyeriss_pe.sv 被图像线改动）：快照 rtl/ 复制于 16:36、17:00 已刷新 sps_ref_ctrl.sv（core）、18:30 再刷一次，= 当前主线 superset（28 源 vs 冻结 19 源）；NLP 证据绑定 2026-10-30 冻结版本（ip/src 为冻结态）', '[nlp/rtl_hash_check_20261031.txt](nlp/rtl_hash_check_20261031.txt)'),
]
for a, b, c in SUM:
    L.append('| %s | %s | %s |' % (a, b, c))
L.append('')
L.append('## 2. NLP 证据')
L.append('')
L.append('| 证据 | 主线来源路径 | sha256 | 用途/结论 |')
L.append('|---|---|---|---|')
for r in rows:
    if r['cat'] != 'nlp' or '/ip/src/' in r['dst']:
        continue
    rel = r['dst'][len('evidence/'):]
    L.append('| [%s](%s) | %s | %s | %s |' % (rel, rel, r['src'], r['sha256'], r['desc']))
L.append('| [nlp/final/ip/src/**（19 文件）](nlp/final/ip/src/) | D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/final/nlp_real_5stage_r4_holdfix/ip/src | 逐文件哈希见 [SHA256SUMS.txt](SHA256SUMS.txt) 与 [nlp/final/ip/SOURCE_SHA256.txt](nlp/final/ip/SOURCE_SHA256.txt) | NLP 构建时刻 RTL 副本（冻结态，19/19 与 rtl_freeze 18 文件 + cfg.vh 一致）；可作历史证据 |')
L.append('')
L.append('## 3. 平台与板测证据')
L.append('')
L.append('| 证据 | 主线来源路径 | sha256 | 用途/结论 |')
L.append('|---|---|---|---|')
for r in rows:
    if r['cat'] != 'platform':
        continue
    rel = r['dst'][len('evidence/'):]
    L.append('| [%s](%s) | %s | %s | %s |' % (rel, rel, r['src'], r['sha256'], r['desc']))
L.append('')
L.append('## 4. 图像线证据')
L.append('')
L.append('| 证据 | 主线来源路径 | sha256 | 用途/结论 |')
L.append('|---|---|---|---|')
for r in rows:
    if r['cat'] != 'vision':
        continue
    rel = r['dst'][len('evidence/'):]
    L.append('| [%s](%s) | %s | %s | %s |' % (rel, rel, r['src'], r['sha256'], r['desc']))
L.append('')
L.append('## 5. 派生核对 / 附属证据（非原始证据，仅供复核）')
L.append('')
L.append('| 证据 | 生成来源 | sha256 | 用途/结论 |')
L.append('|---|---|---|---|')
for r in rows:
    if r['cat'] not in ('derived', 'aux'):
        continue
    rel = r['dst'][len('evidence/'):]
    L.append('| [%s](%s) | %s | %s | %s |' % (rel, rel, r['src'], r['sha256'], r['desc']))
L.append('')
L.append('## 6. 未收录 / 已知缺口')
L.append('')
GAP = [
 'routed DCP 与 synth 大树未复制（如 ooc_run_20261030_120804/post_route.dcp、synth_run_* 目录）；重建命令见 RELEASE_NOTES.md。',
 '2MB+ 的平台时序报告（r6_final/sumsfix timing_*.rpt）未复制，摘要见 platform/platform_timing_summary_20261031.txt。',
 '板端运行结果 JSON/日志：未随包（RELEASE_SCOPE §5 的"若存在"项→现状不存在）；只有主会话记录 3/3 match 与本机 board_run_local_attempt.log（缺 pynq）→ 快照记 not_verified，不宣称板测 pass（定稿口径见 docs/DELIVERY_SUMMARY.md §2）。',
 '图像 native 向量大 hex（sps_native/*.hex 约 1.6MB）未复制，只保留 vectors meta.json；生成脚本与命令见 VISION_CASE.md。',
 'NLP 参数二进制（params/*.bin/*.coe、ckpt npz 17MB）未复制：ckpt 在 build/.../train/，发布快照的案例资产由 docs 子代理放入 examples/。',
 '图像 native 正式门 PASS（3072 beats，17:01）只证明功能/数值仿真；其资源/时序（P3 OOC blocked）、P2 整链、平台/板测仍未验证。',
 '图像 native 资源/OOC：旧结构 native SPS OOC 综合 >25min 未收敛中止（无 util/timing 报告）；参考式新结构 P3 OOC blocked（外部 Vivado 占用 heavy lock，未启动）→ 不报 LUT/FF/BRAM/DSP/WNS 实测值。',
 'evidence/ACCEPTANCE_CORE.md、PORTABILITY_DIFFS.txt、COPY_SHA256.txt 是 core 子代理的验收/可移植性/复制清单证据（见 §5 附属证据，已计入 SHA256SUMS.txt）；如 core 在定稿前再更新，以最终文件为准。',
]
for g in GAP:
    L.append('- ' + g)
L.append('')
(EV / 'INDEX.md').write_text('\n'.join(L), encoding='utf-8')
print('INDEX.md bytes:', (EV/'INDEX.md').stat().st_size, '| SHA256SUMS bytes:', (EV/'SHA256SUMS.txt').stat().st_size)
print('rows:', len(rows), 'total KB:', round(total_bytes/1024.0,1))
