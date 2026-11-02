# 交付摘要（Lead 定稿）— D:/DL_FINAL/SNN 发布快照 0.1.0

- 定稿时间：2026-10-31（Asia/Shanghai）；整理规范：`docs/RELEASE_SCOPE.md`
- 来源：主产品 `D:/dl_snn`（只读复制）、平台/上板包 `D:/spk_platform`（只读复制）
- 发布根：`D:/DL_FINAL/SNN`（自包含，可整目录拷走；与 `D:/DL_FINAL/DL` 并列、互不修改）
- 本文是所有状态的**最终口径**；与其它文档冲突时以本文为准（其余文档保留各自证据细节）。

## 0. 一句话

NLP 情感分类链路（真实语料训练 → 真实 checkpoint → 编译/参数/RTL/调度 → 冻结 RTL 功能回放 0 mismatch → 5 ns routed OOC pass → IP 源包 → 平台 r7 bit）已经闭环，可演示、可复核；
图像识别线是**诚实 partial**（工具链与行为级回放 pass、参考式 SPS 的 small 门今天刚 bit-exact pass、native 资源/OOC/平台/板测未完成，**无精度、未上板，不作演示**）。

## 1. 交付物

| 交付物 | 位置 | 说明 |
|---|---|---|
| 发布说明 | `README.md` `RELEASE_NOTES.md` `VERSION` `environment.md` | 这是什么/收录与未收录/环境 |
| 3 条命令跑通 | `QUICKSTART.md`（实际执行记录 `evidence/ACCEPTANCE_CORE.md`） | 环境检查 → NLP 编译 → 纯 Python 数值自检 |
| 工具链源码 | `tools/`（统一入口 + backend + 训练/导出） | 路径已相对化，可在任意根运行 |
| RTL 算子库 | `rtl/ops/spiking/**` + `operator_library.json` + `source_manifest.json` | 与主线 2026-10-31 当前状态一致（复制时刻 16:36，图像线文件另有更新说明见 §5） |
| PYNQ 运行包 | `runtime/spikformer/` | tokenize/head/run/packing |
| 示例与真实权重 | `examples/`（`examples/nlp_sentiment_b1/ckpt_real_real50.npz`） | 真实 checkpoint 可完整恢复 |
| 上板包 | `board/`（r7-sumsfix bit + hwh + runner + demo_nlp.py + demo 资产 + vec + MANIFEST） | 板卡当前离线，快照未现场复跑 |
| 证据快照 | `evidence/`（89 文件，sha256 清单 + `INDEX.md`） | 每个 PASS 都能指到 JSON/日志 |
| 文档 | `docs/`：USER_GUIDE / OPERATORS / NUMERIC_CONTRACT / PLATFORM_AND_STATUS / DEMO_NLP / VISION_CASE / PORTABILITY | 逐门状态与证据 |

## 2. NLP 演示主线：已验证项（可复核）

| 门 | 结果 | 证据（快照内） |
|---|---|---|
| 编译/权重导入（快照根，真实 checkpoint） | pass：`kind=real_corpus`，54 个 param blob | `evidence/ACCEPTANCE_CORE.md`（A1b）；`evidence/nlp/` |
| 数值合同 M1 | pass：8/8；47 层整数 trace 与 PyTorch 量化前向逐层一致 | `evidence/nlp/math_tests_m1.log` |
| 冻结 RTL 功能回放（4 项） | pass，全部 0 mismatch：small 39,774 / e2e 7,084,772 / e2e 反压 7,084,812 / real 3 样本 10,627,157 cycles | `evidence/nlp/replay_*_holdfix.json` |
| 真实语料精度 | pass（整数域）：dev 0.7592（n=1200，majority 0.5058）/ holdout 0.7562（n=960）；public test 无标签 → 不报 test acc | `evidence/nlp/train_real50.json`、`export_final_real50.json` |
| 5 ns routed OOC | pass：Setup WNS **+0.135 ns** / Hold WHS **+0.023 ns** / TNS、THS 0 / 0 失败 / unrouted 0 | `evidence/nlp/final/reports/ooc_20261030_120804.json` |
| IP 源包 | pass：`timing_qualified=true`，package_sha256 `894599b9…`；scope = OOC 5 ns only | `evidence/nlp/final/ip/ip_manifest.json` |
| 平台 r7 bit | bit/HWH 已生成（r7-sumsfix，Setup WNS +0.051 / Hold WHS +0.011 / 0 failing）；平台集成其余项 partial | `board/`、`evidence/platform/` |
| 板端 3 样本比对 | **主会话记录 3/3 match；原始板端 JSON 未随包 → 本快照 not_verified**（不作已复核声明） | `board/README_BOARD.md` §4、`docs/DEMO_NLP.md` §4 |

## 3. 图像识别线：诚实 partial（不作演示）

| 项 | 状态 | 证据 |
|---|---|---|
| 工具链编译（vis_native 配置） | pass（生成链） | `evidence/vision/toolchain/` |
| 行为级小几何整链回放（SPS→block→sums） | pass（16 beats / 32 channels） | `evidence/vision/behavior/` |
| 参考工程路线 P0（基线/哈希/向量） | pass | `evidence/vision/ref_route/baseline_hashes.json` |
| 参考式 SPS small 门（P1） | **pass**：T2/T4 两配置 16/16 beats bit-exact、结构断言 0 fail（ctrl sha256 `728337f1…`；Lead 独立复跑过）；H32/T4 代理几何（C1-16/C2-32）256 beats PASS（`p1_r7_h32_t4.log`）。历史 FAIL（53/256 nibble）保留为基线 | 主线 `build/spikformer/vision_ref_20261031/sim/p1_r7_small_t2.log`、`p1_r7_small_t4.log`、`p1_r7_h32_t4.log`；快照 `evidence/vision/ref_route/` 副本 |
| 旧 native 资源门 | fail（旧 `spiking_sps_top` 结构约 846k FF 静态评估 > 器件 682k；native SPS 综合 probe >25 min 未收敛） | `evidence/vision/native_resource/` |
| P1 正式 native 门（H32/C1-96/C2-384/T4/TC2=8，3072 beats） | **pass**：3072/3072 beats 全等、98,304 nibbles hamming=0、capture=12288/flush=768 结构断言 0 fail、5,573,066 cycles（ctrl `728337f1…`；Lead 独立 TB 副本 + 独立 work 库复跑确认；r3 的 1,142 nibbles 为旧 ctrl/旧向量对照，作废） | 主线 `tb_sps_native_imgnative_r4_20261031.log` + `tb_sps_native_imgnative_r4_diff_summary.txt` |
| P2 native 整链 bit-exact（单图） | **pass**：3072 block-out beats + 384 sums 全 bit-exact。修复 1 个真 RTL bug：`spikformer_vis_native_top.sv` N_RUN 同一 acc 在 t2 循环多次 NBA → 只累加 t=3（got/want≈1/4）；改组合 `beat_add` 后归零，control 顶层 sha256 `8ece767a…`（sps/block RTL 未动）。Lead 独立 TB 副本 + 独立 work 库复跑确认 | 主线 `tb_vis_native_chain_p2fix_img1_20261031.log`；快照 `evidence/vision/ref_route/` 副本 |
| P2 两图连续 | **fail（已知 RTL 限制）**：img0 全 PASS；img1 卡死 watchdog。根因：`spiking_rgb_encoder.sv` 的 w1_done/bias_done/pix_done/fwin_loaded 与 `spiking_sps_ref_ctrl.sv` 的 enc_started 是 sticky、只在 s_rst 清 → 第二次 start 不 re-arm（对应任务书 §5.2 未实现项）。非 TB/golden 问题；本轮 RTL 冻结未修。输出反压/`+BPS` 气泡 not_run（A-round top 无外部 hook） | 主线 `tb_vis_native_chain_p2fix_img2_20261031.log` |
| P3 OOC（整顶 5ns routed）/ B 轮 / P4 | **in_progress / not_run**：用户关闭外部 Vivado 后 17:35 整顶 OOC（stage=all）已在共享锁内启动（probe 首次因 read_verilog include 语法失败，已修并重跑）；B 轮/P4 平台板测 not_run | `docs/VISION_CASE.md`、OOC run 目录 |
| 图像精度（CIFAR-10） | **not_run / 无**：上游无 checkpoint，只有 fixture 权重 → 只做功能冒烟 | `docs/VISION_CASE.md` |

## 4. 范围与总约束（必须同时引用）

1. **OOC ≠ 集成 ≠ 板级**：NLP 的 5 ns routed OOC 结果是 out-of-context、输入 0 延迟模型、未设 HD.CLK_SRC；集成时必须按真实接口重新约束；输入 setup 预算仅 +0.135 ns。
2. **平台/板测边界**：r7 bit 已生成且有主会话记录 3/3 match，但原始板端输出未随包、板卡离线 → 本快照不复核、不宣称板测 pass；完整协议仿真矩阵 not_run。
3. **快照未重跑 Vivado**：不重跑 synth/OOC/bitstream（规范禁止）；所有物理结果引用 2026-10-30 主线实测证据。
4. **NLP 冻结证据 vs 当前 RTL**：快照 RTL 为 2026-10-31 当前主线状态；NLP 冻结证据绑定 2026-10-29/30 的 18 文件哈希。比对结果 16 SAME / 2 DIFF（`spiking_eyeriss_pe.sv`、`spiking_sps_top.sv`，均为图像线并行改动）；`spikformer_nlp_top.sv` 不实例化 SPS，NLP 功能 RTL 16/16 未变。
5. **数值口径冲突并列**：板端 7,084,772 cycles（35.4 ms）与 handoff 记录 3,389,415 cycles（16.95 ms）不一致，未核实，见 `docs/DEMO_NLP.md` §5。
6. **演示限制**：每句 2–3 s（每样本重载 overlay）、PL 不回 idle、软复位不排空 DMA、r7 bit 的 SUMS 读窗口需 `ROT=64` 软件补偿（详见 `docs/DEMO_NLP.md` §3，含 A/B 两种现场做法）。

## 5. 复核命令（全部在快照根 D:/DL_FINAL/SNN 执行）

```bat
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz --out-dir build/_check_real

E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py
E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py
```

证据哈希复核、迁移目录一致性、manifest/params 摘要见 `evidence/ACCEPTANCE_CORE.md` 与 `evidence/INDEX.md` §0。

## 6. 未收录 / 未验证（如实记录）

- 未收录：`build/` 大树（routed DCP、综合目录）、`sim_work/` 旧工作库、开发期一次性脚本（RELEASE_NOTES §2）。
- 未验证：Vivado 重跑、ModelSim 全 TB 回归（快照只带源码与证据日志）、板端现场复跑、平台 BD 重建、图像线 native 整链/OOC/平台/板测（见 §3）。

## 7. 更新记录（收拢过程时间线）

| 时间 | 事件 | 结果 |
|---|---|---|
| 16:32 | 整理规范 `docs/RELEASE_SCOPE.md` 定稿；发布根 = `D:/DL_FINAL/SNN` | — |
| 16:36 | 首次复制主线源码/文档/上板包（97 文件） | — |
| 16:46–16:52 | core/docs/evidence 三线初版（README/QUICKSTART/USER_GUIDE/OPERATORS/DEMO_NLP/INDEX 等） | — |
| 16:44 / 16:46 / 16:56 / 16:59 | 图像线参考式 SPS small 门四配置 PASS（T2、T4、H32-T4 proxy、H32-T4-TC2=8；ctrl 逐步修到 `728337f1…`） | 见 `docs/VISION_CASE.md` §4 |
| 16:48 / 16:59 / 17:04 | **Lead 独立复跑**：small T2、T4、native SPS（自建 work 库 / 独立 TB 副本） | 全部 ALL PASS |
| 17:00 | 快照 RTL 刷新到主线当时状态；快照根 compile（fixture + 真实权重）复跑 | params 哈希集合不变；kind=real_corpus，54 blob |
| 17:01 | 图像线 **P1 native SPS 正式门 PASS**（3072 beats / 98,304 nibbles hamming=0 / capture 12288 flush 768 / 5,573,066 cycles） | 见 `docs/VISION_CASE.md` §4 |
| 17:10 | 图像线 P3 OOC：probe+tcl 已就绪并自检通过；heavy lock 被外部 Vivado（用户 GUI）与内存不足阻塞，未启动 | 记 blocked（非 fail） |
| 17:50 | 图像线 P2 native 整链回放结论 | 以 `docs/VISION_CASE.md` 最终行为准 |
| 18:25–18:35 | 最终刷新（RTL/TB）+ evidence SHA256SUMS 重算 + 验收记录更新 | 见 `evidence/ACCEPTANCE_CORE.md` 文末续记 |
| 18:40 | Lead 定稿本文 | — |

**口径提醒**：本表只记录事件；每个数字的最终口径以本文 §1–§3 与 `docs/PLATFORM_AND_STATUS.md` 为准。
