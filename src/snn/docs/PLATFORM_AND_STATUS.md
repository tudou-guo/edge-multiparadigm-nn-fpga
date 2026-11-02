# 平台与门状态（NLP 演示主线；图像线 partial）

快照：`D:/DL_FINAL/SNN` 0.1.0；内容冻结 2026-10-31 18:40；本文件随收拢更新。
口径：只写 `pass / fail / partial / not_run`；每个 PASS 指证据；未执行一律 `not_run`；冲突数字并列写来源、不确定写“未核实”。

**最终口径**：`docs/DELIVERY_SUMMARY.md`（Lead 定稿）是发布快照所有状态的最终裁决，本文与它冲突时以它为准；本文保留更细的证据与拆分。
**与任务卡初稿的差异（如实记录）**：任务卡初稿曾要求 platform=pass、board=pass；定稿口径把它们分别降为 **platform=partial**（bit/HWH/时序子项 pass，平台集成其余项未完成）与 **board=记录级 3/3 match / 快照 not_verified**（原始板端 JSON 未随包、板卡离线、不宣称板测 pass）。本文按定稿口径书写。

---

## 1. NLP 门状态总表

| 门 | 状态（定稿口径） | 证据 |
|---|---|---|
| backend_development | **pass** | 编译入口 `tools/toolchain.py` + backend config/manifest 生成；实模型编译产物 `evidence/nlp/final/manifest.json`（快照）/主线 `build/…/train/nlp_real_compile/manifest.json`（model_kind=real_corpus，54 blobs）；快照根重跑实测 exit 0（RTL 17:38 最终刷新后：`manifest_sha256=f727e4c7…911e`、54 blob，详见 `docs/DEMO_NLP.md` §1.4）；九门实测记录 `docs/SPIKFORMER_BACKEND.md` §4 |
| parameterization | **pass** | `tools/backends/spikformer/config.py` 派生 T/depth/通道/缓存/IP 宽 + 非法配置结构化拒绝；三份配置 `examples/spikformer/config/*.json` compile 通过 |
| real_model_assets | **pass** | 真实 checkpoint `examples/nlp_sentiment_b1/ckpt_real_real50.npz`（sha256 `e6a3a411…448a1`，strict import PASS）；`ckpt_real_real50.assets.json`；`corpus_meta.json`；`evidence/nlp/export_final_real50.json` |
| math_contract | **pass** | `evidence/nlp/math_tests_m1.log`（8/8 PASS；full_trace small/target 各 47 层）；快照根复跑实测 8/8 PASS（`build/_selfcheck_math/math_tests_m1.json`）；Lead 独立复核 `math_lead_check.json`（23/23）与 `intref_fix.json`（6/6）在主线 `build/…/reports/`（未复制进快照，见 `evidence/INDEX.md` §6 缺口） |
| target_e2e | **pass** | `evidence/nlp/replay_e2e_noforce_holdfix.json`（cycles 7,084,772，mismatch 0）；`replay_e2e_bpforce_holdfix.json`（7,084,812，14 输出/10 输入 stall，0 mismatch） |
| real_weight_replay | **pass** | `evidence/nlp/replay_real_holdfix.json`（3 固定真实样本，cycles 10,627,157，0 mismatch）；`train_realweight_fidelity_real50.json`（47 层整数前向一致） |
| integer_accuracy | **pass** | `evidence/nlp/export_final_real50.json`：integer_ref dev 0.7592（n=1200，majority 0.5058）/ holdout 0.7562（n=960，majority 0.5000），accuracy_established=true。**public test.tsv 无标签 → 不报 test accuracy** |
| rtl_freezes | **pass（带 16/18 说明）** | `evidence/nlp/rtl_freeze.json`（18 文件冻结点哈希，2026-10-29 22:26）；`evidence/nlp/final/reports/build_identity_20261030.json`。**说明**：快照 rtl/ 取 2026-10-31 16:36 主线状态，与冻结点比对 **16 SAME / 2 DIFF**（`spiking_sps_top.sv`、`spiking_eyeriss_pe.sv`，图像线并行改动）；NLP 功能 RTL **16/16 未变**（`spikformer_nlp_top.sv` 不实例化 SPS），见 §5 |
| **ip** | **pass** | `evidence/nlp/final/ip/ip_manifest.json`（`timing_qualified=true`，19/19 RTL 哈希，package_sha256 `894599b9…`）；`ip/SOURCE_SHA256.txt`；绑定 routed OOC 报告 |
| **ooc** | **pass** | `evidence/nlp/final/reports/ooc_20261030_120804.json`：5ns routed **Setup WNS +0.135 ns / TNS 0 / 0 failing（186,547 endpoints）；Hold WHS +0.023 ns / THS 0 / 0 failing；pulse width 0 failing；unrouted 0**；`evidence/nlp/final/…/ooc_run_20261030_120804/` 报告集；`ooc/analysis5/routed_check/routed_hold_check.log`（完整失败 hold 集 0）。**范围**：OOC、输入 0 延迟模型、HD.CLK_SRC 未设置；WNS 束缚路径是新增输入延时链本身（+0.135 ns），真实上游 max 延时超过即失败 → 不是集成即用时序签核 |
| **platform** | **partial（定稿）** | 子项：① 平台搭建 pass——`evidence/platform/platform_build_20261030.json`（BD validate/generate pass、ModelSim compile pass、CSR/FSM smoke 7/7）、`D:/spk_platform/logs/{build_snn_r2.log,check_project.log}`、`D:/spk_platform/sim/tb_snn_csr/vsim.log`；② r6 实现 pass——WNS +0.245 / WHS +0.010 / 0 failing（`D:/spk_platform/reports/r6_final_timing_*.rpt`、`evidence/platform/CDC_FIX_R6_RESULT_20261030.md`）；③ r7-sumsfix 全片 5ns 实现 + bit/HWH pass——见 §2。**未完成**：完整协议仿真矩阵 not_run、软复位不排空 DMA、写完成常数假设、CDC 1 Critical 未收敛；48h run 的 platform 门在 `final_gates.json` 里是 `not_run`（该文件 2026-10-30 13:10 冻结，早于 r6/r7 实现） |
| **board** | **记录级 3/3 match；快照 not_verified** | 主会话记录：3 个真实样本 `match=True`，cycles≈7,084,772，model_ms≈35.4（`board/README_BOARD.md` §3；`docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2）。**但**原始板端 run JSON/控制台日志未随包、本机尝试因缺 `pynq` 失败（`evidence/platform/board/board_run_local_attempt.log`）、板卡当前离线 → 快照**不宣称板测 pass**，记 `not_verified`；完整协议仿真矩阵 not_run |
| multi_block | **partial** | sequencer DEPTH=3 回放 ALL PASS（权重基址 0/1M/2M、bank 翻转、末 block 才归约）；`tb_block_replay` ALL PASS（16 beats）；multiblock 配置 compile pass；**未做多 block 真实权重整链回放**。记录 `docs/SPIKFORMER_BACKEND.md` §4/§6 |

**上板包**：`board/snn_platform.bit`（sha256 `9b2e318c5eacd2f21c39bd6c8864cd0f41ca47ed069ee6bbd7ea45b0191c267b`，28,700,853 B）+ `board/snn_platform.hwh`（sha256 `fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb`，380,358 B）；HWH 从 `project_1/snn_platform.xsa` 内的 `design_1.hwh` 提取，与 bit 同 stem。`MANIFEST.sha256` 覆盖 20 个文件（2026-10-30 23:18）；4 个 demo 资产（`demo_nlp.py`/`demo_vocab.json`/`demo_head.npz`/`demo_samples.json`，10-31 11:00 后加入）不在原 MANIFEST，哈希见快照 `board/MANIFEST_DEMO_20261031.sha256`（本快照新增的补充分片）。

---

## 2. 平台 r7-sumsfix（实际生成 bit 的版本）

器件 `xczu15eg-ffvb1156-2-i`，Vivado 2021.1，200 MHz（5 ns）**全片**实现。bit 生成 2026-10-30 23:10，xsa 23:13。

| 项 | 值 | 证据 |
|---|---|---|
| Setup | **WNS +0.051 ns**，TNS 0.000，0 failing（334,330 endpoints），WPWS +0.039 | `D:/spk_platform/reports/sumsfix_timing_setup.rpt`；快照派生摘要 `evidence/platform/platform_timing_summary_20261031.txt` |
| Hold | **WHS +0.011 ns**，THS 0.000，0 failing（333,740 endpoints） | `D:/spk_platform/reports/sumsfix_timing_hold.rpt`；同上摘要 |
| 布线 | 0 routing errors；fully routed 180,101 / routable 180,101 | `evidence/platform/r7_sumsfix_route_status.rpt`（原文副本） |
| DRC | 0 error；135 warning（DPIP-2×3、DPOP-3×128、PDCN-1569×3、RTSTAT-10×1） | `D:/spk_platform/reports/sumsfix_drc.rpt` |
| bitstream / HWH | `write_bitstream completed successfully`；`write_hw_platform` 完成 → `project_1/snn_platform.xsa` | `D:/spk_platform/logs/sumsfix.log`（`SUMSFIX_IMPL=write_bitstream Complete!`） |
| 相对 r6 的 RTL 差异 | 仅 `snn_axi_lite_csr.sv` 的 SUMS 窗口解码（32 字 → 128 字；**未修 +64 偏移**） | `D:/spk_platform/src/snn/snn_axi_lite_csr.sv` vs `.bak_sumsfix`；分析见 `docs/DEMO_NLP.md` §3 |
| 对照 r6 | r6：WNS +0.245 / WHS +0.010 / 0 failing（CDC 修复版） | `D:/spk_platform/reports/r6_final_timing_{setup,hold}.rpt` |

**已知平台限制（随 platform=partial 一起引用）**：
1. 完整协议仿真矩阵 `not_run`（只有 CSR/超时 smoke 7/7，`sim/tb_snn_csr/vsim.log` 中 `NLP_RESULT pass=1 errors=0`）。
2. 软复位只 flush 两侧 FIFO，**不排空在途 DMA 事务**；连续第二个样本会超时 → 现靠每样本重载 overlay 绕过。
3. 写完成按常数 4 统计（512 B 输出）；其它长度/4 KB 边界未验证。
4. 顶层 `sys_rst` 无 input delay/false path、DDR4 `reset_n` 无 output delay（check_timing HIGH，对本次功能无影响）。
5. CDC 报告 1 条 Critical（CDC-10，DDR4/复位基础设施路径，已标 False Path，来自基线平台）+ 4 条 CDC-15 warning；未重新收敛。
6. 平台 r7 是 NLP 专用（wrapper + 冻结 NLP 内核 + r4 adapter）；图像线平台未做（`not_run`）。

---

## 3. 板端（记录级 3/3 match；快照 not_verified）

| 项 | 记录值 | 来源 |
|---|---|---|
| 3 真实样本 | 主会话记录 **全部 match=True**（sums 逐通道与 golden 相等） | `board/README_BOARD.md` §3；`docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2 |
| cycles / model_ms | ≈7,084,772 / ≈35.4 ms（200 MHz 域） | `board/README_BOARD.md` §3 |
| 演示时延 | 每句 2–3 s；PL 计算记录 16.95 ms = 3,389,415 cycles（与上一条口径不同，**未核实**） | handoff §1.4；并列见 `docs/DEMO_NLP.md` §5 |
| 快照可复核性 | **not_verified**：原始板端 JSON/日志未随包；本机尝试因缺 `pynq` 失败；板卡 `192.168.2.99` 离线，本次收拢未访问板卡 | `evidence/platform/board/board_run_local_attempt.log`；DELIVERY_SUMMARY §2 |
| 包完整性 | 板包 MANIFEST 20/20 校验通过（快照复制时验证）；demo 4 文件见 `board/MANIFEST_DEMO_20261031.sha256` | 本快照 `board/` |

**SUMS 读窗口（必须与板端结论一起引用）**：r7-sumsfix 的 CSR 解码 `if (araddr_q>=12'h100 && araddr_q<12'h300) rdata = sums_flat[{araddr_q[8:2],5'b0} +: 32]` 含基址 0x100>>2=64 → `rd(SUMS+4*i) = o_sums[(i+64)%128]`。`demo_nlp.py` 的 `ROT=64` 是**必须的补偿**；`snn_platform_run.py` 的 `read_sums()` 没有旋转，在 r7 bit 上原样跑会 64 通道错位。两种正确读法（软件旋转 / 按 handoff §5.1 重出 bit 后 ROT=0）与“原始 JSON 缺失 → 未核实”的说明见 `docs/DEMO_NLP.md` §3。

---

## 4. 图像线门状态（partial；不是演示、无精度、未上板）

> 图像线状态以 `docs/DELIVERY_SUMMARY.md` §3 为准（P1 small 四配置 PASS + P1 native SPS 3072 beats PASS；P2/P3/P4 见该表）。以下保留收拢时刻的门级明细；与 `DELIVERY_SUMMARY.md` 冲突时以它为准。

| 项 | 状态 | 证据 |
|---|---|---|
| 工具链编译（图像配置族 `spikformer-b1-cifar10`） | **pass（生成链）** | `evidence/vision/toolchain/{manifest.json,resolved_config.json,schedule.json,numeric_contract.json,spikformer_config.vh,spikformer.xdc}`；native 配置 `model_kind=development_fixture`、`checkpoint.kind=missing`（只有 fixture 权重） |
| 行为级回放（小几何整链 SPS→block→sums） | **pass** | `evidence/vision/behavior/vsim_{block_r,sps_r,replay_r}.log`：tb_block_replay ALL PASS 16 beats、tb_vis_sps ALL PASS、tb_vis_replay ALL PASS 32 channels（TOKEN fails=0） |
| native 资源门（旧 SPS 结构） | **fail（静态评估 + 综合尝试超时，非实测利用率）** | `evidence/vision/native_resource/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md`：静态 ≈846k FF > 器件 682,560 FF、conv2 2304 MAC/拍 > DSP 1248；native SPS OOC probe >25 min 未收敛、无 util/timing 报告。**不报实测 LUT/FF/BRAM/DSP/WNS 数字** |
| 参考路线 P0（基线冻结/哈希/向量/口径） | **pass** | `evidence/vision/ref_route/baseline_hashes.json`、`sps_small_check_meta.json`、`sps_native_meta.json` |
| 参考路线 P1（参考式 SPS small，16 beats bit-exact） | **pass（四配置；以 `DELIVERY_SUMMARY.md` §3 为准）** | `evidence/vision/ref_route/`（`p1_r2_small.log` 等）；历史 53/256 nibbles FAIL 作基线保留（`tb_sps_ref_small.log`）。**不再复述时间戳** |
| 参考路线 P1 native SPS 仿真（3072 beats） | **pass（3072 beats；以 `DELIVERY_SUMMARY.md` §3 为准）** | 主线 `build/spikformer/vision_ref_20261031/sim/` 与 `evidence/vision/ref_route/`；量级见 `docs/VISION_CASE.md` |
| 参考路线 P2 native 整链 bit-exact | **not_run** | 见 `DELIVERY_SUMMARY.md` §3 |
| 参考路线 P3 同源 OOC 综合/实现（5ns routed） | **not_run** | 见 `DELIVERY_SUMMARY.md` §3 |
| 参考路线 B 轮 block 分块供权 | **not_run** | 按任务书条件触发，未触发 |
| 图像平台/板测（P4） | **not_run** | 无图像 bit/HWH；板卡离线 |
| 图像精度（CIFAR-10） | **无 / not_run** | 上游仓库不带 CIFAR-10 checkpoint，只有 fixture 权重 → 只做功能冒烟，**不报精度** |

**图像线边界（必须随引用）**：不是演示；**没有任何精度数字**；行为级回放 pass ≠ native/板级结果；native 资源门 fail 是静态评估；平台/板测 `not_run`。

---

## 5. 快照 RTL 与冻结证据的一致性

- 快照 `rtl/` 复制的时刻（2026-10-31 16:36）是当时主线状态；NLP 冻结证据绑定 2026-10-29 22:26 的 18 文件哈希。
- 比对结果：**16 SAME / 2 DIFF**——`rtl/ops/spiking/sps/spiking_sps_top.sv`、`rtl/ops/spiking/sps/spiking_eyeriss_pe.sv` 被图像线并行改动（未修改 NLP 数据通路）。
- 结论：NLP 功能 RTL **16/16 未变**（NLP top 不实例化 SPS）；IP 包 `ip/src` 保留冻结态；详见 `evidence/nlp/rtl_hash_check_20261031.txt` 与 `docs/PORTABILITY.md`。
- 该差异**不影响** NLP 的 ooc/ip/回放证据（它们绑定冻结哈希并由 build_identity 记录）。

---

## 6. 明确 `not_run` / 未验证清单（汇总）

1. 图像线：P2 native 整链、P3 OOC、B 轮、平台/板测 = `in_progress / not_run`（P1 small 四配置与 P1 native SPS 3072 beats 已 PASS，最终口径见 `DELIVERY_SUMMARY.md` §3）。
2. 平台完整协议仿真矩阵：`not_run`（仅 CSR/超时 smoke 7/7）。
3. 软复位排空在途 DMA 的完整协议：未实现（现靠重载 overlay 绕过）。
4. 平台写完成非 512 B 长度 / 4 KB 边界：未验证。
5. 平台 CDC 1 Critical（CDC-10）未重新收敛；`sys_rst`/`DDR4 reset_n` I/O 约束缺口未补。
6. OOC 的 HD.CLK_SRC、真实上游接口 min/max、集成顶层时序：未验证（OOC 0 延迟输入模型对 setup 乐观）。
7. 板端现场复跑：`not_verified`（原始 JSON 缺失、板卡离线）；板测不宣称 pass。
8. r7 bit 的 SUMS +64 偏移：RTL 未修，只有软件补偿（`ROT=64`）；需板测复核。
9. NLP 公开 test.tsv 无标签：test accuracy 不报。
10. 快照未重跑 Vivado（规范禁止）；所有物理结果引用 2026-10-30 主线实测证据。

---

## 7. 证据索引（快照内优先；主线路径用于复核）

| 内容 | 快照路径（D:/DL_FINAL/SNN/） | 主线来源 |
|---|---|---|
| NLP 门/证据清单 | `evidence/nlp/final_gates.json`、`evidence/INDEX.md` §2 | `build/spikformer/nlp_48h_20261029_1915/` |
| NLP 数学/准确率 | `evidence/nlp/{math_tests_m1.log,math_tests_m1.json,train_real50.json,export_final_real50.json,train_realweight_fidelity_real50.json}`；`math_lead_check.json`/`intref_fix.json` 未复制（见 `evidence/INDEX.md` §6） | 同左 `reports/` |
| NLP 回放 | `evidence/nlp/replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json` | 同左 `reports/` |
| NLP OOC / IP | `evidence/nlp/final/{reports/ooc_20261030_120804.json,ip/ip_manifest.json}` | `build/…/final/nlp_real_5stage_r4_holdfix/` |
| 平台搭建 | `evidence/platform/platform_build_20261030.json` | `D:/spk_platform/reports/platform_build_20261030.json`、`logs/`、`sim/tb_snn_csr/` |
| 平台 r6/r7 时序 | `evidence/platform/{platform_timing_summary_20261031.txt,r7_sumsfix_route_status.rpt,CDC_FIX_R6_RESULT_20261030.md,r6_final_route_status.rpt,r6_final_util.rpt}` | `D:/spk_platform/reports/` |
| 平台 bit/HWH | `board/{snn_platform.bit,snn_platform.hwh,MANIFEST.sha256,MANIFEST_DEMO_20261031.sha256}` | `D:/spk_platform/board/`、`project_1/snn_platform.xsa` |
| 板端记录 | `board/README_BOARD.md`；`evidence/platform/board/board_run_local_attempt.log`（本机失败尝试，非板测证据） | `D:/spk_platform/board/`；handoff §1 |
| 图像线 | `evidence/vision/`（toolchain/behavior/native_resource/ref_route）；`docs/VISION_CASE.md` | `D:/dl_snn/build/spikformer/{vis_native,vision_ref_20261031}`、`sim_work/vis_sps/` |
| 演示手册 | `docs/DEMO_NLP.md`；`examples/nlp_sentiment_b1/README.md` | — |
| RTL 冻结一致性 | `evidence/nlp/rtl_hash_check_20261031.txt` | 主线 `build/…/reports/rtl_freeze.json` |
