# 图像线现状与证据（partial；不是演示、无精度、未上板）

> **图像线仍在推进，最终状态以 `docs/DELIVERY_SUMMARY.md`（Lead 18:50 前定稿）为准。**
> 本文件记录 2026-10-31 收拢时刻（约 17:00）的快照状态；之后的变化以 Lead 定稿为准。

## 0. 一句话结论

图像识别案例是**诚实 partial 收录**：

- 工具链（native 配置）**编译 pass**，参数/调度产物齐全，但只有 **fixture 权重**；
- 行为级小几何回放**整链 ALL PASS**（tb_block_replay / tb_vis_sps / tb_vis_replay）；
- native 资源门 **fail**（基于旧 SPS 结构的静态评估：≈846k FF、2304 MAC/拍；native OOC 综合 >25min 超时中止，无实测报告）；
- 参考工程路线 **P0 pass、P1 small 四配置 pass（T2/T4/H32-T4 proxy/H32-T4-TC2=8，2026-10-31 16:56–16:59）、P1 native SPS 正式门 pass（3072 beats，17:01）、P2 单图整链 pass（3072 block-out + 384 sums bit-exact，17:26）/ 两图连续 fail（known limitation：sps 参考模块第二次 start re-arm 缺失，冻结未修）**；P3 OOC **fail（synth elaboration，17:39；sps_psum.sv:115 ambiguous clock；impl/routed/资源/时序 not_run）**、B 轮/P4 平台板测 **not_run**；
- 图像平台/板测 **not_run**；上游无 CIFAR-10 checkpoint → **无精度**；native 资源门仍是旧结构的**静态评估 fail**（新参考式结构的资源/时序未测：P3 OOC synth fail，未到 impl/routed）。

**不得当演示、不得报精度、不得把行为级回放当 native/板级结果。**

## 1. 工具链（vis_native 配置编译）

- 配置：`examples/spikformer/config/model_native_b1.json` —— 32×32×3 输入，E=384、H=12、HD=32、MLP=1536、T=4、depth=1、10 类。
- 编译命令（源：`docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §2.1）：

      E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
        --config examples/spikformer/config/model_native_b1.json \
        --fixture-seed 7 --out-dir build/spikformer/vis_native

- 产物（已存在，pass）：`build/spikformer/vis_native/` 下 `manifest.json`、`resolved_config.json`、`schedule.json`、`numeric_contract.json`、`generated/spikformer_config.vh`、`params/*.bin/*.coe`；目录约 13MB（快照只收录小文件，参数二进制未复制）。
- 关键派生（`resolved_config.json` derived / `schedule.json`）：
  - SPS：H=W=32、C0=3、C1=96、C2=384、TOKENS=64；`conv1_beats=324`、`conv2_beats=41472`；
  - block：`block_w_beats=222912`（1,769,472 B）、`block_in_beats=3072`、descriptor 4 beats；
  - `model_kind=development_fixture`、`checkpoint.kind=missing`。
- 状态：**pass（编译/产物）**；注意这是 fixture 权重，不是训练权重。
- 证据：`evidence/vision/toolchain/`（resolved_config.json、schedule.json、manifest.json、numeric_contract.json、spikformer_config.vh、spikformer.xdc）。

## 2. 行为级（小几何回放，pass）

- 回放几何：H=W=8、C0=3、C1=16、C2=32、T=2；block E=32、H=1、HD=32、HIDDEN=64、T=2、TOKENS=4、TL=8、TC=8、RW=4。
- 2026-10-31 13:19 最新回归（ModelSim 2020.4，快照内日志）：
  - `tb_block_replay`：**ALL PASS（16 beats）** —— `evidence/vision/behavior/vsim_block_r.log`
  - `tb_vis_sps`：**ALL PASS（16 beats）** —— `evidence/vision/behavior/vsim_sps_r.log`
  - `tb_vis_replay`（SPS→block→sums 整链）：**ALL PASS（32 channels）**，TOKEN CHECK fails=0，block weight=1136、block output=16，TSUM 4 6 12 16 24 16 16 3 —— `evidence/vision/behavior/vsim_replay_r.log`
- 原始日志位置：`D:/dl_snn/sim_work/vis_sps/vsim_{block_r,sps_r,replay_r}.log`。
- RTL bug 修复史（全部有仿真证据，均在小几何回放范围内）：
  1. **vis_top/SPS 工具链 9 个 bug**（13:10 前）：S_LDT 权重消费条件（每 tile 少 1 拍）、W2B 只算单 token 位置（应含全部 token）、`st` 位宽 3→4、`blk_wdata` 未驱动、`capi` 位宽、`acc` 复位清零、S_RUN 权重指针推进、TB 主机激励顺序、tb_vis_sps 比较格式（假 FAIL）。记录：`evidence/vision/native_resource/VISION_LINE_PROGRESS_20261031.md`。
  2. **block 输出失配根因**：Python golden 落后于 2026-10-29 block 合约——RTL 在 proj/fc2 后先 LIF 再把 0/1 spike 加回残差，旧 golden 直接把完整线性累加值加回。**只改 `scripts/spikformer/gen_block_replay.py`，未改 RTL**；修复前 240/256 nibble mismatch → 修复后 ALL PASS。证据：`evidence/vision/behavior/vsim_block_baseline_20261031.log`（FAIL 基线）、`vsim_block_fixed_model_20261031.log`（PASS）、`vsim_block_sps_tokens_fixed_20261031.log`、`vsim_block_stage_probe_20261031.log`（QKV/AV/两次权重重载全匹配）、`vsim_replay_fixed_model_20261031.log`（整链）。
- 边界：行为级通过 ≠ native 几何通过 ≠ 综合/时序/板级通过。

## 3. native 资源门（静态评估 + 综合尝试，fail / 未产出）

native 几何：32×32×3 → C1=96、C2=384；block E=384、H=12、HD=32、MLP=1536、T=4、TOKENS=64、depth=1。器件 xczu15eg-ffvb1156-2-i（LUT 341,280 / FF 682,560 / BRAM36 744 / DSP48 1248）。

**旧 SPS 结构静态评估（推导，非综合实测）**：

| 项 | 静态评估 | 口径 |
|---|---|---|
| 寄存器数组 | `f1` 1024×384 bit = 393,216 bit（S_P2 组合异步读→FF/LUTRAM）；`prow2` 256×1536 bit = 393,216 bit（drain 异步读）；f0/wbank1/wtile2/bias 等 ≈5.5 万 FF；**合计 ≈846k FF ≈ 器件 124%** | 推导（bit 数直接视作 FF） |
| conv2 数据通路 | `LANES*T*TC2*9 = 8*4*8*9 = 2304 MAC/拍`；每 (lane,t) 为 72 项串行加法链，超 DSP 1248 且 5ns 组合链过不去 | 推导 |
| 结论 | 当前 SPS 只能跑小回放几何；native 必须改微架构（BRAM 化 + 同步读流水 + MAC 时间复用），不是调参数能解决 | 推导 |

> 该评估的估算口径（数组 bit 数视作 FF、门控加权视作 DSP、ROM 容量视作流量等）已被
> `docs/VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md` §3 逐项纠正；本文保留为**历史静态推导**，
> 不作为综合实测数字。**不报 LUT/FF/BRAM/DSP/WNS 实测值。**

**native OOC 综合实测尝试**：

- SPS native probe（`rtl/probe/sps_native_probe.sv` + `scripts/synth_ooc_probe.tcl`）：**>25 分钟未收敛，进程峰值 7.1GB，无 `util.rpt`/`timing.rpt` 产出 → 中止**。报告目录 `reports/native_probe_sps/` 为空。
- block native probe：跑到约 13 分钟暂停；Vivado 日志已见 `Implemented Block Ram Cascade chain of height 8 and width 36/18 for RAM "spiking_block_top:/fc1mem_reg"`（大权重 RAM 的 BRAM 推断路径正常）；**未产出 util/timing 报告** → 方向可行但**未跑完**。
- 结论：native 资源门（**旧 `spiking_sps_top` 结构**）**fail**（静态评估判定不可综合；综合尝试超时无报告）。
- 参考式新结构（§4）的 P1 native SPS **功能/数值仿真已 PASS**（3072 beats，17:01），但**资源/时序仍未测**（P3 OOC synth fail，未到 impl/routed）→ 不能据此判资源门 pass。
- 证据：`evidence/vision/native_resource/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md`。

## 4. 参考工程路线（reference route）

目标：复用参考实现模块（conv_layer1/Multi_add_unit、simple_eyeriss_pe_unit/array、psum_callback/top/ram、SpikesTmpRam、maxpool 行结构），做 BRAM 化 + 同步读流水 + MAC 时间复用。新 RTL 已写并注册 `source_manifest.json v20261031.1`（`spiking_feature_bram.sv`、`spiking_rgb_encoder.sv`、`spiking_eyeriss_array.sv`、`spiking_sps_psum.sv`、`spiking_sps_ref_ctrl.sv`、`spiking_sps_token_adapter.sv`、`spiking_sps_ref_top.sv`、`spikformer_vis_native_top.sv`；旧 `spikformer_vis_top.sv` 保留 legacy 基线）。**控制器 `spiking_sps_ref_ctrl.sv` 最新 sha256 = 728337f1…（16:56 最终修复版）；16:44 的 e6008ad6… 是中间版本，已作废。**

| 门 | 状态 | 证据 |
|---|---|---|
| P0 基线冻结/哈希/run/布局协议 | **PASS** | `evidence/vision/ref_route/baseline_hashes.json` |
| P0 bias_beats 口径统一 | **PASS**（代码已改；未跑全回归） | VISION_REFERENCE_ROUTE_RESULT §1 |
| P0 阶段 golden（small+native） | **PASS**（small 与 legacy exp.hex 16/16 全等；native 向量 ld1=324 / bias=384 / uniq=41472 / pix=1024 / exp=3072） | `evidence/vision/ref_route/sps_small_check_meta.json`、`sps_native_meta.json` |
| P1 参考式 SPS small（四配置） | **PASS（2026-10-31 16:56–16:59，ctrl sha256 728337f1…）**：T=2 16/16 beats、T=4 16/16 beats、H32-T4 proxy 256 beats、H32-T4-TC2=8 256 beats，均 bit-exact；capture=64/1024、flush=16/64 structural ok | `evidence/vision/ref_route/p1_r7_small_t2.log`、`p1_r7_small_t4.log`、`p1_r7_h32_t4.log`、`p1_r8_h32_t4_tc2_8.log` |
| P1 small 历史中间版本 | PASS（16:44，ctrl sha e6008ad6…；中间版本，已作废） | `evidence/vision/ref_route/p1_r2_small.log`、`p1_r2_compile.log` |
| P1 small 历史 FAIL 基线 | FAIL 53/256 nibbles（16:06，已保留作历史） | `evidence/vision/ref_route/tb_sps_ref_small.log` |
| P1 native SPS 正式门（3072 beats） | **PASS（2026-10-31 17:01）**：H32/W32/C1-96/C2-384/T4/TC2=8；3072 beats 全等、98,304 nibbles hamming=0、capture=12288/flush=768 结构断言 0 fail、cycles=5,573,066；ctrl sha256 728337f1…；r1–r3 失败运行记录在 invalidated_runs（不冒充 PASS）。可复核命令/独立 python 复核：`verify_r4_pass_imgnative.py`（快照副本 `native_gate_verify.py`）+ `tb_sps_native_imgnative_r4_diff_summary.txt`（exact 3072/3072、hamming=0）；正式向量目录 `vectors/sps_native_tc2_8/`（meta 快照 `sps_native_tc2_8_meta.json`），旧 `vectors/sps_native`（TC2=4）不配对、仅历史 | `evidence/vision/ref_route/tb_sps_native_imgnative_r4_20261031.log`、`tb_sps_native_imgnative_r4_diff_summary.txt`、`native_gate_result_20261031.json`（绑定 ctrl/向量/哈希）、`native_gate_evidence_sha256.txt`（51 行证据清单）、`native_gate_verify.py`、`sps_native_tc2_8_meta.json`、`native_gate_rtl_hashes.txt`、`native_gate_compile.log` |
| P2 native 整链 | **单图 PASS / 两图连续 FAIL（known limitation）**：单图（2026-10-31 17:26，+IMG2=0）SUMS PASS img0（384 sums）+ BLKOUT PASS img0（3072/3072 beats bit-exact）；TB 终行“2 images”文案瑕疵已由 TB 新版 8aead49c… 修正。两图连续（17:33:42）：同一次运行 img0 全 PASS，img1 卡死（top=N_RUN / blk=S_TOK / sps=C_LDW1）到 watchdog 120M cyc（5:19 wall）→ 根因 `spiking_rgb_encoder.sv` 的 w1_done/bias_done/pix_done/fwin_loaded 与 `spiking_sps_ref_ctrl.sv` 的 enc_started 为 sticky、只在 s_rst 清，第二次 start 不 re-arm（任务书 §5.2 未实现项），本轮 RTL 冻结未修；**不是 TB/golden 问题**。输出反压/输入气泡 not_run。**功能仿真，非资源/时序/板级。** 修复前单图 BLKOUT PASS/SUMS FAIL 382（N_RUN 累加 bug，修复仅 beat_add 组合求和 + 每 lane 单次 acc 累加，control 顶层 8ece767a…；sps ctrl 728337f1…/block_top 841df601… 冻结） | `evidence/vision/ref_route/tb_vis_native_chain_p2fix_img1_20261031.log`（单图 PASS，4ffa5ee8…）、`tb_vis_native_chain_p2fix_img2_20261031.log`（两图 FAIL，9dc10409…）、`tb_vis_native_chain_p1_img1_20261031.log`（修复前，c073917f…）、`VISION_P2_CHAIN_PROGRESS_20261031.md`（98a0d2f1…）、`compile_native_chain_p2_fix_20261031.log` |
| P3 同源 OOC 综合/实现（5ns routed） | **FAIL（synth elaboration，2026-10-31 17:39:25）**：stage=all 首次整顶 OOC synth 失败 6 errors，根因 `[Synth 8-91] ambiguous clock in event control` @ `rtl/ops/spiking/sps/spiking_sps_psum.sv:115`（伴 `case not full/no default` line 119，psum.sv 冻结未改 sha256 8bb4edc2…）；连带 `spiking_sps_psum`/`spiking_sps_ref_top`/`spikformer_vis_native_top`/`vis_native_probe` synth 失败。**impl/routed/资源/时序 not measured（不报 LUT/FF/BRAM/WNS）**。sim 功能门 P1/P2 不受影响；这是首次综合参考式 RTL 暴露的兼容性问题，下一轮修复路径见 `VISION_NATIVE_OOC_PROGRESS_20261031.md` | `evidence/vision/ref_route/ooc_native/vivado.log`（55db145a…）、`facts.json`（17:44 更新版，gate.p3_native_ooc=fail）、`rtl_sha256_at_synth.txt`、`constraints_ooc.xdc`、`heavy_wait_latest.json`、`VISION_NATIVE_OOC_PROGRESS_20261031.md` |
| B 轮 block 分块供权 | **not_run**（按任务书条件触发） | — |
| P4 平台/板测 | **not_run** | — |

- native 门证据绑定：`evidence/vision/ref_route/native_gate_result_20261031.json` 记录 ctrl sha256 728337f1…、TC2=8 向量哈希、3072 beats/98304 nibbles/cycles、以及 r1–r3 作废运行与 not_run 清单；`VISION_REF_P1_PROGRESS_20261031.md`、`VISION_NATIVE_SIM_PROGRESS_20261031.md` 为进度记录。
- 参考路线实测资源/时序：**未测**（P3 OOC synth fail，未到 impl/routed）→ 不报 LUT/FF/BRAM/DSP/WNS；P1 native PASS 只是**功能/数值仿真**，不代表资源/时序可行。
- 容量推导（仅用于 B 轮触发判断，**非实测**）：conv2 唯一权重 331,776B（下限 72 BRAM36）；block 六矩阵 1,769,472B（下限 384 BRAM36）；SPS 双 bank 256KiB；psum 8×256×84b。
- 流量（调度定义，**非实测**）：SPS 每图 host 42,180 beats 约 337KB unique；block 权重流 222,912 beats；SPS→block 3,072 beats。旧"21.2MB/图"与 `bias_beats=30` 已按任务书作废。
- 复现命令（D:/dl_snn）：

      cd D:/dl_snn/sim_work/vis_sps
      vsim -c -do compile_ref_small.do -l compile_ref_small.log
      vsim -c -work work tb_sps_ref_small -do "run 100ms; quit -f"
      vsim -c -work work_native tb_sps_native -do "run -all; quit -f"

  期望输出：small `ALL PASS tb_sps_ref_small (16 beats; capture=64 flush=16 structural ok)`；native `ALL PASS tb_sps_native (3072 beats; capture=12288 flush=768 structural ok)`（日志：`build/spikformer/vision_ref_20261031/sim/p1_r7_*.log`、`tb_sps_native_imgnative_r4_20261031.log`）。
- 注意：P1 small（16-beat）与 P1 native SPS（3072-beat）PASS 都只是**功能/数值仿真**，**不等于** native 整链（P2）/ OOC（P3 synth fail）/ 平台/板级。

## 5. 平台 / 板测

- 图像平台 top（`spikformer_vis_platform_top`）、平台 BD / HWH / XSA、板测：**全部 not_run**。
- 复用 D:/spk_platform 骨架仍是计划（未动平台工程；NLP 演示包未碰）。
- 证据：`evidence/VISION_REFERENCE_ROUTE_RESULT_20261031.md` §7（明确未测项与 B 轮/P4 交接）。

## 6. 精度（CIFAR-10）：无

- 上游仓库**不带 CIFAR-10 checkpoint**；本项目图像线只有 fixture 权重，没有训练、没有精度数字。
- native 向量是 golden 功能向量，不是模型权重。
- 证据：`evidence/vision/native_resource/SPIKFORMER_VISION_HANDOFF_20261031.md` §2/§5.3（"上游仓库不带 checkpoint，要自己训 CIFAR-10 的 Spikformer-S；否则只能 fixture 冒烟，不能报精度"）；`evidence/vision/toolchain/resolved_config.json` 的 `checkpoint.kind=missing`。

## 7. 结论口径（必须照抄，禁止拔高）

- 图像线 = **partial**：工具链编译 pass；行为级小几何整链回放 pass；参考路线 **P0 pass、P1 small 四配置 pass、P1 native SPS 正式门 pass（3072 beats，17:01）、P2 单图整链 pass（17:26）/ 两图连续 fail（known limitation）**；**P3 OOC fail（synth elaboration，17:39:25；根因 sps_psum.sv:115 ambiguous clock；impl/routed/资源/时序 not_run，不报 LUT/FF/BRAM/WNS）**、输出反压 ABI gap not_run、B 轮/P4 平台板测 not_run；native 资源门仍是**旧结构静态评估 fail**（新结构资源/时序未测）；无 CIFAR-10 checkpoint → **无精度**。
- **不是演示、无精度、未上板。**
- 不得把行为级回放当 native 结果；不得把 fixture 当训练权重；不得把 OOC/综合当集成/板级；负 WNS 记 fail，不降频、不加时序例外。

## 8. RTL 冻结哈希核对（快照时刻 2026-10-31 17:00；快照 rtl/ 于 17:00 刷新、18:30 再刷一次，以最终包为准）

NLP 冻结记录 `build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json`（18 文件，frozen_at 2026-10-29T22:26）。

当前主线 vs 冻结：**16 match / 2 changed**：

| 文件 | rtl_freeze sha256 | 当前主线 sha256 | 当前主线 mtime | 原因 |
|---|---|---|---|---|
| `rtl/ops/spiking/sps/spiking_sps_top.sv` | 484e735948b3…ede60 | 84929737178d…2917f | 2026-10-31 12:52 | 图像线（参考路线）修复 S_LDT 等 |
| `rtl/ops/spiking/sps/spiking_eyeriss_pe.sv` | 1f1813ce3a0c…e536c | 0c64d70d69e9…4e468 | 2026-10-31 15:31 | 图像线 PE 单窗语义修复（该 PE 只被新阵列使用） |

- **快照 `rtl/` = 当前主线 superset（28 源，含图像线新文件；复制 16:36，17:00 已刷新 `spiking_sps_ref_ctrl.sv`，18:30 再刷一次）**：不是 2026-10-30 NLP 冻结态。
- **NLP 的 OOC / replay / IP 证据绑定 2026-10-30 冻结版本**：`nlp/final/ip/src` 的 19 个源文件与冻结哈希 18/18 一致，可作历史证据；用快照 `rtl/` 重新综合 NLP 会得到与 2026-10-30 OOC 不同的网表。
- 图像线仍在改 RTL：控制器 `spiking_sps_ref_ctrl.sv` 最新 sha256 728337f1…（旧 e6008ad6 中间版本作废）；P2 修复又改了 `rtl/ops/spiking/control/spikformer_vis_native_top.sv`（sha256 8ece767a…，17:22，vision-only，不在 NLP 冻结 18 文件内）；快照 `rtl/` 的 18:30 刷新须包含此版本。最终以 Lead 定稿为准。
- 证据：`evidence/nlp/rtl_hash_check_20261031.txt`（派生核对表，逐文件 18 行）。

## 9. 证据索引（快照内）

| 证据 | 用途 |
|---|---|
| `evidence/VISION_REFERENCE_ROUTE_RESULT_20261031.md` | 参考路线结果快照（16:11 版本；P1 当时 FAIL，现已被 p1_r2_small.log 的 PASS 取代） |
| `evidence/vision/toolchain/` | native 配置编译产物（manifest/resolved_config/schedule/numeric_contract/config.vh/xdc） |
| `evidence/vision/behavior/` | 行为级回放日志（block/SPS/整链 PASS + 修复前 FAIL 基线） |
| `evidence/vision/ref_route/` | P0 基线哈希、small/native 向量 meta（含正式门 `sps_native_tc2_8_meta.json`）、P1 small 历史 FAIL + 四配置 PASS 日志（p1_r7_*/p1_r8_*）+ P1 native PASS 日志/diff 复核/gate JSON/证据清单/复核脚本/哈希绑定 + P2 单图 PASS/两图 FAIL 日志与 golden 向量 + P3 OOC fail 证据（`ooc_native/`）+ 进度记录（VISION_REF_P1_PROGRESS、VISION_NATIVE_SIM_PROGRESS、VISION_P2_CHAIN_PROGRESS、VISION_NATIVE_OOC_PROGRESS）+ 16:44 中间版本日志 |
| `evidence/vision/native_resource/` | native 资源问题交接（综合尝试/静态评估）、行为级修复记录、图像线交接（无 checkpoint） |
| `evidence/INDEX.md` | 全部证据 + sha256 + 来源路径 |

## 10. 与 NLP 演示线的边界

- NLP 演示线（见 `docs/DEMO_NLP.md`）是**已闭环的真实语料训练 + 真实 checkpoint + 回放 0 mismatch + 5ns routed OOC pass + IP 包**；图像线不是演示线。
- 图像线的工具链/行为级/参考路线结果不得混入 NLP 的状态表，也不得用于支持"图像识别演示"的表述。
