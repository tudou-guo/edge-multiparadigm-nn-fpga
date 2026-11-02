# SNN 发布快照 0.1.0 — Spikformer SNN 加速算子库与工具链

- 版本：`snn-toolchain-0.1.0`（见 `VERSION`）
- 整理规范：`docs/RELEASE_SCOPE.md`
- 源码复制时刻：**2026-10-31 16:36 CST**（sha256 清单：`evidence/COPY_SHA256.txt`）
- 来源：**D:/dl_snn**（只读拷贝，不改主线）、**D:/spk_platform**（平台/上板包）
- 发布根：**D:/DL_FINAL/SNN**（与 D:/DL_FINAL/DL 并列，互不修改）

## 1. 这是什么

把 D:/dl_snn 中**已闭环的 NLP 情感分类案例与工具链**收拢成的**自包含、可拷贝、可复核**发布快照：
配置＋权重 → SNN 参数/RTL 配置/IP 工程/调度/PYNQ 包。目标器件 `xczu15eg-ffvb1156-2-i`，Vivado 2021.1，5 ns。

- **演示主线 = NLP 情感分类**：真实语料训练 → 真实 checkpoint 导入 → 编译/参数/RTL 配置/调度 →
  功能回放 0 mismatch → 5 ns routed OOC → IP 源包 → 平台 r7 bit → 板端样本比对。
- **图像识别案例 = 诚实 partial 收录**：不作演示、不报精度（`docs/VISION_CASE.md`，由 evidence 子代理维护）。

## 2. 目录结构

| 路径 | 内容 |
|---|---|
| `README.md` `QUICKSTART.md` `RELEASE_NOTES.md` `VERSION` `environment.md` | 发布说明、3 条命令、收录/未收录、版本、环境 |
| `docs/` | 规范 + 用户指南 + 算子库 + 可移植性 + 数值合同 + 平台/证据索引（各子代理维护） |
| `tools/` | 工具链源码副本：`toolchain.py` + `backends/spikformer/` + `spikformer_train/` |
| `rtl/` | RTL 源码副本：`ops/spiking/**` + `operator_library.json`；绑定源清单 `ops/spiking/source_manifest.json`、上游映射 `ops/spiking/UPSTREAM_MAP.json` |
| `runtime/` | PYNQ 运行包副本 `runtime/spikformer/` |
| `verification/` | 仿真 TB（`.v`）+ 纯 Python 短例副本，状态见 `verification/README.md` |
| `scripts/` | 精选脚本副本（spec §3 列表），路径统一相对快照根 |
| `examples/` | 案例与状态表、配置、真实 checkpoint 资产（docs 子代理维护） |
| `board/` | 上板包副本（bit/hwh/run 脚本/demo 资产/vec/MANIFEST/README_BOARD） |
| `evidence/` | 关键证据只读快照 + `INDEX.md`（含 sha256 与主线来源路径） |
| `build/_quickstart_check/` | 本快照验收产物（可删，按 QUICKSTART 重建） |

## 3. 快速开始

在快照根（`D:/DL_FINAL/SNN`）执行：

```bash
# 1) 环境检查
E:/miniconda/python.exe -c "import sys, numpy; print(sys.version.split()[0], 'numpy', numpy.__version__)"

# 2) 编译 NLP（fixture 权重，无需 checkpoint）
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --fixture-seed 1 --out-dir build/_quickstart_check

# 3) 纯 Python 数值自检
E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py
E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py
```

期望：`kind=development_fixture` / `FAILURES: 0` / `RESULT: 8/8 PASS -> PASS`。详见 `QUICKSTART.md`。
板端演示不在本机执行，见 `board/README_BOARD.md`。

## 4. 状态摘要（只写实际执行结果）

### 4.1 NLP 演示主线（门状态）

| 门 | 状态 | 证据（快照/主线） |
|---|---|---|
| 数值合同 math_contract | **pass** | `evidence/nlp/math_tests_m1.json`；主线 `build/spikformer/nlp_48h_20261029_1915/reports/math_tests_m1.log` |
| 目标整链回放 target_e2e | **pass**（0 mismatch） | `evidence/nlp/final_gates.json`；主线 `.../reports/replay_e2e_5stage_r2_noforce.json`（cycles 7,084,772） |
| 真实权重回放 real_weight_replay | **pass**（0 mismatch，3 样本） | 主线 `.../reports/replay_real_5stage_r2.json`（cycles 10,627,157） |
| 整数域精度 integer_accuracy | **pass**（dev 0.7592 / holdout 0.7562） | `evidence/nlp/final_gates.json`；主线 `.../reports/train_real50.json` |
| 5 ns routed OOC | **pass**（Setup WNS +0.135 ns / Hold WHS +0.023 ns，0 failing） | `evidence/nlp/final/reports/ooc_20261030_120804.json` |
| IP 源包 | **pass**（timing_qualified=true） | `evidence/nlp/final/ip/ip_manifest.json` |
| 平台 r7 bit | **bit 已生成**（r7-sumsfix，Setup WNS +0.051 / Hold WHS +0.011，0 failing） | `board/snn_platform.bit` + `board/MANIFEST.sha256`；说明见 `board/README_BOARD.md` |
| 板端 3 样本比对 | **记录级 pass（3/3 match=True）**；原始板端 JSON 未随包、板卡离线 → 快照内**未复核（not_verified）** | 主线 `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2、`board/README_BOARD.md` §3；`docs/PLATFORM_AND_STATUS.md` §3 |

> 口径（两说并列，不互相覆盖）：`evidence/nlp/final_gates.json` 是 2026-10-30 13:10 冻结的 48h 门状态（platform/bit_hwh/board = not_run）；
> 平台 r7 bit 与板端 3/3 match 发生在其后（2026-10-30 23:10 之后），记在平台报告与 `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2。
> 快照内**无原始板端 JSON**、板卡离线，本包**不宣称已复核**；最终措辞以 `docs/DELIVERY_SUMMARY.md` 为准。

### 4.2 图像识别案例（partial，不作演示）

| 项 | 状态 |
|---|---|
| 工具链编译（vis_top / vis_native） | pass（生成链） |
| 行为级整链回放 | pass（主线 `sim_work/vis_sps/vsim_replay_r.log`，见 `evidence/vision/`） |
| native 资源门 | **fail**（基于旧 SPS 结构的静态评估：≈846k FF、2304 MAC/拍；native OOC 综合 >25 min 超时中止，无实测报告） |
| 参考式 SPS small 门（P1，四配置） | **pass**：T2 / T4 各 16/16 beats bit-exact，H32-T4 proxy（TC2=4 与 TC2=8 各 256 beats）全 bit-exact，结构断言 0 fail（ctrl sha256 `728337f1…`；Lead 已独立复跑 T2/T4）。历史 FAIL（53/256 nibble）为基线 |
| P1 正式 native 门（H32/C1-96/C2-384/T4/TC2=8，3072 beats） | **pass**：3072/3072 beats 全等、98,304 nibbles hamming=0，capture=12288/flush=768 结构断言 0 fail，5,573,066 cycles（ctrl sha256 `728337f1…`；Lead 已用独立 TB 副本 + 独立 ModelSim work 库复跑确认）。r3 的 1,142 nibbles 残差来自旧 ctrl/旧向量对照，已作废 |
| P2 native 整链 bit-exact | **pass（单图）**：3072 block-out beats + 384 sums 全部 bit-exact；过程中发现并修复 native top 真 RTL bug（N_RUN 同 acc 多 NBA 只累加 t=3，改 `beat_add` 组合求和后归零；control 顶层 sha256 `8ece767a…`）。Lead 独立 TB 副本 + 独立 work 库复跑确认；两图连续/输出反压见 `docs/VISION_CASE.md` |
| P2 两图连续 | **fail（已知 RTL 限制）**：img0 全 PASS；img1 卡死（encoder/ctrl 的 done/enc_started 为 sticky、只在 s_rst 清 → 第二次 start 不 re-arm，对应任务书 §5.2 未实现项）。非 TB/golden 问题，本轮未修；输出反压/`+BPS` not_run |
| P3 OOC / B 轮 / P4 平台板测 | **in_progress / not_run**：用户关闭外部 Vivado 后，17:35 起整顶 OOC（stage=all）已在共享锁内启动；B 轮/P4 平台板测 not_run |

> 图像线在收拢期间仍被并行修改：**快照以复制时刻哈希为准**，详见 `docs/PORTABILITY.md` §4 与 `docs/VISION_CASE.md`。

### 4.3 本快照自检（2026-10-31 本次执行）

| 检查 | 结果 | 说明 |
|---|---|---|
| 快照根 compile（fixture） | **pass** | `build/_quickstart_check/manifest.json`，kind=development_fixture，28 个 param blob |
| 快照根 compile（真实权重） | **pass** | `build/_quickstart_real/manifest.json`，kind=real_corpus，**54 个 param blob，与最终冻结 run `final/nlp_real_5stage_r4_holdfix` 的 params 字节级一致** |
| 迁移目录 compile 一致性 | **pass** | 拷到 `D:/_snn_reloc_check` 后 params 哈希集合与 `manifest_sha256` 均一致（临时目录已删除） |
| `verification/sim/spikformer/test_short.py` | **pass** | 15 项 PASS，`FAILURES: 0` |
| `tools/spikformer_train/test_integer_ref.py` | **pass** | `RESULT: 8/8 PASS -> PASS` |
| NLP 冻结 RTL 18 文件哈希 | **16/18 SAME，2 DIFF** | DIFF：`sps/spiking_eyeriss_pe.sv`、`sps/spiking_sps_top.sv`（图像线并行改动，NLP datapath 不实例化 SPS）；详见 `docs/PORTABILITY.md` §4 |
| 详细记录 | — | `evidence/ACCEPTANCE_CORE.md` |

## 5. 范围声明

**包含**：工具链源码、RTL 源码（复制时刻）、PYNQ 运行包、仿真 TB + 纯 Python 短例、精选脚本、配置、案例资产、上板包、关键证据。

**不包含 / 未执行（如实声明）**：
- 不跑 Vivado 综合/实现/bitstream，不访问板卡（192.168.2.99 离线），不占 heavy lock 跑长任务。
- 不含 routed DCP / synth 中间目录 / 完整 build 树；重建命令见 `RELEASE_NOTES.md`。
- 板端原始结果 JSON 未随包（`board/README_BOARD.md` 给运行方法与预期输出）。
- 图像线 P1（small 四配置 + native SPS 3072 beats）已 pass；P2/P3/P4 未完成（见 §4.2 与 `docs/DELIVERY_SUMMARY.md`），**不得**当演示或报精度。

## 6. 事实纪律

1. 只写 pass / fail / partial / not_run；每个 PASS 指到证据路径。
2. 未执行步骤一律 not_run；不用 fixture/旧结果/行为级回放冒充 native/板级结果。
3. 数字冲突时并列两个数字 + 来源，不擅自选一个；不确定写"未核实"。
4. 负 WNS 记 fail；不降频、不加时序例外；不把 OOC 说成集成/板级。

## 7. 文档索引

| 文档 | 内容 |
|---|---|
| `QUICKSTART.md` | 3 条命令跑通（环境 → 编译 → 自检）+ 板端指路 |
| `RELEASE_NOTES.md` | 0.1.0 收录/未收录/已知限制/未验证项/重建命令 |
| `environment.md` | Python/Vivado/ModelSim/PYNQ/板卡/路径约定 |
| `docs/USER_GUIDE.md` | 工具链用户指南（入口/命令/配置/权重导入/生成物/manifest/物理门） |
| `docs/OPERATORS.md` | 算子库清单 + 各算子参数/数值语义/已验证范围 |
| `docs/PORTABILITY.md` | 路径改写说明 + diff 摘要 + 冻结 RTL 哈希核对 |
| `docs/DEMO_NLP.md` / `docs/PLATFORM_AND_STATUS.md` | NLP 演示手册 / 平台与门状态（docs 子代理） |
| `docs/NUMERIC_CONTRACT.md` / `docs/VISION_CASE.md` | 数值合同 / 图像线现状（evidence 子代理） |
| `docs/DELIVERY_SUMMARY.md` | 一页纸交付摘要（Lead） |
| `verification/README.md` | 各 TB 对应线路与状态 |
| `evidence/INDEX.md` | 证据表：证据/来源路径/sha256/用途 |
