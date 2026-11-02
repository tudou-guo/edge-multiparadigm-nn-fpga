---
name: snn-translate
version: 2.0.1
description: 把 Spikformer 配置 JSON 与全整数 checkpoint 转译为参数 blob/RTL 配置/约束/调度，并按 G0–G6 分级验收（回放 mismatch=0、OOC WNS/WHS≥0、IP 包、平台时序）
applies_to: [snn]
package_paths: [src/snn, data/snn, build/snn, sim/snn, board/snn]
requires: [Python 3.9.x + numpy, ModelSim SE-64 2020.4（回放，可选）, Vivado 2021.1（OOC/平台，可选且受共享 heavy lock）]
last_verified: 2026-11-02
---

# SKILL: snn-translate

## 0 一句话定义

把 `spikformer-config-1` 配置 JSON + 全整数 checkpoint `.npz` 经 `[包内] src/snn/tools/toolchain.py` 转译成全整数 SNN 加速器工程：
**54 个参数 blob、28 份 RTL 源绑定、RTL 配置头、目标顶层适配器（含 hold-margin 单元）、5 ns XDC、OOC/工程 Tcl、调度 JSON、manifest 身份清单**，
再用 RTL 回放（mismatch=0）与 5 ns OOC（WNS/WHS≥0、0 失败端点）等门级判据验收。转译本身是纯 Python，不启动 Vivado。

## 1 何时使用 / 何时不要使用

**使用**：

1. 输入是 Spikformer NLP 家族配置（`family` ∈ `spikformer-nlp-sentiment|intent|match`），且 checkpoint 满足 `export_abi.checkpoint_npz_required_exact`；
2. 需要生成或复核参数包、RTL 结构配置、约束与片上调度；
3. 需要判断一次转译能否进入回放、OOC、IP 打包、平台时序门（对应 `../evidence-gate/SKILL.md` 的 G0–G5）。

**不要使用**：

- 输入是 ONNX 静态图 → `../cnn-translate/SKILL.md`；输入是 LNN/CfC 权重 → `../lnn-export/SKILL.md`；
- 想在本技能得到资源/时序/bitstream 预测：compile 从不调用 Vivado，也不预测资源；
- 想把本整数变体与原版 Spikformer 或任何 FP32 参考逐位比较（见 §8）；
- 图像线 `spikformer-b1-cifar10`（`data/snn/examples/spikformer/config/model_native_b1.json`）当前是 `development_fixture`：只到编译与行为级回放，属 `partial`，**不在本技能验收范围**。

## 2 前置条件与环境自检

[包内] 提交包按用途拆目录，命令工作目录因此分两种：

| 用途 | 包内目录 | 目录内容 |
|---|---|---|
| 源码/工具链/RTL | `src/snn/` | `tools/toolchain.py`、`tools/backends/spikformer/`、`tools/spikformer_train/`、`rtl/ops/spiking/`、`docs/` |
| 配置与 checkpoint | `data/snn/examples/` | `spikformer/config/*.json`、`nlp_sentiment_b1/ckpt_real_real50.npz` |
| 编译生成物/报告/脚本 | `build/snn/` | `generated/_quickstart_real/`、`reports/`、`scripts/spikformer/` |
| TB 与证据 | `sim/snn/` | `verification/sim/spikformer/tb_*.v`、`evidence/nlp/*.json` |
| 板端包 | `board/snn/` | `snn_platform.bit`、`snn_platform.hwh`、`snn_platform_run.py`、`vec/case_real/` |

环境自检（在包根执行，逐条对照期望）：

```bash
python --version                                        # 期望 Python 3.9.x
python -c "import numpy;print(numpy.__version__)"       # 期望 2.x（实测 2.0.2）
ls src/snn/tools/toolchain.py src/snn/rtl/ops/spiking/source_manifest.json
ls data/snn/examples/spikformer/config/nlp_sentiment_b1.json data/snn/examples/nlp_sentiment_b1/ckpt_real_real50.npz
```

[工作区] 完整发布快照根为 `D:\DL_FINAL\SNN`（`tools/ rtl/ scripts/ examples/ verification/ board/ evidence/` 并列）；
回放、`build --stage synth|ooc`、`test_integer_ref` 等按快照根编写的命令在该根执行，包内对应写法见 `reference/commands.md` 的「工作目录」列。

## 3 输入合同

| 输入 | 格式 | 硬约束 | 校验命令 | 不满足时 |
|---|---|---|---|---|
| 配置 JSON | UTF-8 JSON，`schema=spikformer-config-1` | 见下表，缺字段/越界即 `ValueError('config: ...')` | 编译即校验（先跑 compile） | 进程非 0，不落 manifest |
| 整数 checkpoint | `.npz`（numpy，无 pickle） | 键集合、dtype、shape、range、thr≥1 全严格 | 编译；或 `check_contract.py` | `RuntimeError/TypeError`，不落 manifest |
| 目标器件/时钟 | 配置 `target` | `xczu15eg-ffvb1156-2-i`、Vivado 2021.1、`period_ns=5` | 生成 `generated/spikformer.xdc` 只有 1 个 5.000 ns 时钟 | 与冻结证据不可比 |

配置硬约束（`src/snn/tools/backends/spikformer/config.py`，报错原文均 2026-11-02 实测）：

| 约束 | 规则 | 报错原文 |
|---|---|---|
| batch | 只支持 1 | `config: only B=1 implemented` |
| time_steps | ∈ {1,2,4} | `config: time_steps not in legal [1,2,4]` |
| head_dim | 只能 32 | `config: head_dim: only 32 implemented (PE16 lane pair)` |
| embed | = heads×head_dim，且能整除 16、32 | `config: embed != heads*head_dim` |
| mlp_hidden | 16 的倍数 | `config: mlp_hidden must be multiple of 16` |
| depth | 正整数 | `config: depth must be positive int` |
| seq_len（NLP） | 整数 ∈ [1,256] | `config: seq_len must be int in [1,256]` |
| frontend（NLP） | `kind=embed_spike`、`mask=none` | `config: frontend.kind must be embed_spike` |
| lif | `tau=2`、`reset=0` | `config: lif.tau must be 2 for this batch` |
| head | reduction/norm ∈ sum/ps_div/folded | `config: head.reduction must be sum/ps_div/folded` |

checkpoint 必需键（`weights.py` `NPZ_REQUIRED`）：`etab`(V,E int8)、`ptab`(N,E int8)、`thr_e`(1,) int32、
`head_w`(C,E float32)、`head_b`(C,) float32、`meta_bn_state`、`meta_model_kind`、`meta_corpus_sha256`、
`meta_vocab_sha256`、`meta_cfg`；以及每 block `block{b}.{q,k,v,proj,fc1,fc2}_{w int8, b int32, scale(1,) float32, thr(1,) int32}`。
`meta_bn_state` 只接受 `already_folded|absent`；`requires_fold` 被拒（`checkpoint declares BN requires_fold: fold BN before export`）。
本批交付 checkpoint：`data/snn/examples/nlp_sentiment_b1/ckpt_real_real50.npz`（935,756 B，sha256 `e6a3a411…2448a1`，
`meta_model_kind=real_corpus`，BN absent），伴生 `ckpt_real_real50.assets.json`（sha256 `abaa6c1f…86ace`）由 compile 自动绑定。

## 4 执行步骤

### 步骤 1：编译（fixture，验证链路，不需要 checkpoint）

工作目录 `src/snn`（[包内]）：

```bash
python -B -m tools.toolchain --backend spikformer compile \
  --config ../../data/snn/examples/spikformer/config/nlp_sentiment_b1.json \
  --fixture-seed 1 --out-dir <tmp>/fixture
```

期望：`manifest: <tmp>/fixture/manifest.json kind=development_fixture`，exit 0；`params/` 28 个 `.bin`（+28 个 `.coe`）；
`manifest_sha256=835ad38dc06f0546bc08612bf052f085edec7f47b34115cfe5079c7b221381c4`（本机 2026-11-02 实测，与
`sim/snn/evidence/ACCEPTANCE_CORE.md` §A1 一致）。
失败分支：`refusing to overwrite non-empty out_dir`（换 out-dir）；`no weights: pass --fixture-seed N`（fixture 模式必须显式 seed）。

### 步骤 2：编译（真实 checkpoint，交付用参数包）

工作目录 `data/snn`（[包内]），用 `PYTHONPATH` 指向源码，`--weights` 用相对路径以复现冻结清单：

```bash
cd data/snn
export PYTHONPATH=D:/FINAL/src/snn        # bash；PowerShell: $env:PYTHONPATH="D:/FINAL/src/snn"
python -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz \
  --out-dir <tmp>/real
```

期望：`kind=real_corpus`，**54 个 params blob**，`manifest_sha256=f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e`；
生成的 `manifest.json` 文件 sha256 为 `b71e3f17…a6a4`，与 `build/snn/generated/_quickstart_real/manifest.json` **逐字节相同**（2026-11-02 实测）。
失败分支：键/类型不符 → `TypeError: key 'block0.q_w' must be int8, got float32 (no silent casts)`；
`fixture_seed conflicts with real weights`（不要同时给 `--fixture-seed` 与 `--weights`）。

### 步骤 3：核对 manifest 身份

```bash
python -c "import json,hashlib,pathlib; m=json.load(open('<tmp>/real/manifest.json')); \
print(m['model_kind'], len(m['params']), len(m['rtl_sources']), m['manifest_sha256']); \
print('rtl mismatch:', sum(1 for s,h in m['rtl_sha256'].items() if hashlib.sha256(pathlib.Path(s).read_bytes()).hexdigest()!=h))"
```

期望：`real_corpus 54 28 f727e4c7…`、`rtl mismatch: 0`。`manifest_sha256` 算法 = 去掉该字段后对
`json.dumps(manifest, sort_keys=True)` 求 sha256（本机复算一致）。
**不要跨路径比较 `manifest_sha256`**：它含 `assets.checkpoint` 路径字符串，改用绝对路径会得到不同值
（实测 `a2a655d4…`），但 54 个 blob 的 sha256 集合不变。

### 步骤 4：打包板端运行包

```bash
cd src/snn
python -B -m tools.toolchain --backend spikformer package \
  --manifest <tmp>/real/manifest.json --out-dir <tmp>/pkg
```

期望：`packaged 54 param files -> <tmp>/pkg`，产物 `runtime_config.json` + `params/`（54）+ `README.txt`；
`runtime_config.json` 的 `bitstream/hwh` 字段为 `NOT_BUILT (platform stage, shared heavy lock)`（package 不造 bit）。

### 步骤 5：物理门（OOC/综合）——需 Vivado 2021.1 + 共享 heavy lock

[工作区] 在完整快照根执行（该根 `scripts/spikformer/` 存在）：

```bash
cd D:/DL_FINAL/SNN      # [工作区]
# 只排队（默认，不启动 Vivado）
python -B -m tools.toolchain --backend spikformer build \
  --manifest build/.../manifest.json --stage ooc --shared-heavy-lock <abs lock path>
# 真跑：同一命令追加 --execute（需锁 + Vivado 2021.1；5 ns 不降频、不加时序例外）
```

[包内] 只有 `build/snn/scripts/spikformer/run_physical.py`，而 `toolchain build` 会按
`src/snn/scripts/spikformer/...` 找驱动 → 包内直接跑会失败（实测 `can't open file ...src\snn\scripts\spikformer\run_physical.py`）。
包内可做的替代是直接调用驱动做源码冻结检查 + 排队（不启动工具）：

```bash
SNN_ROOT=D:/FINAL/src/snn python -B build/snn/scripts/spikformer/run_physical.py \
  --manifest build/snn/generated/_quickstart_real/manifest.json --stage ooc --lock <tmp>/heavy.lock
```

期望首行 `source frozen check: 28 files match manifest hashes`，随后 `QUEUED (no heavy start)...`、`queued report: ...`，exit 0（2026-11-02 实测）。

### 步骤 6：回放（G1）——需 ModelSim SE-64 2020.4 + 主线向量（`[工作区]`）

```bash
cd D:/DL_FINAL/SNN      # [工作区]
python -B scripts/spikformer/run_nlp_sim.py --case small
python -B scripts/spikformer/run_nlp_sim.py --case e2e --plusarg +NLP_NO_FORCE
python -B scripts/spikformer/run_nlp_sim.py --case e2e        # 反压版（force/release）
python -B scripts/spikformer/run_nlp_sim.py --case real --plusarg +NLP_NO_FORCE
```

期望 TB 行 `NLP_RESULT case=... pass=1 cycles=...`；证据 `sim/snn/evidence/nlp/replay_*_holdfix.json`
（cycles 39774 / 7084772 / 7084812 / 10627157，均 `pass=true`、`fail_lines=0`）。

### 步骤 7：IP 源包（G4）

[工作区]：`python -B scripts/spikformer/run_ip_pack.py --manifest <run>/manifest.json --out-dir <run>/ip --ooc-report <run>/reports/ooc_*.json`。
[包内] 可对已生成 manifest 复打包（同源哈希 + 54 params 拷贝；无 OOC 报告时 `timing_qualified=false`）：

```bash
SNN_ROOT=D:/FINAL/src/snn python -B build/snn/scripts/spikformer/run_ip_pack.py \
  --manifest build/snn/generated/_quickstart_real/manifest.json --out-dir <tmp>/ip
```

期望 JSON：`status=pass, sources=28, params_copied=54, mismatch=[]`（2026-11-02 实测；package_sha256 随输出内容而定）。
冻结证据里的正式包：`sim/snn/evidence/nlp/final/ip/ip_manifest.json`（`timing_qualified=true`、19/19、`package_sha256=894599b9…292f22`）。

### 步骤 8：验收判级

按 §6 表逐级判定；任何一级不达标就停在该级，不得给出越级结论。判定命令与证据落点见 `reference/acceptance.md`。

## 5 产物与清单

| 路径（相对 out-dir） | 生成者 | 关键字段/内容 | 自证 |
|---|---|---|---|
| `manifest.json` | `manifest.write_manifest` | `model_kind`、`config/derived/structure`、`schedule`、`params`(54/28)、`rtl_sources`(28)+`rtl_sha256`、`generated`、`target`、`head`、`numeric`、`assets`、`manifest_sha256` | 复算 `manifest_sha256`；`rtl mismatch: 0` |
| `params/*.bin` + `*.coe` | `weights.py _emit` | int8/int32/float32 定长 blob：embed/pos、thr_e、embed_scale、每 block 6 层 w/b/scale/thr、FP32 head | `manifest.params[*].sha256/nbytes`；集合摘要见 `ACCEPTANCE_CORE.md` §A1b |
| `generated/spikformer_config.vh` | `codegen.generate` | `SPIK_T/SPIK_EMBED/SPIK_RW/SPIK_SUM_W/...` 结构宏 | 与 manifest `structure` 对照 |
| `generated/spikformer_nlp_target_top.sv` | `codegen._adapter` | 目标顶层 + 1192 个 DONT_TOUCH LUT1 hold-margin（8 级×149 bit，含 s_rst） | sha256 绑定在 `build_identity_*.json`/OOC 报告 |
| `generated/spikformer.xdc` | `codegen.generate` | `create_clock -period 5.000 s_clk` + 0 input/output delay，无时序例外 | 冻结证据 sha256 `9fc1c267…10842` |
| `generated/spikformer_ooc.tcl` / `spikformer_project.tcl` / `sources.f` / `cache_key.txt` | `codegen.generate` | OOC 流程、工程装配、源清单、生成器缓存键 | 与 manifest `generated` 一一对应 |
| `resolved_config.json` / `numeric_contract.json` / `schedule.json` | `compile.run` | 展开配置（含本批 thresholds）、数值合同、每 block 权重基址/descriptor beats | `check_contract.py` 108 项 |

## 6 验收（分级；命令见 reference/acceptance.md）

| 级别 | 判据 | 阈值/期望 | 证据落点 |
|---|---|---|---|
| G0 编译-fixture | 退出码 + blob 数 + 身份 | exit 0；28 blobs；`manifest_sha256=835ad38d…381c4` | `<out>/manifest.json`；`sim/snn/evidence/ACCEPTANCE_CORE.md` §A1 |
| G0 编译-真实 | 同上（交付口径） | exit 0；**54 blobs**；`manifest_sha256=f727e4c7…911e`；manifest 文件 sha `b71e3f17…a6a4` | `<out>/manifest.json`；`build/snn/generated/_quickstart_real/manifest.json` |
| G0 合同一致 | 配置/派生/合同/checkpoint 一致性 | `check_contract.py` → `status=pass`、`fails=[]`（108 checks） | 自产 JSON；`sim/snn/evidence/nlp/contract.json` |
| G0 数值自检 | 纯 Python 单测 | `test_short.py` → `FAILURES: 0`（15 项）；`test_integer_ref.py` → `RESULT: 8/8 PASS -> PASS` | `sim/snn/evidence/nlp/math_tests_m1.{log,json}` |
| G1 回放 | 四档 `pass=1` 且 mismatch=0 | cycles 39774 / 7084772 / 7084812 / 10627157；`fail_lines=0` | `sim/snn/evidence/nlp/replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json` |
| G2 OOC 5 ns | 0 失败端点 + WNS/WHS≥0 | Setup WNS **+0.135**、TNS 0、0 failing（186,547）；Hold WHS **+0.023**、0 failing；unrouted 0；latch 0；无时序例外 | `sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json` |
| G4 IP 包 | 同源 + 时序资格 | `timing_qualified=true`；19/19 源哈希；`package_sha256=894599b9…292f22` | `sim/snn/evidence/nlp/final/ip/ip_manifest.json`、`SOURCE_SHA256.txt` |
| G5 平台（r7） | 全片 routed 时序 + bit/HWH | Setup WNS **+0.051**/0 failing（334,330）；Hold WHS **+0.011**/0 failing（333,740）；route 0 errors、fully routed 180,101/180,101 | `build/snn/reports/platform_timing_summary_20261031.txt`、`r7_sumsfix_route_status.rpt` |
| G6 板端 | 客方回传记录 | 3/3 `match=True`（≈7,084,772 cycles、model_ms≈35.4） | `board/snn/README_BOARD.md` §3；`sim/snn/evidence/platform/board/` |

**判级规则**：负 WNS 记 `fail`；不降频、不加时序例外；OOC ≠ 集成/板级；fixture 结果不得当精度或交付结论；
板端以客方回传记录为准（原始输出未随包），不得与本机实测混写。

## 7 故障与修复（高频摘要，四元组全文见 reference/troubleshooting.md）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | 包内 `toolchain build --stage synth/ooc` 报 `can't open file ...src/snn/scripts/spikformer/run_physical.py` | `toolchain.py` 以 ROOT(=src/snn) 为 cwd 调驱动，包内脚本在 `build/snn/scripts` | 用工作区根执行；或直接调 `run_physical.py`（`SNN_ROOT` 指向 `src/snn`）只排队 |
| 2 | `test_short.py` 报 `ModuleNotFoundError: No module named 'reference'` | 脚本按 `parents[3]` 推根，包内落在 `sim/snn` | 加 `SNN_ROOT=D:/FINAL/src/snn` |
| 3 | `test_integer_ref.py` 报 `No module named 'runtime'` | 包内未收录 `runtime/spikformer/`（仅工作区有） | 到 [工作区] `D:\DL_FINAL\SNN` 执行 |
| 4 | 换路径后 `manifest_sha256` 变了 | 该哈希含 `assets.checkpoint` 路径字符串 | 用相同相对路径字符串复跑；比 blob 集合而非比该值 |
| 5 | hold WHS 负、失败路径全起于输入端口 | OOC 0 输入延迟模型下输入到首级寄存器 hold 余量不足 | 生成 adapter 里 8 级 DONT_TOUCH LUT1 链（1192 cells），XDC 逐字节不变；16 级牺牲 setup，10 级余量过薄 |
| 6 | 平台 r7 读 sums 通道错位 64 | CSR SUMS 窗口基址 0x100→64 字偏移未修 | 软件 `ROT=64` 读；或按 handoff 重出 bit |
| 7 | 连续第二个样本超时 | 软复位只 flush FIFO，不排空在途 DMA | 每样本重载 overlay；根治需改平台 |
| 8 | 平台 CDC 报告 1 条 Critical | DDR4/复位基础设施路径（基线平台遗留） | 记录未收敛；不得写成"CDC 全清" |
| 9 | 配置越界/缺字段 | `config.py` 硬校验 | 按 §3 报错原文改配置 |
| 10 | 浮点权重被拒 | `_get` 无静默 cast | 用 `export_integer.py` 重新导出 int8/int32 npz |
| 11 | 冻结 RTL 被改后再跑物理门 | `rtl_freeze.json` 规则：冻结后任何 RTL 改动使 synth/replay/OOC 证据失效 | 重新 freeze 并重跑全部下游门 |
| 12 | 图像线 AV 位宽与 NLP 混用 | NLP 32 token AV=1024(uint11)；图像 native 64 token AV=2048(12bit)；图像的 `numeric_contract.json.av` 文案陈旧 | 以 `resolved_config.derived` 为准，不混用 |

## 8 边界与不适用

1. **数值红线**：本变体是命名的 Spikformer NLP 整数变体，不是原版同构实现——无 softmax、AV 不归一、无 BN/LayerNorm；
   **禁止**与原版 Spikformer 或任何 FP32 参考逐位比较；golden truth 只有 `tools/backends/spikformer/integer_ref.py`
   （见 `src/snn/docs/NUMERIC_CONTRACT.md` §0/§10）。
2. **不预测**：compile 不预测资源、时序、功耗；`package` 的 `bitstream/hwh=NOT_BUILT` 不是 bit。
3. **OOC 边界**：OOC 是 0 输入延迟模型、`HD.CLK_SRC` 未设，setup 偏乐观（WNS 束缚路径就是 hold-margin 链，+0.135 ns）；
   真实上游输入延迟 >0.135 ns 即破坏，**不是集成即用时序签核**。
4. **平台边界**：完整协议仿真矩阵待补、软复位不排空在途 DMA、CDC 1 Critical 待收敛；平台 r7 为 NLP 专用。
   平台 r7 的 WHS **+0.011 ns** 与 r6 基线 +0.010 ns 同级，属薄余量：任何 RTL/权重改动都必须重跑时序并重新配对 bit/HWH。
5. **板端边界**：3/3 为客方回传记录（原始 JSON 未随包），须注明来源。
6. **图像线**：`partial`；无精度、无 native OOC/平台/板测；P2 两图连续有已知 RTL 限制；不纳入本技能验收。
7. **不要做**：手改 `generated/*.tcl`（含生成时绝对根，换根重跑 compile）；手改冻结 RTL/阈值；用 fixture 结果冒充精度/交付；
   把 OOC 说成平台/板级；为了过门降频或加时序例外。

## 9 版本与身份绑定

| 身份 | 内容 | 生成/校验 |
|---|---|---|
| 版本 | `snn-toolchain-0.1.0`（[工作区] `D:\DL_FINAL\SNN\VERSION`；包内见 `src/snn/README.md`） | 包内未收录 VERSION 文件 |
| `manifest_sha256` | 去掉自身字段后对排序 JSON 求 sha256 | 步骤 3 复算 |
| `rtl_sha256` | 28 份源（`src/snn/rtl/ops/spiking/source_manifest.json`）逐文件 sha256 | manifest 内；`run_physical`/`run_ip_pack` 复检 → 0 mismatch |
| `rtl_freeze.json` | 18 文件冻结点（2026-10-29T22:26） | `sim/snn/evidence/nlp/rtl_freeze.json` |
| 快照 RTL 一致性 | 当前 28 源 vs 冻结 18 文件：**16 SAME / 2 DIFF**（`sps/spiking_sps_top.sv`、`sps/spiking_eyeriss_pe.sv`，图像线并行改动） | `sim/snn/evidence/nlp/rtl_hash_check_20261031.txt`；NLP 功能 RTL 16/16 未变（NLP top 不实例化 SPS）；**不得写"整棵 RTL 逐字节一致"** |
| `cache_key` | 生成器缓存键（cfg+sched+part+tool） | `generated/cache_key.txt` |
| 构建身份 | adapter/xdc/ooc_tcl/codegen 哈希 + `manifest_sha256` | `sim/snn/evidence/nlp/final/reports/build_identity_20261030.json` |

规则：任何改 checkpoint/配置/RTL 后必须重新 compile，旧 `manifest_sha256`/`rtl_sha256` 立即失效；证据只能引用同一身份链上的产物。

## 10 证据归档

按 `../evidence-gate/SKILL.md` §10 的标签与目录约定：

| 级 | 标签 | 归档位置（[包内]） |
|---|---|---|
| G0 编译 | `pass/fail` + manifest_sha256 + params 集合摘要 | `build/snn/generated/<run>/`（本包保留 `_quickstart_check`/`_quickstart_real`/`_lead_final_check`） |
| G1 回放 | `pass` + cycles + `fail_lines=0` | `sim/snn/evidence/nlp/replay_*_holdfix.json` |
| G2/G3 物理 | `pass` + WNS/WHS/TNS/THS/failing/unrouted/latch | `sim/snn/evidence/nlp/final/reports/ooc_*.json`；平台 `build/snn/reports/` |
| G4 IP | `pass` + timing_qualified + 19/19 + package_sha256 | `sim/snn/evidence/nlp/final/ip/` |
| G6 板端 | 客方回传记录（注明来源与日期） | `board/snn/README_BOARD.md` §3；`sim/snn/evidence/platform/board/` |
| 索引 | 一行一事实 + 来源 + sha256 | `sim/snn/evidence/INDEX.md`、`ACCEPTANCE_CORE.md` |

新增证据放新目录/新文件名（带日期或 run-id），不覆盖旧证据；不在包目录外写文件。

## 11 参考文件索引

- `reference/commands.md`：可复制命令矩阵（工作目录、参数、退出码、期望输出）
- `reference/artifacts.md`：产物与清单字段表（54 blob 明细、manifest/schedule/numeric 字段、运行包）
- `reference/acceptance.md`：G0–G6 分级验收清单与判定命令
- `reference/troubleshooting.md`：故障四元组（现象/根因/处置/证据）
- `reference/sources.md`：每条关键结论的出处索引（包内文件/字段与工作区来源）
- `../evidence-gate/SKILL.md`：标签、门级、身份哈希与归档约定
