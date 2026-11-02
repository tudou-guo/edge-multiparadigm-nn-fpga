# 故障库 — lnn-export（现象 / 根因 / 处置 / 证据）

> 每条四元组都能回查到真实文件或真实命令输出。标注 **实测**＝2026-11-02 本机复现过；**引用**＝来自开发树报告/发布文档；
> **推导**＝由源码/合同必然推得但无现场报错记录（已注明，不冒充实测故障）。
> 路径：`<包根>` = `D:\FINAL`；[工作区] = `D:\DL_FINAL\LNN` / `D:\DL_LNN`。

## 快速索引

| 现象关键词 | 条目 |
|---|---|
| `manifest.json config/hash check failed`、pyc、第二次运行失败 | T1 |
| `export/ directory ... was not found`、板上缺期望值 | T2 |
| `hardware handoff ... same stem` | T3 |
| `ID mismatch`、`VERSION mismatch`、`CLOCK_HZ mismatch`、三件套不同源 | T4 |
| 门控算错、`W_b` 权重不参与、golden 全错 | T5 |
| `KeyError`、形状/键名不符 | T6 |
| `overflows int16 Q12 range`、`exceeds fixpoint range limit` | T7 |
| 7/7 对拍 `DIFF` | T8 |
| `CFC_SIM_FAIL missing export/gen/...` | T9 |
| `CFC_SIM_FAIL xsim tools not found` | T10 |
| `bitfile : MISSING`、`export : MISSING`（env_check） | T11 |
| 多驱动 8-6858/8-6859、ROM 初值全 0 | T12 |
| 离线 `run_cycles=2910`、周期数被误读 | T13 |
| 状态残留、`retain` 模式结果不一致 | T14 |
| LUT 索引越界 / 夹紧 | T15 |

---

## T1 pycache 让 manifest 哈希失配（板上第二次运行必失败）

- **现象**：板上第一次跑正常，第二次跑报 `BOARD_ERROR: bring-up validation failed: export/manifest.json config/hash check failed`。
- **根因**：驱动 import `export/cfc_spec.py`、`export/int_reference.py` 时 Python 在 `export/` 下生成 `__pycache__/*.pyc`；
  `manifest.json` 的 `files` 表里有 2 条**开发树** pyc 哈希，新生成的 pyc 与它不同 → 驱动判 `mismatch`（`missing` 只告警，`mismatch` 判失败）。
- **处置**：每次运行前 `rm -rf export/__pycache__`，命令始终加 `python3 -B`；PC 端离线自检同理。
- **证据**：`board/lnn/lnn_cfc_run.py`（manifest 校验：`missing` 只记录、`mismatch` 置 `ok=False`，见 `files["mismatch"]` 分支）；
  `data/lnn/export/manifest.json` 的 2 条 `export/__pycache__/*.pyc` 条目；`[工作区] user/README_LNN_DEMO.md` §1.4"pycache 陷阱"（引用开发树文档，原样给出该报错）。

## T2 板上缺 export/ 期望值目录，驱动拒绝运行

- **现象**：`lnn_cfc_run.CfcBoardError: export/ directory (weights, LUTs, golden vectors) was not found; pass --expect-dir or set CFC_EXPECT_DIR`。
- **根因**：只把驱动/位流/Notebook 拷到板上，没拷 `export/`（权重/LUT/整数 golden/manifest）。
- **处置**（三选一）：① `scp -r <包根>/board/lnn/export xilinx@<板>:<驱动目录>/export`；② `os.environ["CFC_EXPECT_DIR"]=...` 后**重启内核**；③ 命令行 `--expect-dir /path/to/export`。
- **证据**：`[工作区] D:/DL_LNN/reports/board_package_report.md` §8.1 现场现象原文与 §8.2 修复；
  `board/lnn/lnn_cfc_run.py` 的 `_load_reference()` 报错文本与 `discover_paths()` 候选列表（`driver/export`、`driver/package/export`、`driver/*/export`、`cwd/export`、父目录、`CFC_EXPECT_DIR`）；`board/lnn/README_BOARD.md` §3。
- **备注**：包内 `board/lnn/export/` 已自带，**整目录拷贝**即可开箱；只拷 `.bit/.hwh` 才会缺。

## T3 bit 与 hwh 不同目录 / 不同 basename

- **现象**：PYNQ `Overlay()` 报 `hardware handoff (.hwh) ... same stem` / `hardware handoff not found next to bitstream`。
- **根因**：`.bit` 与 `.hwh` 必须同目录且同 basename（`lnn_cfc.bit` + `lnn_cfc.hwh`）；改名或分开存放即失败。
- **处置**：从交付包原样拷贝；若必须改名，两份同时改成同一 stem。
- **证据**：`board/lnn/README_BOARD.md` §2、§5 故障表；`src/lnn/docs/USER_GUIDE.md` §5；文件实测 `lnn_cfc.bit`/`lnn_cfc.hwh` 同名。

## T4 三件套不同源（ID/VERSION/时钟不符或哈希不等）

- **现象**：`ID mismatch: read 0x... expect 0x43464331`、`VERSION mismatch`、`CLOCK_HZ mismatch`，或 hash 校验直接失败。
- **根因**：bit 与 hwh/xsa 不是同一次构建；或板上 bit 是旧版本、驱动/期望值是新的。
- **处置**：用本包成对三件套；跑 `sha256sum -c SHA256SUMS` 与 XSA 内嵌比对（§4 步骤 6）；ID/VERSION/CLOCK_HZ 三项必须 `ok=True`。
- **证据**：判定方法可复算——XSA 内 `lnn_cfc.bit` 哈希 `0fcd8d65…`、内 `design_1.hwh` 哈希 `fe2d9f0e…`（本机实算，与外部文件一致）；
  `sim/lnn/evidence/ACCEPTANCE.md` §3"配对检查"；`board/lnn/README_BOARD.md` §5。
- **不要做**：不要用 `--allow-clock-mismatch` 掩盖时钟不符。

## T5 门控公式漏 W_b 项（V1.0 缺陷，最严重数值错误）

- **现象**：导出/仿真的 golden 与需求文档公式对不上；84 个权重里索引 60..79 的 20 个等于白算；换 V1.1 后全部数字漂移。
- **根因**：早期合同把门控写成 `gate_pre=ROUND(lin_a*Δt + bb)`，漏掉 `b_t = W_b z_t + b_b` 的 `W_b` 项。
- **处置**：按合同 V1.1 改为 `gate_pre = ROUND_Q12_SAT16(lin_a*655 + (lin_b<<12))`（单次 Q24 舍入、无中间截断），
  并**重做全部整数 golden、RTL 与验证**；旧 V1.0 golden 一律作废。
- **证据**：`[工作区] D:/DL_LNN/docs/CFC_CONTRACT.md` 版本历史（V1.1 修正说明）；`data/lnn/export/manifest.json` 的 `numeric.gate_semantics`；
  `src/lnn/README.md` §5 已知限制第 5 条；`[工作区] RELEASE_NOTES.md` "重要工程说明 1"。
- **自检**：`python cfc_sim_model.py --selfcheck` 必须出现 `OK gate_pre includes W_b branch (contract V1.1)`。

## T6 权重键名/形状不符（推导，无现场报错记录）

- **现象**：`KeyError: 'wb'` 或 numpy 拼接/矩阵乘维度错误，导出中止。
- **根因**：`model_weights.npz` 缺 `PHASE_KEYS` 之一，或 `w1/w2/wa/wb` 不是 `(4,5)`、`wo` 不是 `(1,4)`、偏置不是 `(4,)`/`(1,)`。
- **处置**：按 §3.1 的 10 个键与形状重存 npz；导出器按 `PHASE_KEYS` 顺序读键，顺序不影响但键名必须一致。
- **证据**：`[工作区] user/gen/generate_export.py` 的 `PHASE_KEYS`/`load_float_params()`/`quantize_weights()`；
  实测 `model/model_weights.npz` 键与形状为 `w1(4,5) w2(4,5) wa(4,5) wb(4,5) wo(1,4) b1(4,) b2(4,) ba(4,) bb(4,) bo(1,)`（全是 float32）。
- **标注**：推导条目——包内没有该故障的现场日志。

## T7 量化溢出（权重/偏置超定点量程）

- **现象**：导出器抛 `ValueError: weight 12345 overflows int16 Q12 range` 或 `bias ... overflows int32 Q24 range`；
  浮点范围检查抛 `parameter w1 max abs 9.1 exceeds fixpoint range limit 8.00`。
- **根因**：训练发散导致 |权重| > 8.0（Q12 可表示上限）或偏置 > 127.99；或权重被放大后直接送入量化。
- **处置**：先查训练日志与 `train_config.json` 指标；确实需要更大动态范围＝改数值合同（改 Q 格式会连带改 RTL），不在本技能内静默处理。
- **证据**：`[工作区] user/gen/generate_export.py` 的 `quantize_weights()/quantize_biases()/check_float_ranges()`（异常文本原文）；
  实测当前权重通过范围检查（`manifest.metrics.float_range_check` 全 pass）。

## T8 7/7 对拍出现 DIFF

- **现象**：`user/export_demo.py` 输出某行 `DIFF`，末行 `RESULT: FAIL`，退出码 1。
- **根因**：权重/配置与仓库 `export/` 不是同一批（换了 `model_weights.npz` 未重生成 `export/`，或 `train_config.json` 被改）。
- **处置**：确认 `model_weights.npz` 与 `train_config.json` 同批；看 `<out>/export.log` 里的 `metrics`（float/int MSE、acceptance）与前一批比对；
  重生成 `export/` 后必须同步更新 `manifest.json`、`board/lnn/export/` 与 `SHA256SUMS`。
- **证据**：`[工作区] user/export_demo.py` 的 `FILES` 列表与 `sha256` 比较逻辑；`user/outputs/video00/summary.json`（7/7 `same=true` 的正常样例）。
- **易错点**：对拍口径是"重算 vs 仓库 `export/`"，不是"重算 vs `board/lnn/export/`"；两者包内同源（`diff -rq` 无差异），但工作区只有前者。

## T9 run_sim.tcl 在包内布局下找不到 export/gen（实测）

- **现象**：在 `<包根>/build/lnn` 下执行 `tclsh scripts/run_sim.tcl` → `CFC_SIM_FAIL missing export/gen/cfc_weight_init.svh`，退出码 **1**。
- **根因**：脚本用 `script_dir/..` 当快照根（`D:/DL_FINAL/LNN` 布局：`rtl/ verification/ export/gen` 是同级）；
  包内这三类文件分别在 `src/lnn/rtl`、`sim/lnn/verification`、`data/lnn/export/gen`，根下没有 `verification/` 与 `export/gen`。
- **处置**：在工作区 `cd D:/DL_FINAL/LNN && vivado -mode batch -notrace -source scripts/run_sim.tcl` 跑；
  或在包内自建符合快照布局的临时树（把 rtl/、verification/、export/ 摆成同级）后跑。
- **证据**：本机 2026-11-02 实测上述命令与输出（exit 1）；脚本原文 `build/lnn/scripts/run_sim.tcl` 的 `set root [file normalize [file join $script_dir ".."]]` 与随后的 `file exists` 检查。

## T10 xsim 不在 PATH（实测）

- **现象**：`CFC_SIM_FAIL xsim tools not found (source settings64.bat first)`，退出码 **1**。
- **根因**：脚本用 `auto_execok` 找 `xvlog/xelab/xsim`，找不到再查 `XILINX_VIVADO` 与 `info nameofexecutable` 目录；未 source 环境时三者皆无。
- **处置**：先 `call E:\Vitis\Vivado\2021.1\settings64.bat` 再跑；或从 Vivado 的 `bin` 目录启动。
- **证据**：本机 2026-11-02 在 `D:/DL_FINAL/LNN` 实跑 `tclsh scripts/run_sim.tcl` 输出与退出码；
  脚本原文 `find_tool` 过程；`[工作区] environment.md` 的 Vivado 位置 `E:/Vitis/Vivado/2021.1`。

## T11 包内 env_check.py 报 bitfile/export MISSING（实测，不是故障）

- **现象**：`cd <包根>/build/lnn && python scripts/env_check.py` → `bitfile : MISSING`、`export : MISSING`，但退出码 0。
- **根因**：脚本以 `__file__/../..` 为快照根，检查 `<快照根>/board/` 下的 `lnn_cfc.bit`（包内为 `board/lnn/lnn_cfc.bit`）与 `<快照根>/export/` 下的 `weights.hex`（包内为 `data/lnn/export/weights.hex`）；
  包内该脚本在 `build/lnn/scripts`，其根是 `build/lnn`，这两个路径本就不存在。
- **处置**：判断"环境是否可用"请用 `board/lnn` 下的驱动/期望值；要跑完整环境+哈希检查，用工作区 `python scripts/env_check.py --check-hashes`（实测 `108 files checked, 0 bad`）。
- **证据**：本机实跑两条命令的输出对照；脚本原文 `ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`。

## T12 Vivado 2021.1 的 include 作用域与多驱动

- **现象（引用开发树记录）**：综合报 `[Synth 8-6858]/[Synth 8-6859] multi-driven net`（`f1/f2/g`）；或 ROM 初值全 0（include 未生效）。
- **根因**：① 同一寄存器在复位块与 FSM 块都被赋值 → 多驱动，上板会被钉 0；② 生成参数的 `include` 放在**文件作用域**，
  Vivado 2021.1 综合不解析文件作用域 include 的 `include_dirs`（xsim 却可以）→ ROM 没有初值。
- **处置**：`include` 必须放在 **module 体内**（当前 `cfc_weight_rom.sv`/`cfc_activation_rom.sv`/`cfc_axi_lite_regs.sv` 已如此）；
  多驱动改为仅主 FSM 块驱动数据、capture 块只复位中间量。
- **证据**：`src/lnn/rtl/cfc_weight_rom.sv` 注释"the initial block is the only initialization path (no `$readmemh`)"；
  `src/lnn/rtl/cfc_axi_lite_regs.sv` 注释"Vivado 2021.1 synthesis does not resolve file-scope include through include_dirs"；
  `[工作区] reports/sim_report.md` §6 第 4/7 条；`[工作区] reports/vivado_build_report.md` §210-214。

## T13 离线模型的 run_cycles 被当成硬件周期

- **现象**：开发机 `simulation-model` 输出 `run_cycles=2910`，被写成"硬件 64 步周期数"，与 RTL 的 9088 矛盾。
- **根因**：`cfc_sim_model` 用经验模型 `cycles_base=30 + cycles_per_step=45`，64 步 → `30+45*64=2910`，只是模型假设。
- **处置**：硬件周期只看板上 `source=hardware` 的 `summary.json` 的 `run_cycles` 与 `run_cycles_source`；
  设计参考值是 RTL 仿真 `RUN_CYCLES=9088`（≈30.3 µs @300 MHz，约 142 周期/步）。
- **证据**：`board/lnn/cfc_sim_model.py` 的 `cycles_base=30, cycles_per_step=45` 与 `run_cycles = self.cycles_base + self.cycles_per_step * n`；
  本机实跑 `--backend simulation-model` 输出 `run_cycles=2910`、`hardware_time = 0.000010 s`——均为模型值；同一次运行的 `clear_each_step task_mse=0.49617` 也只是模型输出；
  `[工作区] user/README_LNN_DEMO.md` §3 口径提醒；`src/lnn/docs/INTERFACE.md` §5。

## T14 状态未初始化 / 状态残留（推导 + 合同保障）

- **现象**：同一样本两次运行结果不同；或 `retain` 模式下输出与 `normal` 不一致；`H_FINAL` 非零但本轮没跑。
- **根因**：`MODE[0]`（`CLEAR_BEFORE_RUN`，默认 1）被写成 0 保留上次末状态；或上次运行中止后未 `CLEAR_STATE` 就再跑。
- **处置**：默认用 `normal`（运行前清状态）；需要对照实验才用 `--mode retain`；异常中断后先 `CLEAR_STATUS`/`CLEAR_STATE` 再跑；
  `--checks` 的 `repeat` 与 `32+32 segments` 就是这条语义的验收项。
- **证据**：`src/lnn/docs/INTERFACE.md` §3（运行协议：`MODE[0]=1` 时先清隐状态）、§2（`MODE` 默认 1）；
  `[工作区] reports/sim_report.md` §2.2（状态清零重复运行一致、32+32 分段=连续 64 步）；驱动 `MODE_MAP` 源码。
- **标注**：推导条目（机制与检测手段真实，无"状态未初始化"现场报错文本）。

## T15 LUT 索引越界

- **现象**：无报错——当前实现把索引夹紧到 `0..1024`，越界输入不会崩，但若移除夹紧会读到越界/回绕值，表现为 tanh/sigmoid 输出跳变。
- **根因**：`idx = (v + 32768 + 32) >> 6`，`v` 为 int16 时理论范围 `[-32768, 32767]` → `idx ∈ [0, 1023]` 加偏置后最大 1024，
  边界靠 `clamp` 兜底；若 `v` 未被限制在 int16（例如上游漏饱和），索引会出界。
- **处置**：保持 `lut_index()` 的 `max(0, min(1024, idx))`；上游所有进入 LUT 的量必须先 `round_q12_sat16` 饱和。
- **证据**：`data/lnn/export/cfc_spec.py` 的 `lut_index()`；`data/lnn/export/manifest.json` 的 `activation.index_rule`；
  实测自检输出 `OK lut_index(-32768)==0`、`OK lut_index(32767)==1024`、`OK lut_index(0)==512`；
  `[工作区] reports/int_reference_report.md` §5"LUT 索引越界夹紧次数：0"。
- **标注**：无现场越界报错记录；本条是防御性条款。

## 附：故障与证据的对应关系

| 条目 | 证据文件（包内可查 / 工作区） |
|---|---|
| T1 | `board/lnn/lnn_cfc_run.py`、`data/lnn/export/manifest.json` / [工作区] `user/README_LNN_DEMO.md` §1.4 |
| T2 | `board/lnn/lnn_cfc_run.py`、`board/lnn/README_BOARD.md` §3 / [工作区] `reports/board_package_report.md` §8 |
| T3–T4 | `board/lnn/SHA256SUMS`、`sim/lnn/evidence/ACCEPTANCE.md` §3、`board/lnn/README_BOARD.md` §5 |
| T5 | `data/lnn/export/manifest.json`、`src/lnn/README.md` §5 / [工作区] `docs/CFC_CONTRACT.md`、`RELEASE_NOTES.md` |
| T6–T8 | [工作区] `user/gen/generate_export.py`、`user/export_demo.py`、`user/outputs/video00/summary.json` |
| T9–T11 | `build/lnn/scripts/run_sim.tcl`、`build/lnn/scripts/env_check.py` + 本机实跑输出 |
| T12 | `src/lnn/rtl/*.sv` 注释 / [工作区] `reports/sim_report.md` §6、`reports/vivado_build_report.md` |
| T13 | `board/lnn/cfc_sim_model.py`、`src/lnn/docs/INTERFACE.md` §5 |
| T14 | `src/lnn/docs/INTERFACE.md` §2/§3、驱动 `MODE_MAP` / [工作区] `reports/sim_report.md` §2.2 |
| T15 | `data/lnn/export/cfc_spec.py`、`manifest.json` `activation.index_rule` / [工作区] `reports/int_reference_report.md` §5 |
