# 事实来源索引 — lnn-export

> 目的：本技能里每条可核查结论 → 对应源文件/字段。冲突时以 `[包内]` 为准；工作区与包不一致处如实并列。
> 标注：`[包内]` = `D:\FINAL` 下可移植；`[工作区]` = `D:\DL_FINAL\LNN` 或 `D:\DL_LNN`（不随包）。
> 核验日期：2026-11-02（本机 Windows / Python 3.9.23 / numpy 2.0.2）。

## 1 输入合同与数值格式

| 结论 | 出处 | 标注 | 用在 |
|---|---|---|---|
| 10 个权重键与形状 `w1/w2/wa/wb(4,5), wo(1,4), b1/b2/ba/bb(4,), bo(1,)`，全 float32 | 实测 `model/model_weights.npz`；导出器 `PHASE_KEYS` | [工作区] | SKILL §3.1 |
| 84 权重 + 17 偏置 = **101 参数** | `data/lnn/export/manifest.json` `model.n_weights/n_biases`；`src/lnn/README.md` §1 | [包内] | SKILL §0/§3.1、artifacts §3.2 |
| 参数顺序（W1 0..19 / W2 20..39 / Wa 40..59 / Wb 60..79 / Wo 80..83；b1 0..3 / b2 4..7 / ba 8..11 / bb 12..15 / bo 16） | `manifest.json` `model.parameter_order`；`data/lnn/export/cfc_spec.py` 的 `W1_BASE…BO_INDEX` | [包内] | SKILL §3.1、artifacts §3.2 |
| int16 Q12（scale 4096）/ int32 Q24（scale 2^24）/ 48 位精确累加、无中间截断 | `manifest.json` 的 `numeric.data` / `numeric.bias` / `numeric.accumulator`；`src/lnn/docs/INTERFACE.md` §4 | [包内] | SKILL §3.2、artifacts §3.2 |
| 舍入＝最近值、中点远离零、饱和 int16；`round_q12_sat16` 实现 | `manifest.json` `numeric.rounding/round_function`；`cfc_spec.py` | [包内] | SKILL §3.2 |
| Δt=655（`0x028F`=0.159912109375；浮点训练 0.16） | `manifest.json` `numeric.delta_t_int*`、`task.delta_t_float` | [包内] | SKILL §3.2、artifacts §3.2 |
| LUT 1025 点、[-8,8]、步长 1/64、索引 `clamp((v+32768+32)>>6,0,1024)` | `manifest.json` `activation`；`cfc_spec.py` 的 `LUT_*` 常量 | [包内] | SKILL §3.2、T15 |
| 溢出护栏文案（`overflows int16 Q12 range` / `exceeds fixpoint range limit`） | `user/gen/generate_export.py` 的 `quantize_weights/quantize_biases/check_float_ranges` | [工作区] | SKILL §3.2、T7 |
| 任务定义（ω=1.0、Δt=0.16、delay=4、n_steps=64、8 相位、skip5） | `manifest.json` `task`；`model/train_config.json` `task` | [包内]/[工作区] | artifacts §3.3 |

## 2 导出链与 7 个文件

| 结论 | 出处 | 标注 | 用在 |
|---|---|---|---|
| 导出器 CLI：`--out-dir`（默认 `user/outputs/demo01`）；非空目录拒绝、退出码 2 | 实跑 `python user/export_demo.py --help`；`user/export_demo.py` 源码 | [工作区] | commands §1.1–1.3、T8 |
| 生成器 CLI：`--weights/--bootstrap/--tag/--no-plot`，二者必选其一 | `user/gen/generate_export.py` 的 `argparse` | [工作区] | commands §1.4–1.5 |
| 7 个文件名与描述 | `user/export_demo.py` 的 `FILES` 列表；`user/outputs/video00/summary.json` | [工作区] | SKILL §5、artifacts §1 |
| 7 文件字节数（420/153/5125/5125/2626/60464/115）与 SHA256 | 本机实算 `data/lnn/export/`；与 `user/outputs/video00/summary.json` 一致 | [包内] | artifacts §1、SKILL §5 |
| 对拍方式＝SHA256 相等（不是文本 diff），且与仓库 `export/` 比 | `user/export_demo.py` 的 `sha256()` 与 `ok = ... and sha256(new)==sha256(old)` | [工作区] | SKILL §4 步骤 2、commands §1.2 |
| 生成器一次输出 `n_files=37`、`float_task_mse_skip5=1.2341206e-05` | `user/outputs/video00/export.log` 的 JSON 段 | [工作区] | commands §1.4、T8 |
| RTL 只从 svh 的 `initial` 块取 ROM 初值（不用 `$readmemh`） | `src/lnn/rtl/cfc_weight_rom.sv` 注释；`cfc_activation_rom.sv` 注释 | [包内] | SKILL §5、T12 |

## 3 ONNX 与定点参考

| 结论 | 出处 | 标注 | 用在 |
|---|---|---|---|
| opset 17 / IR 8；15 节点；算子 counts；允许集 | `data/lnn/export/onnx_verify.json` `graph.operators` | [包内] | SKILL §3.1/§4 步骤 3、artifacts §4 |
| ONNX vs PyTorch `max_abs_err=4.768372e-07`（256 组，阈值 1e-5） | 同上 `pytorch_vs_onnx_random`；`sim/lnn/evidence/ACCEPTANCE.md` §1 | [包内] | SKILL §4 步骤 3、acceptance §2.1 |
| 定点 vs 浮点 MSE `1.171856e-04`（阈值 0.001）、max abs 0.0476 | `manifest.json` `acceptance/metrics` | [包内] | SKILL §4 步骤 4、acceptance §2.2 |
| 浮点任务 MSE `1.2341344e-05`（阈值 0.005） | `manifest.json` `acceptance.float_task_mse_skip5` | [包内] | 同上 |
| 整数参考自检 14 项 OK（含 `gate_pre includes W_b branch`） | 本机实跑 `cfc_sim_model.py --selfcheck`；`board/lnn/cfc_sim_model.py` 源码 | [包内] | SKILL §4 步骤 4、acceptance §2.2 |

## 4 仿真与实现

| 结论 | 出处 | 标注 | 用在 |
|---|---|---|---|
| 一键入口 `vivado -mode batch -notrace -source scripts/run_sim.tcl`；只看 `CFC_TB_PASS/CFC_TB_FAIL/CFC_SIM_PASS` 标记 | `build/lnn/scripts/run_sim.tcl` 原文 | [包内] | SKILL §4 步骤 5、commands §4 |
| 三 TB 数字：cell 21 用例/0 mismatch/139 周期；seq 626 checks/0 mismatch/RUN_CYCLES=9088/STEP_COUNT=64；protocol 155 checks/0 mismatch | `sim/lnn/evidence/ACCEPTANCE.md` §1；`[工作区] reports/sim_report.md` §3 | [包内] + [工作区] | acceptance §2.3、SKILL §4 步骤 5 |
| WNS +0.226 / WHS +0.011 / WPWS +0.039 ns；0 失败端点；57464/57464；DRC 4 Warning/0 Error | `sim/lnn/evidence/ACCEPTANCE.md` §2 | [包内] | acceptance §3、SKILL §6 |
| 资源：LUT 25,221（7.39%）/ FF 32,890（4.82%）/ DSP 4 / BRAM 28.5 / URAM 0 | 同上 §2；`[工作区] reports/vivado_build_metrics.txt` | [包内]+[工作区] | acceptance §3 |
| 8 个 RTL 冻结哈希（前 8 位 7c076f1a…7cf03211） | `[工作区] reports/sim_report.md` §5；包内 `src/lnn/rtl/*.sv` 实算 8/8 一致 | [包内]+[工作区] | artifacts §5、SKILL §9 |
| 器件 `xczu15eg-ffvb1156-2-i`；`npu_clk 300 MHz` 单时钟域；CSR `0x0480000000`（4 KiB） | `board/lnn/lnn_cfc.hwh`（实测 `SYSTEMINFO`/`CLKFREQUENCY`/`BASEVALUE`）；`INTERFACE.md` §1/§5 | [包内] | SKILL §4 步骤 6、§9 |
| 门控 V1.1 修正（漏 `W_b` → 20/84 权重不参与） | `[工作区] docs/CFC_CONTRACT.md` 版本历史；`manifest.json` `numeric.gate_semantics`；`src/lnn/README.md` §5 | [包内]+[工作区] | T5、SKILL §3.2 |
| Vivado 2021.1 include 必须放 module 体内；多驱动 8-6858/8-6859 | `src/lnn/rtl/cfc_axi_lite_regs.sv`/`cfc_weight_rom.sv` 注释；`[工作区] reports/vivado_build_report.md` §210-214 | [包内]+[工作区] | T12 |

## 5 板端与身份

| 结论 | 出处 | 标注 | 用在 |
|---|---|---|---|
| 驱动 CLI 全部选项与默认值 | 实跑 `python lnn_cfc_run.py --help`；`board/lnn/lnn_cfc_run.py` 的 `argparse` | [包内] | commands §5 |
| 4 种模式位（normal 01 / retain 00 / clear-each-step 11 / clear-per-step-only 10） | `lnn_cfc_run.py` 的 `MODE_MAP` | [包内] | commands §5、T14 |
| 板端 PASS 判据（source/ID/VERSION/CLOCK_HZ/64-64/final_state/step_count/error_code） | `src/lnn/docs/USER_GUIDE.md` §4；`board/lnn/README_BOARD.md` §3；`INTERFACE.md` §6 | [包内] | SKILL §4 步骤 7、acceptance §4 |
| 三件套哈希 bit `0fcd8d65…` / hwh `fe2d9f0e…` / xsa `f01d6a8a…` | 本机 `sha256sum` 实算；`board/lnn/SHA256SUMS`；`ACCEPTANCE.md` §3 | [包内] | SKILL §4 步骤 6、artifacts §5 |
| XSA 内 `lnn_cfc.bit` == 外部 bit；XSA 内 `design_1.hwh` == 外部 hwh | 本机 `zipfile+hashlib` 实算；`ACCEPTANCE.md` §3"配对检查" | [包内] | SKILL §4 步骤 6、T4 |
| `SHA256SUMS` 42 行全 OK | 本机 `sha256sum -c SHA256SUMS`（exit 0） | [包内] | commands §3.2 |
| 离线校验 29/29 PASS 但 `source=simulation-model`、`is_hardware_measurement=false` | `sim/lnn/evidence/offline_validation_summary.json` 顶层字段；`board/lnn/README_BOARD.md` §0 | [包内] | SKILL §6、acceptance §4 |
| 板端实测**不在包内**，需现场生成 `source=hardware` 的 summary | `ACCEPTANCE.md` §1 板端行；`README_BOARD.md` §7 已知限制 1 | [包内] | SKILL §6/§8 |
| 离线模型 `run_cycles=2910` 是模型假设（`30+45*n`） | `board/lnn/cfc_sim_model.py` `cycles_base/cycles_per_step` | [包内] | T13 |
| `export/` 缺失时的报错与查找顺序 | `board/lnn/lnn_cfc_run.py` 的 `_load_reference/discover_paths`；`board/lnn/README_BOARD.md` §3 | [包内] | T2 |
| pycache 导致 manifest 校验失败的现场记录 | `[工作区] user/README_LNN_DEMO.md` §1.4；`manifest.json` 2 条 pyc 条目 | [工作区]+[包内] | T1 |

## 6 本机实测记录（2026-11-02，可复算）

| 检查 | 命令 | 结果 |
|---|---|---|
| 历史提炼融合 | 读 `D:\DL_FINAL\_skill_work\history\lnn.md`（B1–B9 踩坑、C1–C7 决策、E 实测数值） | 已并入 troubleshooting T1–T15 与 acceptance；其中 B1→T5、B3→T4、B4→T2、B7→acceptance §4、B8→acceptance §3 |
| 导出器接口 | `python user/export_demo.py --help`（工作区） | 仅 `--out-dir`，退出码 0 |
| 期望值同源 | `diff -rq data/lnn/export board/lnn/export` | 无差异，退出码 0 |
| gen 文件数 | `ls data/lnn/export/gen \| wc -l` | `23` |
| 三件套哈希 | `sha256sum lnn_cfc.{bit,hwh,xsa}` | 三个哈希与文档一致 |
| XSA 内嵌 | `python -c "zipfile+hashlib"` | bit/hwh 均与外部一致 |
| 交付清单 | `sha256sum -c SHA256SUMS` | 42/42 OK |
| 定点自检 | `python cfc_sim_model.py --selfcheck <export>` | 14 OK + PASS |
| 驱动样本枚举 | `python lnn_cfc_run.py --list-samples` | 8 样本，退出码 0 |
| 环境哈希 | `python scripts/env_check.py --check-hashes`（工作区） | `108 files checked, 0 bad` |
| 包内 run_sim 前置 | `tclsh build/lnn/scripts/run_sim.tcl` | exit 1，`missing export/gen/cfc_weight_init.svh` |
| 工作区 run_sim 前置 | `tclsh scripts/run_sim.tcl`（工作区） | exit 1，`xsim tools not found` |
| RTL 哈希 | `sha256sum src/lnn/rtl/*.sv` | 8/8 与冻结值一致 |

## 7 不能复述的说法（已被证据否定或容易误读）

| 错误说法 | 正确口径 | 依据 |
|---|---|---|
| "LNN 有 84 个参数" / "84 个权重就是全部参数" | 84 权重 + 17 偏置 = **101 参数** | `manifest.json` `model.n_weights/n_biases` |
| "门控只用 a，不需要 W_b" | V1.1 必须含 `W_b`：`gate_pre=ROUND(lin_a*655+(lin_b<<12))` | `CFC_CONTRACT.md` 版本历史、`RELEASE_NOTES.md` |
| "离线 29/29 PASS 就是板端通过" | 离线是 `simulation-model`，只能标 `partial` | `offline_validation_summary.json`、`README_BOARD.md` §0 |
| "离线 run_cycles=2910 是硬件周期" | 硬件以板上 `summary.json` 为准；RTL 参考 9088 | `cfc_sim_model.py`、`INTERFACE.md` §5 |
| "仿真通过＝可综合/时序收敛" | 跨级不传递；G1 不证明 G3/G6 | `evidence-gate/reference/gates.md` |
| "bit 与 hwh 可以分别从不同构建取" | 必须同 stem 同次构建，XSA 内嵌逐字节一致 | `ACCEPTANCE.md` §3、`README_BOARD.md` §2 |
| "包内 src/lnn 有导出器/仿真器" | 包内 `src/lnn` 只有 `README/QUICKSTART/docs/rtl`；导出器与 TB 在别处或工作区 | 实测 `find src/lnn`（12 个文件） |
| "包内有 `VERSION` 文件" | **没有**；版本串只在 `src/lnn/README.md` 首行出现 | 实测 `find /d/FINAL -name VERSION` 无结果 |

## 8 未找到证据的项（不得臆造）

| 主题 | 状态 | 已做到的替代 |
|---|---|---|
| 板端实测 `summary.json`（`source=hardware`） | 需真实板卡 | 只写判据；离线 29/29 属流程自检 |
| 导出器/ONNX 导出脚本随包 | 不随包 | 引用 [工作区] `user/gen/generate_export.py`、`D:/DL_LNN/model/export_onnx.py` |
| 位流重建脚本与 Vivado 工程 | 不随包 | 只给"需重新构建并重做三件套配对"的规则 |
| "LUT 越界""状态未初始化"的现场报错文本 | 快照与开发报告中均无实测记录 | T14/T15 标为**推导/防御性条款**，只给机制与检测命令 |
| `run_sim.tcl` 在包内布局下的官方跑法 | 包内脚本沿用快照布局 | T9 给实测失败文本 + 两种可行替代 |
