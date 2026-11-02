---
name: cnn-translate
version: 2.0.1
description: 把静态 INT8 QDQ ONNX（opset 13..19、N=1、affine-int8-v1）编译为指令流/参数包/DDR 镜像/模型专用 RTL/Vivado IP 源码包，并给出 G0~G6 分级验收命令与证据落点。
applies_to: [cnn]
package_paths: [src/cnn, data/cnn, build/cnn, sim/cnn, board/cnn]
requires: [Python 3.9.x, onnx 1.14.1, numpy 2.0.2, pillow 10.1.0, 可选: Vivado 2021.1, 可选: ModelSim SE-64 2020.4]
last_verified: 2026-11-02
---

# SKILL: cnn-translate

## 0 一句话定义

把**静态 INT8 QDQ ONNX** 转译为"可综合的 CNN 加速器工程"：图规划、指令流、参数包、DDR 初始化镜像、
模型专用 RTL 顶层、源码身份清单，以及可选的 Vivado IP Catalog 源码包；本技能同时规定每一级的
判定命令、阈值与证据文件。**编译成功不等于数值正确、不等于时序收敛、不等于板端通过。**

## 1 何时使用 / 何时不要使用

**使用**：

- 已有一份完成 QDQ 量化的静态 ONNX，要生成指令/参数/DDR 镜像与模型专用 RTL；
- 要复核随包清单（`data/cnn/*/compile_manifest.json`）是否与当前源码同源；
- 要把生成的 RTL 打成 Vivado IP Catalog 源码包，或要为 ModelSim 回放、OOC 综合、板端比对准备输入；
- 要把 CNN 某一级结论写进报告（与 [evidence-gate](../evidence-gate/SKILL.md) 的 G0–G6 门级对齐）。

**不要使用**：

- 训练、量化校准、QDQ 导出：本包不含这些脚本（训练脚本只在 [工作区] `D:\DL_FINAL\DL\examples\*`）；
- 期望编译器给出资源/时序预测：编译器只生成 RTL 与身份清单，不综合、不布局布线（`src/cnn/README.md`）；
- 期望 IP 包直接上板：IP 包是 RTL 源码组件，不含 Block Design、bitstream、XSA（`src/cnn/docs/PLATFORM_AND_STATUS.md`）；
- 期望用旧 bit/HWH 证明新源码：硬件证据随硬件包身份失效（见第 9 节）。

## 2 前置条件与环境自检

包根按相对路径书写；本机包根为 `D:\FINAL`。**编译器必须在 `src/cnn` 下执行**（模块入口 `tools.onnx_gen.onnx_gen`）。

| 项 | 要求 | 依据 |
|---|---|---|
| Python | 3.9.x；实测 3.9.23 | 本机 `python --version` |
| Python 包 | `onnx` 1.14.1、`numpy` 2.0.2、`pillow` 10.1.0 | [包内] `build/cnn/scripts/env_check.py` 实测输出 |
| ONNX 前端 | 默认域单一 opset，QDQ 路径要求 **13..19** | `src/cnn/tools/onnx_gen/qdq_frontend.py`（`check_opset`） |
| 目标器件 | `xczu15eg-ffvb1156-2-i`，5 ns（200 MHz），128-bit AXI | `src/cnn/tools/onnx_gen/targets.py`（`targets` 子命令） |
| Vivado | 2021.1（仅 IP 打包需要；不在 PATH 时用 `--no-pack-ip` 或 `--vivado`） | `src/cnn/tools/onnx_gen/onnx_gen.py` 默认 `E:/Vitis/Vivado/2021.1/bin/vivado.bat` |
| ModelSim | SE-64 2020.4（仅 G1 回放需要） | `data/cnn/mobiledw/summary.json`（`simulator_version`） |

自检命令（包根执行，实测通过）：

```powershell
Set-Location D:\FINAL
python -B build\cnn\scripts\env_check.py
```

期望输出（本机 2026-11-02 实测，退出码 0）：首行 `python 3.9.23`，`numpy OK 2.0.2`、`onnx OK 1.14.1`、
`pillow OK 10.1.0`，`vsim/vlog/vlib` 找到（`D:\modelsim\win64\*.EXE`），`vivado/xvlog/xelab/xsim` 显示 `not in PATH (optional)`，
末行 `ENV_CHECK: PASS`。可选工具缺失不阻断编译（只阻断 IP 打包/回放）；`FAIL` 时先修 `onnx/numpy/pillow`。

## 3 输入合同

| 输入 | 格式 | 硬约束 | 校验命令 | 不满足时的行为 |
|---|---|---|---|---|
| 模型 | `.onnx`（QDQ） | 默认域**单一** opset **13..19**；否则拒绝 | 见命令 C2 的 `--report` | exit 2，`E_QDQ_UNSUPPORTED_OPSET` |
| shape | 静态 NCHW | 所有维度 >0、**N=1**；动态/零维拒绝 | 同上 | exit 2，`E_QDQ_DYNAMIC_SHAPE` |
| QDQ 模式 | QuantizeLinear/DequantizeLinear | 激活以 uint8 存储后重编码为 signed INT8（`qs = qu - 128`）；权重 int8、w_zero_point=0；bias int32；per-tensor | 同上 | exit 2，`E_QDQ_PATTERN`/`E_QDQ_DOMAIN_MISMATCH` |
| 数值配置 | `affine-int8-v1` | rounding=`ties_away_from_zero`；M0 31-bit、SHIFT≤62；INT32 累加 | `data/cnn/*/qdq_report.json` 的 `note`/`onnx` | exit 2，`E_QUANT_MISMATCH`/`E_CFG` |
| 输入样本 | `.npy` uint8，NCHW/NHWC | 与模型输入 shape 一致；`--numeric-profile affine-int8-v1` 时**必须**给 `--input-u8` | 命令 C2 | exit 2，`E_CFG: affine numeric profile requires --input-u8` |
| 算子 | 16 个主线操作码 | 逐条见下表；未登记算子/未实现属性直接拒绝，不静默回退 | `registry --out` 导出注册表 | exit 2，`E_UNSUPPORTED_OP`/`E_UNSUPPORTED_ATTR` |
| 并行度 | `--tn/--tm` 1..16；`--mac-impl` **仅 1 或 3** | 公开档 `tn4/tn8/tn16`（默认 tn8/tm8/MAC_IMPL=3）；`--mac-impl 2` 被 CLI 拒绝 | 命令 C5 | exit 2，`E_CFG`/`E_CP_UNSUPPORTED_TILE` |
| Depthwise 参数 | `--dw-pc` 4/8/16，`--dw-rq-lanes` 1/2（PC=16 才允许 2），`--dw-tap-parallel` 仅 1 | `MAX_CH = max(图推导 MAX_CH, align8(DW_PC))` | `engine-profiles` 子命令 | exit 2，`E_CP_*` |
| 输出目录 | 新目录 | 每次编译用**新的空目录**，不复用旧编译目录 | 人工检查 | 旧目录会绑定旧身份，产生身份漂移（见 troubleshooting T2） |

**算子子集（16 操作码 / 10 个主线引擎目录；`0x001E` 为隔离候选）**：

| 操作码 | 算子/类别 | RTL 目录 [包内] | 关键限制（`op_registry.py`） |
|---|---|---|---|
| `0x0001` | Conv（3×3；1×1 降到中心 3×3） | `src/cnn/rtl/ops/conv/` | stride 1/2，pad 0/1，dilation 1，group=1，N=1 |
| `0x0002` | MaxPool 2×2 / 5×5 SPPF 专用核 | `src/cnn/rtl/ops/pool2x2/` | 2×2 stride2 pad0 偶数 H/W，或 5×5 stride1 pad2 形状保持 |
| `0x0010` | Relu | `src/cnn/rtl/ops/eltwise/` | INT8 clamp(min=0)，zero-point 0 |
| `0x0011` | Add（同 shape） | `src/cnn/rtl/ops/eltwise/` | 无广播；同 shape/dtype/layout 饱和加 |
| `0x0012` | Pad | `src/cnn/rtl/ops/copy/` | mode=constant；rank-4 NCHW N=1；仅 H/W |
| `0x0013` | Slice | `src/cnn/rtl/ops/copy/` | step=1；H/W/C 静态 |
| `0x0014` | Concat | `src/cnn/rtl/ops/copy/` | 仅 channel 轴；2..4 个同 layout/quant 输入 |
| `0x0015` | Clip | `src/cnn/rtl/ops/eltwise/` | 静态整数阈值，zero-point 0 |
| `0x0016` | GlobalAveragePool | `src/cnn/rtl/ops/reduce/` | 同 scale 的精确舍入均值 |
| `0x0017` | AveragePool 2×2 | `src/cnn/rtl/ops/pool2x2/` | stride2 pad0 偶数 H/W；四数和按 4 ties-away |
| `0x0018` | Mul（同 shape/scale） | `src/cnn/rtl/ops/eltwise/` | M0≤0xFFFFFF，SHIFT≤31；无广播 |
| `0x0019` | Resize nearest ×2 | `src/cnn/rtl/ops/resize/` | mode=nearest、asymmetric、floor；输出 H/W≤32 |
| `0x001A` | MaxPool 5×5 / SPPF 构件 | `src/cnn/rtl/ops/pool5/` | 见 OPERATORS.md |
| `0x001B` | ActivationLUT（Sigmoid/Tanh/SiLU/LeakyReLU） | `src/cnn/rtl/ops/activation/` | 需显式 LUT 合同（`--activation-contract`）；纯语义节点不直接落硬件 |
| `0x001C` | Depthwise 3×3（group=C、multiplier=1） | `src/cnn/rtl/ops/depthwise/` | V1 ABI；N=1、C≤16、H/W≤32，stride 1/2，pad 0/1 |
| `0x001D` | 通用 MaxPool2D | `src/cnn/rtl/ops/maxpool2d/` | V1 signed INT8、8 lane；物理资格需单独做平台验证 |

`0x001E`（Depthwise Conv V2）在 [工作区] `D:\DL_FINAL\DL\examples\depthwise_conv_v2\candidate` 的隔离编译器里，
**不在包内主线上**；包内主线 CLI 没有 `--depthwise-v2` 选项，不要按 QUICKSTART 之外的写法臆造。

## 4 执行步骤

### 步骤 1：环境自检

命令与期望输出见第 2 节。`ENV_CHECK: PASS` 才进入步骤 2。

### 步骤 2：导出算子注册表并对模型做结构化拒绝检查（可选但推荐）

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.onnx_gen registry --out <new-dir>\registry.json
```

期望：`{"status": "pass", "out": ...}`；文件 `schema=e1-op-registry-1`、`opset_range=[11,19]`、`operators` 26 条。
随后用 `--report <file>` 跑一次目标模型：被拒绝时报告里是 `{"status":"fail","compile_graph":"rejected","error":{"code":...}}`，
且**不会留下半成品输出目录**。

### 步骤 3：编译（G0）

包内自带工具链但没有样例模型（`src/cnn/examples/` 未随包），模型与样本需自备；开发期完整案例在 [工作区]。
先用 `--no-pack-ip` 只生成编译产物：

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.onnx_gen compile `
  --model <model_qdq.onnx> --numeric-profile affine-int8-v1 --input-u8 <input_u8.npy> `
  --mac-impl 3 --tn 4 --tm 4 --name <graph_name> --run-id <run_id> `
  --out-dir <新目录> --no-pack-ip
```

期望（本机用 [工作区] TinyCNN 实测，退出码 0）：

```json
{"status": "pass", "manifest": "...\\compile_manifest.json", "records": 8,
 "graph_digest": "832221ba22961bb73c42dd349d2ae2d5f1d67aeab051ce08c44cadd18143b05a",
 "ip": null, "engine_profile": null, "target": null, "depthwise": null, "generated_parameters": null}
```

失败分支：exit 2=图/合同/配置被拒（看 `error.code`）；exit 4=编译通过但 IP 打包失败（看 `error.detail.log`）；不要复用已有输出目录。

### 步骤 4：核验产物身份自洽（G0 硬门）

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.source_library verify --generated <out-dir>\generated
```

期望：`{"status":"pass","source_files":44,"package_files":0,...}`；`source_digest` 必须等于
`<out-dir>\compile_manifest.json` 与 `<out-dir>\generated\rtl_manifest.json` 里的 `source_digest`。
包内 `src/cnn` 源码集当期的 `source_digest` 实测为 `36ca00cc095ea7bb010d4576942e8a881a099001a109185a36eff006f67b1061`，
与随包 `data/cnn/tinycnn/compile_manifest.json` 一致。

### 步骤 5：IP 打包（G4，可选）

省略 `--no-pack-ip`（默认即打包）或在已生成目录上单独跑打包器：

```powershell
python -B -m tools.onnx_gen.onnx_gen compile ... --ip-version 0.1 --vivado <vivado.bat>
```

产物：`<out-dir>/ip_repo/<ip_name>_<version>/component.xml`、同级 `<ip_name>_<version>.pack_manifest.json`、
日志 `<out-dir>/logs/pack_npu_ip.log`。已存在的 `ip_repo/<ip_name>_<version>` 会被拒绝重打（`IP directory already exists; refusing to repack`）。

### 步骤 6：ModelSim 图级回放（G1）

```powershell
Set-Location D:\FINAL
powershell -ExecutionPolicy Bypass -File build\cnn\scripts\run_p3_graph_shell.ps1 `
  -CompileDir <out-dir> -ResultDir <result-dir> `
  -ProjectRoot <compile 所在工程根> -Simulator modelsim -SimDefine TB_MAC_IMPL_2 `
  -Seed 20261030 -Vlog D:\modelsim\win64\vlog.exe -Vsim D:\modelsim\win64\vsim.exe -Vlib D:\modelsim\win64\vlib.exe
```

阈值：`summary.json` 里 `status=pass`、`vlog_rc=0`、`sim_rc=0`、`pass_marker=true`、`source_unchanged=true`，
且没有 `** Error/Fatal`；`simulator.log` 含 `P3_RESULT: PASS`；快照 mismatch=0。
（需要 ModelSim SE-64 2020.4 与同源 `ProjectRoot`；命令约束见 `reference/commands.md` C7。）

### 步骤 7：OOC 综合（G2）

```powershell
vivado -mode batch -notrace -source build\cnn\scripts\run_e1p2_ooc_synth.tcl `
  -tclargs <out-dir>\generated <ooc-out-dir> <project_root>
```

期望：日志出现 `P2_OOC_SYNTH_DONE=true`，并生成 `utilization.rpt`、`timing_summary.rpt`、`drc.rpt`、
`npu_top_ooc.dcp`。
（需 Vivado 2021.1；脚本硬编码 DSP IP 依赖路径，见第 7 节 T8，需按包内 `src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci` 调整。）

### 步骤 8：平台实现 / BD 集成（G3/G5，需要 Vivado 工程）

包内只提供脚本与历史结果，不提供可用 BD 工程：`configure_e1p3_bd.tcl` 用于把参数钉到 BD 单元
（`-tclargs <root> <cell> <max_cin> <max_cout> <max_ch> <max_w> <num_layers> <instr_words> <instr_addr_w> <instr_init> <mac_impl> <tn> <tm> [dw_pc dw_tap_parallel dw_rq_lanes]`）。
判定阈值见第 6 节 G3。历史 r4 数字（2026-10-29 实现）只作参照，不得当作本次实现结论。

### 步骤 9：板端比对（G6，历史包）

```bash
cd board/cnn
sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh \
  --ctrl-base 0x480000000 --window-base 0x400000000 --out-dir run_out_heads
sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh \
  --ctrl-base 0x480000000 --window-base 0x400000000 --image runtime/test_image.jpg --out-dir run_out_image
```

阈值：6 个 head 全部匹配（`first_mismatch=-1`）、`error_code=0`；image 模式检测结果与 golden 一致。
包内记录是 **2026-10-29 用户提供的 Jupyter 输出**（`MANIFEST.json` 的 `board_status=PASS_user_reported`），
属历史实现（2026-10-29），写报告时须注明来源与日期。

## 5 产物与清单

| 路径（编译输出目录内） | 生成者 | 关键字段/身份 | 如何自证 |
|---|---|---|---|
| `compile_manifest.json` | compile | `schema=e1-p2-compile-manifest-1`、`model_sha256`、`ddr_init_sha256`、`instruction_mem_sha256`、`source_digest`、`gates` | `source_library verify` + 复算 sha256 |
| `ddr_init.bin` + `ddr_init_manifest.json` | compile | `base=0x10000000`、`bytes`、`sha256`、`writes[]` | 复算文件 sha256 与清单 `sha256` 比对 |
| `graph_plan.json` / `layout_plan.json` / `memory_plan.json` | plan/layout/memory | `graph_digest`、`arena.base=0x10000000`、`checks` 全 pass | 人工读取 `checks.*=pass` |
| `lowered_graph_ir.json` / `source_graph_ir.json` / `lowering_manifest.json` | frontend/lowering | `lowered_graph_digest`、`passes[]`、映射明细 | digest 与 compile_manifest 一致 |
| `parameter_manifest.json` + `params/*.bin` | compile | `parameters[].address/file/sha256/requant_records{M0,SHIFT}` | 逐个 `params/*.bin` 复算 sha256 |
| `generated/instruction_stream.json` + `*_instr.mem` | codegen | `schema=e1-instruction-stream-2`、`framing{0xFFFFFFFF}`、`records[]`、`mem_sha256` | `records` 条数>0；mem sha256 与清单一致 |
| `generated/rtl_manifest.json` / `sources.f` / `source_hashes.tsv` / `source_root.txt` | codegen | `schema=e1-graph-rtl-manifest-1`、`source_digest`、`source_files[]`、`files_sha256`、`template`、`vendor_ip` | `source_library verify` |
| `generated/npu_top.v` | codegen | 顶层参数（`MAC_IMPL/TN/TM/MAX_*`） | 与 `rtl_manifest.capacities` 对照 |
| `snapshot_manifest.json` / `snapshot_desc.txt` / `golden/*.bin` | compile | `snapshots[].name/address/bytes/sha256` | 回放快照与 golden 逐字节比对 |
| `ip_repo/<name>_<ver>/component.xml` + `.pack_manifest.json` | Vivado packer | VLNV、端口映射、`counts{total,rtl,new_core,xci}`、每文件 sha256 | `source_library verify --package <ip_dir>` |
| `compile_report.json`（`--report` 或案例产物） | compile | `schema=e1-compile-cli-1`、`requested`、`status`、`records` | 与 manifest 交叉比对 |

字段级明细见 `reference/artifacts.md`。

## 6 验收（分级）

| 级 | 判定命令 | 阈值 | 证据落点 |
|---|---|---|---|
| G0 编译 | 命令 C2 | exit 0；`status=pass`；`records>0`；`gates` 中 frontend/lowering/codegen/memory_planning/parameter_pack/quant_contract/spatial_capacity/affine_frontend/reference 全 pass；`source_library verify` pass | 新输出目录 + `compile_manifest.json` |
| G1 回放 | 命令 C7 | `summary.json` pass、0 mismatch、`source_unchanged=true` | `<result-dir>/summary.json`、`simulator.log`、`snapshots/` |
| G2 OOC | 命令 C8 | 日志 `P2_OOC_SYNTH_DONE=true`、无 error；产出 `utilization/timing/drc` 报告 | `<ooc-out-dir>/*.rpt` |
| G3 实现 | Vivado 平台工程（包内无） | setup WNS≥0、hold WHS≥0、0 failing endpoints、0 unrouted、DRC 0 Error | 平台 `timing_summary.rpt`（历史 r4 见 `board/cnn/MANIFEST.json`） |
| G4 IP 包 | 命令 C5 | `component.xml` 存在；`pack_manifest` 计数自洽；`source_library verify --package` pass | `<out-dir>/ip_repo/` |
| G5 平台集成 | BD + IP + bit/HWH 同源 | bit/HWH/XSA 出自同一次实现；BD 可打开 | 平台工程与 `MANIFEST.json`（历史包） |
| G6 板端 | 命令 C9 | 6/6 head 逐字节一致、`first_mismatch=-1`、`error_code=0`；image 检测与 golden 一致 | `run_out_heads/`、`run_out_image/` 原始输出 |

## 7 故障与修复（摘要）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| T1 | ModelSim/板端 mismatch 上百万且 `first_offset=2` | DDR 输入镜像按 `qu` 打包，RTL 期望 `qs=qu-128` | 用 signed 重编码写 DDR；重编译后 `ddr_init` 哈希变化（T1 四元组见 troubleshooting） |
| T2 | 复用旧输出目录后清单/哈希对不上 | 目录内清单绑定旧模型与旧 RTL 身份 | 每次新空目录；`--no-pack-ip` 也需新目录 |
| T3 | `E_CFG: mac_impl must be 1 (behavioral) or 3 (DSP)` | CLI 只接受 1/3，旧文档写的 0/2/3 是 RTL 参数编码 | 改 `--mac-impl 3`（或 1） |
| T4 | `E_CFG: affine numeric profile requires --input-u8` | 选了 `--numeric-profile` 却没给样本 | 补 `--input-u8 <sample.npy>` |
| T5 | `E_QDQ_DYNAMIC_SHAPE`/batch≠1 | 模型含动态维或 N≠1 | 导出静态图、固定 N=1 |
| T6 | `E_QDQ_UNSUPPORTED_OPSET` | opset 不在 13..19 | 重新导出到 13..19（随包案例为 13） |
| T7 | `E_UNSUPPORTED_OP`/`E_UNSUPPORTED_ATTR` | 算子或属性不在注册表 | 改图或接受拒绝，不要静默替换 |
| T8 | OOC 脚本报 `missing .../dsp_mul_pipe4.xci` | 脚本硬编码 [工作区] 旧路径 | 改用 `src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci` |
| T9 | exit 4 / `E_CP_EXTERNAL: IP packaging failed` | Vivado 不在 PATH 或路径错 | 用 `--no-pack-ip` 或 `--vivado <path>`；看 `logs/pack_npu_ip.log` |
| T10 | replay 报 `SourceRoot differs from compiled source_root` | 编译后源码树变动/指错根 | 用编译时同一 source root，或重新编译 |
| T11 | ModelSim PASS 但综合后板端 `ERR_OPCODE=2` | 综合把引擎侧 RAM 写口优化成常量 | 综合后网表仿真/端口断言（T11 四元组） |
| T12 | 200 MHz 实现 WNS 负且 failing endpoints 上千 | clk_wiz 内部 primary 时钟被重定义 | 改 `PRIM_SOURCE=No_buffer`、重跑实现（T12 四元组） |

完整四元组（现象/根因/处置/证据）见 `reference/troubleshooting.md`（13 条）。

## 8 边界与不适用

- **不预测资源/时序**：编译器只产出 RTL 与清单；`build/cnn/reports/*` 是历史 r4/legacy 报告，
  r4 数字口径见 `board/cnn/MANIFEST.json` 的 `identity.implementation`，legacy 报告是更早平台（191,542 LUT / 527 DSP 等），
  两者都不是本次编译的资源结论。
- **IP 包 ≠ 平台**：不含 BD、bitstream、XSA；MobileDW 的 BD 迁移未完成（IP-XACT 仍含旧绝对依赖路径）。
- **不承诺精度**：QDQ 报告里的 97.9%/63.0% 是案例记录，不是本工具链对新模型的精度承诺。
- **不支持**：训练/量化校准（包内无脚本）、动态 shape、ONNX 广播、非 depthwise 的分组卷积、Resize 非 nearest×2、非 `xczu15eg-ffvb1156-2-i` 器件。
- **不要做**：用旧 bit/HWH 证明新源码；把 ModelSim 行为级 PASS 写成板端通过；把历史实现数字写成本次结果（见 `../evidence-gate/SKILL.md`）。

## 9 版本与身份绑定

- 工具链版本 **0.3.0（2026-10-31 发布整理快照）**：取自 `src/cnn/README.md`；RTL 库 `src/cnn/rtl/operator_library.json`
  的 `version` 为 `20261024.2`、`schema=e1-operator-library-1`。
- `source_digest` 算法：对 `{相对路径: sha256}` 映射（`sources[] + template + vendor_ip`）做
  `json.dumps(mapping, sort_keys=True)` 后 SHA-256（`src/cnn/tools/onnx_gen/source_library.py`）。
  它只含相对路径与字节哈希，与绝对根无关；因此包内 `src/cnn` 与 [工作区] `D:\DL_FINAL\DL` 的
  RTL 字节一致时 digest 相同（实测同为 `36ca00cc...`）。
- 清单绑定：`compile_manifest` 的 `model_sha256/ddr_init_sha256/instruction_mem_sha256/source_digest/quant_contract_sha256` 必须与文件/源码复算一致；`rtl_manifest.files_sha256` 覆盖生成文件。
- 硬件包绑定：`board/cnn/MANIFEST.json` 的 `identity.bit_sha256`（`baeb25fc...`）与
  `data/cnn/yolo/expected_hashes.json` 的 `board_package.bit_sha256`（`619f900c...`）**是两个不同硬件包**。
  旧 YOLO 期望哈希只能配它自己的 bit/HWH；`board/cnn` 的 r4 包只能配 `runtime/` 与 `scripts/board_run_a.py`。
- 失败即失效：源码/RTL/参数任一变化，旧的 G1/G2/G3/G6 结论全部作废，重跑并写 `invalidated_runs`。

## 10 证据归档

按 [evidence-gate](../evidence-gate/SKILL.md) 的门级与标签归档：

- G0：把 `compile_manifest.json`、`generated/rtl_manifest.json`、`source_hashes.tsv` 拷入证据目录；
  条目标 `measured`，带 `model_sha256 / source_digest / instruction_mem_sha256` 三个哈希。
- G1：`summary.json`、`simulator.log`、`snapshots/`、`simulation_identity.json`；标签 `pass` 需本次执行。
- G2/G3：`utilization.rpt`、`timing_summary.rpt`、`drc.rpt`，并在条目写明器件/Vivado 版本/时钟约束。
- G4：`component.xml`、`.pack_manifest.json`、`logs/pack_npu_ip.log`。
- G6：板端原始 JSON/日志 + bit/HWH/ddr_init 三件套哈希；外部提供的结果须注明来源，不与本机实测混写。
- 包内既有证据位置：`data/cnn/*`（G0/G1 记录）、`board/cnn/MANIFEST.json`（G3/G6 记录）、`build/cnn/reports/*`（历史报告）。

## 11 参考文件索引

- [commands.md](reference/commands.md)：可复制命令矩阵（含全部参数、退出码、期望输出）
- [artifacts.md](reference/artifacts.md)：产物与清单字段表（字段/类型/真实样例值/来源）
- [acceptance.md](reference/acceptance.md)：G0–G6 分级验收清单与当前包状态
- [troubleshooting.md](reference/troubleshooting.md)：13 条故障四元组
- [sources.md](reference/sources.md)：本技能每条关键结论的出处
- [evidence-gate](../evidence-gate/SKILL.md)：标签、门级、证据目录的通用规范
