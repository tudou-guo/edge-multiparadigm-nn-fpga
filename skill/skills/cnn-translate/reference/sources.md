# sources.md — 事实来源索引

> 规则：每条关键结论 → 出处文件（包内写相对包根；仅开发机存在写绝对路径并标 `[工作区]`）→ 可在文件里定位的字段/行。
> 包内路径已用 `Test-Path` 逐条核实存在；`[工作区]` 路径同理。

## 1 CLI 与编译器行为

| 结论 | 出处 | 字段/位置 |
|---|---|---|
| 编译命令全部选项、默认值 | `src/cnn/tools/onnx_gen/onnx_gen.py` | `build_parser()` `compile_cmd` 段（`--model`…`--report`） |
| 退出码 0/2/4 语义 | `src/cnn/tools/onnx_gen/onnx_gen.py` | `command_compile`：`return 0` / `return 2` / `return 4` |
| success JSON 字段 | 本机 2026-11-02 实跑输出 | `status/manifest/records/graph_digest/ip/engine_profile/target/depthwise/generated_parameters` |
| `--mac-impl` 只接受 1/3 | `src/cnn/tools/onnx_gen/compile_pipeline.py` | `_resolve_compute_profile`：`if int(legacy_mac) not in (1, 3)` |
| 默认 tn8/tm8/mac_impl3 | `src/cnn/tools/onnx_gen/compile_pipeline.py` | 同上 `legacy_tn = 8 if explicit_tn is None else ...` |
| 公开档位 tn4/tn8/tn16 与 DSP 估算 | `src/cnn/tools/onnx_gen/profiles/tn4.json`、`tn8.json`、`tn16.json`；`compute_profiles.py` | `budget.dsp_array_estimate` 72/288/1152 |
| `--vivado` 默认 `E:/Vitis/Vivado/2021.1/bin/vivado.bat` | `src/cnn/tools/onnx_gen/onnx_gen.py` | `compile_cmd.add_argument("--vivado", default=...)` |
| `--pack-ip` 默认开 | 同上 | `compile_cmd.set_defaults(pack_ip=True)` |
| `--source-root` 默认 = 包内 `src/cnn` | 同上 | `default=Path(__file__).resolve().parents[2]` |
| `--run-id` 默认 `e1p2` | 同上 | `compile_cmd.add_argument("--run-id", default="e1p2")` |
| `--help` 真实输出 | 本机实跑 | commands.md C2 引用的 49 个选项 |

## 2 输入合同

| 结论 | 出处 |
|---|---|
| QDQ 路径 opset 13..19 | `src/cnn/tools/onnx_gen/qdq_frontend.py` `check_opset`（`"opset %d not supported (QDQ graph requires 13..19)"`） |
| 主前端 opset 11..19 | `src/cnn/tools/onnx_gen/onnx_frontend.py` `_parse_opset`；`op_registry.py` `MIN_OPSET=11/MAX_OPSET=19` |
| 动态维/N=1 拒绝 | `src/cnn/tools/onnx_gen/qdq_frontend.py` `_tail_dims`（`E_QDQ_DYNAMIC_SHAPE`） |
| qs=qu-128 重编码 | `src/cnn/tools/onnx_gen/affine_contract.py`（`zero_point_u8_to_s8`）、`board/cnn/runtime/decode/preprocess.py` |
| affine-int8-v1 合同 schema/rounding/requant | `src/cnn/tools/onnx_gen/affine_contract.py`（`v0.2-quant-contract-affine-1`、`ties_away_from_zero`、`m0_bits=31`、`shift_max=62`） |
| 16 操作码与逐算子限制 | `src/cnn/tools/onnx_gen/op_registry.py` `_DEFAULT_SPECS`；`src/cnn/docs/OPERATORS.md` |
| 10 个主线引擎目录 | `src/cnn/rtl/operator_library.json` `engines[]`；`src/cnn/rtl/ops/README.md` |
| Depthwise V1 限制（C≤16、H/W≤32、group=C） | `src/cnn/tools/onnx_gen/op_registry.py` DepthwiseConv 条目 |
| DW_PC/RQ 合法组合 | `src/cnn/rtl/operator_library.json` `depthwise.parameters`；`engine_profiles.py` |
| 特征图轴上限 640、SIM DDR 1..512 MiB | `src/cnn/tools/onnx_gen/spatial_capacity.py`（`TARGET_MAX_AXIS=640`、`MAX_SIM_DDR_LIMIT_MIB=512`） |
| 器件/时钟/AXI 目标 | `src/cnn/tools/onnx_gen/targets.py`（`xczu15eg-ffvb1156-2-i`、`clock_ns=5.0`、`axi_data_width=128`） |
| MAC_IMPL 0/1/2/3 的 RTL 编码 | `src/cnn/rtl/common/npu_backend_contract.vh`（`NPU_MAC_LEGACY/DSP_ADAPTER/BALANCED/BALANCED_DSP`） |

## 3 产物与清单

| 结论 | 出处 |
|---|---|
| 编译输出文件集合 | 本机 2026-11-02 实跑目录；[工作区] `D:\DL_FINAL\DL\examples\tinycnn28_mnist\artifacts\toolchain_compile_tn4_tm4/` |
| compile_manifest 字段与真实值 | `data/cnn/tinycnn/compile_manifest.json`、`data/cnn/mobiledw/compile_manifest.json` |
| gates 12 项真实取值 | 同上 `gates` 对象 |
| rtl_manifest 字段与 source_digest | [工作区] `.../toolchain_compile_tn4_tm4/generated/rtl_manifest.json`；算法见 `src/cnn/tools/onnx_gen/source_library.py` |
| instruction_stream 结构与 17 字载荷 | [工作区] `.../generated/instruction_stream.json`（`payload_fields`）；`data/cnn/tinycnn/compile_manifest.json` `records` |
| parameter_manifest requant 记录 | 本机实跑 `parameter_manifest.json`（`M0=1131972204`、`SHIFT=39`） |
| ddr_init base=0x10000000 | `data/cnn/*/ddr_init_manifest.json`、`board/cnn/runtime/ddr_init_manifest.json` |
| snapshot_manifest 9/15 条 | `data/cnn/tinycnn/snapshot_manifest.json`、`data/cnn/mobiledw/snapshot_manifest.json` |
| qdq_report 字段与案例精度 | `data/cnn/tinycnn/qdq_report.json`、`data/cnn/mobiledw/qdq_report.json` |
| IP pack_manifest/component.xml 结构 | [工作区] `D:\DL_FINAL\DL\examples\mobiledw32_cifar10\artifacts\ip_repo_mobiledw32_tn4_tm4_20261030/` |
| `tools/npu_gen` 无 Python 入口 | `src/cnn/tools/npu_gen/` 仅 `pack_npu_ip.tcl`、`npu_top.v.tpl`；`python -m tools.npu_gen` 实测 `No module named tools.npu_gen.__main__` |

## 4 回放、综合、平台、板端

| 结论 | 出处 |
|---|---|
| ModelSim 回放参数与 PASS 判据 | `build/cnn/scripts/run_p3_graph_shell.ps1`（第 1–25 行参数；第 310 行 `P3_RESULT: PASS`；`summary.json` 字段） |
| 回放 testbench | `sim/cnn/verification/sim/tb_p3_graph_shell.sv` |
| 历史回放结果 | `data/cnn/mobiledw/summary.json`（`status=pass`、`board_status=not_run`、`sim_define=TB_MAC_IMPL_2,TB_LEAN_FUNC`） |
| OOC 用法与标记 | `build/cnn/scripts/run_e1p2_ooc_synth.tcl`（`USAGE` 注释、`P2_OOC_SYNTH_DONE=true`、报告文件名） |
| OOC 的 DSP XCI 硬编码路径 | 同上（`set ipxci_src [file join $root build/mnist_project2_board/reloc_mn8_ip_repo3/...]`） |
| DSP XCI 的真实包内位置 | `src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci`（sha256 `25cbfe8f...`，见 `source_library list` 的 `vendor_ip`） |
| BD 参数钉扎参数表 | `build/cnn/scripts/configure_e1p3_bd.tcl` `USAGE` |
| 平台状态与两种存储器路径 | `src/cnn/docs/PLATFORM_AND_STATUS.md` |
| r4 实现/板测身份与数字 | `board/cnn/MANIFEST.json` `identity`/`board_runs`/`board_validation` |
| 板端命令与结果口径 | `board/cnn/README_BOARD.md` |
| 板端脚本参数与身份检查 | `board/cnn/scripts/board_run_a.py`（`argparse`、`ddr_init.bin identity mismatch`） |
| 板端输入打包与预处理合同 | `board/cnn/runtime/decode/preprocess.py`（常量与 `pack_nhwc8_a16`） |
| 6 head golden 与地址 | `board/cnn/runtime/expected_heads.json`、`board/cnn/runtime/golden/` |
| YOLO 模型/期望哈希 | `data/cnn/yolo/model/{graph,heads,inputs,export_report}.json`、`data/cnn/yolo/expected_hashes.json` |
| 两个不同 bit 包 | `board/cnn/MANIFEST.json` 的 `identity.bit_sha256`（`baeb25fc...`）vs `data/cnn/yolo/expected_hashes.json` 的 `board_package.bit_sha256`（`619f900c...`） |
| 历史报告（非本次） | `build/cnn/reports/r4_platform_utilization.rpt`、`build/cnn/reports/legacy/*.rpt`（报告头含 `Command:` 原路径与 `Date:`） |

## 5 历史踩坑（[工作区] 来源，已被包内事实交叉验证）

| 结论 | 出处 |
|---|---|
| u8/s8 输入重编码根因与两个 ddr_init 哈希 | `[工作区] D:\dl_v1\docs\M4_PLATFORM_A_BOARD_CLOSURE_20261028.md`；包内 `board/cnn/MANIFEST.json` 的 `identity.input_recode_fix` |
| 行为仿真 PASS 但综合后 `ERR_OPCODE=2` | `[工作区] D:\dl_v1.0\docs\PYNQ_ERR_OPCODE_ROOT_CAUSE_FIX_20261017.md` |
| 200 MHz 时序不收敛与 clk_wiz PRIM_SOURCE | `[工作区] D:\dl_v1\docs\M4_PLATFORM_A_IMPL_TIMING_QUALIFIED_20261028.md` |
| 旧 PASS 不能跨源码继承 | `[工作区] D:\dl_v1\docs\RELEASE_STATUS_20261029.md`；`[工作区] D:\dl_v1\docs\TOOLCHAIN.md` |
| 板端 BUSY 采样竞态 | `[工作区] D:\dl_v1.0\docs\F0_BUSY_R2_ROOT_CAUSE_R3_FIX_20261017.md` |
| 历史提炼汇总（命令/决策/数值） | `[工作区] D:\DL_FINAL\_skill_work\history\cnn.md`；被否定说法清单 `evidence_map.md` |

## 6 本技能自身

| 结论 | 出处 |
|---|---|
| 章节骨架与红线 | `[工作区] D:\DL_FINAL\_skill_work\SPEC.md` |
| G0–G6 门级/标签/证据目录 | `skill/skills/evidence-gate/SKILL.md`、`skill/skills/evidence-gate/reference/gates.md` |
| 包内 cnn 路径确实存在 | 逐条 `Test-Path`/`ls` 核实（2026-11-02）；核实结果见 task-2 完成汇报 |
| 包内无 `src/cnn/examples/`、`verification/`、`scripts/`、`requirements.txt` | `ls src/cnn` 实测（只有 `docs/ rtl/ tools/ README.md QUICKSTART.md`） |
