# acceptance.md — CNN 分级验收清单（G0–G6）

> 门级定义与标签沿用 `../evidence-gate/SKILL.md` 与 `../evidence-gate/reference/gates.md`：
> 结论按 `pass / fail / partial / not_run` 分级；历史数据须注明来源与日期；估算须标 `derived`。
> 级不可跳：G1 通过不代表 G2；G3 的历史数据不能证明本次 G3。

## 0 判定前先回答的三个问题

1. 该结论是**本次执行**还是**历史记录**？历史记录须注明来源，不计为本机实测。
2. 证据是否与当前**源码/模型/硬件包**同一身份（`source_digest`、`model_sha256`、bit/HWH 哈希）？
3. 数字是 `measured` 还是 `derived`？估算不得写成资源/时序结论。

## 1 G0 编译

| 项 | 内容 |
|---|---|
| 命令 | commands.md C2（`compile --out-dir <新的空目录>`） |
| 通过阈值 | exit 0；stdout `status=pass`；`records>0`；`compile_manifest.gates` 中 `frontend/lowering/codegen/memory_planning/parameter_pack/quant_contract/spatial_capacity/affine_frontend/reference` 均为 `pass`；`source_library verify --generated` 为 `pass` 且 `source_digest` 三处一致 |
| 证据落点 | `<out-dir>/compile_manifest.json`、`<out-dir>/generated/rtl_manifest.json`、`<out-dir>/generated/source_hashes.tsv` |
| 复核命令 | `python -B -m tools.onnx_gen.source_library verify --generated <out-dir>/generated` |
| fail 判据 | exit 2（结构化拒绝，`error.code` 见 troubleshooting）；exit 4（IP 打包失败，不影响 G0 判定但需单列 G4）；`records=0`；`gates` 任一为 fail |
| 不能证明 | 数值正确、可综合、时序、板端 |

## 2 G1 数值与回放（ModelSim 图级）

| 项 | 内容 |
|---|---|
| 命令 | commands.md C7 `run_p3_graph_shell.ps1 -Simulator modelsim -SimDefine TB_MAC_IMPL_2` |
| 通过阈值 | `summary.json`：`schema=e1-p3-graph-sim-2`、`status=pass`、`vlog_rc=0`、`sim_rc=0`、`pass_marker=true`、`source_unchanged=true`；`simulator.log` 含 `P3_RESULT: PASS` 且无 `** Error/Fatal`；`snapshots/` 与 `golden/` 逐字节一致（mismatch=0） |
| 证据落点 | `<result-dir>/summary.json`、`simulator.log`、`vlog.log`、`snapshots/`、`simulation_identity.json` |
| 复核命令 | 见 `reference/commands.md` C7；抽查 `sha256(snapshots/rNN_x.bin) == golden/x__x.bin` |
| 注意 | `TB_MAC_IMPL_2` 是**行为级** twin 乘法器模型；不验证厂商 DSP 原语、资源或时序 |

## 3 G2 OOC 综合

| 项 | 内容 |
|---|---|
| 命令 | commands.md C8（Vivado 2021.1） |
| 通过阈值 | 日志出现 `P2_OOC_SYNTH_DONE=true`、无 `ERROR`；产出 `utilization.rpt`、`timing_summary.rpt`、`drc.rpt`、`npu_top_ooc.dcp`；`P2_OOC_DSP48_CELLS`>0 与设计一致 |
| 证据落点 | `<ooc-out-dir>/utilization.rpt`、`timing_summary.rpt`、`drc.rpt` |
| 已知阻塞 | 脚本硬编码 DSP XCI 路径不在包内（troubleshooting T8）；Vivado 不在 PATH |

## 4 G3 实现/布线

| 项 | 内容 |
|---|---|
| 命令 | Vivado 平台工程（包内无可用工程；BD 配置用 `build/cnn/scripts/configure_e1p3_bd.tcl`） |
| 通过阈值 | setup WNS≥0、TNS=0、0 failing endpoints；hold WHS≥0、THS=0；pulse WNS≥0；0 unrouted、0 route error；DRC 0 Error；`all_constraints_met=true` |
| 证据落点 | 平台 `timing_summary.rpt`、`route_status.rpt`、`drc.rpt`；汇总进 `MANIFEST.json` |
| 口径警告 | `+0.039 ns` 属于另一份历史包（A_rev6 合格实现，589,345 端点），与 r4 的 `+0.075 ns` 不是同一运行；引用时必须带出处，禁止合并成"约 +0.05" |

## 5 G4 IP 打包

| 项 | 内容 |
|---|---|
| 命令 | commands.md C2 去掉 `--no-pack-ip`（默认打包）或 `--ip-version/--vivado` 指定；底层为 `src/cnn/tools/npu_gen/pack_npu_ip.tcl` |
| 通过阈值 | `<out-dir>/ip_repo/<ip_name>_<ver>/component.xml` 存在；`<ip_name>_<ver>.pack_manifest.json` 的 `counts` 与 `files[]` 自洽；`source_library verify --generated <out-dir>/generated --package <ip_dir>` 为 pass（`package_files>0`） |
| 证据落点 | `ip_repo/`、`.pack_manifest.json`、`logs/pack_npu_ip.log` |
| 不能证明 | Block Design、bitstream、XSA、时序 |

## 6 G5 平台集成

| 项 | 内容 |
|---|---|
| 命令 | Vivado BD：`configure_e1p3_bd.tcl -tclargs <root> <cell> <max_cin> <max_cout> <max_ch> <max_w> <num_layers> <instr_words> <instr_addr_w> <instr_init> <mac_impl> <tn> <tm> [dw_pc dw_tap_parallel dw_rq_lanes]` |
| 通过阈值 | BD 可打开/验证；IP 参数与 `rtl_manifest.capacities` 一致；bit 与 HWH 出自同一次实现（哈希配对） |
| 证据落点 | BD 工程、`platform_top.hwh`、`platform_top.bit`、实现报告 |

## 7 G6 板端

| 项 | 内容 |
|---|---|
| 命令 | commands.md C9（`board_run_a.py` heads/image 两模式） |
| 通过阈值 | 6 个 head 逐字节一致、`first_mismatch=-1`、`error_code=0`；image 模式检测结果与 golden 一致（本次记录：1 个 train，score 0.636，box=(471.2,180.3,641.1,450.7)） |
| 证据落点 | `run_out_heads/`、`run_out_image/annotated_test_image.jpg`、板端原始日志 |
| 不能证明 | VOC/mAP、FPS/带宽、长稳；RTL 回放只到 `conv1 record0`，完整 640 图回放未做 |

## 8 一条结论的写法（与 evidence-gate 对齐）

```text
[G0][pass][measured] 编译 tinycnn28_mnist（TN4/TM4/MAC_IMPL=3）
证据：out/tinycnn/compile_manifest.json（model_sha256=6307d768..., records=8, source_digest=36ca00cc...）
复算：python -B -m tools.onnx_gen.source_library verify --generated out/tinycnn/generated
时间/执行者：2026-11-02 / <who>
```

缺陷示例（禁止）：用"差不多/大体上"一类模糊词描述结论；把仿真通过等同于功能确认；把历史记录写成 pass；跨门级合并证据。
