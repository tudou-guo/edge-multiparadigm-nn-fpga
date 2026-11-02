# snn-translate 故障库（现象 → 根因 → 处置 → 证据）

> 来源：`D:\DL_FINAL\_skill_work\history\snn.md`（codex 会话提炼）+ `[包内]` docs/证据 + 2026-11-02 本机复跑。
> 每条给出可核对证据；本机复跑的条目在证据列注明"2026-11-02 实测"。找不到证据的条目会明写"未找到证据"，不臆造报错文本。

## T1 setup 时序不收敛（5 ns / 200 MHz）

- **现象**：OOC setup 失败；改为五级流水 MAC 后 setup `WNS +0.405 ns / TNS 0 / 0 failing（187,376 端点）`。
- **根因**：三层 MAC 数据通路在 5 ns 内组合路径过长。
- **处置**：RTL 应用"五级 MAC 数据通路"修复（不改协议/不改数值语义），重新 freeze 后重跑。
- **证据**：[工作区] `D:\dl_snn\docs\SPIKFORMER_NLP_48H_TIMING_VIOLATION.md` §0~§1、`SPIKFORMER_NLP_48H_TIMING_FIX_PLAN.md` §4~§5；冻结记录 `sim/snn/evidence/nlp/rtl_freeze.json`（reason 段）。

## T2 hold 违例：失败路径全部起于输入端口（-0.080 ns / 32,852 endpoints）

- **现象**：setup 已过，hold `WHS -0.080 ns、THS -2058.904 ns、32,852 failing（187,376 端点）`；路径 `i_w_data[*] → u_top/thr_*_reg[*]/D`，数据路径逻辑级数 0、route 0.012 ns。
- **根因**：OOC 0 输入延迟模型下，输入端口到首级寄存器 hold 余量不足；不是内部逻辑问题，加时序例外/降频无效。
- **处置**：生成的目标适配器加 **1,192 个 DONT_TOUCH LUT1 输入 hold-margin 单元**（8 级 × 149 sampled bits，含 `s_rst`），**XDC 与失败版本逐字节相同**（未放宽约束、未加例外）；结果 5 ns routed OOC `WNS +0.135 / WHS +0.023 / 0 failing / unrouted 0`。16 级 probe 牺牲 setup（synth WNS -0.191）被拒；10 级 routed 过门但余量仅 +0.039/+0.001，被 8 级取代。
- **证据**：`src/snn/tools/backends/spikformer/codegen.py` `_adapter`/`spk_input_hold_margin`；`sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json`；`hold_closure_20261030.json`；XDC sha256 `9fc1c2676a29bbab1ee6277b9bb60a882c5e868396ab03ee36aa1f8a2eb10842`。

## T3 包内执行 `toolchain build --stage synth|ooc` 找不到物理驱动

- **现象**（2026-11-02 实测）：`E:\miniconda\python.exe: can't open file 'D:\FINAL\src\snn\scripts\spikformer\run_physical.py': [Errno 2] No such file or directory`；先打印了 `physical driver: ...` 行。
- **根因**：`toolchain.py` 的 `ROOT` = `src/snn`（按文件位置 `parents[1]`），`_spik_build` 以 `ROOT` 为 cwd 调 `scripts/spikformer/run_physical.py`；提交包把脚本放在 `build/snn/scripts/`，根被拆开。
- **处置**：物理门在 [工作区] 完整快照根 `D:/DL_FINAL/SNN` 执行；包内只做"源码冻结检查 + 排队"：`SNN_ROOT=<包根>/src/snn python -B build/snn/scripts/spikformer/run_physical.py --manifest ... --stage ooc --lock <tmp>/heavy.lock`（实测首行 `source frozen check: 28 files match manifest hashes`，exit 0）。`--execute` 需 Vivado + 共享 heavy lock。
- **证据**：本机 2026-11-02 终端输出；`src/snn/tools/toolchain.py` 第 20~40 行；`build/snn/scripts/spikformer/run_physical.py` 的 `ROOT = parents[2]`。

## T4 test_short.py 在包内报 `No module named 'reference'`

- **现象**（实测）：`Traceback ... import reference as R / ModuleNotFoundError: No module named 'reference'`。
- **根因**：脚本按 `parents[3]` 推根（快照布局为 `SNN/verification/sim/spikformer/` → 根=`SNN`）；包内路径是 `sim/snn/verification/sim/spikformer/`，推根落到 `sim/snn`，没有 `tools/backends/spikformer`。
- **处置**：加环境变量 `SNN_ROOT=<包根>/src/snn` 再跑，实测 15 项 PASS、`FAILURES: 0`。
- **证据**：`sim/snn/verification/sim/spikformer/test_short.py` 第 12~17 行；本机 2026-11-02 实测输出。

## T5 test_integer_ref.py 在包内报 `No module named 'runtime'`

- **现象**（实测）：`ModuleNotFoundError: No module named 'runtime'`（`import runtime.spikformer.head as RHEAD`）。
- **根因**：包内未收录 `runtime/spikformer/`（`head.py/run.py/tokenize.py/packing.py` 只在 [工作区] `D:\DL_FINAL\SNN\runtime\spikformer\`）；该测试同时需要 torch。
- **处置**：到 [工作区] 快照根执行 `python -B tools/spikformer_train/test_integer_ref.py --out <tmp>/math_tests_m1.json`，实测 `RESULT: 8/8 PASS -> PASS`、exit 0。包内只能引用 `sim/snn/evidence/nlp/math_tests_m1.json` 证据（记 `not_run`/证据级）。
- **证据**：本机 2026-11-02 实测；`sim/snn/evidence/nlp/math_tests_m1.json`（`PASS:true, n_pass:8`）。

## T6 check_contract.py 缺 SNN_ROOT 时导入失败

- **现象**：脚本 `ROOT = parents[2]`（包内=`build/snn`），随后 `sys.path.insert(0, ROOT)` 再 `from tools.backends.spikformer import config` → 找不到 `tools` 包。
- **根因**：包内工具链源码在 `src/snn/tools`，不在 `build/snn`。
- **处置**：`SNN_ROOT=<包根>/src/snn python -B build/snn/scripts/spikformer/check_contract.py --config ... --contract ... --checkpoint ... --out ...`；实测 `{"status":"pass","fails":[],"checks":108}`、exit 0。
- **证据**：本机 2026-11-02 实测；`build/snn/scripts/spikformer/check_contract.py` 第 12~16 行。

## T7 check_ready.py 在包内 src/snn 报 1 个 FAIL

- **现象**（实测）：`config: nlp_sentiment_b1  FAIL  D:\FINAL\src\snn\examples\spikformer\config\nlp_sentiment_b1.json`，`FAIL items: 1`；其余项多 `not_run`。
- **根因**：脚本把 `examples/` 写在工具链根下；包内配置在 `data/snn/examples/`。另 `checkpoint/compile/OOC/bit/board` 依赖主线 build 产物，快照未带，`not_run` 属正常。
- **处置**：到 [工作区] 快照根执行（实测 `FAIL items: 0`）；包内结果不作为门禁证据。
- **证据**：本机 2026-11-02 两处实测输出；`src/snn/tools/spikformer_train/check_ready.py`。

## T8 浮点/未量化 checkpoint 被拒

- **现象**（实测）：把 `block0.q_w` 改成 float32 后编译，报
  `TypeError: key 'block0.q_w' must be int8, got float32 (no silent casts)`，不落 manifest。
- **根因**：`weights._get` 严格校验 dtype/shape（`np.asarray` 后直接比较 dtype），无静默 cast。
- **处置**：用 `tools/spikformer_train/export_integer.py` 从训练 state 重新导出全整数 npz；或确认 checkpoint 来自 `export_abi`。
- **证据**：本机 2026-11-02 实测；`src/snn/tools/backends/spikformer/weights.py` `_get`（第 125~135 行）。
  注：历史提炼中"浮点 checkpoint 被拒"的**专用报错文本**未找到；上面是实测得到的唯一 dtype 报错，不要写成"检测到浮点 checkpoint"。

## T9 配置越界/缺字段被结构化拒绝

- **现象**（实测原文）：`ValueError: config: time_steps not in legal [1,2,4]` / `config: head_dim: only 32 implemented (PE16 lane pair)` /
  `config: only B=1 implemented` / `config: mlp_hidden must be multiple of 16` / `config: seq_len must be int in [1,256]` /
  `config: embed != heads*head_dim` / `config: missing required field 'time_steps'`。
- **根因**：`config.resolve` 逐条硬校验，无静默回退。
- **处置**：按报错改配置；不要改源码绕过。
- **证据**：本机 2026-11-02 实测；`src/snn/tools/backends/spikformer/config.py` 第 33~95 行。

## T10 换路径后 manifest_sha256 变化，被误判为"产物不一致"

- **现象**：同一 checkpoint 用绝对路径编译得 `manifest_sha256=a2a655d4…`，与包内冻结清单 `f727e4c7…` 不同。
- **根因**：`manifest_sha256` 对含 `assets.checkpoint` 路径字符串的排序 JSON 求哈希；`cache_key` 同理。params 的 sha256 集合不变。
- **处置**：用与冻结 run 相同的相对路径字符串（`cd data/snn` + `--weights examples/nlp_sentiment_b1/ckpt_real_real50.npz` + `PYTHONPATH`）复跑，实测得到 `f727e4c7…` 且 manifest 文件与包内 `b71e3f17…` 逐字节相同；跨机比对用 params 哈希集合/文件 sha，不要只比 `manifest_sha256`。
- **证据**：本机 2026-11-02 两次编译实测；`src/snn/tools/backends/spikformer/manifest.py` 末尾哈希段。

## T11 r7 bit 的 SUMS 读窗口 +64 字偏移

- **现象**：板端读 `SUMS+4*i` 得到 `o_sums[(i+64)%128]`，不旋转会 64 通道错位、与 golden 比对失败；`demo_nlp.py` 用 `ROT=64` 才正确。
- **根因**：r7-sumsfix 只把 SUMS 窗口从 32 字扩到 128 字，解码 `sums_flat[{araddr_q[8:2],5'b0} +: 32]` 的基址 `0x100>>2=64` 未减去；RTL 未修偏移。
- **处置**：软件读法 `sums[j]=rd(SUMS+4*((j+64)%128))`；或按 handoff §5.1 改 RTL 重出 bit 后 `ROT=0`。未改 bit 前不要直接 `SUMS+4*j`。
- **证据**：`src/snn/docs/DEMO_NLP.md` §3；`src/snn/docs/PLATFORM_AND_STATUS.md` §3"SUMS 读窗口"。

## T12 连续第二个样本超时（软复位不排空在途 DMA）

- **现象**：同一 overlay 连续跑第二个样本超时；现靠每样本重载 overlay 绕过。
- **根因**：`CTRL_SOFT_RESET` 只 flush 两侧 FIFO + 清状态，不排空在途 DMA 事务，也不等写 DMA 的 B 响应。
- **处置**：演示脚本每样本 `Overlay(bit)` 重载；根治需改平台软复位序列（未实现）。
- **证据**：`src/snn/docs/PLATFORM_AND_STATUS.md` §2"已知平台限制"第 2 条；`CDC_FIX_R6_RESULT_20261030.md`"未完成"第 2 条。

## T13 冻结 RTL 被改动后仍引用旧证据

- **现象**：图像线并行改了两个 SPS 文件；若继续引用 2026-10-30 的 replay/OOC 结论就会张冠李戴。
- **根因**：`rtl_freeze.json` 规则："any post-freeze RTL edit invalidates synth/replay/OOC evidence and requires re-freeze"；快照 28 源 vs 冻结 18 文件为 **16 SAME / 2 DIFF**。
- **处置**：NLP 证据只引用冻结哈希链（`ip/src` 为冻结态）；口径写"NLP 功能 RTL 16/16 未变（NLP top 不实例化 SPS）"，**不得写"整棵 RTL 逐字节一致"**；改动后必须重新 freeze + 重跑下游门。
- **证据**：`sim/snn/evidence/nlp/rtl_freeze.json`、`sim/snn/evidence/nlp/rtl_hash_check_20261031.txt`、`src/snn/docs/PLATFORM_AND_STATUS.md` §5。

## T14 5 ns 约束与 300 MHz DMA 域：CDC 与时钟组

- **现象**：平台有 200 MHz SNN 域（`snn_clk`，5 ns）与 300 MHz DMA 域（`ui_clk/npu_clk`）；OOC 生成的 XDC 只约束 `s_clk 5.000` + 0 I/O delay，**不含跨域时钟组**；平台 CDC 报告 1 条 Critical（CDC-10，DDR4/复位基础设施）+ 4 条 CDC-15 warning。
- **根因**：OOC 与平台是两套约束；跨域同步靠 XPM/事件 FIFO（r6 已改），但 CDC 未重新收敛，`sys_rst` 无 input delay、`DDR4 reset_n` 无 output delay。
- **处置**：平台集成时必须补 CDC clock-group XDC（`platform_build_20261030.json` 的 next 第 1 条）；引用时序时区分"应用逻辑时序 pass"与"CDC 告警未清零"；不得把 300 MHz 域约束套到 SNN 5 ns 证据上。
- **证据**：`build/snn/reports/platform_build_20261030.json`（`clocks`、`next`）；`build/snn/reports/CDC_FIX_R6_RESULT_20261030.md`（r6 跨域 synth WNS +0.776→+4.709；WNS +0.245/WHS +0.010）。

## T15 e2e 回放的 noforce 与 bpforce 两档不能互相替代

- **现象**：`tb_nlp_e2e` 的 force/release 反压模式曾死锁在 `in_cnt=199`（TB 侧），而不加 force 的 `+NLP_NO_FORCE` 正常。
- **根因**：TB 侧 force/release 目标与 `always_ff` 更新冲突（记录在 `rtl_freeze.json.open_tb_items`）；不是 RTL 数据错误。
- **处置**：两档都保留：noforce 证明数据正确（cycles 7,084,772），bpforce 证明反压握手（cycles 7,084,812，14 输出/10 输入 stall 窗口，mismatch=0）；TB 修复后才可替换。block 级反压另有覆盖。
- **证据**：`sim/snn/evidence/nlp/replay_e2e_{noforce,bpforce}_holdfix.json`；`sim/snn/evidence/nlp/rtl_freeze.json` `open_tb_items`。

## T16 门状态文件时间早于后续实现

- **现象**：`sim/snn/evidence/nlp/final_gates.json` 里 `platform/bit_hwh/board = not_run`，但之后已有 r6/r7 实现与 bit/HWH。
- **根因**：该文件 2026-10-30 13:10 冻结，早于 r6（20:00）与 r7（23:10）实现。
- **处置**：引用时并列最新实现证据（`platform_timing_summary_20261031.txt`、`platform_build_20261030.json`、`board/snn/MANIFEST.sha256`），并注明冻结时间；不得用 final_gates 否定后续结果，也不得把后续结果塞回 final_gates。
- **证据**：`sim/snn/evidence/nlp/final_gates.json`（`started_at/deadline_at`）；`src/snn/docs/PLATFORM_AND_STATUS.md` §1 平台行注。

## T17 图像线 AV 位宽与 NLP 混用 / 陈旧合同字段

- **现象**：图像 native `numeric_contract.json` 的 `av` 仍写 "uint11 max 1024"，但该几何是 64 token，上界应为 2048（12 bit）。
- **根因**：合同生成文案沿用 NLP 32-token 口径，未随几何更新；图像 resolved_config 已正确。
- **处置**：以 `resolved_config.derived.av_bits/av_max` 为准（NLP 11/1024；图像 12/2048）；本技能只验收 NLP，不把图像数字混入。
- **证据**：`src/snn/docs/NUMERIC_CONTRACT.md` §4；`sim/snn/evidence/vision/toolchain/{numeric_contract.json,resolved_config.json}`。

## T18 把记录级板测写成 pass（证据事故）

- **现象**：板端 3 样本 `match=True`（cycles≈7,084,772）是主会话记录，原始 run JSON/控制台日志未随包，本机缺 `pynq` 失败、板卡离线。
- **根因**：把"文档记录"当"本次实测"；`board_run_local_attempt.log` 只是失败尝试，不是板测证据。
- **处置**：标 `record_level` + `not_verified`，不写 `pass`；现场复跑前先按 T11 补 `ROT=64`。
- **证据**：`board/snn/README_BOARD.md` §3；`sim/snn/evidence/platform/board/board_run_local_attempt.log`；`src/snn/docs/PLATFORM_AND_STATUS.md` §3。

## T19 "参数 blob 数不符"没有专用报错

- **事实**：blob 数由模式派生——fixture `4+12×depth`（depth2→28）、真实权重 `6+24×depth`（depth2→54），代码里没有"数量必须=N"的断言；
  历史提炼也未找到该报错文本。
- **处置**：核对 blob 数用 manifest `len(params)` 与命名空间（§artifacts.md §2），不要臆造"blob 数不符"报错；数量异常时先查 `model_kind` 与 config `depth`。
- **证据**：`src/snn/tools/backends/spikformer/weights.py` `_shapes`/`import_checkpoint`；本机 2026-11-02 两种模式实测（28/54）。
