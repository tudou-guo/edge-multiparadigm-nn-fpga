# 图像线 native OOC（同源 5 ns 综合/实现）实测报告 — 2026-10-31

状态：**P3 = fail（Vivado 综合 RTL elaboration 失败）**；impl/routed 未测。
本批固定值：xczu15eg-ffvb1156-2-i / Vivado 2021.1 / s_clk 5.000 ns（200 MHz）/ 不降频 / 不加时序例外 / 不生成 bitstream / 不上板 / **未修改任何产品 RTL**。

> 口径声明：本次是**首次对参考式 SPS + native 整顶做 Vivado 综合的实测**。
> 失败发生在 `synth_design` 的 **RTL elaboration**，不是资源不足、不是时序问题、不是仿真功能失败。
> 因此本报告**不报** LUT/FF/BRAM/DSP/WNS/WHS/TNS/THS——这些全部**未测**。
> 「synth 失败 ≠ 设计不可行」：只代表当前冻结 RTL 里的一个可修综合限制。

## 1. 门状态总览

| 项 | 状态 | 依据 |
|---|---|---|
| **P3 整顶 OOC synth（vis_native_probe = SPS ref + block）** | **fail** | 17:35:12 起跑，17:39:25 退出；`Synth 8-91 ambiguous clock in event control` @ `spiking_sps_psum.sv:115` → 6 errors，RTL Elaboration failed |
| P3 整顶 OOC impl（place/phys_opt/route, 5 ns routed） | **not_run** | synth 未通过，未进入 opt/place/route；**routed 数字不存在** |
| 资源（LUT/FF/LUTRAM/BRAM36/18/DSP）与层级占用 | **未测** | synth 未产出 `util_synth.rpt`（未到 report 阶段） |
| 时序（setup/hold/pulse width） | **未测** | 无 routed DCP，无 `timing_impl.rpt` |
| SPS-only OOC（降级项 `spiking_sps_ref_top`） | **not_run（同一根因）** | 它例化 `spiking_sps_psum`，同样失败 |
| block-only OOC（补充数据点，非本门要求） | 运行中→见 §5 | `blk_native_probe_m`，独立于 SPS 缺陷 |
| 平台/bit/HWH/XSA/板测/B 轮 | not_run | 属其它门，本批未做 |

## 2. 失败根因（实测，可复现）

### 2.1 完整错误行（原样摘自 `vivado.log`）

```text
ERROR: [Synth 8-91] ambiguous clock in event control [D:/dl_snn/rtl/ops/spiking/sps/spiking_sps_psum.sv:115]
ERROR: [Synth 8-6156] failed synthesizing module 'spiking_sps_psum' [D:/dl_snn/rtl/ops/spiking/sps/spiking_sps_psum.sv:12]
ERROR: [Synth 8-6156] failed synthesizing module 'spiking_sps_ref_top' [D:/dl_snn/rtl/ops/spiking/sps/spiking_sps_ref_top.sv:10]
ERROR: [Synth 8-6156] failed synthesizing module 'spikformer_vis_native_top' [D:/dl_snn/rtl/ops/spiking/control/spikformer_vis_native_top.sv:12]
ERROR: [Synth 8-6156] failed synthesizing module 'vis_native_probe' [D:/dl_snn/rtl/probe/vis_native_probe.sv:45]
ERROR: [Common 17-69] Command failed: Synthesis failed - please see the console or run log file for details
RTL Elaboration failed
33 Infos, 2 Warnings, 0 Critical Warnings and 6 Errors encountered.
```

即：`spiking_sps_psum` 报错 → 级联到 ref_top → native_top → probe。**只有一条根因**，其余是级联。

### 2.2 同 run 的相关 warning（不是错误，但请 RTL owner 复核）

```text
WARNING: [Synth 8-7071] port 'o_beat_last' of module 'spiking_rgb_encoder' is unconnected for instance 'u_enc' [spiking_sps_ref_top.sv:34]
WARNING: [Synth 8-7023] instance 'u_enc' of module 'spiking_rgb_encoder' has 15 connections declared, but only 14 given [spiking_sps_ref_top.sv:34]
WARNING: [Synth 8-155] case statement is not full and has no default [spiking_rgb_encoder.sv:169]
```

### 2.3 代码定位（未改 RTL，仅阅读）

`rtl/ops/spiking/sps/spiking_sps_psum.sv:115` 是
`always_ff @(posedge s_clk or posedge s_rst)`；
其 124–136 行在**过程块内的 for 循环体里声明了局部 unpacked 数组**：

```systemverilog
for (int a = 0; a < LANES; a = a + 1) begin
  logic signed [ACC_W-1:0] oldv [T];   // 过程块内局部 unpacked array
  logic signed [ACC_W-1:0] newv [T];
  for (int t = 0; t < T; t = t + 1) begin ... end
end
```

推断（未逐条证实，仅供下一轮）：Vivado 2021.1 对 `always_ff` 内 for-body 局部 unpacked 数组支持不完整，报成 “ambiguous clock”。ModelSim 仿真可接受，所以 P1 功能门 PASS 无法暴露该问题。

## 3. run 命令与 heavy lock 证据（实测）

### 3.1 命令（实际执行，整顶）

```bash
E:/miniconda/python.exe -B D:/dl_snn/scripts/spikformer/run_heavy.py \
  --lock D:/dl_v0.1/build/onnx_e1/v02_real_yolo_20261026_2022_b1/jobs/heavy.lock.json \
  --job-id img-ooc-native-all-20261031_1735 --category vivado_ooc --mem-mib 4000 --wait-sec 300 -- \
  E:/Vitis/Vivado/2021.1/bin/vivado.bat -mode batch -notrace \
  -log OUT/vivado.log -journal OUT/vivado.jou \
  -source D:/dl_snn/scripts/spikformer/synth_ooc_vis_native.tcl \
  -tclargs OUT all vis_native_probe
```

OUT = `D:/dl_snn/build/spikformer/vision_ref_20261031/ooc_native_20261031_1657`。
同一脚本的标准锁入口（任务卡原文，`mem=5000`）：`run_locked.py <job-id> -- <vivado …>`；本次因主机 free 长期 2.3–4.9 GiB 波动，按 Lead 授权用同一把共享锁 + 同一 lease 机制的 `run_heavy.py --mem-mib 4000`（历史 probe 同值），不绕锁。

### 3.2 锁 lease 证据（`jobs/heavy_events.jsonl`）

```text
{"event":"acquired","job_id":"img-ooc-native-all-20261031_1735","category":"vivado_ooc","lease_id":"aff6fc90a97345ff803f96cd1578b4dc","mem_required_mib":4000,"probe_blockers":[],"timestamp_local":"2026-10-31T17:35:09+0800"}
{"event":"released","job_id":"img-ooc-native-all-20261031_1735","lease_id":"aff6fc90a97345ff803f96cd1578b4dc","duration_sec":258.7,"timestamp_local":"2026-10-31T17:39:27+0800"}
```

- 未占用、未杀任何外部进程。阻塞期（16:51–17:33）`heavy_lock` 判定 `external_vivado`（用户 GUI PID 504）+ 间歇 `external_simulator` + `insufficient_free_memory`，用户 17:30 关闭 GUI 后才放行（17:35:09 拿到 lease）。

### 3.3 主机资源采样（实测，30 s 间隔）

- `OUT/evidence/memwatch_early.log`、`memwatch_latest.log`：free physical 2278–6001 MiB 波动；Vivado 进程峰值 **3989 MiB**（17:37:27 采样），随后 synth 失败退出，free 回到 6001 MiB。
- 历史同块几何参照（实测，非本次）：NLP 48h OOC 峰值 RSS 3746 MiB / 4216 MiB。

## 4. synth 阶段实测细节

| 项 | 值 |
|---|---|
| 起跑（`OOC_RUN_BEGIN`） | 2026-10-31T17:35:12+0800 |
| `SYNTH_BEGIN` | 17:35:15 |
| 源文件 | 19 个（`OOC_SOURCES 19 files`，product sps/attention/lif/mlp + block_top + native_top + 3 probe） |
| 约束 | 生成物 `constraints_ooc.xdc`：`create_clock -period 5.000 [get_ports s_clk]` + `set_input_delay/output_delay 0.000`；无例外、无降频 |
| 失败时刻 | 17:39:25（wall ≈ 4 min 13 s，其中 lease 258.7 s） |
| 结论 | **FAIL（RTL Elaboration failed，6 errors）** |
| 产出 | 无 `util_synth.rpt` / 无 `timing_synth.rpt` / 无 `post_synth.dcp`（未到该阶段） |
| lip license | 正常获得（`Got license for feature 'Synthesis'`）→ 不是 license 问题 |

## 5. impl 阶段

**not_run**。synth 失败，未执行 `opt_design/place_design/phys_opt_design/route_design`，因此：
- 无 routed DCP，无 setup/hold/pulse-width 报告；
- **不存在任何 routed WNS/WHS/TNS/THS/未布线数字**；
- synth timing 都没有，更不得把任何估算当 routed timing。

### 5.1 block-only 补充数据点（非本门要求，独立于 SPS 缺陷）

`blk_native_probe_m`（`spiking_block_top` native 参数 E384/H12/HD32/MLP1536/T4/TOKENS64/TL8/TC8/RW4 + 输入边界 8×LUT1 补偿）单独 OOC 运行，结果见：
- 目录 `OUT/blk_only_1742/`（`vivado.log`、`facts_synth.json`、`facts_impl.json`、reports/*.rpt、DCP）
- 本文 §9 若已填数字，则为**实测 block-only**；它**不能**代表整顶（SPS 部分完全未测）。

## 6. 实测 vs 推导（严格区分）

- **实测（本报告）**：§2 错误行与根因、§3 命令/lease/资源采样、§4 起止时刻与失败结论、§5.1 block-only（若填）。
- **推导（不作门、不写成功）**：SPS 双 bank 128×8192b ≈ 64 BRAM36；block 六矩阵 1,769,856 B 容量下限 ≈ 384 BRAM36；psum 256×84b；conv2 唯一权重 331,776 B 下限 72 BRAM36；器件量级 LUT 341,280 / FF 682,560 / BRAM36 744 / DSP48 1248。
- **未测**：全部 util/时序/DRC；RTL cycles；平台 BD/HWH/XSA；板端 FPS。

## 7. 下一轮排查建议（不动数值/握手语义，仅消除综合限制）

1. 先做**单模块回归**：`synth_design -top spiking_sps_psum -mode out_of_context`（参数 LANES=8/T=4/ACC_W=21/C2=384/CMAX=384/DEPTH=256），确认根因只在 psum，跑通再往上一层。
2. `spiking_sps_psum.sv:115` 的 `always_ff`：
   - 把 for 循环体内的 `logic signed [ACC_W-1:0] oldv [T]; / newv [T];` **提到过程块外**（模块级声明），或改成 packed 向量 `logic signed [T*ACC_W-1:0]`；
   - 给 `case (pst)` 补 `default:`（同时消 `Synth 8-155` 一类 warning）。
3. 顺手确认 `spiking_sps_ref_top.sv:34` `u_enc.o_beat_last` 未连接是否故意（15 declared / 14 given）。
4. 修好后按 `scripts/spikformer/synth_ooc_vis_native.tcl` 原样跑：先 `-tclargs OUT synth vis_native_probe`（约 15–20 min 出 util/timing/synth-DCP），通过再 `-tclargs OUT impl vis_native_probe`（约 20–35 min 出 routed）。
5. 注意：**synth 阶段通过不等于 routed 5 ns 通过**；本次连资源数字都没有，不得引用任何推导值当门结论。

## 8. 交付索引

```text
OUT = D:/dl_snn/build/spikformer/vision_ref_20261031/ooc_native_20261031_1657
OUT/vivado.log                                  # 失败 run 完整日志（167 行）
OUT/evidence/vivado_whole_top_synth_fail_1739.log   # 同上，冻结证据副本
OUT/evidence/vivado_fail_include_option_1734.log    # 更早一次尝试：read_verilog include 选项在 non-project 模式不存在
OUT/evidence/heavy_wait_latest.json             # 阻塞期 heavy_lock 判定（external_vivado 等）
OUT/evidence/memwatch_latest.log                # free/Vivado RSS 采样
OUT/commands.txt                                # 实际命令与降级命令
OUT/identity/rtl_sha256_at_synth.txt            # 冻结源 sha256（与 img-p1 清单逐条一致）
OUT/facts.json                                  # 机器可读 facts（status=fail）
OUT/constraints_ooc.xdc                         # 生成的 OOC 约束
OUT/blk_only_1742/                              # block-only 补充 run
scripts/spikformer/synth_ooc_vis_native.tcl     # 新增：synth+impl 全流程
rtl/probe/vis_native_probe.sv                   # 新增：整顶 probe
rtl/probe/sps_ref_probe.sv                      # 新增：SPS-only 降级 probe
rtl/probe/blk_native_probe_m.sv                 # 新增：block-only probe（含边界补偿）
```

## 9. block-only 补充实测（若已跑出；否则 not_run）

（本节在 run 结束后填；只放实测数字，且标注 block-only ≠ 整顶。）

