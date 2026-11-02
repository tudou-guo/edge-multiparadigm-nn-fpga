# 图像线 native 综合资源问题交接（2026-10-31 14:10）

> 本文作为历史快照保留。最新执行任务书：
> [VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md](VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md)。
> 本文将数组 bit 数直接视作 FF、将 conv2 门控加权项视作 DSP、将 ROM 容量视作每图流量，
> 以及部分 BRAM/bias 拍数估算不准确；新任务书第 3 节已逐项纠正，以新任务书为准。
> 不要把本文“846k FF / 2304 DSP”当作已完成 native 综合的实测数字。

> 用途：把“native 几何综合资源爆表”这件事交给另一个对话解决。本文自包含，
> 读完可直接开工。行为级验证已经全绿，本文件只谈资源/微架构。

## 0. 一句话结论

行为级（回放小几何）已闭环；**native 几何（32×32×3 → E384/H12/HD32/MLP1536/T4/64 token）
的 `spiking_sps_top` 按当前 RTL 直接综合不可行**：寄存器数组约 **846k FF > 器件 682,560 FF**，
且 conv2 每拍 **2304 次乘加**超器件 DSP（~1248），5ns 组合链也过不去。
block 方向可行（Vivado 已把它的大 RAM 推成 BRAM cascade），但完整 util/timing 因暂停未跑完。
必须改 SPS 微架构（BRAM 化 + 同步读流水 + MAC 时间复用），不是调参数能解决。

## 1. 当前验证状态（已完成，无需重做）

| 测试 | 结果 |
|---|---|
| `tb_block_replay`（默认随机 token） | **ALL PASS（16 beats）** |
| `tb_block_replay`（SPS token） | **ALL PASS（16 beats）** |
| `tb_vis_sps` | **ALL PASS（16 beats）** |
| `tb_vis_replay`（SPS→block→sums） | TOKEN CHECK fails=0；32 sums 全 match；**ALL PASS** |

- 本轮之前已修并保留的 bug（回放几何）：
  - `spiking_sps_top.sv` S_LDT 只在 `i_weight_valid && o_weight_ready` 时消费（否则 ROM 指针每 tile 少 1）。
  - `spikformer_vis_top.sv`：`W2B` 改为含全部 token 位置（`NTOK*((C2*C1*9+7)/8)`）；`st` 4bit；`blk_wdata` 驱动；`capi` 位宽 `$clog2(NTOK*G)`；S_START 清 acc；S_RUN 只在 valid&&ready 推进权重指针。
  - 两个 TB 的主机顺序/token 比较格式修正。
- block golden 修复：`scripts/spikformer/gen_block_replay.py` 里 proj/fc2 后都应先 LIF 再把 0/1 spike 加回残差（2026-10-29 block 合约）；RTL `spiking_block_top.sv` 未改（sha256 `841df6017e51af858d3b128263b0478fe45cc93093863fbc7423f648759f9d19`）。

日志：`sim_work/vis_sps/vsim_{block_r,sps_r,replay_r}.log`。

## 2. native 目标几何与工具链产物

配置：`H=W=32, C0=3, C1=96, C2=384, T=4, TOKENS=64`；
block：`E=384, HEADS=12, HEAD_DIM=32, HIDDEN=1536, DEPTH=1, TL=8, TC=8, RW=4`。

器件：`xczu15eg-ffvb1156-2-i`（Vivado 2021.1，s_clk 5.000ns / 200MHz，不降频、不加例外）。
器件量级：**LUT 341,280 / FF 682,560 / BRAM36 744 / DSP48 1248**（写方案时以此为准，可用
`report_utilization` 复核）。

工具链产物（已生成，可直接用）：
`build/spikformer/vis_native/`
- `generated/spikformer_config.vh`（`SPIK_SPS_*`、`SPIK_*` 宏）
- `params/*.bin` / `*.coe`（block0 权重、bias）
- `resolved_config.json`（含 streaming/geometry/derived 数据）
- `schedule.json`（权重 DDR base/bytes、memory_map）

关键流长度（来自 resolved_config.derived）：
- `conv1_beats=324`，`conv2_beats=41,472`（每 token 位置一次），`bias_beats=30`
- `block_w_beats=222,912`（=1,783,296 B = 1.78MB），`block_in_beats=3,072`
- `sps.conv2_weights=331,776` B（整个 tile set 一遍）

## 3. 综合尝试与证据

### 3.1 native SPS OOC 综合：失败（结构性问题）

- Probe：`rtl/probe/sps_native_probe.sv`（把 `spiking_sps_top` 例化在 native 参数下，
  所有流输入引到顶层端口，避免被常量折叠）。
- 脚本：`scripts/synth_ooc_probe.tcl`（read_verilog + `synth_design -top … -mode out_of_context`
  + `create_clock -period 5.000 [get_ports s_clk]`）。
- 实测：跑了 **>25 分钟未收敛**，进程峰值 **7.1GB**，无 `util.rpt`/`timing.rpt` 产出 → 中止。
- 静态原因（这是主要依据，比跑挂更硬）：
  - `f1`：`logic [C1*T-1:0] f1 [H][W_IN]` = 1024 × 384 bit = **393,216 bit**；
    在 `S_P2` 里被**组合逻辑异步读**（`f1[2*wy+fy][2*wx+fx][ch*T+tt]`）→ 推不成 BRAM36，
    只能 FF/LUTRAM。
  - `prow2`：256 × 1536 bit = **393,216 bit**，drain 同样异步读 → FF。
  - 其他：`f0` 24,576 FF；`wbank1`（conv1 权重）96×3×9×8 = **20,736 FF**；
    `wtile2` 4,608 FF；bias1/bias2f ≈ 10,080 FF。
  - **合计 ≈ 846k FF ≈ 器件 124%**（还没算控制/流水/adder tree 的 FF）。
  - conv2 数据通路：`LANES*T*TC2*9 = 8*4*8*9 = **2304 MAC/拍**`；
    且每个 (lane,t) 是 `9*TC2 = 72` 项**串行加法链**（RTL 里 `part = part + …` 的 for 循环
    在单拍内综合成级联加法）→ DSP 数超 1248，组合链也远超 5ns。
- 关键代码位置（改的时候按这些行找）：
  - `rtl/ops/spiking/sps/spiking_sps_top.sv:61` `wbank1` 声明
  - `:67` `f1` 声明；`:192` `f1[...] <= lif1_spk` 写
  - `S_P2` 段（约 `:213-250`）f1 异步读 + 2304 MAC 循环
  - `prow2` 写（LIF2 后）与 drain 读（组合）
- 结论：**当前 SPS 只能跑小回放几何**（H=W=8,C1=16,C2=32,T=2,TC2=4 时数组小、仿真通过）。
  native 必须改微架构。

### 3.2 native block OOC 综合：可行方向，未跑完（暂停）

- Probe：`rtl/probe/blk_native_probe.sv`。
- 跑到 ~13 分钟时（暂停前）Vivado 日志已显示：
  `Implemented Block Ram Cascade chain of height 8 and width 36/18 for RAM "spiking_block_top:/fc1mem_reg"`
  → 大权重 RAM 的 BRAM 推断路径正常。
- 未产出 `util.rpt`/`timing.rpt`（被暂停）。
- 静态预算（可先按此评估）：
  - QKV/proj 权重 banks：4 矩阵 × TL=8 banks × `BANK_DEPTH=(E/TL)*(E/TC)=48*48=2304` × 64 bit
    = 589,824 B
  - `fc1mem`/`fc2mem`：各 `FCD=HIDDEN*E/8=73,728` beats × 64 bit = 589,824 B
  - 合计 = 1,769,856 B = 14.16 Mbit → **≥384 BRAM36（52% of 744）**
  - 数据通路宽度与已验证的 NLP block 相同（TL*TC=64 MAC、5 级流水、8bit × 8bit）；
    NLP 在 5ns 平台已收敛（r6 WNS +0.051），native 预计同量级 → **5ns 有希望**，但需实测。
  - 风险点：TOKENS=64（NLP 32）、HEADS=12（NLP 4）、`att` PE 128bit；计数器/多路选择会变大。

## 4. 平台/数据流事实（改造与平台设计都要用）

- **repack**：SPS `o_tok_data` 是 32 bit/拍（8 lane × T × 1bit spike）；
  block `i_in_data` 是 128 bit/拍（8 ch × T × RW4）。中间要一个展宽级（1bit→RW nibble），
  共 3072 拍。
- block 输出 3072 拍 × 128bit；最终 sums = 384 ch × 32bit = 1536 B。
- conv2 tile set：每位置 41,472 拍（331,776 B），64 个位置 = **2.65M 拍 / 21.2MB 每图**。
  平台里应把整个 tile set 缓存在 BRAM（331,776 B ≈ 92 BRAM36），按位置回放，
  否则每图多读 21MB DDR（200MHz 64bit ≈ 13ms）。
- 单图 DDR 流量（conv2 缓存后）≈ block 权重 1.78MB + conv1 3.2KB + pixels 3KB + sums 1.5KB
  ≈ **1.8MB**（64bit@200MHz 理论 1.6GB/s → ~1.1ms）。
- 计算量粗估：block 每图 ~20–35M 周期（NLP 已验证 3.39M 的 6–10×），即 **100–175ms/图**，
  仅“功能可用”级别，不能报性能。

## 4.1 重要提醒：回放几何 vs native 内存路由

- `spikformer_vis_top` 的 `w2rom`（`NTOK*((C2*C1*9+7)/8)`）在 native 是 2.65M × 8B = 21MB，
  **不能**片上实现；native 必须走“DDR→（可选 BRAM 缓存）→SPS 权重流”的平台路径。
- 因此 native 平台 top（`spikformer_vis_platform_top`）需要：
  1) SPS 权重/像素流：conv1 324 + bias（板级按 SPS 的 BIB 实际拍数，注意 native
     `BIB=max(C1,C2)=384`，与 config 里 `bias_beats=30` 字段不一致，以 RTL 为准）+
     pixels 1024（24bit/拍）+ conv2 tile set 41,472（BRAM 缓存后按位置回放 64×）；
  2) block 权重流 222,912 拍（1.78MB）；
  3) repack（32→128 bit）；
  4) sums 收集（384ch 累加）→ 写 DMA；
  5) CSR/descriptor 与 NLP 平台一致（4 descriptor beats + thr q/k/v/proj/fc1/fc2）。

## 5. 给“资源问题”对话的建议路线

**选项 A（最小改动，推荐先做）**
1. `f1` / `prow2` 改成 **simple dual-port BRAM**（读地址打一拍），S_P2 与 drain 各加 1 级流水；
   预计 BRAM 增加 ~22–44，FF 降到 ~60–80k。
2. conv2 MAC 时间复用：`TC2` 串行（每拍 1 个输入通道）→ 72 MAC/拍，周期 ×8；
   或 `TC2=2` → 576 MAC/拍（DSP 576 可接受），周期 ×4。
3. `wbank1` 20.7k FF 可保留或改流式装载（省 FF，但要改 conv1 时序）。
4. 改完再跑 OOC `synth_design` + `report_utilization`/`report_timing_summary` 验证。

**选项 B（进一步省资源）**
- conv1 权重流式；f1 只缓存当前行窗口（line buffer）；prow2 分块累加。
- 这样 BRAM/FF 都更小，但控制复杂度上升。

**选项 C（换几何）**
- 若图像分类精度允许，缩小 `C1/C2`（如 C2=192/96）或串行化通道；
- 或把 SPS 输出 token 降到 32 位置（pool 更大）——会改变模型，需要重训/重评。

**顺序建议**：A 先跑通（BRAM + 读流水 + TC2 串行）→ synth 出数 → 不够再 B/C。

## 6. 复现/工具命令

行为级回归：
```bash
cd D:/dl_snn
python -B scripts/spikformer/gen_sps_replay.py
TOKSRC=build/spikformer/sps_replay/exp.hex python -B scripts/spikformer/gen_block_replay.py
python -B scripts/spikformer/gen_vis_tok.py
python -B scripts/spikformer/gen_vis_sums.py
cd sim_work/vis_sps
vsim -c -do compile_vision.do -l compile_vision.log
vsim -c -do compile_block.do -l compile_block.log
vsim -c -work work tb_block_replay -l vsim_block.log -do "run 4000 us; quit -f"
vsim -c -work work tb_vis_sps -l vsim_sps.log -do "run 40000 us; quit -f"
vsim -c -work work tb_vis_replay -l vsim_replay.log -do "run 80000 us; quit -f"
```

OOC 综合（**一次只跑一个，SPS native 预期跑不完，注意 4GB+ 内存**）：
```bash
cd D:/dl_snn
python D:/spk_platform/scripts/run_locked_impl.py job_name \
  E:/Vitis/Vivado/2021.1/bin/vivado.bat -mode batch -notrace \
  -log D:/dl_snn/logs/probe_blk.log -journal D:/dl_snn/logs/probe_blk.jou \
  -source D:/dl_snn/scripts/synth_ooc_probe.tcl \
  -tclargs blk_native_probe D:/dl_snn/reports/native_probe_blk
# SPS: -tclargs sps_native_probe D:/dl_snn/reports/native_probe_sps
```
- 环境现状：无残留 Vivado 进程，heavy lock 已释放（free ≈ 9GB）。
- 注意：`run_locked_impl.py` 是单 heavy 租约，两个综合不要并行。

## 7. 文件清单（本文件相关）

- `rtl/ops/spiking/control/spikformer_vis_top.sv`（回放几何 top，本轮已修）
- `rtl/ops/spiking/sps/spiking_sps_top.sv`（native 需改造的重点）
- `rtl/ops/spiking/control/spiking_block_top.sv`（native 方向可行，未改）
- `rtl/probe/sps_native_probe.sv`、`rtl/probe/blk_native_probe.sv`、`rtl/probe/probe.xdc`
- `scripts/synth_ooc_probe.tcl`
- `build/spikformer/vis_native/`（工具链产物）
- `scripts/spikformer/gen_block_replay.py`（golden 已修）
- 进度与复验：`docs/VISION_LINE_PROGRESS_20261031.md`、`docs/VISION_LINE_RETURN_20261031.md`
