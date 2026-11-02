# Spikformer SNN 交接文档（2026-10-31）

> 最新资源修复执行入口：
> [VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md](VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md)。
> 该任务书按用户选定的参考工程路线给出实施步骤，覆盖本文 §11 及旧资源交接中的不准确估算。
> 可直接转交的开场指令：`NEW_CHAT_OPENING_VISION_REFERENCE_ROUTE_20261031.txt`。
> 新架构当前为 plan_only；下文保留历次行为验证与平台状态，不代表新 native 已完成。

给下一个对话窗口用。目标：**先把图像分类这条线做完**，NLP 线已上板跑通可直接复用。

---

## 0. 环境与路径（照抄即可）

| 项 | 值 |
|---|---|
| 产品根目录（SNN） | `D:/dl_snn` |
| 平台 Vivado 工程 | `D:/spk_platform/project_1/project_1.xpr` |
| 板子 | PYNQ @ `192.168.2.99`，用户 `xilinx` / 密码 `xilinx`，PYNQ 2.7.0 |
| 板端目录 | `/home/xilinx/jupyter_notebooks/board` |
| Vivado | `E:\\Vitis\\Vivado\\2021.1\\bin\\vivado.bat` |
| ModelSim | `d:/modelsim/win64/vlog`、`d:/modelsim/win64/vsim` |
| Python（PC） | `E:/miniconda/python.exe` |
| 器件/频率 | `xczu15eg-ffvb1156-2-i`，Vivado 2021.1，5 ns（200 MHz） |

---

## 1. NLP 线：已完成，可演示

### 1.1 硬件

- bit：`D:/spk_platform/board/snn_platform.bit`（r7-sumsfix，2026-10-30 23:10）
- 时序：Setup WNS **+0.051 ns** / TNS 0 / 0 failing；Hold WHS **+0.011 ns** / THS 0 / 0 failing
- sha256：`9b2e318c5eacd2f21c39bd6c8864cd0f41ca47ed069ee6bbd7ea45b0191c267b`

### 1.2 板端验证结果

3 个真实 checkpoint 样本 **全部 match=True**（读完 sums 后与 golden 逐通道相等）。

**重要：这里有一个软件补偿**，RTL 仍待修（见 §5.1）：
```python
# 硬件读地址 i 返回 sums[(i+64) % 128]，所以读第 j 通道要用 (j+64)%128
sums = [dev.rd(SUMS + 4*((j+64) % 128)) for j in range(128)]
```

### 1.3 演示脚本（板端）

- `demo_nlp.py`：`SnnNLP().show("这个手机很好用")` → 打印正面/负面概率条 + 结论
- 资产：`demo_vocab.json`（4000 字表）、`demo_head.npz`（FP32 head 2×128）、`demo_samples.json`
- **准确性：dev 集 74.7%（多数类基线 63.3%）**，演示时不能说"很准"

### 1.4 已知限制

- 每句约 2–3 s：其中 PL 计算只有 **16.95 ms**（3,389,415 cycles @200 MHz），其余是每句重载 overlay + MMIO 写 0.92 MB 刺激
- PL 跑完不回 idle，**必须每句重新 `Overlay()`**，否则第二个样本会超时

---

## 2. 图像线：工具链已完成，RTL 端到端未通过

### 2.1 工具链（已完成，有产物）

配置族 `spikformer-b1-cifar10`，示例配置：
- `examples/spikformer/config/model_native_b1.json`（32×32×3，E=384，H=12，HD=32，MLP=1536，T=4，depth=1，10 类）
- `examples/spikformer/config/model_multiblock_b.json`（E=192，H=6，MLP=768，T=2，depth=2）

编译命令：
```bash
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/model_native_b1.json \
  --fixture-seed 7 --out-dir build/spikformer/vis_native
```

产出（已验证存在）：`params/sps_conv1_w(2592B)`、`sps_conv1_b`、`sps_conv2_w(331776B)`、`sps_conv2_b` + block 全套；
`generated/spikformer_config.vh` 含 `SPIK_SPS_H/W/C0/C1/C2/TOKENS/CONV1_BEATS/CONV2_BEATS`；schedule/manifest/numeric_contract 齐全。

**本轮新增的推导规则**（`tools/backends/spikformer/config.py`）：
- `C1 = embed/4`（可被配置 `sps_c1` 覆盖），`C2 = embed`
- `tokens = (image_h/4)*(image_w/4)`，要求正方形、可被 4 整除
- 卷积 beat 数：`conv1_beats = ceil(C1*C0*9/8)`，`conv2_beats = ceil(C2*C1*9/8)`

### 2.2 RTL（已写，编译通过，回放未过）

`rtl/ops/spiking/control/spikformer_vis_top.sv`
```
image → SPS(conv1+LIF+2×maxpool+conv2) → spiking_block_top → 逐通道求和
```

主机喂数顺序（与 SPS 回放 TB 完全一致，**不能改**）：
```
1. block 权重 1136 拍 → blkrom
2. conv1 权重 54 拍  → 直通给 SPS
3. 偏置对 32 拍     → 直通给 SPS
4. 像素 64 拍       → 直通给 SPS
5. conv2 tile 576 拍 → w2rom
6. S_TREP 阶段从 w2rom 循环重放给 SPS
```

### 2.3 回放验证状态

| TB | 用途 | 结果 |
|---|---|---|
| `tb_vis_sps.v` | **只测 SPS 一级（快，几十秒）** | **FAIL** |
| `tb_vis_replay.v` | 整链（SPS→block→sums） | FAIL |

golden 生成链：
```bash
E:/miniconda/python.exe -B scripts/spikformer/gen_sps_replay.py            # 图像→token golden
TOKSRC=build/spikformer/sps_replay/exp.hex \
  E:/miniconda/python.exe -B scripts/spikformer/gen_block_replay.py          # token→block 输出
E:/miniconda/python.exe -B scripts/spikformer/gen_vis_tok.py               # token 展平供 TB 比对
E:/miniconda/python.exe -B scripts/spikformer/gen_vis_sums.py              # 逐通道 sums
```

回放尺度：H=W=8，C0=3，C1=16，C2=32，T=2，LANES=8，TC2=4 → 4 token × 32 通道；
block：E=32，H=1，HD=32，MLP=64，T=2，TOK=4，TL=8，TC=8，RW=4。

---

## 3. 当前卡点（下一个窗口从这里开始）

### 3.1 已验证正确的部分（有探测数据）

| 探测项 | 值 | 结论 |
|---|---|---|
| `u_sps.f0[0][0][0:2]` | `59 3f 48` | 与 pix.hex 原值一致 → **像素通路正确** |
| `u_sps.wbank1[0][0][0]` | 16 | 与 w_ld1.hex 首字节一致 → **conv1 权重正确** |
| `u_sps.bias1[0]` | -142 | 与 b_ld.hex 低 21 位一致 → **conv1 偏置正确** |
| `u_sps.f1[0][0][15:0]` 等 | `3fcf / b333 / c3f3 / 30ff` | 有区分度、未饱和 → **conv1+LIF+池化正确** |
| `u_sps.bias2f[0]` | -92 | 与 b_ld.hex 高 21 位一致 → **conv2 偏置正确** |
| `u_sps.tbeat` / tile 流 | `bi` 正常递增 | 修复后 **conv2 tile 供给正常** |

### 3.2 症状

SPS 输出 16 个 token beat，但**几乎全是 1**：
```
got  = 1111101111111111
want = 0011001100000000
```

### 3.3 下一步（按顺序，每步都很快）

1. **比 f1 的数值**（不是有没有值，是逐值 vs golden）
   在 `tb_vis_sps.v` 里 dump `u_sps.f1[y][x][(g+1)*16-1 -: 16]`，与 gen_sps_replay 的 `f1` 逐值比。
   如果 f1 已经不对 → 问题在 conv1（但 wbank1/bias1/f0 都对，那就是 **CALC 窗口取指或 padding 规则**）。
   如果 f1 对 → 问题在 conv2/P2。

2. **比 psum2**：第一个 tile 累加完成后 dump `u_sps.psum2[a][tt]`。

3. 定位后修 `spikformer_vis_top.sv` 或走线，重跑 `tb_vis_sps`（单级，几十秒）。

4. `tb_vis_sps` PASS 后跑 `tb_vis_replay`（整链）。

### 3.4 怀疑点（供参考）

- SPS 的 P2 是"按 tile 分组累加"，tile 顺序必须与 `w_tiles.hex` 的 576 拍顺序严格一致；
  我目前是全局递增取模，需确认**每次 SPS 重新请求时 ROM 指针是否从该 tile 的起点开始**
- 也可能 token 捕获采样沿不对（`sps_tvalid` 常高时同一拍被重复捕获）

---

## 4. 本轮在 `spikformer_vis_top.sv` 上修掉的 bug（都是真 bug，别回退）

| # | bug | 后果 | 修法 |
|---|---|---|---|
| 1 | `sps_wvalid` 常拉高 | SPS 每周期吞一拍垃圾 | 由主机 `i_w_valid` 门控 |
| 2 | `o_w_ready` 用寄存器 | 与主机 valid 差一拍，静默丢拍 | 改组合（always_comb 按状态选） |
| 3 | `o_pix_ready` 被 assign 和时序块双驱动 | 像素通路随机挂死 | 只保留 `assign o_pix_ready = sps_pready` |
| 4 | 顶层自数拍子（54/32）决定相位 | 与 SPS 内部计数器错位 | 相位由 SPS 的 ready 驱动 |
| 5 | `bicnt` 位宽 `clog2(BIB+1)=6` | 装不下 576 个 tile，永远到不了 575 | 改为 `clog2(max(BIB,W2B)+1)` |
| 6 | `sps_wdata` 恒等于主机总线 | S_TREP 重放阶段喂残留值，conv2 权重全 X | `st==S_TREP` 时取 `w2rom[bicnt]` |

---

## 5. 其他未完成项

### 5.1 RTL 缺陷：SUMS 读窗口偏移 64（**必须修**）

`D:/spk_platform/src/snn/snn_axi_lite_csr.sv`：
```verilog
// 现状（错）：araddr_q[8:2] 含基地址 0x100>>2 = 64
if (araddr_q >= 12'h100 && araddr_q < 12'h300)
  s_axil_rdata = sums_flat[{araddr_q[8:2], 5'b0} +: 32];
// 应为：
  s_axil_rdata = sums_flat[{(araddr_q[8:2] - 7'd64), 5'b0} +: 32];
```
修完需要重新综合+实现出 bit（约 65 分钟），然后删掉 `demo_nlp.py` 里的 `ROT = 64`。

改之前的版本备份在 `snn_axi_lite_csr.sv.bak_sumsfix`。

### 5.2 PL 不回 idle

连续跑第二个样本会超时。根因是软复位只 flush FIFO、不排空在途 DMA。当前靠"每样本重载 overlay"绕过。
修好后每句可降到约 17 ms（现在是 2–3 s）。

### 5.3 图像线上板

还需要：
- `spikformer_vis_platform_top`（把 NLP 平台 top 的 embed 前端换成 SPS）
- 图像侧向量生成器 + 板端 run 脚本
- 真实权重：**上游仓库不带 checkpoint**，要自己训 CIFAR-10 的 Spikformer-S

---

## 6. 纪律（不要违反）

1. **未执行的步骤一律写 `not_run`**，不得用 fixture 结果或 NLP 结果冒充图像线 PASS。
2. **负 WNS 记 fail**；不降频、不加时序例外。
3. 重任务走共享锁，不杀别人进程：
   ```bash
   E:/miniconda/python.exe -B D:/spk_platform/scripts/run_locked.py <job_id> <cmd...>
   ```
   锁文件：`D:/dl_v0.1/build/onnx_e1/v02_real_yolo_20261026_2022_b1/jobs/heavy.lock.json`
4. 新 RTL 必须同时登记：
   - `rtl/ops/spiking/source_manifest.json` 的 `sources`（**写全路径** `rtl/ops/spiking/...`）
   - `rtl/operator_library.json` 的 `sources`
   否则 compile 时报 `FileNotFoundError`。
5. 重生成 bit 的脚本模板：`D:/spk_platform/scripts/run_sumsfix_bitstream.tcl`
   （reset synth_1 → 综合 → reset impl_1 → 到 write_bitstream → 报告 + `write_hw_platform`）
   跑法：`cmd //c D:/spk_platform/scripts/run_sumsfix.bat`
6. .hwh 从 xsa 里取：`zipfile.ZipFile('project_1/snn_platform.xsa').read('design_1.hwh')`，
   与 bit 同名同目录（PYNQ Overlay 要求）。

---

## 7. 关键文件清单

```
D:/dl_snn/
  rtl/ops/spiking/control/spikformer_vis_top.sv     ← 图像端到端顶层（在改）
  rtl/ops/spiking/control/spiking_block_top.sv      ← 已验证的单 block
  rtl/ops/spiking/sps/*.sv                          ← SPS 前端（回放 8x8 通过）
  tools/backends/spikformer/{config,weights,codegen,compile,schedule}.py
  scripts/spikformer/gen_sps_replay.py              ← 图像→token golden
  scripts/spikformer/gen_block_replay.py            ← 支持 TOKSRC 环境变量
  scripts/spikformer/gen_vis_tok.py / gen_vis_sums.py
  verification/sim/spikformer/tb_vis_sps.v          ← SPS 单级（快，先跑这个）
  verification/sim/spikformer/tb_vis_replay.v       ← 整链
  docs/SPIKFORMER_NLP_RUNBOOK.md                    ← NLP 上板手册
  docs/SPIKFORMER_NLP_REVIEW_PLAN.md                ← 方案评审稿
D:/spk_platform/
  board/                                            ← NLP 上板包（bit/hwh/vec/demo_*）
  src/snn/snn_axi_lite_csr.sv                       ← SUMS 偏移 bug 在这里
  scripts/run_sumsfix_bitstream.tcl / run_sumsfix.bat
  project_1/project_1.xpr
```

---

## 8. 第一个动作建议

```bash
# 1) 确认现状（应看到 token 仍错、f0/wbank1/f1 正常）
cd D:/dl_snn
d:/modelsim/win64/vlog -sv rtl/ops/spiking/control/spikformer_vis_top.sv \
  rtl/ops/spiking/sps/*.sv rtl/ops/spiking/lif/*.sv \
  rtl/ops/spiking/attention/spiking_popcount.sv rtl/ops/spiking/attention/spiking_attention_pe.sv \
  rtl/ops/spiking/mlp/spiking_mlp_controller.sv rtl/ops/spiking/control/spiking_block_top.sv \
  verification/sim/spikformer/tb_vis_sps.v
d:/modelsim/win64/vsim -c -do "run 20000 us; quit" tb_vis_sps

# 2) 然后按 §3.3 第 1 步比 f1
```

---

## 9. 2026-10-31 13:20 状态更新（行为级已闭环，独立复验）

本轮（另一窗口）修复了 block golden 落后于 2026-10-29 block 合约的问题：
`gen_block_replay.py` 中 proj 与 fc2 后都应先 LIF 再把 0/1 spike 加回残差，
旧模型把完整线性累加值直接加回。RTL `spiking_block_top.sv` 未改
（sha256 `841df601…f9d19`）。

本窗口独立复验结果（重新生成全部 golden 后重跑）：

| 测试 | 结果 |
|---|---|
| `tb_block_replay`（默认随机 token） | **ALL PASS（16 beats）** |
| `tb_block_replay`（SPS token） | **ALL PASS（16 beats）** |
| `tb_vis_sps` | **ALL PASS（16 beats）** |
| `tb_vis_replay` | TOKEN CHECK fails=0；TSUM 4 6 12 16 24 16 16 3；**ALL PASS（32 channels）** |

日志：`sim_work/vis_sps/vsim_{block_r,sps_r,replay_r}.log`。

同时，本窗口此前修掉的图像线 RTL bug（已保留）：
- `spiking_sps_top.sv`: S_LDT 只在 `i_weight_valid && o_weight_ready` 时消费（原先进周期抢拍导致 ROM 指针每 tile 少 1）。
- `spikformer_vis_top.sv`: W2B 应含全部 token 位置（`NTOK*((C2*C1*9+7)/8)`，本几何 9216）；`st` 4 bit；`blk_wdata` 驱动；`capi` 位宽 `$clog2(NTOK*G)`；S_START 清 acc；S_RUN 只在 valid&&ready 推进权重指针。
- 两个 TB 的激励顺序/token 比较格式修正。

## 10. 下一步（图像线）

行为级回放已闭环，剩下是平台化与上板：

1. **native 配置的综合/实现**：`build/spikformer/vis_native` 工具链产物已存在
   （`generated/spikformer_config.vh`、params、schedule、manifest）。但
   `spikformer_vis_top` 目前是回放几何（片上 ROM 全量权重），native 32×32 需要
   DDR 流式权重路径。
2. **`spikformer_vis_platform_top`**：仿照 NLP 平台 wrapper，把 embed 前端换成
   SPS，权重/图像从 PL DDR4 经 DMA 流式输入，输出 sums 到 DDR；再做综合实现。
3. **向量 + 板端脚本**：图像侧向量生成器（pix/conv 权重/DDR 布局）与 PYNQ run 脚本。
4. **真实权重**：上游不带 CIFAR-10 checkpoint，需要先训 Spikformer-S；否则上板只能
   用 fixture 权重做功能冒烟，不能报精度。

## 11. 2026-10-31 14:10 资源问题交接

native 几何综合评估结论、证据、SPS 改造方案建议，见
`docs/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md`（自包含，可直接交给另一个对话）。
一句话：行为级已全绿；native `spiking_sps_top` 按当前 RTL 不可综合（≈846k FF > 682k FF，
conv2 2304 MAC/拍 > DSP 1248），需 BRAM 化 + 同步读流水 + MAC 时间复用；
block 方向可行（大 RAM 已被推成 BRAM cascade），完整 util/timing 待续跑。
