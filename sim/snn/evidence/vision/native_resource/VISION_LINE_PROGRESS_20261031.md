# 图像线进展（2026-10-31 13:10）

## 结论

- `tb_vis_sps`（SPS 单级回放）：**ALL PASS（16 beats）** ✅
- `tb_vis_replay`（SPS→block→sums）：**SPS token 全对（TOKEN CHECK fails=0）**，但 **block 输出与 golden 不符**，sums 错。
- 新发现：`spiking_block_top` 在当前小几何（E=32,H=1,HD=32,HIDDEN=64,T=2,TOKENS=4,TL=8,TC=8,RW=4）下与 `gen_block_replay` 模型不一致；旧的 `tb_block_replay.v` 已失效（引用已删除的 `i_thr_m` 端口、阈值位宽 20），无法作为“block 已验证”的证据。

## 本轮修掉的 bug（都已复现/验证）

| # | 文件 | bug | 影响 | 修法 |
|---|---|---|---|---|
| 1 | `rtl/ops/spiking/sps/spiking_sps_top.sv` | `S_LDT` 在进入周期的 ready 仍为 0 时就吃掉一拍权重 | 每 tile 少推进 1 个 ROM 指针，tile1+ 权重整体错位（f1 正确但 psum2 错） | 消费条件改为 `i_weight_valid && o_weight_ready` |
| 2 | `rtl/ops/spiking/control/spikformer_vis_top.sv` | `W2B` 按单个 token 位置算（576），但 SPS 每个 token 都重读 tile 窗口（256 次 load） | 权重 ROM 只有 1/16，回放中段重复旧权重 | `W2B = NTOK*((C2*C1*9+7)/8)`（本几何 9216）；TB 主机写入数改 9216 |
| 3 | 同上 | `st` 声明 3 bit，但状态到 S_DONE=9 | S_HEAD/S_DONE 被截断成 S_BW，整链挂死 | `logic [3:0] st` |
| 4 | 同上 | `blk_wdata` 从未赋值 | block 收到的权重全是 X | `assign blk_wdata = blkrom[bcnt];` |
| 5 | 同上 | `capi` 位宽 `$clog2(NTOK)`=2 bit，但 block 输出 16 拍 | 输出全写到 token0 | `$clog2(NTOK*G)` |
| 6 | 同上 | `acc` 只在复位清 | 多次运行/回归会累加 | S_START 清 acc |
| 7 | 同上 | S_RUN 在 `!wloaded` 时用 `blk_wready` 推进 bcnt（valid 还是 0） | 跳过 blkrom[0]，权重错位 | 只在 `blk_wvalid && blk_wready` 推进 |
| 8 | `verification/sim/spikformer/tb_vis_replay.v` | 主机激励顺序与 vis_top FSM 不一致（先 conv1/conv2 后 block 权重） | block 权重装进 blkrom 错误位置 | 改为 S_BW→S_W1→S_BI→S_PIX→S_TILE，tile 数 9216 |
| 9 | `verification/sim/spikformer/tb_vis_sps.v` | 用 `expb[got][15:0]` 直接比 16-bit token，但 `exp.hex` 是 nibble 打包 32-bit | 假 FAIL（格式错误） | 按 `tb_sps_replay` 的 nibble 提取比较 |

## 当前卡点（下一步）

`tb_block_replay` 用同一套 golden（`gen_block_replay.py`）也失败：
- 用 TOKSRC（SPS token）golden：247/256 nibble 错；
- 用默认随机 token golden：同样失败。

因此不是 vis_top 的问题，而是 **block_top 在当前小几何下与模型不一致**（或 2026-10-29 的 block contract 改写后模型/TB 未更新）。

下一步建议（按顺序）：
1. 用 token0 对比 block 内部阶段：QKV LIF 输出（`sbrow`/`lifspk`）、`avf`、`ares`、`fc1 hid`、最终 `o_out_data`，定位第一个不一致阶段。
2. 若 QKV 就错 → 查 block 权重 bank 写/读映射（`ld_addr` 公式在 E=32 时是否仍成立）。
3. 若 QKV 对、AV 错 → 查 attention PE 的 head/group 映射。
4. 定位后修 block_top，并恢复/更新 `tb_block_replay.v`（端口 `i_thr_proj/fc1/fc2`、阈值 32-bit）。

## 复现命令

```bash
cd D:/dl_snn/sim_work/vis_sps
# 编译（XPM 无关，纯 RTL）
vsim -c -do compile_vision.do -l compile_vision.log
# SPS 单级
vsim -c -work work tb_vis_sps -l vsim_sps.log -do "run 40000 us; quit -f"
# 整链
vsim -c -work work tb_vis_replay -l vsim_replay.log -do "run 80000 us; quit -f"
# block 单独（先修 TB 端口）
vsim -c -work work tb_block_replay -l vsim_block.log -do "run 4000 us; quit -f"
```

## 修改文件

- `rtl/ops/spiking/sps/spiking_sps_top.sv`
- `rtl/ops/spiking/control/spikformer_vis_top.sv`
- `verification/sim/spikformer/tb_vis_sps.v`
- `verification/sim/spikformer/tb_vis_replay.v`
- `verification/sim/spikformer/tb_block_replay.v`（端口修复，仍待对齐 golden）