# 图像线问题处理返回 / 交接（2026-10-31）

承接 `docs/VISION_LINE_PROGRESS_20261031.md`。本轮目标是定位并解决
`tb_block_replay` / `tb_vis_replay` 的 block 输出失配。

## 结论

问题已解决，图像行为级回放整链通过：

- `tb_block_replay`（默认随机 token）：**ALL PASS（16 beats）**
- `tb_block_replay`（SPS token）：**ALL PASS（16 beats）**
- `tb_vis_sps`：**ALL PASS（16 beats）**
- `tb_vis_replay`：**TOKEN CHECK fails=0，ALL PASS（32 channels）**

本轮没有修改 `spiking_block_top.sv`。根因是独立 Python golden
`scripts/spikformer/gen_block_replay.py` 落后于 2026-10-29 block 合约：RTL 在
`proj` 和 `fc2` 后均执行 LIF，再将 0/1 spike 加回残差；旧 golden 却把完整
线性累加值直接加回残差，因此会产生大面积、但稳定可复现的失配。

## 根因证据

### 1. 旧 golden 与当前合约不一致

当前权威模型 `scripts/spikformer/gen_nlp_common.py` 的 block 语义是：

1. `proj psum -> LIF(thr_proj) -> ares = input + proj_spike`
2. `fc1 psum -> LIF(thr_fc1) -> hid`
3. `fc2 psum -> LIF(thr_fc2) -> out = ares + fc2_spike`

RTL 也按该语义实现：共享 LIF 的 threshold mux 包含 `i_thr_proj` 和
`i_thr_fc2`，`S_PROJ` 与 `S_MLP` 均把 `lifspk`（而非 psum）加回残差。

旧 `gen_block_replay.py` 则执行：

```text
ares = (tok + proj_acc) mod 16
out  = (ares + fc2_acc) mod 16
```

因此旧期望值不是当前 block 合约的输出。

### 2. RTL 内部阶段没有出现首个偏差

使用已有定向多头 stage probe 对当前 RTL 做独立检查，结果：

- Q/K/V LIF spike bank：全部 token × 全部 channel × T 全匹配
- 双 head QK popcount / AV 映射：token0 全部 channel × T 全匹配
- 连续两次完整权重重载及最终残差输出：全部匹配
- 终止标记：`TB_BLOCK_MULTIHEAD: ALL PASS`

日志：`sim_work/vis_sps/vsim_block_stage_probe_20261031.log`

此外，当前 `spiking_block_top.sv` SHA256 为
`841df6017e51af858d3b128263b0478fe45cc93093863fbc7423f648759f9d19`，
与既有 NLP small 0-mismatch 回放报告中的 RTL SHA 相同：
`build/spikformer/nlp_48h_20261029_1915/reports/replay_small.json`。

### 3. 单变量修复验证了因果关系

保持 RTL、TB、权重和输入不变，仅修正 Python golden：

- 修复前：`FAILURES=240 tb_block_replay`
- 修复后：`ALL PASS tb_block_replay (16 beats)`

这排除了 block 权重 bank 地址、MAC 流水或 attention head/group 映射作为本次
失配根因。

## 本轮修改

### `scripts/spikformer/gen_block_replay.py`

1. `proj` 线性累加后执行 `lif_chain(..., THRPROJ)`，残差只加 proj spike。
2. `fc2` 线性累加后执行 `lif_chain(..., THRFC2)`，最终残差只加 fc2 spike。
3. 将 `proj/fc1/fc2` threshold 常量拆分命名，明确三段语义。
4. 新增诊断向量：
   - `g_proj.hex` / `g_proj1.hex`
   - `g_hid.hex`
   - `g_fc2.hex` / `g_fc21.hex`
5. 更新模块说明，明确当前 block 合约。

未修改任何 RTL。`build/spikformer/block_replay/` 与
`build/spikformer/vis_replay/` 是按修复后模型重新生成的回放产物；当前
`block_replay` 产物对应 `TOKSRC=build/spikformer/sps_replay/exp.hex`。

## 验证矩阵

| 门槛 | 结果 | 关键证据 | 日志 |
|---|---|---|---|
| 编译（ModelSim 2020.4） | PASS | Errors=0 | `compile_vision_baseline_20261031.log`、`compile_block_baseline_20261031.log` |
| 修复前 block / 默认随机 token | FAIL（预期基线） | 240/256 nibble mismatch | `vsim_block_baseline_20261031.log` |
| 定向 QKV/AV stage probe | PASS | QKV PASS、AV PASS、连续两次输出 PASS | `vsim_block_stage_probe_20261031.log` |
| 修复后 block / 默认随机 token | PASS | 16 beats | `vsim_block_fixed_model_20261031.log` |
| 修复后 block / SPS token | PASS | 16 beats | `vsim_block_sps_tokens_fixed_20261031.log` |
| SPS 单级 | PASS | 16 beats | `vsim_sps_regression_20261031.log` |
| SPS→block→sums 整链 | PASS | token fails=0；block weight=1136；block output=16；32 sums 匹配 | `vsim_replay_fixed_model_20261031.log` |

整链日志中的首 8 个通道和 golden 一致：

```text
TSUM 4 6 12 16 24 16 16 3
ALL PASS tb_vis_replay (32 channels)
```

## 从生成到回归的复现命令

PowerShell：

```powershell
Set-Location D:/dl_snn

python -B scripts/spikformer/gen_sps_replay.py
$env:TOKSRC = 'D:/dl_snn/build/spikformer/sps_replay/exp.hex'
python -B scripts/spikformer/gen_block_replay.py
Remove-Item Env:TOKSRC
python -B scripts/spikformer/gen_vis_tok.py
python -B scripts/spikformer/gen_vis_sums.py

Set-Location D:/dl_snn/sim_work/vis_sps
vsim -c -do compile_vision.do -l compile_vision.log
vsim -c -do compile_block.do -l compile_block.log
vsim -c -work work tb_block_replay -l vsim_block.log -do "run 4000 us; quit -f"
vsim -c -work work tb_vis_sps -l vsim_sps.log -do "run 40000 us; quit -f"
vsim -c -work work tb_vis_replay -l vsim_replay.log -do "run 80000 us; quit -f"
```

若要复现默认随机 token 的 block 门槛，执行 `gen_block_replay.py` 前不要设置
`TOKSRC`；随后重跑 `tb_block_replay` 即可。

## 当前状态与下一步

`VISION_LINE_PROGRESS_20261031.md` 中记录的 block/整链行为级阻塞已经关闭。
下一步可进入图像配置的综合、实现和板级验证；行为仿真结果不能替代时序收敛、
CDC、实现后仿真或实板结果。

注意：`D:/dl_snn` 当前不是 Git worktree，无法用 `git status/diff` 自动区分历史
未提交修改。本轮人工确认的唯一源码修改是
`scripts/spikformer/gen_block_replay.py`，其余源码保持进入本轮时的状态。
