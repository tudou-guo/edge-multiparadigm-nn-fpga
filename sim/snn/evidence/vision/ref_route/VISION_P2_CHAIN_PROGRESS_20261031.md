# 图像线 P2：native 整链回放进度（2026-10-31）

状态（只记录实测；未跑写 not_run）：
- **P2-1 单图（无反压，修复后）：PASS —— BLKOUT 3072/3072 beats bit-exact + 384 sums 全等 + last/done。**
- **P2-2 两图连续（修复后）：FAIL（阻塞在 RTL 第二次 start 的 re-arm 缺失，非 TB/golden）**——同一运行里 img0 SUMS+BLKOUT 均 PASS，img1 卡在 SPS=C_LDW1/block=S_TOK 直至 watchdog；根因见 §2.1。
- 修复前单图：BLKOUT PASS、SUMS FAIL（382/384），作为 before 对照保留。
- 说明：单图 PASS 日志 `tb_vis_native_chain_p2fix_img1_20261031.log` 是 `+IMG2=0` 单图运行；该次 TB 终行文案曾硬编码 “(2 images, …)”，**TB 已改为按实际 imgs_done/en2 打印**（新 TB sha 8aead49c…），日志本体保留作单图证据。
- 输出侧反压（block-out/sums 外部 stall）：not_run（A 轮 top 的 blk_ordy 在 N_RUN 恒 1，无外部 hook；本轮不改该结构）。
- OOC 综合/实现、平台、板测：not_run。
- RTL 只改一处（lead 批准）：spikformer_vis_native_top.sv sums 累加修复（6fe26228… → 8ece767a…）；sps/block/lif/attention/mlp 冻结未动（ctrl 728337f1…；block_top 841df601…）。

## 1. 结论与证据

| 检查 | 状态 | 证据 |
|---|---|---|
| block-out 3072 beats 逐 nibble（img1，修复前） | PASS | `tb_vis_native_chain_p1_img1_20261031.log`：`BLKOUT PASS img0 (3072 beats bit-exact, last ok)` |
| 384 sums（img1，修复前） | FAIL（RTL 缺陷，382/384） | 同日志：`SUMS FAIL img0 mismatches=382` |
| **单图 img1（修复后）** | **PASS** | `tb_vis_native_chain_p2fix_img1_20261031.log`（+IMG2=0）：`SUMS PASS img0 (384 sums)` + `BLKOUT PASS img0 (3072 beats bit-exact, last ok)` + `ALL PASS`；3:22 wall、Errors 0 |
| 两图连续（修复后） | **FAIL（RTL restart gap）**：img0 PASS；img1 hang（SPS=C_LDW1/block=S_TOK，watchdog 120M cyc） | `tb_vis_native_chain_p2fix_img2_20261031.log`（sha 9dc10409…）：`SUMS PASS img0`+`BLKOUT PASS img0`，然后 `FAIL timeout img=1 top=2 blk=2 sps=1` |
| 输入侧气泡（+BPS） | not_run | 本轮时间不够，未跑 |
| 输出侧反压 | not_run | 无 RTL hook（ABI gap） |

单图日志：`build/spikformer/vision_ref_20261031/sim/tb_vis_native_chain_p1_img1_20261031.log`
（sha256 c073917f…；Time 351,297,595 ns、7:25 wall、Errors 0）
编译日志：`compile_native_chain_p1_20261031.log`（sha 05186c62…，work_chain 独立库）。

## 2. sums FAIL 的根因（已用 golden 复算证明，非 TB/golden 差异）

- 现象：img1 382/384 channel 不等，got 显著偏小（例 ch6 got=64 want=256；ch15 got=128 want=512）。
- 证明：用同一 golden block out[64][384][4] 复算单时间片和 → **got sums 精确等于仅 t=3 的逐 beat 累加**（15 个打印 channel 全等；mean got/want=0.2484≈1/4）。脚本：`sim_work/vis_sps/check_chain_sums.py`。
- 缺陷位置：`rtl/ops/spiking/control/spikformer_vis_native_top.sv` N_RUN 累加段：
  `for (a) for (t2) acc[(capi%G)*TL+a] <= acc[(capi%G)*TL+a] + {{(OW-RW){1'b0}}, blk_odata[(a*T+t2+1)*RW-1 -: RW]};`
  同一 `acc` 在 t2 循环内多次 non-blocking 赋值：每次右值读的都是同一旧 `acc`，只有循环最后一次赋值生效 → 每个 block-out beat 只累加 t2=T-1 一个时间片，与 `gen_vis_sums.py` 的“sum over all tokens and all T timesteps”合约不符。
- 为什么不是 golden/TB：block-out 3072/3072 全等已反证 block 输出与 golden 完全一致；sums golden = block out 在 token×T 上的和（`gen_vis_sums.py`：`out.sum(axis=(0,2))`）。
**修复（lead 批准，已应用；仅 `spikformer_vis_native_top.sv` 一个文件）**
- 修法（img-native 独立复核同结论，等效“方案B”）：新增 module 级 `beat_add[TL]` always_comb，把一个 beat 的 T 个 RW nibble 先组合求和；N_RUN 里每 lane 只做一次 `acc[(capi%G)*TL+a] <= acc[...] + beat_add[a]`。保持“每 beat 只在 blk_ov&&blk_ordy 拍累加一次”的握手语义。
- 修复前 sha256 `6fe26228fea27d5f70fe0f0cca027a5bd1fee9097c3878e27d9b620d1a5c63f9`（备份 `sim_work/vis_sps/backup_vis_native_top_before_p2fix_1722.sv`）；
  修复后 sha256 **`8ece767a61823b062bfae8c9c74164d8ec9da0efdc53fcc092e34d1993ad5ad0`**。
- 数据/握手/ABI/数值合约不变；只是累加语义从“每 beat 只加 t=T-1”改为“每 beat 加 T 个时间片之和”。
- **不影响 native SPS 门**：`tb_sps_native` 只实例化 `spiking_sps_ref_top`，不实例化 `spikformer_vis_native_top`；sps RTL 仍冻结在 ctrl 728337f1…，r4 native SPS PASS 证据不受本修复影响。


## 2.1 两图连续 FAIL 的根因（RTL 第二次 start 未 re-arm；不是 TB/golden）

- 现象：修复后两图运行中 img0 的 SUMS PASS + BLKOUT PASS；img1 在 `top=N_RUN, blk=S_TOK, sps=C_LDW1` 静止到 watchdog（120M cycles），`blkw_r=0, spsw_r=0`（TB 正在等发 SPS w1 beat 的 fire）。
- 机理（RTL 只读分析，未改）：
  1. `spiking_rgb_encoder.sv` 的 `w1_done/bias_done/pix_done/fwin_loaded` 是“一次装载完成即置位、永不复位”的状态（只有 s_rst 清）。第二次 i_start 时 encoder 直接进到最后分支 `o_w_ready<=0`，**不再打开权重接收**。
  2. `spiking_sps_ref_ctrl.sv` 的 `enc_started` 亦是 sticky（image0 置 1 后不清），第二次 C_ENC 不会再发 `o_enc_start`。
  3. 因此 SPS 第二次进 C_LDW1 后 `o_weight_ready = i_enc_w_ready = 0`，主机 w1 拍永远不 fire → SPS 不前进；block 在 S_TOK 等 token → 两图连续死锁。
- 证据：timeout 现场 `sps=1 (C_LDW1)` 且 `spsw_r=0`，与“encoder 不再 ready”一致；img0 在同一次运行中 SUMS/BLKOUT 全 PASS，排除 golden/TB。
- 修复方向（下一轮 RTL，本轮冻结未改）：encoder 在 i_start（或 SPS 给的 o_enc_start）时清 w1_done/bias_done/pix_done/fwin_loaded 并重新收权重/像素；SPS ctrl 在 i_start 时清 enc_started；对应 plan §5.2 “第二次 start 必须真正 IDLE/重装、不依赖全局 reset”的要求。
- 影响面：该 gap 只影响“同一 SPS 实例连续第二图”；单图回放（P2-1）与 native SPS 正式门（重新例化/单图）不受影响。

## 3. golden 生成方法与可信度（关键）

新增参数化生成器 `sim_work/vis_sps/gen_block_param.py`（sha 394605c4…），方程同 `scripts/spikformer/gen_block_replay.py` 推广到 HEADS>1/TOKENS>4/HIDDEN>64；权重流顺序与位序从 `spiking_block_top.sv` S_LDW 逐相位读出：
wq,wk,wv,wp（各 EEB=E*E/8=18432 beats，行主序 8 int8/beat）→ bq,bk,bv,bp（各 E/2=192 beats，2x signed32，lo=偶 hi=奇）→ fc1（FCD=HIDDEN*E/8=73728）→ bf1（HIDDEN/2=768）→ fc2（73728）→ bf2（E/2=192）= **222912 beats**。

**先自检再用于 native**（`--small-selftest`，全部逐字节通过）：
- 用 E=32/H=1/HD=32/HIDDEN=64/T=2/TOKENS=4/seed=52 复现旧 `build/spikformer/block_replay/` 的 wq/wk/wv/wp/bq/bk/bv/bp/wfc1/bf1/wfc2/bf2 `.hex` 全部相同；
- 用旧 `tok.hex` 复现旧 `exp.hex` 相同（方程正确）；
- `blk_w.hex` == wq..bf2 顺序拼接（1136 beats）。
另：native 第二图 `gen_sps_img2.py`（sha 98396cfd…）用与 `vectors/sps_native_tc2_8` 相同的 SPS 权重（seed=7 前段 draws）、新像素（独立 seed=99），输出 pix/exp；未重生成任何 conv 权重。

## 4. 交付向量（build/spikformer/vision_ref_20261031/vectors/）

| 文件 | 行数/内容 | sha256 |
|---|---|---|
| block_native/blk_w.hex | 222912 beats（S_LDW 顺序） | 157a00be… |
| block_native/exp_out_img1.hex | 3072 beats（128b, TL*T*RW nibble） | cef6a85d… |
| block_native/exp_sums_img1.hex | 384 x 32b | c65360d6… |
| block_native/exp_sums_img2.hex | 384 x 32b | e5f6899c… |
| block_native/exp_out_img2.hex | 3072 beats | （生成时同 tag img2） |
| block_native/meta_img1.json / meta_img2.json | 阈值全 60、seed=52、顺序/位序约定 | — |
| sps_native_img2/pix.hex | 1024 beats | 80ff0595… |
| sps_native_img2/exp.hex | 3072 beats | 0e31e8e3… |

未覆盖旧证据（`build/spikformer/block_replay/`、`vectors/sps_native*` 保留）。TB 的 SPS 权重流取 `vectors/sps_native_tc2_8`（TC2=8 正确口径）。

### 4.1 RTL 改动（本轮唯一）与冻结声明

| 文件 | 修前 sha256 | 修后 sha256 | 说明 |
|---|---|---|---|
| `rtl/ops/spiking/control/spikformer_vis_native_top.sv` | 6fe26228… | **8ece767a61823b062bfae8c9c74164d8ec9da0efdc53fcc092e34d1993ad5ad0** | lead 批准的唯一 RTL 修复：sums 每 beat 先组合求 beat_add[TL]=sum_t nibble，再单次累加 |

- 未改：`sps/*`（ctrl 仍 728337f1…）、`spiking_block_top.sv`（841df601…）、lif/attention/mlp/linear/embed/reduce、tools、平台、参考工程。
- native SPS 门不受影响：`tb_sps_native` 只实例化 `spiking_sps_ref_top`，不实例化 `spikformer_vis_native_top`；native SPS r4 PASS 证据（3072/3072、ctrl 728337f1）保持有效。

## 5. TB 改动（verification/sim/spikformer/tb_vis_native_chain.v，sha 8aead49c…）

- 修骨架的向量路径与加载：SPS 流改用 `sps_native_tc2_8`（原骨架指向旧 tc2=4 目录）；block-out golden `exp_out_img1/2`；两图 sums。
- 新增 **3072 block-out 逐 nibble 比较**（每 image），last 位置检查，失败计数；修 gotb 每图清零。
- 两图默认开启（+IMG2=0 可只跑单图）；watchdog 120M cycles。
- 终行按实际完成图数打印：`ALL PASS tb_vis_native_chain (images=%0d, each: 3072 block-out beats bit-exact + 384 sums; en2=%0d)`，避免单图运行被文案误标成两图（修复前骨架的问题）。
- 保留：+BPS 主机流气泡；block 每图重装权重；sums/last/done 检查。

## 6. 复现命令

```bash
cd /d/dl_snn
# 1) golden（不覆盖旧证据）
E:/miniconda/python.exe sim_work/vis_sps/gen_sps_img2.py --out-dir build/spikformer/vision_ref_20261031/vectors/sps_native_img2
E:/miniconda/python.exe sim_work/vis_sps/gen_block_param.py --e 384 --heads 12 --hd 32 --hidden 1536 --t 4 --tokens 64 --tl 8 --tc 8 --rw 4 --seed 52 --thr 60 --tok-file build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/exp.hex --out-dir build/spikformer/vision_ref_20261031/vectors/block_native --tag img1
E:/miniconda/python.exe sim_work/vis_sps/gen_block_param.py ... --tok-file build/spikformer/vision_ref_20261031/vectors/sps_native_img2/exp.hex --tag img2
E:/miniconda/python.exe sim_work/vis_sps/gen_block_param.py --small-selftest   # 生成器自检
# 2) 编译+跑（work_chain 独立库）
cd /d/dl_snn/sim_work/vis_sps
/d/modelsim/win64/vsim -c -do 'do compile_native_chain.do; quit -f' -l D:/dl_snn/build/spikformer/vision_ref_20261031/sim/compile_native_chain_p1_20261031.log
/d/modelsim/win64/vsim -c -work work_chain tb_vis_native_chain -l <log> -do 'run -all; quit -f'            # 两图
/d/modelsim/win64/vsim -c -work work_chain tb_vis_native_chain +IMG2=0 -l <log> -do 'run -all; quit -f'   # 单图
```

## 7. 未测项 / 后续入口

- 两图连续：**FAIL**（RTL 第二次 start 未 re-arm，§2.1）；img0 的 BLKOUT+SUMS 在该运行中仍 PASS。修复需改 `spiking_rgb_encoder.sv` + `spiking_sps_ref_ctrl.sv`（本轮冻结未改）。
- 输入侧气泡（+BPS=1）、输出侧反压（无 hook）、连续两图状态残留的“通过”判定：not_run / 待两图结果。
- native top sums 累加 RTL 缺陷：本轮冻结未改；下一轮按 §2 建议修 1 处，再重跑本 TB 即可闭合 P2 的 sums 门。
- 综合/实现/平台/板测：not_run。

## 8. 结果时间线

| 时间 | 项 | 结果 | 日志 |
|---|---|---|---|
| 17:18:39 | 单图（img1，无反压，修复前） | block-out PASS；sums FAIL（RTL bug，382/384） | tb_vis_native_chain_p1_img1_20261031.log |
| ~17:22 | 两图连续（修复前） | 按 lead 指示 kill（会撞同一 sums bug），partial 留作对照 | tb_vis_native_chain_p1_img2_prefix_20261031.log |
| ~17:24 | 修复 native top sums 累加（8ece767a…）并重编译 | 编译 Errors 0 | compile_native_chain_p2_fix_20261031.log |
| **17:26:10** | **单图（img1，修复后，+IMG2=0）** | **PASS：SUMS 384/384 + BLKOUT 3072/3072 + last/done；3:22 wall** | tb_vis_native_chain_p2fix_img1_20261031.log |
| ~17:29 | TB 终行改为按实际 imgs_done/en2 打印（sha 8aead49c…），重编译 | Errors 0 | compile_native_chain_p2_fix2_20261031.log |
| 17:29:40-17:33:42 | 两图连续（修复后，默认 IMG2=1） | **FAIL**：img0 PASS；img1 hang 于 SPS=C_LDW1（encoder 第二次 start 不 re-arm），watchdog 120M；5:19 wall | tb_vis_native_chain_p2fix_img2_20261031.log（sha 9dc10409…） |
