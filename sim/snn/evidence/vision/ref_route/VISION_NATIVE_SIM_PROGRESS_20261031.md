# 图像线 native SPS / 整链仿真进度（2026-10-31）

状态（17:10 快照，后续以本文"结果时间线"为准）：
native SPS TB 已建成并按 RTL TC2=8 口径跑出 r4 = **PASS**（3072/3072 beats、98304/98304 nibble、last/done/watchdog、capture/flush 结构断言全过，日志见 §7）；r1~r3 为定位过程。native 整链 TB 已建成并通过编译，但缺 block_native 权重/3072 block-out/384 sums golden 与第二张图向量，运行按门控报 **NOT_RUN**（§6）。RTL 只读，未修改；未跑综合/实现，未访问板卡。文中 `build/.../` 为 `build/spikformer/vision_ref_20261031/` 的缩写（已全文展开）。

## 1. 交付物与可复制命令

| 交付物 | 路径 | 状态 |
|---|---|---|
| native SPS TB | verification/sim/spikformer/tb_sps_native.v | PASS（r4） |
| native SPS 编译脚本 | sim_work/vis_sps/compile_ref_native.do（独立库 work_native） | PASS |
| native 整链骨架 TB | verification/sim/spikformer/tb_vis_native_chain.v | 编译 PASS / 运行 NOT_RUN（缺 golden） |
| 整链编译脚本 | sim_work/vis_sps/compile_native_chain.do（独立库 work_chain） | PASS |
| 本文 | docs/VISION_NATIVE_SIM_PROGRESS_20261031.md | 完成 |

```text
# native SPS 正式门（重跑用；库独立，不碰别人的 work/日志）
cd D:/dl_snn/sim_work/vis_sps
vsim -c -do "do compile_ref_native.do; quit -f"
vsim -c -work work_native tb_sps_native -l D:/dl_snn/build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_20261031.log -do "run -all; quit -f"

# native 整链骨架（当前应报 NOT_RUN 并列出缺向量；不造 golden）
cd D:/dl_snn/sim_work/vis_sps
vsim -c -do "do compile_native_chain.do; quit -f"
vsim -c -work work_chain tb_vis_native_chain -l D:/dl_snn/build/spikformer/vision_ref_20261031/sim/tb_vis_native_chain_gate_imgnative_20261031.log -do "run -all; quit -f"
```

## 2. native SPS TB 参数与协议

- DUT：spiking_sps_ref_top，H=32、W_IN=32、C0=3、C1=96、C2=384、T=4、LANES=8、TC2=8、RM=4、ACC_W=21。
- 输出：64 token × 48 group = 3072 beats；每 beat=8 lane × T=4 × RW=4 = 128 bit（adapter 把 1bit spike 展成 nibble 0/1）。
- 主机流顺序（与 tb_sps_ref_small 一致）：w_ld1 324 beats → b_ld 384（21b pair，lo=C1、hi=C2）→ pix 1024 → w_sched 41472（(g2,tile) 顺序）。
- 检查：逐 beat 每 lane/t 与 exp.hex 比（exp 每行 8 个 hex char = lane 0..7，char 内 bit t = spike time），比较总数 3072×32=98304 nibble；last 只在 beat3071；done sticky；watchdog 12M cycle（实测 r4 约 5.57M cycle 结束）。
- 结构断言（只读层级探针，不 force）：psum capture 顺序 (pos,grp)==(cap_n%256, cap_n/256)，总数 12288=(H1*W1)*G2；每次 row flush 入口 (p2_g,p2_y)==(flush_n/16, flush_n%16) 且 p2cnt==ROWW，总数 768=G2*H1；C_DRAIN 入口 beat_pushed==12288 且 p2cnt==0（不允许 pad-flush 残留）。
- 只有数值/结构两类 checker 都 0 才打印 ALL PASS；正常退出 $finish。

## 3. 向量来源与关键发现（旧 native 向量 meta 的 tc2 与 RTL 不一致）

**发现**：build/spikformer/vision_ref_20261031/vectors/sps_native/meta.json 写 "tc2": 4（gen_sps_param.py 默认），但 native RTL（spiking_sps_ref_top 默认 + spikformer_vis_native_top 传参 + 计划书 §4 的 8 lane × 8 ch）是 **TC2=8**。RTL 的 C_C2TILE 按 (g2,c_tile) 顺序、每 tile 连续读 TILEB=LANES*TC2*9/8 拍；w_sched 必须按此 tile 顺序排。

处理（不覆盖旧证据）：用同一 seed=7/thr1=200/thr2=60 重新生成到新目录
`build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/`（命令见 §8）。逐文件 sha256：

| 文件 | 旧 sps_native（tc2=4） | 新 sps_native_tc2_8 | 结论 |
|---|---|---|---|
| w_ld1.hex | 相同 | 相同 | 与 tile 无关 |
| b_ld.hex | 相同 | 相同 | 与 tile 无关 |
| pix.hex | 相同 | 相同 | 与 tile 无关 |
| exp.hex | 相同 | 相同 | golden 数学与 tile 顺序无关（交叉验证） |
| w_uniq.hex | 相同 | 相同 | canonical (oc,ic,tap) 展平，不是流顺序 |
| w_sched.hex | 9ce5b23b… | 0f9e8f69… | **唯一不同：tile 内 (a,k,b) 排列**（老 36 beats/tile×24 tiles；新 72 beats/tile×12 tiles；总 41472 beats 不变） |

自检脚本 sim_work/vis_sps/check_native_sched_imgnative.py：按 RTL spiking_eyeriss_array 的 wtile[a][c][b] 解码规则，对 tc2=8 抽 4 个 tile（g2=0/1/47、tile=0/1/11）逐字节 MATCH；tc2=4 旧向量在其对应 RTL 下也 MATCH；w_uniq 与 canonical 展平逐字节一致（证明 w_uniq 不能直接当 w_sched 流用，TB 实际喂 w_sched）。

## 4. 结果时间线（每个 PASS 指日志路径）

| 轮次 | ctrl sha256 | 向量 | FAILURES（nibble/98304） | 首个偏差 | 结论 |
|---|---|---|---|---|---|
| r1pref | 旧版（<e6008ad6，16:41 编译） | 旧 tc2=4 w_sched | 44360 | beat0/lane0/t0 | 无效对照：向量/RTL TC2 不配对；旧 ctrl 亦有 small 53 nibble 问题 |
| r2 | e6008ad6（small 修复版） | 旧 tc2=4 w_sched | 44793 | beat0/lane0/t0 | 同上，作废 |
| r3 | e6008ad6 | 新 tc2=8 w_sched | 1142 | beat153/pos3(x3,y0)/g9/lane5/t1 | tile 配对修复有效（-97%）；残余集中在 token x=3..6、每 g 每行稀疏、hamming 1~5bit |
| ctrl | e6008ad6 | 旧 tc2=4 w_sched + RTL TC2=4（对照） | 512 | **同为 beat153/pos3/g9/lane5/t1** | 两 TC2 设置首个偏差相同 → 残余是共享链/时序边界，不是 tile 顺序 |
| r4 | **728337f1**（flush_now 收窄到 {WIN,TILE}） | 新 tc2=8 w_sched | **0** | 无 | **PASS（正式门）**：3072 beats + 12288 capture + 768 flush 结构断言全过 |

r3→r4 的 ctrl 差异（img-p1，非本任务改动）：flush 抢占 C_C2ROWS 会把 b1 读地址默认 0 打进 BRAM raddr，恢复后 ph2 采到 mem[0]，污染"最后一个输入 tile"窗口（native NT2=12、H1=16 更易触发）；修复为 flush_now={C_C2WIN,C_C2TILE}。
r4 结果：**ALL PASS tb_sps_native（3072 beats；capture=12288 flush=768 structural ok），FAILURES=0**，无 BEATFAIL/LAST/STRUCTFAIL，o_done sticky 后正常 $finish；**5,573,066 cycles、2:00 wall、Errors 0**；日志
`build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_20261031.log`；编译日志 compile_ref_native_imgnative_r4_20261031.log（Errors 0；Warnings 为既有 SVCHK/TFMPC 类，未改动 RTL）；RTL 哈希清单 tb_sps_native_imgnative_r4_rtl_hashes.txt（ctrl=728337f1…）。
独立复核：sim_work/vis_sps/verify_r4_pass_imgnative.py 对 r4 dump 与 exp.hex 做离线 diff → 3072/3072 beats 全等、total hamming=0、nibble 全在 {0,1}（输出 tb_sps_native_imgnative_r4_diff_summary.txt）。
旁证（img-p1，同一 ctrl 728337f1）：H32/W32/C1-16/C2-32/T4/**TC2=8** 代理 256/256 beats PASS、array tile 2048/2048、f1/pool1 bank 全等，日志 build/spikformer/vision_ref_20261031/sim/p1_r8_h32_t4_tc2_8.log。
说明：本轮正式门口径是 RTL TC2=8 + tc2=8 w_sched；旧 vectors/sps_native（meta tc2=4）不删除、仅作历史证据，不能用于 TC2=8 的 RTL。

附：TC2=4 对照（同一 ctrl 728337f1、旧 tc2=4 w_sched、RTL 例化 TC2=4 的 sim_work 对照 TB）：**ALL PASS 3072 beats / capture=12288 flush=768**，日志 tb_sps_native_tc2_4_control2_imgnative_20261031.log（说明修复后两种 TC2 装载口径都自洽；官方门仍以 TC2=8 为准）。

## 5. r3 残余偏差的定位证据（供复盘）

- 292/3072 beat 错，hamming 1~5 bit，失配 token x∈{3,4,5,6}（对应 spk2 列 6..13），y 均匀；capture 顺序断言全对（12288 个 (grp,pos) row-major 无错位）。
- 对照 TB（TC2=4 RTL+旧 tc2=4 向量）首个偏差与 TC2=8 完全相同 → 排除 tile 顺序/TC2 分支。
- img-p1 独立 H32/T4 边界回归定位到"每组最后一个输入 tile 的 array 结果错"（flush 抢占 C_C2ROWS 污染 BRAM 读地址），与本现象一致；新 ctrl 修复后 native r4 归零。
- TB 侧两个断言 bug 已修（不影响 RTL）：o_b0_we 是寄存器输出，flush 探针改为 C_C2W 入口沿 (prev_st) 检测；C_C2FL 首拍 beat_pushed=12287/p2cnt=15 是 in-flight 末拍的正常瞬态，改为在 C_DRAIN 入口检查无 pad-flush 残留（r4 全过）。
- 对照补充（新 ctrl 728337f1）：同一 native 几何、RTL 例化 **TC2=4** + 旧 tc2=4 w_sched 的 control TB（sim_work/vis_sps/tb_sps_native_tc2_4_control.v，独立 work_ctrl 库）**ALL PASS**：3072 beats、capture=12288、flush=768、1:56 wall，日志 tb_sps_native_tc2_4_control2_imgnative_20261031.log。即旧向量本身自洽，只是与 TC2=8 RTL 不配对；修复后两种 TC2 装载口径都归零。

## 6. native 整链 TB 骨架（tb_vis_native_chain.v）

状态：**编译 PASS、运行 NOT_RUN**（缺向量/golden，按门控显式打印缺失项后 $finish；不造 golden）。按 lead 分工，本 TB + TODO 自 2026-10-31 17:10 起交 **img-p1** 接手维护（交接时 sha256 2f57fcf9…；本会话只读、不再修改）。编译库 work_chain（独立），日志
`build/spikformer/vision_ref_20261031/sim/compile_native_chain_imgnative_20261031.log`、门控运行日志 `.../tb_vis_native_chain_gate_imgnative_20261031.log`。

- DUT：spikformer_vis_native_top，native 参数（E=384、HEADS=12、HEAD_DIM=32、HIDDEN=1536、TOKENS=64、TL=TC=8、RW=4、PSUM_SLICE=32、SUM_W=6、OW=32）。
- 每图主机 ABI（由 RTL 读出门控推导，未改 RTL）：i_blk_w 222912 beats → i_sps_w 324+384 → i_pix 1024 → i_sps_w 41472；i_start 后 o_done/o_sums_valid sticky。block 每图重新装权重（A 轮无片上跨图保留）。
- 骨架已实现：向量存在性门控、两图循环、+BPS 主机流气泡（反压）、block-out beat 计数与采样、384 sums 对比、last/done/watchdog；sums 输出侧反压无外部 hook（A 轮 top 内 blk_ordy 硬连），列为 ABI gap。
- 222912 beats 来源核对（spiking_block_top S_LDW 相位）：4×EEB(18432)=73728 + 4×192 偏置=768 + fc1 73728 + bf1 768 + fc2 73728 + bf2 192 = 222912。

### TODO（P2 整链门前必须补齐，已与 img-p1/lead 对齐）

1. native block golden 生成器（扩展 gen_block_replay.py 或新 gen_block_param.py）：E=384/HEADS=12/HD=32/HIDDEN=1536/T=4/TOKENS=64/TL=TC=8/RW=4，方程同 gen_block_replay.py；token 源用 vectors/sps_native_tc2_8/exp.hex（token-major，48 group/beat）。输出向量（新目录 vectors/block_native/，不得覆盖旧证据）：
   - blk_w.hex：222912 beats，严格按 S_LDW 顺序 wq,wk,wv,wp → bq,bk,bv,bp（每 beat 2×PSUM_SLICE signed，lo=偶 hi=奇）→ fc1 → bf1 → fc2 → bf2；
   - exp_out.hex：3072 个 block-out beat（TL*T*RW=128bit nibble 展开；须与 RTL o_out_data 位序对齐后定稿）；
   - exp_sums_img1.hex / exp_sums_img2.hex：384×32bit；
   - meta.json：thr_q/k/v/proj/fc1/fc2、seed、顺序/位序约定、beat 数。
2. 第二张不同输入：同一 SPS 权重、新像素；产出 vectors/sps_native_img2/{pix.hex,exp.hex} 与对应 exp_sums_img2.hex（不要重生成 conv 权重）。
3. 逐 beat 比较 exp_out.hex（当前只计数+采样前 8 个），P2 门要求 3072 个 block-out 全比 + 384 sums。
4. 输出侧反压：A 轮 native top 无 sums/block-out 外部反压 hook（N_RUN 内 blk_ordy 恒 1）；外部只能做输入侧气泡（+BPS=1）。若 P2 需要结果反压稳定，需加 TB-only hook 或下一轮 RTL 测试点，禁止本轮改 RTL。
5. 连续两图状态残留检查已按 topology 写好（第二图重发全部权重+像素），但需 golden 才能验证。

### 6.1 P2 后续结果（img-p1 接手后，17:36 交接；非本门 PASS）

- **P2-1 单图整链：PASS**——block-out 3072/3072 beats bit-exact + 384 sums 全等 + last/done；日志 `build/spikformer/vision_ref_20261031/sim/tb_vis_native_chain_p2fix_img1_20261031.log`（sha256 4ffa5ee8…，+IMG2=0 单图；3:22 wall、Errors 0）。
- **P2-2 两图连续：FAIL**——同一次运行 img0 SUMS+BLKOUT 均 PASS；img1 卡在 SPS=C_LDW1 / block=S_TOK 至 watchdog 120M cyc。日志 `.../tb_vis_native_chain_p2fix_img2_20261031.log`（sha256 9dc10409…）。根因（img-p1 只读分析，未改 RTL）：`spiking_rgb_encoder` 的 w1_done/bias_done/pix_done/fwin_loaded 与 `spiking_sps_ref_ctrl` 的 enc_started 均为 sticky、只在 s_rst 清 → 第二次 start 后 encoder 不再收权重；按 plan §5.2「第二次 start 真 IDLE/重装」下轮修。
- 修复前对照：`.../tb_vis_native_chain_p1_img1_20261031.log`（BLKOUT PASS / SUMS FAIL 382，即 native top 同索引多 NBA 缺陷）。
- native top RTL 修复（本轮唯一 RTL 变更）：6fe26228… → **8ece767a…**（sums 累加改为每 beat 先 always_comb 求 beat_add[TL]=Σ_t nibble，再每 lane 单次 acc+=；ABI/数值/握手不变）；sps ctrl 728337f1…、block_top 841df601… 未动。**本门 r4 不实例化 spikformer_vis_native_top，故 native SPS 门 PASS 证据不受该修复影响。**
- 详见 img-p1 文档 `docs/VISION_P2_CHAIN_PROGRESS_20261031.md`（sha256 11f243f7…）；golden/TB 自检见该文（gen_block_param.py 394605c4…、tb_vis_native_chain.v 8aead49c…）。

## 7. 证据索引

- native SPS 正式门 PASS：build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_20261031.log
- 过程（作废/对照，保留）：tb_sps_native_imgnative_r1prefix_20261031.log、_r2_、_r3_、tb_sps_native_tc2_4_control_imgnative_20261031.log
- 编译：build/spikformer/vision_ref_20261031/sim/compile_ref_native_imgnative_r4_20261031.log、compile_native_chain_imgnative_20261031.log
- RTL 哈希：build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_rtl_hashes.txt
- 向量：build/spikformer/vision_ref_20261031/vectors/sps_native（旧，tc2=4，保留）、sps_native_tc2_8（新，正式门用）
- 分析脚本：sim_work/vis_sps/analyze_native_imgnative.py、check_native_sched_imgnative.py、analyze_r3_imgnative.py
- 整链骨架：verification/sim/spikformer/tb_vis_native_chain.v（TODO 在文件尾注释）

## 8. 向量重生成命令（可复制）

```text
cd D:/dl_snn
E:/miniconda/python.exe scripts/spikformer/gen_sps_param.py --h 32 --w 32 --c1 96 --c2 384 --t 4 --lanes 8 --tc2 8 --thr1 200 --thr2 60 --seed 7 --out-dir build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8
```

## 9. 未完成项 / 明确 not_run

- native 整链 bit-exact（3072 block-out + 384 sums + 两图）：**not_run**（缺 §6 TODO 1/2 的向量与 golden；骨架仅编译/门控）。
- 输出侧反压验证：**not_run**（A 轮 top 无外部 hook）。
- 连续两图状态残留实测：**not_run**（同上）。
- OOC 综合/实现、平台、板测：**not_run**（不在本任务，未跑 Vivado、未访问板卡）。
- 本文所有 PASS 均指有日志路径的仿真结果；未跑的写 not_run，不用推算值冒充实测。

## 10. 最终证据清单（native SPS 门，冻结口径 2026-10-31）

### 10.1 门状态

| 门 | 状态 | 证据 |
|---|---|---|
| native SPS（H32/W32/C1-96/C2-384/T4/TC2=8，3072 beats）| **PASS** | tb_sps_native_imgnative_r4_20261031.log（ALL PASS；capture=12288 flush=768 structural ok；5,573,066 cyc；Errors 0） |
| native SPS 独立复核 | **PASS** | tb_sps_native_imgnative_r4_diff_summary.txt（3072/3072 beats，hamming=0） |
| 门结果 JSON（机读）| — | build/spikformer/vision_ref_20261031/sim/tb_sps_native_gate_result_imgnative_20261031.json |
| TC2=4 对照（同 ctrl+旧 tc2=4 向量）| **PASS（旁证，非门）** | tb_sps_native_tc2_4_control2_imgnative_20261031.log |
| native 整链（3072 block-out + 384 sums + 两图）| **not_run** | tb_vis_native_chain_gate_imgnative_20261031.log（编译 PASS/门控 NOT_RUN，缺向量） |
| 结果反压 / 连续两图实测 | **not_run** | 见 §6 TODO 4/5 |
| OOC/平台/板测 | **not_run** | 不在本任务 |

### 10.2 可复核命令（复制即用）

```text
# 1) 编译 native SPS TB（独立库 work_native，不碰别人的 work 库/日志）
cd D:/dl_snn/sim_work/vis_sps
vsim -c -do "do compile_ref_native.do; quit -f"

# 2) 跑正式门
cd D:/dl_snn/sim_work/vis_sps
vsim -c -work work_native tb_sps_native \
  -l D:/dl_snn/build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_20261031.log \
  -do "run -all; quit -f"
# 预期尾行: "# ALL PASS tb_sps_native (3072 beats; capture=12288 flush=768 structural ok)"

# 3) 独立 Python 复核（读 r4 dump + exp.hex）
E:/miniconda/python.exe D:/dl_snn/sim_work/vis_sps/verify_r4_pass_imgnative.py
# 预期: exact beats=3072/3072, total hamming bits=0, nibbles outside {0,1}=0
```

### 10.3 向量目录与身份（sha256 全表见 manifest 文件）

- 正式门向量：`build/spikformer/vision_ref_20261031/vectors/sps_native_tc2_8/`
  - 生成：`E:/miniconda/python.exe scripts/spikformer/gen_sps_param.py --h 32 --w 32 --c1 96 --c2 384 --t 4 --lanes 8 --tc2 8 --thr1 200 --thr2 60 --seed 7 --out-dir <上目录>`
  - 关键 sha256：w_ld1 2e99eba8…、b_ld c1a61406…、pix e27327ab…、exp 3a0993ea…、w_uniq b026fd62…、**w_sched 0f9e8f69…（TC2=8 tile 顺序）**、meta a2807fc3…（"tc2":8）。
- 旧向量：`build/spikformer/vision_ref_20261031/vectors/sps_native/`（meta "tc2":4；w_ld1/b_ld/pix/exp/w_uniq 与新目录逐字节相同，仅 w_sched 9ce5b23b… 不同；不删除，仅作历史证据，**不能配 TC2=8 RTL**）。
- RTL 冻结哈希（本门）：ctrl 728337f1…；其余 sps RTL 见 `build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_rtl_hashes.txt`。
- TB/脚本/文档哈希与全部日志哈希：`build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_final_evidence_sha256.txt`（由 `sha256sum` 生成，含作废过程日志单列）。
- 证据索引：`build/spikformer/vision_ref_20261031/sim/`（本任务新文件均带 imgnative 后缀；未覆盖 img-p1 的 p1_* 日志）。

### 10.4 定位过程（作废，仅留痕；不得当 PASS 引用）

| 轮次 | 日志 | 失效原因 |
|---|---|---|
| r1pref | tb_sps_native_imgnative_r1prefix_20261031.log（44360 fails）| 旧 ctrl（pre-e6008ad6）+ 旧 tc2=4 向量喂 TC2=8 RTL，双因叠加 |
| r2 | tb_sps_native_imgnative_r2_20261031.log（44793 fails）| e6008ad6 + 旧 tc2=4 向量；向量/RTL TC2 不配对 |
| r3 | tb_sps_native_imgnative_r3_20261031.log（1142 fails）| e6008ad6 的 flush 抢占 C_C2ROWS bug（bug D）仍在；向量已对 |
| 对照旧版 | tb_sps_native_tc2_4_control_imgnative_20261031.log（512 fails）| 仅用于排除 tile 顺序的对照，旧 ctrl 下作废 |

### 10.5 flush 结构断言 bug 修法（TB 侧，非 RTL）

问题：旧断言写 `st==9 && o_b0_we && p2fl==0`。`o_b0_we` 是寄存器输出（C_C2W 内 `<=` 置 1），到下一拍才为 1；而 `p2fl` 在同一拍已从 0 变 1，两者同拍窗口永远错开 → flush_n 恒 0（r3 的 "flush total=0" 是 TB 断言 bug，不是 RTL 没 flush）。
最终版断言（r4 使用）：**检测进入 C_C2W 状态的那一拍**（`st==9 && prev_st!=9`，prev_st 用非阻塞采样上拍状态），在该拍检查 `p2cnt==ROWW` 与锁存行号 `(p2_g,p2_y)==(flush_n/H1, flush_n%H1)`，每次 flush 只计一次；总数断言 768=G2*H1。
末行/pad-flush 断言同理：不在 C_C2FL 首拍查 `beat_pushed`（末拍 in-flight 时会看到 12287/p2cnt=15 的正常瞬态），改为 **C_DRAIN 入口**（`st==11 && prev_st!=11`）检查 `beat_pushed==12288 && p2cnt==0`（无 pad 残留）。
r4 实测：capture=12288、flush=768、C_DRAIN 入口 clean，全部通过。注意 TB 用层级只读探针引用了 ctrl 的 p2_g/p2_y/p2cnt/p2fl/beat_pushed 与状态枚举（C_C2W=9、C_DRAIN=11，ctrl 728337f1 下核对过）；换 ctrl 版本需重核这些名字/枚举。

### 10.6 交接与边界

- 本任务只读 RTL：未修改任何 rtl/、未跑 Vivado、未访问板卡、未覆盖 frozen 文件或他人日志。
- native 整链 TB（tb_vis_native_chain.v + compile_native_chain.do + TODO 清单）按 lead 分工交 img-p1 接手（交接 17:10，交接 hash 2f57fcf9…；之后由 owner 修改，本会话不再并发改该 TB，也不把其后续 hash 计入本门证据）。
- 若 sps RTL 再次变更（hash != 728337f1），native SPS 门需重跑并以新日志替换 §10.1；旧证据保留。

### 10.7 OOC standby check（只读旁证，非本门 PASS；供 P3 负责人核对 probe 与 native 门口径）

按 lead 分工做的待命核对（未跑 Vivado、未改 probe/RTL）：
- OOC 入口 `scripts/spikformer/synth_ooc_vis_native.tcl` 的整顶 probe `rtl/probe/vis_native_probe.sv` 用 `build/spikformer/vis_native/generated/spikformer_config.vh` 绑定参数；ModelSim 展开（Xilinx LUT1 用 sim-only stub，仅仿真）实测：
  - SPS：H=32、W_IN=32、C0=3、C1=96、C2=384、T=4、LANES=8、**TC2=8**、WT_W=8、ACC_W=21、RW=4；
  - block：E=384、HEADS=12、HEAD_DIM=32、HIDDEN=1536、T=4、TOKENS=64、TL=8、TC=8、RW=4、PSUM_SLICE=32、SUM_W=6；
  - 端口宽度：pix=24b、blk/sps 权重=64b、o_sums=384×32b。
  与本文 native SPS 门（r4）完全一致；config header 的 SPIK_SPS_CONV1_BEATS=324、CONV2_BEATS=41472 也与主机流拍数一致。
- OOC 源清单（19 文件）与 native 门 RTL 集合一致（ctrl 728337f1 等；未包含 legacy spiking_sps_top，spiking_maxpool2x2 无实例引用）。
- **注意（P3 交接）**：`rtl/probe/sps_native_probe.sv` 实例化的是**旧 legacy `spiking_sps_top`**，不是 native 门的 `spiking_sps_ref_top`；旧入口 `scripts/synth_ooc_probe.tcl` 也不含新 control 顶层。native OOC 必须用 `synth_ooc_vis_native.tcl`（整顶）或 `rtl/probe/sps_ref_probe.sv`（SPS-only 降级），否则测的是旧架构。
- 证据：`sim_work/vis_sps/tb_probe_paramcheck_imgnative.v`、`compile_ooccheck_imgnative.do`、`xilinx_lut1_stub_sim.v`（sim-only）；日志 `build/spikformer/vision_ref_20261031/sim/probe_paramcheck_imgnative_20261031.log`（PROBE_PARAM/PROBE_PORT/PROBE_CHECK_DONE，Errors 0）、`compile_ooccheck_imgnative_20261031.log`。

## 11. 收尾确认（给快照 evidence 的核对清单，2026-10-31 17:15）

以下六项已在本文件写明，快照 evidence 可直接按本文件校对：

1. **r1pref/r2/r3 作废、r4 有效**：见 §4 结果时间线与 §10.4；r4 = ALL PASS（3072 beats，capture=12288，flush=768，structural ok）。
2. **旧 vectors/sps_native = tc2=4，不配对**：见 §3 与 §10.3（meta "tc2":4；仅 w_sched 9ce5b23b… 与 TC2=8 RTL 不符，其余文件 sha256 相同）。
3. **正式向量 = sps_native_tc2_8**：见 §10.3（w_sched 0f9e8f69…，meta "tc2":8；生成命令见 §8）。
4. **flush 断言修法**：见 §10.5（旧写法 o_b0_we 寄存器晚一拍、与 p2fl 同拍窗口错开 → 恒 0；最终版检测进入 C_C2W 那一拍 st==9 && prev_st!=9；末行改 C_DRAIN 入口检查 beat_pushed==12288 && p2cnt==0）。
5. **OOC probe 参数核对**：见 §10.7（vis_native_probe 展开 SPS H32/W32/C0-3/C1-96/C2-384/T4/TC2=8、block E384/H12/HD32/1536/T4/TOK64/TL8/TC8/RW4/PSUM32/SUM6，sums 384×32b，与 native 门一致；警告 sps_native_probe.sv 是旧 legacy SPS，native OOC 用 synth_ooc_vis_native.tcl）。
6. **可复核命令/日志/哈希**：见 §10.1–10.3 与 manifest `build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_final_evidence_sha256.txt`。

快照应交 evidence 的小文件（本任务产出，均为只读复制对象）：
- `build/spikformer/vision_ref_20261031/sim/tb_sps_native_gate_result_imgnative_20261031.json`
- `build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_final_evidence_sha256.txt`
- `build/spikformer/vision_ref_20261031/sim/tb_sps_native_imgnative_r4_rtl_hashes.txt`
- 本文：`docs/VISION_NATIVE_SIM_PROGRESS_20261031.md`

RTL 边界重申：本任务未修改任何 rtl/ 文件；native SPS 门的 RTL 身份 = ctrl 728337f1… 及其余哈希见 r4_rtl_hashes.txt。