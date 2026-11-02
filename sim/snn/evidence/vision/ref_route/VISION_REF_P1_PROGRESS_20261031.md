# 图像线 P1 进度：参考式 SPS small 门 PASS 与 native 边界修复（2026-10-31）

状态（本文件只记录实测，计划不写成 PASS）：
- **P1 small 门（H8/C1-16/C2-32/T2，16 beats）PASS**：16/16 token beats 全等 + last/done/watchdog，另加结构断言。
- **P1 T=4 同几何（H8/C1-16/C2-32/T4）PASS**。
- **P1 H32/T4 native 边界代理（H32/W32/C1-16/C2-32/T4/TC2=4）PASS**，含 f1/pool1/array 逐点 golden 对照。
- **P1 native 全尺寸（H32/C1-96/C2-384/T4/TC2=8，3072 beats）PASS**（img-native 执行，同一 ctrl 728337f1…；日志 `tb_sps_native_imgnative_r4_20261031.log`，独立 Python 复核 3072/3072 全等、hamming=0）。
- P2 native 整链 / P3 OOC / B 轮 / P4 平台与板测：**not_run**。
- 只读参考工程 `D:/dl_v0.1/FPGA_SpikingTransformer-main` 未改动；未跑 Vivado；未访问板卡。

同一版本 ctrl（**sha256 728337f10bff4e7d3f3c6a2ec0ea060446b5bc3a07e72182a4ecfeae22c02244**）跑出以下 small/边界代理四份 PASS 日志（native 全尺寸 PASS 见 §1/§5）。

## 1. 门状态与证据索引

| 项目 | 状态 | 证据（build/spikformer/vision_ref_20261031/sim/） |
|---|---|---|
| small T=2 门（16 beats） | PASS | `p1_r7_small_t2.log`：`ALL PASS tb_sps_ref_small (16 beats; capture=64 flush=16 structural ok)` |
| T=4 同几何（16 beats） | PASS | `p1_r7_small_t4.log`：`ALL PASS tb_sps_ref_small_t4 (16 beats; capture=64 flush=16 structural ok)` |
| H32/T4 边界代理（256 beats，TC2=4） | PASS | `p1_r7_h32_t4.log`：`ALL PASS tb_sps_ref_small_h32_t4 (256 beats; capture=1024 flush=64 structural ok)` |
| H32 边界逐点 golden（TC2=4） | PASS | f1 bank 512/512、pool1 bank 128/128、array tile 结果 4096/4096（dump 与分析见 §4） |
| H32/T4 边界代理（256 beats，TC2=8） | PASS | `p1_r8_h32_t4_tc2_8.log`：`ALL PASS tb_sps_ref_small_h32_t4_tc2_8 (256 beats; capture=1024 flush=64 structural ok)` |
| H32 边界逐点 golden（TC2=8） | PASS | f1 bank 512/512、pool1 bank 128/128、array tile 结果 2048/2048（NT2=2） |
| 编译（三 TB 同库） | PASS | `p1_r7_compile.log`，Errors: 0，Warnings: 0（vsim 阶段） |
| native 全尺寸 3072 beats（TC2=8） | PASS | `tb_sps_native_imgnative_r4_20261031.log`：`ALL PASS tb_sps_native (3072 beats; capture=12288 flush=768 structural ok)`，5,573,066 cycles；独立 Python 复核 `tb_sps_native_imgnative_r4_diff_summary.txt`（3072/3072 exact，hamming=0） |
| 长反压/连续两图/供权气泡 | not_run | P1 任务书第 3 条，本批未做 |
| OOC 综合/实现、P2/P3/P4/B 轮 | not_run | — |

## 2. 失败基线（修复前）

| 证据 | 内容 |
|---|---|
| `tb_sps_ref_small_acc.log`（16:09，旧 ctrl sha256 74452587…） | small FAIL：**53/256 nibble**；BEATFAIL 分布见日志，beat0/beat4 恰好通过 |
| `p1_r6_h32_t4_dbg.log`（H32 代理，中间版 ctrl e6008ad6…） | **FAILURES=121/8192**，chkfail=0（行对齐/flush 结构与计数全对） |
| 最小观测（small 基线） | CAP=64、PSUMBEAT=64、C2W_ENTRY=16；`C2W_ENTRY g=1 fpr=0 p2row0=xxxxxx03`，逐 lane 解码该 word0=3 属 **g0 的 row3**（g1r0 应为 0），即"行内容对、行地址错一位" |

由此得到四条独立根因（A/B/C 修 small 53 nibble；D 由 H32/T4 边界回归发现，是 native 尺寸的隐患）。

## 3. 四条根因与修复（before → after）

**A. o_psum_beat_ready 寄存器吞拍**
- 现象：ready 为寄存器输出且在 C_C2TILE/C_C2ROWS/C_C2WIN/C_C2FL 恒 1，不看 p2cnt；capture 却要求 `p2cnt<ROWW`。p2row 满而 flush 未起的周期里 beat 被 ready 消费但不落盘。
- 修复：改组合输出 `((st==C_C2TILE)||(st==C_C2ROWS)||(st==C_C2WIN)||(st==C_C2FL)) && (p2cnt<ROWW)`；capture 与 fire 同表达式。
- 观测：CAP=64 与 PSUMBEAT=64 一一对应（修复前基线存在 beat 丢失窗口）；after 见 `p1_r7_small_t2.log` 的 `capture=64`。

**B. 全局 flush trigger 被同拍 case 状态赋值覆盖**
- 现象：旧代码在 case 前 `if(...) st<=C_C2W`，同拍 `case(st)` 又对 st 赋值（如 C_C2WIN phase2 的 `c_g2++/c_fpr<=0/st<=C_C2TILE`），后者胜出 → flush 被跳过，而计数器已推进。
- 修复：删除全局 trigger；`flush_now` 在 C_C2WIN/C_C2TILE 分支顶部用 if/else 抢占；C_C2TILE 同拍把 `o_w_ready/o_arr_w_valid` 用 `!flush_now` 关门，避免权重 beat 被 array 消费而 cnt 未加。
- 观测：before `C2W_ENTRY` 序列 g0 fpr0..2 → g1 fpr0（第 4 行写进新组）；after 每个 g 组 4 行，H32 代理 flush=64 且行序断言全过。

**C. flush 地址必须用捕获时锁存的行身份，不能用运行计数器**
- 现象：每个 g 组**最后一行**的最后一个 psum beat 比 tile/g2 推进晚 2~5 拍（array→psum RMW→LIF 延迟），触发 flush 时 `c_g2` 已 +1、`c_fpr` 已被清 0 → 该行被写到新组 row0，后续整组行错位。这是 53 nibble 的主因。
- 修复：首拍捕获时锁存 `p2_g<=i_psum_beat_grp`、`p2_y<=i_psum_beat_pos/ROWW`；C_C2W 地址改为 `(p2_g*H1+p2_y)*8+p2fl`。`c_fpr` 仅保留 debug 计数。
- 观测：H32 代理行序断言（第 n 次 flush 的 (p2_g,p2_y) 必须 ==(n/H1,n%H1)）chkfail=0；small 16/16。

**D. flush 抢占 C_C2ROWS 污染 b1 行载入的 BRAM 读流水（native 尺寸隐患）**
- 现象（H32/T4 代理首跑）：token 121/8192 错但 chkfail=0。dump 全 f1 bank **512/512 全等**、pool1 bank **128/128 全等**、array tile 结果**只有每个 g 组最后一个输入 tile 错（348/4096）**，tile0..NT2-2 全等。
- 根因：flush 8 拍内每拍默认 `o_b1_raddr<=0`，把 feature BRAM 的 `raddr_q` 走到 0；被抢占的 C_C2ROWS 恢复后 ph2 采到 `mem[0]` 而非所发行 word。psum beat 只在最后输入 tile finalize（LIF），所以 flush 恰只在最后 tile 期间发生 → 只污染最后 tile 的窗口数据。H8 的 H1=4 行载入次数少，侥幸未撞；H1=16 的 native 必撞。
- 修复：`flush_now` 收窄为 `{C_C2WIN,C_C2TILE}`，C_C2ROWS 不被抢占（行长 ≤12 word；p2row 满时 beat 由 ready=0 反压，不丢）。
- 观测：修复后 H32 代理 256/256 beats PASS，array 结果 **4096/4096 全等**；small T=2/T=4 仍 PASS。

附带：p2row 复位清 0（消 X）；C_C2TILE 的 flush 同拍关权重握手。

## 4. 改动清单与身份

| 文件 | sha256 | 说明 |
|---|---|---|
| `rtl/ops/spiking/sps/spiking_sps_ref_ctrl.sv` | `728337f10bff4e7d3f3c6a2ec0ea060446b5bc3a07e72182a4ecfeae22c02244` | A/B/C/D 修复；改动前备份 `sim_work/vis_sps/backup_sps_ref_ctrl_p1_before_1641.sv`（`74452587bab70934acbbb642f2399f59d17021eec64ec20f169579f8a3147f5d`） |
| `verification/sim/spikformer/tb_sps_ref_small.v` | `f15cd29cfd934110d5213e03e10b8533ad5e966c956c04ef666fb52a92478224` | 主门 TB + 新增结构断言 |
| `verification/sim/spikformer/tb_sps_ref_small_t4.v`（新） | `65aa6b470bc1b09dad9c4c5dba47c3576ceb6ecb0351eccb061fe8a5b2aa67c6` | T=4 同几何 TB（T=4/RW4/128b token） |
| `verification/sim/spikformer/tb_sps_ref_small_h32_t4.v`（新） | `5ea04930937b7429474fda6a764d45ab94fd1da7fd81618f4f25ccb395ba4173` | H32/T4 边界代理 TB + f1/pool1/array dump（TC2=4） |
| `verification/sim/spikformer/tb_sps_ref_small_h32_t4_tc2_8.v`（新） | `ffe162dc209e1c3bd37119af4d5d29848cd99ec58e52cd5df0a69bdbb589be46` | 同上的 TC2=8 版本（覆盖 native 的 array TC2=8 路径、NT2=2、SUBT=1） |
| `build/.../vectors/sps_small_t4_check/` | meta `5ae4707e…` | `gen_sps_param.py --h 8 --w 8 --c1 16 --c2 32 --t 4 --tc2 4 --seed 11` |
| `build/.../vectors/sps_h32_t4_check/` | meta `caa99f9a…` | `--h 32 --w 32 --t 4 --tc2 4 --seed 11` |
| `build/.../vectors/sps_h32_t4_tc2_8_check/` | meta `a82771fc…` | `--h 32 --w 32 --t 4 --tc2 8 --seed 11`（TC2=8 代理） |

未改动：frozen NLP RTL（block_top/lif/attention/mlp/linear/embed/reduce/nlp_top/sps_top 等）、tools 语义、平台工程、参考工程、Vivado/板卡。

## 5. PASS 日志（run 命令见 §6）

| 日志 | sha256 | 关键行 |
|---|---|---|
| `p1_r7_small_t2.log` | `57eb962f6fa9239ec8015785955b7733da75b72ba35669e77b0ba0c0477b737c` | `ALL PASS tb_sps_ref_small (16 beats; capture=64 flush=16 structural ok)`；Errors: 0 |
| `p1_r7_small_t4.log` | `269556ee81435c5a5480d0941f5cc65b5845971bbabddcd912e7299a823b6261` | `ALL PASS tb_sps_ref_small_t4 (16 beats; capture=64 flush=16 structural ok)`；Errors: 0 |
| `p1_r7_h32_t4.log` | `dd3771c4be960c8378d01436a1fb5dc7f329617793d836dd2a4c0613bc4cee58` | `ALL PASS tb_sps_ref_small_h32_t4 (256 beats; capture=1024 flush=64 structural ok)`；Errors: 0 |
| `p1_r7_compile.log` | `5bc6fc42…` | 编译通过，Warnings 仅 adder_tree 越界提示（未使用实例） |
| `tb_sps_native_imgnative_r4_20261031.log`（native 正式门，img-native） | `e7a14f8c…` | `ALL PASS tb_sps_native (3072 beats; capture=12288 flush=768 structural ok)`；Errors 0；2:00 wall |
| `tb_sps_native_imgnative_r4_diff_summary.txt` | `65e5684b…` | `exact beats=3072/3072`、`total hamming nibbles and bits=0`、`nibbles outside {0,1}=0` |
| `tb_sps_native_imgnative_r4_rtl_hashes.txt` | `346f25cd…` | 7 个 sps RTL 哈希，含 ctrl `728337f1…` |
| `p1_r8_h32_t4_tc2_8.log` | `c281c84efaac8dbae548a9a57fca5eb312b3aa51baa0c17f53cd45afdd17ae59` | `ALL PASS tb_sps_ref_small_h32_t4_tc2_8 (256 beats; capture=1024 flush=64 structural ok)` |
| `p1_h32_arrres_fixed.txt`（4096 行） | `acb57afa…` | H32 代理 array tile 结果，逐 lane/t 与 golden 全等 |
| `p1_h32tc28_arrres.txt`（2048 行） | — | TC2=8 代理 array tile 结果，逐 lane/t 与 golden 全等 |
| 分析脚本 `analyze_h32_tc2_8.py` | — | TC2=8 代理的 f1/pooled/array 逐点 golden 对照 |
| `p1_h32_b0dump_fixed.hex` / `p1_h32_b1dump_fixed.hex` | — | f1 bank 512 words / pooled bank 128 words，与 golden 全等 |
| 分析脚本 | — | `sim_work/vis_sps/analyze_h32.py`、`analyze_h32c.py`、`analyze_h32d.py`、`analyze_h32e.py` |

修复前对照：`tb_sps_ref_small_acc.log`（53 nibble，sha `a911729e…`）、`p1_r6_h32_t4_dbg.log`（121/8192，sha `efd68b3f…`）。

## 6. TB 新增结构断言（独立于 golden，失败计入 chkfail）

1. 每个被捕获的 psum beat，`(grp,pos)` 必须严格等于第 cap_n 个调度位置（`pos=cap_n%(H1*W1), grp=cap_n/(H1*W1)`），无缺口 → 可检测丢拍/重拍。
2. 每次 flush 入口 `p2cnt==ROWW`，且第 n 次 flush 的 `(p2_g,p2_y)` 必须等于 row-major 期望 `(n/H1, n%H1)`（small 4/4，H32 16/16）。
3. 总数断言：`capture==(H1*W1)*G2`（small 64，H32 1024）、`flush==G2*H1`（small 16，H32 64）。
4. token 无 X；last 只在最后一拍；done 等待 + 3M 周期 watchdog。日志中出现 `CHKFAIL` 即失败。

## 7. 复现命令（Windows bash 口径，工作区 `D:/dl_snn`）

```bash
cd /d/dl_snn/sim_work/vis_sps
# 编译（small 主 TB + sps RTL；T4/H32 代理 TB 用 vlog 单编）
/d/modelsim/win64/vsim -c -do 'do compile_ref_small.do; quit -f' -l /d/dl_snn/build/spikformer/vision_ref_20261031/sim/p1_r7_compile.log
/d/modelsim/win64/vlog -sv -work work /d/dl_snn/verification/sim/spikformer/tb_sps_ref_small_t4.v
/d/modelsim/win64/vlog -sv -work work /d/dl_snn/verification/sim/spikformer/tb_sps_ref_small_h32_t4.v
/d/modelsim/win64/vlog -sv -work work /d/dl_snn/verification/sim/spikformer/tb_sps_ref_small_h32_t4_tc2_8.v
# 三份回归（+acc 供 TB 层级观测）
/d/modelsim/win64/vsim -c -voptargs=+acc -work work tb_sps_ref_small      -l /d/dl_snn/build/spikformer/vision_ref_20261031/sim/p1_r7_small_t2.log -do 'run 100ms; quit -f'
/d/modelsim/win64/vsim -c -voptargs=+acc -work work tb_sps_ref_small_t4   -l /d/dl_snn/build/spikformer/vision_ref_20261031/sim/p1_r7_small_t4.log -do 'run 200ms; quit -f'
/d/modelsim/win64/vsim -c -voptargs=+acc -work work tb_sps_ref_small_h32_t4 -l /d/dl_snn/build/spikformer/vision_ref_20261031/sim/p1_r7_h32_t4.log -do 'run 400ms; quit -f'
/d/modelsim/win64/vsim -c -voptargs=+acc -work work tb_sps_ref_small_h32_t4_tc2_8 -l /d/dl_snn/build/spikformer/vision_ref_20261031/sim/p1_r8_h32_t4_tc2_8.log -do 'run 400ms; quit -f'
# golden 向量重生成（不覆盖旧证据目录）
E:/miniconda/python.exe /d/dl_snn/scripts/spikformer/gen_sps_param.py --h 8  --w 8  --c1 16 --c2 32 --t 4 --lanes 8 --tc2 4 --seed 11 --out-dir /d/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_small_t4_check
E:/miniconda/python.exe /d/dl_snn/scripts/spikformer/gen_sps_param.py --h 32 --w 32 --c1 16 --c2 32 --t 4 --lanes 8 --tc2 4 --seed 11 --out-dir /d/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_h32_t4_check
E:/miniconda/python.exe /d/dl_snn/scripts/spikformer/gen_sps_param.py --h 32 --w 32 --c1 16 --c2 32 --t 4 --lanes 8 --tc2 8 --seed 11 --out-dir /d/dl_snn/build/spikformer/vision_ref_20261031/vectors/sps_h32_t4_tc2_8_check
```

## 8. 未测项、风险与交接

- **native 全尺寸已 PASS**（img-native，同一 ctrl `728337f1…`，TC2=8 RTL + tc2=8 重生成向量；3072/3072 beats + 独立 Python 全等）。我的 H32 代理另有 **TC2=4 与 TC2=8 两版**逐点 golden PASS，但代理规模为 C1-16/C2-32（NT2=2、G2=4），不等于 native 边界密度证据；native 的 PASS 以 r4 日志为准。
- native 的 TB、编译脚本与 tc2=8 向量重生成由 img-native 维护（`compile_ref_native_imgnative_r4_20261031.log`；向量口径：RTL TC2=8 必须配 `--tc2 8` 的 w_sched，否则权重 tile 排列错位——r1pref/r2/r3 的 early FAIL 即由此/旧 ctrl 造成，作废仅留定位记录）。
- 长反压、连续两图、DMA 供权气泡、T=4 多通道边界之外的协议压力：not_run。
- native SPS 综合/OOC/时序、P2 整链、P3 物理门、P4 平台/板测、B 轮：not_run；不得用本文件的 small/代理 PASS 替代。
- 观测入口（native 调试用）：`u_ctrl.p2_g/p2_y/p2cnt/p2fl/c_ret/o_dbg_c2w/flush_now` 可层级引用；`c_fpr` 仅 debug 计数，不再参与寻址；`o_psum_beat_ready` 已为组合信号。
