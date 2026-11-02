# 算子库（rtl/ops/spiking）— 参数 / 数值语义 / 已验证范围

清单权威：`rtl/operator_library.json`（schema `e1-operator-library-1`，variant `spikformer-snn-1`）；
构建源绑定：`rtl/ops/spiking/source_manifest.json`（28 份，v`20261031.1`）；上游映射：`rtl/ops/spiking/UPSTREAM_MAP.json`。
本文件只描述**已入快照**的 RTL，不代表全部已流片验证。**NLP 路径 = embed + block（attention/linear/lif/mlp/reduce）+ wrapper**；
SPS 是图像前端的 patch embedding，**NLP 不实例化**。

## 1. 总表

| 引擎 | 目录 | 顶层 | 关键参数 | 功能 | 已验证范围（证据） |
|---|---|---|---|---|---|
| sps（native） | `sps/` | `spiking_sps_top` | H/W_IN/C0/C1/C2/T/LANES/TC2/WT_W/ACC_W/RW | conv3×3(C0→C1)+LIF+2×maxpool+conv3×3(C1→C2)+LIF → token | **8×8 小图回放 PASS（16 beats 逐 lane）**：`tb_sps_replay.v`；native 32×32 整链 **not_run** |
| sps（参考式，图像线新增） | `sps/` | `spiking_sps_ref_top` | H/W_IN/C0/C1/C2/T/LANES/TC2/WT_W/ACC_W/RW | 参考工程路线的 patch embed（rgb_encoder+eyeriss_array+psum+token_adapter） | **P1 small 门 PASS（主线，16/16 beats，2026-10-31 16:44/16:46 两配置）**；快照 TB 为 16:36 副本、不重跑；历史 FAIL 53/256 nibble 为修复前基线；**P1 正式 native 门 fail（1,142/98,304 nibbles）**、native 整链 **not_run** |
| linear | `linear/` | `spiking_systolic_array` | ROWS/COLS/T/SPIKE_BITS/W/PSUM_SLICE | 2-bit 脉冲脉动阵列（QKV/proj/fc1/fc2） | block 回放 **PASS**（`tb_block_replay.v`）；NLP 整链 **PASS**（`replay_e2e_5stage_r2_noforce.json`） |
| lif | `lif/` | `spiking_lif_group` | T/BITS（cell 另 BITS） | LIF：tau=2、算术移位、硬复位 | **PASS**（`tb_spiking_lif_group.v`；整数参考 8/8 PASS） |
| attention_qk | `attention/` | `spiking_attention_pe` | HEAD_DIM=32/T/SUM_W=6/OUT_W=T*SUM_W | 每 T 位与后 popcount（不除 head_dim） | popcount 全 1=32 **PASS**（`tb_spiking_popcount.v`）；整链 PASS |
| attention_av | `attention/` | `spiking_linemac_pe` | HEAD_DIM/T/SUM_W/ACC_W/TOKENS | 未归一 AV（每 token 求和，uint11，max 1024） | 整链 PASS（`tb_nlp_e2e.v`，0 mismatch） |
| mlp | `mlp/` | `spiking_mlp_controller` | C/HIDDEN/T/RW/W/PSUM_SLICE | fc1+LIF，fc2+LIF，残差加回 | **PASS**（`tb_mlp_replay.v` 4 beats；block 整链 PASS） |
| embed | `embed/` | `spiking_embed_top` | V/N/E/T/LANES/ACC_W | token 表 + 位置表 + LIF（`delta=etab+ptab`，t 恒定） | **PASS**（`tb_embed_replay.v` 16 beats） |
| reduce | `reduce/` | `spiking_head_reduce` | CH/TOKENS/T/HL/SUM_W | PL 每通道 T*N 求和（归一在 PS 或 folded） | **PASS**（`tb_head_replay.v` 32 通道） |
| control（block） | `control/` | `spiking_block_top` | E/HEADS/HEAD_DIM/HIDDEN/T/TOKENS/DEPTH… | 单 Transformer block：QKV+LIF→QK popcount→AV→proj+残差→MLP+残差（5 级 MAC 流水） | **PASS**（`tb_block_replay.v` 16 beats；NLP 整链 PASS，0 mismatch） |
| control（sequencer） | `control/` | `spikformer_block_sequencer` | DEPTH/CH | 权重基址切换、bank ping-pong、末 block 才归约、二次启动 | **PASS**（`tb_sequencer.v`） |
| control（NLP top） | `control/` | `spikformer_nlp_top` | E/HEADS/HEAD_DIM/HIDDEN/T/TOKENS/V/DEPTH/… | NLP 目标顶层：embed → DEPTH×block → head reduce | **PASS**（`tb_nlp_e2e.v`，7,084,772 cycles，0 mismatch；真实权重 10,627,157 cycles，0 mismatch） |
| control（图像 top） | `control/` | `spikformer_vis_top`（legacy）/ `spikformer_vis_native_top`（参考式） | H/W/C0/C1/C2/T/E/HEADS/HIDDEN… | 图像整链 SPS→block→sums | legacy 行为级整链 **PASS（32 channels）**；参考式 native 整链 **not_run** |
| wrappers | `wrappers/` | `spikformer_platform_wrapper` | ID/VERSION/T/DEPTH/CH/HEADS… | CSR（ID/VERSION/CONTROL/STATUS/ERROR/BASE/STRIDE/DESC/TIMEOUT/ACK/TAG）+ DDR 窗口 + 生命周期 | **PASS**（`tb_wrapper.v`；done 为安全系结，待子引擎对接）；平台 r7 上板 UIO 走通（主会话记录） |
| sps 子模块 | `sps/` | `spiking_adder_tree` `spiking_eyeriss_pe` `spiking_eyeriss_array` `spiking_feature_bram` `spiking_maxpool2x2` `spiking_rgb_encoder` `spiking_sps_psum` `spiking_sps_token_adapter` `spiking_sps_ref_ctrl` | 见各文件 `parameter` | 参考式 SPS 的乘法树/PE/阵列/帧存/池化/编码/psum/skid 适配/控制器 | 参考式 small **P1 PASS（16/16 beats，2026-10-31 16:44）**；修复前基线 FAIL 53/256 nibble 与逐级定位见 `docs/VISION_REFERENCE_ROUTE_RESULT_20261031.md` |

> 每个算子的参数以 RTL `module` 头部 `parameter` 为准；本表只列常用项。
> `source_manifest.json` 的 28 份源包含 9 份仅图像线使用的 SPS/vis 模块；NLP 目标顶层只例化 `spiking_embed_top` + `spiking_block_top`。

## 2. 数值语义（与 `numeric_contract.json` 一致）

| 项 | 语义 |
|---|---|
| LIF | `mem = (pre + delta) >>> 1`（算术右移，tau=2）；`pre+delta` **先符号扩展到 33 bit** 再移位（无 32-bit 回绕）；发放条件 `valid && mem>=0 && mem>=thr`；**硬复位 0**；`t=0` 初值 0；`thr>=1` |
| QK | 每 t 按 `head_dim` 位 AND 后 popcount，**不除 head_dim**；`SUM_W=clog2(head_dim+1)=6`，全 1=32，求和域 6 bit |
| AV | 未归一：对 TOKENS 个源 token 求和，uint11，`max = TOKENS*32`（seq32 → 1024） |
| 残差 | 位宽 `ceil(log2(max+1))`，RW 分配 4 bit（min 需求单列）；SPS 上界 2、block 输入 2+2b、Attention 后 3+2b、MLP 后 4+2b、D 个 block 后 2+2D |
| Head | `reduction=sum`、`norm=ps_div`：PL 输出每通道 uint32 和 → PS 除以 `T*N` → **FP32 Linear + argmax**（唯一 FP32 阶段） |
| 量化 | int8 权重、per-tensor 对称 scale（maxabs/127）、int32 bias（累加域）、thresholds 显式入 params/manifest；BN 状态显式（本案例 absent） |
| 累加域 | 32-bit 累加器、`PSUM_SLICE=32`、`lif_add_bits=33`；INT32 边界检查在导出/编译期做（越界报错，不静默截断） |

## 3. 结构约束（config 层拒绝，不静默回退）

- `B=1`；`T ∈ {1,2,4}`；`head_dim=32`；`E = heads*head_dim` 且 `E%32==0`；`mlp_hidden%16==0`；`depth>=1`。
- NLP：`seq_len ∈ [1,256]`，`frontend.kind="embed_spike"`，`mask="none"`。
- 图像：方形且 `%4==0`；`C1` 默认 `E/4`、8 的倍数；矩形输入直接拒绝（池化几何）。

## 4. 上游映射

`rtl/ops/spiking/UPSTREAM_MAP.json` 记录与 `D:/dl_v0.1/FPGA_SpikingTransformer-main` 的逐文件哈希与差异
（上游**不随包**）。自查：`scripts/spikformer/hash_upstream.py`（需上游树存在）。

## 5. 状态口径提醒

- **NLP 功能 RTL 的冻结哈希**（18 文件）中 16 个与 `rtl_freeze.json` 一致；2 个 SPS 文件已被图像线并行改动，**NLP datapath 不受影响**，详见 `docs/PORTABILITY.md` §4。
- 参考式 SPS 相关算子（`spiking_sps_ref_*` 等）**P1 small 已过门（16/16 beats，2026-10-31 16:44）**，但其 native 仿真/整链/OOC/平台/板测 **not_run**，**不得**宣称图像加速器已闭环。
- 本快照**没有**重跑任何 TB 或综合/实现；本文件的“已验证范围”全部指向**主线历史证据**（路径已标注），不冒充快照内新结果。
