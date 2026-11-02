# 数值合同（NLP Spikformer 整数变体 0.1.0）

本文件描述发布快照 D:/DL_FINAL/SNN 的**数值语义权威口径**，面向复核者：每个条目都注明来源文件。
快照内证据副本在 `evidence/nlp/`（见 `evidence/INDEX.md`）。

## 0. 范围与来源

**范围声明**：本变体是**命名的 Spikformer NLP 整数变体，不是原版 Spikformer 的同构实现**：
无 softmax、AV 不归一、无 BN/LayerNorm、仅新增 proj 与 fc2 两个分支输出 LIF。
因此**禁止**把本变体与上游 Spikformer 或任何 FP32 参考实现做逐位比较。

| 权威来源 | 位置 | 说明 |
|---|---|---|
| 冻结合同 contract.json | `build/spikformer/nlp_48h_20261029_1915/contract.json`（快照：`evidence/nlp/contract.json`） | 数值语义 + stream ABI 权威；2026-10-29 19:30 冻结 |
| manifest 绑定的 numeric_contract.json | `.../final/nlp_real_5stage_r4_holdfix/numeric_contract.json`（快照：`evidence/nlp/final/numeric_contract.json`） | 由 manifest 绑定的数值合同摘要 |
| 后端说明 | `docs/SPIKFORMER_BACKEND.md` §2（源文件；快照内 docs/OPERATORS.md 为算子清单） | LIF/QK/AV/残差/head/量化文字口径 |
| 整数参考实现 | `tools/backends/spikformer/integer_ref.py` | replay 线的 golden truth（纯 NumPy 整数数据通路，唯一 FP32 阶段是 PS head） |
| 数学短测证据 | `build/.../reports/math_tests_m1.{log,json}`（快照：`evidence/nlp/math_tests_m1.*`） | 8/8 PASS：torch 量化前向 vs integer_ref 全层整数 trace 一致 |

模型结构（contract.json `model` / numeric_contract.json `structure`）：
E=128、HEADS=4、HEAD_DIM=32、seq_len(TOKENS)=32、MLP_HIDDEN=512、DEPTH=2、T=4、BATCH=1、V=4000、2 类；
target xczu15eg-ffvb1156-2-i，Vivado 2021.1，5.000 ns / 200 MHz，**无时序例外、不降频**。

## 1. 数据流（逐层）

1. **embed**：`delta[n][e] = etab[id[n]][e] + ptab[pos=n][e]`（同 delta 在 T 个时间步重复）→ LIF(thr_e) → spike[B,T,N,E] ∈ {0,1}。
2. **block（每个 block 各自的阈值）**：
   - Q/K/V = linear(input_spikes) + int32 bias → LIF(q/k/v)；
   - 每 head、每 t：`score[i][j] = popcount(Q[i] & K[j])`（head_dim 位按位 AND，**不除 head_dim**，max 32）；
   - `AV[i][a] = Σ_j score[i][j] · V[j][a]`（j 遍历 TOKENS，**未归一**，max 32×32=1024）；
   - proj：AV → linear+int32 bias → LIF(proj) → **残差加**；
   - MLP：fc1(residual) → linear+int32 bias → LIF(fc1)；fc2(fc1_spike) → linear+int32 bias → LIF(fc2) → **残差加**。
3. **head**：PL 每通道对 T*N 个样本求 uint32 整数和 → PS 除以 T*N（FP32）→ FP32 Linear + bias → argmax。

来源：`contract.json` `dataflow`；`integer_ref.py` 模块 docstring / `run_sample`；`SPIKFORMER_BACKEND.md` §2。

## 2. LIF 语义

| 条目 | 口径 | 来源 |
|---|---|---|
| 加法位宽 | pre 与 delta **先符号扩展到 33 bit 再加**，绝不在 32 bit 回绕（`lif_add_bits=33`） | contract `numeric.lif.add_semantics`；numeric_contract `lif_add_bits` |
| 膜电位 | `V = floor((pre+delta)/2)`，即算术右移 1（tau=2） | contract `numeric.lif.mem`；integer_ref `lif_over_time` / `arith_shr1` |
| 发放条件 | `valid && mem >= 0 && mem >= thr`；thr 为 int32 且 **thr >= 1** | contract `numeric.lif.spike`/`thr_min`；integer_ref 断言 |
| 复位 | 发放后**硬复位 mem=0**（无软复位/无泄漏） | contract `numeric.lif.reset` |
| 初值 | 每个 token/通道在 t=0 时 pre=0；跨 token、跨 sample、跨 block 不泄漏、状态隔离 | contract `numeric.lif.init` |
| LIF 位置（本批 7 处） | embed、q、k、v、proj、fc1、fc2 | contract `numeric.lif_placements` |
| 本批禁止 | post_AV attn_lif、BN、softmax、LayerNorm | contract `numeric.lif_placements_forbidden_this_batch` |

补充（来自 integer_ref 实现）：`pre+delta` 用 int64 精确计算并断言落在 signed 33-bit 域（不静默回绕）；
`mem` 存 floor 值；`valid` 为可选的采样门控。数学短测覆盖 LIF 阈值边界与 floor、负累加不发放。

## 3. QK popcount

| 条目 | 口径 | 来源 |
|---|---|---|
| 计算 | 每 head、每 t：Q/K 的 head_dim 位按位 AND 后 popcount；**不除 head_dim** | contract `numeric.qk`；numeric_contract `qk_popcount` |
| 位宽 | `SUM_W = clog2(head_dim+1)`；head_dim=32 → **SUM_W=6**，all-ones=32 可表示 | contract `numeric.qk.sum_w_bits`；backend §1/§2 |
| attn 字 | `T*SUM_W` 位（T=4 → 24 bit） | `nlp/final/resolved_config.json` derived.attn_word=24 |
| 已知差异 | 开源 5b 截断是本项目已记录的差异，**不当 golden** | backend §1 |

## 4. AV（不归一）

| 条目 | 口径 | 来源 |
|---|---|---|
| 公式 | `AV[i][a] = Σ_j score[i][j] · V_spike[j][a]`（j 为 TOKENS，score 0..32，V_spike 0/1） | contract `numeric.av.formula` |
| 上界 | NLP 32 token：**max = 32×32 = 1024**，uint11 | contract `numeric.av.bits_required/max`；numeric_contract `av` |
| 归一 | **不做** softmax、不除 TOKENS/head_dim；AV 直接进 proj 线性 | contract `scope_note`；backend §2 |
| 图像变体差异 | 图像 native 64 token 时上界 2048（12 bit）——**不同口径，勿混用**；其 numeric_contract.json 的 `av` 字段仍是 32-token 的 uint11/1024 文案（陈旧），以 resolved_config.derived 为准 | `build/spikformer/vis_native/resolved_config.json` derived.av_bits=12 / av_max=2048 |

## 5. 残差规则

| 条目 | 口径 | 来源 |
|---|---|---|
| 加什么 | 残差只加 **LIF 后的 0/1 spike**，不加线性累加值；proj 与 fc2 后都先 LIF 再加回 | contract `dataflow`；`docs/VISION_LINE_RETURN_20261031.md`（golden 修复因果证据） |
| 上界 | embed spike=1；block 内 Attention 残差=2；MLP 残差=3；depth=2 最坏 `1+2*depth=5` | contract `numeric.residual_bounds`；numeric_contract `residual.bounds/worst` |
| 位宽 | 最坏 5 → 需 3 bit，实际分配 **RW=4 bit**（uint4） | contract `rw_alloc_bits=4`、`res_bits_min_required=3`；numeric_contract `residual.bits=4/rw_alloc=4` |
| 截断策略 | **不对连续累加器做低位截断后塞进残差**；LIF 输出（0/1）喂残差 | contract `numeric.downshift_policy` |
| 乘数类型 | 残差作为 linear 输入时为 uint4 | contract `numeric.mul_inputs.residual` |

## 6. Head（PL/PS 分工）

| 条目 | 口径 | 来源 |
|---|---|---|
| PL 归约 | `o_sums[c] = Σ_{n=0..TOKENS-1} Σ_{t=0..T-1} last_block_out[n][c][t]`（RW 位无符号残差），**PL 只做 sum、不除**，输出 uint32/通道 | contract `hardware_abi.head_sums_definition` |
| PS 归一 | `x = o_sums/(T*N)`（FP32，T*N=128）→ FP32 `W@x+b` → argmax | contract `numeric.head.ps`；numeric_contract `head` |
| 分支 | 无 folded 分支（`no_folded_branch=true`） | contract `numeric.head.no_folded_branch` |
| 一致性 | RTL `o_sums` 与 integer_ref PL 和逐通道相等 → 绑定 FP32 head 的 logits/argmax 与硬件路径一致 | `final_gates.json` real_weight_replay.detail；`replay_real_holdfix.json` |

## 7. 量化

| 条目 | 口径 | 来源 |
|---|---|---|
| 权重 | signed **int8**，per-tensor 对称 scale = maxabs/127；zero scale → 1.0；范围 [-127,127] | contract `numeric.weights`；numeric_contract `quant` |
| bias | signed **int32**（整数累加域） | contract `numeric.bias` |
| 累加 | 每个线性级 signed **int32**，导出时按实际 tensor 检查边界 | contract `numeric.accum`；integer_ref `_assert_i32` |
| 阈值/scale | 每层 per-tensor scale/threshold/bias 进 params + manifest；阈值 int32、thr>=1 | contract `numeric.lif.thr_min`；`manifest.json` |
| BN | BN 状态显式；本批 **absent** | contract `numeric.quant`；numeric_contract `quant` |
| 标定规则 | 整数域一次规则：正膜电位中位数，clamp>=1；dead LIF 直接报错 | `reports/export_final_real50.json` calibration；`SPIKFORMER_BACKEND.md` §7 |

## 8. ABI / 打包

| 条目 | 口径 | 来源 |
|---|---|---|
| 权重 beat | 64-bit = 8 个连续 signed int8，row-major [out][in] | contract `hardware_abi.weight_beat_format` |
| bias beat | 64-bit = `{bias[2m+1][31:0], bias[2m][31:0]}` signed int32；**不是 20-bit 字段** | contract `hardware_abi.bias_beat_format` |
| block in/out beat | 128-bit = TL=8 lanes × T=4 × RW=4 bit；lane-major，lane a、时隙 tt 的位偏移 `(a*T+tt)*RW`；beat 顺序 token-major → 通道组 | contract `hardware_abi.block_in_out_beat_format`；integer_ref `pack_block_io_beats` |
| descriptor | 每 block 4 个 64-bit beat：`{thr_q,thr_k}`、`{thr_v,thr_proj}`、`{thr_fc1,thr_fc2}`、`{0,thr_e}`（thr_e 仅 block0 有意义） | contract `hardware_abi.descriptor_beats_64b` |
| 每 block 权重拍数 | E=128、HIDDEN=512：**25152 beats/block**（q/k/v/proj_w 各 2048，bias 各 64；fc1/fc2_w 各 8192，bias 256/64） | contract `hardware_abi.beats_per_block_weights` |
| 样本序列 | `i_start` → tab（etab V*E bytes 然后 ptab N*E bytes，64-bit 字）→ tok N 拍 16-bit id → 每 block 4 desc + 25152 权重拍 → block 运行（内部产生/收集 512 block-in/out 拍）→ head sum → `o_sums_valid`+`o_done` | contract `hardware_abi.sample_sequence` |
| T*N | head 求和覆盖 **T*N=128** 个样本（T=4 × N=32） | contract `hardware_abi.head_sums_definition` |
| 握手 | `valid && ready` 同拍消费一拍；stall 时 payload 稳定；计数器只在 `valid&&ready` 推进；无 ready-ahead-of-valid 隐式契约 | contract `hardware_abi.handshake` |
| 输出 | `o_sums[E]` 每通道 OW=32 bit；`o_sums_valid` 每样本与 `o_done` 同拍脉冲一次 | contract `hardware_abi.top_ports_exact` |
| 顶层/端口 | `spikformer_nlp_top #(E=128,HEADS=4,HEAD_DIM=32,HIDDEN=512,T=4,TOKENS=32,DEPTH=2,TL=8,TC=8,RW=4,PSUM_SLICE=32,SUM_W=6,LANES=8,VTAB=4000,OW=32)`；端口冻结见 contract `top_ports_exact` | contract `hardware_abi`；TB/replay 必须匹配 |

## 9. 数值边界（导出时检查）

| 层 | pos 上界 | neg 下界 | 说明 | 来源 |
|---|---|---|---|---|
| QKV | 16256 | -16256（contract）/ **-16384（resolved_config derived）** | block0 spike 输入（0/1）× 128 输入；block>0 输入为残差（block1 输入 max 3）：worst 48768 / -49152。两处口径不同，并列记录 | contract `numeric.linear_bounds_checked_at_export.qkv`；`nlp/final/resolved_config.json` derived.accum_bounds.qkv |
| proj | 16646144 | -16777216 | AV max 1024 × 权重 127/-128 × 128 输入 | 同上 .proj |
| fc1 | 81280 | -81920 | residual max 5 | 同上 .fc1 |
| fc2 | 65024 | -65536 | fc1 spike 0/1 × 512 hidden | 同上 .fc2 |

注意：这些是合同给的 sanity 下界；导出器必须按**实际 per-block 输入域 + bias** 计算并在超 int32 时报错（contract 原文要求）。
实际最终 checkpoint（real50）的逐层整数 trace 与边界已由 `train_realweight_fidelity_real50.json`（47 层）复核 PASS。

## 10. 与上游 Spikformer / 图像变体的差异

| 项 | 本 NLP 变体 | 上游 Spikformer | 图像 native 变体（fixture） |
|---|---|---|---|
| attention 归一 | 无 softmax、AV 不归一 | softmax attention | 同 NLP 口径（未归一） |
| 归一化层 | 无 BN / 无 LayerNorm | LayerNorm | 无 |
| LIF 位置 | 7 处（embed/q/k/v/proj/fc1/fc2），proj/fc2 为新增分支输出 LIF | 不同 | 同 NLP 口径 |
| 几何 | E128/H4/HD32/N32/T4/D2 | Spikformer-S 等 | E384/H12/HD32/N64/T4/D1 |
| AV 上界 | 1024（uint11） | 不适用 | 2048（12 bit；resolved_config derived.av_max） |
| 权重/精度 | 真实语料训练 checkpoint（real_corpus） | 预训练权重 | **development_fixture**（无 checkpoint，无精度） |

来源：`contract.json` `scope_note`；`build/spikformer/vis_native/resolved_config.json` derived；`docs/SPIKFORMER_VISION_HANDOFF_20261031.md`。

## 11. 复核命令（发布根 D:/DL_FINAL/SNN 下执行）

    # 1) 纯 Python 数值自检（M1 等价：torch 量化 vs integer_ref 全层整数 trace）
    E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py

    # 2) 检查合同/manifest 一致性（若脚本在快照内）
    E:/miniconda/python.exe -B scripts/spikformer/check_contract.py

    # 3) RTL 回放（需要 ModelSim；TB 在 verification/sim/spikformer/）
    E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py

预期：与 `evidence/nlp/math_tests_m1.log` 记录的 8/8 PASS 一致（脚本版本若不同，以日志为准）。
更完整的演示与板端路径见 `docs/DEMO_NLP.md`、`docs/PLATFORM_AND_STATUS.md`；证据索引见 `evidence/INDEX.md`。
