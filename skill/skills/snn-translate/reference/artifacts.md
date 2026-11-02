# snn-translate 产物与清单字段

> 样本取值来自 [包内] `build/snn/generated/_quickstart_real/`（real_corpus，E128/H4/HD32/T4/seq32/MLP512/depth2/V4000）
> 与 `sim/snn/evidence/nlp/final/`；字段名逐字取自真实 JSON。生成者列对应 `src/snn/tools/backends/spikformer/` 模块。

## 1 out-dir 树（compile 产物）

| 路径 | 生成者 | 说明 |
|---|---|---|
| `manifest.json` | `manifest.write_manifest` | 权威身份清单（见 §3） |
| `resolved_config.json` | `compile.run` | 展开后的配置，含 `derived`、本批 `thresholds`、`assets` |
| `numeric_contract.json` | `manifest.write_manifest` | 数值合同摘要（LIF/QK/AV/残差/head/量化/结构） |
| `schedule.json` | `compile.run` | 片上调度（每 block 权重基址、descriptor beats、memory_map） |
| `params/*.bin` + `*.coe` | `weights.py _emit` | 参数 blob；每个 blob 同时写 `.bin` 与 Xilinx `.coe` |
| `generated/spikformer_config.vh` | `codegen.generate` | RTL 结构宏 |
| `generated/spikformer_nlp_target_top.sv` | `codegen._adapter` | 目标顶层适配器（含 hold-margin 单元） |
| `generated/spikformer.xdc` | `codegen.generate` | 5 ns 约束（1 clock + 0 input/output delay，无例外） |
| `generated/spikformer_ooc.tcl` | `codegen.generate` | OOC synth/impl 脚本 |
| `generated/spikformer_project.tcl` | `codegen.generate` | Vivado 工程装配脚本 |
| `generated/sources.f` | `codegen.generate` | RTL 源清单（绝对路径） |
| `generated/cache_key.txt` | `codegen.generate` | 生成器缓存键（sha256(cfg+sched+part+tool)） |

## 2 参数 blob（54 个，real_corpus）

命名空间：`embed_w/pos_w/thr_e/embed_scale` + `block{b}.{q,k,v,proj,fc1,fc2}_{w,b,scale,thr}` + `head_w/head_b`。
数量公式：**真实权重 54 = 6 + 24×depth**（depth=2）；**fixture 28 = 4 + 12×depth**（fixture 只发 w/b，且额外带 NLP 未用的 `sps_conv1_w/sps_conv1_b` 占位）。

| 组 | dtype | shape | 样例 nbytes | 依据 |
|---|---|---|---|---|
| `embed_w` | int8 | (vocab_size, E) = (4000,128) | 512,000 | `weights.import_checkpoint` |
| `pos_w` | int8 | (seq_len, E) = (32,128) | 4,096 | 同上 |
| `thr_e` | int32 | (1,) | 4 | 标量阈值 ≥1 |
| `embed_scale` | float32 | (1,) | 4 | 来自 npz `embed_scale`；缺失时按 `maxabs/127` 推导并记录来源 |
| `block{b}.{q,k,v,proj}_w` | int8 | (E,E) = (128,128) | 16,384 | 每 block 4 个 |
| `block{b}.{q,k,v,proj}_b` | int32 | (E,) = (128,) | 512 | 累加域 bias |
| `block{b}.{q,k,v,proj}_scale` | float32 | (1,) | 4 | per-tensor 对称 scale |
| `block{b}.{q,k,v,proj}_thr` | int32 | (1,) | 4 | 每 LIF 阈值 ≥1 |
| `block{b}.fc1_w` / `fc2_w` | int8 | (MLP,E)=(512,128) / (E,MLP)=(128,512) | 65,536 | |
| `block{b}.fc1_b` / `fc2_b` | int32 | (512,) / (128,) | 2,048 / 512 | |
| `block{b}.fc1_scale` / `fc2_scale` | float32 | (1,) | 4 | |
| `block{b}.fc1_thr` / `fc2_thr` | int32 | (1,) | 4 | |
| `head_w` / `head_b` | float32 | (C,E)=(2,128) / (C,)=(2,) | 1,024 / 8 | PS FP32 head |

每个 blob 在 manifest 里的元数据：`{"bin": "<name>.bin", "coe": "<name>.coe", "sha256": "<64hex>", "shape": [...], "dtype": "int8|int32|float32", "nbytes": <int>}`。

## 3 manifest.json 字段

| 字段 | 类型/样例 | 含义 |
|---|---|---|
| `backend` / `schema` | `"spikformer"` / `"spikformer-manifest-1"` | 后端与清单 schema |
| `model_kind` | `"real_corpus"` 或 `"development_fixture"` | 权重来源，不允许第三种 |
| `fixture_seed` | `null`（真实权重）/ `1`（fixture） | fixture 模式记录 seed |
| `config` | 配置对象（不含 `derived`） | 本次输入，含 `assets.checkpoint` 路径字符串 |
| `derived` | `tokens=32, sum_w=6, attn_word=24, rw=4, av_bits=11, av_max=1024, worst_residual=5, stream{...}` | 派生量 |
| `structure` | `{"E":128,"HEADS":4,"HEAD_DIM":32,"HIDDEN":512,"T":4,"TOKENS":32,"DEPTH":2,"BATCH":1,"TL":8,"TC":8,"RW":4,"PSUM_SLICE":32,"SUM_W":6,"lif_add_bits":33,"accum_bits":32}` | RTL 结构绑定 |
| `schedule` | 同 `schedule.json` | 调度内嵌 |
| `params` | 54 项字典 | 参数元数据（见 §2） |
| `rtl_sources` | 28 项相对路径 | `src/snn/rtl/ops/spiking/source_manifest.json` 的 `sources` |
| `rtl_sha256` | 28 项 `{path: sha256}` | 生成时逐文件哈希 |
| `generated` | `{header,adapter,sources_f,xdc,ooc_tcl,project_tcl,cache_key}` | 生成物相对路径与缓存键 |
| `target` | `{"part":"xczu15eg-ffvb1156-2-i","tool":"Vivado 2021.1","period_ns":5}` | 目标 |
| `head` | `{"reduction":"sum","norm":"ps_div","params":{...}}` | PS head 合同 |
| `numeric` | 数值合同 + `thresholds/scales/model_assets` | 见 §5 |
| `assets` | checkpoint 路径/sha256、corpus/vocab sha256、bn_state | 伴生资产 |
| `manifest_sha256` | 64 hex | 去掉该字段后对 `json.dumps(sort_keys=True)` 的 sha256 |

## 4 schedule.json 字段（真实样本）

| 字段 | 样例 | 含义 |
|---|---|---|
| `sps_once` | `true` | SPS 前端只跑一次（NLP 无 SPS，字段保留） |
| `blocks[i].block/first/last` | `0/true/false`、`1/false/true` | 块序 |
| `blocks[i].weights.{q_w,...}` | `{"base":1879048192,"bytes":16384}` | DDR 权重窗口基址；block0 起 `0x70000000`，stride `0x100000` |
| `blocks[i].lif_thr` | block0 `{"q":128,"k":100,"v":92,"proj":546,"fc1":57,"fc2":100}`；block1 `{"q":190,"k":190,"v":203,"proj":1168,"fc1":96,"fc2":64}` | 本 block 6 个 LIF 阈值（来自 checkpoint） |
| `blocks[i].descriptor_beats_64b` | block0 `["0x0000008000000064","0x0000005c00000222","0x0000003900000064","0x000000000000001a"]` | 4 个 64-bit descriptor：`{thr_q,thr_k}`、`{thr_v,thr_proj}`、`{thr_fc1,thr_fc2}`、`{0,thr_e}`（仅 block0 的 thr_e） |
| `blocks[i].bank_sel` | `0` / `1` | 残差 bank 翻转 |
| `blocks[i].window` | `{"base":1879048192,"bytes":196608}` | 本 block 权重窗口 |
| `head_after_block` | `1`（= depth-1） | 只有最后一个 block 进 head 归约 |
| `memory_map` | `{"sps_out_tokens":32,"residual_banks":2,"bank_words_64b":1024,"attn_tmp_words":96}` | 片上缓冲 |
| `weight_stride` | `1048576` | 每 block 权重窗口步长（字节） |
| `thr_e` | `26` | 样本级阈值 |
| `descriptor_note` | `"4 beats per block, Verilog-concat order ..."` | 与 contract 的 ABI 对应 |

## 5 numeric_contract.json 字段（真实样本）

| 字段 | 样例 | 含义 |
|---|---|---|
| `lif` | `"V=(delta+pre)>>>1 tau=2, spike iff valid and 0<=V and V>=THR, reset 0"` | LIF 语义 |
| `lif_add_bits` | `33` | pre+delta 符号扩展位数（不回绕） |
| `qk_popcount` | `"per-T sum width clog2(head_dim+1)=6, all-ones=32 representable, no /head_dim"` | QK |
| `av` | `"unnormalized sum over TOKENS of score*V_spike, uint11 max 1024"` | AV 不归一 |
| `residual` | `{"bounds":{"embed":1,"attn_res":2,"mlp_res":3},"worst":5,"bits":4,"rw_alloc":4}` | 残差 0/1 spike 相加，RW=4 |
| `head` | `{"reduction":"sum","norm":"ps_div",...}` | PL 求和 + PS 除 T*N |
| `quant` | int8 per-tensor 对称 scale、int32 bias、BN 显式（本批 absent） | 量化合同 |
| `structure` | 同 manifest `structure` | |
| `thresholds` | `thr_e=26`；blocks 同 schedule `lif_thr` | 13 个阈值 |
| `scales` | `embed=1.0`；block0 proj 0.15748031437397003、fc1 0.6220472455024719、fc2 0.8818897604942322；block1 proj 0.10236220806837082、fc1 0.6456692814826965、fc2 0.8976377844810486 | per-tensor scale |
| `model_assets` | `model_kind/corpus_sha256/vocab_sha256/checkpoint/checkpoint_sha256/bn_state` | 资产绑定（checkpoint sha `e6a3a411…2448a1`） |

## 6 包内已冻结的生成物（可直接当样本）

| 路径 | model_kind | blob 数 | manifest_sha256 |
|---|---|---|---|
| `build/snn/generated/_quickstart_check/` | development_fixture | 28 | `835ad38d…381c4` |
| `build/snn/generated/_quickstart_real/` | real_corpus | 54 | `f727e4c7…911e`（文件 sha `b71e3f17…a6a4`） |
| `build/snn/generated/_lead_final_check/` | real_corpus | 54 | `1047d0d0…dbf9`（2026-10-31 17:00 旧 RTL 刷新前读数） |
| `sim/snn/evidence/nlp/final/` | real_corpus | 54 | `79b1e115…ed926`（冻结 run；19 源清单） |

## 7 板端运行包（package 产物）

| 文件 | 内容 |
|---|---|
| `runtime_config.json` | `backend/model_kind/manifest_sha256`、`csr`（ID=0x00 … TAG=0x34，13 个偏移）、`derived`、`schedule`、`head`、`target`、`param_files_copied`（54）、`bitstream/hwh = NOT_BUILT ...` |
| `params/` | 54 个参数 `.bin` 副本（无 `.coe`） |
| `README.txt` | 上板三步（program bitstream → load params → PYNQ MMIO run） |

## 8 IP 源包（run_ip_pack 产物）

| 路径 | 内容 |
|---|---|
| `ip_manifest.json` | `timing_qualified`、`ooc_status`、`source_hashes_match_sim`、`source_mismatch`、`generated_files`、`hardware_artifacts`、`assumptions`、`package_sha256`、`package_files` |
| `SOURCE_SHA256.txt` | 19 个源文件（18 冻结 RTL + `include/spikformer_cfg.vh`）sha256 |
| `src/rtl/ops/spiking/**`（IP 包内相对路径） | 与 manifest 同源的 RTL 副本（19 文件） |
| `generated/` | adapter/config.vh/xdc/ooc_tcl/project_tcl/sources.f |
| `params/` | 54 个参数 `.bin` |
| `manifest.json` / `numeric_contract.json` / `resolved_config.json` / `schedule.json` | 编译清单副本 |
| `README.txt` | SCOPE/ASSUMPTIONS（OOC 5 ns only、0 输入延迟、保留 1192 DONT_TOUCH cells、平台/bit/board not_run） |

冻结证据：`sim/snn/evidence/nlp/final/ip/ip_manifest.json`：
`timing_qualified=true`、`package_sha256=894599b92c8de7d4edf65320db5f23a1413e7e9fabc81740b357e2134b292f22`、
`source_hashes_match_sim=true`、`source_mismatch=[]`、5 条 `assumptions`（其中 OOC WNS 束缚路径=hold-margin 链、+0.135 ns 上游预算）。

## 9 板端包（board/snn）

| 文件 | 关键值 |
|---|---|
| `snn_platform.bit` | 28,700,853 B，sha256 `9b2e318c5eacd2f21c39bd6c8864cd0f41ca47ed069ee6bbd7ea45b0191c267b` |
| `snn_platform.hwh` | 380,358 B，sha256 `fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb`（与 bit 同 stem，PYNQ Overlay 必需） |
| `MANIFEST.sha256` | 20 个文件（2026-10-30 23:18） |
| `MANIFEST_DEMO_20261031.sha256` | 5 条补充清单（4 个 demo 资产 demo_nlp.py/demo_vocab.json/demo_head.npz/demo_samples.json + `README_SNAPSHOT.md`） |
| `snn_platform_run.py` | MMIO 写刺激 → start → 轮询 STATUS → 读 sums → 与 `vec/case_real/exp_sums_s*.hex` 比对；**r7 bit 需 ROT=64**（脚本原样未修） |
| `vec/case_real/` | 3 真实样本的 `etab/ptab/toks/stream_b{0,1}_s{0,1,2}.hex` + `exp_sums_s{0,1,2}.hex` + `golden.json`（golden sha `065c5056…bbba9`，与回放 real 用例同源） |
| `demo_nlp.py` 等 | 交互演示（文本→token→硬件→PS head），读 sums 用 `ROT=64` |

## 10 证据文件（G0–G6）

| 级 | 文件 | 关键字段 |
|---|---|---|
| G0 | `sim/snn/evidence/ACCEPTANCE_CORE.md` | §A1/A1b 的 exit、blob 数、集合摘要、manifest sha |
| G0 | `sim/snn/evidence/nlp/contract.json` | `model/numeric/hardware_abi/scope_note`（数值+ABI 权威） |
| G0 | `sim/snn/evidence/nlp/math_tests_m1.{log,json}` | `PASS:true, n_pass:8, n_total:8` |
| G0 | `sim/snn/evidence/nlp/export_final_real50.json` | `n_params_blobs=54`、`checkpoint_sha256`、`calibration`、`eval_integer_ref` |
| G0 | `sim/snn/evidence/nlp/train_realweight_fidelity_real50.json` | `layers_compared=47`、`first_mismatch=null`、`pl_sums_equal=true`、`PASS=true` |
| G1 | `sim/snn/evidence/nlp/replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json` | `pass/cycles/steps[2].result_line/steps[2].fail_lines` |
| G1 | `sim/snn/evidence/nlp/rtl_freeze.json` | `frozen_at`、18 文件 `files{path:sha256}`、`rule` |
| G2 | `sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json` | `status=pass, wns_setup_ns=0.135, wns_hold_ns=0.023, unrouted_nets=0, latch_cells=0, utilization{...}` |
| G2 | `sim/snn/evidence/nlp/final/reports/build_identity_20261030.json` | manifest/adapter/xdc/ooc_tcl/codegen 哈希 + cache_key |
| G4 | `sim/snn/evidence/nlp/final/ip/ip_manifest.json` | 见 §8 |
| G5 | `build/snn/reports/platform_timing_summary_20261031.txt` / `r7_sumsfix_route_status.rpt` / `platform_build_20261030.json` | r7 WNS +0.051 / WHS +0.011 / 0 failing；route 0 errors；build 阶段 smoke 7/7 |
| G6 | `board/snn/README_BOARD.md` §3；`sim/snn/evidence/platform/board/` | 3/3 match 记录、cycles≈7,084,772、`board_run_local_attempt.log`（本机缺 pynq 的失败尝试，非板测证据） |
