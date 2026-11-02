# snn-translate 事实来源索引

> 规则：每条关键结论对应到**可打开、可核对**的文件/字段；`[包内]` 路径相对提交包根（`D:/FINAL`），
> `[工作区]` 路径只在开发机存在（`D:/DL_FINAL/...`、`D:/dl_snn`、`D:/spk_platform`），不随包。
> 标注"2026-11-02 实测"的条目是本技能写作时在本机重新执行得到的输出；未复跑的物理/板端结论标注来源日期。

## 1 身份与版本

| 结论 | 出处 |
|---|---|
| 版本 `snn-toolchain-0.1.0` | [包内] `src/snn/README.md` 第 1 节；[工作区] `D:\DL_FINAL\SNN\VERSION` |
| 发布快照 ERC/结构/范围说明 | [包内] `src/snn/docs/RELEASE_SCOPE.md`；`src/snn/README.md` §4/§5 |
| 交付状态终裁（platform=partial、board=record_level） | [包内] `src/snn/docs/PLATFORM_AND_STATUS.md` 开头"与任务卡初稿的差异"；`src/snn/docs/DELIVERY_SUMMARY.md` |
| FROZEN RTL 18 文件 + 规则 | [包内] `sim/snn/evidence/nlp/rtl_freeze.json`（`frozen_at=2026-10-29T22:26:00+08:00`、`files`、`rule`、`open_tb_items`） |
| 快照 RTL 16 SAME / 2 DIFF | [包内] `sim/snn/evidence/nlp/rtl_hash_check_20261031.txt`；`src/snn/docs/PLATFORM_AND_STATUS.md` §5 |
| 当前源清单 28 | [包内] `src/snn/rtl/ops/spiking/source_manifest.json`（`version=20261031.1`、`sources` 28 项） |

## 2 CLI 与参数

| 结论 | 出处 |
|---|---|
| 子命令 `compile/build/package` 与全部选项 | [包内] `src/snn/tools/toolchain.py`（argparse 第 53~77 行）；本机 2026-11-02 实测 4 条 `--help` 输出 |
| `build --stage synth\|ooc` 走 `scripts/spikformer/run_physical.py`，`--execute` 才真跑 | `src/snn/tools/toolchain.py` `_spik_build`；[包内] `build/snn/scripts/spikformer/run_physical.py` 头注释与 `main` |
| `build --stage ip\|platform` 只打印入口 | `src/snn/tools/toolchain.py` `_spik_build` 分支（返回 0） |
| `package` 产出 `runtime_config.json` + params + README，bit/HWH=NOT_BUILT | [包内] `src/snn/tools/backends/spikformer/package.py` `run()` |
| `compile` 拒绝覆盖非空 out-dir | `src/snn/tools/backends/spikformer/compile.py` `run()` 第 1 个判断 |
| `--fixture-seed` 与 `--weights` 互斥；`--params-dir` 未实现 | `src/snn/tools/backends/spikformer/compile.py` `run()`；本机 2026-11-02 实测报错文本 |

## 3 配置合同与报错原文

| 结论 | 出处 |
|---|---|
| 必填字段、家族差异、硬校验（B=1、T∈{1,2,4}、HD=32、E=heads*HD、E%16/32、MLP%16、seq∈[1,256]） | [包内] `src/snn/tools/backends/spikformer/config.py` 第 20~95 行；报错前缀 `raise ValueError('config: '+msg)` |
| 7 条配置报错原文（实测） | 本机 2026-11-02 实测；对应 `config.py` 各 `_err(...)` |
| 派生量 `tokens/sum_w=6/av_bits=11/av_max=1024/rw=4/res_bits_min=3/worst_residual=5/stream.beats` | [包内] `build/snn/generated/_quickstart_real/resolved_config.json` `derived`；`src/snn/tools/backends/spikformer/config.py` `resolve()` |
| 结构常量 TL8/TC8/RW4/PSUM_SLICE32/LIF_ADD_BITS33/ACCUM32 | `config.py` 头部常量；`manifest.py` `structure`；[包内] `sim/snn/evidence/nlp/final/numeric_contract.json` `structure` |

## 4 checkpoint 合同与严格导入

| 结论 | 出处 |
|---|---|
| 必需键（etab/ptab/thr_e/head_w/head_b/meta_* + 每 block w/b/scale/thr） | [包内] `src/snn/tools/backends/spikformer/weights.py` `NPZ_REQUIRED` 与 `import_checkpoint()` |
| BN 状态只允许 absent/already_folded；requires_fold 报错 | `weights.py` `BN_STATES` 与 `import_checkpoint` 第 3 段 |
| dtype/shape 严格、无静默 cast；实测报错 `must be int8, got float32 (no silent casts)` | `weights.py` `_get()`；本机 2026-11-02 实测 |
| `meta_cfg` 与配置逐字段一致才导入 | `weights.py` `import_checkpoint`（meta_cfg 比较循环） |
| 交付 checkpoint sha256 `e6a3a411…2448a1`、935,756 B、real_corpus | [包内] `data/snn/examples/nlp_sentiment_b1/README.md` §1；`sim/snn/evidence/nlp/export_final_real50.json` `checkpoint_sha256` |
| 伴生 assets（corpus/vocab/label/split + 标定阈值）自动绑定 | `compile.py` 自动发现 `<weights stem>.assets.json`；[包内] `data/snn/examples/nlp_sentiment_b1/ckpt_real_real50.assets.json` |
| ABI 细节（64-bit 权重 beat=8×int8、bias beat=2×int32、block in/out 128-bit、4 descriptor beats、25152 beats/block） | [包内] `src/snn/docs/NUMERIC_CONTRACT.md` §8；`sim/snn/evidence/nlp/contract.json` `hardware_abi`；`resolved_config.json` `derived.stream.block_w_beats=25152` |

## 5 产物与身份哈希

| 结论 | 出处 |
|---|---|
| manifest 字段与生成顺序（manifest_sha256 最后计算） | [包内] `src/snn/tools/backends/spikformer/manifest.py` `write_manifest()` |
| `manifest_sha256` 算法（去字段+排序 JSON sha256）与实测复算一致 | 同上；本机 2026-11-02 实测 `recompute: True` |
| 54 blob（real）/28 blob（fixture）与命名空间 | [包内] `build/snn/generated/_quickstart_real/manifest.json` `params`；`build/snn/generated/_quickstart_check/manifest.json`；本机 2026-11-02 两种模式实测 |
| 冻结 real manifest：`f727e4c7…911e`、文件 sha `b71e3f17…a6a4`、54 blob、28 源 | [包内] `build/snn/generated/_quickstart_real/manifest.json`；`sim/snn/evidence/ACCEPTANCE_CORE.md` §A1b；本机 2026-11-02 复现（逐字节一致） |
| fixture manifest：`835ad38d…381c4`、28 blob | `build/snn/generated/_quickstart_check/manifest.json`；`ACCEPTANCE_CORE.md` §A1；本机 2026-11-02 复现 |
| params 集合摘要 `315171a2…`（real）/ `000b0b5b…`（fixture） | `ACCEPTANCE_CORE.md` §A1/A1b；本机 2026-11-02 复算 params sha256 集合与包内一致（54/54） |
| 生成物 7 件（header/adapter/sources.f/xdc/ooc_tcl/project_tcl/cache_key） | [包内] `src/snn/tools/backends/spikformer/codegen.py` `generate()` 返回值；`build/snn/generated/_quickstart_real/generated/` 实际 7 个文件 |
| XDC 内容（1 个 5.000 ns clock + 0 input/output delay，无例外），冻结 sha `9fc1c267…10842` | `codegen.py` xdc 段；[包内] `sim/snn/evidence/nlp/final/generated/spikformer.xdc`；`INDEX.md` §2 |
| adapter 含 1192 个 DONT_TOUCH LUT1 hold-margin（8 级×149 bit） | `codegen.py` `HOLD_MARGIN_STAGES=8`/`_adapter()`；`sim/snn/evidence/nlp/final/generated/spikformer_nlp_target_top.sv`；`INDEX.md` §2 |
| schedule 字段：block0 `base=1879048192`、descriptor beats、`memory_map`、`weight_stride=1048576` | [包内] `build/snn/generated/_quickstart_real/schedule.json`；`src/snn/tools/backends/spikformer/schedule.py` |
| 13 个阈值（thr_e=26；block0 128/100/92/546/57/100；block1 190/190/203/1168/96/64） | [包内] `build/snn/generated/_quickstart_real/numeric_contract.json` `thresholds`；`sim/snn/evidence/nlp/export_final_real50.json` `calibration.thresholds_after` |

## 6 数值合同

| 结论 | 出处 |
|---|---|
| 全整数变体、禁止与上游/FP32 逐位比较、无 softmax/AV 不归一/无 BN-LN | [包内] `src/snn/docs/NUMERIC_CONTRACT.md` §0、§10；`sim/snn/evidence/nlp/contract.json` `scope_note` |
| LIF：33 bit 符号扩展、V=floor((pre+delta)/2)、spike iff valid&&V>=0&&V>=thr、硬复位、thr≥1 | `NUMERIC_CONTRACT.md` §2；`numeric_contract.json` `lif/lif_add_bits`；`integer_ref.py` |
| QK：按位 AND popcount、不除 head_dim、SUM_W=6、all-ones=32 | `NUMERIC_CONTRACT.md` §3；`numeric_contract.json` `qk_popcount`；`math_tests_m1.json` `popcount32_avmax1024` |
| AV 不归一、NLP 32 token max=1024(uint11) | `NUMERIC_CONTRACT.md` §4；`resolved_config.json` `derived.av_bits/av_max` |
| 残差只加 0/1 spike、worst=5、RW=4 bit | `NUMERIC_CONTRACT.md` §5；`numeric_contract.json` `residual` |
| Head：PL 对 T*N 求和（不除）、PS 除 128 后 FP32 Linear+argmax | `NUMERIC_CONTRACT.md` §6；`numeric_contract.json` `head` |
| 量化：int8 per-tensor 对称、int32 bias/累加、BN absent、标定规则（正膜电位中位数 clamp≥1、dead LIF 报错） | `NUMERIC_CONTRACT.md` §7；`export_final_real50.json` `calibration` |
| 整数边界（qkv/proj/fc1/fc2）与 int32 检查 | `NUMERIC_CONTRACT.md` §9；`resolved_config.json` `derived.accum_bounds`；`check_contract.py` 108 项 |

## 7 回放（G1）

| 结论 | 出处 |
|---|---|
| 四档回放 pass 与 cycles（39774/7084772/7084812/10627157）、fail_lines=0 | [包内] `sim/snn/evidence/nlp/replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json` `pass/cycles/steps` |
| mismatch=0 的显式文字 | `sim/snn/evidence/nlp/final_gates.json` `gates.target_e2e.detail`/`gates.real_weight_replay.detail`；`rtl_freeze.json.reason` |
| 回放命令与参数 | [包内] `build/snn/scripts/spikformer/run_nlp_sim.py` argparse（实测 `--help`）；`src/snn/docs/DEMO_NLP.md` §1.3 |
| 回放比对范围（embed beat/block in-out/o_out_last/o_sums[128]/done） | `final_gates.json` `target_e2e.detail`；`replay_*.json` `rtl_sources`（TB sha 绑定） |
| ModelSim 版本 2020.4 | `replay_*.json` `modelsim` 字段 |

## 8 OOC / IP / 平台 / 板端

| 结论 | 出处 |
|---|---|
| OOC 5 ns：WNS +0.135 / WHS +0.023 / TNS 0 / THS 0 / 0 failing（186,547 端点）/ unrouted 0 / latch 0 / 无时序例外 | [包内] `sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json` `wns_setup_ns/wns_hold_ns/unrouted_nets/latch_cells`；`PLATFORM_AND_STATUS.md` §1 |
| OOC 资源 LUT 40,964 / FF 71,173 / BRAM 186.5 / DSP 128（measured） | `ooc_20261030_120804.json` `utilization` |
| hold 收口过程（32,852 输入起点/1,192 cells/16 级拒/10 级过/8 级最终） | `sim/snn/evidence/nlp/final/reports/hold_closure_20261030.json`；`INDEX.md` §2；`codegen.py` 注释 |
| IP：`timing_qualified=true`、19/19、`package_sha256=894599b9…292f22`、5 条 assumptions | [包内] `sim/snn/evidence/nlp/final/ip/ip_manifest.json` `timing_qualified/package_sha256/assumptions`；`SOURCE_SHA256.txt`（19 行） |
| 平台 r7：WNS +0.051（334,330）/ WHS +0.011（333,740）/ 0 failing / WPWS +0.039 / route 0 errors、fully routed 180,101 | [包内] `build/snn/reports/platform_timing_summary_20261031.txt`（派生摘要，注明原始报告在 `D:/spk_platform/reports/`）；`r7_sumsfix_route_status.rpt` |
| 平台 r6：WNS +0.245 / WHS +0.010 / 0 failing；CDC 1 Critical 未收敛；软复位不排空 DMA；`sys_rst` 无 input delay | `build/snn/reports/CDC_FIX_R6_RESULT_20261030.md`；`build/snn/reports/platform_build_20261030.json` `next`；`src/snn/docs/PLATFORM_AND_STATUS.md` §2 |
| 平台 r7 的 SUMS +64 偏移（读法 `ROT=64`） | `src/snn/docs/DEMO_NLP.md` §3；`PLATFORM_AND_STATUS.md` §3"SUMS 读窗口" |
| bit sha256 `9b2e318c…c267b`（28,700,853 B）/ hwh `fe2d9f0e…374fb`（380,358 B）/ MANIFEST 20 文件 | [包内] `board/snn/MANIFEST.sha256`；`PLATFORM_AND_STATUS.md` §1"上板包"；`board/snn/README_BOARD.md` |
| 板端记录 3/3 match、cycles≈7,084,772、model_ms≈35.4、record_level/not_verified | `board/snn/README_BOARD.md` §3；`src/snn/docs/PLATFORM_AND_STATUS.md` §3；`sim/snn/evidence/platform/board/board_run_local_attempt.log`（失败尝试，非板测证据） |

## 9 本机 2026-11-02 复跑记录（可直接复核）

| 命令（见 reference/commands.md） | 结果 |
|---|---|
| C5 fixture 编译（包内 `src/snn`） | exit 0；28 blob；`manifest_sha256=835ad38d…381c4` |
| C6 真实权重编译（包内 `data/snn` + `PYTHONPATH`） | exit 0；54 blob；`f727e4c7…911e`；manifest 文件 sha `b71e3f17…a6a4`，与包内逐字节一致 |
| C8 package | exit 0；`packaged 54 param files`；bit/hwh=NOT_BUILT |
| C10 物理驱动排队（`SNN_ROOT=src/snn`） | `source frozen check: 28 files match manifest hashes`；`QUEUED`；exit 0 |
| C12 test_short | 15 项 PASS，`FAILURES: 0`（需 `SNN_ROOT`） |
| C13 check_contract | `status=pass`、`fails=[]`、108 checks |
| C14 test_integer_ref（[工作区]） | `RESULT: 8/8 PASS -> PASS`，exit 0 |
| C15 check_ready（[工作区]） | `FAIL items: 0`（包内 `src/snn` 跑则 1 FAIL：config 路径） |
| C17 IP 复打包（`SNN_ROOT=src/snn`） | `status=pass, sources=28, params_copied=54, mismatch=[]` |
| 配置越界/浮点 checkpoint 报错 | 见 troubleshooting T8/T9 的原文 |
| C19 冻结 RTL 哈希 | 预期 `freeze files=18 mismatch=2`（`sps/spiking_sps_top.sv`、`sps/spiking_eyeriss_pe.sv`） |

## 10 包内 vs 工作区差异（引用时必须区分）

| 项 | [包内] `D:/FINAL` | [工作区] `D:/DL_FINAL/SNN` |
|---|---|---|
| 布局 | 按 `src/snn data/snn build/snn sim/snn board/snn` 拆分 | 快照根下 `tools/ rtl/ scripts/ examples/ verification/ board/ evidence/` 并列 |
| compile/package | 可直接跑（见 commands.md C5/C6/C8） | 按 QUICKSTART 直接跑 |
| `build --stage synth\|ooc` | 驱动路径断裂，只能排队（C10） | 可排队/可 `--execute`（受锁） |
| 回放 | 只有 TB 源码 + 证据 JSON，无主线向量 | 有 `scripts/spikformer/`、`verification/`，可跑（需 ModelSim） |
| `test_integer_ref` | 缺 `runtime/spikformer/`，不可跑 | 可跑（需 torch） |
| `test_short` | 可跑，需 `SNN_ROOT=src/snn` | 直接可跑 |
| `check_ready` | 1 FAIL（examples 路径） | `FAIL items: 0` |
| 原始物理报告（2 MB+ / DCP） | 未收录，只有摘要/报告 JSON | 主线 `D:/dl_snn/build/...`、`D:/spk_platform/reports/` 可复核 |

## 11 未闭合项（写作时如实记录）

1. 回放（ModelSim）、OOC/平台（Vivado+heavy lock）、板端均需对应环境；本技能对它们只做**证据级判定**并标注来源日期。
2. 包内无 `runtime/spikformer/`，`test_integer_ref` 只能在 [工作区] 跑；`test_runtime.py` `[工作区]` 亦为 pre-existing fail（`ACCEPTANCE_CORE.md` §A3）。
3. 板端原始 JSON/日志未随包、板卡离线 → G6 只能 `record_level`。
4. 平台 `final_gates.json` 早于 r6/r7；`platform=not_run` 是 2026-10-30 13:10 的冻结口径，不是当前状态。
5. 图像线为 `partial`，本技能不做其验收；其 `numeric_contract.json.av` 字段陈旧（以 `resolved_config.derived` 为准）。
6. 历史提炼提到的"浮点 checkpoint 专用拒绝文本""blob 数不符报错"未找到证据，本技能只写实测到的 dtype 报错与派生计数。
