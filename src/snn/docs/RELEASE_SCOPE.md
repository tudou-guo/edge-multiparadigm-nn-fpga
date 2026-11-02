# 发布快照整理规范 D:/DL_FINAL/SNN 0.1.0（2026-10-31）

本文件是 D:/DL_FINAL/SNN 发布快照的整理规范与交付范围说明（Lead 维护）。

## 0. 目标与定位

把 D:/dl_snn（Spikformer SNN 加速算子库与工具链）中**已经闭环的 NLP 情感分类案例与现有工具链**收拢为
**自包含、可拷贝、可复核的发布快照**，放在 D:/DL_FINAL/SNN（与 D:/DL_FINAL/DL 并列，但互不修改）。
这是**保底交付**：即使后续图像线没有进展，这份快照也必须是完整、可运行、诚实的状态。

- 演示主线 = NLP 情感分类：真实语料训练 + 真实 checkpoint 导入 + 编译/参数/RTL/调度/PYNQ 包 +
  功能回放 0 mismatch + 5ns routed OOC pass + IP 包 + 平台 r7 bit + 板端 3/3 样本 match。
- 图像识别案例 = 诚实 partial 收录：工具链编译 pass、回放几何行为级整链 pass、native 资源门 fail、
  参考式 SPS P1 未过门、native 仿真/OOC/平台/板测 not_run。**不得当演示、不得报精度。**
- 图像线同时在另一个工作序列里推进：若收拢期间主线 RTL 有变化，快照以**复制时刻**的哈希为准，
  最终由 Lead 在交付前核对并更新 docs/VISION_CASE.md 的状态。

来源（只读）：D:/dl_snn（源码/文档/证据/build）、D:/spk_platform（平台工程与上板包）。
发布根目录：D:/DL_FINAL/SNN。**不得修改 D:/dl_snn 主线源码、D:/DL_FINAL/DL、D:/spk_platform 工程。**
内容冻结：2026-10-31 18:40（Lead 18:40-19:05 复核）。截止 19:15。

## 1. 事实纪律（最重要）

1. 只写 pass / fail / partial / not_run 的事实；每个 PASS 必须能指到证据文件路径。
2. 未执行步骤一律 not_run；不得用 fixture 结果、旧 NLP 结果、行为级回放冒充 native/板级结果。
3. 数字口径冲突时并列写两个数字 + 各自来源，不擅自选一个；不确定的写"未核实"。
4. 负 WNS 记 fail；不降频、不加时序例外；不把 OOC 说成集成/板级。
5. 英文技术词保留原文。

## 2. 目录结构（发布根 = D:/DL_FINAL/SNN/）

| 路径 | 内容 | 负责人 |
|---|---|---|
| README.md | 发布说明：是什么/来源/结构/快速开始/状态摘要/范围声明 | core |
| QUICKSTART.md | 3 条命令跑通（环境检查 → 编译 NLP → 纯 Python 数值自检；板端演示指路） | core |
| RELEASE_NOTES.md | 0.1.0 收录/未收录/已知限制/未验证项 | core |
| VERSION | snn-toolchain-0.1.0 | core |
| environment.md | 环境与路径（Vivado/ModelSim/Python/PYNQ/板卡；不随包工具） | core |
| docs/USER_GUIDE.md | 工具链用户指南（入口/命令/配置/权重导入/生成物/manifest/重跑物理门） | core |
| docs/OPERATORS.md | 算子库（rtl/ops/spiking 清单 + 各算子参数/数值语义/已验证范围） | core |
| docs/PORTABILITY.md | 快照内可移植性路径改写说明 + 与主线 diff 摘要 + 冻结 RTL 哈希核对 | core |
| docs/DEMO_NLP.md | NLP 演示手册（PC 侧数值自检 + PYNQ 板端；录制结果；已知限制） | docs |
| docs/PLATFORM_AND_STATUS.md | 平台与门状态（NLP 各门；上板 r7 结果；图像线 partial） | docs |
| docs/NUMERIC_CONTRACT.md | 数值合同（LIF/QK/AV/残差/head/量化/ABI） | evidence |
| docs/VISION_CASE.md | 图像线现状与证据索引（partial，为什么未做成演示） | evidence |
| docs/RELEASE_SCOPE.md | 本文件 | Lead |
| docs/DELIVERY_SUMMARY.md | 一页纸交付摘要（做了什么/证明什么/没做什么/怎么复核） | Lead |
| tools/ | 源码副本：toolchain.py + backends/spikformer + spikformer_train | core |
| rtl/ | 源码副本：ops/spiking/** + operator_library.json + source_manifest.json + UPSTREAM_MAP.json | core |
| runtime/ | PYNQ 运行包副本 runtime/spikformer/** | core |
| verification/ | 仿真 TB 与纯 Python 短例副本 + README 说明 | core |
| scripts/ | 精选脚本副本（见 §3） | core |
| examples/README.md + examples/spikformer/ + examples/nlp_sentiment_b1/ | 案例与状态表、配置、真实 checkpoint 资产 | docs |
| board/ | 上板包副本（bit/hwh/run 脚本/demo 资产/vec/MANIFEST/README_BOARD） | docs |
| evidence/ | 关键证据只读快照 + INDEX.md（含 sha256 与来源路径） | evidence |

## 3. 源码副本规则（core）

- 用 cp -r 复制，排除 __pycache__、*.pyc、一次性调试脚本（fix_*.py、patch_*.py、probe*.py、mon*.py、cmp_runs.py 等）。
- scripts/spikformer 收录：run_nlp_sim.py, run_physical.py, run_ip_pack.py, run_heavy.py,
  gen_nlp_common.py, gen_nlp_small.py, gen_nlp_e2e.py, gen_nlp_multihead.py,
  gen_sps_param.py, gen_sps_replay.py, gen_block_replay.py, gen_embed_replay.py,
  gen_mlp_replay.py, gen_head_replay.py, gen_vis_tok.py, gen_vis_sums.py,
  make_fixture_ckpt.py, check_contract.py, hash_upstream.py
- verification/sim/spikformer：收录全部 .v/.py，新增 verification/README.md 标注各 TB 对应线路与状态。
- 复制后立刻记录每个复制文件 sha256 清单（至少 RTL 全部 + 工具链 .py），保存到 docs/PORTABILITY.md 或 evidence/。

## 4. 可移植性改写（core，timebox 30 分钟）

现状：tools/backends/spikformer/{codegen,manifest}.py、tools/toolchain.py、tools/spikformer_train/check_ready.py、
scripts/spikformer/{run_nlp_sim,run_physical,run_ip_pack}.py、verification/sim/spikformer/{test_short,test_runtime}.py
硬编码 D:/dl_snn。发布副本内改为相对根解析：ROOT = pathlib.Path(__file__).resolve().parents[N]
（N：tools/toolchain.py=1，tools/backends/spikformer/*=3，tools/spikformer_train/*=2，scripts/spikformer/*=2，
verification/sim/spikformer/*=3），或环境变量 SNN_ROOT 覆盖。
要求：不改 RTL、不改生成算法、语义等价（同一根下生成物一致）。diff 摘要写 docs/PORTABILITY.md。
若 30 分钟无法稳定跑通，回退：保留原常量，docs/PORTABILITY.md + README 声明"快照源码根绑定 D:/dl_snn，
收拢包用于交付/复核，重跑命令以主线根执行"，禁止半改半留。

## 5. 证据清单（evidence，只读复制，总计 <= 8MB）

- build/spikformer/nlp_48h_20261029_1915/contract.json
- .../final/nlp_real_5stage_r4_holdfix/{manifest.json,numeric_contract.json,resolved_config.json,schedule.json}
- .../final/nlp_real_5stage_r4_holdfix/reports/{build_identity_20261030.json,hold_closure_20261030.json,ooc_20261030_120804.json,synth_20261030_115812.json}
- .../final/nlp_real_5stage_r4_holdfix/generated/{spikformer.xdc,spikformer_ooc.tcl,spikformer_nlp_target_top.sv,spikformer_config.vh}
- .../final/nlp_real_5stage_r4_holdfix/ip/{ip_manifest.json,SOURCE_SHA256.txt,README.txt,ooc_report.json}（>2MB 的 src 可不复制）
- .../final_gates.json、.../reports/rtl_freeze.json、.../report 下 replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json
- .../reports/{train_realweight_fidelity_real50.json,export_final_real50.json,math_tests_m1.log}
- 平台：D:/spk_platform/reports/platform_build_20261030.json、r6 时序/利用率关键文件（小文件即可）
- 上板：D:/spk_platform/board/MANIFEST.sha256、板端运行结果 JSON/日志（若存在，找 spk_platform 与 dl_snn/build 下 board* 结果）
- 图像线：sim_work/vis_sps/vsim_replay_r.log（整链 PASS 日志）等关键回放日志；docs/VISION_REFERENCE_ROUTE_RESULT_20261031.md 的副本
- evidence/INDEX.md：表格 | 证据 | 主线来源路径 | sha256 | 用途/结论 |
- 不复制大 build 树（routed DCP、synth 目录）；RELEASE_NOTES 写重建命令。

## 6. 验收（core 执行并记录）

1. 在发布根执行：E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile
   --config examples/spikformer/config/nlp_sentiment_b1.json [--weights examples/nlp_sentiment_b1/ckpt_real_real50.npz]
   --out-dir build/_quickstart_check → 生成 manifest（model_kind=real_corpus 或 fixture）；记录 params/*.bin sha256 集合。
2. 可移植性证明：整目录拷到 D:/_snn_reloc_check 后再跑同一 compile，params 哈希集合一致；完成后删除临时目录。
3. 纯 Python 自检：verification/sim/spikformer/test_short.py、tools/spikformer_train/test_integer_ref.py 记录结果。
4. 核对 NLP 冻结 RTL 18 文件哈希与 build/.../reports/rtl_freeze.json 一致（写进 docs/PORTABILITY.md）。
5. 不跑 Vivado 综合/实现/bitstream，不占 heavy lock 跑长任务。

## 7. 汇报与时间

- 各子代理 17:40 向 Lead 发一次进度（send_message target=lead），18:35 发完成/未完成清单，hard stop 18:50。
- 优先顺序（保底优先）：① 目录+源码+board+assets 可运行 → ② README/QUICKSTART/状态表 → ③ 其余文档 → ④ 附加验证。
- 不追求文档长篇；准确、可复核、命令可复制 > 篇幅。
