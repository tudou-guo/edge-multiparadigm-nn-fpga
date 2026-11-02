# 事实来源索引

> 本技能每条关键结论对应的源文件/字段。核验时按行回查。

| 结论 | 出处 |
|---|---|
| 标签只有 pass/fail/partial/not_run；禁止"基本通过" | `skill/README.md` v1 的通用约定 + `report/collaboration_record.md` §2「验收口径」 |
| 板测 3/3 为记录级、原始 JSON 未随包 | 包根 `README.md` 边界段；`sim/snn/evidence/INDEX.md` 板测行 |
| 30 秒复核、无无源数字 | v1 `skills/evidence-gate/SKILL.md` 验收段；本包 `sim/lnn/evidence/ACCEPTANCE.md` 的"实测/证据位置"结构 |
| SHA256SUMS 复算 recipe（files/bad） | `sim/snn/evidence/INDEX.md` §0 校验方法（原文可复制命令） |
| 133 个文件 / 1063.2 KB / 8MB 预算 | `sim/snn/evidence/INDEX.md` 开头说明 |
| COPY_SHA256.txt 150 文件 + 2 行注释 | `sim/snn/evidence/ACCEPTANCE_CORE.md` §A5、§A7.2 |
| 三类哈希（文件/manifest_sha256/集合摘要） | `ACCEPTANCE_CORE.md` §A1（`000b0b5b…`、`315171a2…`、`c8a45b94…`、`835ad38d…`） |
| 确定性复跑方法 | `ACCEPTANCE_CORE.md` §A1（另起目录同 seed → 字节相同后删除） |
| invalidated_runs | `sim/snn/evidence/vision/ref_route/native_gate_result_20261031.json` |
| 冻结哈希 + SAME/DIFF 比对 | `sim/snn/evidence/nlp/rtl_freeze.json`、`rtl_hash_check_20261031.txt` |
| 配对检查（bit/XSA/hwh） | `sim/lnn/evidence/ACCEPTANCE.md` §3 |
| CNN 历史 r4 实现数值（WNS +0.075 等） | `board/cnn/MANIFEST.json` `identity.implementation` |
| CNN 回放 schema 与 board_status=not_run | `data/cnn/mobiledw/summary.json` |
| SNN OOC 数值（+0.135/+0.023，LUT 40964…） | `sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json` |
| SNN 平台 partial（CDC 1 Critical） | `sim/snn/evidence/platform/platform_build_20261030.json` |
| LNN 构建数值（+0.226/+0.011，57464/57464） | `sim/lnn/evidence/ACCEPTANCE.md` §2 |
| LNN 离线 29/29 = 流程自检 | `sim/lnn/evidence/ACCEPTANCE.md` §1、§5 |
| IP 包 ≠ BD ≠ bitstream | `src/cnn/docs/PLATFORM_AND_STATUS.md` |
| 每次编译用新输出目录 | `src/cnn/docs/USER_GUIDE.md` |
| 估算 vs 实测（846k FF） | `sim/snn/evidence/vision/native_resource/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md` |
| 重活锁/内存采样 | `sim/snn/evidence/vision/ref_route/ooc_native/heavy_wait_latest.json`、`memwatch_latest.log` |

## 核验边界

- 所有 Oligo/Vivado 类结论（G2–G6）来自证据文件，本机无 Vivado/ModelSim/板卡，未复跑；
- 复算类结论（SHA256、索引格式）已在包内证据上验证可执行。
