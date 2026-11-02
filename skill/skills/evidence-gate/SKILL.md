---
name: evidence-gate
version: 2.0.1
description: 对转译/仿真/综合/实现/IP 打包/平台集成/板端任何一步做分级门禁与证据归档——结论必须带标签、证据、复算命令与身份哈希，禁止把仿真当实测、把历史当本次。
applies_to: [all]
package_paths: [src, data, sim, build, board, report]
requires: [python>=3.9（hashlib 即可复算）, 可选: sha256sum / Vivado / ModelSim]
last_verified: 2026-11-02
---

# SKILL: evidence-gate

## 0 一句话定义

把"某一步到底过没过"写成可机检的断言：**结论标签 + 证据文件路径/字段 + 复算命令 + 身份哈希**，四件缺一即不算通过。
本技能只做判定与归档，不代替编译/仿真/综合/板端工具。

## 1 何时使用 / 何时不要使用

**使用**：

- 任一转译步骤（CNN 编译、SNN 编译、LNN 导出）完成后，要对外声明"通过了"；
- 仿真、OOC 综合、平台实现、IP 打包、板端比对任一步要写进报告或交接；
- 把历史结论（旧报告、旧位流、旧板测）搬进新包、新版本、新报告之前；
- 需要解释"为什么这里写 not_run 而不是 pass"时。

**不要使用**：

- 不能替代 vivado / vsim / 板端驱动，本技能不产生新的物理证据；
- 不发明指标口径（"硬件耗时 vs 端到端"由各库文档定义，如 `src/snn/docs/NUMERIC_CONTRACT.md`、`src/lnn/docs/USER_GUIDE.md`）；
- 不要用本技能"美化"结论：标签只有 `pass / fail / partial / not_run` 四种，另加 `record_level`（记录级）与 `measured / derived`（实测/推导）两个限定字段。

## 2 前置条件与环境自检

| 项 | 要求 | 自检方式 |
|---|---|---|
| 证据文件 | 必须来自**本次**运行；与当前源码/RTL/位流**同一身份** | 比对 `manifest_sha256`、`rtl_sha256`、`VERSION` |
| 复算环境 | Python ≥ 3.9 的 `hashlib` 即可；不需要第三方库 | `python -c "import hashlib;print('ok')"` |
| 重活串行 | 同一台机同时只跑一个 Vivado/ModelSim 重活，冲突时记录等待与外部 PID | 见 `reference/troubleshooting.md`「重活锁」 |
| 只读原始证据 | 归档目录内的原始证据不得改写、不得重命名（改名=换身份） | `sha256sum -c` / 复算脚本 bad=0 |

复算现有索引（包内可直接跑，SNN 证据快照的真实 recipe）：

```bash
cd sim/snn/evidence    # [包内] 包根下相对路径；开发树为 D:/DL_FINAL/SNN/evidence
python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"
```

期望输出形如 `files=133 bad=0`（数字随快照而定，`bad=0` 是硬要求）。

## 3 输入合同

| 输入 | 最低要求 | 不合格时的行为 |
|---|---|---|
| 结论声明 | 一句话，主语是"哪一步、哪个对象、哪个版本" | 拒绝归档，先补对象与版本 |
| 证据文件 | 本次运行的 JSON/日志/报告，含时间戳与执行者 | 标 `not_run`，不得用旧文件顶替 |
| 身份哈希 | 源码哈希（`rtl_sha256`/源码清单）、产物哈希（bit/hwh/xsa/DDR 镜像）、清单自描述哈希（`manifest_sha256`） | 缺哈希 → 结论降为 `partial` |
| 环境记录 | 工具版本、器件、时钟、OS、Python 路径 | 缺环境 → 结论不可跨机复现，标 `partial` |
| 复现命令 | 可直接粘贴的原文命令（含工作目录） | 缺命令 → 该证据不满足"30 秒复核" |

三件套/多件套一致性是**硬门**：CNN `bit+hwh+xsa`、SNN `bit+HWH`、LNN `bit/hwh/xsa` 必须同一次构建；
LNN 的配对检查已给出可复制做法（XSA 内嵌 bitstream 与 `.bit` 比对、`design_1.hwh` 与 `.hwh`/hw_handoff 逐字节比对）。

## 4 执行步骤

### 步骤 1：定位门级

先查 `reference/gates.md` 的门禁矩阵，确定这条结论属于哪一级：
`G0 编译/导出 → G1 数值与回放 → G2 OOC 综合 → G3 实现布线 → G4 IP 打包 → G5 平台集成 → G6 板端`。
**级不能跳**：G1 通过不代表 G2 通过；G3 的 r4 历史数据不能证明本次 G3。

### 步骤 2：打标签（唯一合法取值）

| 标签 | 含义 | 附加要求 |
|---|---|---|
| `pass` | 本次执行、证据齐、阈值达标 | 必须给出复算命令与阈值来源 |
| `fail` | 本次执行、明确不达标 | 必须给出失败现象与日志位置；不得"待观察" |
| `partial` | 部分子项通过，或产物齐但未复跑 | 逐条列出通过/未通过子项 |
| `not_run` | 未执行（含环境缺失、被规范禁止） | 必须写原因与下一步 |
| `record_level`（限定字段） | 历史记录存在但本次未复跑（如板端 3/3 记录、原始 JSON 未随包） | 必须与 `pass` 区分；**不得**计入 pass |
| `measured / derived`（限定字段） | 数字来自实测，还是静态推导/估算 | 估算值不得写成资源/时序结论 |

### 步骤 3：绑定身份

对每条 `pass` 记录至少三个哈希（缺一降级）：

1. 输入身份：模型/权重/checkpoint/配置的 sha256，以及源码清单哈希（`rtl_sha256`、`VERSION`、`cache_key`）；
2. 产物身份：`bit/hwh/xsa`、DDR 镜像、IP `package_sha256`、`params/*.bin` 集合摘要；
3. 清单身份：`manifest_sha256`（清单自描述哈希，≠ 清单文件的 sha256）。

规则：**证据随身份失效**。源码/RTL/权重任一变化，旧证据进入 `invalidated_runs`（SNN 的 `tb_sps_native_gate_result_*.json` 就带这个字段），
新结论必须重跑；禁止"旧报告配新源码"。

### 步骤 4：写证据条目

一条证据一个条目，字段固定（与 `templates/evidence_index.md` 对应）：

```
主题 | 标签 | 门级 | 证据路径（相对包根） | 复算命令 | 身份哈希 | 时间 | 执行者 | 备注（record_level/measured）
```

### 步骤 5：生成索引与目录自校验

1. 写 `<lib>/evidence/INDEX.md`（全量状态摘要，一行一事实）与 `SHA256SUMS.txt`（文件哈希清单）；
2. 复算 `SHA256SUMS.txt`：`bad=0`；
3. 在报告/交接中只引用索引条目，不另起数字；
4. 索引末尾写"未执行（not_run）"表与"更新记录"（初版、复跑、最终刷新）。

## 5 产物与清单

| 产物 | 位置（包内） | 关键字段/格式 |
|---|---|---|
| 状态索引 | `sim/snn/evidence/INDEX.md`、`sim/lnn/evidence/ACCEPTANCE.md`、包根 `README.md` 的「边界」段 | 一行一事实；每行必须带证据链接 |
| 哈希清单 | `sim/snn/evidence/SHA256SUMS.txt`、`sim/snn/evidence/COPY_SHA256.txt` | `<sha256>  <相对路径>`，行首 `#` 为注释；开发期 SNN 快照 133 个文件计入 SHA256SUMS、COPY_SHA256 为 150 文件 + 2 行注释 |
| 验收记录 | `sim/snn/evidence/ACCEPTANCE_CORE.md` 形态 | 执行人/执行根/时间/环境 → 每项命令+结果表 → 未执行表 → 刷新与清理记录 |
| 交接/复核文档 | `report/`、各库 `docs/` 下 `*_PROGRESS_*.md` | 根因、measured_vs_derived、invalidated_runs、not_run 清单 |
| 索引模板 | `templates/evidence_index.md`、`templates/acceptance_record.md` | 复制后填字段，不得删行 |

## 6 验收（本技能自身）

自检四项，全部满足才算"证据闭环"：

1. **30 秒复核**：随机抽 3 条 `pass`，用条目里的命令能算出条目里的数字；
2. **无源数字为零**：全文搜索数字，无出处的删掉或降级为 `derived`；
3. **记录级未混入 pass**：所有 `record_level` 条目在索引里单独成段或在标签后标注；
4. **自校验通过**：`SHA256SUMS.txt` 复算 `bad=0`，索引条目与文件一一对应。

## 7 故障与修复（摘要，详见 `reference/troubleshooting.md`）

| 现象 | 根因 | 处置 |
|---|---|---|
| 报告里的"综合通过"其实是历史 r4 数据 | 把历史结果当本次 | 改成 `record_level`，补日期与身份；本次未跑标 `not_run` |
| 复算 `manifest_sha256` 不一致 | 缓存/运行参数/源清单变化 | 清缓存、固定参数与路径、写检查单后重跑 |
| 板端"3/3 通过"被写成 pass | 原始板端 JSON 未随包 | 标 `record_level=true` 且结论 `not_verified` |
| 平台集成写成 pass | CDC 1 Critical 未收敛、软复位不排空 DMA | 只能 `partial`，逐条列未收敛项 |
| RTL 刷新后仍引用旧回放 | 证据未与身份绑定 | 维护 `invalidated_runs`，重跑或标失效 |
| bit 与 hwh 不同次构建 | 分两次导出 | 配对哈希检查；不一致直接 `fail` |
| 估算资源写成实测 | 静态推导（如 ≈846k FF）被当结论 | 标 `derived`，只写"估算，未测" |
| IP 包验收写成"平台完成" | IP 包≠BD≠bitstream | 按 G4/G5/G6 分级，缺 BD/bit 的标 `not_run` |

## 8 边界

- 不产生物理证据，不代替 `vsim`/`vivado`/`pynq`；
- 不重定义指标口径；不合并冲突来源（并列列出，不取平均）；
- 不修改原始证据文件；索引与清单可重建，原始证据只读；
- `fail` 不是失败，隐瞒 `fail` 才是失败。

## 9 版本与身份绑定

- `manifest_sha256`：清单**自描述**哈希（对清单内容规范化后计算），用于绑定"这次编译的输入+配置+RTL"；
- 文件 sha256：清单/bit/hwh/xsa 等**字节**哈希，用于搬运与配对；
- 集合摘要：`sha256sum params/*.bin | sort -k2` 结果的 sha256，用于 28/54 个参数 blob 的整体一致性（SNN 实践）；
- `VERSION`/`cache_key`/`build_identity`：工具链版本与构建身份，升级即换版本号，不覆盖旧记录；
- 一切证据默认与 `<lib>/VERSION`、`rtl_freeze.json`、`build_identity_*.json` 绑定，缺绑定视为无身份。

## 10 证据归档

```
<lib>/evidence/            # 包内实际位于 sim/<lib>/evidence/
├── INDEX.md                 # 状态索引（一行一事实）
├── SHA256SUMS.txt           # 文件哈希清单（bad=0）
├── ACCEPTANCE*.md           # 验收记录（可选，命名带日期）
├── <topic>/                 # 按主题分目录：nlp/ platform/ vision/ board/ ...
└── _copy_manifest.json      # 复制清单（复制了哪些文件、从哪来）
```

规则：新证据放新目录、新文件名带日期或 run-id；禁止覆盖旧证据；`build/` 与 `evidence/` 不参与哈希清单的排除规则要写进记录（SNN 快照即"排除 evidence/ 与 build/"）。

## 11 参考文件索引

- `reference/gates.md`：G0–G6 门禁矩阵（按 CNN/SNN/LNN 列出阈值、入口、证据样例）
- `reference/commands.md`：复算命令矩阵 E1–E10（哈希清单、单文件、确定性复跑、集合摘要、配对、字段提取、30 秒复核模板）
- `reference/artifacts.md`：证据产物字段表（INDEX/SHA256SUMS/ACCEPTANCE/_copy_manifest/单条结论 JSON/命名规则）
- `reference/evidence-format.md`：索引/清单/哈希/集合摘要的格式与算法
- `reference/acceptance.md`：本技能自检清单 A1–A9（30 秒复核、无源数字、记录级隔离、自校验、失效检查、措辞）
- `reference/troubleshooting.md`：证据类故障四元组（现象/根因/处置/证据，14 条）
- `reference/sources.md`：本技能每条结论的出处
- `templates/evidence_index.md`：索引模板
- `templates/acceptance_record.md`：验收记录模板
