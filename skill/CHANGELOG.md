# 变更记录

## 2.1.0 — 2026-11-02

**新增通用板端技能 \`pynq-overlay\`（与题目、板卡解耦）。**

- 覆盖：三件套配对 → Overlay 加载与 ip_dict 元数据核对 → 寄存器映射 → 内存分配与 DMA/缓存一致性 → 黄金比对 → 性能口径 → 证据归档；
- 通用性：只依赖 \`pynq.Overlay\`/\`pynq.allocate\`/MMIO 与 hwh 元数据；换板卡只替换 bit/hwh 与地址/时钟常量（复用清单见 \`reference/acceptance.md\` §3）；
- 判据与故障来自本队三块板卡的真实驱动与记录（LNN CSR 表、CNN heads 比对、SNN 逐样本比对、BUSY 采样竞态、综合后错误码等）；
- 交付：\`SKILL.md\` + \`reference/{commands,artifacts,acceptance,troubleshooting,sources}.md\`。

## 2.0.1 — 2026-11-02

**表述收敛：技能正文只保留"输入合同 / 执行步骤 / 判据 / 故障 / 出处"，不再在正文里汇报某次执行状态。**

- 删除各技能正文中的"包内状态 / 本次结果 / 当前状态"列与"本次实际跑通"段落；
- G0–G6 门禁表改为纯判据（判定命令 + 阈值 + 证据落点），gates.md 同步重写；
- "未跑/未执行"类标注替换为依赖说明（如"需 Vivado 2021.1 + 同源工程根"）；
- 保留数据字段原值（如 manifest 中的 board_status 取值）与历史数值的来源日期；
- 真实状态记录仍在包根 `README.md` 边界段与 `sim/*/evidence/`，技能包不再重复。

## 2.0.0 — 2026-11-02

**主题：从"声明式短文档"升级为"可执行、可验收、有证据的操作手册"。**

### 新增结构

- 每个技能：`SKILL.md`（固定 11 节骨架）+ `reference/{commands,artifacts,acceptance,troubleshooting,sources}.md`；
- `evidence-gate` 额外提供 `templates/evidence_index.md`、`templates/acceptance_record.md`；
- 包级 `README.md` 增加技能索引、端到端执行链、通用约定（路径/标签/身份/平台默认值）、维护规则。

### 内容扩展（每个技能）

- 输入合同：格式/位宽/shape/算子/配置字段的硬约束与"不满足时的行为"；
- 真实命令：来自实际 CLI（`--help`）与随包脚本，含完整参数、期望输出、失败分支；
- 产物与清单：逐个文件 + 字段表 + 真实取值（来自 `data/`、`build/`、`sim/*/evidence/`）；
- 分级验收：G0–G6 门级、阈值、证据落点，明确区分"本次实测"与"历史记录级"；
- 故障库：现象/根因/处置/证据四元组，来自开发期真实事故；
- 事实出处：`reference/sources.md` 逐条登记。

### 素材来源

- 提交包 `D:/FINAL`（权威、可移植）；
- 开发工作区 `D:/DL_FINAL`（`DL/`、`SNN/`、`LNN/` 的 docs/tools/verification/evidence）；
- codex 工作区与会话日志（`D:/dl_v0.1`、`D:/dl_v1`、`D:/dl_v1.0`、`D:/dl_snn`、`C:/Users/27934/Documents/ChatGPT/LNN`、
  `~/.codex/sessions` 33 个会话）——用于真实命令、踩坑与决策演进。

### 相对 v1 的修订

- 删除"基本通过/应该没问题"类表述，统一为四种标签；
- 修正随包不存在的路径引用（`src/cnn/examples/` 为空目录）；
- 明确 IP 包 ≠ 平台 ≠ bitstream、记录级 ≠ 实测、估算 ≠ 资源结论；
- 增加"包内/工作区"路径标注规范。

## 1.0.0 — 2026-10（项目结束后初版）

4 个技能各 30~40 行：适用、输入、步骤、验收、边界、失效条件。
