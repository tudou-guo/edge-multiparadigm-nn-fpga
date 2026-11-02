# 技能包（skill/，v2）

面向"**把神经网络模型转译成 FPGA 可执行工程，并给出可核查证据**"的技能集合。

每个技能是一份独立 `SKILL.md` + 若干 `reference/` 参考资料，可被智能体或人工按文档直接执行；
v2 相对 v1 的变化是**从"声明"变成"可执行条目"**：真实命令参数、真实清单字段、真实失败现象与修复、
可复现的验收阈值与证据路径，全部来自本提交包与开发工作区的实际产物。

## 1 技能索引

| 技能 | 输入 → 输出 | 适用库 | 关键门禁 | 主文档 |
|---|---|---|---|---|
| `cnn-translate` | 静态 INT8 QDQ ONNX → 指令流/参数/DDR 镜像/模型专用 RTL/Vivado IP 源码包 | CNN | G0 编译 → G1 ModelSim 回放 0 mismatch → G4 IP 包 | [SKILL](skills/cnn-translate/SKILL.md) |
| `snn-translate` | Spikformer 配置 + 整数 checkpoint → 参数 blob/RTL 配置/调度/约束/运行包 | SNN | G0 编译（54 blob/28 RTL）→ G1 四档回放 0 mismatch → G2 OOC → G5 平台 | [SKILL](skills/snn-translate/SKILL.md) |
| `lnn-export` | CfC 训练权重（1→4→1，101 参数）→ 单步 ONNX → int16 Q12 定点 → RTL ROM 初值 + 期望值 | LNN | G0 导出 7/7 字节一致 → G1 0 mismatch → G3 实现 → G6 板端 64 步 | [SKILL](skills/lnn-export/SKILL.md) |
| `pynq-overlay` | 任意 PL 加速器 overlay + 板卡 → 板端运行与可信结果（配对/元数据/寄存器/DMA/比对/性能/证据） | 任意 PYNQ 板卡（与题目解耦） | G6 板端 | [SKILL](skills/pynq-overlay/SKILL.md) |
| `evidence-gate` | 任意一步的结论 → 分级标签 + 证据路径 + 复算命令 + 身份哈希 | 三库通用 | G0–G6 全门级 | [SKILL](skills/evidence-gate/SKILL.md) |

每个技能的 `reference/` 至少包含：`commands.md`（可复制命令）、`artifacts.md`（产物与清单字段）、
`acceptance.md`（分级验收）、`troubleshooting.md`（故障四元组）、`sources.md`（事实出处）；
`pynq-overlay` 另含 `scaffold.md`（契约→驱动/测试骨架与报告生成）。

## 2 使用流程

1. **选技能**：按输入类型选转译技能；任何"要对外声称通过了"的步骤都叠加 `evidence-gate`。
2. **备输入**：按 `SKILL.md` 第 3 节"输入合同"逐项核对；不满足硬约束就停在输入层，不要进编译。
3. **执行**：按第 4 节步骤逐条跑；命令直接从 `reference/commands.md` 复制，别改参数。
4. **判级**：用第 6 节验收表 + `evidence-gate/reference/gates.md` 确定门级与标签；
   `pass` 必须本次执行 + 证据文件 + 复算命令，历史数据只能 `record_level`。
5. **归档**：按 `evidence-gate` 第 10 节目录约定写索引与哈希清单，30 秒可复核。

## 3 端到端执行链

```
CNN：ONNX(QDQ) ─compile─▶ 指令流/参数/DDR 镜像/模型 RTL ─pack─▶ IP 源码包 ─BD─▶ 平台实现 ─▶ 板端比对
SNN：配置+整数 ckpt ─compile─▶ 参数 blob/RTL 配置/调度 ─build─▶ 约束/Tcl ─package─▶ 运行包 ─▶ 平台 bit ─▶ 板端
LNN：训练权重 ─export─▶ 7 个 ROM/头文件 + 期望值 ─RTL 仿真─▶ 0 mismatch ─实现─▶ bit/hwh/xsa ─▶ 板端 64 步
                    每一级都要落证据（标签 + 路径 + 复算命令 + 身份哈希）
```

## 4 通用约定

### 4.1 路径

- **包内**路径相对包根写：`src/cnn/tools/onnx_gen/onnx_gen.py`、`sim/snn/evidence/INDEX.md`；
  提交包可整体搬移，包内文档不写死绝对路径。
- **工作区**路径才写绝对路径并标注 `[工作区]`：开发树为 `D:/DL_FINAL/{DL,SNN,LNN}`，不随包。
- 引用技能用相对链接；引用证据用相对 `evidence/` 的路径。

### 4.2 结论标签

只允许 `pass / fail / partial / not_run`，另有两个限定字段：

| 限定 | 含义 | 用法 |
|---|---|---|
| `record_level` | 历史或客方回传的数据 | 须注明来源与日期，不与本机实测混写 |
| `measured / derived` | 实测 / 静态推导估算 | 资源、时序、时延 |

禁止措辞："基本通过""应该没问题""大致一致""已基本完成"。

### 4.3 身份绑定

每个 `pass` 至少绑定三类哈希之一以上：**输入身份**（模型/权重/源码清单）、**产物身份**（bit/hwh/xsa/DDR 镜像/IP 包）、
**清单身份**（`manifest_sha256`）。证据随身份失效：源码/RTL/权重一变，旧证据进 `invalidated_runs`。

### 4.4 平台默认值（三库一致）

| 项 | 值 |
|---|---|
| 器件 | `xczu15eg-ffvb1156-2-i` |
| Vivado | 2021.1 |
| NPU/CNN 编译目标频率 | 200 MHz（5 ns） |
| SNN 约束 | 5 ns；OOC 与平台均不得有 false path / multicycle 例外 |
| LNN 应用时钟 | 300 MHz 单时钟域（平台派生工程） |
| Python | 3.9.x（CNN：onnx 1.14.1 / numpy 2.0.2；SNN：torch 2.8.0+cpu） |

### 4.5 环境边界

- 本包**不含**模型训练数据与完整案例（`src/cnn/examples/` 目录未随包），CNN 编译需自备 ONNX 与样本；
  随包的真实清单样例在 `data/cnn/mobiledw/`。
- 本包**不含** Vivado 工程与临时构建目录，综合/实现/板端需按 `build/*/README.md` 重建并重新验收。
- 依赖缺失时按 `evidence-gate` 的判据记录依赖与结论边界，不要用推断填结论。

## 5 失效条件（通用）

- 模型图/结构与技能声明的合同不符（算子、位宽、shape 超出范围）→ 在输入层拒绝，不静默回退；
- 工具链版本/器件/时钟与技能内的平台假设不同 → 先改技能的平台假设，再执行；
- 缺少对应库的期望值/证据目录（如 LNN 无 `export/` 期望值）→ 驱动会拒绝运行，按失效处理；
- 证据与源码/RTL 版本不匹配 → 结论作废，重跑。

## 5.1 已知不一致（随包状态，读技能前先知道）

| 项 | 现状 | 对执行的影响 |
|---|---|---|
| `src/cnn/examples/` 未随包 | `src/cnn/README.md`、`QUICKSTART.md` 仍链接 `examples/README.md` | CNN 技能的命令用 `<model_qdq.onnx>` 占位，样例需自备；随包清单样例在 `data/cnn/{tinycnn,mobiledw,yolo}/` |
| 三个库均无 `VERSION` 文件 | `src/{cnn,snn,lnn}/README.md` 引用 `VERSION` | 技能里的版本身份改用 README 首行版本串 + 清单哈希，不引用不存在的路径 |
| `sim/snn/evidence/` 现 142 个文件 | `INDEX.md` 记录 137 条目（133 个计入 SHA256SUMS），差额为新增索引脚本 | 复算仍应 `files=133 bad=0`（2026-11-02 实测） |

## 6 维护规则

1. 每次改动工具链/操作码/RTL 冻结集，同步 bump 技能 `version`（major=合同变化，minor=新增步骤/字段）；
2. `last_verified` 改为实际复跑日期；未复跑的技能不得升 major；
3. 新增故障必须先进 `reference/troubleshooting.md` 四元组，再决定是否上升为 SKILL 主体条目；
4. `reference/sources.md` 必须能指向真实文件；被删除的源文件要同步清理引用；
5. 技能包自身也受 `evidence-gate` 约束：本文件的每条数字都能在包内找到出处。

## 7 版本

- 当前：**2.1.0（2026-11-02）**——新增 `pynq-overlay`（通用板端技能）；4 个既有技能维持"只讲判据与执行方法"；
- 上一版：1.0.0——4 份声明式短文档（30~40 行/份）；
- 变更明细见 [CHANGELOG.md](CHANGELOG.md)。
