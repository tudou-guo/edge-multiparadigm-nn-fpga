# 面向边缘智能的多范式神经网络 FPGA 加速平台

本仓库是作品的正式提交包：在同一颗 AMD ZU15EG（xczu15eg-ffvb1156-2-i）上实现 **CNN、SNN、LNN 三个神经网络加速库**，
并统一了从模型到板卡的转译与验证方法。目录按竞赛指南 3.3.5.4 的推荐结构组织，只保留可复现所需内容。

## 1 项目简介

### 1.1 要解决的问题

边缘设备要同时处理图像、文本和时序信号，而 CNN（密集卷积）、SNN（稀疏脉冲）、LNN（连续时间递推）的计算形态差异很大。
常见做法是为每个模型单独做一套加速器，模型一换就要重做 RTL、重写驱动、重跑验证；从训练框架到板卡的链路割裂，
中间没有统一的数值合同与身份绑定，仿真、时序、板端结果难以复核。

本作品把三类模型放进同一个平台，用同一套“模型—数值合同—转译—仿真—时序—上板—证据”流程交付：换模型只需重新转译或重新导出，
RTL 库与驱动协议不变；每次转译都生成绑定源码哈希的 manifest，全部结论按 pass / fail / partial / not_run 分级记录。

### 1.2 三个加速库

| 库 | 输入 | 输出 | 关键结果 |
|---|---|---|---|
| **CNN 库** | 静态 INT8 QDQ ONNX 图 | 整图指令流、参数包、DDR 镜像、模型专用 RTL、Vivado IP Catalog 源码包 | 16 个操作码 / 11 类算子核；5 个案例（MNIST、CIFAR-10、Depthwise V2、MaxPool2D、YOLOv8n）；YOLO 103 条指令；历史 r4 平台 LUT 140,028 / DSP 471、WNS +0.075 ns；16/8 并行度板端端到端 **51.4 ms**（PS 等待约 20 ms + PL 推理约 31 ms） |
| **SNN 库** | 配置 JSON + 整数 checkpoint | 参数 blob、RTL 结构配置、目标顶层、约束与片上调度 | Spikformer 中文情感分类（E128 / H4 / T4 / seq32 / depth2）；0.7 s 转译出 54 个参数 blob、绑定 28 份 RTL 源；冻结 RTL 回放 **0 mismatch**；OOC WNS +0.135 / WHS +0.023，平台 r7 WNS +0.051 / WHS +0.011；板端 3/3（记录级） |
| **LNN 库** | 训练权重 + 配置 | int16 Q12 权重、int32 Q24 偏置、两张 1025 点 LUT、RTL ROM 初值 | CfC 单步图（1 输入 / 4 隐状态 / 1 输出，101 参数）；单 MAC 时分复用，CfC 核 1 个 DSP48E2；64 步 9088 拍、300 MHz 约 30.3 µs；板端 **64/64 与末状态逐位一致**；导出重算 **7/7 逐字节一致** |

### 1.3 方法与平台

- **统一数值合同**：CNN 为 affine-int8-v1；SNN 为全整数变体（LIF 33 位扩展、QK 位与加 popcount、AV 不归一、残差只加 0/1 脉冲）；
  LNN 为 int16 Q12 / int32 Q24 / 48 位累加；三处实现（Python 参考、寄存器模型、RTL）用同一套舍入与饱和规则复核。
- **统一证据链**：每次转译生成 manifest，绑定模型结构、参数清单、RTL 源码哈希与数值合同版本；结果按四级标注，冲突数字并列来源。
- **平台**：PS 为 ARM Cortex-A53 ×4 + PYNQ Linux（PYNQ 2.7），PL 侧含 PL DDR4、读/写 DMA 与 AXI-Lite CSR；
  应用时钟 CNN / SNN 为 200 MHz、LNN 为 300 MHz。PC 侧工具链不启动 Vivado，物理实现由构建脚本单独调用。
- **研发方式**：大模型智能体在人工主导下完成工程实现与证据整理，过程与人工把关记录见 `report/collaboration_record.md`。

## 2 目录结构

| 目录 | 内容 |
|---|---|
| `src/` | 三库源码：`cnn/`（ONNX→指令/RTL 生成器 + RTL 算子库）、`snn/`（Spikformer 工具链 + RTL）、`lnn/`（CfC 导出器 + RTL + 驱动） |
| `sim/` | 仿真与回放：ModelSim 脚本、TB、整数参考与证据 JSON |
| `build/` | 构建入口与实现报告：Tcl/PS 脚本、OOC 与平台时序/资源报告（cnn / snn / lnn 分列） |
| `board/` | 板端交付包与实测输出：`cnn/`（历史 r4 位流 + runtime）、`snn/`、`lnn/`（bit / hwh / xsa、驱动、summary） |
| `data/` | 测试数据与参考结果：CNN 样例 JSON、SNN 配置/checkpoint、LNN export 期望值 |
| `skill/` | 技能包 v2：4 个技能（cnn-translate / snn-translate / lnn-export / evidence-gate），每个 SKILL.md + reference/（命令、产物字段、分级验收、故障库、出处），含输入合同与 G0–G6 门禁；变更见 `skill/CHANGELOG.md` |
| `report/` | 设计报告、作品说明文档、大模型协作记录 |

## 3 快速入口

| 文档 | 说明 |
|---|---|
| `report/设计报告.pdf` | 设计报告：选题背景与创新点、设计原理、软硬件划分、优化对比、协作记录、技能包、复现说明 |
| `report/技术文档.pdf` | 作品说明文档：概述、系统组成、完成情况与性能参数、总结 |
| `report/collaboration_record.md` | 大模型协作记录（分工、指令设计、人工把关、协作规模、边界） |
| `skill/README.md` | 技能包索引与使用方法 |
| `build/README.md`、`board/*/README.md` | 构建入口与板端运行说明 |

## 4 复现步骤（简版）

完整步骤、命令与期望输出见 `report/设计报告.pdf` 第 7 章；以下为索引：

1. **环境**：Vivado 2021.1、ModelSim SE-64 2020.4、Python 3.9（numpy / onnx / onnxruntime / torch CPU）、PYNQ 2.7 板卡；
2. **CNN 转译**：`src/cnn` 下运行 `tools.onnx_gen` → `tools.npu_gen`，得到指令流、参数与 `ip_repo/model_0.1`；
3. **CNN 上板**：Vivado 导入 IP → 接 Block Design → 综合/实现 → 用 `board/cnn` 的位流与 `board_run_a.py` 跑固定样本；
4. **SNN 转译**：`src/snn` 下 `tools.toolchain compile / build / package`，得到 54 个参数 blob 与工程入口；
5. **SNN 验证**：按 `sim/snn` 跑 RTL 回放（期望 0 mismatch）；按 `build/snn` 的报告核对 OOC 与平台时序；
6. **LNN 导出**：`src/lnn` 下运行导出脚本（`user/export_demo.py`），期望 7/7 文件逐字节一致；
7. **LNN 上板**：整包拷贝 `board/lnn` 到板卡，运行 `lnn_cfc_run.py --checks`，期望 64/64、final_state_match=True；
8. **证据核对**：按 `report/设计报告.pdf` 表 17 的清单逐项核对（manifest、回放、时序、板端）。

## 5 边界与口径

- CNN 主线四案例的综合/时序/上板仍为 `not_run`；资源/时序取自历史 r4 实现，板测为固定样本记录；
- SNN 板端 3/3 为记录级（原始 JSON 未随包），OOC 与平台时序分开记录；
- LNN 板端离线模型 29/29，真实板测需现场复跑；64/64 与 7/7 为仓库内可复算结果；
- 所有 `pass / fail / partial / not_run` 均可在对应库的 `evidence/` 或本包 `report/` 中追溯。
