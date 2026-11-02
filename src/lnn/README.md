# LNN 发布快照 1.0.0 — CfC 液态神经网络 PYNQ 演示

- 版本：`lnn-cfc-demo-1.0.0`（见 `VERSION`）
- 收录时刻：**2026-10-31**
- 来源：**D:/DL_LNN**（开发树，保持不变；本快照为面向用户的只读收录）
- 发布根：**D:/DL_FINAL/LNN**（与 `D:/DL_FINAL/DL`、`D:/DL_FINAL/SNN` 并列，互不修改）

## 1. 这是什么

把 **CfC（Closed-form Continuous-time）液态神经网络** 的**单步计算图**映射为可综合 RTL，在 ZU15EG 的 PL 上
连续完成 64 步时序推理，并通过 PYNQ + AXI-Lite 由 PS 主机驱动与展示。

- 网络：CfC default 门控单元，无 backbone；输入 1 / 隐状态 4 / 输出 1；84 权重 + 17 偏置 = 101 参数
- 任务：单频正弦输入，输出 = 输入延迟 4 步；8 组固定相位样本 × 64 步，跳过前 5 步评估
- 硬件：单时钟域 CfC 子系统（CSR / 序列控制 / 时分复用 MAC / 权重 ROM / 激活 ROM / 状态寄存器 / 输入输出 RAM）
  挂在平台 `npu_clk = 300 MHz`，CSR 窗口 `0x0480000000`（4 KiB）
- 器件：`xczu15eg-ffvb1156-2-i`；Vivado 2021.1；PYNQ overlay
- 数值：数据/权重/隐状态 int16 Q12，偏置 int32 Q24，48 位精确累加，1025 点 tanh/sigmoid 查表，Δt = 655（Q12）

## 2. 目录结构

| 路径 | 内容 |
|---|---|
| `README.md` `QUICKSTART.md` `RELEASE_NOTES.md` `VERSION` `environment.md` | 发布说明、快速开始、版本与已验证结果、环境 |
| `board/` | **上板交付包**：`lnn_cfc.bit` / `.hwh` / `.xsa`、PYNQ 驱动、整数参考模型、演示 Notebook、期望值 `export/`、`README_BOARD.md`、`SHA256SUMS` |
| `model/` | 单步 ONNX、训练权重（npz）、训练配置、浮点参考输出、训练曲线与对比图 |
| `export/` | 数值导出（权重/偏置/激活表 hex、gen/ 期望向量与样本、manifest.json、整数参考 `int_reference.py`、`cfc_spec.py`）；工程/仿真用副本 |
| `rtl/` | 8 个可综合 CfC RTL 模块（MAC、ROM、状态、cell 核、序列控制、AXI-Lite、平台封装） |
| `verification/` | 3 个自检测试台 + 整数参考模型 + 编译列表（xsim） |
| `scripts/` | `run_sim.tcl`（一键仿真）、`env_check.py`（环境与哈希检查） |
| `docs/` | `USER_GUIDE.md`（部署/演示/验收/排错）、`INTERFACE.md`（CFC1 寄存器与数值接口） |
| `evidence/` | `ACCEPTANCE.md`（验收数字与哈希索引）、`offline_validation_summary.json`（离线校验原始结果） |
| `MANIFEST.sha256` | 本快照全部文件哈希（`python scripts/env_check.py --check-hashes` 可复核） |

> `board/export/` 是板端部署用的自包含副本，`export/` 是工程/仿真用的数值导出，两者内容一致。

## 3. 已验证结果（实测，非估算）

| 项目 | 阈值（需求文档 §6.1） | 实测 | 结论 |
|---|---|---|---|
| 浮点模型任务 MSE（8 样本，跳过前 5 步） | ≤ 0.005 | **1.234e-05** | PASS |
| 单步 ONNX vs PyTorch 最大绝对误差 | ≤ 1e-5 | **4.77e-07**（Gemm×5/Concat/Tanh/Sigmoid/Mul/Add/Sub，opset 17） | PASS |
| 定点参考 vs 浮点输出 MSE | ≤ 0.001 | **1.172e-04**（最大绝对误差 0.0476） | PASS |
| RTL 逐位仿真（64 步输出 + 每步 4 状态） | mismatches = 0 | **0**（cell 21 用例 / seq 626 checks / protocol 155 checks） | PASS |
| 300 MHz 时序（npu_clk 单域） | WNS/WHS ≥ 0 | **WNS +0.226 ns / WHS +0.011 ns / WPWS +0.039 ns**，0 失败端点 | PASS |
| 布线 / DRC | unrouted 0、0 Error | 57464/57464 fully routed、routing errors 0、DRC 4 Warning / **0 Error** | PASS |
| bit / hwh / XSA 配对 | 同 stem、同一次构建 | XSA 内嵌 bit 与 `board/lnn_cfc.bit` 哈希一致；`hwh` == XSA/`hw_handoff` | PASS |
| 板端（真实 ZU15EG） | 64/64 输出 = 定点参考 | 由现场执行；驱动/期望值/校验流程已就绪（见 `docs/USER_GUIDE.md` §4） | 现场验收 |
| 离线寄存器模型（开发机） | 64/64 | **29/29 用例 PASS**（`source=simulation-model`，非板端） | PASS |

关键文件哈希：

| 文件 | 字节 | SHA256 |
|---|---|---|
| `board/lnn_cfc.bit` | 28,700,861 | `0fcd8d65d0f8a101d795270741ca7c0cef21f1904590de78226a094b74be30a9` |
| `board/lnn_cfc.hwh` | 380,358 | `fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb` |
| `board/lnn_cfc.xsa` | 2,203,972 | `f01d6a8af07fc74acae3dbc30e362db333151978064d3584fcdfc611dbc27032` |

## 4. 从这里开始

- [快速开始](QUICKSTART.md)
- [用户指南](docs/USER_GUIDE.md)
- [寄存器与数值接口](docs/INTERFACE.md)
- [上板说明](board/README_BOARD.md)
- [版本与已验证结果](RELEASE_NOTES.md)

## 5. 边界与限制

1. 本快照**不包含板端实测数据文件**：板端结果需现场用 `board/lnn_cfc_run.py` 生成（`summary.json`，`source=hardware`），
   并建议放入 `evidence/board_run/` 后归档。离线模型结果（`source=simulation-model`）不得当作板端实测。
2. 位流为单次构建产物；若改动 RTL/数值导出，需重新综合实现并重新验收时序（本包时序余量：WNS +0.226 / WHS +0.011 ns）。
3. DRC 4 条 Warning 全部来自平台既有 IP（dbg_hub / MIG），应用逻辑 critical warning 为 0。
4. 性能：64 步片上计算 RUN_CYCLES = 9088（300 MHz 下约 30.3 µs）；端到端耗时以板端实测为准，
   首版不承诺相对 Python 的加速倍数（PYNQ MMIO 开销主要在 PS 侧）。
5. 门控公式采用需求文档 §2.1 的完整形式 `g = sigmoid(a·Δt + b)`、`b = W_b z + b_b`（实现期间修正过一次
   遗漏 W_b 项的早期约定，已重做全部 golden 与验证）。
