# 验收证据索引 — LNN CfC PYNQ 演示 1.0.0

本文件汇总本快照各项验收的**实测数字**与**原始证据位置**。原始日志/报告保留在开发树 `D:/DL_LNN`（只读参考）。

## 1. 验收对照（需求文档 §6.1）

| 层级 | 通过条件 | 实测 | 结论 |
|---|---|---|---|
| 浮点模型 | 8 条固定序列，skip5 MSE ≤ 0.005 | **1.234e-05** | PASS |
| ONNX | 两输出 vs PyTorch ≤ 1e-5 | **4.768e-07**（256 组随机用例；算子 Gemm×5/Concat/Tanh×2/Sigmoid/Mul×3/Add×2/Sub，opset 17） | PASS |
| 定点参考 | 相对浮点 MSE ≤ 0.001 | **1.172e-04**（max abs 0.0476；饱和 0、LUT 夹紧 0） | PASS |
| RTL 单步与序列 | 64 步输出与每步 4 状态逐位一致 | **0 mismatch**（tb_cfc_cell 21 用例；tb_cfc_seq 626 checks；RUN_CYCLES=9088） | PASS |
| 协议与状态 | 定义用例全部通过 | **155 checks / 0 mismatch**（tb_cfc_protocol） | PASS |
| 构建 | WNS/WHS ≥ 0、unrouted 0、DRC 0 Error、bit/hwh 配对 | **WNS +0.226 / WHS +0.011 / WPWS +0.039 ns**；0 失败端点；57464/57464 全布线；DRC 4 Warning / 0 Error；配对 SHA256 校验 PASS | PASS |
| 板端 | 输出/末状态/ID/STEP_COUNT | 未随包提供实测文件（现场执行，见 `docs/USER_GUIDE.md` §4；离线模型 29/29 PASS 仅作流程自检） | 现场验收 |

## 2. 构建事实（位流来源）

| 项 | 值 |
|---|---|
| 器件 | xczu15eg-ffvb1156-2-i（-2，I 级） |
| 工程 | 由平台 `D:/spk_platform/project_1` 派生的 LNN 工程（BD design_1 未改动） |
| 应用时钟 | npu_clk 300 MHz 单时钟域；无 false_path / clock group 例外 |
| 实现策略 | Performance_ExplorePostRoutePhysOpt；route 后 `phys_opt_design -directive Explore` 实跑（日志可查 2 次） |
| 综合 | 0 error / 0 critical warning（应用逻辑 own = 0） |
| 资源（final） | CLB LUT 25,221（7.39%）；FF 32,890（4.82%）；DSP48E2 4；BRAM Tile 28.5；URAM 0 |
| DRC | 4 条 Warning（PDCN-1569 ×3、RTSTAT-10 ×1，均为平台既有 IP），0 Error / 0 Critical |
| 关键路径 | 最差 setup 为平台复位分配网络 → CfC 寄存器复位端；CfC 乘加逻辑非关键路径 |

## 3. 交付文件哈希

| 文件 | 字节 | SHA256 |
|---|---|---|
| `board/lnn_cfc.bit` | 28,700,861 | `0fcd8d65d0f8a101d795270741ca7c0cef21f1904590de78226a094b74be30a9` |
| `board/lnn_cfc.hwh` | 380,358 | `fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb` |
| `board/lnn_cfc.xsa` | 2,203,972 | `f01d6a8af07fc74acae3dbc30e362db333151978064d3584fcdfc611dbc27032` |
| `model/onnx_model.onnx` | 1,406 | `2108484039d91c2943d981f8c2410c4eff324b5933d0d5ff940fa496a160978b` |
| `export/manifest.json` | — | `2b4196a273d614788b020ae99a2829924d27575b29adbd14df1b99422745ae0a` |

配对检查：XSA 内嵌 bitstream 与 `board/lnn_cfc.bit` 哈希一致；XSA 内 `design_1.hwh`、`board/lnn_cfc.hwh` 与 Vivado hw_handoff 哈希一致（逐字节）。

## 4. 原始证据（开发树，不随包）

| 内容 | 位置 |
|---|---|
| 最终报告与验收对照 | `D:/DL_LNN/reports/FINAL_REPORT.md` |
| 浮点模型 | `D:/DL_LNN/reports/float_model_report.md`、`model/logs/train.log` |
| ONNX 验证 | `D:/DL_LNN/reports/onnx_verify_report.md`、`export/onnx_verify.json` |
| 定点参考 | `D:/DL_LNN/reports/int_reference_report.md` |
| RTL 仿真 | `D:/DL_LNN/reports/sim_report.md`、`D:/DL_LNN/sim/logs/` |
| 构建 | `D:/DL_LNN/reports/vivado_build_report.md`、`reports/vivado_final_*.rpt`、`reports/vivado_build_metrics.txt` |
| 离线校验 | `D:/DL_LNN/reports/board_package_report.md`、`board/offline_check/offline_summary.json` |
| 交付清单（开发树全域） | `D:/DL_LNN/reports/DELIVERY_MANIFEST.json` |

## 5. 复现要点

- 仿真：`vivado -mode batch -notrace -source scripts/run_sim.tcl`（期望 CFC_SIM_PASS）
- 离线驱动自检：`cd board && python lnn_cfc_run.py --sample sample_000 --checks`（source=simulation-model，非板端）
- 位流重建需要 Vivado 工程与平台 BD（不随包）；重建后须重新验收时序与 bit/hwh 配对。
