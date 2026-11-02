# DL FPGA 模型工具链

**发布整理快照：0.3.0（2026-10-31）。** 本目录汇集 `D:\dl_v0.1` 和 `D:\dl_v1` 中面向用户的编译器、RTL 算子库、验证入口、模型案例和历史板卡包。原目录保持不变。

主流程接收静态 INT8 QDQ ONNX，生成指令流、参数和 DDR 镜像、模型专用 RTL 顶层以及源码身份清单；Vivado 可用时还会生成 NPU IP Catalog 源码包。该 IP 包是 RTL IP 组件，不包含板级 Block Design、bitstream，也不代表已完成板级验证。

## 从这里开始

- [快速开始](QUICKSTART.md)
- [工具入口](tools/README.md)
- [用户指南](docs/USER_GUIDE.md)
- [支持的算子](docs/OPERATORS.md)
- [平台与案例状态](docs/PLATFORM_AND_STATUS.md)
- [案例目录](examples/README.md)

## 目录结构

| 路径 | 内容 |
|---|---|
| `tools/onnx_gen/` | ONNX 前端、量化合同、图降低、规划、代码和指令生成 |
| `tools/npu_gen/` | NPU RTL 模板和 Vivado IP Catalog 打包器 |
| `tools/quant/` | 为旧案例保留的 ABI 1.0 量化参数工具 |
| `rtl/` | 主线算子库、NPU 公共逻辑、ABI 定义和厂商 IP 描述 |
| `verification/` | ModelSim 案例流程和 MaxPool2D 功能检查 |
| `examples/` | TinyCNN、MobileDW、Depthwise V2 候选、MaxPool2D、YOLO 及旧案例参考包 |
| `scripts/` | 环境检查脚本 |
| `docs/` | 面向用户的工具链、算子、ABI 和平台说明 |

## 默认目标参数

- 器件：`xczu15eg-ffvb1156-2-i`
- Vivado：2021.1
- NPU 编译目标频率：200 MHz
- 示例模型中的 Dense Conv 默认并行度：`TN=4`、`TM=4`、`MAC_IMPL=3`
- 主量化数值配置：`affine-int8-v1`

编译目标频率和所选 FPGA 器件都不表示每个模型已通过 RTL 仿真、综合、时序或板级运行。使用生成物前，请先查看对应案例的状态，确认它是否具备所需的硬件验证结果。
