# 主线算子 RTL

本目录保存主线编译器会选用的算子 RTL。源码集合和引擎登记见 [operator_library.json](../operator_library.json)；ONNX 属性和降低限制概述见[支持的算子](../../docs/OPERATORS.md)。

| 算子类别 | 主线目录 | 操作码 |
|---|---|---:|
| Dense Conv 3×3 及受限的 1×1 中心 tap 路径 | `conv/` | `0x0001` |
| MaxPool 2×2 | `pool2x2/` | `0x0002` |
| ReLU、Add、Clip、同形状 INT8 Mul | `eltwise/` | `0x0010`、`0x0011`、`0x0015`、`0x0018` |
| Pad、Slice、Concat | `copy/` | `0x0012`–`0x0014` |
| GlobalAveragePool | `reduce/` | `0x0016` |
| AveragePool 2×2 | `pool2x2/` | `0x0017` |
| Resize nearest ×2 | `resize/` | `0x0019` |
| MaxPool 5×5 | `pool5/` | `0x001A` |
| 支持的激活 LUT | `activation/` | `0x001B` |
| Depthwise Conv V1 | `depthwise/` | `0x001C` |
| 通用 MaxPool2D | `maxpool2d/` | `0x001D` |

Depthwise Conv V2（`0x001E`）是隔离候选，位于 [examples/depthwise_conv_v2](../../examples/depthwise_conv_v2/README.md)，没有登记到主线算子库。主线与候选路径的区别见 [ABI 与数值配置](../../docs/ABI.md)。

算子名称不代表所有 ONNX 形状和属性组合都受支持。请以编译器合同和[算子支持表](../../docs/OPERATORS.md)中的限制为准。
