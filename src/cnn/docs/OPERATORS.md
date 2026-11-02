# 支持的算子类别

主线编译器通过 `rtl/operator_library.json` 登记 RTL 源码，并由 `tools/onnx_gen/op_registry.py` 与前端合同限定可接受的 ONNX 属性和量化形式。下表按算子类别概述，不代表任意 ONNX 变体或模型形状都受支持。

| 类别 | 主线实现 | 指令 / 降低方式 |
|---|---|---|
| Dense Conv 3×3 和受限的 1×1 中心 tap 路径 | `rtl/ops/conv/` | `0x0001` |
| MaxPool 2×2 | `rtl/ops/pool2x2/` | `0x0002` |
| ReLU、Add、Clip、同形状 INT8 Mul | `rtl/ops/eltwise/` | `0x0010`、`0x0011`、`0x0015`、`0x0018` |
| Pad、Slice、Concat；静态 Split 降低 | `rtl/ops/copy/` | `0x0012`–`0x0014` |
| GlobalAveragePool | `rtl/ops/reduce/` | `0x0016` |
| AveragePool 2×2 | `rtl/ops/pool2x2/` | `0x0017` |
| Resize nearest ×2 | `rtl/ops/resize/` | `0x0019` |
| MaxPool 5×5 / SPPF 构件 | `rtl/ops/pool5/` | `0x001A` |
| Sigmoid、Tanh、受限 SiLU、LeakyReLU LUT 路径 | `rtl/ops/activation/` | `0x001B` |
| Depthwise 3×3，`group=C`、`multiplier=1` | `rtl/ops/depthwise/` | `0x001C`（V1） |
| 通用 MaxPool2D | `rtl/ops/maxpool2d/` | `0x001D` |

## 隔离候选

Depthwise Conv V2 位于 `examples/depthwise_conv_v2/candidate`，包含独立的 QDQ→RTL 编译路径，操作码为 `0x001E`。它仍与主线算子库隔离，尚未并入 V1 ABI。案例说明列出了通过与未通过的兼容性证据。

具体支持的形状、padding、dilation、scale 和 zero-point 规则，请查看案例说明和编译器合同。不能只凭算子名称推断任意 ONNX 图都受支持。
