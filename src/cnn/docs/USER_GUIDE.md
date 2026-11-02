# 用户指南

## 编译器会生成什么

主入口：

```powershell
python -B -m tools.onnx_gen.onnx_gen compile --help
```

对于受支持的静态 INT8 QDQ ONNX 图，编译器会生成图规划、降低后的图、指令流、打包参数、输入和输出元数据、DDR 初始化镜像、RTL 顶层以及源码身份清单。常规编译流程也可以调用 Vivado，生成 IP Catalog 源码包。

IP 包由 `component.xml` 描述，是可复用的 RTL 组件；它不包含针对具体板卡的 PS/PL Block Design、综合网表、布线结果、bitstream 或板测结果。

## 输入

- 静态 ONNX 模型；其量化形式必须符合所选数值配置。
- 输入清单；需要具体输入数据的编译流程还要提供匹配的 `uint8` 样本数组。
- 匹配的数值配置；随附案例通常使用 `affine-int8-v1`。
- 算子源码根目录。省略 `--source-root` 时使用本发布目录；只有案例明确要求时才指定候选根目录。

每次运行都使用新的输出目录。不要复用旧编译目录，因为清单会绑定本次模型和 RTL 源码身份。

## 源码身份

每次编译都会在 `generated/rtl_manifest.json` 等清单中记录源码根目录、源文件及其哈希、模板、厂商 IP、所选引擎和生成文件。编译、仿真与 IP 包必须使用相同的源码身份。位于 `examples/` 的候选实现不会自动成为主线算子库。

## Vivado 打包

Vivado 可用时，默认编译流程会在编译输出目录中生成 IP 包。添加 `--no-pack-ip` 可在编译器生成阶段结束，不调用 Vivado。需要使用其他 Vivado 安装时，可通过 CLI 的 `--vivado` 选项指定路径。

打包不会执行 OOC 综合、整机综合、布局布线、时序资格验证、bitstream 生成或板端运行。这些是独立的平台步骤，应分别记录状态。

## 训练案例

`examples/tinycnn28_mnist` 和 `examples/mobiledw32_cifar10` 包含模型源码、权重、FP32 ONNX、QDQ ONNX、样本输入和已有编译产物。训练或重新量化需要 PyTorch 和 torchvision。仅运行编译器时不需要训练环境。

## 旧版量化工具

`tools/quant/export_quant_params.py` 为冻结 ABI 1.0 的 Conv/Pool 案例保留。它与当前 ONNX QDQ 流程分开；使用时请参考[工具入口](../tools/README.md)，不要将其 ABI 1.0 记录交给其他 ABI 的运行端。
