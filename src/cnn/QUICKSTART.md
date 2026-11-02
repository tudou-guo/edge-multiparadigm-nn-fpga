# 快速开始

## 1. 检查编译环境

在发布目录根路径执行：

```powershell
Set-Location D:\DL_FINAL\DL
python scripts\env_check.py
```

编译依赖列在 `requirements.txt` 中。模型训练和 QDQ 导出需要 PyTorch；随附训练案例使用了带 GPU 的 `pytorch_clean` 环境。只有运行对应流程时才需要 ModelSim 或 Vivado。

## 2. 转译随附的 TinyCNN QDQ 模型

每次编译都要使用一个新的空输出目录。下面的命令会在 `build` 下建立独立结果：

```powershell
python -B -m tools.onnx_gen.onnx_gen compile `
  --model examples\tinycnn28_mnist\artifacts\quantized\tinycnn28_mnist_qdq.onnx `
  --numeric-profile affine-int8-v1 `
  --input-u8 examples\tinycnn28_mnist\artifacts\mnist_sample_u8.npy `
  --source-root D:\DL_FINAL\DL `
  --mac-impl 3 --tn 4 --tm 4 `
  --name tinycnn28_mnist --run-id tinycnn28_release_demo `
  --out-dir build\examples\tinycnn28_release_demo
```

如果 Vivado 已加入 `PATH`，或通过 `--vivado` 指定了 Vivado 路径，编译器默认还会在输出目录下生成 IP Catalog 源码包。若只需要编译结果而不启动 Vivado，可添加 `--no-pack-ip`。

## 3. 查看编译结果

编译目录包含图与内存规划、打包参数、`ddr_init.bin`、生成的 RTL、指令文件、`sources.f` 和源码身份清单。IP Catalog 包含 `component.xml` 与 RTL 源码。要生成 bitstream，仍需在 Vivado 中建立适配具体板卡的 Block Design 并完成相应硬件验证。

## 4. 试用 Depthwise Conv V2 候选

MobileDW-32 使用 `examples\depthwise_conv_v2\candidate` 下隔离的编译器和 RTL 候选。命令及兼容范围见该案例说明。V2 操作码 `0x001E` 与主线 Depthwise V1 操作码 `0x001C` 保持分离。

在把已有编译、仿真、打包或板测结果用于其他模型或源码版本前，请先查看[案例状态](examples/README.md)。
