# 工具入口

## 当前 ONNX→RTL 流程

- `python -B -m tools.onnx_gen.onnx_gen compile ...`：将静态 INT8 QDQ ONNX 编译为图规划、指令、内存镜像和模型专用 RTL。
- Vivado 可用时，编译流程会打包 Vivado IP Catalog 源码包。只生成编译结果时可使用 `--no-pack-ip`。
- `tools/npu_gen/pack_npu_ip.tcl`：当前编译器调用的底层 IP 打包入口。

编译前请先查看[快速开始](../QUICKSTART.md)、[用户指南](../docs/USER_GUIDE.md)和[ABI 说明](../docs/ABI.md)。

## 旧案例使用的 ABI 1.0 量化工具

`quant/export_quant_params.py` 从 `dl_v0.1` 保留，用于旧版 Conv/Pool 案例。它会将各层的 scale 和 zero-point 转换为冻结的 ABI 1.0 二进制记录。它不是当前 ONNX 编译器使用的 QDQ 前端，输出只能交给 ABI 1.0 的运行端处理。

```powershell
python -B tools/quant/export_quant_params.py --help
```

旧版 `tools/restart` 项目 launcher 和阶段构建脚本依赖原目录中的 `baselines/`、`releases/` 及内部 run 布局，因此不作为当前通用工具发布。原板卡包包含的运行时源码随对应案例保存在 `examples/legacy_board_packages/` 中，具体以各包自带说明为准。
