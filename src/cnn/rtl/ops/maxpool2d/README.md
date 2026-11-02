# 通用 MaxPool2D V1

`maxpool2d_core` 对有符号 INT8 输入选取最大值，不执行 ReLU 或重新量化。核心复用 8 路比较器，包含一行输入 RAM、水平结果行环形缓存和一行输出 RAM。三类存储器均为 128-bit 宽、同步读延迟 1 周期，RAM 内容不复位。每个真实输入行最多读取一次；每个输出行写入一次，且只有最后一次写响应成功后才拉高 `core_done`。

支持范围：`KH/KW=1–7`、`SH/SW=1–8`、`DH/DW=1–4`、各方向 `PT/PL/PB/PR=0–24`、`N=1`、`C=1–512`，输入和输出的 `H/W=1–640`。对齐后的单行最大 65,520 字节。输入和输出必须使用相同的正 scale 和有符号 zero-point；编译器会拒绝不同量化域。ONNX `ceil_mode` 与 `storage_order` 必须为 0。支持 `auto_pad=NOTSET` 和 `VALID`，不支持输出索引。ONNX 未提供 stride 或 dilation 时默认 `[1,1]`，未提供 padding 时默认 0。

逻辑张量为 NCHW，物理存储为 NHWC8_A16。位置 `(y,x,c)` 的字节偏移为 `y * row_stride + x * align8(C) + c`。128-bit 数据拍中，最低地址字节位于 `[7:0]`。空间 padding 的值为 -128；未使用的通道 lane 和行尾对齐字节写为 0。

除非需要回放显式保留的旧指令记录，或 `auto` 可安全选择旧 5×5/s1/p2 核，否则编译器选择 `0x001D`。自动选择 Pool5 还要求 `H>=3`、`W>=4`、有符号 zero-point 为 0，且八通道组数为 1 或偶数；其他有效几何形状使用通用核。旧 2×2/s2 核在当前生成构建中带有 clamp，因此普通 MaxPool 不会自动选择它。`--maxpool-backend generic` 强制使用新核；默认值 `--maxpool-backend auto`。旧 `0x0002` 和 `0x001A` 指令负载格式保持不变。

## 配置与生命周期

空闲时，通过 `pool_cfg_addr[3:0]` 和 `pool_cfg_we` 将 16 个配置字各写入一次，再脉冲触发 `core_start`。核心会锁存完整配置，并清除影子寄存器的写入有效位；下一次启动前必须重新写入全部 16 个配置字。计算期间以及已接受服务请求尚未结束时，`core_busy` 保持为高。`core_done` 和 `core_error` 保留到下一次启动或 `core_soft_reset`。忙碌时写配置或再次启动属于协议错误；abort 或错误发生后，核心会先排空已发出或已接受的事务，再退出 busy。执行 `core_soft_reset` 前，外围服务必须已复位或处于空闲状态。配置检查失败时不会发出 DDR 命令。

| 字序号 | 内容 |
|---:|---|
| 0 | 版本标志，必须为 `1` |
| 1–2 | 输入和输出字节地址 |
| 3–5 | `Hin/Win`、`C/C_alloc`、`Hout/Wout`，各字段占低/高 16 位 |
| 6–8 | `KH/KW`、`SH/SW`、`DH/DW`，各字段占低/高 16 位 |
| 9–10 | `PT/PL`、`PB/PR`，各字段占低/高 16 位 |
| 11–12 | 输入和输出行字节数 |
| 13–14 | 输入和输出 span 字节数 |
| 15 | 保留，必须为 0 |

Sequencer 头为 `0x001D0010`。启动前会检查长度、版本、保留位和地址重定位。核心会校验尺寸、形状、stride、span、地址对齐、溢出、地址重叠和综合时确定的容量。服务端口沿用统一的 128-bit 读写协议：读请求 tag 为 1，写请求 tag 为 2。遇到反压时，命令和写数据保持不变；读数据拍仅在 `valid && ready` 时被接收。

错误码：`0001` 配置错误，`0002` 格式不支持，`0003` 未对齐，`0004` 地址范围或重叠，`0005` 行 stride 或容量错误，`0009` 服务协议错误，`000A` 内部错误或 abort。若服务端提供非零错误码，核心会保留该错误码。

## 容量与验证

`MAX_IN_ROW_BYTES`、`MAX_OUT_ROW_BYTES`、`MAX_EH` 和 `CACHE_ROWS` 描述的是一个实例，而非单独某层。导出器会根据所选 `0x001D` 记录推导这些值。`CACHE_ROWS` 是覆盖最大有效高度 `DH*(KH-1)+1` 的最小 2 的幂；RAM 地址间距使用 `MAX_OUT_ROW_BYTES`。清单记录 8 lane 能力、实际容量、操作码、逻辑存储字节数、实例掩码和未运行的物理资格项。逻辑字节数不能直接推算综合后 BRAM 块数。

如需复现 ModelSim 检查，在 `D:\DL_FINAL\DL` 根目录执行：

```powershell
python verification\operators\maxpool2d\run_modelsim.py
```

该流程会生成合成张量，检查几何参数和编译器，再以固定 seed 运行独立核心、sequencer/router 及生成顶层的图级回归。日志和向量分别保存在 `sim_out/maxpool2d_modelsim` 与 `sim_out/maxpool2d_compiled_graphs`。独立测试日志会记录总周期、读、水平、垂直、写、响应等待、验证/调度周期及字节数。完整图回放使用行为级 MAC 实现；因未使用的引擎可能已被裁剪，不执行深层 Conv 性能探测。

这些是行为仿真结果，不是时序或资源证据。本次发布整理没有重新运行 Vivado OOC 综合、IP 打包、布局/时序或板端运行；这些项目的状态为 `not_run`。
