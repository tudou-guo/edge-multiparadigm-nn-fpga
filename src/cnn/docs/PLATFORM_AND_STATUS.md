# 平台产物与验证状态

## IP 包与板级平台

工具链的 Vivado 打包器会生成 Vivado IP Catalog 源码包。用户还需要将 IP 接入适配具体板卡的 Block Design，选择存储器路径和时钟，并执行平台构建。单独的 IP 包不是完整板级平台，也不是 bitstream。

## 随附的 MobileDW 平台案例

`examples/mobiledw32_cifar10/vivado_platform_psddr/project_1/project_1.xpr` 是使用 MobileDW-32 NPU IP 的 PS-DDR Block Design：

- PS `M_AXI_HPM0_LPD` → AXI-Lite SmartConnect → NPU `s_axil`，地址映射为 `0x80000000`。
- NPU `m_axi` → AXI SmartConnect → PS `S_AXI_HP0_FPD` → PS DDR；映射到 NPU 地址空间的低 2 GiB。
- PL0 目标频率约为 200 MHz；原工程记录值约为 199.998 MHz。

随附了复制后的 Vivado 工程和生成的 BD handoff。整理前的源工程有 BD 验证通过记录；复制后的工程尚未在 Vivado 中重新打开。虽然 XPR 已指向本地 IP 包目录，但 IP-XACT 元数据仍记录着指向原 `D:\dl_v1` 候选目录的绝对 DSP 依赖路径，因此迁移和独立复用状态为 `not_run`。综合、实现、资源报告、时序、bitstream、XSA 导出和板端运行也均为 `not_run`。生成的 HWH 是 BD 阶段产物，不是综合完成或通过板级资格验证的结果。

该 MobileDW 案例通过 HP0 使用 PS DDR。历史 YOLO 板卡包则通过 PL 侧 DDR4 MIG 为 NPU 提供数据存储器（包内 HWH 含 `ddr4_0` 控制器和 AXI 存储器范围）。它的 CPU 可见 DDR 窗口和板端运行时均针对该平台。不要在不同模型间替换 bit/HWH 或地址映射。

## 各案例状态

详细列表见[案例索引](../examples/README.md)。ModelSim 行为仿真、RTL 编译、IP 源码打包、综合、时序和板端运行是不同的验证关卡。较早关卡通过不代表后续关卡也通过。
