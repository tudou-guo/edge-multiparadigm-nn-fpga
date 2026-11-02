# M4 A-profile 板卡包（timing-fix r4）

本目录包含 2026-10-29 固定样本板测所用的 bit/HWH、运行时数据和 PYNQ runner。bit 与 HWH 来自同一平台实现；signed-INT8 输入镜像、golden、预处理/后处理和 runner 沿用已验证版本。包内 MANIFEST.json 记录产物身份与测试数据。

## 平台实现

- 器件：xczu15eg-ffvb1156-2-i；Vivado 2021.1；NPU 200 MHz；5 ns 时序约束。
- 实现版本：impl_rtlfix_1029_r4。
- 建立时序：WNS +0.075 ns，TNS 0，0 条违例端点。
- 保持时序：WHS +0.010 ns，THS 0，0 条违例端点；脉宽 WNS +0.039 ns，0 条违例端点。
- Route errors 0；DRC errors 0，warnings 1385。
- NPU 资源：468 DSP48E2、72 URAM288。

## 关键产物

| 文件 | 身份信息 |
|---|---|
| fpga/platform_top.bit | 28,700,862 B；SHA-256 baeb25fc59d7d99fe4a29e169f65367f461d4c7e9787518a2f4acb939ecca7dd |
| fpga/platform_top.hwh | 373,118 B；SHA-256 63a11375621c316ee80822fce3e2cd1a07b5b0982b8614b40856f2d681e04908 |
| runtime/ddr_init.bin | SHA-256 eac1c430b9c425513ee906e0607fead94e16b3d076eb7f1a1fce345a7facb1be |

## 上板命令

在本目录执行：

    sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh --ctrl-base 0x480000000 --window-base 0x400000000 --out-dir run_out_heads
    sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh --ctrl-base 0x480000000 --window-base 0x400000000 --image runtime/test_image.jpg --out-dir run_out_image

## 板测结果（2026-10-29，用户提供的 Jupyter 输出）

| 模式 | 结果 |
|---|---|
| heads | PASS；conv48/51/54/57/60/63 全部匹配，first_mismatch=-1；PL start-to-DONE 0.239510 s；error_code=0 |
| image | PASS；六个 head 匹配；检测到 1 个 train，score 0.636，box=(471.2,180.3,641.1,450.7)，与 golden 一致；PL start-to-DONE 0.239369 s；error_code=0 |

image 模式的软件前处理约 1.7317 s；PL 推理计时不含该前处理。decode、draw 和 DDR 写入/读回也分别计时，不应把 0.239 s 写成完整图像应用的端到端时间。

## 资格边界

- 板测覆盖固定 heads 数据与一张固定测试图；标记为用户提供结果。它证明该包在这两个固定输入上的功能闭环，不代表 VOC/mAP 或数据集精度。
- ModelSim 示例只跑到 conv1 record0；没有完整 640 图的 RTL 仿真回放。
- 已有单样本 PL 延迟记录；未做持续吞吐/带宽基准、多次统计分布或长时间稳定性测试。
- Vivado BD 工程、XSA 和完整 implementation 报告未随板卡包发布；板上运行只需本目录的 bit、HWH、runtime 和脚本。
