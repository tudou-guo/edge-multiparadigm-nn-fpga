# 上板说明 — SNN 平台（ZU15EG / RIGUKE）

硬件：r7-sumsfix 版本（Setup WNS +0.051 / Hold WHS +0.011，0 failing）
工程：`D:/spk_platform/project_1/project_1.xpr`，bit 生成于 2026-10-30 23:10。相对 r6 的唯一 RTL 改动：修复 AXI-Lite SUMS 读窗口地址解码（原来只映射 32 个字，128 个通道每 32 个回绕一次，导致 sums[32+i] == sums[i]）。

## 1. 文件夹内容

| 文件 | 说明 |
|---|---|
| `snn_platform.bit` | PL 配置码流（28.7 MB） |
| `snn_platform.hwh` | 硬件 handoff（bit 同名，PYNQ Overlay 必需） |
| `snn_platform_run.py` | PYNQ 运行脚本（MMIO 写刺激 → start → 读 sums → 比对 golden） |
| `vec/case_real/` | 真实 checkpoint 的 3 个样本：etab/ptab/toks/stream/exp_sums + golden.json |
| `MANIFEST.sha256` | 全部文件哈希 |

## 2. 板卡/环境要求

- ZU15EG（RIGUKE ZU15EG）板卡，PL DDR4 正常。
- PYNQ 镜像（含 `pynq` Python 包，能 `from pynq import Overlay, MMIO`）。
- 板端文件系统可写，串口/网络可访问 `/home/xilinx`。

## 3. 运行步骤

```bash
# 1) 把整个 board 文件夹拷到板子
scp -r board xilinx@<板卡IP>:/home/xilinx/snn_board

# 2) 板端执行（首次建议先跑 1 个样本）
cd /home/xilinx/snn_board
python3 snn_platform_run.py \
    --bit snn_platform.bit \
    --vecdir vec/case_real \
    --samples 1 \
    --json-out snn_board_result_s0.json

# 3) 三个样本全跑
python3 snn_platform_run.py \
    --bit snn_platform.bit \
    --vecdir vec/case_real \
    --samples 3 \
    --json-out snn_board_result.json
```

预期输出（关键行）：

```text
SNN id=0x534E4E31 version=0x00020000
sample 0: match=True cycles=7084772 board_ms=... model_ms=35.4
sample 1: match=True cycles=...
sample 2: match=True cycles=...
wrote snn_board_result.json
```

- `cycles` 在 200 MHz 域计数，真实样本约 **7.08M**（仿真/回放参考值 7,084,772 / 7,084,812）。
- `board_ms` 主要花在 MMIO 写 DDR 刺激（约 0.92 MB/样本），不是 PL 计算时间；`model_ms = cycles/200MHz ≈ 35.4 ms`。
- `all_match=true` 即通过。

## 4. 失败排查

| 现象 | 原因/处理 |
|---|---|
| `hardware handoff not found next to bitstream` | `.bit` 与 `.hwh` 必须同目录同 stem（snn_platform.bit + snn_platform.hwh） |
| ID 不是 0x534E4E31 / version 不是 0x00020000 | bit 没加载成功，或 AXI-Lite 地址窗口不对（应为 0x4_8000_0000） |
| `PL error: TIMEOUT (1)` | DMA 没返回数据：检查 PL DDR4 窗口/地址（0x4_0000_0000 起的 1 GiB）与 DDR4 校准完成 |
| `PL error: CONFIG (2)` | 长度寄存器被改坏（TAB/TOK/W bytes 必须 16 字节对齐且非 0） |
| `PL error: OUT_TIMEOUT (3)` | 写 DMA 的 B 响应没到（检查写通道/地址） |
| `PL error: code 5` | 结果帧不完整（写数据 beat 不足 32 或 last 位没置起） |
| 卡在 `busy` 不结束 | 用 `CTRL_SOFT_RESET` 复位后重跑；检查参考时钟和 MIG 校准 |
| MMIO 写很慢 | 正常：PYNQ MMIO 逐字写 DDR 刺激约几百 KB/s～MB/s；后续可换 PS-DDR + DMA 提速 |

寄存器/接口说明见 `docs/SNN_PLATFORM_BUILD_20261030.md`（CSR 偏移、DDR 布局、每样本流顺序）。

## 5. 版本与证据（r6）

- 时序：Setup WNS +0.051 / TNS 0 / 0 failing（334,330 endpoints）；Hold WHS +0.011 / THS 0 / 0 failing（333,740）；WPWS +0.039。
- 报告：`D:/spk_platform/reports/cdc_fix_r6_20261030_2000/`、`reports/r6_final_*.rpt`。
- 关键改动相对上一版：去掉了 OOC 的 8 级 LUT1 输入延迟链（平台专用适配层），残余 CDC 改 XPM 事件/包 FIFO。

## 6. 已知限制

1. 完整协议仿真矩阵未跑（只有 CSR/超时 smoke PASS）。
2. 软复位是 flush 两侧 FIFO，不是“排空在途 DMA 事务”的完整协议。
3. 写完成 B 数按常数 4（512 B 输出）统计，其它长度/4KB 边界未验证。
4. 顶层 `sys_rst` 没有 input delay / false path（check_timing HIGH），DDR4 `reset_n` 没有 output delay；对本次功能无影响，下一轮补约束。
