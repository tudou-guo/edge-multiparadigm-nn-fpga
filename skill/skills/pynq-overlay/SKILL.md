---
name: pynq-overlay
version: 1.0.0
description: 在任意 PYNQ 板卡（Zynq-7000 / Zynq UltraScale+，PYNQ 2.7+）上通用地加载并运行 PL 加速器 overlay：三件套配对 → overlay/元数据校验 → 寄存器映射 → 内存与 DMA/缓存一致性 → 黄金比对 → 性能采集 → 证据归档。
applies_to: [pynq, board]
package_paths: [board/cnn, board/lnn, board/snn]
requires: [PYNQ 2.7+（板端 Python 3.8/3.10）, 同 stem 的 .bit + .hwh（可选 .xsa）, 可选 Vivado 2021.1（仅重生成件）]
last_verified: 2026-11-02
---

# SKILL: pynq-overlay

## 0 一句话定义

把"一个已综合好的 PL 加速器（overlay）"在 PYNQ 板卡上跑起来并证明结果可信：
**同源三件套校验 → 加载与元数据核对 → 寄存器/DMA 交互 → 黄金比对 → 性能与证据归档**。
本技能只依赖 PYNQ 通用 API（`pynq.Overlay`、`pynq.allocate`、MMIO/RegisterMap）与 hwh 元数据，
**不绑定具体题目、模型或板卡**：换一块板、换一个 IP，只需替换 bit/hwh 与地址/寄存器常量。

## 1 何时使用 / 何时不要使用

**使用**（典型场景，均与具体题目无关）：

- 新拿到一个 overlay（.bit + .hwh），要在板端先冒烟、再跑业务数据；
- 用 `pynq.allocate` 分配缓冲区、通过 DMA 与 PL 交换数据，需要确认缓存一致性/可见性；
- 通过 `overlay.ip_dict` / `register_map` 校验地址映射和寄存器位域；
- 板端结果与黄金参考不一致，要定位是"数据没进 PL""PL 算错""读回的是旧缓存"还是"驱动时序问题"；
- 要给一次上板运行留可复核证据（哈希、ID/VERSION、比对结果、耗时口径）。

**不要使用**：

- 不做 RTL/综合/时序判断：overlay 只是位流，时序结论来自 Vivado 报告；
- 不用本技能替代 `evidence-gate`：门级与标签由 `../evidence-gate/SKILL.md` 定义；
- 不把离线寄存器模型（如 `--backend simulation-model`）的结果当硬件实测；
- 不假设所有板卡地址相同：地址窗、DDR 基址、时钟频率必须从 hwh/BD 读取或由构建记录给出。

## 2 前置条件与环境自检

| 项 | 要求 | 板端自检 |
|---|---|---|
| 板卡可达 | 能 ssh 或通过 Jupyter 终端执行 | `ssh xilinx@<board_ip>` |
| PYNQ 版本 | ≥ 2.7（`pynq.Overlay` 标准 API） | `python3 -c "import pynq; print(pynq.__version__)"` |
| 平台信息 | 与 bit 的器件一致 | `python3 -c "from pynq import PL; print(PL.device_name)"` |
| 三件套 | `.bit` 与 `.hwh` 同目录同 stem（XSA 可选） | `ls -l *.bit *.hwh` |
| 驱动与期望值 | 驱动脚本 + 黄金向量（或生成规则）同目录 | `ls driver.py golden/` |
| 板端路径 | 所有输入在板端可读（相对路径即可） | 见 `reference/commands.md` C1 |

本机（PC 侧）无板卡时，只能做三件套哈希配对与 hwh 元数据静态核对，不能宣称 G6 通过。

## 3 输入合同

| 输入 | 格式 | 硬约束 | 校验命令 | 不满足时 |
|---|---|---|---|---|
| bitstream | `.bit` | 与 hwh 同 stem；同一次实现导出的位流 | `sha256sum` | 哈希/生成时间不配对 → 不同源，停止 |
| hardware handoff | `.hwh` | 含 `MEMRANGE`（地址/范围）、`CLKFREQUENCY`、`DEVICE/PACKAGE/SPEEDGRADE` | `grep -o 'MEMRANGE[^>]*'` | 缺地址/时钟 → 无法核对映射，退回构建方 |
| 板级导出（可选） | `.xsa` | 内嵌 bitstream 与 `.bit` 逐字节一致、内嵌 hwh 与 `.hwh` 一致 | `zipfile` 比对（C2） | 不配对 → 部署包不可信 |
| 寄存器契约 | 驱动内 CSR 常量表或 `register_map` | 偏移、位宽、读写属性、reset 值、ID/VERSION 期望值 | 读 `ID/VERSION` 比对 | 读回不符 → 载入的不是同一 overlay |
| 内存/DMA 契约 | 分配大小、对齐、PL 侧地址窗 | 缓冲区地址对齐（DMA 建议 4 KiB/64 B 对齐）；PL 可见窗口与 PS 物理地址一致 | `buffer.physical_address` 与 hwh 地址窗比对 | 映射错 → DMA 超时或数据错位 |
| 期望值 | 十六进制/二进制/JSON | 与驱动读取的数据布局（字节序、位宽、符号）一致 | 驱动内置 `--expect-dir` 校验 | 布局不符 → 比对必失败，先对齐布局 |

## 4 执行步骤

### 步骤 1：同源三件套配对（PC 侧，先做）

```bash
sha256sum overlay.bit overlay.hwh
python -c "import hashlib,zipfile;z=zipfile.ZipFile('overlay.xsa');print(hashlib.sha256(z.read('overlay.bit')).hexdigest());print(hashlib.sha256(z.read('design_1.hwh')).hexdigest())"
```

判据：XSA 内嵌 bit/hwh 与外部文件逐字节一致；bit/hwh 的 mtime 与构建记录同批。不配套的 bit/hwh 直接判 `fail`（详见 `reference/troubleshooting.md` T1）。

### 步骤 2：加载 overlay 并核对元数据

```python
from pynq import Overlay
ol = Overlay("/home/xilinx/overlay.bit")        # 同目录同 stem 的 .hwh 会被自动找到
print(ol.ip_dict.keys())                         # 期望包含 <ip_name>
print(ol.ip_dict["<ip_name>"]["phys_addr"], ol.ip_dict["<ip_name>"]["addr_range"])
```

判据：`ip_dict` 含目标 IP；地址/范围与 hwh `MEMRANGE` 及驱动常量三处一致。
失败分支：`Overlay` 找不到 hwh → 同 stem 缺失；`ip_dict` 无 IP → bit/hwh 不是同一次导出或 IP 被优化掉。

### 步骤 3：寄存器映射与 ID 自检

```python
ol = Overlay("/home/xilinx/overlay.bit")
csr = ol.<ip_name>                                # DefaultIP → MMIO
csr.write(0x008, 0x1)                             # CONTROL
print(hex(csr.read(0x000)))                       # ID
```

范式要点：

- 地址一律相对 IP 基址写偏移；**不要**把 PS 物理地址直接加进 MMIO；
- 先读 `ID`/`VERSION` 与驱动常量比对，再写控制寄存器；
- 位域读写用「读-改-写」或 `register_map` 字段，避免覆盖其它位；
- reset 值、只读位、写 1 清零位必须按契约处理（详见 `reference/artifacts.md` 的寄存器表模板）。

### 步骤 4：内存分配与 DMA / 缓存一致性

```python
import numpy as np
from pynq import allocate
buf = allocate(shape=(n,), dtype=np.uint32)       # 或 pynq.allocate(shape=..., dtype=...)
buf[:] = data; buf.flush()                        # 非相干口：写后 flush
...                                               # 触发 DMA / START
buf.invalidate()                                  # 读前 invalidate（非相干口）
res = np.array(buf)
```

判据与要点：

- **相干口（HP/ACP 由 BD 决定）与非相干口的差别决定是否需要 `flush/invalidate`**；不确定时按"非相干"处理（写后 flush、读前 invalidate），并记录到证据里；
- DMA 完成判据必须明确：完成位轮询 / 中断 / 超时三者之一，且超时值 > 最坏用例（如 `timeout_cycles`）；
- 缓冲区地址对齐、长度、跨 4 KiB 边界要按 IP 的 AXI burst 规则处理；
- 不要用 PS 物理地址直接构造 numpy 视图（PYNQ 不保证映射）。

### 步骤 5：黄金比对（先小样例，再全量）

- 先用 1 个最小样例打通"加载 → ID 校验 → 单次计算 → 读回 → 比对"，再跑全量；
- 比对必须给出 `first_mismatch` 与计数（不是只给 bool）；
- 字节序/符号扩展是最高频错因：读 int16 要按 `(v | 0xFFFF0000) if v & 0x8000` 还原，写同理；
- 结构断言（done 位、错误码、通道数）与数值比对分开记录。

### 步骤 6：性能采集（口径必须写清）

```python
import time
t0 = time.perf_counter(); start(); wait_done(); t1 = time.perf_counter()
print("wall_s=%.6f" % (t1 - t0))                  # 端到端（含 PS 提交/轮询）
print("hw_cycles=%d hw_s=%.6f" % (cycles, cycles / clock_hz))   # 硬件计数口径
```

判据：报告里每个耗时数字都要带口径（端到端 / 提交到完成 / 核内 cycles）与采样次数（建议 ≥3 次取中位数）；
不同口径的数字不得相加或混写。

### 步骤 7：证据归档

写 `summary.json`（schema 见 `reference/artifacts.md`）：来源、PYNQ/平台版本、bit/hwh 哈希、
ID/VERSION/CLOCK 校验结果、样本数、逐位/容差比对结果、错误码、耗时口径、日志路径。
外部数据**必须**标 `source=hardware` 才能进 G6；离线模型只能标 `partial`。

## 5 产物与清单

| 产物 | 位置 | 关键字段 |
|---|---|---|
| 运行摘要 | `results/<run>/summary.json` | `source`、`pynq_version`、`platform`、`bit_sha256`/`hwh_sha256`、`id_check`/`version_check`/`clock_hz_check`、`samples`/`matched`/`first_mismatch`、`error_code`、`wall_s`、`hw_cycles`、`hw_time_s` |
| 输出数据 | `results/<run>/outputs.npy`（或 .hex） | 与期望值同布局 |
| 比对明细 | `results/<run>/compare.json` | 逐样本结果、首个不一致索引、期望/实际片段 |
| 日志 | `results/<run>/*.log` | 环境打印、寄存器读写、DMA 起止 |
| 哈希清单 | `results/<run>/SHA256SUMS` | 上述文件 + bit/hwh |

## 6 验收（G6 板端）

| 判据 | 阈值 | 证据 |
|---|---|---|
| 来源 | `source=hardware` 且 `is_hardware_measurement=true` | summary.json |
| 身份 | bit/hwh(/xsa) 与部署包哈希一致；XSA 内嵌比对通过 | C2 输出 + summary 哈希 |
| 硬件自检 | `ID`/`VERSION`/`CLOCK_HZ` 与期望一致 | summary 的 check 字段 |
| 功能 | 固定样本 N/N 逐位一致（或按契约容差），`error_code=0` | compare.json |
| 性能 | 至少一项口径可复算（cycles 或 wall_s），标注采样次数 | summary 的 timing 段 |

"换板卡/换 IP 仍可用"是通用性验收：替换 bit/hwh 与地址常量后，上述 5 条应能原样复用（见 `reference/acceptance.md` §3）。

## 7 故障与修复（高频摘要，全文见 `reference/troubleshooting.md`）

| 现象 | 根因 | 处置 |
|---|---|---|
| `Overlay` 报找不到 hwh / same stem | bit 与 hwh 不同目录或不同名 | 同目录同 stem 放好 |
| `ip_dict` 里没有目标 IP | bit/hwh 非同一构建，或 IP 被优化/裁掉 | 重新导出配对文件；核对 `synthesized` 列表 |
| 读回 ID/版本不符 | 部署的不是同一 overlay | 重新配对；检查板上已有旧 bit 加载 |
| DMA 超时 | 地址窗/对齐/长度/使能配置错，或未 flush | 查 `physical_address` 与 hwh 地址窗；补 flush；加长超时定位 |
| 数据"写进去了但读回旧值" | 非相干口未 invalidate；numpy 视图与 buffer 不同步 | 读前 `invalidate()`；用 `np.array(buf)` 复制 |
| 读回值符号错/高位脏 | 16/32 位符号扩展或位域拼接错 | 按契约做符号还原；打印原始字 |
| BUSY 位观察不到 | 采样窗口错位（轮询间隔 > 忙窗口） | 读状态与写 START 在同一线程紧邻执行，记录时间戳 |
| 行为仿真过、板端错误码 | 综合优化改变网表语义（如写使能接常量） | 综合后网表仿真/端口断言，不能只信行为仿真 |

## 8 边界与不适用

- 不生成/不修改 bitstream：overlay 由 Vivado 构建，本技能只消费；
- 不给时序/资源结论：WNS/WHS、LUT/FF 来自实现报告，不是板端；
- 不假设时钟频率：`CLOCK_HZ` 可能来自 PL 时钟而非 PS，必须读寄存器或 hwh `CLKFREQUENCY`；
- 不跨板卡复用地址常量：`0x480000000` 这类 64 位 PL 地址窗因 BD 而异；
- 板卡离线时只能做静态核对（哈希/元数据），不能写"板端通过"。

## 9 版本与身份绑定

| 身份项 | 来源 | 校验 |
|---|---|---|
| bit/hwh 哈希 | 文件 sha256 | 与部署记录、XSA 内嵌比对 |
| hwh 元数据 | `DEVICE/PACKAGE/SPEEDGRADE`、`CLKFREQUENCY`、`MEMRANGE` | 与板卡平台、驱动常量比对 |
| 寄存器身份 | IP 内 `ID`/`VERSION` 只读寄存器 | 每次运行读一次并写入 summary |
| 软件身份 | PYNQ 版本、驱动脚本 sha256 | summary 记录 |

任一身份变化 → 旧运行摘要作废，重新跑并重新归档。

## 10 证据归档

```
results/<run>/
├── summary.json       # 本技能第 5 节字段
├── outputs.npy        # 板端读回数据
├── compare.json       # 与期望值的逐项比对
├── board.log          # 环境/寄存器/DMA 日志
└── SHA256SUMS         # 上述 + bit/hwh
```

归档规则沿用 `../evidence-gate/SKILL.md` §10：只增不改、带 run-id、外部数据标注来源。

## 11 参考文件索引

- `reference/commands.md`：可复制命令矩阵（环境自检 / 三件套配对 / 加载 / 寄存器 / DMA / 比对 / 性能 / 归档）
- `reference/artifacts.md`：summary.json 与 hwh 字段表、寄存器表模板、目录约定
- `reference/acceptance.md`：G6 判据 + 通用性验收（换板卡/换 IP 复用清单）
- `reference/troubleshooting.md`：板端故障四元组（现象/根因/处置/证据）
- `reference/scaffold.md`：接口契约（registers.json）→ 驱动/测试骨架生成 + golden/性能/资源报告自动汇总
- `reference/sources.md`：本技能事实来源
