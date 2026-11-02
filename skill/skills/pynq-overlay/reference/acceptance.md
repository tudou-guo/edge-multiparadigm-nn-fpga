# 板端验收（G6）与通用性验收（PYNQ overlay）

> 门级定义见 `../evidence-gate/reference/gates.md`。本文件只列板端（G6）判据与"换板卡仍可用"的复用验收。

## 1 板端运行判据（G6）

| # | 判据 | 通过标准 | 证据 |
|---|---|---|---|
| 1 | 来源 | `source=hardware`、`is_hardware_measurement=true` | `results/<run>/summary.json` |
| 2 | 身份配对 | bit/hwh 哈希与部署包一致；XSA 内嵌比对通过 | summary 哈希 + C2 输出 |
| 3 | 平台一致 | `PL.device_name` 与 bit 器件一致 | summary `platform` |
| 4 | 硬件自检 | ID/VERSION/CLOCK_HZ 三项 check 全 true | summary 三个 check 字段 |
| 5 | 功能 | 固定样本 N/N 逐位一致（或按契约容差），`error_code=0` | `compare.json` |
| 6 | 性能口径 | 至少一项可复算（`hw_cycles` 或 `wall_s`），标注采样次数 | summary `timing`/`notes` |
| 7 | 原始日志 | 环境打印、寄存器读写、DMA 起止都在 `board.log` | 日志文件 |

**一票否决**（出现即不能判 G6 pass）：

1. `source=simulation-model` 或离线模型结果写成 hardware；
2. bit 与 hwh 不同次构建却当同源；
3. 只有"板端跑过"的口头记录，没有 summary/日志/输出文件；
4. 用其它板卡/其它 bit 的运行结果充当本包结果；
5. 报告里的耗时口径混用（端到端 ↔ 核内 cycles）。

## 2 冒烟顺序（10 分钟内）

1. 加载 overlay → 打印 `ip_dict`（确认 IP 在）；
2. 读 ID/VERSION/CLOCK_HZ → 与期望常量比对；
3. 写 CONTROL 触发一次最小任务 → 轮询 DONE；
4. 读回最小输出 → 与 1 个 golden 样例比对；
5. 写 summary.json（此时只标 `smoke`，不标功能 pass）。

全部通过后再跑全量样本；冒烟失败不要继续做全量。

## 3 通用性验收（换板卡 / 换 IP 复用清单）

| 项 | 换板卡时 | 换 IP 时 |
|---|---|---|
| bit/hwh | 替换为同 stem 新文件，重做 C2 配对 | 替换 IP，重跑 hwh 解析 |
| 地址常量 | 从 hwh `MEMRANGE` 重取（如 `0x480000000` 依 BD 而变） | 从 IP 基址重取 |
| 时钟常量 | 从 `CLKFREQUENCY`/驱动 `CLOCK_HZ` 重取 | 同左 |
| 寄存器表 | 通常不变 | 按新 IP 契约更新偏移/位域 |
| 期望值 | 不变（同模型） | 重生成 golden |
| 判据 | 本文件 §1 原样复用 | 本文件 §1 原样复用 |

**通用性声明**（提交/复用时应写清）：

- 本技能不依赖具体模型：CNN/SNN/LNN/图像预处理/DMA 搬运任一 IP 都适用；
- 不依赖具体板卡：Zynq-7000 与 Zynq UltraScale+ 的 PYNQ 2.7+ 均可，仅地址/时钟常量按 hwh 替换；
- 不依赖本队构建：任何 `.bit`+`.hwh` 三件套即可接入；
- 不替代时序/资源结论：那属于 Vivado 报告（G2/G3）。

## 4 无板卡时的降级判定

| 可做 | 只能得出 |
|---|---|
| 三件套哈希配对、hwh 元数据解析 | 部署包自洽（G0/G4 级身份） |
| 读 summary.json 的既有运行记录 | 客方回传记录（须注明来源与日期） |
| 驱动静态检查（寄存器表、符号扩展、超时值） | 代码级风险清单 |

以上任一项都不能写成"板端通过"。
