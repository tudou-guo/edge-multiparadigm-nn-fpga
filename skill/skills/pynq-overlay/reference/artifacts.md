# 产物与字段表（PYNQ overlay）

## 1 summary.json（板端运行摘要）

| 字段 | 类型 | 必填 | 含义 / 真实取值示例 |
|---|---|---|---|
| `source` | str | 是 | `hardware`（板端实测）或 `simulation-model`（离线模型，只能 partial） |
| `is_hardware_measurement` | bool | 是 | 与 `source` 一致 |
| `pynq_version` | str | 是 | 如 `2.7` |
| `platform` | str | 是 | `PL.device_name`，如 `xczu15eg` |
| `bit_sha256` / `hwh_sha256` | hex64 | 是 | 部署包哈希（与 XSA 内嵌比对过） |
| `id_check` / `version_check` / `clock_hz_check` | bool | 是 | 读 IP 内只读寄存器与期望值比对 |
| `samples` / `matched` | int | 是 | 样本数与逐位一致数（如 64/64、3/3） |
| `first_mismatch` | int/null | 是 | 首个不一致索引，全对为 `null` |
| `error_code` | int | 是 | IP 错误寄存器值，正常为 0 |
| `wall_s` | float | 是 | 端到端耗时（PS 提交 + 轮询 + 读回） |
| `hw_cycles` | int | 否 | 硬件计数寄存器值（若 IP 提供） |
| `hw_time_s` | float | 否 | `hw_cycles / CLOCK_HZ` |
| `notes` | list | 否 | 已知限制、外部数据来源、板卡地址等 |

## 2 hwh 关键字段

| 字段 | 示例 | 用途 |
|---|---|---|
| `DEVICE`/`PACKAGE`/`SPEEDGRADE` | `xczu15eg`/`ffvb1156`/`-2` | 与板卡/bit 器件一致性 |
| `CLKFREQUENCY` | `300000000` | 换算 cycles→秒；与驱动常量比对 |
| `MEMRANGE` | `BASEVALUE="0x480000000"`、`HIGHVALUE="0x480000FFF"` | IP 地址窗；驱动常量来源 |
| `INSTANCE` | `M00_AXI_0` | BD 内连接关系 |

hwh 地址是 64 位字符串，驱动里必须用 64 位整型保存（Python int 即可），不要截断成 32 位。

## 3 寄存器表模板（放进驱动或文档）

| 偏移 | 名称 | 位宽 | 读写 | reset | 说明 |
|---|---|---|---|---|---|
| `0x000` | ID | 32 | RO | 固定 | 期望 `0x....` |
| `0x004` | VERSION | 32 | RO | 固定 | 主版本<<16\|次版本 |
| `0x008` | CONTROL | 32 | RW | 0 | bit0=START，bit1=SOFT_RESET |
| `0x00C` | STATUS | 32 | RO | 0 | bit0=BUSY，bit1=DONE |
| `0x010` | ERROR_CODE | 32 | RO | 0 | 0=正常 |
| `0x020` | RUN_CYCLES | 32 | RO | 0 | 硬件周期计数 |
| `0x02C` | CLOCK_HZ | 32 | RO | 固定 | 运行时钟频率 |

写寄存器用读-改-写；写 1 清零位、只读位按契约单独处理。

## 4 结果目录约定

```
results/<run>/
├── summary.json      # §1 字段
├── outputs.npy       # 板端读回（与期望同 dtype/布局）
├── compare.json      # {samples:[{idx, matched, first_mismatch}], total, passed}
├── board.log         # 环境、寄存器读写、DMA 起止、耗时
└── SHA256SUMS        # 上述 + overlay.bit/hwh
```

## 5 期望值目录约定

```
golden/
├── exp.npy | exp.hex        # 期望输出（标注 dtype/字节序/字节序）
├── inputs/*.hex             # 输入向量
└── meta.json                # 生成规则、来源、sha256
```

期望值与读回值必须同布局；不同布局（如 int16 补码 vs uint16）是最常见的"看起来像算法错"的根因。
