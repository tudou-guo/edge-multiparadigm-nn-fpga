# 上板说明 — LNN CfC PYNQ 演示（CFC1 AXI-Lite / ZU15EG RIGUKE）

本目录是 **PYNQ 板端交付包**（LNN CfC 演示发布快照的 `board/`）：CfC 单步计算图（1 输入 / 4 隐状态 / 1 输出）
的 PL overlay、PYNQ 驱动、演示 Notebook、定点期望值与哈希清单。
接口说明见 `../docs/INTERFACE.md`，部署与验收见 `../docs/USER_GUIDE.md`。

---

## 0. 证据口径（必须先读）

| 结果来源 | summary.json / offline_summary.json 中的标记 | 能否写入验收报告 |
| --- | --- | --- |
| 真实板卡（overlay + MMIO） | `source = "hardware"`, `is_hardware_measurement = true` | 可以，必须附日志 |
| 离线 Python 寄存器模型 | `source = "simulation-model"`, `is_hardware_measurement = false` | **不可以**（标注"未在真实板卡上运行"） |

- `../evidence/offline_validation_summary.json` 是开发机上用严格按契约实现的 Python 寄存器模型
  （`cfc_sim_model.py`）跑出的**离线校验**，只证明驱动寄存器序列、位打包、比对逻辑、
  Notebook 流程和寄存器语义自洽，**不是板端实测**。
- 真正的板端验收必须来自板上运行的 `lnn_cfc_run.py` 生成的 `summary.json`
  （`source=hardware`）。

---

## 1. 目录内容

| 文件 | 说明 |
| --- | --- |
| `lnn_cfc.bit` | PL 配置码流（Vivado 构建，与 .hwh 同一次构建、同 basename） |
| `lnn_cfc.hwh` | PYNQ Overlay 必需的 hardware handoff（与 bit 同 stem） |
| `lnn_cfc.xsa` | 同一次构建的硬件平台导出（存档用） |
| `lnn_cfc_run.py` | PYNQ 驱动：bring-up 校验 → 写输入/配置 → START → 轮询 → 读回 → 逐点比对 → summary.json |
| `cfc_sim_model.py` | 契约 V1.1 整数参考 + 离线寄存器模型（板端只用于生成每步清状态模式的期望值） |
| `cfc_board_ui.py` | Notebook 的显示/绘图辅助函数 |
| `lnn_cfc_demo.ipynb` | 六步现场演示 Notebook（方案 §7.1） |
| `export/` | 权重/偏置/LUT/hex、整数 golden（`gen/`）、8 个样本、manifest.json、cfc_spec.py、int_reference.py。manifest 中 export/ 内的 33 个文件随包分发且哈希可校验；其余条目指向开发树（reports/model），板上不存在属正常，驱动只告警不判失败 |
| `SHA256SUMS` | 本目录全部交付文件哈希（`sha256sum -c SHA256SUMS` 校验） |

> 离线校验摘要与验收数字放在快照根 `../evidence/`；构建/时序证据见 `../evidence/ACCEPTANCE.md`。

---

## 2. 板卡与环境要求

- 板卡：RIGUKE ZU15EG（`xczu15eg-ffvb1156-2-i`），PL DDR4 正常；PYNQ 镜像可 `from pynq import Overlay, MMIO`。
- 板端 Python：numpy、matplotlib、jupyter（PYNQ 镜像自带）。
- `lnn_cfc.bit` 与 `lnn_cfc.hwh` **必须同目录同 basename**，否则 PYNQ `Overlay()` 报
  "hardware handoff not found next to bitstream"。
- CSR 窗口：物理地址 **0x0480000000**（M00_AXI_0，4 KiB，64 位地址，驱动不做 32 位截断）。
- 应用时钟：300 MHz（`CLOCK_HZ` 寄存器 RO 值，驱动在 bring-up 时校验）。

---

## 3. 部署与板端运行

**部署方式**：把本 `board/` 目录**整体**拷到板子（已自带 `export/` 期望值，必须让 `lnn_cfc_run.py` 与 `export/` 同目录）：

```bash
scp -r <主机>/D:/DL_FINAL/LNN/board xilinx@<板IP>:/home/xilinx/jupyter_notebooks/board
```

> 只拷贝了驱动/位流而没有 `export/` 时，驱动会报
> `export/ directory (weights, LUTs, integer golden) was not found`。
> 修复（三选一）：
> 1. `scp -r <主机>/board/export xilinx@<板IP>:<驱动所在目录>/export`（最快）；
> 2. 在 Notebook 里设 `os.environ["CFC_EXPECT_DIR"] = "/path/to/export"` 后**重启内核**；
> 3. 命令行加 `--expect-dir /path/to/export`。
> 驱动会在 `driver目录/export`、`driver目录/package/export`、`driver目录/*/export`、`当前目录/export`、
> 父目录 `export` 以及 `CFC_EXPECT_DIR` 中自动查找（要求同时存在 weights.hex 与 int_reference.py），
> 找不到时错误信息会列出搜索过的路径。

```bash
# 1) 拷贝本目录到板子并登录
scp -r <主机>/D:/DL_FINAL/LNN/board xilinx@<板卡IP>:/home/xilinx/jupyter_notebooks/board
ssh xilinx@<板卡IP>

# 2) 进入目录并校验文件哈希
cd /home/xilinx/jupyter_notebooks/board
sha256sum -c SHA256SUMS

# 3) 单样本运行（默认样本 sample_000，64 步，含状态测试）
python3 lnn_cfc_run.py \
    --bit lnn_cfc.bit \
    --expect-dir export \
    --sample sample_000 \
    --out-dir results/s0_normal \
    --checks
```

预期关键输出（数值以现场日志为准，不在此预填）：

```text
CFC1 bring-up  source=hardware  hardware_measurement=True
  CSR base : 0x480000000 (64-bit=True)
  ID       : 0x43464331  ok=True
  VERSION  : 0x00010000  ok=True
  CLOCK_HZ : 300000000  ok=True
  ...
run sample_000:normal  mode=normal  n_steps=64
  status=done error_code=0 step_count=64 run_cycles=<硬件周期数；RTL xsim 参考值 9088>
  outputs matched 64/64  bit_exact=True  final_state_match=True
  task_mse = <按 export 模型实测>
  RESULT: PASS
...
  repeat               PASS ...
  segments             PASS ...
  clear_each_step      PASS ...
```

PASS 判据（全部满足才算板端 PASS）：

1. `source = hardware`，ID/版本/CLOCK_HZ 校验通过；
2. 64/64 输出与 `export/gen/y_expected.hex`（或样本 `int_y`）**逐位一致**；
3. `H_FINAL` 与 `h_expected` 末状态一致，`STEP_COUNT == N_STEPS`，`ERROR_CODE == 0`；
   RTL xsim 的 64 步 RUN_CYCLES 参考值为 9088（`≈30.3 µs @300 MHz`），板端应以实测 `summary.json` 为准；
4. `--checks` 的 repeat / 32+32 / 每步清状态三项 PASS。

常用参数：

```bash
# 只跑正常模式，写入指定目录
python3 lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --sample sample_000 --out-dir results/s0

# 换样本（export/gen/sample_00X.json，共 8 个）
python3 lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --sample sample_003 --out-dir results/s3

# 列出可用样本
python3 lnn_cfc_run.py --list-samples --expect-dir export

# 每步清状态模式（对比状态记忆作用）
python3 lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --mode clear-each-step --out-dir results/s0_ce

# 开发机无板时（仅流程自检；summary 会标注 simulation-model）
python3 lnn_cfc_run.py --backend simulation-model --expect-dir export --sample sample_000 --out-dir results/dev
```

单次运行生成的产物：

| 文件 | 内容 |
| --- | --- |
| `summary.json` | sample id、匹配数/总数、末状态、任务 MSE、周期数、硬件耗时（RUN_CYCLES/CLOCK_HZ）、端到端耗时、bit/hwh/manifest SHA256、source |
| `outputs.json` | 64 点整数/实值输出、末状态、周期数、状态 |
| `outputs.npy` | 64 点 int16 输出（便于复算） |
| `compare.json` | 逐点比对明细（期望/实际/mismatch 索引/MSE） |
| `state_checks.json` | repeat / 32+32 / 每步清状态结果（`--checks`） |

---

## 4. Notebook 演示（方案 §7.1 六步）

```bash
cd /home/xilinx/jupyter_notebooks/board
jupyter notebook lnn_cfc_demo.ipynb
```

按顺序执行单元格：

1. **步骤 1**：加载 overlay，显示 CFC1 ID / VERSION / CLOCK_HZ / bit·hwh·manifest 哈希（source 横幅）；
2. **步骤 2**：固定样本的 64 点输入、ΔT=655、初始状态，以及浮点/整数参考曲线；
3. **步骤 3**：一次写入输入与配置 → START → 等 DONE/ERROR（独立墙钟超时）→ 读回；
4. **步骤 4**：先整数逐点比对；通过后才画 输入 / 目标 / 浮点参考 / FPGA 输出 四条曲线，显示任务 MSE、周期数、硬件耗时、端到端耗时；
5. **步骤 5**：每步清状态模式对比（任务 MSE 变差 → 说明历史状态的作用）；
6. **步骤 6**：恢复正常模式重复运行 + 32+32 分段保留状态，确认可复现；保存本次结果到 `results/`。

可调整环境变量（在 Notebook 首包 cell 中读取）：`CFC_BIT`、`CFC_XSA`、`CFC_EXPECT_DIR`、
`CFC_SAMPLE`、`CFC_BACKEND`（默认 auto）、`CFC_TIMEOUT_S`、`CFC_RESULTS_DIR`。

> 会前可用"预存结果"核对，但现场结果必须标注是否为本次实测；若步骤 1 显示
> `simulation-model`，说明当前不是板卡环境，所有数字不得当作板端实测。

---

## 5. 失败排查

| 现象 | 原因 / 处理 |
| --- | --- |
| `hardware handoff (.hwh) ... same stem` | bit 与 hwh 不同目录或不同 basename；从交付包原样拷贝 |
| `ID mismatch: read 0x... expect 0x43464331` | overlay 未加载成功、CSR 地址窗口不对（应为 0x0480000000）或加载了旧 bit |
| `VERSION mismatch` | bit 与驱动/期望值不是同一版本；用本包成对的 bit + 驱动，并核对 `SHA256SUMS` |
| `CLOCK_HZ mismatch` | 该构建应用时钟不是 300 MHz；重跑时序正确的构建（禁止用 `--allow-clock-mismatch` 掩盖） |
| `ERROR_CODE=1`（CONFIG） | N_STEPS 不在 1..64 或 DELTA_T 不在 1..32767；驱动默认写 64/655，检查是否有外部程序改写 |
| `ERROR_CODE=2`（BUSY_START） | 上一次运行未结束就再次 START；等 DONE/ERROR 或复位 overlay |
| `ERROR_CODE=3`（TIMEOUT） | 硬件 `TIMEOUT_CYCLES`（默认 1,000,000 ≈ 3.33 ms @300MHz）内未完成；检查 RTL 卡死，必要时提高 TIMEOUT_CYCLES 或排查 |
| 驱动报 `wall-clock timeout` | PS 侧独立超时（默认 10 s）内未见 DONE/ERROR；SEU/时钟/复位或 overlay 加载问题，抓 `diagnostics` |
| `input write-back ok = False` | INPUT 窗口位打包或 AXI 写通道问题；检查 bit 与地址窗口 |
| `manifest ... hash mismatch` | export/ 被改动或与 bit 不同批次；用交付包原文件 |
| Notebook 显示 `simulation-model` | 当前没有 pynq/bit；确认在板子上运行、`CFC_BIT` 指向正确 bit |

---

## 6. 验收记录要点（方案 §6.2）

- 同时保留两类误差：**任务 MSE**（FPGA 输出 vs 目标 `x[t-4]`，跳过前 5 步预热）反映模型效果；
  **定点误差**（FPGA 输出 vs 定点 golden）反映硬件正确性，必须为 0。
- 硬件耗时 = `RUN_CYCLES / CLOCK_HZ`，只含片上计算；端到端耗时由板端单调时钟独立测量，
  含 PS 写入/轮询/读回，报告时分开列出。
- 循环播放曲线时要注明绘图帧率，不能与真实硬件计算时间混用。
- 每次验收保存 `summary.json`、`outputs.npy`、对比曲线 PNG 与日志，并记录
  bit/hwh/manifest 的 SHA256。

---

## 7. 已知限制

1. 本快照不包含板端实测结果文件：`../evidence/offline_validation_summary.json` 为离线模型校验，
   板端实测结论需在板卡上运行本 README 第 3/4 节命令后生成（`source=hardware`）。
2. 首版仅支持固定 ΔT=655、N_STEPS≤64、单时钟域 300 MHz；不支持运行中改配置（忙时写入被硬件丢弃）。
3. PYNQ MMIO 逐字访问较慢：端到端耗时主要来自 PS 侧写入/轮询，不能当作 PL 性能。
4. 运行中断/复位后建议重载 overlay 再跑；异常状态先 `CLEAR_STATUS`。
5. 板端依赖 PYNQ 镜像自带 `pynq`/`numpy`/`matplotlib`；未在其它 Python 版本上验证。
