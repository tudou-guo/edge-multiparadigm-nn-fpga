# 用户指南 — LNN CfC PYNQ 演示（D:/DL_FINAL/LNN 1.0.0）

本指南面向**使用与演示**：把 CfC 液态神经网络单步计算图的 PL overlay 部署到 ZU15EG 板卡，
用 PYNQ 主机驱动连续完成 64 步推理，并与定点参考逐点比对。接口细节见 `docs/INTERFACE.md`。

## 1. 交付内容

| 内容 | 位置 | 说明 |
|---|---|---|
| 位流三件套 | `board/lnn_cfc.bit` `board/lnn_cfc.hwh` `board/lnn_cfc.xsa` | bit 与 hwh 同 stem、同一次构建；XSA 存档 |
| PYNQ 驱动 | `board/lnn_cfc_run.py` | overlay 加载、ID/版本/时钟校验、写输入、START、轮询、读回、逐点比对、summary.json |
| 整数参考/寄存器模型 | `board/cfc_sim_model.py` | 契约语义的逐位整数参考；无板时作离线后端 |
| 演示 Notebook | `board/lnn_cfc_demo.ipynb` | 六步现场流程与四曲线 |
| 期望值 | `board/export/` | 权重/LUT golden、8 组样本、manifest（33 文件） |
| 校验清单 | `board/SHA256SUMS`、`MANIFEST.sha256` | 文件完整性 |
| 模型资产 | `model/` | 单步 ONNX、训练权重、训练配置、浮点参考、曲线 |
| RTL / 仿真 | `rtl/` `verification/` `scripts/run_sim.tcl` | 可综合源码与自检测试台 |

## 2. 部署

~~~bash
# 主机（Windows）→ 板卡；整个 board/ 目录整体拷贝（自带 export/，开箱可跑）
scp -r D:/DL_FINAL/LNN/board xilinx@<板卡IP>:/home/xilinx/jupyter_notebooks/board
ssh xilinx@<板卡IP>
cd /home/xilinx/jupyter_notebooks/board
sha256sum -c SHA256SUMS
~~~

> 只拷了驱动/位流而缺少 `export/` 时：把 `export/`（或 `board/export/`）拷到驱动同目录，
> 或设置 `CFC_EXPECT_DIR` / 命令行 `--expect-dir`。驱动会自动在 `driver目录/export`、
> `driver目录/package/export`、`driver目录/*/export`、`cwd/export`、父目录 `export`、`CFC_EXPECT_DIR` 中查找。

## 3. 运行

### 3.1 板端命令行（推荐验收用）

~~~bash
python3 lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export \
    --sample sample_000 --out-dir results/s0_hw --checks
~~~

关键输出：

~~~text
CFC1 bring-up  source=hardware  hardware_measurement=True
  CSR base : 0x480000000 (64-bit=True)
  ID       : 0x43464331  ok=True
  VERSION  : 0x00010000  ok=True
  CLOCK_HZ : 300000000  ok=True
run sample_000:normal  mode=normal  n_steps=64
  status=done error_code=0 step_count=64 run_cycles=<实测值；参考 9088>
  outputs matched 64/64  bit_exact=True  final_state_match=True
  task_mse = <按 export 模型实测>
  RESULT: PASS
state checks: repeat PASS / segments PASS / clear_each_step PASS
~~~

模式：`--mode normal`（运行前清状态，默认）、`retain`（保留上次末状态）、`clear-each-step`（每步清状态）、
`clear-per-step-only`。样本：`--sample sample_00X`（0..7），`--n-steps N` 可跑 1..64 步（分段实验用）。
`--checks` 追加 重复运行 / 32+32 保留状态 / 每步清状态 三项一致性测试。

### 3.2 Notebook 现场演示

打开 `board/lnn_cfc_demo.ipynb`，顺序执行：

1. 加载 overlay，核对 CFC1 ID / VERSION / CLOCK_HZ 与部署文件 SHA256；
2. 选固定样本，显示 64 个输入点、Δt 与初始状态，跑浮点与整数参考；
3. 一次写入输入与配置，START，等待 DONE/ERROR（独立墙钟超时），读回全部输出与末状态；
4. **先做整数逐点比对**，通过后叠加 输入 / 目标 / 浮点参考 / FPGA 输出 四条曲线，显示任务 MSE、RUN_CYCLES、硬件耗时与端到端耗时；
5. 切换"每步清状态"模式，比较任务误差，解释状态记忆；
6. 恢复正常模式，确认状态清零与重复运行可复现，保存结果。

### 3.3 开发机离线自检（无板）

~~~bat
cd /d D:/DL_FINAL/LNN/board
E:/miniconda/envs/pytorch_clean/python.exe lnn_cfc_run.py --sample sample_000 --out-dir results/s0 --checks
~~~

输出 `source=simulation-model`：只验证驱动寄存器序列、位打包、比对逻辑与期望值，**不是板端实测**。

## 4. 验收标准（板端 PASS）

| 项 | 通过条件 |
|---|---|
| 身份 | `source=hardware`；ID=0x43464331；VERSION=0x00010000；CLOCK_HZ=300000000 |
| 功能 | `status=done`、`error_code=0`、`step_count=64`、`outputs matched 64/64 bit_exact=True`、`final_state_match=True` |
| 任务效果 | `task_mse` 与 export 记录的整数任务 MSE 一致（skip 5 步口径） |
| 状态语义 | `--checks`：重复运行一致、32+32 分段=连续 64 步、每步清状态与参考一致 |
| 性能 | RUN_CYCLES / CLOCK_HZ 得片上计算时间；端到端耗时以驱动独立计时为准（PYNQ MMIO 开销主要在 PS 侧） |

结果文件：`results/<run>/summary.json`（含 source、匹配数、末状态、MSE、周期数、耗时、文件哈希）、
`outputs.npy`（原始输出）、`state_checks.json`（--checks 时）。建议连同日志归档到 `evidence/board_run/`。

## 5. 故障排查

| 现象 | 原因/处理 |
|---|---|
| `export/ directory ... not found` | 未拷期望值：把 `export/` 拷到驱动同目录，或 `--expect-dir` / `CFC_EXPECT_DIR` |
| `board/cfc_sim_model.py is required next to the driver` | 驱动与 `cfc_sim_model.py` 必须同目录 |
| `ID mismatch` / `VERSION mismatch` | 位流与驱动不是同一版本：用本包成对的 bit + 驱动 |
| `hwh ... same stem` 报错 | bit 与 hwh 必须同目录同 basename（`lnn_cfc.bit` + `lnn_cfc.hwh`） |
| `pynq not available` | 板端需 PYNQ 镜像；开发机无 pynq 时驱动自动走 simulation-model |
| DONE 一直不置位 / 超时 | 检查 bit 是否加载成功（ID/版本）、`TIMEOUT_CYCLES`，必要时先 `CLEAR_STATUS` |
| ERROR_CODE=2 | 忙时重复 START（驱动不会这样做；检查是否有其它进程访问 CSR） |
| ERROR_CODE=3 | 运行超时（默认 1,000,000 周期 ≈3.33 ms，正常 64 步约 9.1k 周期） |
| ERROR_CODE=1 | 配置非法（N_STEPS 1..64、DELTA_T 1..32767） |
| `manifest ... mismatch` | export 被改动：用本包原文件，核对 `MANIFEST.sha256` |

## 6. 边界

- 本包不含板端实测结果文件；验收需现场生成 `source=hardware` 的 summary.json。
- 不支持运行中改配置；不支持可变 Δt、多网络规模（后续版本）。
- 位流为该版本 RTL/权重的一次构建产物；改动任一输入需重新构建并重新验收时序（本次 WNS +0.226 / WHS +0.011 ns）。
