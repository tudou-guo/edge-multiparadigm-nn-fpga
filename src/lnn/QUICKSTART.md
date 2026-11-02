# QUICKSTART — 3 条命令跑通 LNN CfC 演示

前置：
- **开发机（Windows）**：Python 3.9+（本快照参考环境 `E:/miniconda/envs/pytorch_clean/python.exe`，numpy 必需，matplotlib 可选）
- **板卡**：PYNQ 镜像（`pynq` + numpy + matplotlib），ZU15EG（RIGUKE）平台
- **仿真（可选）**：Vivado 2021.1（含 xsim），本机 `E:/Vitis/Vivado/2021.1`
- 全部命令在**快照根**（`D:/DL_FINAL/LNN`）执行；板端命令在该目录下的 `board/` 执行

## 0. 环境检查

~~~bat
cd /d D:/DL_FINAL/LNN
E:/miniconda/envs/pytorch_clean/python.exe scripts/env_check.py
~~~

期望看到 `numpy 2.0.2`、`bitfile : present`、`export : present`。

## 1. 开发机离线自检（无需板卡）

用 Python 寄存器模型代替 FPGA，验证驱动、位打包、逐点比对逻辑与期望值：

~~~bat
cd /d D:/DL_FINAL/LNN/board
E:/miniconda/envs/pytorch_clean/python.exe lnn_cfc_run.py --sample sample_000 --out-dir results/s0 --checks
~~~

期望输出（`source=simulation-model`，**不是板端实测**）：

~~~text
CFC1 bring-up  source=simulation-model  hardware_measurement=False
run sample_000:normal  mode=normal  n_steps=64
  status=done error_code=0 step_count=64
  outputs matched 64/64  bit_exact=True  final_state_match=True
  RESULT: PASS
state checks: repeat PASS / segments PASS / clear_each_step PASS
~~~

## 2. 上板部署与实跑（推荐把整个 `board/` 目录整体拷贝）

~~~bash
# 主机 -> 板卡（board/ 自带 export/ 期望值，开箱可跑）
scp -r D:/DL_FINAL/LNN/board xilinx@<板卡IP>:/home/xilinx/jupyter_notebooks/board
ssh xilinx@<板卡IP>

cd /home/xilinx/jupyter_notebooks/board
sha256sum -c SHA256SUMS                 # 校验交付文件
python3 lnn_cfc_run.py \
    --bit lnn_cfc.bit \
    --expect-dir export \
    --sample sample_000 \
    --out-dir results/s0_hw \
    --checks
~~~

板端 PASS 判据（全部满足）：`source=hardware`、ID=`0x43464331`、VERSION=`0x00010000`、
CLOCK_HZ=`300000000`、`outputs matched 64/64 bit_exact=True`、末状态一致、`step_count=64`、`error_code=0`。

## 3. Notebook 现场演示（方案 §7.1 六步）

在板卡 Jupyter 中打开 `board/lnn_cfc_demo.ipynb`，依次运行：
加载 overlay 并核对 ID/版本/时钟/哈希 → 选样本显示 64 输入/Δt/初始状态并跑浮点与整数参考 →
写输入启动、等待 DONE、读回 → **先整数逐点比对**再画四曲线（输入/目标/浮点参考/FPGA 输出）与指标 →
切换"每步清状态"模式对比 → 恢复正常模式确认可复现并保存结果。

> 若只拷了驱动/位流而缺少 `export/`：把本快照的 `export/` 或 `board/export/` 拷到驱动同目录，
> 或设置 `CFC_EXPECT_DIR` / 命令行 `--expect-dir`。驱动会自动在 `driver目录/export`、`driver目录/package/export`、
> `driver目录/*/export`、`cwd/export`、父目录 `export`、`CFC_EXPECT_DIR` 中查找。

## 4.（可选）RTL 自检仿真

~~~bat
cd /d D:/DL_FINAL/LNN
call E:\Vitis\Vivado\2021.1\settings64.bat
vivado -mode batch -notrace -source scripts/run_sim.tcl ^
    -journal verification\logs\vivado.jou -log verification\logs\vivado.log
~~~

期望最后一行 `CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol`，exit code 0；
日志在 `verification/logs/`。仿真自检不需要板卡，可验证 RTL 与 `export/gen` 期望值逐位一致。
