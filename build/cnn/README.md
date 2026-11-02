# build/cnn

CNN 库的构建入口与实现报告。

## 脚本

| 文件 | 用途 |
|---|---|
| `scripts/configure_e1p3_bd.tcl` | 配置 PS-DDR 平台 Block Design（E1-P3 平台） |
| `scripts/run_e1p2_ooc_synth.tcl` | OOC 综合入口 |
| `scripts/run_hw1_ooc_synth.tcl` | 硬件包 OOC 综合 |
| `scripts/pack_npu_ip.tcl` | Vivado IP Catalog 打包 |
| `scripts/run_p4_stage1.ps1` / `run_p4_host_preflight.ps1` | Windows 主机预检与阶段构建 |
| `scripts/env_check.py` | 环境自检 |

## 报告

- `reports/legacy_full_engine_placed_utilization.rpt`：旧全算子平台基线（191,542 LUT / 527 DSP）
- `reports/r4_platform_utilization_historical.rpt` / `r4_hierarchical_utilization_historical.rpt`：历史 r4 实现（140,028 LUT / 471 DSP）
- `reports/legacy/`：早期板卡包（R1 / MNIST 200 MHz）的时序与利用率报告

## 流程与期望

1. 转译：`python -m tools.onnx_gen --onnx model.onnx --out out/cnn --tn 4 --tm 4 --mac-impl 3`
2. IP 打包：`python -m tools.npu_gen --generated out/cnn/generated --out out/cnn/ip_repo`
3. Vivado：加 IP → 接 BD（可参考 `configure_e1p3_bd.tcl`）→ 综合/实现
4. 期望：5 ns 约束下 0 失败端点；YOLO 103 条指令记录；ModelSim 4 案例 0 mismatch

> 注意：CNN 主线四案例的综合/时序为 not_run，报告中的资源/时序来自历史 r4 实现；16/8 并行度
> 端到端 51.4 ms（PS 20 + PL 31）为板端记录。
