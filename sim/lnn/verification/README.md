# verification/ — CfC RTL 自检测试台

三个自检测试台 + 逐位整数参考模型，可直接用 Vivado xsim 重跑，验证 RTL 与 `export/gen` 期望值逐位一致。

| 文件 | 内容 |
|---|---|
| `tb_cfc_cell.sv` | 单步核：21 组边界/随机 (x,h,Δt)，逐位比对 y 与 4 个新状态（期望 0 mismatch，约 139 周期/步） |
| `tb_cfc_seq.sv` | AXI-Lite 端到端 64 步：OUTPUT/H_FINAL vs `export/gen/y_expected.hex`、`h_expected.hex` 与内嵌参考；状态清零重复运行、32+32 分段、每步清状态；期望 RUN_CYCLES=9088、STEP_COUNT=64、0 mismatch |
| `tb_cfc_protocol.sv` | 协议矩阵：独立 AW/W 三种顺序、WSTRB、B 延迟、读反压/2 拍读延迟、忙时 START（错误 2）、非法配置（错误 1）、超时（错误 3）与恢复、CLEAR_STATUS/CLEAR_STATE、未实现偏移读 0 |
| `cfc_ref_model.svh` | 契约 V1.1 整数参考（48 位精确累加、Q12 舍入、LUT 索引、完整门控 b_t = W_b z + b_b） |
| `filelist.f` | 编译列表（8 个 RTL + 3 个 TB；路径相对本目录） |

运行（在快照根，先 source Vivado settings64）：

~~~bat
cd /d D:/DL_FINAL/LNN
call E:\Vitis\Vivado\2021.1\settings64.bat
vivado -mode batch -notrace -source scripts/run_sim.tcl ^
    -journal verification\logs\vivado.jou -log verification\logs\vivado.log
~~~

期望 `CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol`，exit code 0；日志写入 `verification/logs/`。
