# SNN 平台时序收敛结果（r6，2026-10-30）

工程：`D:/spk_platform/project_1/project_1.xpr` ｜ xczu15eg-ffvb1156-2-i ｜ Vivado 2021.1

## 结论：routed 时序已收敛

| 项 | 结果 |
|---|---|
| Setup | **WNS +0.245 ns / TNS 0.000 / 0 failing**（315,896 endpoints） |
| Hold | **WHS +0.010 ns / THS 0.000 / 0 failing**（315,306 endpoints） |
| 脉宽 | WPWS +0.039 / TPWS 0 / 0 failing |
| 布线 | unrouted 0，routing errors 0 |
| DRC | 132 条全 Warning（DSP PREG 128、LUT 方程 3、无负载 1），0 Error/0 Critical |
| Bus skew | 0 violation |
| CDC | 用户逻辑无未解释项；1 条 CDC-10 Critical 属 MIG/BD 复位 IP 内部（已有 IP False Path）；CDC-15 全在 dbg_hub |
| 用户时序例外 | 0（无 set_false_path / set_clock_groups；例外均来自 IP XDC） |
| 资源 | LUT 72,167 (21.15%)、FF 110,508 (16.19%)、BRAM 227 (30.51%)、DSP 131 (3.71%) |

**产物**：
- routed DCP：`project_1/project_1.runs/impl_1/top_routed.dcp`
- bitstream：`project_1/project_1.runs/impl_1/top.bit`（28.7 MB）
- XSA：`project_1/snn_platform.xsa`（4.7 MB，含 bit）
- 报告：`reports/cdc_fix_r6_20261030_2000/`

## 本轮按工程师方案做了什么

1. **平台专用适配层**（新增 `src/snn/spikformer_nlp_platform_top.sv`）
   - 与 OOC 适配层端口/参数完全一致，直接实例化 `spikformer_nlp_top`，**不再带 8 级 LUT1 hold 延迟链**。
   - OOC 适配层 `spikformer_nlp_target_top.sv` 及其产物/哈希未动。
   - 平台 wrapper 的 `u_core` 切换到新模块。

2. **残余 CDC 改为 XPM/事件 FIFO**
   - start：删除独立 toggle，**配置包 FIFO 到达即启动**；
   - done/error：删除 toggle，**状态包 FIFO 为事件载体**（code=0 成功，非 0 错误）；
   - busy → `xpm_cdc_single`；软复位、flush、写完成 b_done → `xpm_cdc_pulse`；
   - 写完成 arm 移到 DMA 域写命令握手处；IRQ 改为 DMA 域本地 sticky。
   - 结果：200↔300 MHz 跨域综合 WNS 从 +0.776 提升到 **+4.709**，routed 无跨域失败。

3. **结果帧/完成发布（部分落地）**
   - 帧计数在写命令握手时重建；只有 32 拍且 last=1 才发布；done 要求完整帧，否则报 error code 5。

4. **最终实现**：`Performance_ExplorePostRoutePhysOpt`，route 后跑 post-route `phys_opt_design -directive Explore`，再 `write_bitstream`，最后生成 XSA 和全套报告。

## 与 r5 对比

| 指标 | r5 | r6 |
|---|---:|---:|
| Setup WNS | -0.886 | **+0.245** |
| Setup failing | 9,728 | **0** |
| Hold WHS | +0.001 | **+0.010** |
| LUT | 76,489 | 72,167 |
| bit/XSA | 无 | **有** |

## 未完成 / 已知限制

1. 完整协议仿真矩阵未跑（FIFO 双向回绕/随机反压/多相位、配置快照、32 拍首末暂停、软复位排空、B 延迟、4KB 边界）；目前只有原 CSR/超时 smoke PASS（DMA 数据事务为 0）。
2. 软复位仍是“flush 两侧 FIFO + 清状态”，未实现方案要求的“排空在途 DMA 事务后复位”；写 DMA 已发事务的 B 响应未被等待。
3. 写完成 B 数仍用常数 `WR_EXPECT`（假设 512 B 输出），不覆盖其它长度/4KB 边界拆分。
4. 顶层 `sys_rst` 无 input delay / false path（check_timing HIGH），`C0_DDR4_0_reset_n` 无 output delay；二者是板级异步复位/静态输出，建议下一轮补约束。
5. r6 的 hold 最差路径是新增的 `sums_shadow → sums_pub`（+0.010 ns，thin margin），后续改动需复查。

## 证据

| 内容 | 路径 |
|---|---|
| 最终 setup | `reports/r6_final_timing_setup.rpt` |
| 最终 hold | `reports/r6_final_timing_hold.rpt` |
| CDC | `reports/r6_final_cdc.rpt` |
| DRC / 布线 / 资源 / methodology | `reports/r6_final_*.rpt` |
| 本轮汇总 | `reports/cdc_fix_r6_20261030_2000/RESULT.md`、`CHANGELOG.md` |
| 运行日志 | `logs/impl_snn_r6.log`、`logs/final_r6.log` |