# 证据类故障四元组

> 每条来自真实记录（开发树/证据快照/协作记录）。格式：现象 → 根因 → 处置 → 证据。
> 本表只收"导致结论不可信"的故障，不收录工具本身的普通报错。

## T1 历史结果被当成本次结论

- 现象：报告写"综合、时序已通过"，实际引用的是 2026-10-29 的 r4 实现报告。
- 根因：历史数据与本次任务没有身份绑定，只在目录里同名。
- 处置：结论改为 `record_level`；补写 run 名与日期；本次未跑的项目单列 `not_run`。
- 证据：`board/cnn/MANIFEST.json`（`identity.revision = "A timing-fix implementation r4"`）、包根 `README.md` 边界段。

## T2 manifest 哈希漂移

- 现象：复算 `manifest_sha256` 与记录不一致，疑似产物被污染。
- 根因：运行环境变化（缓存、Python 路径、输出目录）、源清单刷新、checkpoint 路径字符串不同。
- 处置：清理缓存、固定运行参数与路径、写入检查单；用"新输出目录 + 同 seed"复跑证明确定性。
- 证据：`sim/snn/evidence/ACCEPTANCE_CORE.md` §A1（`manifest.json` 文件 sha256 在 16:50/17:00/17:38 三值，但 params 集合摘要不变）。

## T3 板端记录级当 pass

- 现象：报告写"板端 3/3 通过"，但原始板端 JSON/日志未随包。
- 根因：把"主会话口头记录"当证据。
- 处置：按客方回传记录处理，注明来源与日期；写明原始文件未随包，不与本机实测混写。
- 证据：`sim/snn/evidence/INDEX.md` §1 板测行。

## T4 partial 被写成 pass（平台集成）

- 现象："平台集成通过"。
- 根因：只看了 bit/HWH 已生成与时序为正，忽略 CDC 1 Critical 未收敛、软复位不排空 DMA、协议仿真矩阵待补。
- 处置：`partial`，逐条列未收敛项；只有 G5 全部子项齐才可 pass。
- 证据：`sim/snn/evidence/platform/platform_build_20261030.json`、`CDC_FIX_R6_RESULT_20261030.md`。

## T5 @simulation-model 写成 hardware

- 现象：离线驱动自检通过，被当作板端实测。
- 根因：LNN 离线模型（`source=simulation-model`）与真实板端共用同一输出格式。
- 处置：证据里保留 `source` 字段；板端结论必须来自硬件运行并留波形/日志；离线只作流程自检。
- 证据：`sim/lnn/evidence/ACCEPTANCE.md` §1 板端行（"离线模型 29/29 PASS 仅作流程自检"）。

## T6 RTL 刷新后仍引用旧回放

- 现象：RTL 改了，旧回放证据仍在索引里当有效证据。
- 根因：证据未绑定 RTL 哈希。
- 处置：维护冻结哈希清单与 `invalidated_runs`；回归时按"冻结集"比对，SAME/DIFF 逐文件记录。
- 证据：`sim/snn/evidence/nlp/rtl_freeze.json`（18 文件冻结）、`rtl_hash_check_20261031.txt`（16 SAME / 2 DIFF）。

## T7 bit 与 hwh 不同次构建

- 现象：板端行为与预期不符，复位/地址异常。
- 根因：bit 与 hwh 来自两次实现（或 XSA 内 bitstream 与 .bit 不一致）。
- 处置：做配对哈希；不一致直接 `fail`，重新导出；三件套同源后再测。
- 证据：LNN `ACCEPTANCE.md` §3 配对检查（XSA 内嵌 bitstream 与 .bit 一致、hwh 逐字节一致）；CNN `MANIFEST.json` 的 `bit_sha256`/`hwh_sha256`。

## T8 估算资源写成实测

- 现象：写"资源超标 ≈846k FF、2304 MAC/拍"。
- 根因：静态推导被当成综合结果；同一报告后文又用另一口径纠正。
- 处置：标 `derived`，只写"估算、未测"；实测必须来自综合/实现报告（`utilization.rpt`/`util` JSON）。
- 证据：`sim/snn/evidence/vision/native_resource/VISION_NATIVE_RESOURCE_HANDOFF_20261031.md`。

## T9 IP 包验收写成平台完成

- 现象："IP 已交付，平台完成"。
- 根因：混淆 G4 与 G5/G6。
- 处置：按门级拆开：IP 包 = G4；BD/bit = G5；板端 = G6；缺 BD 的写 `not_run`。
- 证据：`src/cnn/docs/PLATFORM_AND_STATUS.md`（"单独的 IP 包不是完整板级平台，也不是 bitstream"）。

## T10 证据不可移植（绝对路径硬编码）

- 现象：证据目录搬到新机后复算失败、脚本跑不起来。
- 根因：脚本/TB/清单里写死 `D:/dl_snn`、`D:/DL_FINAL` 等绝对路径。
- 处置：包内路径改相对；保留必要绝对路径时在索引里登记"已知硬编码"清单。
- 证据：`sim/snn/evidence/PORTABILITY_DIFFS.txt`（16 文件/343 行差异）；`ACCEPTANCE_CORE.md` §A5 记录 TB 内 70 处 `D:/dl_snn` 属规范范围外。

## T11 证据目录被覆盖

- 现象：同名输出目录复跑，旧证据与本次混在一起。
- 根因：复用 run 目录；工具默认输出到同一路径。
- 处置：每次新 run-id/新目录；证据目录只增不改；旧目录归档不改名。
- 证据：`src/cnn/docs/USER_GUIDE.md`（"每次运行都使用新的输出目录，不要复用旧编译目录，因为清单会绑定本次模型和 RTL 源码身份"）。

## T12 重活锁冲突

- 现象：OOC 综合长时间未收敛/内存不足，疑似 RTL 问题。
- 根因：同机已有外部 Vivado 进程占用内存与 license，测量环境不干净。
- 处置：先记 `heavy_wait`（外部 PID、free memory 采样），等资源后重跑；结论里注明等待时间线。
- 证据：`sim/snn/evidence/vision/ref_route/ooc_native/heavy_wait_latest.json`、`memwatch_latest.log`。

## T13 行为仿真 PASS，综合后 FAIL

- 现象：XSim/ModelSim 图级功能通过，板端却在 descriptor checker 返回 `ERR_OPCODE=2`。
- 根因：Vivado 2021.1 把"两份内容相同的数组"中的引擎副本写端口优化成常量（`WEBWE[3:0]=<const0>`、`ENBWREN=<const1>`），
  行为仿真看不到该优化。
- 处置：证据链补"综合后网表级验证/端口断言"（如 `assert_task_table_ports.tcl` 在 synth/routed 后拒绝写使能悬空）；
  G1 通过**不覆盖** G2 之后；报告里 G1 结论不要外推到板端。
- 证据：`[工作区] D:/dl_v1.0/docs/PYNQ_ERR_OPCODE_ROOT_CAUSE_FIX_20261017.md`。

## T14 时序数字的时钟口径错误

- 现象：实现报告写"200 MHz 满足"，实际 `create_clock` 作用在错误网络上，周期模型是 4.999 ns 而真实为 4.997 ns；
  或出现 `WNS -0.281 / 1519 failing`。
- 根因：clk_wiz 的普通 IP XDC 在内部 `clk_in1` 重建 `create_clock -period 3.333`，覆盖上游 MIG 的 `mmcm_clkout0`；
  BD 未改 `PRIM_SOURCE=No_buffer`。
- 处置：证据里记录"约束文件 + 时钟树来源 + 实测周期 + `check_timing` 结果"；不降频、不加 false path 换数字。
- 证据：`[工作区] D:/dl_v1/docs/M4_PLATFORM_A_IMPL_TIMING_QUALIFIED_20261028.md` §1~§4。
