# 门禁判据（G0–G6）

> 本文件只定义**怎么判、看哪里、阈值多少**；某一级是否通过由每次实际执行的结果与证据决定。
> 判据来源：随包清单/报告的字段与脚本，逐条可在 `sim/*/evidence/`、`data/`、`build/`、`board/` 中找到出处。

## 0 门级总览

| 级 | 名称 | 通过的含义 | 不能证明什么 |
|---|---|---|---|
| G0 | 编译/导出 | 工具链成功产出全部清单化产物 | 数值正确、时序、资源、板端 |
| G1 | 数值与回放 | 整数参考/行为仿真与参考一致（0 mismatch） | 可综合、可布线 |
| G2 | OOC 综合 | 单一顶层可综合、可布到目标器件（0 failing endpoints） | 平台集成、板端 |
| G3 | 实现/布线 | 平台实现收敛（WNS/WHS ≥ 0、0 unrouted） | 板端实测 |
| G4 | IP 打包 | IP Catalog 源码包自洽（component.xml/清单/源哈希） | Block Design、bitstream |
| G5 | 平台集成 | BD + IP + 约束 + bit/HWH 同源可构建 | 板端运行正确 |
| G6 | 板端 | 硬件实测逐位一致并留原始输出 | 泛化到其它输入/版本 |

级不可跳：G1 通过不代表 G2 通过；G3 的历史报告不能证明本次 G3。

## 1 G0 编译/导出

| 库 | 判定命令 | 通过阈值 | 证据落点 [包内] |
|---|---|---|---|
| CNN | `cd src/cnn && python -B -m tools.onnx_gen.onnx_gen compile ...` | 退出码 0；`records>0`；`compile_manifest.gates` 全 pass；`source_library verify` 的 `source_digest` 三处一致 | `data/cnn/{tinycnn,mobiledw,yolo}/compile_manifest.json` |
| SNN | `cd src/snn && python -m tools.toolchain --backend spikformer compile --config ... --out-dir ...` | 退出码 0；`manifest.json` 写入；`rtl mismatch: 0`（28/28）；params blob 数 28（fixture）/ 54（真实权重） | `build/snn/generated/_quickstart_real/manifest.json`、`sim/snn/evidence/nlp/final/manifest.json` |
| LNN | `cd D:/DL_FINAL/LNN && python -B user/export_demo.py --out-dir <新目录>` [工作区] | 输出 `RESULT: PASS  (7/7 files byte-identical)`；退出码 0 | `data/lnn/export/gen/`（23 文件）、`data/lnn/export/manifest.json` |

## 2 G1 数值与回放

| 库 | 判定 | 阈值 | 证据 [包内] |
|---|---|---|---|
| CNN | ModelSim 图级回放 | `status=pass`、`vlog_rc=0`、`sim_rc=0`、0 mismatch、`source_unchanged=true` | `data/cnn/mobiledw/summary.json`（schema `e1-p3-graph-sim-2`） |
| SNN | 整数参考 + 四档回放 | `math_tests_m1` 8/8 PASS；四档 `fail_lines=0`（cycles 39774 / 7084772 / 7084812 / 10627157） | `sim/snn/evidence/nlp/replay_*_holdfix.json`、`math_tests_m1.json` |
| LNN | ONNX 等价 / 定点参考 / RTL 逐位 | ONNX ≤1e-5（实测 4.768e-07）；定点 MSE ≤1e-3（实测 1.172e-04）；64 步 0 mismatch（21 用例 / 626 checks / RUN_CYCLES=9088） | `sim/lnn/evidence/ACCEPTANCE.md` §1 |

回放证据必须绑定 RTL 身份：SNN 用 `rtl_freeze.json`（18 文件冻结哈希）与 `rtl_hash_check_20261031.txt`（16 SAME / 2 DIFF，DIFF 属图像线并行改动）做比对，
"证据 ≠ 当前源码"时不得直接引用。

## 3 G2 OOC 综合

| 判据 | 阈值 | 证据形态 |
|---|---|---|
| 综合完成 | 无 `ERROR`；`P2_OOC_SYNTH_DONE=true`（CNN）或 `synth_*.json` 生成（SNN） | `utilization.rpt`、`timing_summary.rpt`、`drc.rpt`、`ooc_*.json` |
| 时序收敛 | WNS≥0、WHS≥0、TNS=THS=0、0 failing endpoints；无时序例外 | OOC 报告 JSON |
| 示例数值 | SNN NLP 5 ns：WNS **+0.135** / WHS **+0.023** / 0 failing / unrouted 0；LUT 40964 / FF 71173 / BRAM 186.5 / DSP 128 | `sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json` |

综合若在 elaboration 阶段报错（示例：`[Synth 8-91] ambiguous clock in event control`），结论判为 `fail` 并**不报**资源/时序数字；
impl/routed 与 UTL/TIM 报告均视为未产生。

## 4 G3 实现/布线

| 判据 | 阈值 | 证据形态 |
|---|---|---|
| 时序 | setup WNS≥0 / TNS=0 / 0 failing；hold WHS≥0 / THS=0；pulse WNS≥0 | `timing_summary.rpt`、`*_timing_*.rpt` |
| 布线 | 0 routing errors、fully routed = routable nets | `route_status.rpt` |
| DRC | 0 Error（Warning 需记录） | `drc.rpt` |
| 示例数值 | CNN r4（2026-10-29）：WNS +0.075 / WHS +0.010 / pulse +0.039 / 0 unrouted / DRC 0 Error；LNN：WNS +0.226 / WHS +0.011 / WPWS +0.039、57464/57464 全布线；SNN r7：WNS +0.051 / WHS +0.011 / 0 failing | `board/cnn/MANIFEST.json`、`sim/lnn/evidence/ACCEPTANCE.md` §2、`sim/snn/evidence/platform/` |

## 5 G4 IP 打包

| 库 | 判据 | 阈值 | 证据 |
|---|---|---|---|
| CNN | Vivado IP Catalog 源码包 | `component.xml` + 全部 RTL 源 + `sources.f`；`pack_manifest.counts` 与 `files[]` 自洽；`source_library verify --package` 通过 | 编译输出目录 `ip_repo/<name>_<ver>/` |
| SNN | `run_ip_pack.py` 产物 | `timing_qualified=true`、19/19 源哈希、`source_hashes_match_sim=true`、scope=OOC 5ns only | `sim/snn/evidence/nlp/final/ip/ip_manifest.json` |

IP 包不是平台工程：不含 Block Design、bitstream、XSA（见 `src/cnn/docs/PLATFORM_AND_STATUS.md`）。

## 6 G5 平台集成

| 检查项 | 判据 | 证据形态 |
|---|---|---|
| Block Design | 可打开/验证；IP 参数与 `rtl_manifest.capacities` 一致 | BD 工程、`*.hwh` |
| 全片实现 | 0 failing endpoints；bit 与 HWH 同一次实现（哈希配对） | `timing_summary.rpt`、`MANIFEST.sha256` |
| 平台限制 | 逐条列出（协议矩阵、软复位排空、CDC、平台专用性） | 平台 build 报告 JSON |
| 示例 | SNN r7：Setup +0.051 / Hold +0.011 / 0 failing、bit+HWH 已生成；CDC-10 与软复位排空列为已知限制 | `sim/snn/evidence/platform/platform_build_20261030.json` |

## 7 G6 板端

| 检查项 | 判据 | 证据形态 |
|---|---|---|
| 逐位比对 | 固定样本 N/N 一致、error_code=0 | 板端 run 日志 + 结果目录 |
| 身份配对 | bit/hwh/xsa 同次构建；驱动 ID/VERSION/CLOCK_HZ 与硬件一致 | `MANIFEST.json` 哈希、驱动输出 |
| 原始输出 | 板端 JSON/控制台日志随结论归档 | `results/<run>/` |
| 示例 | LNN 64/64 `bit_exact=True`、`final_state_match=True`、step_count=64；CNN 6 个 head 逐字节一致 | `board/lnn/README_BOARD.md`、`board/cnn/runtime/expected_heads.json` |

## 8 门链

```
G0 编译/导出 ──▶ G1 回放 ──▶ G2 OOC 综合 ──▶ G3 实现布线 ──▶ G4 IP 包 ──▶ G5 平台集成 ──▶ G6 板端
   产物齐         0 mismatch      0 failing          WNS/WHS≥0        component.xml     BD+bit 同源       逐位一致
   每级都要身份哈希；跨级不传递
```
