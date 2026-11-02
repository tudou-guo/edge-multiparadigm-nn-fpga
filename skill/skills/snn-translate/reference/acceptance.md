# snn-translate 分级验收

> 标签只允许 `pass / fail / partial / not_run`，另加限定字段 `record_level`（历史记录、本次未复跑）与
> `measured / derived`（实测/推导）。判定遵循 `../evidence-gate/SKILL.md`：标签 + 证据路径/字段 + 复算命令 + 身份哈希，缺一不算 pass。
> 每个 PASS 必须来自**本次**执行；引用包内已冻结证据时，要注明"证据是 <日期> 的冻结 run，本次未复跑"——按 `../evidence-gate` 只能记
> `pass（证据级）` 或 `record_level`，不得冒充本次实测。

## 0 判级总表

| 级 | 门 | 判据摘要 | 阈值为负时 |
|---|---|---|---|
| G0 | 编译/合同/数值 | exit 0 + blob 数 + 合同 108 项 + 单测 | `fail` |
| G1 | RTL 回放 | 四档 `pass=1`、mismatch=0 | `fail` |
| G2 | OOC 5 ns | WNS/WHS≥0、TNS/THS=0、0 failing、unrouted=0、latch=0、无例外 | 负 WNS 记 `fail` |
| G3 | 实现/布线（平台或 OOC routed） | 0 routing errors、fully routed=routable | `fail` |
| G4 | IP 源包 | `timing_qualified=true`、同源哈希 19/19、绑定 OOC 报告 | `fail` |
| G5 | 平台集成 | bit/HWH + 全片时序 + 平台限制清单 | 子项缺失 → `partial` |
| G6 | 板端 | 逐样本 match、all_match=true | 客方回传记录须注明来源 |

## 1 G0 编译（fixture / 真实权重 / 合同 / 单测）

| # | 检查 | 判定命令 | pass 条件 | 证据落点 |
|---|---|---|---|---|
| A1 | fixture 编译 | `reference/commands.md` C5 | exit 0；28 blobs；`manifest_sha256=835ad38dc06f0546bc08612bf052f085edec7f47b34115cfe5079c7b221381c4` | 自产 out-dir；`sim/snn/evidence/ACCEPTANCE_CORE.md` §A1 |
| A2 | 真实权重编译 | C6 | exit 0；**54 blobs**；`manifest_sha256=f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e`；manifest 文件 sha `b71e3f17…a6a4` | 同上；包内 `build/snn/generated/_quickstart_real/` |
| A3 | 确定性 | 同 seed/同权重换 out-dir 再跑 A1/A2 | `manifest_sha256` 与 params sha256 集合不变（fixture 实测三个时点一致） | `ACCEPTANCE_CORE.md` §A1/A2 |
| A4 | RTL 源哈希 | C7 `rtl mismatch` | `rtl mismatch: 0`（28/28） | manifest `rtl_sha256` |
| A5 | 合同一致 | C13 | `status=pass`、`fails=[]`、108 checks | 自产 JSON；`sim/snn/evidence/nlp/contract.json` |
| A6 | 纯 Python 单测 | C12 | 15 项 PASS，`FAILURES: 0` | `sim/snn/evidence/nlp/math_tests_m1.log`（8/8） |
| A7 | 整数参考保真 | C14（[工作区]） | `RESULT: 8/8 PASS -> PASS`；47 层 trace 的 `first_mismatch=null` | `sim/snn/evidence/nlp/math_tests_m1.json`、`train_realweight_fidelity_real50.json` |

**G0 通过条件**：A1+A2+A4+A5 全 pass；A6/A7 需 torch/runtime 环境。
**常见 fail**：配置越界（§3 报错原文）、checkpoint dtype 不符、out-dir 已存在、fixture 与 weights 同时给。

## 2 G1 回放（冻结 RTL，mismatch=0）

| # | 用例 | 判定字段 | pass 条件 | 证据 |
|---|---|---|---|---|
| B1 | small | `pass`、`cycles`、`steps[2].result_line`、`steps[2].fail_lines` | `pass=True`、`cycles=39774`、`fail_lines=0` | `sim/snn/evidence/nlp/replay_small_holdfix.json` |
| B2 | e2e +NLP_NO_FORCE | 同上 | `pass=True`、`cycles=7084772`、`fail_lines=0` | `replay_e2e_noforce_holdfix.json` |
| B3 | e2e 反压（bpforce） | 同上 | `pass=True`、`cycles=7084812`、`fail_lines=0`（14 输出/10 输入 stall 窗口） | `replay_e2e_bpforce_holdfix.json` |
| B4 | real 3 样本 | 同上 | `pass=True`、`cycles=10627157`、`fail_lines=0` | `replay_real_holdfix.json` |

判定公式（可脚本化）：

```bash
python - <<'PY'
import json,glob
for p in sorted(glob.glob('sim/snn/evidence/nlp/replay_*_holdfix.json')):
    d=json.load(open(p,encoding='utf-8-sig'))
    s=[x for x in d['steps'] if x.get('step')=='vsim'][0]
    ok=d['pass'] is True and d['result']=='pass' and s['exit']==0 and s['fail_lines']==0
    print(('PASS' if ok else 'FAIL'), p.split('/')[-1], d['cycles'], s['result_line'])
PY
```

期望 4 行全 `PASS`，cycles 39774 / 7084772 / 7084812 / 10627157。
"mismatch=0" 的显式文字记录在 `sim/snn/evidence/nlp/final_gates.json`（target_e2e/real_weight_replay 的 `detail`）与
`rtl_freeze.json.reason`；回放 JSON 本身用 `pass/result/fail_lines` 表达。
**注意**：回放命令需 ModelSim + 主线向量目录；判定以 2026-10-30 冻结 run 的证据为准，并注明证据日期。

## 3 G2 OOC 5 ns（同源 RTL + 生成 adapter）

判定命令 C20；pass 条件（全部满足）：

| 字段 | 阈值 | 冻结证据值 |
|---|---|---|
| `status` | `pass` | `pass` |
| `wns_setup_ns` | ≥ 0（负值 fail） | `0.135` |
| `wns_hold_ns` | ≥ 0（负值 fail） | `0.023` |
| `unrouted_nets` | `0` | `0` |
| `latch_cells` | `0` | `0` |
| `facts.wns_*_ns` / `facts.unrouted_nets` | 与解析字段一致 | 一致 |
| `target` | `xczu15eg-ffvb1156-2-i` / Vivado 2021.1 / `period_ns=5` | 一致 |
| 时序例外 | 无（`exceptions.rpt` 0 例外） | 0 |
| pulse width | 0 failing | 0 failing |

证据：`sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json`（含 `utilization`：LUT 40,964 / FF 71,173 / BRAM 186.5 / DSP 128，
`measured`）、`ooc_run_20261030_120804/{timing_route.rpt,route_status.rpt,pulse_width.rpt,clocks.rpt,exceptions.rpt}`
（原始报告未随包，只在 [工作区] 主线；包内保留摘要 JSON 与 `hold_closure_20261030.json`）。
**OOC 不是集成/板级签核**：0 输入延迟模型、`HD.CLK_SRC` 未设、WNS 束缚路径是 hold-margin 链（上游预算 +0.135 ns）。

## 4 G3 实现/布线

| 对象 | pass 条件 | 证据 |
|---|---|---|
| OOC routed | `route_status.rpt`：0 routing errors、unrouted 0、fully routed=routable | 同上 `ooc_*/route_status.rpt` |
| 平台 r7 全片 | 0 routing errors、fully routed 180,101/180,101、WNS +0.051/0 failing（334,330）、WHS +0.011/0 failing（333,740）、WPWS +0.039 | `build/snn/reports/platform_timing_summary_20261031.txt`、`r7_sumsfix_route_status.rpt` |
| 平台 r6 对照 | WNS +0.245 / WHS +0.010 / 0 failing | `build/snn/reports/CDC_FIX_R6_RESULT_20261030.md`、`r6_final_route_status.rpt` |

## 5 G4 IP 源包

判定命令 C22；pass 条件：

- `timing_qualified=true`；
- `source_hashes_match_sim=true`、`source_mismatch=[]`，19/19 源文件（18 冻结 RTL + `include/spikformer_cfg.vh`）；
- `package_sha256=894599b92c8de7d4edf65320db5f23a1413e7e9fabc81740b357e2134b292f22`（冻结证据）；
- 绑定同一身份的 OOC 报告（`ooc_report.json` 与 `reports/ooc_20261030_120804.json` sha 相同 `94f0e125…`）；
- `assumptions` 至少声明：OOC 5 ns only、0 输入延迟、保留 1192 DONT_TOUCH cells、不是集成签核；平台/bit/board 另需独立验证。

## 6 G5 平台集成（判据与已知限制）

| 检查 | 判定命令 / 字段 | 证据 |
|---|---|---|
| 平台搭建 | BD validate/generate；CSR/FSM smoke 7/7 | `build/snn/reports/platform_build_20261030.json`（`status.modelsim_compile=pass`、`csr_fsm_smoke="pass (7/7)"`） |
| 全片实现 + bit/HWH | 0 failing endpoints；bit/HWH 同源 | 同 §4；`board/snn/snn_platform.bit`、`board/snn/snn_platform.hwh` + `MANIFEST.sha256`（20 文件） |
| 完整协议仿真矩阵 | 目前只有 CSR/超时 smoke | 补齐协议矩阵前不得写平台 pass |
| 软复位排空在途 DMA | 连续第二样本超时 | 现靠每样本重载 overlay 绕过 |
| CDC 1 Critical（CDC-10） | 记录为已知限制 | 不得写"CDC 全清" |
| 平台专用性 | r7 为 NLP 专用（wrapper + 冻结 NLP 核 + adapter） | `src/snn/docs/PLATFORM_AND_STATUS.md` §2 |

**判级**：G5 需逐子项列出（通过项 + 待补项），不得整包写 `pass`。
**薄余量提醒**：r7 的 WHS +0.011 ns 与 r6 基线 +0.010 ns 同级，不构成「安全余量」；任何 RTL/权重/约束改动后必须重跑实现并重新配对 bit/HWH。

## 7 G6 板端（判据）

| 检查 | 判定 | 证据 / 说明 |
|---|---|---|
| 板包完整性 | `sha256sum -c MANIFEST.sha256`（20 文件）+ DEMO 清单（5 条：4 demo 资产 + README_SNAPSHOT） | 快照复制时 20/20 通过；4 demo 文件见 `MANIFEST_DEMO_20261031.sha256` |
| 逐样本比对 | `snn_platform_run.py ... --samples 3` → 每行 `match=True`、`all_match=true` | 3/3 match（客方回传记录，见 `README_BOARD.md` §3） |
| 原始输出 | 板端 run JSON / 控制台日志 | 须与结论一并归档；本机缺 `pynq` 时只能做离线检查（`sim/snn/evidence/platform/board/board_run_local_attempt.log`） |
| 板卡 | 现场复跑 | 板卡 `192.168.2.99` 需在线 |
| sums 通道序 | r7 bit 必须 `ROT=64` 读 | 不加旋转会 64 通道错位，比对必失败 |

**判级**：G6 结论以客方回传记录为限，须注明"记录级、原始输出未随包"，不得写"板测 pass"。

## 8 最终结论模板（照抄改数字）

```text
G0 编译：pass（fixture 28 blob / real 54 blob；manifest_sha256 见上；注明本次/证据级）
G1 回放：pass（2026-10-30 冻结 RTL；4 档 cycles=39774/7084772/7084812/10627157，fail_lines=0）
G2 OOC ：pass（WNS +0.135 / WHS +0.023 / TNS 0 / 0 failing / unrouted 0 / latch 0）
G3 实现：pass（OOC routed 0 routing errors）；平台 r7（WNS +0.051 / WHS +0.011 / 0 failing）
G4 IP  ：pass（timing_qualified=true，19/19，package_sha256=894599b9…292f22）
G5 平台：BD/compile/CSR smoke 7/7 + r7 时序通过；协议矩阵、软复位排空、CDC-10 待收敛
G6 板端：3/3 match（客方回传记录，注明来源与日期）
未闭合：<逐条列>
```

## 9 一票否决（出现即不能判 pass）

1. 负 WNS/WHS 仍写 pass；2. 为过门降频或加 `set_false_path`/multicycle；
3. 用 fixture 结果声称精度或交付；4. 把 OOC 说成集成/板级；5. 用历史 run 冒充本次；
6. 板端只有记录却写 pass；7. 图像线结果混入 NLP 验收；8. 冻结 RTL 改动后仍引用旧回放/OOC 证据。
