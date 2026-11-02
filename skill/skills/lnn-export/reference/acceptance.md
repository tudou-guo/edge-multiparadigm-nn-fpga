# 分级验收清单 — lnn-export

> 门级定义与标签规则来自 `../evidence-gate/reference/gates.md`（G0–G6）。本文件只列本技能相关的 G0/G1/G3/G6。
> 铁律：**跨级不传递**；历史数据须注明来源与日期；离线模型（`simulation-model`）不能当板端证据。


## 1 G0 导出（7/7 逐字节）

| 项 | 内容 |
|---|---|
| 判定命令 | `cd D:/DL_FINAL/LNN && python -B user/export_demo.py --out-dir user/outputs/<新目录>` |
| 通过阈值 | 输出含 `RESULT: PASS  (7/7 files byte-identical)`；退出码 **0**；`<out>/summary.json` 的 `all_same=true` 且 `generator_exit=0` |
| 证据落点 | `<out>/summary.json`（7 条 `same=true` + `bytes`）、`<out>/export.log`（生成器完整输出） |
| 不能证明 | 数值正确性、RTL 一致、时序、板端 |
| 失败处置 | 退出码 2 → 换新 `--out-dir`；出现 `DIFF` → 见 troubleshooting T8 |

## 2 G1 数值与回放（三个子项，需同时满足）

### 2.1 G1a ONNX 等价

| 项 | 内容 |
|---|---|
| 判定命令 | `cd <包根>/data/lnn/export && python -c "import json;d=json.load(open('onnx_verify.json',encoding='utf-8'));print(d['pytorch_vs_onnx_random']['max_abs_err'],d['graph']['operators']['all_allowed'])"` |
| 通过阈值 | `max_abs_err ≤ 1e-5`（实测 `4.768372e-07`）；`all_allowed=True`；`opset=17`、`ir_version=8` |
| 证据落点 | `data/lnn/export/onnx_verify.json`（256 组随机 + 8×64 序列；序列级 max abs 同为 4.768372e-07） |
| 不能证明 | 定点误差、RTL 行为 |

### 2.2 G1b 定点参考

| 项 | 内容 |
|---|---|
| 判定命令 | `cd <包根>/board/lnn && python cfc_sim_model.py --selfcheck <包根>/data/lnn/export` |
| 通过阈值 | 末行 `== cfc_sim_model selfcheck: PASS ==`；14 项 `OK`（含 `gate_pre includes W_b branch (contract V1.1)`、`lut_index(-32768)==0`、`lut_index(32767)==1024`）；退出码 0 |
| 补充阈值 | `manifest.acceptance`：浮点任务 MSE `1.234134e-05 ≤ 0.005`；定点 vs 浮点 MSE `1.171856e-04 ≤ 0.001`；max abs `0.0476`；饱和计数 0、LUT 夹紧 0 |
| 证据落点 | 命令输出；`data/lnn/export/manifest.json` 的 `acceptance`/`metrics` |
| 不能证明 | RTL 实现是否同语义（要看 G1c） |

### 2.3 G1c RTL 逐位回放

| 项 | 内容 |
|---|---|
| 判定命令 | `call E:\Vitis\Vivado\2021.1\settings64.bat` → `cd D:/DL_FINAL/LNN && vivado -mode batch -notrace -source scripts/run_sim.tcl` |
| 通过阈值 | 末行 `CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol`；三个 TB 各出现 `CFC_TB_PASS` 且无 `CFC_TB_FAIL`；mismatch=0；`RUN_CYCLES=9088`、`STEP_COUNT=64` |
| 分项数字 | `tb_cfc_cell` 21 用例 / 0 mismatch / 平均 139 周期每步；`tb_cfc_seq` 626 checks / 0 mismatch；`tb_cfc_protocol` 155 checks / 0 mismatch（errors=0） |
| 证据落点 | `verification/logs/{compile,elab_*,sim_*}.log`；包内摘要 `sim/lnn/evidence/ACCEPTANCE.md` §1 |
| 不能证明 | 可综合/可布线（G2/G3）、板端 |

## 3 G3 实现（时序/布线/DRC）

| 项 | 内容 |
|---|---|
| 判定命令 | 读 `sim/lnn/evidence/ACCEPTANCE.md` §2；或在有 Vivado 工程时读 `reports/vivado_final_*.rpt` [工作区] 与 `reports/vivado_build_metrics.txt` |
| 通过阈值 | setup WNS `≥0`（实测 **+0.226 ns**）、hold WHS `≥0`（**+0.011 ns**）、WPWS `+0.039 ns`；失败端点 **0**；routable`=`fully routed **57464/57464**；routing errors **0**；DRC **0 Error / 0 Critical**（4 Warning 来自平台既有 IP） |
| 资源（记录） | CLB LUT 25,221（7.39%）、FF 32,890（4.82%）、DSP48E2 4、BRAM Tile 28.5、URAM 0、CARRY8 136 |
| 身份 | 三件套哈希 + XSA 内嵌比对见 `reference/commands.md` §3；RTL 8 个冻结哈希见 `reference/artifacts.md` §5 |
| 不能证明 | 板端功能正确 |
| 注意 | hold 余量仅 +0.011 ns（与平台 r6 基线 +0.010 同级）；任何 RTL/约束改动必须重跑实现，旧 WNS 立即失效 |

## 4 G6 板端 64 步

| 项 | 内容 |
|---|---|
| 判定命令 | 板上：`rm -rf export/__pycache__` → `python3 -B lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --sample sample_000 --out-dir results/s0_hw --checks` |
| 通过阈值（全部满足） | ① `source=hardware`、`is_hardware_measurement=true`；② `ID=0x43464331`、`VERSION=0x00010000`、`CLOCK_HZ=300000000` 三项 `ok=True`；③ `outputs matched 64/64`、`bit_exact=True`、`mismatch_indices=[]`；④ `final_state_match=True`、`step_count=64`、`error_code=0`、`status=done`；⑤ `--checks` 的 `repeat / segments / clear-each-step` 三项 PASS |
| 证据落点 | 板上 `results/s0_hw/{summary,outputs,compare,state_checks}.json` + `outputs.npy`；建议归档 `evidence/board_run/` |
| 不可替代品 | `sim/lnn/evidence/offline_validation_summary.json`：`source=simulation-model`、`is_hardware_measurement=false`、29/29 PASS——**只能标 partial**，证明驱动寄存器序列/位打包/比对逻辑自洽，不能写进板端验收 |
| 失败处置 | 见 troubleshooting T1–T6（pycache/缺 export/stem 不符/不同源/错误码/超时） |

## 5 复跑顺序（从零到板端）

1. `python scripts/env_check.py --check-hashes`（工作区）→ 环境与 108 文件哈希；
2. `python -B user/export_demo.py --out-dir <新目录>` → G0 7/7；
3. 读 `onnx_verify.json` + `manifest.json` → G1a/G1b；
4. `vivado -mode batch -notrace -source scripts/run_sim.tcl` → G1c（需 settings64）；
5. `cd board/lnn && sha256sum -c SHA256SUMS` + XSA 内嵌比对 → G3 身份；
6. 板上 `python3 -B lnn_cfc_run.py ... --checks` → G6；
7. 每级把标签 + 证据路径 + 复算命令写入记录（格式见 `../evidence-gate/templates/acceptance_record.md`）。

## 6 明确不可从本技能得到

| 结论 | 原因 |
|---|---|
| 板端实测通过 | 需真实 ZU15EG + PYNQ；包内只有离线模型 |
| 位流可重建 | Vivado 工程、平台 BD、构建脚本不随包 |
| 时序/资源"本次"实测 | 只有开发树报告记录值，须注明来源与日期 |
| 新网络规模/新 Δt 可跑 | 结构固定 1→4→1、Δt=655、n_steps≤64 |
| 加速比 / 端到端性能承诺 | 只有 `RUN_CYCLES=9088`（片上计算）；端到端含 PS MMIO，首版不承诺倍数 |
