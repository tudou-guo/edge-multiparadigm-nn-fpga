# build/snn

SNN 库的构建清单与物理报告。

## 结构

| 路径 | 内容 |
|---|---|
| `generated/_quickstart_real/` | compile 产物：manifest、resolved_config、numeric_contract、schedule |
| `generated/_lead_final_check/` | 最终检查用的同一组产物 |
| `reports/platform_timing_summary_20261031.txt` | 平台 r7 全片时序（WNS +0.051 / WHS +0.011 / 0 失败） |
| `reports/r6_final_util.rpt` / `r6_final_route_status.rpt` | r6 资源与布线 |
| `reports/r7_sumsfix_route_status.rpt` | r7 布线状态（fully routed） |
| `reports/CDC_FIX_R6_RESULT_20261030.md` | r5→r6 CDC 修复记录（WNS −0.886 → +0.245） |
| `reports/final_gates.json` / `rtl_freeze.json` | 冻结 RTL 哈希与门禁结果 |

## 命令

`python -m tools.toolchain --backend spikformer compile --config cfg.json --weights ckpt.npz --out out/snn`
`python -m tools.toolchain --backend spikformer build --manifest out/snn/manifest.json --stage ooc --execute`
`python -m tools.toolchain --backend spikformer package --manifest out/snn/manifest.json --out out/snn/package`

> 物理门需 Vivado 2021.1 + 共享 heavy lock；快照内 RTL 已非 10-30 冻结态，
> 重跑不会逐位复现冻结 WNS/WHS，报告中已注明。
