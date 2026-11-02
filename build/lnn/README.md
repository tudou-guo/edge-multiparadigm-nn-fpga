# build/lnn

LNN 库的构建入口与验收报告。

| 路径 | 内容 |
|---|---|
| `scripts/env_check.py` | 环境自检 |
| `scripts/run_sim.tcl` | ModelSim 回放入口（RTL 逐位比对） |
| `reports/ACCEPTANCE.md` | 验收记录：WNS +0.226 / WHS +0.011、0 失败端点、DRC 0 Error、三件套哈希配对 |
| `reports/offline_validation_summary.json` | 板端离线模型 29/29 校验 |

## 说明

LNN 的位流由平台工程派生构建（D:/spk_platform/project_1 派生的 LNN 工程），
构建产物（bit/hwh/xsa）在 `board/lnn/`；本目录保留入口脚本与验收证据，
不复制 Vivado 工程与临时目录。
