# 事实来源（PYNQ overlay）

| 结论 | 出处 |
|---|---|
| 三件套配对（XSA 内嵌 bit/hwh 逐字节比对） | `sim/lnn/evidence/ACCEPTANCE.md` §3；本机 zipfile+hashlib 复核 |
| 64 位 CSR 基址 `0x0480000000`、同 stem hwh 强制检查 | `board/lnn/lnn_cfc_run.py`（CSR_BASE、discover_paths） |
| 寄存器表（ID/VERSION/CONTROL/STATUS/ERROR_CODE/RUN_CYCLES/CLOCK_HZ） | `board/lnn/lnn_cfc_run.py` 的 CSR 常量表 |
| overlay 加载与 ip_dict 打印 | `board/cnn/scripts/board_run_a.py`（`load_overlay`）、`board/snn/snn_platform_run.py`（Overlay+MMIO） |
| 板端 6 head 逐字节比对与 `first_mismatch` | `board/cnn/README_BOARD.md`、`board/cnn/scripts/board_run_a.py` |
| 逐样本 match、sums 比对 | `board/snn/README_BOARD.md`、`board/snn/snn_platform_run.py` |
| 硬件实测与离线模型区分（`source=hardware/simulation-model`） | `board/lnn/lnn_cfc_run.py`（summary 字段）、`sim/lnn/evidence/offline_validation_summary.json` |
| hwh 元数据（DEVICE/PACKAGE/SPEEDGRADE、CLKFREQUENCY、MEMRANGE） | `board/lnn/lnn_cfc.hwh` 解析（命令见 C3） |
| BUSY 采样竞态与 r3 修复 | `[工作区] D:/dl_v1.0/docs/F0_BUSY_R2_ROOT_CAUSE_R3_FIX_20261017.md` |
| 行为仿真通过但综合后错误码 | `[工作区] D:/dl_v1.0/docs/PYNQ_ERR_OPCODE_ROOT_CAUSE_FIX_20261017.md` |
| u8→s8 输入重编码 | `board/cnn/MANIFEST.json` 的 `input_recode_fix`；`board/cnn/runtime/decode/preprocess.py` |
| 板端离线尝试日志 | `sim/snn/evidence/platform/board/board_run_local_attempt.log` |
| 性能口径（cycles vs 端到端） | `board/lnn/lnn_cfc_run.py`（RUN_CYCLES、wall time）、`report/design_report` 相关段落 |

## 核验边界

- 板端命令按随包驱动（\`board/*/\`）与既有证据整理，可直接在 PYNQ 板卡上复跑；
- 通用性声明（换板卡/换 IP 复用）为设计目标与验收清单，换板时按 \`reference/acceptance.md\` §3 逐项过一遍。
