# 命令矩阵 — lnn-export

> 所有命令按"可直接复制"给出；参数取自真实 `--help` / argparse 源码 / 实际运行，未臆造。
> 标记：**已实测**＝2026-11-02 在本机（Windows，Python 3.9.23）真实执行；**未复跑**＝缺 Vivado/PYNQ 或需板卡，只给依赖与期望。
> 路径约定：`<包根>` = `D:\FINAL`（git-bash 写作 `/d/FINAL`）；`<工作区>` = `D:\DL_FINAL\LNN`（不随包）。

## 0 环境与工作目录

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 0.1 | `python --version` | 已实测 | 0 / `Python 3.9.23` |
| 0.2 | `python -c "import numpy;print(numpy.__version__)"` | 已实测 | 0 / `2.0.2` |
| 0.3 | `cd <工作区> && python scripts/env_check.py` | 已实测 | 0 / 含 `bitfile : present`、`export : present` |
| 0.4 | `cd <工作区> && python scripts/env_check.py --check-hashes` | 已实测 | 0 / `hashes : 108 files checked, 0 bad` |
| 0.5 | `cd <包根>/build/lnn && python scripts/env_check.py` | 已实测 | 0 / **`bitfile : MISSING`、`export : MISSING`**（脚本按 `__file__/../..` 解析根，包内该布局下必现；不是故障，见 troubleshooting T10） |

## 1 导出链（仅工作区；包内不含生成器）

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 1.1 | `cd <工作区> && python user/export_demo.py --help` | 已实测 | 0 / `usage: export_demo.py [-h] [--out-dir OUT_DIR]`，仅 2 个选项 |
| 1.2 | `cd <工作区> && python -B user/export_demo.py --out-dir user/outputs/video01` | 已实测（video00 实跑） | 0 / 7 行 `SAME ...` + `RESULT: PASS  (7/7 files byte-identical)`；`generator_wall_s≈0.585`、`generator_exit=0` |
| 1.3 | `cd <工作区> && python -B user/export_demo.py` | 已实测（参数默认值） | 默认 `--out-dir user/outputs/demo01`；目录非空时退出码 **2** + `[ERROR] refusing to overwrite non-empty out-dir` |
| 1.4 | `cd <out>/export_run && python -B model/generate_export.py --weights model/model_weights.npz --tag demo --no-plot` | 未复跑（导出器内部调用） | 0 / 打印 `{tag,n_files:37,float_task_mse_skip5,...,manifest}` JSON；`--bootstrap` 与 `--weights` 互斥，二者必须给一个 |
| 1.5 | `cd <工作区> && python -B model/generate_export.py --weights model/model_weights.npz --tag trained` | 未复跑（需工作区 model/ 完整；会改写 export/） | 0 / 生成 7 个文件 + `manifest.json`（**会改变 manifest 哈希，勿对交付包执行**） |

## 2 期望值 / 清单核对（只读，包内可跑）

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 2.1 | `cd <包根> && ls data/lnn/export/gen \| wc -l` | 已实测 | 0 / `23` |
| 2.2 | `cd <包根> && diff -rq data/lnn/export board/lnn/export` | 已实测 | 0 / 无输出（两处期望值同源） |
| 2.3 | `cd <包根>/data/lnn/export && python -c "import json;a=json.load(open('manifest.json',encoding='utf-8'))['acceptance'];print(a['float_task_mse_skip5']['value'],a['int_vs_float_mse_skip5']['value'],a['onnx_max_abs_err'])"` | 已实测 | 0 / `1.234134383167709e-05 0.00011718563615310465 4.76837158203125e-07` |
| 2.4 | `cd <包根>/board/lnn && python cfc_sim_model.py --selfcheck <包根>/data/lnn/export` | 已实测 | 0 / 14 行 `OK` + `== cfc_sim_model selfcheck: PASS ==` |
| 2.5 | `cd <包根>/board/lnn && python lnn_cfc_run.py --list-samples` | 已实测 | 0 / `export dir: <包根>\board\lnn\export` + `sample: sample_000`…`sample_007` |
| 2.6 | `cd <包根>/board/lnn && python cfc_sim_model.py --make-fixture <目录>` | 未复跑（写盘，仅流程测试） | 0 / 生成契约合规的假 export 树；板端需 `--allow-fixture` 才接受 |
| 2.7 | `cd <包根>/data/lnn/export && python -c "import json;d=json.load(open('manifest.json',encoding='utf-8'));print(d['contract_version'],len(d['files']))"` | 已实测（等价） | 0 / `1.1 40`（其中 7 条包内缺失：2 条 `__pycache__` + 5 条 `model/`；可校验 33 条） |

## 3 三件套配对与身份（只读，包内可跑）

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 3.1 | `cd <包根>/board/lnn && sha256sum lnn_cfc.bit lnn_cfc.hwh lnn_cfc.xsa` | 已实测 | 0 / 三个哈希见下表 |
| 3.2 | `cd <包根>/board/lnn && sha256sum -c SHA256SUMS` | 已实测 | 0 / 42 行全 `OK` |
| 3.3 | `cd <包根>/board/lnn && python -c "import hashlib,zipfile;z=zipfile.ZipFile('lnn_cfc.xsa');print(hashlib.sha256(z.read('lnn_cfc.bit')).hexdigest());print(hashlib.sha256(z.read('design_1.hwh')).hexdigest())"` | 已实测 | 0 / 输出必须分别等于 3.1 的 bit 与 hwh 哈希 |
| 3.4 | `cd <包根>/board/lnn && grep -o 'DEVICE="[^"]*" PACKAGE="[^"]*" SPEEDGRADE="[^"]*"' lnn_cfc.hwh` | 已实测 | 0 / `DEVICE="xczu15eg" PACKAGE="ffvb1156" SPEEDGRADE="-2"` |
| 3.5 | `cd <包根>/board/lnn && grep -o 'CLKFREQUENCY="300000000" DIR="O" NAME="npu_clk"' lnn_cfc.hwh` | 已实测 | 0 / 命中 1 行 |
| 3.6 | `cd <包根>/board/lnn && grep -o 'BASEVALUE="0x480000000"[^/]*' lnn_cfc.hwh` | 已实测 | 0 / 行内含 `INSTANCE="M00_AXI_0"`、`HIGHVALUE="0x480000FFF"`（4 KiB） |
| 3.7 | `cd <包根>/src/lnn/rtl && sha256sum *.sv` | 已实测 | 0 / 8 行；前 8 位须与 `reference/artifacts.md` §5 冻结值一致 |

三件套真实哈希（2026-11-02 实算，与 `ACCEPTANCE.md` §3 一致）：

```text
0fcd8d65d0f8a101d795270741ca7c0cef21f1904590de78226a094b74be30a9 *lnn_cfc.bit    (28,700,861 B)
fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb *lnn_cfc.hwh    (380,358 B)
f01d6a8af07fc74acae3dbc30e362db333151978064d3584fcdfc611dbc27032 *lnn_cfc.xsa    (2,203,972 B)
```

## 4 RTL 仿真（需 Vivado 2021.1）

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 4.1 | `call E:\Vitis\Vivado\2021.1\settings64.bat` | 未复跑 | 0 / `xvlog/xelab/xsim` 进入 PATH |
| 4.2 | `xvlog -version` | 未复跑 | 0 / 版本行；本机未 source 时 `command not found` |
| 4.3 | `cd <工作区> && vivado -mode batch -notrace -source scripts/run_sim.tcl` | 未复跑 | 0 / 末行 `CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol` |
| 4.4 | `cd <包根>/build/lnn && tclsh scripts/run_sim.tcl` | 已实测（Tcl 前置检查路径） | **1** / `CFC_SIM_FAIL missing export/gen/cfc_weight_init.svh`（包内脚本沿用快照根布局，见 T9） |
| 4.5 | `cd <工作区> && tclsh scripts/run_sim.tcl` | 已实测（Tcl 前置检查路径） | **1** / `CFC_SIM_FAIL xsim tools not found (source settings64.bat first)`（未 source settings，见 T10） |
| 4.6 | `grep -c CFC_TB_PASS <工作区>/verification/logs/sim_*.log` | 未复跑 | 每个 TB 日志各 1 行 `CFC_TB_PASS`；出现 `CFC_TB_FAIL` 即失败 |

脚本内部实际执行（摘自 `scripts/run_sim.tcl`，用于排错）：

```text
xvlog -sv -i ../export/gen -f filelist.f          # cwd = verification/
xelab -debug typical tb_cfc_cell     -s cfc_cell_sim
xelab -debug typical tb_cfc_seq      -s cfc_seq_sim
xelab -debug typical tb_cfc_protocol -s cfc_proto_sim
xsim cfc_cell_sim -runall / xsim cfc_seq_sim -runall / xsim cfc_proto_sim -runall
```

## 5 板端（需 PYNQ 板卡）

| # | 命令 | 状态 | 退出码 / 期望输出 |
|---|---|---|---|
| 5.1 | `scp -r <包根>/board xilinx@<板卡IP>:/home/xilinx/jupyter_notebooks/board` | 未复跑 | — / 整个 `board/` 目录（含 `export/`） |
| 5.2 | `cd /home/xilinx/jupyter_notebooks/board && sha256sum -c SHA256SUMS` | 未复跑 | 0 / 42/42 `OK` |
| 5.3 | `rm -rf export/__pycache__` | 未复跑（必须） | — / 清掉 pyc，避免 manifest `mismatch`（T1） |
| 5.4 | `python3 -B lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --sample sample_000 --out-dir results/s0_hw --checks` | 未复跑 | 0 / `source=hardware`、`outputs matched 64/64 bit_exact=True final_state_match=True`、`step_count=64`、`error_code=0`、`RESULT: PASS` |
| 5.5 | `python3 lnn_cfc_run.py --list-samples --expect-dir export` | 未复跑 | 0 / 8 个样本 id |
| 5.6 | `python3 -B lnn_cfc_run.py --expect-dir export --sample sample_003 --out-dir results/s3` | 未复跑 | 0 / 换样本；期望值取 `export/gen/sample_003.json` |
| 5.7 | `python3 -B lnn_cfc_run.py --expect-dir export --mode clear-each-step --out-dir results/s0_ce` | 未复跑 | 0 / 每步 `h_old=0`；任务 MSE 明显变差（正常 ≈1e-4，清状态 ≈0.5） |
| 5.8 | `python3 -B lnn_cfc_run.py --expect-dir export --n-steps 32 --out-dir results/seg1` | 未复跑 | 0 / 分段实验用，`N_STEPS` 合法 1..64 |
| 5.9 | `cd <包根>/board/lnn && python -B lnn_cfc_run.py --backend simulation-model --sample sample_000 --out-dir <临时目录> --checks` | **已实测** | 0 / `source=simulation-model`、`hardware_measurement=False`、`same_stem=True`、`manifest files: total=33 verified=33 mismatch=0 missing=7`、`outputs matched 64/64 bit_exact=True final_state_match=True`、`step_count=64`、`run_cycles=2910`（模型假设）、`checks RESULT: PASS`；**不得当板端证据** |

板端驱动完整选项（`--help` 实测，2026-11-02）：

```text
usage: lnn_cfc_run.py [-h] [--bit BIT] [--hwh HWH] [--xsa XSA] [--expect-dir EXPECT_DIR]
                      [--sample SAMPLE] [--mode {clear-each-step,clear-per-step-only,normal,retain}]
                      [--n-steps N_STEPS] [--timeout TIMEOUT] [--out-dir OUT_DIR]
                      [--backend {auto,hardware,simulation-model}] [--allow-clock-mismatch]
                      [--allow-fixture] [--checks] [--list-samples] [--quiet]
```

模式位（驱动 `MODE_MAP` 源码）：`normal=0b01`（运行前清状态）、`retain=0b00`（保留上次末状态）、
`clear-each-step=0b11`（运行前清 + 每步 `h_old=0`）、`clear-per-step-only=0b10`。

## 6 退出码与判据速查

| 命令 | 成功 | 失败特征 |
|---|---|---|
| `user/export_demo.py` | 0 + `RESULT: PASS (7/7)` | 1：有 `DIFF` 或生成器非零；2：out-dir 非空 |
| `generate_export.py` | 0 + 末段 JSON | 非零 + `ValueError ... overflows ... range` / `exceeds fixpoint range limit` |
| `run_sim.tcl` | 0 + `CFC_SIM_PASS` | 1 + `CFC_SIM_FAIL <原因>` |
| `lnn_cfc_run.py` | 0 + `RESULT: PASS` | 非零 + `BOARD_ERROR: ...` / `CfcBoardError` |
| `cfc_sim_model.py --selfcheck` | 0 + `selfcheck: PASS` | 非零 + `FAIL` 行 |
| `sha256sum -c` | 0 + 全 `OK` | 非零 + `FAILED` 行 |

## 7 命令来源（可回查）

| 命令组 | 来源 |
|---|---|
| 1.1–1.3 | `[工作区] user/export_demo.py` 的 `argparse` 与 `main()`；`user/outputs/video00/summary.json` 为实跑记录 |
| 1.4–1.5 | `[工作区] user/gen/generate_export.py` 的 `argparse`（`--weights/--bootstrap/--tag/--no-plot`） |
| 2.x | `data/lnn/export/manifest.json`、`board/lnn/lnn_cfc_run.py`（`--list-samples` 分支）、`cfc_sim_model.py` 的 `--selfcheck/--make-fixture` |
| 3.x | `board/lnn/SHA256SUMS`、`lnn_cfc.xsa` 内 `lnn_cfc.bit`/`design_1.hwh`、`lnn_cfc.hwh` 的 `SYSTEMINFO/MEMRANGE` |
| 4.x | `[包根] build/lnn/scripts/run_sim.tcl` 与 `[工作区] scripts/run_sim.tcl` 原文 |
| 5.x | `[包根] src/lnn/docs/USER_GUIDE.md` §3、`board/lnn/README_BOARD.md` §3、驱动 `--help` 实测 |
