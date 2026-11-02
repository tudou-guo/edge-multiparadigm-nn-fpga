# QUICKSTART — 3 条命令跑通快照

前置：Windows + Python 3.9（本机 `E:/miniconda/python.exe`，已装 numpy；`torch` 仅 trainer 与 integer_ref 短测需要）。
不需要 Vivado、不需要板卡。全部命令在**快照根**执行；换根后路径自动相对解析（可用环境变量 `SNN_ROOT` 覆盖）。

```bat
cd /d/DL_FINAL/SNN
```

## 1. 环境检查

```bat
E:/miniconda/python.exe -c "import sys, numpy; print(sys.version.split()[0], 'numpy', numpy.__version__)"
E:/miniconda/python.exe -c "import torch; print('torch', torch.__version__)"
```

期望：`3.9.x numpy 2.0.2`；`torch 2.8.0+cpu`（没有 torch 也能跑第 2 步与 test_short，跳过 test_integer_ref）。

## 2. 编译 NLP（fixture 权重，无需 checkpoint）

```bat
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --fixture-seed 1 --out-dir build/_quickstart_check
```

期望输出：

```text
manifest: build/_quickstart_check/manifest.json kind=development_fixture
```

产物（约 5.5 MB）：

| 文件 | 说明 |
|---|---|
| `manifest.json` | 绑定配置/结构/调度/params/28 个 RTL 源 sha256/生成物/数值合同 |
| `numeric_contract.json` `resolved_config.json` | 数值合同与展开后的配置 |
| `params/*.bin`（28） | 每块 q/k/v/proj/fc1/fc2 的 w/b（+ `.coe` 供 IP） |
| `generated/spikformer_config.vh` | RTL 配置头 |
| `generated/spikformer_nlp_target_top.sv` | 目标顶层适配器（含 hold-margin 单元） |
| `generated/spikformer.xdc` `spikformer_ooc.tcl` `spikformer_project.tcl` `sources.f` | 约束/OOC 与工程 Tcl/源清单（路径指向本快照根） |

可选：真实权重编译（checkpoint 由 docs 子代理放入 `examples/nlp_sentiment_b1/`；不存在则跳过）：

```bat
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz ^
  --out-dir build/_quickstart_real
```

期望：`kind=real_corpus`，54 个 params blob（含 embed/pos 表、全部阈值/scale、FP32 head）。
本快照 2026-10-31 16:57 实测、17:38 最终刷新后复跑：**pass**，且 params 哈希集合与已闭环案例的最终冻结 run
（主线 `final/nlp_real_5stage_r4_holdfix/params`）**字节级一致**（见 `evidence/ACCEPTANCE_CORE.md` A1b）。

## 3. 纯 Python 数值自检

```bat
E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py
E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py
```

期望：

```text
FAILURES: 0                      # test_short.py：15 项 PASS
RESULT: 8/8 PASS -> PASS         # test_integer_ref.py（需要 torch）
```

## 4. 板端演示（不在本机执行）

见 `board/README_BOARD.md`：把 `board/` 拷到 ZU15EG（PYNQ 镜像）后

```bash
cd /home/xilinx/snn_board
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real --samples 3 --json-out snn_board_result.json
```

板卡 192.168.2.99 当前离线；本快照**未跑板端**，原始板端输出**未随包**（见 `RELEASE_NOTES.md` §3）。

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: tools` | 命令必须在**快照根**执行（`cd D:/DL_FINAL/SNN`），不要到 `tools/` 里跑 |
| `FileNotFoundError ... rtl/ops/spiking/source_manifest.json` | 用 `SNN_ROOT` 指定快照根，或确认命令 cwd 是快照根 |
| 编译后在别的机器复用 `generated/*.tcl` | Tcl 里记录生成时的绝对根；在新根重新 compile 即可（不要手改 Tcl） |
