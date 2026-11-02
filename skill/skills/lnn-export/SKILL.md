---
name: lnn-export
version: 2.0.1
description: 把 CfC 液态神经网络（1 输入 / 4 隐状态 / 1 输出，84 权重 + 17 偏置 = 101 参数）的训练权重导出为 int16 Q12 / int32 Q24 定点 ROM 初值与 1025 点激活 LUT，并验证 7/7 文件逐字节可复算、RTL 0 mismatch、板端 64/64
applies_to: [lnn]
package_paths: [src/lnn, data/lnn/export, sim/lnn, build/lnn, board/lnn]
requires: [Python>=3.9, numpy, (可选) Vivado 2021.1 含 xsim, (可选) PYNQ 2.7 + ZU15EG 板卡]
last_verified: 2026-11-02
---

# SKILL: lnn-export

## 0 一句话定义

把训练好的 CfC 浮点权重（`model_weights.npz`，10 组张量）按冻结数值合同量化成 **int16 Q12 数据 + int32 Q24 偏置 + 两张 1025 点激活 LUT**，生成 7 个可直接被 RTL `include` 的 ROM 初值文件，并用"重算逐字节比对 + 整数参考 + RTL 逐位仿真 + 板端 64 步"四级证据证明这条链可复算。产物为 `data/lnn/export/`（包内期望值）与 `board/lnn/`（位流三件套 + 驱动）。

## 1 何时使用 / 何时不要使用

**使用**：已有训练好的 CfC 权重要换成 RTL ROM 初值并验证导出确定性；复核"仓库 `export/` 是否由这份 `model_weights.npz` 生成"（7/7 逐字节）；核对定点格式（Q12/Q24/48 位累加/LUT 索引）与 RTL 是否同一套语义；在板上跑 64 步并与定点 golden 逐点比对（含三件套配对）。

**不要使用**：不要用它训练（`train_cfc.py`、`export_onnx.py` 不随包，只在 [工作区] `D:/DL_LNN/model/`）；不要用它改网络结构（固定 1→4→1、84+17 参数，换规模=重做 RTL）；不要用离线模型结果冒充板端实测；不要用"仿真 0 mismatch"推断时序或板端——G1/G3/G6 跨级不传递。

## 2 前置条件与环境自检

| 项 | 要求 | 自检命令 | 期望输出 |
|---|---|---|---|
| Python / numpy | ≥3.9；numpy 2.x | `python -c "import numpy;print(numpy.__version__)"` | `2.0.2` |
| 期望值目录 | `data/lnn/export/` 与 `board/lnn/export/` 一致 | `diff -rq data/lnn/export board/lnn/export` | 无输出（退出码 0） |
| 驱动可用 | 驱动 + `cfc_sim_model.py` 同目录 | `cd board/lnn && python lnn_cfc_run.py --list-samples` | `export dir: ...` + `sample_000`…`sample_007`，退出码 0 |
| 三件套完整 | bit/hwh/xsa 同目录同 stem | `cd board/lnn && sha256sum -c SHA256SUMS` | 42 行全 `OK`，退出码 0 |
| 仿真（可选） | Vivado 2021.1 的 `xvlog/xelab/xsim` 在 PATH | `xvlog -version` | 版本行；未 source settings 时报 `command not found` |
| 板卡（可选） | PYNQ 2.7 + `xczu15eg-ffvb1156-2-i` | 板上 `python3 -c "import pynq"` | 无异常 |

工作区自检（导出链需要生成器，生成器只在工作区）：`cd D:/DL_FINAL/LNN` 后跑 `python scripts/env_check.py`
（期望 `bitfile : present` / `export : present`）与 `python user/export_demo.py --help`（只有 `--out-dir` 一个参数）。

## 3 输入合同

### 3.1 权重与配置（导出器唯一数值输入）

| 文件 | 格式 | 硬约束 | 校验命令 | 不满足时 |
|---|---|---|---|---|
| `model_weights.npz` | numpy `.npz`，float32 | 键必须是 `w1 w2 wa wb wo b1 b2 ba bb bo`；`w1/w2/wa/wb` 形状 `(4,5)`，`wo` `(1,4)`，`b1/b2/ba/bb` `(4,)`，`bo` `(1,)` | `python -c "import numpy as np;d=np.load('model/model_weights.npz');print(list(d.keys()))"` | 缺键 → `KeyError`；形状不符 → 拼接/矩阵乘报错，导出中止 |
| `train_config.json` | JSON | 含 `task`/`model`/`quantization`；`delta_t_int=655`、`lut_size=1025`、`lut_min=-8.0`、`lut_max=8.0`、`data_frac_bits=12`、`bias_frac_bits=24`、`acc_bits=48`、`n_steps=64` | 逐字段比对（见 `reference/artifacts.md`） | 缺字段 → `KeyError`；数值与合同不符 → ROM 初值不可信，先改配置 |
| `model/onnx_model.onnx` | ONNX，opset 17 / IR 8 | 输入 `x[1,1] h[1,4] delta_t[1,1]`，输出 `y[1,1] h_next[1,4]`；15 节点；算子 ⊆ {Gemm,MatMul,Concat,Tanh,Sigmoid,Mul,Add,Sub} | 读 `data/lnn/export/onnx_verify.json` 的 `graph.inputs/outputs`、`graph.operators.all_allowed` | 动态 shape / 未允许算子 → 不能作数值规范，RTL 无法对齐 |

参数量恒等式：`4×(4×5) + 1×4 = 84` 权重，`4×4 + 1 = 17` 偏置，合计 **101 参数**。参数顺序冻结：
权重 `0..19=W1[o][i]`（`idx=o*5+i`）、`20..39=W2`、`40..59=Wa`、`60..79=Wb`、`80..83=Wo[i]`；
偏置 `0..3=b1`、`4..7=b2`、`8..11=ba`、`12..15=bb`、`16=bo[0]`；输入通道序 `[x,h0,h1,h2,h3]`。

### 3.2 数值合同（int16 Q12 / int32 Q24）

| 对象 | 格式 | 真值换算 / 规则 | 出处 |
|---|---|---|---|
| 输入 / 权重 / 隐状态 / 输出 | int16，Q12 | `int / 4096` | `manifest.json` `numeric.data` |
| 偏置 `b1 b2 ba bb bo` | int32，Q24 | `int / 16777216` | `numeric.bias` |
| 累加器 | signed 48 位精确，无中间截断 | 乘积 int32 精确 + 偏置 Q24 精确相加 | `numeric.accumulator` |
| 舍入 / 饱和 | 最近值、中点远离零；每次舍入后饱和到 int16 | `round_q12_sat16(v)`：`v>=0 ? (v+2048)>>12 : -(((-v)+2048)>>12)` | `numeric.rounding` |
| Δt | 整数 `655 = 0x028F = 0.159912109375`（浮点训练 0.16） | — | `numeric.delta_t_int` |
| 激活 | 1025 点 LUT，区间 `[-8,8]`，步长 `1/64` | `idx=clamp((v+32768+32)>>6,0,1024)`；`value=sat16(round_half_away(act((i-512)/64)*4096))` | `numeric`/`activation` |

生成器两道溢出护栏（触发即抛异常、不写文件）：权重 `|q|>32767` → `ValueError: weight ... overflows int16 Q12 range`；
偏置超 int32 → `bias ... overflows int32 Q24 range`；浮点 `|w|>8.0` 或 `|b|>127.99` → `exceeds fixpoint range limit`。
V1.1 门控必须含 `W_b` 分支：`gate_pre = ROUND_Q12_SAT16(lin_a*655 + (lin_b<<12))`。漏掉 `W_b` 会让 84 个权重中
索引 `60..79` 的 20 个不参与计算（V1.0 缺陷，必须重做全部 golden）。

## 4 执行步骤

### 步骤 1：核对包内期望值齐备（只读，可随时做）

```bash
cd <包根>
ls data/lnn/export/gen | wc -l          # 期望 23
for f in cfc_weight_init.svh cfc_lut_init.svh cfc_rom_params.svh x_input.hex y_expected.hex h_init.hex; do
  test -f data/lnn/export/gen/$f && echo "OK $f" || echo "MISS $f"
done
```

期望 6 行 `OK`、`gen/` 共 23 个文件（3 svh + 4 hex + 8×2 个 sample）。
失败分支：`MISS` → 期望值不完整，驱动会拒绝运行，先补齐 `export/`（§7 第 2 条）。

### 步骤 2：重算导出并逐字节对拍（G0）

只在**工作区**可做（包内不含生成器）：

```bash
cd D:/DL_FINAL/LNN
python -B user/export_demo.py --out-dir user/outputs/video01
```

机制：把 `user/gen/generate_export.py` 与 `model/` 输入复制到 `<out>/export_run/` 临时树，在临时树跑
`python -B model/generate_export.py --weights model/model_weights.npz --tag demo --no-plot`，
再把 7 个产物与仓库 `export/` 做 SHA256 逐字节比较，写 `<out>/summary.json`。

期望（末行 + 7 行 `SAME`；实测 0.585 s、exit 0）：

```text
SAME weights.hex                  84 x int16 Q12                 420 bytes
SAME biases.hex                   17 x int32 Q24                 153 bytes
SAME tanh_lut.hex                 1025 x int16 Q12               5125 bytes
SAME sigmoid_lut.hex              1025 x int16 Q12               5125 bytes
SAME gen/cfc_weight_init.svh      RTL weight/bias ROM init       2626 bytes
SAME gen/cfc_lut_init.svh         RTL activation ROM init        60464 bytes
SAME gen/cfc_rom_params.svh       N_W / N_B / N_LUT / DT params  115 bytes
RESULT: PASS  (7/7 files byte-identical)
```

失败分支：`[ERROR] refusing to overwrite non-empty out-dir: ...` + 退出码 **2** → 换新 `--out-dir`（默认
`user/outputs/demo01`，video00 已占用）；某行 `DIFF` → 权重/配置与仓库 `export/` 不同批次，看 `<out>/export.log`
的 `metrics` 段；生成器非零退出 → 脚本会打印 `[generator log tail]` 15 行。

### 步骤 3：核对单步 ONNX 等价性（G1 数值侧）

包内只有证据文件，导出/比对脚本不随包（[工作区] `D:/DL_LNN/model/export_onnx.py`）：

```bash
cd <包根>/data/lnn/export
python -c "import json;d=json.load(open('onnx_verify.json',encoding='utf-8'));print(d['opset'],d['ir_version'],d['graph']['operators']['all_allowed']);print(d['pytorch_vs_onnx_random']['max_abs_err'])"
```

期望：`opset 17 / IR 8 / all_allowed True` 与 `4.768372e-07`（阈值 `1e-5`；`y` 4.768e-07、`h_next` 2.384e-07，256 组随机用例）。
失败分支：`all_allowed=False` → 图含未允许算子；误差 >1e-5 → ONNX 与 PyTorch 不是同一份权重/opset。

### 步骤 4：核对定点参考与 manifest 验收字段

```bash
cd <包根>/data/lnn/export
python -c "import json;a=json.load(open('manifest.json',encoding='utf-8'))['acceptance'];print(a['float_task_mse_skip5']['value'],a['int_vs_float_mse_skip5']['value'],a['onnx_max_abs_err'])"
python <包根>/board/lnn/cfc_sim_model.py --selfcheck <包根>/data/lnn/export
```

期望：`1.234134383167709e-05 0.00011718563615310465 4.76837158203125e-07`；自检末行
`== cfc_sim_model selfcheck: PASS ==`（14 项 `OK`，含 `gate_pre includes W_b branch (contract V1.1)`）。
阈值：浮点任务 MSE `≤0.005`、定点 vs 浮点 MSE `≤0.001`（实测 max abs 0.0476）。
失败分支：出现 `gate_pre ...` 之外的 FAIL → 整数语义与合同 V1.1 不一致，禁止进入 G1。

### 步骤 5：RTL 逐位仿真（G1 RTL 侧）

```bash
call E:\Vitis\Vivado\2021.1\settings64.bat
cd D:/DL_FINAL/LNN
vivado -mode batch -notrace -source scripts/run_sim.tcl
```

脚本行为：检查 `export/gen` 6 个必需文件 → `cd verification` → `xvlog -sv -i ../export/gen -f filelist.f` →
对 3 个 TB 逐个 `xelab -debug typical` + `xsim <snap> -runall` → 只看日志标记（`CFC_TB_PASS`/`CFC_TB_FAIL`），不靠退出码猜测。
期望末行 `CFC_SIM_PASS tb_cfc_cell tb_cfc_seq tb_cfc_protocol`、exit 0，日志在 `verification/logs/`。
实测：`tb_cfc_cell` 21 用例 / 0 mismatch / 平均 139 周期每步；`tb_cfc_seq` 626 checks / 0 mismatch / `RUN_CYCLES=9088` /
`STEP_COUNT=64`；`tb_cfc_protocol` 155 checks / 0 mismatch。
失败分支：`CFC_SIM_FAIL missing export/gen/...` → 根布局不对（包内 `build/lnn/scripts/run_sim.tcl` 沿用快照布局，本机实跑即报此行、exit 1）；
`CFC_SIM_FAIL xsim tools not found (source settings64.bat first)` → 未 source settings（本机实跑同此、exit 1）；
`CFC_TB_FAIL ... mismatch` → RTL 与 golden 不同源。

### 步骤 6：三件套哈希配对（G3 身份，独立于构建日志）

```bash
cd <包根>/board/lnn
sha256sum lnn_cfc.bit lnn_cfc.hwh lnn_cfc.xsa
python -c "import hashlib,zipfile;z=zipfile.ZipFile('lnn_cfc.xsa');print(hashlib.sha256(z.read('lnn_cfc.bit')).hexdigest());print(hashlib.sha256(z.read('design_1.hwh')).hexdigest())"
grep -o 'DEVICE="[^"]*" PACKAGE="[^"]*" SPEEDGRADE="[^"]*"' lnn_cfc.hwh
grep -o 'BASEVALUE="0x480000000"[^/]*' lnn_cfc.hwh
```

期望（实测）：

| 文件 | 字节 | SHA256 |
|---|---|---|
| `lnn_cfc.bit` | 28,700,861 | `0fcd8d65d0f8a101d795270741ca7c0cef21f1904590de78226a094b74be30a9` |
| `lnn_cfc.hwh` | 380,358 | `fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb` |
| `lnn_cfc.xsa` | 2,203,972 | `f01d6a8af07fc74acae3dbc30e362db333151978064d3584fcdfc611dbc27032` |

判据：XSA 内 `lnn_cfc.bit` == 外部 bit；XSA 内 `design_1.hwh` == 外部 hwh（本机实算一致）。hwh 另给两项接口证据：
`DEVICE="xczu15eg" PACKAGE="ffvb1156" SPEEDGRADE="-2"`；`BASEVALUE="0x480000000"` 行含 `INSTANCE="M00_AXI_0"` 与
`HIGHVALUE="0x480000FFF"`（4 KiB 窗口，对应驱动 `CSR_BASE`）；`CLKFREQUENCY="300000000" ... NAME="npu_clk"` 有命中。
失败分支：三个哈希任一不符 → 不同次构建（§7 第 4 条）；XSA 内 bit 不等 → XSA 非 `-include_bit` 导出，不可部署。

### 步骤 7：板端 64 步核对（G6）

```bash
cd /home/xilinx/jupyter_notebooks/board        # 整个 board/ 目录拷到同一目录，驱动与 export/ 同级
rm -rf export/__pycache__
python3 -B lnn_cfc_run.py --bit lnn_cfc.bit --expect-dir export --sample sample_000 --out-dir results/s0_hw --checks
```

期望关键行：`source=hardware hardware_measurement=True`、`ID : 0x43464331 ok=True`、`VERSION : 0x00010000 ok=True`、
`CLOCK_HZ : 300000000 ok=True`、`outputs matched 64/64 bit_exact=True final_state_match=True`、`step_count=64`、
`error_code=0`、`RESULT: PASS`，且 `--checks` 的 `repeat / segments / clear-each-step` 三项 PASS。
开发机无板时可在**包内**实跑（本次实测，退出码 0）：

```bash
cd <包根>/board/lnn && python -B lnn_cfc_run.py --backend simulation-model --sample sample_000 --out-dir <临时目录> --checks
```

期望关键行：`source=simulation-model hardware_measurement=False`、`same_stem=True`、`bit_sha256 : 0fcd8d65…`、
`ID/VERSION/CLOCK_HZ ok=True`、`manifest files: total=33 verified=33 mismatch=0 missing=7`（missing 是开发树残留条目，见 §5 注）、
`outputs matched 64/64 bit_exact=True final_state_match=True`、`step_count=64`、`run_cycles=2910`（**模型假设值**，见 §7 第 12 条）、
`checks RESULT: PASS`（repeat/segments/clear_each_step）。
失败分支：§7 第 1/2/5/6 条。

## 5 产物与清单

| 路径 | 生成者 | 关键内容 | 自证（字节 / SHA256 前 8 位） |
|---|---|---|---|
| `data/lnn/export/weights.hex` | `generate_export.py` | 84 行 × 4 hex（int16 补码），第 n 行 = 权重索引 n | 420 / `1addd01a` |
| `data/lnn/export/biases.hex` | 同上 | 17 行 × 8 hex（int32 补码） | 153 / `c5a65206` |
| `data/lnn/export/tanh_lut.hex` | 同上 | 1025 行；`i=0→-4096`、`i=1024→+4096` | 5125 / `a7ec65c0` |
| `data/lnn/export/sigmoid_lut.hex` | 同上 | 1025 行；`i=0→1`、`i=1024→4095` | 5125 / `84f1b475` |
| `data/lnn/export/gen/cfc_weight_init.svh` | 同上 | `initial begin` 内 84 条 `w_rom[i]=16'hXXXX` + 17 条 `b_rom[i]=32'hXXXXXXXX` | 2626 / `3f5f21f9` |
| `data/lnn/export/gen/cfc_lut_init.svh` | 同上 | `tanh_rom[0..1024]` + `sigmoid_rom[0..1024]` 共 2050 条赋值 | 60464 / `d640fab6` |
| `data/lnn/export/gen/cfc_rom_params.svh` | 同上 | `CFC_N_W 84`/`CFC_N_B 17`/`CFC_N_LUT 1025`/`CFC_DT_INT 16'd655`/`CFC_N_STEPS_MAX 64` | 115 / `f3ea39ea` |
| `data/lnn/export/gen/x_input.hex` `y_expected.hex` `h_init.hex` `h_expected.hex` | 同上 | sample_000 的 64 点输入 / 64 点输出 / 4 点初态 / 256 行每步 4 状态 | 320/320/20/1280；`4295e535`/`fa4ce2a2`/`a3c7c781`/`2ab9405f` |
| `data/lnn/export/manifest.json` | 同上 | `schema=cfc-manifest-v1`、`contract_version=1.1`、40 条文件哈希、`acceptance`/`metrics`/`task`/`hardware` | 24045 / `2b4196a2` |
| `data/lnn/export/gen/sample_000.json` … `sample_007.npz`（8×2） | 同上 | 每样本 `x_int/y_int/h_int_steps` 与浮点参考、`mse_task_*` | 8×2 文件；板端绘图/比对 |
| `board/lnn/lnn_cfc.bit`、`lnn_cfc.hwh`、`lnn_cfc.xsa` | Vivado 构建（脚本不随包） | 位流 + hardware handoff + 平台导出，同 stem 同次构建 | 见 §4 步骤 6 |
| `board/lnn/SHA256SUMS` | 打包脚本 | 本目录 42 行哈希清单 | `sha256sum -c` 42/42 OK |

注：`manifest.json` 的 `files` 表 40 条 = `export/` 内 33 个随包文件 + 2 条开发树 `export/__pycache__/*.pyc` +
5 条开发树 `model/*`（后 7 条包内不存在）；驱动对 `missing` 只告警、对 `mismatch` 判失败——这是 §7 第 1 条 pycache 陷阱的根源。

## 6 验收（分级，按 evidence-gate 门级）

| 级别 | 判定命令 | 阈值 | 证据落点 |
| --- | --- | --- | --- |
| **G0 导出** | 工作区 `python -B user/export_demo.py --out-dir <新目录>` | 7/7 `SAME`；exit 0；`summary.json` 的 `all_same=true` | `<out>/summary.json`、`<out>/export.log` |
| **G1a ONNX** | 读 `onnx_verify.json` | `max_abs_err ≤ 1e-5`（实测 4.768e-07）；`all_allowed=True` | `data/lnn/export/onnx_verify.json` |
| **G1b 定点参考** | 读 `manifest.json` 的 `acceptance` | 浮点任务 MSE ≤0.005（1.234e-05）；定点 vs 浮点 MSE ≤0.001（1.172e-04） | `data/lnn/export/manifest.json` |
| **G1c RTL 回放** | `vivado -mode batch -notrace -source scripts/run_sim.tcl` | `CFC_SIM_PASS` + 3 个 `CFC_TB_PASS`；0 mismatch；`RUN_CYCLES=9088`、`STEP_COUNT=64` | `verification/logs/sim_*.log`；摘要 `sim/lnn/evidence/ACCEPTANCE.md` |
| **G3 实现** | 读 `sim/lnn/evidence/ACCEPTANCE.md` §2 | WNS ≥0（+0.226）、WHS ≥0（+0.011）、0 失败端点、57464/57464 全布线、DRC 0 Error | `sim/lnn/evidence/ACCEPTANCE.md` §2 |
| **G6 板端** | 板上 `python3 -B lnn_cfc_run.py ... --checks` | `source=hardware`、64/64 `bit_exact=True`、`final_state_match=True`、`step_count=64`、`error_code=0`、三项 checks PASS | 板端 `results/<run>/summary.json` |

口径红线：`sim/lnn/evidence/offline_validation_summary.json` 的 `is_hardware_measurement=false`、`source=simulation-model`，
29/29 PASS 只证明驱动寄存器序列与比对逻辑自洽，不能写进板端验收。

## 7 故障与修复（高频摘要；完整四元组见 `reference/troubleshooting.md`）

| # | 现象 | 根因 | 处置 |
|---|---|---|---|
| 1 | `bring-up validation failed: export/manifest.json config/hash check failed` | 首次运行在 `export/` 写 `__pycache__/*.pyc`，与 manifest 里开发树 pyc 哈希不符（`mismatch`） | 每次跑前 `rm -rf export/__pycache__` 且命令加 `-B` |
| 2 | `export/ directory (weights, LUTs, integer golden) was not found` | 只拷驱动/位流，没拷 `export/` | 把 `export/` 拷到驱动同目录，或 `--expect-dir`/`CFC_EXPECT_DIR` |
| 3 | PYNQ 报 `hardware handoff ... same stem` | `.bit` 与 `.hwh` 不同目录或不同 basename | 同目录同 stem 原样拷贝 |
| 4 | `ID mismatch` / `VERSION mismatch` / `CLOCK_HZ mismatch` | bit 与驱动/期望值不是同一次构建 | 用本包成对三件套，核对 `SHA256SUMS` 与 §4 步骤 6 |
| 5 | `ERROR_CODE=1`（CONFIG） | `N_STEPS` 不在 1..64 或 `DELTA_T` 不在 1..32767 | 驱动默认 64/655；排查外部进程改写 CSR |
| 6 | `ERROR_CODE=3`（TIMEOUT）/ `wall-clock timeout` | 硬件 `TIMEOUT_CYCLES`（默认 1,000,000 ≈3.33 ms）内未完成；正常 64 步仅 9088 拍 | 先 `CLEAR_STATUS` 重跑；持续超时查 overlay 是否真加载 |
| 7 | `weight ... overflows int16 Q12 range` | 权重绝对值超出 Q12 可表示范围 | 查训练是否发散；改数值格式=改合同，需重做 golden/RTL |
| 8 | 7/7 对拍出现 `DIFF` | 权重/配置与仓库 `export/` 不同批次 | 确认 `model_weights.npz` 与 `train_config.json` 同批；看 `export.log` `metrics` |
| 9 | `CFC_SIM_FAIL missing export/gen/...` | `run_sim.tcl` 按 `script_dir/..` 解析根，包内 `build/lnn/scripts` 下无 `verification/` 与 `export/gen` | 在工作区 `D:/DL_FINAL/LNN` 跑，或按快照布局摆好三目录 |
| 10 | `CFC_SIM_FAIL xsim tools not found (source settings64.bat first)` | `xvlog/xelab/xsim` 不在 PATH | 先 `call E:\Vitis\Vivado\2021.1\settings64.bat` |
| 11 | 综合报 `f1/f2/g` 多驱动（8-6858/8-6859）或 ROM 初值为 0 | `include` 在文件作用域，Vivado 2021.1 不解析 `include_dirs` | `include` 必须在 module 体内（当前 RTL 已如此） |
| 12 | 离线 summary 的 `run_cycles=2910` 被当硬件周期 | `cfc_sim_model` 用 `cycles_base=30 + 45*n_steps` 模型假设 | 硬件周期只看 `source=hardware` 的 `summary.json`；RTL 参考值 9088 |

## 8 边界与不适用

- **不含训练**：`train_cfc.py`/`export_onnx.py`/`verify_export_consistency.py` 只在 [工作区] `D:/DL_LNN/model/`；本技能从"已有 `model_weights.npz`"开始。
- **不含位流重建**：Vivado 工程与平台 BD 不随包；改权重/RTL 后必须重新构建并重做 §4 步骤 6。
- **结构固定**：1 输入 / 4 隐状态 / 1 输出，84+17 参数，Δt=655，n_steps≤64；换规模属重做 RTL。
- **不给性能承诺**：`RUN_CYCLES=9088`（300 MHz ≈30.3 µs）只含片上计算；端到端耗时含 PS MMIO，两者不可混用，首版不承诺加速倍数。
- **不产生板端证据**：本机无板卡；`simulation-model` 结果一律标"未在真实板卡上运行"。
- **不要做**：不要把 `simulation-model` 写成 `hardware`；不要把历史实现的 WNS/WHS 写成"本次实现"；不要用 `--allow-clock-mismatch` 掩盖时钟不符；不要手改 `export/` 内文件（manifest 哈希会失配）。

## 9 版本与身份绑定

| 身份项 | 真实值 | 校验方式 |
|---|---|---|
| 发布版本串 | `lnn-cfc-demo-1.0.0`（见 `src/lnn/README.md` 首行；**包内未随 `VERSION` 文件**，勿引用不存在的路径） | `grep -n 'lnn-cfc-demo' src/lnn/README.md` |
| 数值合同版本 | `contract_version = 1.1`（含 `W_b` 门控修正） | `manifest.json` 的 `contract_version` |
| 期望值清单 | `manifest.json` 40 条哈希（`export/` 内 33 条可校验，其余 7 条指向开发树；`files_meta` 同步） | 驱动 bring-up 校验；`mismatch` 即失败 |
| 交付清单 | `board/lnn/SHA256SUMS` 42 行 | `sha256sum -c SHA256SUMS` |
| 位流三件套 | bit `0fcd8d65…` / hwh `fe2d9f0e…` / xsa `f01d6a8a…` | §4 步骤 6（含 XSA 内嵌比对） |
| RTL 冻结集 | 8 个 `.sv`；包内哈希与开发树冻结值逐一对齐（如 `cfc_mac.sv = 7c076f1a…`、`cfc_cell_core.sv = 086ff95a…`） | `cd src/lnn/rtl && sha256sum *.sv` 比对 `reference/artifacts.md` |
| 器件/时钟 | `xczu15eg-ffvb1156-2-i`；`npu_clk = 300 MHz` 单时钟域 | `lnn_cfc.hwh` 的 `SYSTEMINFO` 与 `CLKFREQUENCY` |

规则：任一输入（权重/配置/RTL/约束）变化 → 旧证据立即失效（标 `invalidated_runs`），必须重跑 G0→G6；
重新生成 `manifest.json` 会改变其哈希，需同步 `board/lnn/export/` 与 `SHA256SUMS`。

## 10 证据归档

按 `../evidence-gate/SKILL.md` 的标签与目录约定：导出重算 7/7 标 `pass`（本次执行），落 `<out>/summary.json` +
`export.log`；ONNX 等价落 `data/lnn/export/onnx_verify.json`；
定点参考落 `manifest.json` 的 `acceptance`/`metrics`；RTL 回放标 `pass` 或 `not_run`，落 `verification/logs/`；
时序/DRC 标 `pass` + `record_level` + `measured`，落 `sim/lnn/evidence/ACCEPTANCE.md` §2；板端仅当
`source=hardware` 才可标 `pass`，落板上 `results/<run>/summary.json` + `outputs.npy` + `state_checks.json`；
离线模型 29/29 只能标 `partial`（流程自检），落 `sim/lnn/evidence/offline_validation_summary.json`。

## 11 参考文件索引

| 文件 | 内容 |
|---|---|
| `reference/commands.md` | 可复制命令矩阵（导出/ONNX/自检/仿真/板端/配对/哈希），含真实参数与退出码 |
| `reference/artifacts.md` | 产物与清单字段表（7 文件 + manifest + 三件套 + RTL 冻结哈希） |
| `reference/acceptance.md` | 分级验收清单（G0/G1/G3/G6 的判定命令、阈值、证据落点） |
| `reference/troubleshooting.md` | 故障库：现象/根因/处置/证据 四元组（12 条） |
| `reference/sources.md` | 事实来源索引：每条关键结论 → 出处文件/字段 |
| `../evidence-gate/SKILL.md` | 门级、标签与证据归档规则 |
