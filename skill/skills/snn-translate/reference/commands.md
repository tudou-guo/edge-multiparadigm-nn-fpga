# snn-translate 命令矩阵（可复制）

> 约定：`<包根>` = 提交包根（本机 `D:/FINAL`）；`<tmp>` = 任意可写临时目录（不要写进包内生成物目录，除明确写出的 `build/snn/generated/*`）。
> 包内 Python 为 3.9.x + numpy；`python` 即实际解释器。所有命令 2026-11-02 在本机复跑过，特别标注的除外。
> 退出码约定：0 = 成功；非 0 = 失败。Windows 下 `python -B` 抑制 __pycache__，建议始终带上。

## 1 环境与入口自检

| # | 工作目录 | 命令 | 期望 |
|---|---|---|---|
| C1 | `<包根>` | `python --version` | `Python 3.9.x` |
| C2 | `<包根>` | `python -c "import numpy;print(numpy.__version__)"` | `2.x`（本机 2.0.2） |
| C3 | `<包根>` | `ls src/snn/tools/toolchain.py src/snn/rtl/ops/spiking/source_manifest.json data/snn/examples/nlp_sentiment_b1/ckpt_real_real50.npz` | 3 个路径均存在 |
| C4 | `src/snn` | `python -B -m tools.toolchain --help` | 见 §2（真实 argparse 输出） |

## 2 CLI 真实接口（argparse，非文档转述）

`src/snn/tools/toolchain.py` 的顶层：`--backend {spikformer}` + 子命令 `{compile,build,package}`。
**注意**：`--backend` 只有 `spikformer`；传其他值打印 `unknown backend: <x>` 并 exit 2。

```text
# python -B -m tools.toolchain --help
usage: toolchain.py [-h] --backend BACKEND {compile,build,package} ...

# python -B -m tools.toolchain compile --help
usage: toolchain.py compile [-h] --config CONFIG --out-dir OUT_DIR
                            [--weights WEIGHTS] [--params-dir PARAMS_DIR]
                            [--input INPUT] [--fixture-seed FIXTURE_SEED]

# python -B -m tools.toolchain build --help
usage: toolchain.py build [-h] --manifest MANIFEST --stage STAGE
                          [--host-tools-root HOST_TOOLS_ROOT]
                          --shared-heavy-lock SHARED_HEAVY_LOCK [--execute]
                          [--wait-sec WAIT_SEC]

# python -B -m tools.toolchain package --help
usage: toolchain.py package [-h] --manifest MANIFEST --out-dir OUT_DIR
```

| 参数 | 默认 | 语义 / 坑 |
|---|---|---|
| `compile --config` | 必填 | 配置 JSON；编译前由 `config.resolve` 全量校验 |
| `compile --out-dir` | 必填 | 已存在 `manifest.json` 时拒绝覆盖（`refusing to overwrite non-empty out_dir`） |
| `compile --weights` | None | 真实 checkpoint `.npz`，触发严格导入（model_kind=real_corpus） |
| `compile --fixture-seed` | None | fixture 模式必须显式给；与 `--weights` 互斥（同时给报 `fixture_seed conflicts with real weights`） |
| `compile --params-dir` | None | **未实现**：给了且无 weights 会报 `params_dir reuse not implemented; pass --weights` |
| `compile --input` | None | 预留；正常转译不需要 |
| `build --stage` | 必填 | 见 §5：`synth\|ooc` 走物理驱动；`ip\|platform` 只打印入口 |
| `build --shared-heavy-lock` | 必填 | 共享 heavy lock 路径；不允许私有锁 |
| `build --execute` | False | 不加只排队；**加了才启动 Vivado** |
| `build --host-tools-root` | `D:/dl_v0.1/verification/host` | 传给物理驱动，包内路径不存在也不影响排队 |
| `build --wait-sec` | 0 | 等锁秒数 |
| `package --manifest / --out-dir` | 必填 | 已存在 `runtime_config.json` 时拒绝覆盖 |

## 3 编译（G0）

### C5 编译 fixture（[包内]，工作目录 `src/snn`）

```bash
cd <包根>/src/snn
python -B -m tools.toolchain --backend spikformer compile \
  --config ../../data/snn/examples/spikformer/config/nlp_sentiment_b1.json \
  --fixture-seed 1 --out-dir <tmp>/fixture
```

期望输出（原文）：

```text
manifest: <tmp>/fixture/manifest.json kind=development_fixture
```

- exit 0；`params/` = 28 `.bin` + 28 `.coe`；`generated/` = 7 个文件；
- `manifest_sha256 = 835ad38dc06f0546bc08612bf052f085edec7f47b34115cfe5079c7b221381c4`（与 `sim/snn/evidence/ACCEPTANCE_CORE.md` §A1 一致）。

### C6 编译真实权重（[包内]，交付口径；工作目录 `data/snn`）

```bash
cd <包根>/data/snn
export PYTHONPATH=<包根>/src/snn          # PowerShell: $env:PYTHONPATH="<包根>/src/snn"
python -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz \
  --out-dir <tmp>/real
```

期望：`kind=real_corpus`；**54 blob**（108 个 `.bin`/`.coe`）；
`manifest_sha256 = f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e`；
`manifest.json` 文件 sha256 = `b71e3f17373beeb3e66e85a462769cb9a16b172db9ad5db71a43d56c29a8a6a4`，
与 `build/snn/generated/_quickstart_real/manifest.json` **逐字节相同**（2026-11-02 实测）。
换用绝对 `--weights` 路径会得到 `a2a655d4…`（路径字符串进 manifest），但 params 哈希集合不变。

### C7 manifest 身份复算

```bash
python -c "import json,hashlib,pathlib; p='<tmp>/real/manifest.json'; m=json.load(open(p,encoding='utf-8-sig')); \
c=dict(m); s=c.pop('manifest_sha256'); \
print('recompute:', hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()==s); \
print('blobs:', len(m['params']), 'rtl:', len(m['rtl_sources'])); \
print('rtl mismatch:', sum(1 for s2,h in m['rtl_sha256'].items() if hashlib.sha256(pathlib.Path(s2).read_bytes()).hexdigest()!=h))"
```

期望：`recompute: True`；`blobs: 54 rtl: 28`；`rtl mismatch: 0`。
（`rtl_sha256` 的键相对 `src/snn`，必须在该目录运行本命令。）

## 4 打包（板端 bundle，不造 bit）

### C8 package

```bash
cd <包根>/src/snn
python -B -m tools.toolchain --backend spikformer package \
  --manifest <tmp>/real/manifest.json --out-dir <tmp>/pkg
```

期望：`packaged 54 param files -> <tmp>/pkg`；产物 `runtime_config.json`、`params/`(54)、`README.txt`；
`runtime_config.json` 里 `"bitstream": "NOT_BUILT (platform stage, shared heavy lock)"`、`"hwh": "NOT_BUILT ..."`。

## 5 物理门（G2/G3；需 Vivado + 共享 heavy lock）

### C9 工作区根执行（推荐；`scripts/spikformer/` 在根下）

```bash
cd D:/DL_FINAL/SNN        # [工作区]，完整发布快照根
# 排队（默认；不启动 Vivado，只写 queued report）
python -B -m tools.toolchain --backend spikformer build \
  --manifest build/spikformer/nlp_48h_20261029_1915/final/nlp_real_5stage_r4_holdfix/manifest.json \
  --stage ooc --shared-heavy-lock <abs lock path>
# 真跑（需锁与 Vivado 2021.1；5 ns 不降频、不加时序例外）
python -B -m tools.toolchain --backend spikformer build ... --execute --wait-sec <sec>
```

期望（排队）：`physical driver: ...` + 物理驱动打印 `source frozen check: 28 files match manifest hashes`（或旧 run 的 19）+ `QUEUED (no heavy start)...`、`queued report: ...`。

### C10 包内替代（只做源码冻结检查 + 排队，不启动工具；[包内] 工作目录 `<包根>`）

```bash
cd <包根>
SNN_ROOT=<包根>/src/snn python -B build/snn/scripts/spikformer/run_physical.py \
  --manifest build/snn/generated/_quickstart_real/manifest.json --stage ooc --lock <tmp>/heavy.lock
```

期望：首行 `source frozen check: 28 files match manifest hashes`；随后 `QUEUED (no heavy start)...`、`queued report: ...`；exit 0（2026-11-02 实测）。
**不要**在包内直接跑 `toolchain build --stage synth|ooc`：驱动路径解析到 `src/snn/scripts/...`，实测报
`can't open file '...\src\snn\scripts\spikformer\run_physical.py': [Errno 2] No such file or directory`。

### C11 build ip/platform 分支（只打印，不执行）

```bash
python -B -m tools.toolchain --backend spikformer build --manifest <m.json> --stage ip --shared-heavy-lock <lock>
```

期望：`stage ip: project tcl = generated/spikformer_project.tcl` + `ip/platform packaging entry not executed here; use scripts/spikformer/run_ip_pack.py once the OOC stage is frozen.`，exit 0。

## 6 回放（G1；需 ModelSim + 主线向量）`[工作区]` 

```bash
cd D:/DL_FINAL/SNN        # [工作区]
python -B scripts/spikformer/run_nlp_sim.py --case small
python -B scripts/spikformer/run_nlp_sim.py --case e2e --plusarg +NLP_NO_FORCE
python -B scripts/spikformer/run_nlp_sim.py --case e2e
python -B scripts/spikformer/run_nlp_sim.py --case real --plusarg +NLP_NO_FORCE
```

脚本真实参数（2026-11-02 `--help`）：`--case {e2e,multihead,real,small}`、`--gen`、`--compile-only`、`--elaborate-only`、
`--trace`、`--plusarg`（可重复）、`--heavy {auto,on,off}`、`--wait-sec`、`--timeout`、`--tag`、`--samples`（real 1–3）。
每个 run 从新目录编译（`build/.../replay/sim_<case>_<timestamp>/`），从 TB 行解析结果，不凭文件存在判 PASS。

期望：`NLP_RESULT case=small pass=1 cycles=39774` / `case=e2e pass=1 cycles=7084772`（noforce）/
`7084812`（bpforce）/ `10627157`（real 3 样本）；`fail_lines=0`。对应证据 `sim/snn/evidence/nlp/replay_*_holdfix.json`。

## 7 纯 Python 自检与合同检查（包内可跑）

### C12 test_short（[包内] 工作目录 `<包根>`）

```bash
cd <包根>
SNN_ROOT=<包根>/src/snn python -B sim/snn/verification/sim/spikformer/test_short.py
```

期望：15 行 `PASS ...`，末行 `FAILURES: 0`，exit 0（2026-11-02 实测）。
不加 `SNN_ROOT` 会报 `ModuleNotFoundError: No module named 'reference'`。

### C13 check_contract（[包内]；验证配置/派生/合同/checkpoint 一致性）

```bash
cd <包根>
SNN_ROOT=<包根>/src/snn python -B build/snn/scripts/spikformer/check_contract.py \
  --config data/snn/examples/spikformer/config/nlp_sentiment_b1.json \
  --contract sim/snn/evidence/nlp/contract.json \
  --checkpoint data/snn/examples/nlp_sentiment_b1/ckpt_real_real50.npz \
  --out <tmp>/check_contract.json
```

期望：`{"status": "pass", "fails": [], "checks": 108, "out": ...}`，exit 0（2026-11-02 实测）。

### C14 test_integer_ref（需 torch + `runtime/spikformer/`，仅 [工作区]）

```bash
cd D:/DL_FINAL/SNN        # [工作区]
python -B tools/spikformer_train/test_integer_ref.py --out <tmp>/math_tests_m1.json
```

期望：`RESULT: 8/8 PASS -> PASS`，exit 0（2026-11-02 实测）。包内缺 `runtime/`，执行会报 `No module named 'runtime'`。

### C15 check_ready（就绪度自查；[工作区]）

```bash
cd D:/DL_FINAL/SNN        # [工作区]
python -B tools/spikformer_train/check_ready.py
```

期望：表格 + `FAIL items: 0`（2026-11-02 实测）。其中 checkpoint/compile/OOC/bit/board 多为 `not_run`（快照未携带主线 build 产物），属正常。
在包内 `src/snn` 直接跑会因 `examples/` 不在该根而多 1 行 `FAIL`（config 行）。

## 8 IP 源包（G4）

### C16 工作区正式打包（同源 + 绑 OOC 报告）

```bash
cd D:/DL_FINAL/SNN        # [工作区]
python -B scripts/spikformer/run_ip_pack.py --manifest <run>/manifest.json \
  --out-dir <run>/ip --ooc-report <run>/reports/ooc_*.json
```

### C17 包内复打包（无 OOC 报告时 `timing_qualified=false`；[包内] 工作目录 `<包根>`）

```bash
cd <包根>
SNN_ROOT=<包根>/src/snn python -B build/snn/scripts/spikformer/run_ip_pack.py \
  --manifest build/snn/generated/_quickstart_real/manifest.json --out-dir <tmp>/ip
```

期望：`{"status": "pass", "sources": 28, "params_copied": 54, "mismatch": [], "timing_qualified": false, "package_sha256": ...}`（2026-11-02 实测）。
正式包证据：`sim/snn/evidence/nlp/final/ip/ip_manifest.json`（`timing_qualified=true`、`package_sha256=894599b9…292f22`、19/19 源哈希）。

## 9 板端（G6；`[工作区]`）

```bash
# 包内 board/snn 整个目录拷到板子 /home/xilinx/snn_board 后执行
cd /home/xilinx/snn_board
sha256sum -c MANIFEST.sha256              # 20 文件
sha256sum -c MANIFEST_DEMO_20261031.sha256  # 5 条补充清单（4 demo 资产 + README_SNAPSHOT）
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real --samples 1 --json-out snn_board_result_s0.json
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real --samples 3 --json-out snn_board_result.json
```

期望关键行：`SNN id=0x534E4E31 version=0x00020000`、`sample N: match=True cycles=... model_ms=35.4`、`all_match=true`。
**r7 bit 的 sums 有 +64 字偏移**：`snn_platform_run.py` 原样读会 64 通道错位，需按 `ROT=64` 补偿（`src/snn/docs/DEMO_NLP.md` §3）。

## 10 证据核对命令

| # | 目标 | 命令 |
|---|---|---|
| C18 | 证据哈希清单完整性 | `cd sim/snn/evidence && python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"` → `bad=0` |
| C19 | 冻结 RTL 哈希 | `cd src/snn && python -c "import json,hashlib,pathlib; m=json.load(open('../sim/snn/evidence/nlp/rtl_freeze.json')); bad=[s for s,h in m['files'].items() if hashlib.sha256(pathlib.Path(s).read_bytes()).hexdigest()!=h]; print('freeze files=%d mismatch=%d' % (len(m['files']), len(bad))); print(bad[:4])"` → `mismatch=2`（`sps/spiking_sps_top.sv`、`sps/spiking_eyeriss_pe.sv`，图像线并行改动；NLP 功能 RTL 16/16 未变） |
| C20 | OOC 结果读字段 | `python -c "import json; d=json.load(open('sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json')); print(d['status'], d['wns_setup_ns'], d['wns_hold_ns'], d['unrouted_nets'], d['latch_cells'], d['facts'])"` → `pass 0.135 0.023 0 0 {...}` |
| C21 | 回放结果读字段 | `python -c "import json; d=json.load(open('sim/snn/evidence/nlp/replay_e2e_noforce_holdfix.json')); print(d['pass'], d['cycles'], d['steps'][2]['result_line'], d['steps'][2]['fail_lines'])"` → `True 7084772 NLP_RESULT case=e2e pass=1 cycles=7084772 0` |
| C22 | IP 包资格 | `python -c "import json; d=json.load(open('sim/snn/evidence/nlp/final/ip/ip_manifest.json')); print(d['timing_qualified'], d['package_sha256'], len(d['assumptions']))"` → `True 894599b92c8d…292f22 5` |
| C23 | 平台 r7 时序摘要 | `cat build/snn/reports/platform_timing_summary_20261031.txt`（含 WNS +0.051 / WHS +0.011 / 0 failing） |
| C24 | 板包哈希 | `cd board/snn && sha256sum -c MANIFEST.sha256`（Windows 无 sha256sum 时用 `python -c` 逐行复算；bit sha256 `9b2e318c…c267b`、hwh `fe2d9f0e…374fb`） |

## 11 依赖与执行环境

| 命令 | 依赖 | 状态 |
|---|---|---|
| C9/C10 `--execute` 物理门 | Vivado 2021.1 + 共享 heavy lock（`D:/dl_v0.1/.../heavy.lock.json`） | ；仅 C10 排队分支实测 |
| C11 `build --stage ip\|platform` | 无（只打印） | 可跑，不产生工具执行 |
| C6 回放四档 | ModelSim SE-64 2020.4 + 主线向量（`D:/dl_snn/build/.../replay/case_*`） | ；证据 JSON 已在包内 |
| C14 test_integer_ref | torch（本机 2.8.0+cpu）+ 工作区 `runtime/` | 工作区实测通过；包内不可跑 |
| C15 板端 | ZU15EG 板卡 + PYNQ 2.7 | 板卡离线，`record_level` only |
