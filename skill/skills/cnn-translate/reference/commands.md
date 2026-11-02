# commands.md — CNN 编译/打包/回放/综合/板端 命令矩阵

> 所有命令都可直接复制。**工作目录**栏是硬约束：`tools.onnx_gen` 模块入口只在 `src/cnn`（或其等价源码根）下可导入。
> 参数只写 `--help` 与源码里真实存在的选项；未在本机执行过的命令标  并写依赖。
> 退出码来源：`src/cnn/tools/onnx_gen/onnx_gen.py`（0=pass、2=编译拒绝、4=编译通过但 IP 打包失败）。
> 方括号路径：`[包内]` 相对包根（本机 `D:\FINAL`）；`[工作区]` 仅开发机存在。

## 0 约定

| 项 | 值 |
|---|---|
| 包根 | `D:\FINAL`（提交包，可整体搬移） |
| 编译器工作目录 | `D:\FINAL\src\cnn`（`-m tools.onnx_gen.onnx_gen`） |
| 开发期完整案例 | `[工作区] D:\DL_FINAL\DL`（工具链 0.3.0，含 `examples/`） |
| Python | 3.9.x；onnx 1.14.1 / numpy 2.0.2 / pillow 10.1.0 |
| 失败语义 | exit 0=通过；exit 2=图/合同/参数被拒（结构化 `error.code`）；exit 4=IP 打包失败 |

## C1 环境自检

```powershell
Set-Location D:\FINAL
python -B build\cnn\scripts\env_check.py
```

- 来源：[包内] `build/cnn/scripts/env_check.py`（与 `[工作区] D:\DL_FINAL\DL\scripts\env_check.py` 内容相同）。
- 期望：`ENV_CHECK: PASS`，exit 0；打印 python 版本与 numpy/onnx/pillow 版本，以及 vsim/vlog/vlib/vivado/xvlog/xelab/xsim 的 PATH 查找结果。
- 本机实测（2026-11-02）：python 3.9.23、numpy 2.0.2、onnx 1.14.1、pillow 10.1.0、vsim/vlog/vlib 在 `D:\modelsim\win64\`、vivado/xvlog/xelab/xsim 不在 PATH。
- 失败分支：`ENV_CHECK: FAIL` → 先装 `onnx/numpy/pillow`；可选工具缺失不阻断编译（只阻断 IP 打包/回放）。

## C2 编译（最常用：只出编译产物）

`[工作区] 完整案例`（TinyCNN-28，本机实测 exit 0）：

```powershell
Set-Location D:\DL_FINAL\DL
python -B -m tools.onnx_gen.onnx_gen compile `
  --model examples\tinycnn28_mnist\artifacts\quantized\tinycnn28_mnist_qdq.onnx `
  --numeric-profile affine-int8-v1 `
  --input-u8 examples\tinycnn28_mnist\artifacts\mnist_sample_u8.npy `
  --source-root D:\DL_FINAL\DL `
  --mac-impl 3 --tn 4 --tm 4 `
  --name tinycnn28_mnist --run-id tinycnn28_writer_verify `
  --out-dir build\examples\tinycnn28_writer_verify --no-pack-ip
```

`[包内] 自备模型`（包内 `src/cnn/examples/` 未随包，QUICKSTART 的样例路径不可直接执行）：

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.onnx_gen compile `
  --model <你的 model_qdq.onnx> --numeric-profile affine-int8-v1 --input-u8 <你的 input_u8.npy> `
  --mac-impl 3 --tn 4 --tm 4 --name <graph_name> --run-id <run_id> `
  --out-dir <新的空目录> --no-pack-ip
```

- 参数来源：`python -B -m tools.onnx_gen.onnx_gen compile --help`（本机实跑，见 `[包内] src/cnn/docs/USER_GUIDE.md`）。
- 默认值：`--source-root`=`src/cnn`（脚本自身根）；`--run-id`=`e1p2`；`--profile`=`first_fit`；
  `--ip-version`=`0.1`；`--pack-ip` 默认开；`--engine-pruning` 对 affine 图为 `graph`；`--maxpool-backend`=`auto`。
- 期望 stdout（本机实测）：
  ```json
  {"status": "pass", "manifest": ".../compile_manifest.json", "records": 8,
   "graph_digest": "832221ba22961bb73c42dd349d2ae2d5f1d67aeab051ce08c44cadd18143b05a",
   "ip": null, "engine_profile": null, "target": null, "depthwise": null, "generated_parameters": null}
  ```
  产物约 50 个文件/534 KB（含 `generated/`、`params/`、`golden/`、`model/`、`inputs/`）。
- 失败分支：见 C3/C4；输出目录复用会覆盖/污染身份，必须先换新目录。

## C3 结构化拒绝：并行度参数非法

```powershell
Set-Location D:\DL_FINAL\DL
python -B -m tools.onnx_gen.onnx_gen compile --model examples\tinycnn28_mnist\artifacts\quantized\tinycnn28_mnist_qdq.onnx `
  --numeric-profile affine-int8-v1 --input-u8 examples\tinycnn28_mnist\artifacts\mnist_sample_u8.npy `
  --mac-impl 2 --out-dir <新目录> --no-pack-ip
```

- 本机实测：exit 2，stdout 为 `{"schema":"e1-compile-cli-1","command":"compile","status":"fail","compile_graph":"rejected","error":{"code":"E_CFG","message":"mac_impl must be 1 (behavioral) or 3 (DSP)"},...}`。
- 且**不会创建输出目录**（结构化拒绝发生在产物写入前）。
- 规则来源：`src/cnn/tools/onnx_gen/compile_pipeline.py` `_resolve_compute_profile`（`int(legacy_mac) not in (1, 3)`）。
- 说明：RTL 参数本身支持 0/1/2/3（`src/cnn/rtl/common/npu_backend_contract.vh`：`NPU_MAC_LEGACY/DSP_ADAPTER/BALANCED/BALANCED_DSP`），
  但当前编译 CLI 只接受 1 与 3；`TB_MAC_IMPL_2` 只是 ModelSim 回放的 testbench 宏。

## C4 结构化拒绝：affine 配置缺样本

```powershell
python -B -m tools.onnx_gen.onnx_gen compile --model <model_qdq.onnx> `
  --numeric-profile affine-int8-v1 --out-dir <新目录> --no-pack-ip --report <dir>\reject.json
```

- 本机实测：exit 2；stdout/report 内 `{"status":"fail","compile_graph":"rejected","error":{"code":"E_CFG","message":"affine numeric profile requires --input-u8"}}`。
- `--report` 会在被拒时也落盘（报告文件是该 JSON），适合 CI 留证。

## C5 算子注册表 / 目标 / 引擎档位

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.onnx_gen registry --out <dir>\registry.json
python -B -m tools.onnx_gen.onnx_gen targets
python -B -m tools.onnx_gen.onnx_gen engine-profiles
```

- 本机实测 `registry`：stdout `{"status": "pass", "out": "..."}`；文件 `schema=e1-op-registry-1`、`opset_range=[11,19]`、`operators` 26 项。
- `targets`：默认 `project2_zu15eg_200mhz`，part `xczu15eg-ffvb1156-2-i`，`clock_ns=5.0`，`axi_data_width=128`，`hash=8c7c2e2d...`。
- `engine-profiles`：默认 `dw_c8_t1_q1`；可选 `dw_c4_t1_q1`、`dw_c16_t1_q1`、`dw_c16_t1_q2`，每个档位 `dense{tn8,tm8,mac_impl3}`。

## C6 源码身份核验（编译后必跑）

```powershell
Set-Location D:\FINAL\src\cnn
python -B -m tools.onnx_gen.source_library list --source-root .
python -B -m tools.onnx_gen.source_library verify --generated <out-dir>\generated
```

- 本机实测 `list --source-root .`：`status=pass`、`rtl_sources` 42 项、`source_digest=36ca00cc095ea7bb010d4576942e8a881a099001a109185a36eff006f67b1061`、
  `template.sha256=b181dd9c...`、`vendor_ip=rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci`。
- 本机实测 `verify --generated <新编译目录>/generated`：`{"status":"pass","source_files":44,"package_files":0,"optional_not_packaged":[]}`；
  `source_digest` 与 `compile_manifest.json` 一致。
- 失败分支：sha 不一致 → `CompileError(E_CFG, "source file changed ...")` 类错误（见 `source_library.py` `verify_generated`）。

## C7 ModelSim 图级回放（G1）

```powershell
Set-Location D:\FINAL
powershell -ExecutionPolicy Bypass -File build\cnn\scripts\run_p3_graph_shell.ps1 `
  -CompileDir <out-dir> -ResultDir <result-dir> `
  -ProjectRoot <compile 所在工程根> -Simulator modelsim -SimDefine TB_MAC_IMPL_2 `
  -Seed 20261030 `
  -Vlog D:\modelsim\win64\vlog.exe -Vsim D:\modelsim\win64\vsim.exe -Vlib D:\modelsim\win64\vlib.exe
```

- 脚本参数来源：[包内] `build/cnn/scripts/run_p3_graph_shell.ps1` 第 1–25 行；testbench 在
  `sim/cnn/verification/sim/tb_p3_graph_shell.sv`（`run_p3_graph_shell.ps1` 在 `sim/cnn/verification/sim/` 也有一份，与包内副本相同）。
- 强制约束（脚本内检查）：`-CompileDir`/`-ResultDir` 必须位于 `-ProjectRoot` 之下；
  `rtl_manifest.source_root` 会回绑 `-ProjectRoot`/`-SourceRoot`，不一致直接 `throw "SourceRoot differs from compiled source_root"`；
  `modelsim` 分支必须带 `-SimDefine TB_MAC_IMPL_2`（`behavioral twin`），否则 `throw`。
- 产物：`<result-dir>/summary.json`（`schema=e1-p3-graph-sim-2`）、`simulation_identity.json`、`snapshots/`、
  `vlog.log`、`simulator.log`、`work/`。
- 通过判据：`status=pass`（`vlog_rc=0`、`sim_rc=0`、`pass_marker=true`、`source_unchanged=true`），
  `simulator.log` 含 `P3_RESULT: PASS` 且无 `** Error/Fatal`。
- 依赖：ModelSim SE-64 2020.4 + 同源 `ProjectRoot`；一次图级回放耗时较长。

## C8 OOC 综合（G2）

```powershell
vivado -mode batch -notrace -source build\cnn\scripts\run_e1p2_ooc_synth.tcl `
  -tclargs <out-dir>\generated <ooc-out-dir> <project_root>
```

- 参数来源：[包内] `build/cnn/scripts/run_e1p2_ooc_synth.tcl` 头部 `USAGE`（`run_hw1_ooc_synth.tcl` 同构）。
- 前置检查：`<generated>` 下必须有 `sources.f`、`npu_top.v`、且**恰好一个** `*_instr.mem`；
  `<project_root>` 必须同时能解析 `sources.f` 的相对路径与 DSP XCI。
- 期望标记：`P2_OOC_IP_OUTPUTDIR=`、`P2_OOC_SOURCES=<n>`、`P2_OOC_RAM_CELLS=...`、`P2_OOC_BLOCK_RAM_CELLS=...`、
  `P2_OOC_DSP48_CELLS=...`、`P2_OOC_SYNTH_DONE=true`；报告 `utilization.rpt`、`utilization_hier.rpt`、
  `timing_summary.rpt`、`timing_max.rpt`、`timing_min.rpt`、`drc.rpt`、`design_analysis.rpt`；checkpoint `npu_top_ooc.dcp`。
- 器件与约束：脚本内写死 `xczu15eg-ffvb1156-2-i` 与 `create_clock -period 5.000`。
- 已知依赖缺口：脚本第 39 行附近把 DSP XCI 源写死为
  `$root build/mnist_project2_board/reloc_mn8_ip_repo3/mn8_1.0/src/dsp_mul_pipe4/dsp_mul_pipe4.xci`，
  **包内没有该路径**；应改为 [包内] `src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci`（同一文件，sha256 `25cbfe8f...`）。
- 依赖：Vivado 2021.1（本机不在 PATH）。

## C9 板端比对（G6，历史 r4 包）

```bash
cd board/cnn
sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh \
  --ctrl-base 0x480000000 --window-base 0x400000000 --out-dir run_out_heads
sudo python3 scripts/board_run_a.py --bitstream fpga/platform_top.bit --hwh fpga/platform_top.hwh \
  --ctrl-base 0x480000000 --window-base 0x400000000 --image runtime/test_image.jpg --out-dir run_out_image
```

- 来源：[包内] `board/cnn/README_BOARD.md`「上板命令」；参数全集见 `scripts/board_run_a.py`（`--ctrl-name`、`--arena-base`、
  `--ddr-init`、`--expected-heads`、`--golden-dir`、`--mode auto|heads|image`、`--conf-thres 0.5`、`--iou-thres 0.3`、
  `--timeout-seconds 10`、`--timeout-cycles 0x0FFFFFFF`、`--poll-sleep 0.0002`、`--skip-ddr-write`、`--smoke-only`、
  `--use-golden-heads`、`--skip-head-compare`）。
- 运行时自检：读 `runtime/ddr_init_manifest.json`，若 `ddr_init.bin` 的字节数或 sha256 不匹配 → `SystemExit("ddr_init.bin identity mismatch")`。
- 通过判据：6 个 head 全匹配（`first_mismatch=-1`）、`error_code=0`；image 模式检测结果与 golden 一致。
- 包内记录：`board_status=PASS_user_reported`（2026-10-29 用户提供 Jupyter 输出），引用时须注明外部来源与日期。
- 依赖：PYNQ 板卡 + 对应 overlay 环境；`scripts/board_run_a.py` 在主机上不能真正跑。

## C10 板端包身份复核（主机可跑，纯哈希）

```bash
cd board/cnn
python - <<'PY'
import hashlib, json, pathlib
man = json.loads(pathlib.Path('MANIFEST.json').read_text(encoding='utf-8'))
bad = []
for rel, row in man['files'].items():
    p = pathlib.Path(rel); data = p.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if len(data) != row['bytes'] or sha != row['sha256']:
        bad.append((rel, len(data), sha))
print('files=%d bad=%d' % (len(man['files']), len(bad)))
for row in bad: print('BAD', row)
PY
```

- 期望：`files=18 bad=0`（`MANIFEST.json` 里 `files` 共 18 项：bit/hwh/ddr/runtime/decode/expected_heads/6×golden/test_image/script；本机实测 18/0）。
- 这是 `board/cnn` 硬件包的**离线身份复核**，不能替代 G6 上板。

## C11 历史报告与数据的读取（不产生新结论）

```bash
# G0 记录：随包真实清单
python -c "import json;d=json.load(open('data/cnn/tinycnn/compile_manifest.json',encoding='utf-8'));print(d['graph_name'],d['status'],len(d['records']),d['instruction_mem_sha256'])"
# G1 记录：ModelSim 回放摘要
python -c "import json;d=json.load(open('data/cnn/mobiledw/summary.json',encoding='utf-8'));print(d['status'],d['simulator'],d['board_status'])"
# G3/G6 记录：历史 r4 实现与板测
python -c "import json;d=json.load(open('board/cnn/MANIFEST.json',encoding='utf-8'));print(d['identity']['implementation']);print(d['board_status'])"
```

- 期望第一行：`tinycnn28_mnist pass 8 503c39a69aa1b27fc09b32829a46468ba0ae8a7d23a2ac37d99196d4a00bd408`。
- 第二行：`pass Model Technology ModelSim SE-64 vsim 2020.4 Simulator ... not_run`（`board_status=not_run`）。
- 第三行：`{'run': 'impl_rtlfix_1029_r4', 'clock_period_ns': 5.0, 'setup_wns_ns': 0.075, ...}`；`PASS_user_reported`。

## C12 旧 ABI 1.0 量化参数导出（与本流程分开）

```powershell
Set-Location D:\FINAL\src\cnn
python -B tools\quant\export_quant_params.py --help
```

- 真实选项：`--layer-json`、`--out-dir`、`--name`、`--self-test`、`--print-json`。
- 只服务冻结 ABI 1.0 的旧 Conv/Pool 案例，**不是** current ONNX QDQ 前端；不要把它产出的 `quant.bin` 交给本工具的 QDQ 流程。

## C13 未收录的入口（不要臆造）

- `python -m tools.npu_gen` **不存在**：`src/cnn/tools/npu_gen/` 只有 `pack_npu_ip.tcl` 与 `npu_top.v.tpl`；
  IP 打包由编译器内部调用 Vivado 执行 `pack_npu_ip.tcl`。v1 文档里的 `python -m tools.npu_gen --generated ... --out ...` 是错的。
- 包内也没有 `src/cnn/scripts/`、`src/cnn/verification/`、`src/cnn/requirements.txt`；对应能力在 `build/cnn/scripts/`、
  `sim/cnn/verification/`，依赖清单在 [工作区] `D:\DL_FINAL\DL\requirements.txt`（`numpy>=1.21, onnx>=1.12, pillow>=9.0`）。
