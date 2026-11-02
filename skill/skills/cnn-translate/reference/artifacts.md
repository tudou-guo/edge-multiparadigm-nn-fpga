# artifacts.md — 产物与清单字段表

> 每条字段都在随包文件或本机实跑产物中核对过；"真实样例值"取自 [包内] `data/cnn/` 或反引号内文件路径。
> 编译输出目录指 `compile --out-dir` 指定的新目录；`[工作区]` 指 `D:\DL_FINAL\DL`。

## 1 编译输出目录（本机实跑 TinyCNN-28，53 个文件 / 534 KB）

```text
<out-dir>/
├── compile_manifest.json              # 总清单（身份+门状态）
├── compile_report.json                # 仅当 --report 或案例产物
├── ddr_init.bin / ddr_init_manifest.json
├── graph_plan.json / layout_plan.json / memory_plan.json
├── source_graph_ir.json / lowered_graph_ir.json / lowering_manifest.json
├── parameter_manifest.json / node_outputs.json / snapshot_manifest.json / snapshot_desc.txt
├── model/            model.onnx, affine_contract.json, quant_contract.json, input_manifest.json, input_data/__input__.bin
├── inputs/           input_manifest.json, input_manifest_data/__input__.bin
├── params/           <node>_weight.bin, <node>_bias.bin, <node>_quant.bin
├── golden/           <node>__<output>.bin, output__<name>.bin
└── generated/        npu_top.v, rtl_manifest.json, sources.f, source_hashes.tsv,
                      source_root.txt, instruction_stream.json, <name>_instr.mem,
                      tb_params.json, tb_params.vh
```

IP 打包开启时另增：`ip_repo/<ip_name>_<version>/{component.xml, ...RTL...}`、
`ip_repo/<ip_name>_<version>.pack_manifest.json`、`logs/pack_npu_ip.log`。

来源：[工作区] `D:\DL_FINAL\DL\examples\tinycnn28_mnist\artifacts\toolchain_compile_tn4_tm4/` 与本机 2026-11-02 实跑目录（文件集合一致）。

## 2 `compile_manifest.json`

- `schema`：`e1-p2-compile-manifest-1`（[包内] `data/cnn/tinycnn/compile_manifest.json`、`data/cnn/mobiledw/compile_manifest.json`）。
- 顶层 45 个键（mobiledw）/ 38 个键（tinycnn），关键字段：

| 字段 | 类型 | 含义 | 真实样例值 |
|---|---|---|---|
| `graph_name` | str | 图名 | `tinycnn28_mnist` / `mobiledw32_cifar10` |
| `status` | str | 编译总状态 | `pass` |
| `numeric_profile` | str | 数值配置 | `affine-int8-v1` |
| `model_sha256` | hex64 | 输入 QDQ ONNX 的 sha256 | tinycnn `6307d768...e1907cc`；mobiledw `bcf829ca...483a8f4` |
| `source_graph_digest` | hex64 | 源图 IR digest | tinycnn `5db3529a...73aec95` |
| `lowered_graph_digest` | hex64 | 降低后图 digest（= CLI 输出的 `graph_digest`） | tinycnn `832221ba...143b05a`；mobiledw `4b7f5a9d...64bf0e` |
| `source_digest` | hex64 | RTL 源集合摘要（见第 12 节算法） | 两者均 `36ca00cc...67b1061`（tinycnn） |
| `source_root` | str | 编译时源码根 | `D:/dl_v1`（历史）；本机新编译为传入的 `--source-root` |
| `source_files` | list | `[{path, sha256}]` 逐源文件 | 44 项 |
| `instruction_mem` / `instruction_mem_sha256` | str/hex64 | 指令 ROM 镜像与其哈希 | `generated/tinycnn28_mnist_instr.mem` / `503c39a6...00bd408` |
| `instruction_stream` | str | 指令流 JSON 相对路径 | `generated/instruction_stream.json` |
| `ddr_init` / `ddr_init_sha256` | str/hex64 | DDR 镜像与哈希 | `ddr_init.bin` / tinycnn `15cb770c...43c61`；mobiledw `65ad7be9...fb06ebb` |
| `generated_rtl_manifest` | str | 生成 RTL 清单路径 | `generated/rtl_manifest.json` |
| `records` | list | 指令记录 `[{index,node,op,opcode,payload_words}]` | tinycnn 8 条；mobiledw 14 条（含 4 条 `0x001E`） |
| `gates` | object | 12 个门状态 | 见第 3 节 |
| `dense` | `object\|null` | `{mac_impl, tn, tm}` | tinycnn `{"mac_impl":3,"tm":4,"tn":4}` |
| `depthwise` | `object\|null` | Depthwise ABI 与 lane 配置 | mobiledw：`abi=depthwise-conv-v2`、`opcode=0x001E`、`dw_pc=16`、`dw_rq_lanes=2` |
| `engine_profile` | object | 引擎档位来源 | `{"provenance":{...},"requested":{"kind":"flags","overrides":{"depthwise.pc":16,...}}}` |
| `physical_instances` / `pruning` | object | 物理实例裁剪位图 | `allowed_mask=6143`（tinycnn）/ `14335`（mobiledw） |
| `memory_limits` | object | TB/DDR 预算 | `{"mem_bytes":196608,"mem_words":12288,"ddr_guard_bytes":65536,"sim_ddr_limit_mib":256,...}` |
| `simulation_budget` | object | 仿真周期上界（非性能承诺） | `{"cycle_upper_bound_estimate":83472,"performance_claim":"none; simulation budget only","watchdog_cycles":20000000}` |
| `tb_params` | object | TB 参数文件身份 | `{"json":"generated/tb_params.json","json_sha256":"e6ca46bb...","vh":"generated/tb_params.vh",...}` |
| `quant_contract_sha256` | hex64 | 量化合同哈希 | tinycnn `3355ed10...d01aa57` |
| `input_manifest_sha256` | hex64 | 输入清单哈希 | tinycnn `5a6ed76b...bacced` |

## 3 `compile_manifest.gates`（真实取值）

tinycnn：`affine_frontend=pass, board=not_run, codegen=pass, frontend=pass, lowering=pass,
memory_planning=pass, parameter_pack=pass, quant_contract=pass, reference=pass,
reserved_engines=not_used_by_graph, rtl=not_run, spatial_capacity=pass`。

mobiledw 多两项：`engine_parameters=pass, engine_verification=not_run`。
**读法**：`rtl/board=not_run` 是正常的——编译不给 RTL 仿真与板端结论；`reserved_engines=not_used_by_graph` 表示未用引擎被裁掉。

## 4 `generated/rtl_manifest.json`

- `schema=e1-graph-rtl-manifest-1`；字段：`name, run_id, graph_digest, source_digest, source_root, source_files[],
  sources_f, sources_f_pruned[], rtl_sources[], new_core_sources[], files_sha256{}, library_catalog{path,sha256,legacy_snapshot},
  template{path,sha256}, vendor_ip[{path,sha256}], dense, depthwise, engine_profile, engine_pruning,
  capacities{dense_max_in_row_bytes,...}, engine_capacity{}, engine_row_capacity{}, generated_parameters{},
  instr_words, instr_addr_w, instruction_stream, mem, mem_file, npu_top, opcodes[], physical_instances{}, pruning{},
  qualification, quant_allocations{}, records`。
- 真实样例值（[工作区] tinycnn 编译产物）：`instr_addr_w=8`、`instr_words=256`、`records=8`、`opcodes=[1,29,1,29,1,1,29,1]`、
  `capacities={dense_max_in_row_bytes:224, dense_max_weight_tiles:12, max_ch:32, tn:4, tm:4, mac_impl:3,...}`、
  `template={path:"tools/npu_gen/npu_top.v.tpl", sha256:"b181dd9c..."}`、`vendor_ip=[{path:"rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci", sha256:"25cbfe8f..."}]`。
- `qualification` 固定写 `compile-generated RTL only; no functional/OOC/platform/board claim is made by code generation`。

## 5 `generated/instruction_stream.json`

| 字段 | 类型 | 含义 | 真实样例值 |
|---|---|---|---|
| `schema` | str | 版本 | `e1-instruction-stream-2` |
| `framing` | object | 起止标记 | `{"start_marker":"0xFFFFFFFF","terminator":"0xFFFFFFFF","word_count":142}` |
| `instr_words` / `instr_addr_w` | int | ROM 深度/地址位宽 | `256` / `8` |
| `word_count` / `words[]` | int/list | 实际字与 hex 字序列 | `144`（tinycnn 本机实跑）；`["0xFFFFFFFF","0x0000008E",...]` |
| `mem_file` / `mem_sha256` | str/hex64 | ROM 镜像与哈希 | `tinycnn28_mnist_instr.mem` / `503c39a6...` |
| `opcodes` | map | 本轮操作码→算子 | `{"0x0001":"Conv","0x0002":"MaxPool",...,"0x001D":"MaxPool2D"}` |
| `payload_fields` | map | 每操作码 17/16 字载荷布局 | `0x0001: [in_addr, weight_addr, bias_addr, quant_addr, out_addr, h_w, cin_cout, conv_cfg, quant_cfg, act_cfg, in_row_stride, out_row_stride, in_bytes, weight_bytes, bias_bytes, quant_bytes, out_bytes]` |
| `records[]` | list | `{header, index, node_id, op, opcode, payload[], payload_words}` | `{"index":0,"node_id":"conv1_activation","opcode":1,"payload_words":17}` |

## 6 `parameter_manifest.json` 与 `params/`

| 字段 | 类型 | 含义 | 真实样例值（本机实跑 tinycnn） |
|---|---|---|---|
| `schema` | str | 版本 | `e1-p3-parameter-manifest-1` |
| `parameters[].kind` | str | `weight`/`bias`/`quant` | `weight` |
| `parameters[].node` | str | 节点 | `conv1_activation` |
| `parameters[].address` | hex | DDR 地址 | `0x10007000` |
| `parameters[].bytes` / `file` / `sha256` | int/str/hex64 | 长度/相对路径/字节哈希 | `576` / `params/conv1_activation_weight.bin` / `f11f446d...` |
| `parameters[].source_initializer` | str | ONNX 初始化器名 | `conv1_activation.weight` |
| `parameters[].requant_records` | `list\|null` | quant 记录的 `M0/SHIFT/bias_scale/oc/out_zero_point/relu/weight_scale` | `{"M0":1131972204,"SHIFT":39,"weight_scale":0.005006497260183096,"out_zero_point":-128,"relu":false,...}` |

## 7 `ddr_init_manifest.json` + `ddr_init.bin`

| 字段 | 类型 | 含义 | 真实样例值 |
|---|---|---|---|
| `schema` | str | 版本 | `e1-p2-ddr-init-1` |
| `base` | hex | NPU 地址空间基址（ABI） | `0x10000000` |
| `bytes` / `sha256` | int/hex64 | 镜像长度/哈希 | tinycnn `81920` / `15cb770c...`；board r4 `8482816` / `eac1c430...` |
| `writes[]` | list | `{address, offset, bytes, label, sha256}` | `{"address":"0x10004000","bytes":8192,"label":"input:__input__","sha256":"369b2486..."}` |

## 8 规划类清单

| 文件 | schema | 关键字段（真实值） |
|---|---|---|
| `graph_plan.json` | `e1-graph-plan-1` | `stats{compute_nodes:8, nodes:8, tensors:19, max_live_tensors:10, dead_tensors:[]}`、`serial_policy=stable_topological_order_no_independent_reordering`、`schedule[]` |
| `layout_plan.json` | `e1-layout-plan-1` | `policy=int8_nhwc8_a16; explicit_view_copy; logical_fallback_for_non_int8`、`stats{abi_compatible_tensors:35,...}`、`tensors{}.storage` |
| `memory_plan.json` | `e1-memory-plan-1` | `arena{base:"0x10000000", alignment:4096, used_bytes:61440, used_pages:15}`、`checks{alias_binding\|alignment\|live_overlap\|view_binding\|copy_buffers}` 全 `pass` |
| `lowering_manifest.json` | `e1-p3-lowering-1` | `passes=["batch_norm_exact_fold","constant_fold","conv1x1_center_expand","static_split_to_slice","depthwise_recognition"]`、`lowered_nodes`、`mapping[]`（`disposition` 如 `direct/expanded_3x3/depthwise_direct`） |
| `node_outputs.json` | `e1-p2-node-outputs-1` | `node_golden[]/output_golden[]`：`{node,output,file,bytes,sha256}` |
| `snapshot_manifest.json` | `e1-p2-snapshot-manifest-1` | `snapshot_count`（tinycnn 9、mobiledw 15）、`snapshots[].{name,address,bytes,sha256,golden,node,op,output,record_index,when}` |
| `snapshot_desc.txt` | 文本 | 每行 `snap <when> <index> <name> <addr> <bytes> <golden>`；TB 的 `+SNAP_DESC` 输入 |

## 9 QDQ 报告（案例产物，非编译器输出）

| 字段 | 真实样例值 |
|---|---|
| `schema` | `tinycnn28-toolchain-qdq-1` / `mobiledw32-cifar10-qdq-1` |
| `onnx` | `{"opset":13,"sha256":"6307d768...","input":[1,1,28,28],"output":[1,10,1,1]}` |
| `qdq_frontend` | `{"status":"pass","nodes":8,"operators":["Conv","MaxPool",...]}` |
| `calibration` | `{"dataset":"MNIST official train split","examples":10000,"activation_scale_margin":1.05}` |
| `test_accuracy_sample` / `onnxruntime_comparison` | tinycnn `{"qdq_onnxruntime":0.979,"float_pytorch":0.98,"examples":1000}`；mobiledw `{"qdq_accuracy":0.63,"float_accuracy_same_subset":0.636}` |
| `note` | `Static post-training QDQ calibration. Compiler and RTL accuracy are separate gates.` |

## 10 `compile_report.json`（`--report` 结构）

`schema=e1-compile-cli-1`；字段：`command, status, compile_graph(\"pass\"|\"rejected\"), ip_pack(pass|fail|not_run), ip{...},
manifest, records, graph_digest, engine_profile, target, dense, depthwise, generated_parameters, spatial_capacity,
engine_capacity, memory_limits, simulation_budget, reference_cache, gates, requested{...}, out_dir,
engine_pruning, error{code,message}`。
本机实测被拒时：`{"status":"fail","compile_graph":"rejected","error":{"code":"E_CFG","message":"..."}}`，exit 2。

## 11 IP 包清单（`ip_repo/`）

| 文件 | 关键字段（真实样例，[工作区] MobileDW 包） |
|---|---|
| `<ip_name>_<ver>.pack_manifest.json` | `schema=e1-npu-ip-pack-manifest-1`、`mode=full`、`net_name=npu_mobiledw32`、`ip_version=1.0`、`part=xczu15eg-ffvb1156-2-i`、`sources_f{path,sha256}`、`top{path,sha256}`、`instr_mem{path,name,sha256}`、`counts{total:33,rtl:31,new_core:1,xci:1}`、`skipped_sim[]`、`files[]` |
| `<ip_name>_<ver>/component.xml` | IP-XACT：`vendor=user.org`、`library=npu`、`name=npu_mobiledw32`、`version=1.0`、`m_axi`/`s_axil` 总线接口与端口映射 |

## 12 身份与哈希算法

| 身份 | 算法 | 复算命令 |
|---|---|---|
| 文件字节哈希 | SHA-256 | `python -c "import hashlib,pathlib;print(hashlib.sha256(pathlib.Path('<f>').read_bytes()).hexdigest())"` |
| `source_digest` | `sha256(json.dumps({rel_path: sha256}, sort_keys=True))`（sources + template + vendor_ip） | `python -B -m tools.onnx_gen.source_library list --source-root <root>` |
| `lowered_graph_digest` / `source_graph_digest` | `sha256_json`（canonical JSON：`sort_keys=True, separators=(",",":")`，`graph_ir.sha256_json`） | 编译器内部计算，CLI 输出 `graph_digest` 可比对 |
| 包身份 | `board/cnn/MANIFEST.json` 的 `identity` 段 汇总 bit/hwh/ddr_init/source_digest | 见 commands.md C10 |

## 13 随包数据清单（`data/cnn/`）

| 文件 | 内容（真实值） |
|---|---|
| `data/cnn/tinycnn/compile_manifest.json` | `graph_name=tinycnn28_mnist`、8 records、`model_sha256=6307d768...`、`instruction_mem_sha256=503c39a6...`、`source_digest=36ca00cc...` |
| `data/cnn/tinycnn/compile_report_tn4_tm4.json` | `schema=e1-compile-cli-1`、`records=8`、`status=pass`、`dense=null`，并行度在 `requested.dense_overrides={"mac_impl":3,"tn":4,"tm":4}`（mobiledw 的 compile_report 则 `dense` 非空） |
| `data/cnn/tinycnn/qdq_report.json` | opset 13、`qdq_onnxruntime=0.979`、1000 样本 |
| `data/cnn/mobiledw/compile_manifest.json` | 14 records（4×`0x001E`）、`depthwise.abi=depthwise-conv-v2`、`source_root=D:/dl_v1/examples/temp/...` |
| `data/cnn/mobiledw/summary.json` | `schema=e1-p3-graph-sim-2`、`status=pass`、`vlog_rc=0`、`sim_rc=0`、`board_status=not_run`、`sim_define=TB_MAC_IMPL_2,TB_LEAN_FUNC` |
| `data/cnn/mobiledw/snapshot_manifest.json` | `snapshot_count=15` |
| `data/cnn/yolo/model/` | `graph.json`（`nodes=95, tensors=104`）、`heads.json`（6 head、`total_bytes=739200`、strides `[8,16,32]`）、`inputs.json`（`[1,3,640,640]`、`scale=0.00787017...`、packed sha `4aba865d...`）、`export_report.json`（`node_counts={Add:6,Concat:13,Conv:63,MaxPool:3,Resize:2,Split:8}`） |
| `data/cnn/yolo/expected_hashes.json` | `instr_sha256=d841e29b...`、`model_onnx_sha256=ba470414...`、`board_package.bit_sha256=619f900c...`（≠ `board/cnn` 的 `baeb25fc...`） |

## 14 板端包（`board/cnn/`）

| 文件 | 关键字段/真实值 |
|---|---|
| `MANIFEST.json` | `schema=m4-a-board-package-7`；`identity{revision="A timing-fix implementation r4", source_digest=f387744a..., bit_sha256=baeb25fc..., hwh_sha256=63a11375..., ddr_init_sha256=eac1c430..., implementation{run=impl_rtlfix_1029_r4, clock_period_ns=5.0, setup_wns_ns=0.075, setup_failing_endpoints=0, hold_whs_ns=0.01, pulse_wns_ns=0.039, route_errors=0, drc_errors=0, drc_warnings=1385, all_constraints_met=true}, npu_resources{DSP48E2:468, URAM288:72}, npu_ip_version=0.8.0, input_recode_fix="graph-input DDR bytes are signed-INT8 two's complement (qu-128)"}` |
| `runtime/expected_heads.json` | `schema=m4-a-board-heads-1`；6 head：`conv48@0x100AF000/409600B/sha 0452edb5...`、`conv51@0x10113000/153600B`、`conv54@0x10000000/102400B`、`conv57@0x10019000/38400B`、`conv60@0x10023000/25600B`、`conv63@0x1002A000/9600B` |
| `runtime/golden/output__conv*.bin` | 6 个 golden head 原始字节（sha 与 expected_heads 一致） |
| `runtime/ddr_init_manifest.json` | `bytes=8482816`、`sha256=eac1c430...`、`base=0x10000000` |
| `runtime/decode/preprocess.py` | `INPUT_SCALE=0.007870171219110489`、`INPUT_H=INPUT_W=640`、`INPUT_PHYS_C=8`、`INPUT_ROW_STRIDE=5120`；NHWC8_A16 打包、signed 重编码（`qu-128`）、row 按 align16 补齐 |
| `runtime/decode/heads.json` | 6 head 的 `{node, role, bytes, layout=NHWC8_A16, offset_bytes, row_stride, phys_c}` |
| `runtime/test_image.jpg` | sha256 `649c9780871519d61cd5be930fed97b443180a19ab63fce5b1e99d5479564dcd`（= `data/cnn/yolo/model/inputs.json` 的 sample file_sha256） |
