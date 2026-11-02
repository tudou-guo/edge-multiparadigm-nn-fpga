# troubleshooting.md — CNN 故障库（现象/根因/处置/证据）

> 每条包含四元组。**来源标注**：`[包内]` = 可在提交包复核；`[工作区]` = 开发机路径；`[已否定]` = 该说法被实测反例推翻。
> 未找到证据的传闻另列第 14 节，禁止写成故障结论。

## T1 板端/回放 mismatch 上百万：输入镜像按 u8 存，RTL 按 s8 读 `[已否定旧说法]`

- **现象**：ModelSim 有界 twin 对 revision6 全量部署 FAIL，`mismatch=1,175,149/1,638,400`、`first_offset=2`；板端 heads 无法 6/6。
- **根因**：部署 DDR 镜像按 raw uint8 码（`qu`）打包，而 RTL/整数参考按 signed INT8（`qs = qu - 128`）解释。
  不是平台、`cc_data`、DDR 事务问题。
- **处置**：`affine_frontend.py` 增加 graph input `u8_to_s8` 重编码；`preprocess.py` 的 `pack_nhwc8_a16` 存 signed 字节；
  重编译后 `ddr_init` 哈希由 `d541d998...` 变为 `eac1c430...`，同一 bit 上 6/6 match。
- **证据**：[包内] `board/cnn/MANIFEST.json` 的 `identity.input_recode_fix`（`signed-INT8 two's complement (qu-128)`）；
  [包内] `board/cnn/runtime/decode/preprocess.py` 的 `pack_nhwc8_a16` 注释；[工作区] `D:\dl_v1\docs\M4_PLATFORM_A_BOARD_CLOSURE_20261028.md`。

## T2 复用旧输出目录导致身份漂移 `[已否定旧说法：旧目录可继续用]`

- **现象**：在新输出目录跑通后，为省时间把下一次编译指到旧目录；manifest 里的 `model_sha256`/`source_digest` 与本次模型/源码不符，
  IP 打包报 `IP directory already exists; refusing to repack`。
- **根因**：清单绑定"模型 + RTL 源码字节"；复用目录等于把新旧身份混在一处。历史 r1/r2 板端数据混用也是同一类问题。
- **处置**：每次编译用**新的空目录**（`src/cnn/QUICKSTART.md` 与 `docs/USER_GUIDE.md` 明写）；旧目录只读归档。
- **证据**：[包内] `src/cnn/docs/USER_GUIDE.md`「每次运行都使用新的输出目录」；[包内] `src/cnn/tools/onnx_gen/export_model.py` 的拒绝重打逻辑。

## T3 `E_CFG: mac_impl must be 1 (behavioral) or 3 (DSP)`（本机实测复现）

- **现象**：`--mac-impl 2` 编译直接 exit 2，stdout 是结构化拒绝 JSON，且不创建输出目录。
- **根因**：当前编译 CLI 只接受 1/3（`compile_pipeline.py`）；0/2 只是 RTL 参数编码（`npu_backend_contract.vh`），
  `TB_MAC_IMPL_2` 只是 ModelSim testbench 宏。旧版技能写的"0=串行、2=平衡树、3=DSP"对 CLI 不成立。
- **处置**：生产用 `--mac-impl 3`；功能回放可用 1；不要试图用 CLI 传 0/2。
- **证据**：命令见 `reference/commands.md` C3（本机 2026-11-02 实测 exit 2）。

## T4 `E_CFG: affine numeric profile requires --input-u8`

- **现象**：给了 `--numeric-profile affine-int8-v1` 但漏掉 `--input-u8`，exit 2。
- **根因**：affine 流程需要 uint8 样本生成 DDR 输入与整数参考；在产物写入前就拒绝。
- **处置**：补 `--input-u8 <sample.npy>`；用 `--report <file>` 留证。模型 shape 与样本不一致时同样会被拒。
- **证据**：`src/cnn/tools/onnx_gen/onnx_gen.py` 的 `"affine numeric profile requires --input-u8"`；本机实测（commands.md C4）。

## T5 动态 shape / batch≠1 被拒

- **现象**：编译报 `E_QDQ_DYNAMIC_SHAPE`，消息形如 `"<tensor> has a dynamic/zero dimension: shape [...]"`，或 `"batch must be 1"`。
- **根因**：QDQ 前端要求全静态且 N=1（`qdq_frontend.py` `_tail_dims`）。
- **处置**：重新导出固定 shape、batch=1 的 ONNX；不要指望编译器推导动态维。
- **证据**：`src/cnn/tools/onnx_gen/qdq_frontend.py` L89 定义 `E_QDQ_DYNAMIC_SHAPE`；L219（动态/零维）、L223（batch≠1）用该错误码 `_reject`。
- **说明**：错误码与消息来自**源码拒绝分支**（静态核对），未提供运行复现。

## T6 opset 不在 13..19 被拒

- **现象**：`E_QDQ_UNSUPPORTED_OPSET`，消息 `"opset %d not supported (QDQ graph requires 13..19)"`。
- **根因**：QDQ 前端只接受默认域单一 opset 13..19；随包两个案例都是 **opset 13**（`data/cnn/*/qdq_report.json`），
  不是旧文档写的"opset 17"。
- **处置**：把导出目标改到 13..19；不要改 opset 号而不重新导出（会破坏 QDQ 语义）。
- **证据**：`qdq_frontend.py` L344 调用 `_reject(E_QDQ_UNSUPPORTED_OPSET, ...)`、L345 为该消息；`data/cnn/tinycnn/qdq_report.json` `onnx.opset=13`。

## T7 算子/属性不在注册表

- **现象**：`E_UNSUPPORTED_OP: operator '<X>' is not registered in the E1 subset` 或
  `E_UNSUPPORTED_ATTR: <Attr> attribute(s) not implemented: <name>`。
- **根因**：`op_registry.py` 数据驱动白名单，未登记算子/属性不静默回退。
- **处置**：换算子或从图里消除；不要假设"名字叫 Conv 就都支持"（stride 3、dilation 2、非 depthwise 分组卷积都会被拒）。
- **证据**：`src/cnn/tools/onnx_gen/op_registry.py` `_DEFAULT_SPECS`、`_check_attrs`。

## T8a OOC 脚本报 `missing .../dsp_mul_pipe4.xci`（包内脚本依赖外部构建目录）

- **现象**：`vivado -source run_e1p2_ooc_synth.tcl` 启动即报 `error "missing <project_root> build/mnist_project2_board/reloc_mn8_ip_repo3/mn8_1.0/src/dsp_mul_pipe4/dsp_mul_pipe4.xci"`。
- **根因**：脚本 L32 把 DSP XCI 源写死为旧工程的相对路径，提交包没有 `build/mnist_project2_board/`；L33 检查缺文件即报错。
- **处置**：把 L32 改成 `[包内] src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci`（sha256 `25cbfe8f9485a7d68bfd4ad75979a63f184c301574c288bf06b324d059c7b4de`，同一 DSP IP）；
  或先 `python -B -m tools.onnx_gen.source_library list --source-root <包内 src/cnn>` 看 `vendor_ip` 路径再改。
- **证据**：`build/cnn/scripts/run_e1p2_ooc_synth.tcl` L32（`set ipxci_src [file join $root build/mnist_project2_board/...]`）、L33（`error "missing $ipxci_src"`）；
  `src/cnn/rtl/ip/dsp_mul_pipe4/dsp_mul_pipe4.xci` 在包内存在。
- **说明**：这是**脚本依赖缺口**，不是"绝对路径导致综合失败"的实测记录（history B7 未找到该报错）。

## T8b IP 独立复用/迁移未完成（IP-XACT 残留绝对 DSP 依赖）

- **现象**：把 IP 包搬到新工程/新目录后，不能声明"已迁移、可独立复用"；迁移结论须待重新综合与 BD 验证后才成立。
- **根因**：MobileDW 复制后的 Vivado 工程虽已把 XPR 指向本地 IP 包目录，但 IP-XACT 元数据仍记录指向原 `D:/dl_v1` 候选目录的绝对 DSP 依赖路径。
- **处置**：以当前包根重新打包 IP（让 `pack_manifest`/`component.xml` 重新解析 `source_root`），再重判 G4/G5；
  不得沿用旧 IP 包的"已迁移"表述，也不得引用旧 `repo_root`。
- **证据**：`[包内] src/cnn/docs/PLATFORM_AND_STATUS.md` L15（"IP-XACT 元数据仍记录着指向原 `D:\dl_v1` 候选目录的绝对 DSP 依赖路径，因此迁移和独立复用尚未完成"）；
  `[工作区] D:/DL_FINAL/DL/examples/mobiledw32_cifar10/artifacts/ip_repo_mobiledw32_tn4_tm4_20261030/npu_mobiledw32_1.0.pack_manifest.json` 的 `repo_root` 仍为旧路径。

## T9 exit 4 / `E_CP_EXTERNAL: IP packaging failed`

- **现象**：编译本身 pass，但 IP 打包阶段 exit 4，`error.detail={rc,log,ip_dir}`。
- **根因**：Vivado 不可调用（默认 `E:/Vitis/Vivado/2021.1/bin/vivado.bat` 不存在/不在 PATH）、`pack_npu_ip.tcl` 失败、
  或 `component.xml` 未生成。
- **处置**：只出编译产物用 `--no-pack-ip`；需要 IP 时 `--vivado <真实 vivado.bat>`；查 `<out-dir>/logs/pack_npu_ip.log`。
- **证据**：`src/cnn/tools/onnx_gen/export_model.py` `_pack_ip`（`E_CP_EXTERNAL`、`pack_rc`、`component_xml_sha256`）。

## T10 replay 报 `SourceRoot differs from compiled source_root`

- **现象**：ModelSim 回放启动即 `throw "SourceRoot differs from compiled source_root"`（或 `compile dir escaped project`）。
- **根因**：`rtl_manifest.json.source_root` 与 `-ProjectRoot/-SourceRoot` 不一致；或编译后源码树被改动/替换。
- **处置**：用编译时的同一 source root 跑回放；源码已变则重新编译（旧回放随之失效）。
- **证据**：`build/cnn/scripts/run_p3_graph_shell.ps1` 第 35–59 行；[工作区] `D:\dl_v1\docs\TOOLCHAIN.md`
  「源在编译后发生变化时需用新目录重新编译，不能通过重写旧 manifest 延续 PASS」。

## T11 行为仿真 PASS、综合后板端 `ERR_OPCODE=2` `[已否定旧说法：行为仿真过=RTL 对]`

- **现象**：XSim 行为仿真通过；板端在 descriptor checker 就返回 `ERR_OPCODE=2`（期望 `ERR_BANK_RANGE=5`）。
- **根因**：Vivado 2021.1 综合网表把 `task_table_reg` 引擎副本的 `WEBWE[3:0]` 接成常量 0、`ENBWREN` 接常量 1；
  host 与引擎各读一份内容相同的数组，引擎那份的写端口被优化掉。资源数不变，看不出问题。
- **处置**：改成一份 true dual-port 任务 RAM（A 口写/引擎读、B 口 host 回读），2×RAMB18 → 1×RAMB36；
  增加 `assert_task_table_ports.tcl` 在综合/routed 后拒绝"写使能悬空/常量"。
- **证据**：[工作区] `D:\dl_v1.0\docs\PYNQ_ERR_OPCODE_ROOT_CAUSE_FIX_20261017.md`（含 before/after 仿真日志与引脚证据）。

## T12 200 MHz 实现时序不收敛（clk_wiz 内部 primary 时钟被重定义）

- **现象**：impl_1 默认策略 setup `WNS -0.281 ns / TNS -153.344 / 1519 failing endpoints`；methodology `TIMING-4/TIMING-27` critical。
- **根因**：`clk_wiz_npu` 的 `CONFIG.PRIM_SOURCE = Single_ended_clock_capable_pin`，其 IP XDC 在内部 `clk_in1` 上生成
  `create_clock -period 3.333`，覆盖上游 MIG UI 的 `mmcm_clkout0` 3.331 ns，导致 200 MHz（4.999 ns）约束模型错误。
- **处置**：BD 改 `PRIM_SOURCE = No_buffer`（M/D/O 不变，仍走 BUFG）；新建 run 用 `Performance_Explore`
  （PHYS_OPT_DESIGN 开、POST_ROUTE 关、jobs=4）；不降频、不加 false path。
- **结果（该历史包）**：setup `+0.039 / TNS 0 / 0 failing`（589,345 端点）、hold `+0.010`、pulse `+0.039`、0 unrouted、DRC 0 Error。
- **证据**：[工作区] `D:\dl_v1\docs\M4_PLATFORM_A_IMPL_TIMING_QUALIFIED_20261028.md` §1–§4。
- **口径**：`+0.039` 属于 A_rev6 合格实现；r4 板卡包记录的是 `+0.075`，两者不可合并。

## T13 两个不同 bit/HWH 包被当成同一个

- **现象**：用 `data/cnn/yolo/expected_hashes.json` 的 `board_package.bit_sha256` 去核对 `board/cnn/fpga/platform_top.bit`，永远对不上。
- **根因**：`619f900c...` 属于 A_rev6 历史合格包；`board/cnn` 是 timing-fix r4 包，bit 为 `baeb25fc...`、HWH 为 `63a11375...`。
  另外 MobileDW 走 PS DDR（HP0），历史 YOLO 走 PL DDR4 MIG，存储器映射不可互换。
- **处置**：先按 `MANIFEST.json` 确认要引用哪个硬件包；bit/HWH/ddr_init/runtime/脚本必须来自同一包；跨包一律 `fail`。
- **证据**：[包内] `board/cnn/MANIFEST.json` 的 `identity` 段、`data/cnn/yolo/expected_hashes.json`、
  `src/cnn/docs/PLATFORM_AND_STATUS.md`（两种存储器路径不可替换）。

## 14 未找到证据、禁止写成故障的传闻

| 说法 | 状态 | 处理 |
|---|---|---|
| "DSP IP 绝对路径依赖导致迁移失败（有具体报错）" | 未找到证据 | 只写 T8a 的"脚本依赖缺口"与 T8b 的包内文档"迁移未完成" |
| "ModelSim license 初始化失败" | 未找到证据 | 不写；回放失败先看 `vlog.log/simulator.log` |
| "动态 shape 的拒绝报错文本"（运行实测） | 源码有错误码/消息，无运行复现记录 | 按 T5 写"源码拒绝分支"，注明未复跑 |
| "yolov8 板端 DMA 写不可见已定案为 MMU/cache" | 未定案 | 不写结论 |

## 15 处置清单（按现象快速索引）

| 现象关键字 | 条目 |
|---|---|
| mismatch 百万/位偏移 | T1 |
| 哈希不符/清单漂移 | T2 |
| 参数被拒 `E_CFG` | T3、T4 |
| 图被拒 `E_QDQ_*` | T5、T6、T7 |
| Vivado/打包失败 | T8a、T8b、T9 |
| 回放启动失败 | T10 |
| 仿真过、上板错 | T11 |
| 时序不收敛 | T12 |
| 板包身份不符 | T13 |
