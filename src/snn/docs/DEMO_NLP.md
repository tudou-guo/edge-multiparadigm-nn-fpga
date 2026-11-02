# NLP 情感分类演示手册（PC 侧自检 + PYNQ 板端）

演示主线：**ChnSentiCorp 真实语料训练 → 整数 checkpoint 导出 → 编译/参数包 → 冻结 RTL 回放 0 mismatch → 5ns routed OOC pass → 平台 r7 bit → 板端 3/3 样本 match（主会话记录；快照 not_verified）**。
本文件所有 `pass / fail / partial / not_run` 都指证据；未执行写 `not_run`。演示包在 `../board/`，资产在 `../examples/nlp_sentiment_b1/`，门状态总表见 `PLATFORM_AND_STATUS.md`。

> 板卡 `192.168.2.99` 当前离线：本文 §2 是**可照做的操作步骤**，§4 的板端结果是**主会话记录**（原始 run JSON/控制台日志不在当前工作区 → 快照 `board = not_verified`，不宣称板测 pass）。最终口径以 `docs/DELIVERY_SUMMARY.md` 为准。

---

## 1. PC 侧数值自检

### 1.1 纯 Python 短测（本快照内可直接跑）

```bash
cd <快照根>    # D:/DL_FINAL/SNN
E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py
```

本次在快照根执行结果：**15 项 PASS，FAILURES: 0**（exit 0）。覆盖：LIF 阈值/负移位、popcount=32、多 block 残差、packing 往返、单 block/多 block 生命周期（双 start 拒绝/ack 清 done）、conv tap、OR 池化。
脚本：`verification/sim/spikformer/test_short.py`。

### 1.2 M1 数学/接口自检：test_integer_ref 8/8 PASS，47 层整数 trace

```bash
cd /d/dl_snn    # 需要 torch 环境与完整主线
E:/miniconda/envs/pytorch_clean/python.exe -B tools/spikformer_train/test_integer_ref.py \
  --out build/spikformer/nlp_48h_20261029_1915/reports/math_tests_m1.json
```

已执行结果：**8/8 PASS**（主线 `reports/math_tests_m1.log`）。**快照根复跑实测**（2026-10-31 17:1x，`E:/miniconda/envs/pytorch_clean/python.exe -B tools/spikformer_train/test_integer_ref.py --out build/_selfcheck_math/math_tests_m1.json`）：**8/8 PASS，RESULT: 8/8 PASS -> PASS**（47 层 trace 两项都在；报告 `build/_selfcheck_math/math_tests_m1.json`）。两次数值一致性由断言保证；fixture 的数值字段可能随 fixture 生成代码版本略有差异（如 `pl_sum_max` 13→10），不影响 PASS/FAIL 结论。

| # | 检查 | 结果 |
|---|---|---|
| 1 | LIF 阈值边界/floor/负值/hard reset | PASS |
| 2 | popcount=32（SUM_W=6）、AV max=1024 | PASS |
| 3 | 跨 token 且跨 t 的 one-hot、Q/K 非对称、per-head 隔离 | PASS |
| 4 | 负累加不发放 | PASS |
| 5 | embed 共享量化域 + PS head 与 `runtime/spikformer/head.py` 逐 bit 一致 | PASS |
| 6 | STE 前向 == round(clamp(w)) | PASS |
| 7 | **full_trace_small_E32：PyTorch 量化前向 vs integer_ref 逐层整数 trace 完全一致，47 层** | PASS |
| 8 | **full_trace_target_E128：同上，47 层**（目标配置 E128/H4/HD32/seq32/T4/D2） | PASS |

补充独立复核：Lead `math_lead_check.json` 23/23 pass（attention 轴/head 隔离/popcount/AV/LIF 边界/fixture 前向）；`intref_fix.json` 6/6 pass（LIF 时间轴、Q LIF 时变、lane 打包、残差上界）。
证据：`math_tests_m1.log/.json` 快照副本 `evidence/nlp/`；`math_lead_check.json`、`intref_fix.json` 在主线 `build/spikformer/nlp_48h_20261029_1915/reports/`（未复制进快照，见 `evidence/INDEX.md` §6）。

### 1.3 冻结 RTL 回放：4 组 mismatch=0

```bash
cd /d/dl_snn
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case small
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case e2e --plusarg +NLP_NO_FORCE
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case e2e                 # 反压版
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case real --plusarg +NLP_NO_FORCE
```

| 用例 | cycles | mismatch | 证据（主线 `build/spikformer/nlp_48h_20261029_1915/reports/`） |
|---|---:|---:|---|
| small | 39,774 | 0 | `replay_small_holdfix.json` |
| e2e +NLP_NO_FORCE | 7,084,772 | 0 | `replay_e2e_noforce_holdfix.json` |
| e2e 反压（14 输出/10 输入 stall 窗口） | 7,084,812 | 0 | `replay_e2e_bpforce_holdfix.json` |
| real 3 样本 | 10,627,157 | 0 | `replay_real_holdfix.json` |

比对范围：embed 每个 beat、block-in/block-out 每个 beat、`o_out_last`、`o_sums[128]`、done；回放覆盖 noforce 与反压（force，14 输出/10 输入 stall）两种模式，均 0 mismatch。冻结 RTL 18 文件哈希见 `reports/rtl_freeze.json`（回放与 OOC 同源）。快照证据副本：`evidence/nlp/replay_*_holdfix.json`、`evidence/nlp/rtl_freeze.json`。

### 1.4 编译自检

```bash
cd <快照根>
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz \
  --out-dir build/nlp_sentiment_b1
```

快照根实测（2026-10-31 17:41，RTL 17:38 最终刷新后，exit 0）：manifest `model_kind=real_corpus`、54 个 param blob（`params/` 108 个 .bin/.coe），`manifest.manifest_sha256=f727e4c7…911e`，manifest 文件级 sha256 `b71e3f17…a6a4`。主线同源产物 `afc12870…2c23`；**进程内 manifest_sha256 不同是正常的**——它绑定本次输出目录/config 路径与 `rtl_sha256` 字段（17:38 快照刷新后 2 个图像线 RTL 文件哈希变了，17:00 的旧读数 `1047d0d0…` 已作废），模型语义（real_corpus/54 blob，params 集合不变）一致；复核以 `model_kind` 与 blob 集合为准（详见 `examples/nlp_sentiment_b1/README.md` §3.4）。

---

## 2. PYNQ 板端演示步骤

### 2.0 需要的东西

| 项 | 要求 |
|---|---|
| 板卡 | ZU15EG（RIGUKE ZU15EG），PYNQ 2.7.0，PL DDR4 正常 |
| 网络 | 能 ssh/scp 到 `xilinx@<板卡IP>`（历史 IP `192.168.2.99`，密码 `xilinx`） |
| 运行目录 | `/home/xilinx/snn_board`（或 `/home/xilinx/jupyter_notebooks/board`，两者历史都用过） |

### 2.1 放置 bit/hwh（必须同目录同 stem）

`snn_platform.bit` 与 `snn_platform.hwh` **必须同目录、同名**（PYNQ `Overlay` 靠 .hwh 解析 IP）。

```text
board/
├── snn_platform.bit      ← r7-sumsfix，sha256 9b2e318c5eacd2f21c39bd6c8864cd0f41ca47ed069ee6bbd7ea45b0191c267b
├── snn_platform.hwh      ← sha256 fe2d9f0edcde37d4bfd574d61c079eb892c89de30a052c59d093ad59be6374fb
├── snn_platform_run.py   ← PYNQ 批量 runner（向量→DDR→start→读 sums→比对 golden）
├── demo_nlp.py           ← 交互演示（文本→token→硬件→PS head→中文结论）
├── demo_vocab.json / demo_head.npz / demo_samples.json
├── MANIFEST.sha256 / MANIFEST_DEMO_20261031.sha256   ← 包完整性校验
├── board_bringup_cell.py ← 早期 bring-up 单元（可选）
└── vec/case_real/        ← 3 个真实样本 etab/ptab/toks/stream/exp_sums + golden.json
```

无 hwh 时典型报错：`hardware handoff not found next to bitstream`。

### 2.2 scp 整个目录到板子

```bash
scp -r board xilinx@<板卡IP>:/home/xilinx/snn_board
```

到板后先在板端校验包完整性（MANIFEST 覆盖 20 个文件，2026-10-30 23:18 生成；**4 个 demo 资产在此后加入、不在原 MANIFEST 内**，其哈希见 `board/MANIFEST_DEMO_20261031.sha256`）：

```bash
cd /home/xilinx/snn_board && sha256sum -c MANIFEST.sha256
sha256sum -c MANIFEST_DEMO_20261031.sha256
```

### 2.3 先跑 1 个样本，再跑 3 个样本

```bash
cd /home/xilinx/snn_board

# 单样本（首次建议）
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real \
    --samples 1 --json-out snn_board_result_s0.json

# 三样本全跑
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real \
    --samples 3 --json-out snn_board_result.json
```

关键输出行（`README_BOARD.md` §3）：

```text
SNN id=0x534E4E31 version=0x00020000
sample 0: match=True cycles=7084772 board_ms=... model_ms=35.4
sample 1: match=True cycles=...
sample 2: match=True cycles=...
wrote snn_board_result.json
```

`all_match=true` + exit 0 为通过。判定内容：PL 写回的 128×u32 sums 与 `exp_sums_sN.hex` 逐通道相等。`board_ms` 主要花在 MMIO 写 PL DDR4 刺激（约 0.92 MB/样本），不是 PL 计算时间；`model_ms = cycles/200 MHz`。

> **重要（r7 bit 的 sums 读法，见 §3）**：当前 bit 的 SUMS 窗口有 +64 字偏移。`snn_platform_run.py` 的 `read_sums()` **没有**做旋转，直接在 r7 bit 上跑会得到 64 通道错位、与 golden 比对失败。两种正确做法：
> - **A（现场补丁，最小改动）**：把 `snn_platform_run.py` 的 `read_sums` 改为 `[self.rd(CSR['SUMS'] + 4*((i+64)%128)) for i in range(128)]`（与 `demo_nlp.py` 的 `ROT=64` 一致）；
> - **B（重出 bit）**：按 `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §5.1 把 RTL 解码改成 `sums_flat[{(araddr_q[8:2] - 7'd64), 5'b0} +: 32]` 重新综合实现，此后 `ROT=0`、runner 不用改。
>
> handoff §1.2 明确记录了“读地址 i 返回 sums[(i+64)%128] → 读第 j 通道用 (j+64)%128”的补偿读法并称 3 样本 match=True；据此推断记录中的 3/3 match 用的是补偿读法（原始 run JSON 缺失，无法直接核实）。README_BOARD §3 的 runner 原样输出与当前 RTL 分析存在矛盾 → **该矛盾未核实**；现场复跑请按 A 先自测单样本。

### 2.4 demo_nlp.py 交互演示（文本输入 → 中文结论）

```bash
cd /home/xilinx/snn_board
python3 -c "from demo_nlp import SnnNLP; d=SnnNLP(); d.show('这个手机很好用')"
# 批量（6 句演示样本，带人工标注）
python3 -c "from demo_nlp import SnnNLP; SnnNLP().show_many()"
```

输出形如（**数值仅示意，不是录制结果**）：

```text
┌────────────────────────────────────────────────────────
│ 输入：这个手机很好用
├────────────────────────────────────────────────────────
│ 负面 ████···················· 17.3%
│ 正面 ████████████████████···· 82.7%
├────────────────────────────────────────────────────────
│ 结论：正面 / POSITIVE   置信度 82.7%（高）
└────────────────────────────────────────────────────────  PL cycles=7084772
```

实现链（`demo_nlp.py`）：按字切词（`demo_vocab.json` 4000 字表，保留前 32 字、pad=0/unk=1）→ 复用 `snn_platform_run.load_sample` 写表/权重 → 只覆盖 tokens → `program()/start_and_wait()` → 读 128 通道 sums（`ROT=64`）→ 除以 T*N=128 → FP32 head（`demo_head.npz` 的 2×128 权重 + 2 偏置）→ softmax → 正面/负面。与 `tools/backends/spikformer/integer_ref.py` 的 PS head 语义一致；资产是 JSON + 纯数值 npz（无 pickle）。

**注意**：`predict()` 每句都 `Overlay(bit)` 重载一次（PL 跑完不回 idle），所以每句约 2–3 s；演示前可先单跑一句确认 `id=0x534E4E31 / version=0x00020000`。

---

## 3. 必查项结论：r7 bit 是否还需要 ROT=64？

**结论：需要。r7-sumsfix 只修了 SUMS 读窗口的“宽度”，没修“基址偏移”，当前 bit 仍必须用 `ROT=64` 读。**

### 3.1 证据：两版 RTL 的差异（`D:/spk_platform/src/snn/`）

| 版本 | 解码代码 | 行为 |
|---|---|---|
| 修复前 `snn_axi_lite_csr.sv.bak_sumsfix`（r6 及更早） | `if (araddr_q[11:8] == 4'h1) s_axil_rdata = sums_flat[{araddr_q[6:2], 5'b0} +: 32];` | 只映射 0x100–0x1FF 的 32 个字，**每 32 通道回绕**（板端看到 `sums[32+i]==sums[i]`） |
| **r7-sumsfix（当前，bit 即此版）** | `if (araddr_q >= 12'h100 && araddr_q < 12'h300) s_axil_rdata = sums_flat[{araddr_q[8:2], 5'b0} +: 32];` | 窗口扩到 128 个字（0x100–0x2FC），**但 `araddr_q[8:2]` 含基址 0x100>>2=64** |

RTL 文件 mtime 2026-10-30 21:56 < bit 生成时间 23:10/23:15，且 `board/snn_platform.bit` 与 `project_1.runs/impl_1/top.bit` sha256 相同（`9b2e318c…`），可确认 **bit 内就是右列的当前解码**。

### 3.2 推导：当前 bit 读第 i 个字返回 `sums[(i+64)%128]`

设字地址 `i`，AXI 地址 `0x100+4i`。`araddr_q[8:2] = ((0x100+4i)>>2) mod 128 = (64+i) mod 128`，而 `sums_flat[32*ch +: 32] = o_sums[ch]`（CSR 端口注释：`sums_flat[32*i +: 32] == o_sums[i]`）。因此：

```text
rd(SUMS + 4*i) = o_sums[(i + 64) % 128]        # r7-sumsfix 当前 bit
```

取 j = (i+64)%128 反解即：`sums[j] = rd(SUMS + 4*((j+64)%128))` —— 这就是 `demo_nlp.py` 的 `ROT=64`，**它是正确的**，不是遗留物。

### 3.3 两种读法（现场任选其一）

```python
# 读法 A：当前 r7 bit + 软件旋转（demo_nlp.py 已在用）
sums = [dev.rd(SUMS + 4*((j + 64) % 128)) for j in range(128)]

# 读法 B：按 handoff §5.1 重出 bit（RTL 减去 64）后
sums = [dev.rd(SUMS + 4*j) for j in range(128)]
```

现场不重出 bit 时用 A；若重出 bit，A 的 `ROT` 改 0（或直接用 B），同时修正/删掉 `demo_nlp.py` 的 `ROT=64`。**在未改的 r7 bit 上直接读 `SUMS+4*j`，逐通道错位 64，必然比对失败**；完整协议仿真矩阵也 `not_run`，该偏移只能靠板测复核。

### 3.4 文档口径冲突（并列，不隐瞒）

- `board/README_BOARD.md` §1：r7 相对 r6 的“唯一 RTL 改动：修复 SUMS 读窗口地址解码（原来只映射 32 个字…）” —— **只描述了宽度修复**，未提 +64 偏移。
- `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2 / §5.1：明确写“硬件读地址 i 返回 `sums[(i+64)%128]`”，并给出“必须修”的 RTL 改法与“修完删掉 ROT=64”。
- 结论：**两者不矛盾——README_BOARD 说的是“已修一半”，handoff §5.1 说的是剩余 +64 偏移；当前 bit 必须 ROT=64（读法 A）。** 原始板端 run JSON 不在工作区，按“未核实：r7 后建议 ROT=0，需板测复核”同时给出 A/B 两种读法。

---

## 4. 已记录的板端结果（主会话记录；快照 not_verified）

板卡离线（`192.168.2.99`），本次收拢**未访问板卡**；以下为主会话文档记录。**快照口径（与 `docs/DELIVERY_SUMMARY.md` 一致）：board = 记录级 3/3 match / not_verified**——原始板端 run JSON 与控制台日志未随包（本机尝试因缺 `pynq` 失败，见 `evidence/platform/board/board_run_local_attempt.log`），因此快照**不宣称板测 pass**。

| 项 | 记录值 | 来源 |
|---|---|---|
| 样本 match | **3/3 全部 match=True**（sums 逐通道与 golden 相等） | `README_BOARD.md` §3；`SPIKFORMER_VISION_HANDOFF_20261031.md` §1.2 |
| cycles | **≈7,084,772**（200 MHz 域；与 e2e 回放参考值 7,084,772 / 7,084,812 同量级） | `README_BOARD.md` §3 |
| model_ms | **≈35.4 ms**（= cycles/200 MHz） | `README_BOARD.md` §3 |
| bit | r7-sumsfix，sha256 `9b2e318c…267b`，2026-10-30 23:10 生成 | `SPIKFORMER_VISION_HANDOFF_20261031.md` §1.1；MANIFEST |
| 全片时序 | Setup WNS **+0.051 ns**（TNS 0，0 failing，334,330 endpoints）/ Hold WHS **+0.011 ns**（THS 0，0 failing，333,740）/ WPWS +0.039 | `D:/spk_platform/reports/{sumsfix_timing_setup.rpt,sumsfix_timing_hold.rpt}` |
| 布线 | 0 routing errors，fully routed 180,101/180,101 | `reports/sumsfix_route_status.rpt` |
| 附加冲突数字 | “PL 计算只有 16.95 ms（3,389,415 cycles @200 MHz）” | handoff §1.4；与上表 7,084,772 cycles/35.4 ms **口径不同/未核实**，并列保留 |

`model_ms` 是 PL 计算时间的口径；`board_ms`（整样本墙钟）主要被 MMIO 写 DDR 刺激拖慢，两者不可混用。

---

## 5. 数字口径冲突与准确率口径（并列，不择一）

### 5.1 数字口径冲突：PL 计算时间

| 口径 | 数字 | 来源 | 说明 |
|---|---:|---|---|
| 板端记录 cycles / model_ms（与回放同量级） | ≈7,084,772 cycles / ≈35.4 ms | `board/README_BOARD.md` §3 | 与 e2e 回放参考值 7,084,772 / 7,084,812 一致；**主记录口径** |
| 文档记录的 PL 计算时间 | 16.95 ms（3,389,415 cycles @200 MHz） | `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.4 | 与上一条**互不一致**；报告 JSON 中未找到溯源 → **未核实** |

两个数字**并列保留，不择一**；引用板端性能时必须同时说明“未核实”。`model_ms=cycles/200 MHz` 是 PL 计算口径；`board_ms`（整样本墙钟）主要被 MMIO 写 DDR 刺激（约 0.92 MB/样本）拖慢，两者不可混用。

### 5.2 准确率口径

| 口径 | dev | holdout | 来源 |
|---|---:|---:|---|
| **integer_ref（准确率口径 of record，纯 NumPy 整数、回放 golden truth）** | **75.92%**（n=1200，majority 50.58%） | **75.62%**（n=960，majority 50.00%） | `reports/export_final_real50.json`（confusion dev [[551,56],[233,360]]） |
| torch 整数等价前向（训练过程指标） | 78.33% | 76.77% | `reports/train_real50.json` |
| 第三方文档口径 | 74.7%（多数类基线 63.3%） | — | `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.3；报告 JSON 未找到 → **未核实** |

- 公开 `test.tsv` 1200 行**无标签**，不报 test accuracy。
- class-1 recall 60.7% vs class-0 90.8%（dev），模型真实但偏置；演示时说“不完美”，不要说“很准”。
- `demo_samples.json` 的 6 句是**挑选样本**（3 正 3 负），`show_many()` 里的命中数不代表整体准确率。
- **板端 3/3 match 是数值一致性（sums 与 golden 相等），不是准确率。**

---

## 6. 已知限制

1. **每句 2–3 s**：PL 跑完不回 idle，必须每句重新 `Overlay()`；时间主要花在重载 overlay + MMIO 写约 0.92 MB 刺激，PL 计算记录为 16.95 ms（3,389,415 cycles@200 MHz，handoff §1.4；与 §5.1 的 7,084,772 cycles/35.4 ms 口径冲突，未核实）。
2. **软复位不排空在途 DMA**：`CTRL_SOFT_RESET` 只 flush 两侧 FIFO，不完整排空 DMA 事务；连续跑第二个样本会超时 → 演示脚本用“每样本重载 overlay”绕过，未根治。
3. **SUMS 读窗口 +64 偏移**：r7 bit 仍需 `ROT=64` 软件补偿（§3）；`snn_platform_run.py` 原样直接读会错位，需按 §3.3 读法 A 补丁或重出 bit。
4. **板卡离线 + 原始结果缺失**：`192.168.2.99` 不在线，本次收拢无法现场复跑；§4 是主会话记录（原始板端 JSON/日志未随包；本机尝试因缺 `pynq` 失败，`evidence/platform/board/board_run_local_attempt.log`）→ 快照口径 board=**not_verified**，不宣称板测 pass。
5. **协议仿真矩阵 not_run**：平台只有 CSR/超时 smoke 7/7 PASS（`D:/spk_platform/reports/platform_build_20261030.json`；`sim/tb_snn_csr/vsim.log`），完整协议矩阵未跑。
6. **写完成按常数 4 统计**：wrapper 以写 DMA 的 `dbg_b_count==4` 判 sums 落盘（512 B 输出）；其它长度/4 KB 边界未验证。
7. **平台约束遗留**：顶层 `sys_rst` 无 input delay/false path、DDR4 `reset_n` 无 output delay（check_timing HIGH，对本次功能无影响）；CDC 报告 1 条 Critical（CDC-10，DDR4/复位基础设施路径，已标 False Path，来自基线平台，未重新收敛）。
8. **演示资产边界**：`demo_head.npz` 是 FP32 head 权重（2×128），仍是真实 checkpoint 的 head；demo 文本按训练同款分词（截前 32 字）。

---

## 7. 证据索引

| 内容 | 路径 |
|---|---|
| 纯 Python 短测（快照内可跑） | `verification/sim/spikformer/test_short.py` |
| M1 数学/接口 8/8 PASS、47 层整数 trace | 快照 `evidence/nlp/math_tests_m1.log`、`evidence/nlp/math_tests_m1.json`；主线 `build/spikformer/nlp_48h_20261029_1915/reports/` |
| Lead 独立数学复核 23/23 | 主线 `reports/math_lead_check.json`；`intref_fix.json`（6/6）（未复制进快照） |
| 回放 mismatch=0 ×4 | 同上 `reports/replay_{small,e2e_noforce,e2e_bpforce,real}_holdfix.json` |
| 冻结 RTL 18 文件哈希 | 同上 `reports/rtl_freeze.json` |
| 训练/导出/准确率 | 同上 `reports/{train_real50.json,export_final_real50.json,train_realweight_fidelity_real50.json}` |
| 板端记录（文档级） | `board/README_BOARD.md`；主线 `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1；本机失败尝试 `evidence/platform/board/board_run_local_attempt.log` |
| 快照内 NLP 证据副本 | `evidence/nlp/`（math/replay/final_gates/rtl_freeze 等，详见 `evidence/INDEX.md` §2） |
| r7 时序/布线（快照派生摘要） | `evidence/platform/platform_timing_summary_20261031.txt`、`evidence/platform/r7_sumsfix_route_status.rpt`；原始报告 `D:/spk_platform/reports/{sumsfix_timing_setup.rpt,sumsfix_timing_hold.rpt,sumsfix_route_status.rpt,sumsfix_drc.rpt}` |
| CSR 解码两版 RTL 对照 | `D:/spk_platform/src/snn/{snn_axi_lite_csr.sv,snn_axi_lite_csr.sv.bak_sumsfix}` |
| 本快照副本 | `board/`（bit/hwh/runner/demo/vec/MANIFEST）、`examples/nlp_sentiment_b1/`（checkpoint/assets/corpus_meta） |

---

> 图像线不在本手册范围；其状态以 `docs/DELIVERY_SUMMARY.md` §3 为准（P1 small 四配置 PASS + P1 native SPS 3072 beats PASS；P2/P3/P4 见该表），本手册不复述图像线时间戳细节。
