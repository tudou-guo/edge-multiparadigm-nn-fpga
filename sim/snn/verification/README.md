# verification/ — 仿真 TB 与纯 Python 短例（状态说明）

本目录是 `D:/dl_snn/verification/` 的副本（复制于 2026-10-31 16:36 CST）。
**快照内不重跑 TB**（需要 ModelSim/Vivado 与 `build/` 向量；见下“运行前提”），因此所有 `.v` 的“快照内重跑”一律 **not_run**；
下表“主线历史结果”只引用**已有证据文件**，不把历史 PASS 当成本快照新跑结果。

## 1. 纯 Python（快照内实际执行，2026-10-31）

| 脚本 | 快照内结果 | 说明 |
|---|---|---|
| `test_short.py` | **PASS（15 项，FAILURES: 0）** | LIF 阈值/负移位、popcount=32、多 block 残差>3、packing 往返、生命周期（double-start 拒绝/ack 清 done/二次启动）、conv/gated/pool 语义 |
| `test_runtime.py` | **FAIL（pre-existing，非本快照引入）** | 主线同错：① `import run as R` 触发 `runtime/spikformer/run.py` 的包内相对导入失败；② 修掉导入后 `FakeMMIO` 无真实 memory reader，`read_sums` 按设计拒绝返回占位值。**未修测试逻辑**，仅改其 sys.path 为相对根 |
| `tools/spikformer_train/test_integer_ref.py` | **PASS（RESULT: 8/8 PASS -> PASS）** | 整数参考实现 vs torch 量化前向逐层一致（47 层×2 配置）、LIF 边界、popcount=32/AV=1024、负累加、onehot 跨 token/t、PS head bit 一致、STE。需要 torch |

复现：

```bat
E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py
E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py
```

详见 `evidence/ACCEPTANCE_CORE.md`。

## 2. TB 清单与线路

“主线历史结果”来自 `docs/SPIKFORMER_BACKEND.md`、`docs/SPIKFORMER_NLP_48H_HOLD_CLOSURE_RESULT.md`、
`docs/VISION_LINE_RETURN_20261031.md`、`docs/SPIKFORMER_VISION_HANDOFF_20261031.md` 与 `build/` 下的报告。

| TB | 覆盖 | 主线历史结果（来源/日期） | 快照内重跑 |
|---|---|---|---|
| `tb_min.v` `tb_clkonly.v` | 环境/时钟/复位冒烟 | pass（bring-up，早期） | not_run |
| `tb_spiking_popcount.v` | QK popcount 全 1=32 | **PASS**（`docs/SPIKFORMER_BACKEND.md` §6，2026-10-29） | not_run |
| `tb_spiking_lif_group.v` | LIF 静默/发放/边界 | **PASS**（同上） | not_run |
| `tb_sps_replay.v` | native SPS 8×8 回放（16 beats 逐 lane） | **PASS**（同上） | not_run |
| `tb_vis_sps.v` | legacy 图像 SPS 单级（小图 16 beats） | **PASS（16 beats）**（`docs/VISION_LINE_RETURN_20261031.md`，2026-10-31）；另有 `build/vis_sps.log` 12:33 的 `FAILURES=16` 记录，**口径冲突、未核实**（疑为中间调试 run） | not_run |
| `tb_vis_replay.v` | legacy 图像整链 SPS→block→sums | **PASS（32 channels，TOKEN CHECK fails=0）**（同上文档，2026-10-31） | not_run |
| `tb_sps_ref_small.v` | **参考式** SPS small（P1 门） | **PASS（16/16 beats bit-exact，2026-10-31 16:44）**；历史 FAIL 53/256 nibble 为修复前基线（`docs/VISION_REFERENCE_ROUTE_RESULT_20261031.md`、主线 `build/spikformer/vision_ref_20261031/sim/p1_r2_small.log`） | not_run |
| `tb_embed_replay.v` | token/pos 表 + LIF | **PASS（16 beats）**（`docs/SPIKFORMER_BACKEND.md` §6） | not_run |
| `tb_mlp_replay.v` | fc1+LIF+fc2+残差 | **PASS（4 beats）**（同上） | not_run |
| `tb_head_replay.v` | PL 每通道 T*N 求和 | **PASS（32 通道）**（同上） | not_run |
| `tb_sequencer.v` | DEPTH=3 权重基址/末 block 归约/二次启动 | **PASS**（同上） | not_run |
| `tb_wrapper.v` | wrapper ID/busy/double-start/ack | **PASS**（done 路径待子引擎对接；`docs/SPIKFORMER_BACKEND.md` §5 已知债务 3） | not_run |
| `tb_block_replay.v` `tb_block_mon.v` | 单 block 整链（QKV+LIF+QK popcount+AV+proj+残差+MLP） | **PASS（16 beats）**，含 SPS token 变体（`docs/VISION_LINE_RETURN_20261031.md`） | not_run |
| `tb_nlp_small.v` | NLP small 回放（E32/H1/T2/D2） | **PASS**（`build/spikformer/nlp_48h_20261029_1915/reports/replay_small.json`，cycles 39,774） | not_run |
| `tb_nlp_multihead.v` | NLP 多头短回放（E64/H2，head 交换） | **PASS**（`.../reports/replay_multihead.json`） | not_run |
| `tb_nlp_e2e.v` | NLP 目标整链（E128/H4/seq32/T4/D2） | **PASS +NLP_NO_FORCE**（`.../reports/replay_e2e_5stage_r2_noforce.json`，cycles 7,084,772，0 mismatch）；**+force 背压窗口死锁为 TB 侧已知项**（`.../reports/rtl_freeze.json` open_tb_items） | not_run |

## 2.1 图像线新增 TB（2026-10-31 17:38 最终刷新收录，证据源而非自包含源码）

按 Lead 收尾清单，`verification/sim/spikformer/` 于 **2026-10-31 17:38 CST** 最终刷新（RTL 已冻结），
新增/更新以下图像线 TB（共 23 个 `.v`，与主线一致）：

| TB | 用途 | sha256（快照=主线） | 状态（主线证据） |
|---|---|---|---|
| `tb_sps_ref_small.v`（更新，含结构断言） | 参考式 SPS small（T=2） | `f15cd29cfd934110d5213e03e10b8533ad5e966c956c04ef666fb52a92478224` | pass（16/16 beats，2026-10-31 16:56） |
| `tb_sps_ref_small_t4.v` | 参考式 SPS small（T=4） | `65aa6b470bc1b09dad9c4c5dba47c3576ceb6ecb0351eccb061fe8a5b2aa67c6` | pass（16/16 beats） |
| `tb_sps_ref_small_h32_t4.v` | 参考式 SPS H32/T4 代理几何 | `5ea04930937b7429474fda6a764d45ab94fd1da7fd81618f4f25ccb395ba4173` | pass（256 beats repr） |
| `tb_sps_ref_small_h32_t4_tc2_8.v` | 参考式 SPS H32/T4/TC2=8 | `ffe162dc209e1c3bd37119af4d5d29848cd99ec58e52cd5df0a69bdbb589be46` | pass |
| `tb_sps_native.v` | 参考式 SPS 正式 native（3072 beats） | `8e9a7f8fd3e98e3626f9d48562ea7e972e5cd315b7b07cab1ef8a7cc52d47463` | pass（2026-10-31 17:01，hamming=0） |
| `tb_vis_native_chain.v` | native 整链（SPS→block→sums）骨架 | `8aead49c5a262cf9957314d0a3f2c6f4c255f66d126d364a5397729d4e15583e` | 编译通过；整链门 **not_run/in_progress** |

> **这些新增 TB 是“证据源”，不是快照自包含运行单元**：TB 内 `$readmemh` 引用主线绝对向量路径（`D:/dl_snn/build/...`），
> 运行需主线 `build/` 向量与 ModelSim；快照内**不重跑**。原 18 个 TB 的时点见下。

## 2.2 副本时点说明

- TB 与短例是 **2026-10-31 16:36 CST** 的复制时刻版本。
- **17:00 CST 只刷新了 RTL**（`rtl/ops/spiking/sps/`），TB 未同步；图像线在 16:36–18:30 期间新增/改动过
  `tb_sps_ref_small.v`、`tb_sps_native.v`、`tb_vis_native_chain.v` 等（图像线 WIP），**快照内不含这些后续改动**；
  因此“参考式 SPS P1 PASS”的主线证据所用 TB 可能比本副本新，**快照内不重跑 TB，也不据此声称快照内可复现**。
- RTL 刷新后，快照 `rtl/` = 当时主线状态（详见 `docs/PORTABILITY.md` §4）；NLP 证据绑定 2026-10-30 冻结版本。

## 3. 运行前提（为什么快照内没重跑）

1. **向量与路径**：TB 依赖 `build/spikformer/<case>/` 下的 `.hex`/`.npy`（golden 生成器 `scripts/spikformer/gen_*.py` 产出），
   这些大目录**不随包**；且 **8 个 TB 里共 70 处 `$readmemh` 仍写死 `D:/dl_snn/build/...`**（spec §4 只要求改写 `.py`，TB 未改）。
   在快照内回放需：先把这些前缀替换为快照根（或在那台机器上把 `build/` 向量放回 `D:/dl_snn/build/`），再重建向量（见 `RELEASE_NOTES.md` §5）。
2. **仿真器**：ModelSim 在 `D:/modelsim/win64/`（`vlog`/`vsim`，不在 PATH）；本快照不启动。
3. **NLP 回放驱动**：`scripts/spikformer/run_nlp_sim.py`（需要 build 树与仿真器），本快照不改其逻辑，仅保留副本。

## 4. 已知 TB 侧未结项（来自冻结记录，不属 RTL bug）

- `tb_nlp_e2e` force/release 背压窗口在 `in_cnt=199` 死锁（TB 侧；`+NLP_NO_FORCE` 通过，block 级背压仍有覆盖）。
- `tb_block_target_golden` weights readmem 范围 off-by-one（25,155 vs 25,156 words）→ 只修 TB。
