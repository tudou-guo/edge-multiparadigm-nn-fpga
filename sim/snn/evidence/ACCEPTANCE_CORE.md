# 验收记录（core，spec §6）— 2026-10-31

执行人：consolidate-core；执行根：`D:/DL_FINAL/SNN`；记录时间 **2026-10-31 16:36–17:38 CST**（含 17:00 与 17:38 两次 RTL/TB 刷新后复跑）。
环境：Windows + `E:/miniconda/python.exe`（Python 3.9.23，numpy 2.0.2，torch 2.8.0+cpu）。
本文件只记录**实际执行**的命令与结果；未执行的写 not_run。**刷新与清理记录见 §A7**。

## A1 快照根 compile（spec §6.1）

```bat
cd D:/DL_FINAL/SNN
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json --fixture-seed 1 ^
  --out-dir build/_quickstart_check
```

结果：**pass**（exit 0）

```text
manifest: build/_quickstart_check/manifest.json kind=development_fixture
```

| 项 | 值 |
|---|---|
| out-dir | `build/_quickstart_check/`（5.5 MB） |
| model_kind | `development_fixture`（fixture seed=1；真实权重见 A1b） |
| params | 28 个 `.bin`（+28 个 `.coe`） |
| params 集合摘要 | `sha256sum params/*.bin | sort -k2` 的 sha256 = **`000b0b5ba6fe53efa897dd8cf4b9ff4b7f4ecfb31516a60698b845599ee08dc4`** |
| manifest.json 文件 sha256 | **`c8a45b94761b03eafa2724cfccdc6e892b255213f4a3715f7ceb1ddfa3023738`**（17:38 最终刷新后重跑；17:00 为 `afd9671a…`、16:50 为 `6edb8f44…`） |
| manifest 内 `manifest_sha256` | `835ad38dc06f0546bc08612bf052f085edec7f47b34115cfe5079c7b221381c4`（17:00 为 `5d1415ff…`；差异来自 28 源 rtl_sha256 中的 `sps/spiking_sps_ref_ctrl.sv` 与 `control/spikformer_vis_native_top.sv`） |
| `rtl_sources` / `rtl_sha256` | 28 / 28，**与快照 RTL 字节全部一致（mismatch 0）** |
| 生成物 | `generated/{spikformer_config.vh, spikformer_nlp_target_top.sv, spikformer.xdc, spikformer_ooc.tcl, spikformer_project.tcl, sources.f, cache_key.txt}` |

确定性复跑：另起 `--out-dir build/_accept_recheck`（同 seed）→ params 集合摘要与上述**完全相同**，`manifest.json` 字节相同（后删除该目录）。

### A1b 真实权重 compile（checkpoint 已由 docs 子代理放入，16:57 补跑）

```bat
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz ^
  --out-dir build/_quickstart_real
```

checkpoint：`examples/nlp_sentiment_b1/ckpt_real_real50.npz`，
sha256 `e6a3a411e269f3b144ddb3efee77309c32ac9a53197c9071c224e1d0dd2448a1`（与主线最终 checkpoint 相同）。

结果：**pass**（exit 0）

```text
manifest: build/_quickstart_real/manifest.json kind=real_corpus
```

| 项 | 值 |
|---|---|
| model_kind | **`real_corpus`** |
| params | **54 个**（含 embed/pos 表、每 block 6 组 w/b/scale/thr、FP32 head） |
| params 集合摘要 | **`315171a23d3fa97f140cf7fb4fa958cb44350742352b17113b842e9cb797aca4`** |
| 与主线**最终冻结 run** 对比 | `final/nlp_real_5stage_r4_holdfix/params/*.bin` 哈希集合**逐行相同 → 字节级一致** |
| `manifest.params` 元数据 | 与最终冻结 manifest **完全相等** |
| `schedule` `config`（除 assets.checkpoint 记录路径） `derived`（除新增 `.sps` 键） | 与最终冻结 manifest **相等** |
| `rtl_sha256` 差异 | 仅 **SPS/vis 11 项**（2 个被图像线改动 + 9 个图像线新增）；**NLP 功能 RTL 全部相同** |
| `rtl_sources` | 28（快照 source_manifest v20261031.1）vs 冻结 run 19（当时清单） |
| manifest.json 文件 sha256 | `b71e3f17373beeb3e66e85a462769cb9a16b172db9ad5db71a43d56c29a8a6a4`（17:38 最终刷新后重跑；params 集合摘要在 16:57/17:00/17:38 三个时点均为 `315171a2…`） |
| `manifest_sha256` | `f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e`（17:00 为 `1047d0d0…`；与冻结 run `79b1e115…` 不同：源清单与 checkpoint 记录路径不同所致，见上两行） |
| `rtl_sha256` 自校验 | **mismatch 0**（28/28 与快照 RTL 一致） |

> 结论：**快照的真实权重编译产物 = 已闭环 NLP 案例的同一套模型资产**（params 字节级一致、调度与合同一致），
> 差异只在“图像线新增的源清单条目”和“checkpoint 路径字符串”。

## A2 可移植性证明：迁移目录 compile 一致性（spec §6.2）

```bash
# 快照整目录（排除 build/）拷到 D:/_snn_reloc_check，再跑同一 compile
tar -cf - --exclude=./build . | (cd /d/_snn_reloc_check && tar -xf -)
cd /d/_snn_reloc_check
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json --fixture-seed 1 --out-dir build/_quickstart_check
```

结果：**pass**

| 项 | 快照根 | 迁移目录 | 一致 |
|---|---|---|---|
| params 哈希集合摘要 | `000b0b5b…` | `000b0b5b…` | **是**（`diff` 逐行相同；刷新前后不变） |
| manifest.json 文件 sha256 | `c8a45b94…` | `c8a45b94…` | **是**（17:38 最终刷新后重测） |
| `manifest_sha256` 内字段 | — | — | **是** |
| manifest `params/rtl_sha256/structure/numeric/schedule` | — | — | **全等** |
| 生成 Tcl 根 | `set snn_root {D:/DL_FINAL/SNN}` | `set snn_root {D:/_snn_reloc_check}` | 跟随根（预期） |

迁移目录 `D:/_snn_reloc_check` **已删除**（验收后清理）。

## A3 纯 Python 自检（spec §6.3）

| 命令 | 结果 | 摘要 |
|---|---|---|
| `E:/miniconda/python.exe -B verification/sim/spikformer/test_short.py` | **pass**（exit 0） | 15 个 PASS，`FAILURES: 0`（LIF/popcount/残差/packing/生命周期/conv/gated/pool） |
| `E:/miniconda/python.exe -B tools/spikformer_train/test_integer_ref.py` | **pass**（exit 0） | `RESULT: 8/8 PASS -> PASS`（47 层整数 trace ×2 配置、LIF 边界、popcount=32/AV=1024、PS head bit 一致、STE） |
| `E:/miniconda/python.exe -B verification/sim/spikformer/test_runtime.py` | **fail（pre-existing，非本快照引入）** | 主线同错：`import run as R` → `runtime/spikformer/run.py` 包内相对导入失败；修导入后 `FakeMMIO` 无 memory reader，`read_sums` 拒绝返回占位。**测试逻辑未改**，仅 sys.path 改为相对根 |

## A4 NLP 冻结 RTL 哈希核对（spec §6.4）

基准：`D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json`
（schema `spikformer-nlp-48h-rtl-freeze-1`，frozen_at `2026-10-29T22:26:00+08:00`，18 文件）。
对象：快照 `rtl/ops/spiking/**`（2026-10-31 16:36 复制时刻）。

**结果：16 SAME / 2 DIFF / 0 MISSING。**

| 文件 | 结果 | 冻结值 | 快照值 |
|---|---|---|---|
| `sps/spiking_eyeriss_pe.sv` | **DIFF** | `1f1813ce3a0c5cf31eb4bcdf65eab20e2e99ce54139fe0aed35ed5ce4dee536c` | `0c64d70d69e9a86a0cf4b2bdfd1f7f2c2b2ec11e11eb5820137f84323274e468` |
| `sps/spiking_sps_top.sv` | **DIFF** | `484e735948b361d0361d273556a7162bc4d3360b7869a4571f826d8d390ede60` | `84929737178de4218d1987e7df886b79135b360b85798100862beeeb3472917f` |
| 其余 16 个 | SAME | — | — |

17:38 最终 RTL 刷新后复核：结果**不变**（16 SAME / 2 DIFF）。两次刷新更新的文件（`sps/spiking_sps_ref_ctrl.sv` 728337f1、`control/spikformer_vis_native_top.sv` 8ece767a）**均不在** 18 文件冻结集内；其对应主线证据绑定的仍是 2026-10-30 冻结哈希。
结论与影响（详见 `docs/PORTABILITY.md` §4）：2 个 DIFF 均为**图像线并行改动**的 SPS 文件；NLP datapath 不实例化 SPS
（`spikformer_nlp_top.sv` 只例化 `spiking_embed_top` + `spiking_block_top`），16 个 NLP 功能 RTL 哈希全部 SAME。
快照内**未重跑** synth/OOC（spec 禁止），因此不宣称复现冻结切片。

## A5 复制清单与联调

| 项 | 结果 |
|---|---|
| 快照文件 sha256 清单 | `evidence/COPY_SHA256.txt`（152 行 = 150 个文件 + 2 行注释；排除 `evidence/` 与 `build/`；17:38 最终刷新后重算） |
| 可移植性 diff | `evidence/PORTABILITY_DIFFS.txt`（16 文件，343 行） |
| Python 源码内 `D:/dl_snn` 硬编码 | **0**（`grep -rn 'D:/dl_snn' --include=*.py .` 无命中；数据文件 `rtl/operator_library.json` 的 `root` 字段按规范保留） |
| TB（`.v`）内 `D:/dl_snn` | **70 处（8 个 TB，未改，spec §4 范围外）**：tb_block_replay 14、tb_vis_replay 18、tb_vis_sps 17、tb_sps_ref_small 5、tb_sps_replay 5、tb_mlp_replay 5、tb_embed_replay 4、tb_head_replay 2；快照内不重跑 TB，故不影响验收（见 `docs/PORTABILITY.md` §3.3/§8） |
| 全部 .py 语法 | `python -B -m py_compile` **pass**（tools/scripts/verification/runtime 全部） |
| 入口 `--help` 冒烟 | `run_physical.py` / `run_ip_pack.py` / `check_contract.py` / `check_ready.py` 均 OK |

## A6 未执行（not_run，如实记录）

| 项 | 原因 |
|---|---|
| Vivado synth/impl/bitstream、OOC 重跑 | spec §6.5 禁止；不占 heavy lock |
| TB 仿真（23 个 `.v`，含 17:38 最终刷新的 6 个图像线 TB） | 需 `build/` 向量 + ModelSim；快照只带源码（TB 内含主线绝对向量路径，非自包含） |
| 板端运行 | 板卡离线；原始板端输出未随包（见 `RELEASE_NOTES.md` §3） |
| 平台工程重建 | r7 bit/HWH 直接收录，不重建 |

## A7 RTL 刷新与清理（2026-10-31 17:00 / 17:38 CST）

### A7.1 RTL 刷新（对齐当前主线）

- 刷新范围：`rtl/ops/spiking/sps/` 及 `rtl/ops/spiking` 下与主线不同的文件（`rtl/probe` 按规范不收录）。
- 实际变化（1 个文件）：

| 文件 | 16:36 复制版 | 17:00 刷新后（= 当前主线） |
|---|---|---|
| `rtl/ops/spiking/sps/spiking_sps_ref_ctrl.sv` | `74452587bab70934acbbb642f2399f59d17021eec64ec20f169579f8a3147f5d` | `728337f10bff4e7d3f3c6a2ec0ea060446b5bc3a07e72182a4ecfeae22c02244`（图像线 16:56 最终修复版，P1 三配置 PASS 的版本） |

- 刷新后 `diff -rq rtl 主线/rtl`：**无差异**（除未收录的 `rtl/probe`）。**快照 RTL = 当前主线 superset**；
  NLP 证据仍绑定 2026-10-30 冻结版本（`rtl_freeze.json`，16 SAME / 2 DIFF 不变）。

### A7.4 最终刷新（17:38 CST，RTL 已最终冻结）

Lead 确认 RTL 最终冻结后执行的最后一次刷新：

| 类别 | 刷新内容 | 关键 sha256 |
|---|---|---|
| RTL（1 个） | `rtl/ops/spiking/control/spikformer_vis_native_top.sv`（图像线 P2 sums 累加修复） | `8ece767a61823b062bfae8c9c74164d8ec9da0efdc53fcc092e34d1993ad5ad0` |
| TB 新增（5 个） | `tb_sps_native.v`、`tb_sps_ref_small_t4.v`、`tb_sps_ref_small_h32_t4.v`、`tb_sps_ref_small_h32_t4_tc2_8.v`、`tb_vis_native_chain.v` | tb_vis_native_chain=`8aead49c…`、tb_sps_native=`8e9a7f8f…` |
| TB 更新（1 个） | `tb_sps_ref_small.v`（加结构断言） | `f15cd29cfd934110d5213e03e10b8533ad5e966c956c04ef666fb52a92478224` |

- 刷新后：`diff -rq rtl 主线/rtl` **无差异**（除未收录的 `rtl/probe`）；`verification/sim/spikformer` 的 `.v` 与主线**无差异**；
  冻结 18 文件仍 **16 SAME / 2 DIFF**（2 个 SPS 文件为图像线历史改动；本次刷新的 2 个 RTL 均不在冻结集内）。
- `evidence/COPY_SHA256.txt` 重算：**152 行**（150 个文件 + 2 行注释）。

### A7.2 清单与复跑

| 项 | 结果 |
|---|---|
| `evidence/COPY_SHA256.txt` 重算 | **152 行**（最终刷新后；排除 `evidence/` 与 `build/`；含 `#` 注释 2 行） |
| compile（fixture，17:38 最终刷新后） | **pass**（params 集合摘要 `000b0b5b…` 三个时点不变；manifest 文件 sha `c8a45b94…`，内部 `manifest_sha256 835ad38d…`；rtl mismatch 0） |
| compile（真实权重，17:38 最终刷新后） | **pass**（54 params；params 集合摘要 `315171a2…` 三个时点不变；manifest 文件 sha `b71e3f17…`，内部 `manifest_sha256 f727e4c7…`；rtl mismatch 0） |
| 迁移一致性（刷新后重测） | **pass**（`D:/_snn_reloc_check`：params 摘要、manifest 文件 sha、`manifest_sha256` 全一致；Tcl `snn_root` 跟随新根；临时目录已删） |
| golden 生成器自测（`SNN_ROOT` 重定向） | **pass**：7 个 `gen_*` 全部跑通；embed/head/mlp/sps 产物与主线逐字节一致；block_replay 在 `TOKSRC=SPS-token` 下 30/30 逐字节一致（临时目录已删） |

### A7.3 清理

| 项 | 状态 |
|---|---|
| `evidence/_tmp_*.py` | **已清理为 0**（曾用临时脚本 `_tmp_freeze_check.py`、`_tmp_port*.py`、`_tmp_fixslash.py` 均已删除） |
| 快照内 `__pycache__` / `*.pyc` | **0**（已删，交付前复核） |
| `D:/_snn_reloc_check`、`D:/_snn_gentest`、`D:/_snn_gentest2` | **已删除** |
| `build/_accept_recheck`、`build/_pkg_check` | **已删除**（仅保留 `build/_quickstart_check`、`build/_quickstart_real` 两份验收产物） |

## 更新记录

- 2026-10-31 17:00：初版（A1–A5 完成；A1b 真实权重补跑）。
- 2026-10-31 17:05：A7 刷新/复跑/清理记录；A1/A1b/A2/A4 数字按 RTL 刷新后复跑更新。
- 2026-10-31 17:38：**最终刷新**（A7.4）：RTL 1 个（vis_native_top 8ece767a）+ TB 6 个；重算 COPY_SHA256（152 行）；复跑 compile fixture/real、迁移一致性、test_short、test_integer_ref 全部 **pass**；params 集合摘要不变（`000b0b5b…` / `315171a2…`）；冻结 18 文件仍 16 SAME / 2 DIFF。此后不再改快照。
