# 可移植性说明（D:/DL_FINAL/SNN 0.1.0）

本文件记录：发布副本内的**路径改写清单**、**与主线的 diff 摘要**、**NLP 冻结 RTL 哈希核对**。
- 源码/其他文件复制时刻：**2026-10-31 16:36 CST**
- **RTL 刷新时刻：2026-10-31 17:00 CST 与 17:38 CST（最终，RTL 已冻结）**；17:38 刷新包含图像线 P2 修复的 `control/spikformer_vis_native_top.sv`（`8ece767a…`）与 6 个新/改 TB。刷新后快照 `rtl/` 与主线 `rtl/`（除未收录的 `rtl/probe`）**无差异**
- 本文件内容以此后模型可见时钟为准；sha256 清单见 `evidence/COPY_SHA256.txt`（刷新后已重算）

## 1. 结论

- **采用完整相对根改写（spec §4 主方案），未启用回退方案。**
- 快照内 **16 个 .py 文件**已把硬编码 `D:/dl_snn` 替换为相对根解析；**源码内已无 `D:/dl_snn` 硬编码**（`grep -rn 'D:/dl_snn' --include=*.py .` 无命中）。
- 验证：快照根 compile **pass**；整目录拷到 `D:/_snn_reloc_check` 后同一 compile 的 **params sha256 集合与 `manifest_sha256` 完全一致**；生成的 Tcl 根指向新快照根。
- RTL 未改一个字节（见 §4 的哈希核对）；生成算法未改；语义等价。

## 2. 相对根规则

所有改写的文件统一使用：

```python
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[N]).resolve()
```

- 默认取文件自身位置的第 N 层父目录 = **快照根**；把整目录拷走即自动生效。
- 环境变量 `SNN_ROOT` 可强制覆盖（便于生成物落到别处或脚本被移出原位调试）。
- `N` 取值表（与 spec §4 一致）：

| 文件位置 | N | 例 |
|---|---|---|
| `tools/toolchain.py` | 1 | `tools/toolchain.py` → 快照根 |
| `tools/backends/spikformer/*.py` | 3 | `tools/backends/spikformer/manifest.py` → 快照根 |
| `tools/spikformer_train/*.py` | 2 | 主线原本已用相对根，无需改（见 §3.3） |
| `scripts/spikformer/*.py` | 2 | `scripts/spikformer/run_physical.py` → 快照根 |
| `verification/sim/spikformer/*.py` | 3 | `verification/sim/spikformer/test_short.py` → 快照根 |

## 3. 改动清单

完整 diff：`evidence/PORTABILITY_DIFFS.txt`（16 个文件，343 行）。下表“改动”= 旧常量/旧写法 → 新写法。

### 3.1 spec §4 指定的文件（9）

| 文件 | 改动 | ± |
|---|---|---|
| `tools/toolchain.py` | `cwd=str(pathlib.Path('D:/dl_snn'))` → `cwd=str(ROOT)`；`ROOT` = parents[1] | +4/-2 |
| `tools/backends/spikformer/codegen.py` | `HOST_ROOT = pathlib.Path('D:/dl_snn')` → `HOST_ROOT = ROOT`（parents[3]）；docstring 去掉主线绝对路径 | +4/-3 |
| `tools/backends/spikformer/manifest.py` | `pathlib.Path('D:/dl_snn/rtl/ops/spiking/source_manifest.json')` → `(ROOT / 'rtl/ops/spiking/source_manifest.json')`；`_sha('D:/dl_snn/' + s)` → `_sha(ROOT / s)`；`ROOT` = parents[3] | +4/-3 |
| `tools/spikformer_train/check_ready.py` | `ROOT = pathlib.Path('D:/dl_snn')` → parents[2] + `SNN_ROOT` 覆盖 | +2/-2 |
| `scripts/spikformer/run_physical.py` | `ROOT = pathlib.Path('D:/dl_snn')` → parents[2] + 覆盖 | +2/-1 |
| `scripts/spikformer/run_ip_pack.py` | 同上 | +2/-2 |
| `verification/sim/spikformer/test_short.py` | `sys.path.insert(0, 'D:/dl_snn/tools/backends/spikformer')` → `sys.path.insert(0, str(ROOT / 'tools/backends/spikformer'))`（parents[3]） | +3/-2 |
| `verification/sim/spikformer/test_runtime.py` | `sys.path.insert(0, 'D:/dl_snn/runtime/spikformer')` → `str(ROOT / 'runtime/spikformer')`（parents[3]） | +3/-2 |
| `scripts/spikformer/run_nlp_sim.py` | **无需改动**：主线已用 `REPO = dirname(dirname(dirname(abspath(__file__))))` | 0 |

### 3.2 额外改写的文件（7，spec §4 清单外，避免“半改半留”）

这些 golden 生成脚本原先把输出目录写成主线 `build/`，若保留会在快照内运行时**写到主线**，因此一并改为 `ROOT / 'build/...'`：

| 文件 | 改动 | ± |
|---|---|---|
| `scripts/spikformer/gen_block_replay.py` | `o = 'D:/dl_snn/build/spikformer/block_replay/'` → `str(ROOT / 'build/spikformer/block_replay/')`；`np.save(...)` 同类；补 `import os, pathlib` | +5/-2 |
| `scripts/spikformer/gen_embed_replay.py` | 输出目录 → ROOT | +4/-1 |
| `scripts/spikformer/gen_head_replay.py` | 输出目录 → ROOT | +4/-1 |
| `scripts/spikformer/gen_mlp_replay.py` | 输出目录 → ROOT；补 import | +4/-1 |
| `scripts/spikformer/gen_sps_replay.py` | 输出目录 → ROOT；补 import | +4/-1 |
| `scripts/spikformer/gen_vis_sums.py` | 输入金标/输出目录 → ROOT；补 import | +5/-2 |
| `scripts/spikformer/gen_vis_tok.py` | 输入/输出路径 → ROOT；补 import | +4/-2 |

> 这些脚本的输出目录（`build/spikformer/<case>/`）**不随包**；在快照内重放需先建目录（见 `RELEASE_NOTES.md` §5 重建命令）。

### 3.3 明确未改的

| 文件/项 | 原因 |
|---|---|
| `tools/spikformer_train/{train_nlp,export_integer,prepare_corpus,verify_export,test_integer_ref}.py` | 主线原本已用 `ROOT = pathlib.Path(__file__).resolve().parents[2]`（无 `SNN_ROOT` 覆盖），已可移植，不改 |
| `scripts/spikformer/hash_upstream.py` | 依赖上游树 `D:/dl_v0.1/FPGA_SpikingTransformer-main`（不随包）；保留原样，文档标注只能在上游机器使用 |
| `verification/sim/spikformer/*.v`（8 个 TB） | spec §4 只要求改写 `.py`；TB 里的 `$readmemh("D:/dl_snn/build/...` 共 **70 处**（tb_block_replay 14、tb_vis_replay 18、tb_vis_sps 17、tb_sps_ref_small 5、tb_sps_replay 5、tb_mlp_replay 5、tb_embed_replay 4、tb_head_replay 2）**未改**：快照内不重跑 TB（无 `build/` 向量与仿真器），因此不影响本次验收；若要在此快照跑 TB，需先把 `D:/dl_snn` 前缀替换为本快照根（或在那台机器上把 `build/` 向量放到同名路径） |
| `rtl/operator_library.json` `rtl/ops/spiking/source_manifest.json` `UPSTREAM_MAP.json` | **数据文件不改**：保留主线 `root` 字段作为来源标注，代码不读该字段；改它会破坏与冻结哈希/上游映射的可比性 |
| RTL（`rtl/ops/spiking/**`） | spec §4 明确不改 RTL；快照 RTL 为 16:36 复制时刻的字节拷贝 |

## 4. NLP 冻结 RTL 哈希核对（spec §6.4）

基准：`D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json`
（schema `spikformer-nlp-48h-rtl-freeze-1`，frozen_at `2026-10-29T22:26:00+08:00`，18 个文件）。
比对对象：快照 `rtl/ops/spiking/**`（2026-10-31 16:36 复制时刻）。

**结果：16/18 SAME，2/18 DIFF，0 MISSING。**（16:36 复制后核对；17:00 RTL 刷新后复核结果不变）

> **快照 RTL = 当前主线 superset**：源清单 28 份（冻结时 19 份），多出 9 份图像线新增的 SPS/vis 模块；
> 冻结的 18 文件中 16 个与快照字节一致，2 个为图像线改动过的 SPS 文件。
> **NLP 证据（回放/OOC/IP）绑定 2026-10-30 冻结版本**（`rtl_freeze.json`），不是快照当前字节；本包不重跑物理门。

| # | 文件 | 结果 | 冻结值 / 快照值 |
|---|---|---|---|
| 1 | `attention/spiking_attention_pe.sv` | SAME | 44325e51… |
| 2 | `attention/spiking_linemac_pe.sv` | SAME | 1ee83f17… |
| 3 | `attention/spiking_popcount.sv` | SAME | 7b47a053… |
| 4 | `control/spikformer_block_sequencer.sv` | SAME | 37778a38… |
| 5 | `control/spikformer_nlp_top.sv` | SAME | 3802aacc… |
| 6 | `control/spiking_block_top.sv` | SAME | 841df601… |
| 7 | `embed/spiking_embed_top.sv` | SAME | 958fd8cb… |
| 8 | `lif/spiking_lif_cell.sv` | SAME | 18427f9d… |
| 9 | `lif/spiking_lif_group.sv` | SAME | 5f32f9fd… |
| 10 | `linear/spiking_systolic_array.sv` | SAME | 1039aa12… |
| 11 | `linear/spiking_systolic_pe.sv` | SAME | 16589ff7… |
| 12 | `mlp/spiking_mlp_controller.sv` | SAME | 59405bfb… |
| 13 | `reduce/spiking_head_reduce.sv` | SAME | 1d295115… |
| 14 | `sps/spiking_adder_tree.sv` | SAME | 1c7f14ea… |
| 15 | `sps/spiking_maxpool2x2.sv` | SAME | a132e0fc… |
| 16 | `wrappers/spikformer_platform_wrapper.sv` | SAME | ed02c6b1… |
| **17** | **`sps/spiking_eyeriss_pe.sv`** | **DIFF** | 冻结 `1f1813ce3a0c5cf31eb4bcdf65eab20e2e99ce54139fe0aed35ed5ce4dee536c` / 快照 `0c64d70d69e9a86a0cf4b2bdfd1f7f2c2b2ec11e11eb5820137f84323274e468` |
| **18** | **`sps/spiking_sps_top.sv`** | **DIFF** | 冻结 `484e735948b361d0361d273556a7162bc4d3360b7869a4571f826d8d390ede60` / 快照 `84929737178de4218d1987e7df886b79135b360b85798100862beeeb3472917f` |

完整 sha256：`evidence/COPY_SHA256.txt`（快照源码/文档/配置副本，**排除 evidence/ 与 build/**；17:38 最终刷新后重算，152 行）。

### 4.1 这 2 个 DIFF 是什么（如实记录，不掩盖）

- 两个文件都在 **SPS（图像 patch embed）目录**，属**图像线参考式 SPS 改造**（`docs/VISION_REFERENCE_ROUTE_RESULT_20261031.md` §3：新增/修改 `spiking_eyeriss_pe.sv`、`spiking_sps_top.sv` 等，修 pad+shift 丢最后一列 tap 的真 bug）。
- 图像线在收拢期间**仍在并行修改** `D:/dl_snn`；快照以**复制/刷新时刻**的字节为准。**17:00 + 17:38 两次刷新后，快照 `rtl/` 与主线 `rtl/`（除 `rtl/probe`，按规范未收录）无差异**：
  - `sps/spiking_sps_ref_ctrl.sv` = `728337f10bff4e7d3f3c6a2ec0ea060446b5bc3a07e72182a4ecfeae22c02244`（16:56 修复版，P1 四配置 PASS）
  - `control/spikformer_vis_native_top.sv` = `8ece767a61823b062bfae8c9c74164d8ec9da0efdc53fcc092e34d1993ad5ad0`（17:22 图像线 P2 sums 累加修复，vision-only）
  - 两者**均不在** 18 文件冻结集内；NLP 功能 RTL 16 个哈希仍与冻结值 SAME
- **NLP datapath 不实例化 SPS**：NLP 目标顶层 `spikformer_nlp_top` 只例化 `spiking_embed_top` + `spiking_block_top`（`rtl/ops/spiking/control/spikformer_nlp_top.sv:125,140`）；SPS 文件只出现在 `source_manifest.json` 的编译源清单里（28 份中的 9 份 SPS/vis），**不进入 NLP 的功能路径**。因此 16 个 NLP 功能 RTL 哈希全部 SAME，NLP 回放/OOC/IP 证据对应的 RTL 未被改动。
- **但必须注意**：18 文件冻结集是“整份 NLP 冻结切片”的哈希锚点。**在快照内新跑 synth/OOC 不会复现 2026-10-30 的冻结切片**（源清单里有 2 个文件已变），所以快照内**没有、也不宣称**重跑物理门。要复现冻结 OOC/IP 证据，必须用冻结哈希对应的历史 RTL 版本（主线 `build/.../ip/src/` 与 `evidence/nlp/final/ip/src/` 保留了打包时的同源副本）。
- **给 Lead 的提示**：若 18 文件哈希需要“全 SAME”，只能二选一：①以图像线新 RTL 重新冻结并重跑证据（不可行，需 Vivado 且改变结论）；②声明“NLP 冻结锚点 16/18 SAME，2 个 SPS 文件被图像线并行改动，NLP datapath 不受影响”。本快照采用 ②，并在此明示。

### 4.2 冻结哈希核对命令（可复现）

```bash
E:/miniconda/python.exe -B - <<'PY'
import json, hashlib, pathlib
frz = json.load(open(r'D:/dl_snn/build/spikformer/nlp_48h_20261029_1915/reports/rtl_freeze.json'))['files']
SNAP = pathlib.Path(r'D:/DL_FINAL/SNN')
for rel, want in sorted(frz.items()):
    got = hashlib.sha256((SNAP / rel).read_bytes()).hexdigest()
    print(('SAME' if got == want else 'DIFF'), rel)
PY
```

## 5. 语义等价性证明（“同一根下生成物一致”）

改写只动“根目录如何解析”，不动生成算法、参数、RTL。六项证据：

| 证据 | 结果 |
|---|---|
| ① 快照根直接 compile（fixture seed=1，**17:38 最终刷新后重跑**） | `kind=development_fixture`，28 个 params blob；`manifest.json` 文件 sha256 `c8a45b94761b03eafa2724cfccdc6e892b255213f4a3715f7ceb1ddfa3023738`，内部 `manifest_sha256` `835ad38dc06f0546bc08612bf052f085edec7f47b34115cfe5079c7b221381c4` |
| ② 整目录拷到 `D:/_snn_reloc_check` 后同一 compile（17:38 最终刷新后重测） | `params/*.bin` 的 sha256 排序集合**逐行相同**（集合摘要 `000b0b5ba6fe53efa897dd8cf4b9ff4b7f4ecfb31516a60698b845599ee08dc4`，16:50/17:00/17:38 三个时点均相同）；`manifest.json` 文件 sha256 相同（`c8a45b94…`）、内部 `manifest_sha256` 相同；`params/rtl_sha256/structure/numeric/schedule` 字段全等 |
| ③ 生成器确实跟随新根 | 快照生成的 `generated/spikformer_ooc.tcl` 首行 `set snn_root {D:/DL_FINAL/SNN}`；迁移目录生成的是 `set snn_root {D:/_snn_reloc_check}`（同一份代码，不同根） |
| ④ manifest 自校验 | `manifest.rtl_sha256` 的 28 条与快照当前 RTL 字节**全部一致**（0 mismatch），说明 `manifest.py` 的 `ROOT / s` 解析正确 |
| ⑤ 真实权重编译等价 | 快照 `build/_quickstart_real`（54 params）与主线最终冻结 run `final/nlp_real_5stage_r4_holdfix` 的 **params 哈希集合逐行相同**（`315171a2…`，刷新前后不变），`manifest.params` 元数据、`schedule`、`config`（除 checkpoint 记录路径）、`derived`（除新增 `.sps` 键）全等；`rtl_sha256` 仅 SPS/vis 11 项不同（图像线文件）。快照 manifest 文件 sha256 `b71e3f17373beeb3e66e85a462769cb9a16b172db9ad5db71a43d56c29a8a6a4`，内部 `manifest_sha256` `f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e` |
| ⑥ golden 生成器自测（SNN_ROOT 重定向） | 7 个 `gen_*` 生成器在 `SNN_ROOT=D:/_snn_gentest` 下全部跑通（输出 48 个向量文件）；`embed/head/mlp/sps_replay` 产物与主线 `build/` 副本**逐字节一致**；`block_replay` 在 `TOKSRC=<sps_replay>/exp.hex`（与主线同条件）下 30/30 文件**逐字节一致**（临时目录已删） |

迁移测试目录 `D:/_snn_reloc_check` **已删除**。

## 6. 与主线的 diff 摘要（代表样例）

```diff
--- tools/backends/spikformer/manifest.py
 import hashlib, json, os, pathlib
+ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()
 def _sha(p):
-    srcs = json.loads(pathlib.Path('D:/dl_snn/rtl/ops/spiking/source_manifest.json').read_text())['sources']
+    srcs = json.loads((ROOT / 'rtl/ops/spiking/source_manifest.json').read_text())['sources']
-        'rtl_sha256': {s: _sha('D:/dl_snn/' + s) for s in srcs},
+        'rtl_sha256': {s: _sha(ROOT / s) for s in srcs},
```

```diff
--- verification/sim/spikformer/test_short.py
 import json, os, pathlib, sys
+ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()
-sys.path.insert(0, 'D:/dl_snn/tools/backends/spikformer')
+sys.path.insert(0, str(ROOT / 'tools/backends/spikformer'))
```

其余 14 个文件的 diff 见 `evidence/PORTABILITY_DIFFS.txt`；全部为同类“常量 → `ROOT` 表达式”替换（`codegen.py` 另含 docstring 文案）。

## 7. 回退方案（未启用）

spec §4 的回退方案（保留原常量 + 声明“源码根绑定 D:/dl_snn”）**未启用**：改写后编译、迁移一致性、短例全部通过，
不需要回退。若未来某文件被移到快照外运行，用 `SNN_ROOT` 指回快照根即可。

## 8. 已知限制

1. 迁移一致性只证明“生成侧可移植”；物理门（synth/OOC/bit）与 TB 回放依赖 Vivado/ModelSim（不随包），本快照未重跑。
2. 生成器输出的 Tcl 记录**生成时刻的绝对根**，拷到新机器后应重新 compile，而不是手改 Tcl。
3. `scripts/spikformer/hash_upstream.py` 与 `tools/spikformer_train/*` 里对上游/语料的绝对路径假定仍指向主线环境（上游不随包），见 §3.3。
4. **TB（`.v`）未做路径改写**：8 个 TB 共 70 处 `$readmemh` 仍指向 `D:/dl_snn/build/...`（spec §4 范围外）。快照内不重跑 TB，所以不影响验收；需要回放时按 §3.3 替换前缀。
