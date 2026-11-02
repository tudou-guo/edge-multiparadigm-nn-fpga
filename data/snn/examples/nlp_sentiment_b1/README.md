# NLP 情感分类案例 nlp_sentiment_b1 — 资产与全链复现

真实语料（ChnSentiCorp）训练 → 整数导出 → 编译/参数包 → 冻结 RTL 回放 → PYNQ 平台 r7 上板。本目录只放**最小资产**；大文件（语料编码数组、torch 训练状态、回放同源 golden 向量）不随快照复制，来源路径写在下面。

## 1. 本目录内容与哈希

| 文件 | 大小 | sha256 | 来源（主线） | 说明 |
|---|---:|---|---|---|
| `ckpt_real_real50.npz` | 935,756 B | `e6a3a411e269f3b144ddb3efee77309c32ac9a53197c9071c224e1d0dd2448a1` | `build/spikformer/nlp_48h_20261029_1915/train/ckpt_real_real50.npz` | 最终整数 checkpoint（contract `export_abi` 全字段，model_kind=real_corpus） |
| `ckpt_real_real50.assets.json` | 2,423 B | `abaa6c1f40ffbd94b8354c5a1efb00976a214053d0f6c640b79b3cc545386ace` | 同上目录 | 资产绑定：vocab/label_map/splits 哈希 + 64 样本整数域标定阈值 |
| `corpus_meta.json` | 3,276 B | `00645501d74090bf1c7bc2b3e8e7e51a411efbabfecb222102dc1816bd340b97` | `build/spikformer/nlp_48h_20261029_1915/corpus/corpus_meta.json` | 语料溯源：URL/MD5/SHA256、三份 tsv 哈希、固定 holdout、词表、编码数组哈希 |

主线同目录还有（**未复制**）：`best_state_real50.pt`（3.6 MB，torch 训练状态）、`corpus/corpus_encoded.npz`（3.0 MB，编码数组）、`replay/case_real/`（回放向量）、`nlp_real_compile/`（54 blob 参数包）。

## 2. 模型与数值合同（摘要）

| 项 | 值 |
|---|---|
| 配置 | `../spikformer/config/nlp_sentiment_b1.json`：seq_len 32、vocab 4000、E128 / H4 / HD32 / MLP512 / depth2 / T4 / 2 类 |
| 前端 | 按字切词（`runtime/spikformer/tokenize.py`），保留前 32 字符，右 pad=0，unk=1；etab+ptab 共享量化域 → LIF(thr_e) |
| LIF | `mem=floor((pre+delta)/2)`（33 bit 符号扩展后算术右移），`valid && mem>=0 && mem>=thr` 发放并硬复位 0，thr>=1；13 个 LIF 阈值固定 |
| QK/AV | QK 位与 popcount（不除 head_dim）；AV 不归一（最大 32×32=1024）；无 softmax |
| 输出 | PL 每通道 T*N 整数和 → PS 除 128 → FP32 Linear(2×128)+argmax（`norm=ps_div`） |
| 出口 checkpoint | 13 阈值：thr_e=26；block0 q/k/v/proj/fc1/fc2=128/100/92/546/57/100；block1=190/190/203/1168/96/64（来源 `export_final_real50.json`） |

数值语义权威：`build/spikformer/nlp_48h_20261029_1915/contract.json`（主线）；独立参考实现 `tools/backends/spikformer/integer_ref.py`。

## 3. 全链命令（按执行顺序）

> 步骤 1–3 需要主线目录 `D:/dl_snn`（语料/训练状态/导出脚本齐全）；步骤 4–5 可在本快照根执行（脚本已改为相对根解析）；步骤 6 在板上执行。窗口命令用 Windows 路径。

### 3.1 语料（已执行，provenance 见 corpus_meta.json）

```bash
cd /d/dl_snn
# 官方 ChnSentiCorp.zip（MD5 7ef61b08ad10fbddf2ba97613f071561 校验通过）
E:/miniconda/python.exe -B tools/spikformer_train/prepare_corpus.py \
  --zip build/spikformer/nlp_48h_20261029_1915/corpus/ChnSentiCorp.zip \
  --out build/spikformer/nlp_48h_20261029_1915/corpus --vocab-size 4000 --seq-len 32
```

- train.tsv 9600 行（划出固定分层 holdout 960 条后 train=8640）/ dev.tsv 1200（带标签）/ test.tsv 1200（**无标签，不用于 accuracy**）。
- 词表 4000 只由 train 构建；编码一次成 `corpus_encoded.npz`（pad=0/unk=1，截断=保留前 32 字）。

### 3.2 真实训练（已执行，冻结命令）

```bash
cd /d/dl_snn
bash build/spikformer/nlp_48h_20261029_1915/train/train_cmd.txt
# 等价展开（heavy lock 内）：pytorch_clean python -B tools/spikformer_train/train_nlp.py
#   --config examples/spikformer/config/nlp_sentiment_b1.json --data .../corpus_encoded.npz
#   --run-dir build/spikformer/nlp_48h_20261029_1915 --stage train --tag real50 \
#   --epochs 5 --lr 0.1 --head-lr 2e-3 --recalibrate 0 --patience 2
```

结果（torch 整数等价前向，5 epoch，约 16 s/epoch）：dev 61.6→67.7→74.5→77.25→**78.33%**（best ep4），holdout **76.77%**，train 90.54%；证据 `reports/train_real50.json`。

### 3.3 导出整数 checkpoint（已执行）

```bash
cd /d/dl_snn
E:/miniconda/envs/pytorch_clean/python.exe -B tools/spikformer_train/train_nlp.py \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --data build/spikformer/nlp_48h_20261029_1915/corpus/corpus_encoded.npz \
  --run-dir build/spikformer/nlp_48h_20261029_1915 --stage export --tag real50 \
  --state build/spikformer/nlp_48h_20261029_1915/train/best_state_real50.pt

E:/miniconda/python.exe -B tools/spikformer_train/export_integer.py \
  --candidate build/spikformer/nlp_48h_20261029_1915/train/ckpt_candidate_real50.npz \
  --tag real50 --run-dir build/spikformer/nlp_48h_20261029_1915 \
  --config examples/spikformer/config/nlp_sentiment_b1.json
```

产出 `train/ckpt_real_real50.npz`（即本目录副本，sha256 见 §1）。导出保真：torch 量化整数前向 vs `integer_ref` 在 2 条真实样本上 47 层逐一相等，PASS（`reports/train_realweight_fidelity_real50.json`）。

### 3.4 编译（本快照内可复跑）

```bash
cd <快照根>    # D:/DL_FINAL/SNN
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile \
  --config examples/spikformer/config/nlp_sentiment_b1.json \
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz \
  --out-dir build/nlp_sentiment_b1
```

期望：`manifest.json` 中 `model_kind=real_corpus`、54 个 param blob（`params/` 下 108 个 .bin/.coe 文件）、13 阈值/全部 scale/FP32 head 绑定。

实测（2026-10-31 17:41，快照 RTL 17:38 最终刷新后，快照根，exit 0）：`model_kind=real_corpus`、54 blob、`manifest.manifest_sha256=f727e4c7920990ab03c11d3b2f26d075e12f27cf30f72d7fb7938a58b2c0911e`（manifest 文件级 sha256 `b71e3f17…a6a4`）。主线同源产物 `train/nlp_real_compile/manifest.json` 的 manifest_sha256 为 `afc128700ac5cbf47e41ef942a9e09e7380111a617a1ee84354f2a54412c2c23`；**两者不同属预期**——manifest_sha256 绑定输出目录/config 路径与 `rtl_sha256` 字段（快照 17:38 刷新后 2 个图像线 RTL 文件哈希变化，17:00 旧读数 `1047d0d0…` 作废），模型语义与 params 集合一致。复核以 `model_kind` 与 blob 集合为准。

### 3.5 冻结 RTL 回放（ModelSim；需主线 build 目录里已生成的向量）

```bash
cd /d/dl_snn
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case small
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case e2e --plusarg +NLP_NO_FORCE
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case e2e          # 反压版
E:/miniconda/python.exe scripts/spikformer/run_nlp_sim.py --case real --plusarg +NLP_NO_FORCE
```

| 用例 | cycles | mismatch | 证据（主线 `build/.../reports/`） |
|---|---:|---:|---|
| small（E32/H1/T2/seq4/D2） | 39,774 | 0 | `replay_small_holdfix.json` |
| e2e +NLP_NO_FORCE（目标 E128/H4/T4/seq32/D2） | 7,084,772 | 0 | `replay_e2e_noforce_holdfix.json` |
| e2e 反压（14 输出/10 输入 stall 窗口） | 7,084,812 | 0 | `replay_e2e_bpforce_holdfix.json` |
| real 3 样本（dev#0/dev#1/holdout#0） | 10,627,157 | 0 | `replay_real_holdfix.json` |

说明：`case real` 的向量目录 `replay/case_real/` 不在快照内（大文件），需主线根执行；回放对比 embed/block-in/block-out 每个 beat + `o_sums[128]` + done，mismatch=0。

### 3.6 上板（PYNQ，板卡离线时 not_run）

```bash
# 板端（见 ../../board/README_BOARD.md 与 ../../docs/DEMO_NLP.md）
cd /home/xilinx/snn_board
python3 snn_platform_run.py --bit snn_platform.bit --vecdir vec/case_real --samples 3 \
    --json-out snn_board_result.json
```

**注意 1（r7 bit 的 sums 读法）**：r7-sumsfix bit 的 AXI-Lite SUMS 读窗口仍有 +64 字偏移，直接读 `SUMS+4*i` 得到的是 `sums[(i+64)%128]`；`snn_platform_run.py` 未做旋转（详见 `docs/DEMO_NLP.md` §3 与 §4）。带补偿的读法/重出 bit 的修法都已写明。

**注意 2（快照口径）**：板端 3/3 match 是主会话记录；原始板端 run JSON 未随包、板卡当前离线 → 快照记 `board = not_verified`（不宣称板测 pass），见 `docs/PLATFORM_AND_STATUS.md` §3。

## 4. 数字与来源一览（口径并列，不择一）

| 指标 | 数字 | 口径/来源 |
|---|---|---|
| integer_ref dev acc | **75.92%**（n=1200，majority 50.58%） | 整数域评测（准确率口径 of record）：`reports/export_final_real50.json` |
| integer_ref holdout acc | **75.62%**（n=960，majority 50.00%） | 同上（固定分层 holdout，索引有哈希） |
| integer_ref train subset | 83.98%（n=256） | 同上 |
| torch 整数等价 dev / holdout / train | **78.33% / 76.77% / 90.54%** | `reports/train_real50.json`（torch 前向，非整数参考） |
| "dev 74.7%（多数类 63.3%）" | 74.7% / 63.3% | 仅见 `docs/SPIKFORMER_VISION_HANDOFF_20261031.md` §1.3；在报告中**未找到**对应 JSON（未核实口径，可能来自早期运行） |
| 类别偏差 | class-1 recall 60.7% vs class-0 90.8%（dev） | `reports/export_final_real50.json` confusion [[551,56],[233,360]] |
| 公开 test.tsv | 无标签，不报 test accuracy | corpus_meta.json |

> 表中 `reports/…` 的主线前缀是 `build/spikformer/nlp_48h_20261029_1915/`；快照内副本在 `evidence/nlp/`（`export_final_real50.json`、`train_real50.json` 等，见 `evidence/INDEX.md` §2）。

**准确率继续以 integer_ref 为准（它是回放 golden truth）；torch 数字只作训练过程指标。演示 6 句是挑选样本，板端 3/3 match 是数值一致性，不是准确率。**

## 5. 复现边界

1. 训练/导出需要 torch 2.8 环境（`E:/miniconda/envs/pytorch_clean/python.exe`）与主线会话中记录的重型锁；快照只保证资产可读 + 编译入口可用。
2. `run_nlp_sim.py --case real` 需要主线生成的 `replay/case_real/`（本快照未复制）。
3. 板端复跑需板卡在线（`192.168.2.99`，PYNQ 2.7.0），当前离线；原始板端结果 JSON 未随包、本机尝试因缺 `pynq` 失败 → 快照记 `board = not_verified`（主会话记录 3/3 match 见 `docs/DEMO_NLP.md` §4）。
