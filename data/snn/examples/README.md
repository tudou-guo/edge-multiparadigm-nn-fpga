# 案例与状态表（examples/）

本快照收录两类 Spikformer 案例：**NLP 情感分类**（演示主线，已闭环）与**图像分类**（partial，仅收录配置与工具链证据，不是演示、无精度）。状态口径：`pass / fail / partial / not_run`，每个 PASS 指证据路径；完整门表见 [docs/PLATFORM_AND_STATUS.md](../docs/PLATFORM_AND_STATUS.md)，图像线最终措辞以 Lead 定稿的 docs/DELIVERY_SUMMARY.md 为准。

## 1. 案例状态

| 案例 | 配置 | 数据/权重 | 状态 | 证据 |
|---|---|---|---|---|
| **NLP 情感分类**（nlp_sentiment_b1） | `spikformer/config/nlp_sentiment_b1.json` | ChnSentiCorp 真实语料训练的整数 checkpoint `nlp_sentiment_b1/ckpt_real_real50.npz`（sha256 `e6a3a411…448a1`） | **pass（演示主线）**：编译/参数化/实模型资产/math/4 组回放 0 mismatch/IP/5ns routed OOC 均 pass；平台 r7 bit+HWH 与全片时序已生成（平台集成其余项 partial）；板端 3/3 match 为主会话记录（快照 not_verified，不宣称板测 pass） | `docs/DEMO_NLP.md`；`docs/PLATFORM_AND_STATUS.md` §1；`evidence/INDEX.md` |
| 图像 native 单 block（原生 32×32） | `spikformer/config/model_native_b1.json` | 无真实 checkpoint（fixture 参数；上游仓库不带 CIFAR-10 权重） | **partial**：工具链 compile pass；行为级回放 pass（小几何）；参考路线 P1 small 四配置 PASS + P1 native SPS 3072 beats PASS；native 资源门 **fail**（静态评估，非实测）；P2/P3/P4 见 `DELIVERY_SUMMARY.md` §3。**不得当演示、不得报精度** | `docs/PLATFORM_AND_STATUS.md` §4；`docs/VISION_CASE.md` |
| 图像 multiblock（T2/depth2） | `spikformer/config/model_multiblock_b.json` | fixture 参数 | **partial**：生成链 pass、双 block 回放 pass；P1/native/OOC/平台/板测见 `DELIVERY_SUMMARY.md` §3 | 同上 |

## 2. 目录说明

```text
examples/
├── README.md                       本文件
├── spikformer/                     配置族（来自主线 D:/dl_snn/examples/spikformer）
│   ├── README.md
│   └── config/
│       ├── model_native_b1.json    图像 native 单 block：32×32×3, E384/H12/HD32/MLP1536/T4/depth1/10 类
│       ├── model_multiblock_b.json 图像非默认样本：E192/H6/MLP768/T2/depth2
│       └── nlp_sentiment_b1.json   NLP：seq32/vocab4000/E128/H4/HD32/MLP512/T4/depth2/2 类
└── nlp_sentiment_b1/               NLP 演示资产（真实 checkpoint + 溯源）
    ├── ckpt_real_real50.npz        最终整数 checkpoint（936 KB）
    ├── ckpt_real_real50.assets.json 资产绑定（vocab/label_map/splits 哈希 + 标定阈值）
    ├── corpus_meta.json            语料溯源（来源 URL/MD5/SHA256/切分/词表/编码数组哈希）
    └── README.md                   语料→训练→导出→编译→回放→上板全链命令与数字来源
```

## 3. 注意

- `spikformer/README.md` 是主线原文（图像配置口径；其中 “Real weights (blocked until checkpoint arrives)” 指图像线：上游不带 CIFAR-10 checkpoint）。**NLP 已有真实 checkpoint**，见 `nlp_sentiment_b1/`。

## 4. 边界（诚实说明）

1. 快照只复制 NLP 演示所需的 3 个资产文件；训练状态 `best_state_real50.pt`、语料编码数组 `corpus_encoded.npz`（约 3 MB）、回放 golden 向量等大文件**不在快照内**，来源路径与哈希写在各 README 与 `corpus_meta.json` 中。
2. 图像案例只证明"工具链能生成"与"行为级小几何回放能过"；native 资源门 fail 是**静态评估**（寄存器阵列规模推导 + 综合跑挂），不是实测利用率；平台/板测 `not_run`。
3. NLP 的演示句是挑选样本，板端 3/3 match 是"sums 逐通道与 golden 相等"，**不是**准确率指标；且原始板端 JSON 未随包、板卡离线 → 快照记为 **not_verified**。准确率口径与数字冲突并列见 `docs/DEMO_NLP.md` §5。
