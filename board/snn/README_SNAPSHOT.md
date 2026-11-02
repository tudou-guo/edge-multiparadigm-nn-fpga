# 快照补充说明（D:/DL_FINAL/SNN/board）

本目录是 `D:/spk_platform/board` 的只读副本（2026-10-31 收拢，原包 2026-10-30 23:18 生成）。
发布快照内新增了两个说明/校验文件，`README_BOARD.md` 等原包文件逐字节未改：

| 文件 | 说明 |
|---|---|
| `MANIFEST_DEMO_20261031.sha256` | 4 个 demo 资产（10-31 11:00 后加入原包、**不在原 MANIFEST.sha256 内**）的哈希 |
| `README_SNAPSHOT.md` | 本说明 |

**板端使用前必读的一点**：r7-sumsfix bit 的 AXI-Lite SUMS 读窗口仍有 +64 字偏移——读字地址 i 返回 `sums[(i+64)%128]`。

- `demo_nlp.py` 已用 `ROT=64` 做补偿（正确）；
- `snn_platform_run.py` 的 `read_sums()` **没有**旋转，原样在 r7 bit 上跑会 64 通道错位、与 golden 比对失败；
- 现场两种做法（补丁读法 A / 按 handoff §5.1 重出 bit 读法 B）见快照 `docs/DEMO_NLP.md` §3。

运行步骤、预期输出与已知限制见 `README_BOARD.md`（§3 运行、§4 失败排查、§6 已知限制）以及快照 `docs/DEMO_NLP.md`。
