# board/cnn

CNN 库的板端交付包（历史 r4 固定样本板测）。

| 路径 | 内容 |
|---|---|
| `fpga/platform_top.bit` / `platform_top.hwh` | 历史 r4 平台位流（xczu15eg-ffvb1156-2-i） |
| `MANIFEST.json` | 位流/实现/板测记录的身份信息 |
| `runtime/` | DDR 初始化镜像、期望 head、测试图与 golden |
| `scripts/board_run_a.py` | 板端运行脚本 |

## 运行

1. 将 `fpga/`、`runtime/`、`scripts/` 拷到板卡同一目录；
2. 加载 `platform_top.bit` + `.hwh`；
3. 运行 `python3 board_run_a.py`，核对 6 个 head 与 golden 一致。

> 记录：历史固定样本板测 0.2395 s；16/8 并行度平台端到端 51.4 ms（PS 20 + PL 31），
> 两者不是同一位流。CNN 主线四案例的综合/时序为 not_run。
