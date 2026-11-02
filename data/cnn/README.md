# data/cnn

CNN 库的测试数据与参考结果（小文件随包，大权重/模型不复制）。

| 目录 | 内容 |
|---|---|
| `yolo/model/` | YOLOv8n-ReLU VOC20 的 graph/heads/inputs/export 报告 |
| `yolo/expected/` | 期望 head 输出与期望哈希 |
| `tinycnn/` | TinyCNN-28 MNIST 的编译清单、图规划、QDQ 报告 |
| `mobiledw/` | MobileDW-32 CIFAR-10 的编译清单、图规划、ModelSim 回放摘要、QDQ 报告 |

> 大文件（ONNX 权重、图片、DDR 镜像）在原工程 `DL/examples/` 下，按 README 索引取用；
> 本目录只保留能支撑"数值/哈希/编译结果"复核的 JSON。
