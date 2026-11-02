# data/

测试数据与参考结果。三个库的数据分布：

- cnn/：ONNX 模型、输入样本、期望输出（大文件按索引说明，不随包复制）；
- snn/：examples/（3 份配置 + 整数 checkpoint + 测试向量）；
- lnn/：export/（int16 Q12 权重、Q24 偏置、两张 1025 点 LUT、ROM 初始化头文件、整数参考与 manifest）。
