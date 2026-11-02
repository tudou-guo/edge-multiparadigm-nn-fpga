# 板端故障四元组（PYNQ overlay）

> 每条：现象 → 根因 → 处置 → 证据。前 4 条来自本机三块板卡的真实记录，其余为通用故障模式。

## T1 bit 与 hwh 不同源

- 现象：overlay 能加载，但 IP 行为异常/读回 ID 不符；或 PYNQ 报 hwh 与 bit 不匹配。
- 根因：bit 与 hwh 来自两次实现导出，或 XSA 内嵌 bitstream 与外部 .bit 不一致。
- 处置：按 C2 做配对哈希；不一致直接停，重新导出；三件套同源后再测。
- 证据：LNN 配对记录（XSA 内嵌 bit 与外部 .bit 逐字节一致、hwh 一致，见 `sim/lnn/evidence/ACCEPTANCE.md` §3）；CNN 部署包哈希见 `board/cnn/MANIFEST.json`。

## T2 overlay 加载找不到 hwh / same stem

- 现象：`Overlay("x.bit")` 报 hwh 缺失，或提示 must have the same stem。
- 根因：.hwh 与 .bit 不同目录、不同名，或只拷了 .bit。
- 处置：同目录同 stem 原样放置；驱动里显式传 bit 路径，让 PYNQ 自动找同名 hwh。
- 证据：LNN 驱动 `discover_paths` 的 same-stem 检查（`board/lnn/lnn_cfc_run.py`）。

## T3 ip_dict 里没有目标 IP

- 现象：加载成功但 `ol.ip_dict` 无目标实例，驱动 KeyError。
- 根因：bit/hwh 不是本次构建；IP 在 BD 层级下名字不同；或综合时被优化掉。
- 处置：打印 `ip_dict` 全量名字核对层级（如 `<top>_0`）；核对构建报告中的 IP 清单。
- 证据：三块板卡驱动均先打印 `ip_dict` 再取实例。

## T4 64 位地址被截断

- 现象：读写"成功"但无反应，或驱动直接抛地址越界。
- 根因：PL 地址窗是 64 位（如 `0x480000000`），代码用了 32 位掩码/常量。
- 处置：地址常量保持 64 位；写寄存器时只掩低 32 位，做基址/偏移分离。
- 证据：LNN 驱动显式断言 `CSR_BASE` 必须是 64 位 `0x0480000000`（`lnn_cfc_run.py` L65/L123）。

## T5 DMA 超时 / 数据不可见

- 现象：START 后一直 BUSY，或 DONE 但读回数据是旧值/全零。
- 根因：地址窗、对齐、长度、跨 4 KiB 边界配置错；非相干口未 `flush/invalidate`；PL 侧 DMA 与 PS 缓存看到不同副本。
- 处置：先打印 buffer `physical_address` 与长度、比对 hwh 地址窗；写后 flush、读前 invalidate；把"预估最坏耗时×2"设为超时并记录。
- 证据：通用做法；本队三块板均采用 flush/invalidate 与显式超时（见各 `board/*/*_run.py`）。

## T6 符号扩展/字节序错（最像"算法错"的错）

- 现象：数值大面积接近但不相等，或首批数据全错、后续正常。
- 根因：硬件是 int16 补码/INT8 signed，host 按 uint 读取；或字节序/打包顺序不一致。
- 处置：按契约还原符号（int16：`v | 0xFFFF0000 if v & 0x8000`）；读回先打印原始字再解释。
- 证据：CNN 部署镜像 u8→s8 重编码修复（`board/cnn/runtime/decode/preprocess.py`、`board/cnn/MANIFEST.json` 的 `input_recode_fix`）。

## T7 BUSY 位观察不到（采样窗口错位）

- 现象：测试报告 `status_busy_observations=0`，误判"硬件忙拒绝失效"。
- 根因：轮询间隔大于忙窗口，START 与 STATUS 读取不在同一紧邻时序；或测试线程被 gate 阻塞。
- 处置：写 START 后立即读 STATUS（同线程、记录时间戳）；必要时加长任务制造可观测忙窗口；重试逻辑与 DONE 判定分开。
- 证据：`[工作区] D:/dl_v1.0/docs/F0_BUSY_R2_ROOT_CAUSE_R3_FIX_20261017.md`。

## T8 行为仿真通过、板端错误码

- 现象：XSim/ModelSim 功能仿真过，板端返回 `ERR_OPCODE` 类错误。
- 根因：综合把 RAM 副本写端口优化成常量（行为仿真看不到）。
- 处置：综合后网表仿真/端口断言；不能只凭行为仿真判功能正确。
- 证据：`[工作区] D:/dl_v1.0/docs/PYNQ_ERR_OPCODE_ROOT_CAUSE_FIX_20261017.md`。

## T9 `.pyc` 污染导致哈希校验失败

- 现象：`manifest hash check failed`，或驱动拒绝启动。
- 根因：首次运行在期望值目录生成 `__pycache__/*.pyc`，与清单中的哈希不符。
- 处置：运行前 `rm -rf __pycache__`，命令加 `-B`；清单只覆盖输入文件。
- 证据：LNN 驱动 pycache 陷阱（`board/lnn/lnn_cfc_run.py`、`src/lnn/docs/USER_GUIDE.md`）。

## T10 板卡离线/无法连接

- 现象：ssh 超时、Jupyter 无响应。
- 根因：板卡未上电、IP 变化、网络隔离。
- 处置：串口/显示器确认启动；重设静态 IP；离线时只做静态核对并标注。
- 证据：SNN 板端离线尝试记录（`sim/snn/evidence/platform/board/board_run_local_attempt.log`）。

## T11 PYNQ 版本差异

- 现象：`pynq.allocate` 行为不同、`Overlay` 参数变化、老教程的 `Xlnk` 不可用。
- 根因：板卡镜像版本与文档不一致（2.5/2.6/2.7+ 差异）。
- 处置：先打印版本；统一用 `pynq.allocate`；驱动记录版本到 summary。
- 证据：三块板卡运行摘要均记录 pynq 版本。
