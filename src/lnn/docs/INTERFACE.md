# CFC1 寄存器与数值接口

器件 `xczu15eg-ffvb1156-2-i`，CfC 子系统单时钟域 `npu_clk = 300 MHz`（平台 MIG UI 域）。
主机经 AXI4-Lite 访问 4 KiB 寄存器窗口，基地址 `0x0480000000`（PS 地址空间，35 位内；驱动保留完整 64 位）。

## 1. AXI4-Lite 从机

- 32 位数据、12 位窗口内字节地址；4 字节对齐访问
- 独立 AW / W 通道（任意先后或同拍），WSTRB 字节写；B 响应单拍
- 读数据 1 拍延迟；支持 rvalid 反压前的一笔挂起读
- 忙时（BUSY=1）：输入窗口与配置寄存器写入被丢弃（AXI 正常响应），不改变正在执行的任务

## 2. 寄存器映射

| 偏移 | 名称 | 属性 | 说明 |
|---|---|---|---|
| 0x000 | ID | RO | `0x43464331`（"CFC1"） |
| 0x004 | VERSION | RO | `0x00010000`（V1.0） |
| 0x008 | CONTROL | W1P | bit0 START，bit1 CLEAR_STATE（仅空闲生效），bit2 CLEAR_STATUS；其余位忽略 |
| 0x00C | STATUS | RO | bit0 BUSY，bit1 DONE（sticky），bit2 ERROR（sticky） |
| 0x010 | ERROR_CODE | RO | 0 正常，1 配置错误，2 忙时 START，3 运行超时 |
| 0x014 | N_STEPS | RW | 默认 64，合法 1..64 |
| 0x018 | DELTA_T | RW | 低 16 位正整数（Q12），默认 655（0.159912），合法 1..32767 |
| 0x01C | MODE | RW | bit0 运行前清状态（默认 1）；bit1 每步用 h_old=0（默认 0） |
| 0x020 | RUN_CYCLES | RO | 最近一次运行周期数（START 接受 → 最后输出提交） |
| 0x024 | STEP_COUNT | RO | 已提交输出个数（成功 = N_STEPS） |
| 0x028 | TIMEOUT_CYCLES | RW | 默认 1,000,000（≈3.33 ms @300 MHz） |
| 0x02C | CLOCK_HZ | RO | `300000000` |
| 0x080..0x08C | H_FINAL | RO | 4 个末状态，每字低 16 位有符号，高 16 位符号扩展 |
| 0x100..0x1FC | INPUT | RW | 64 个采样点，每字低 16 位有符号（高 16 位忽略） |
| 0x200..0x2FC | OUTPUT | RO | 64 个输出点，每字低 16 位有符号，高 16 位符号扩展 |

未实现偏移读 0、写忽略。

## 3. 运行协议

1. 空闲时写入 64 个输入样本与配置（N_STEPS / DELTA_T / MODE / TIMEOUT_CYCLES）；
2. 写 `CONTROL.START`（W1P）：硬件锁存配置、清 DONE/ERROR/RUN_CYCLES/STEP_COUNT、置 BUSY；
   MODE[0]=1 时先清隐状态；MODE[1]=1 时每步以 h_old=0 计算（状态仍逐步提交，供 H_FINAL 观测）；
3. 轮询 STATUS：DONE=1 成功；ERROR=1 时读 ERROR_CODE；
4. 读 OUTPUT / H_FINAL / STEP_COUNT / RUN_CYCLES；
5. 清 DONE/ERROR 用 `CONTROL.CLEAR_STATUS`（或下一次合法 START）。

超时（RUN_CYCLES 达到 TIMEOUT_CYCLES）会停止本次任务、置 ERROR、ERROR_CODE=3，输出视为无效；
重新运行前建议显式 CLEAR_STATUS/CLEAR_STATE。

## 4. 数值格式

| 对象 | 格式 |
|---|---|
| 输入 / 权重 / 隐状态 / 输出 | int16，Q12（实值 = 整数 / 4096） |
| 偏置（b1,b2,ba,bb,bo） | int32，Q24（实值 = 整数 / 2^24） |
| 累加器 | signed 48 位精确累加，无中间截断 |
| 舍入 | 最近值、中点远离零，再饱和到 int16 |
| Δt | 整数 655（Q12；浮点训练用 0.16） |
| 激活 | 1025 点 tanh / sigmoid 表，区间 [-8,8]，步长 1/64，索引 `clamp((v+32768+32)>>6, 0, 1024)`，输出 Q12 |

单步计算（与需求文档 §2.1 一致）：

~~~text
z   = [x, h0, h1, h2, h3]
f1  = tanh(W1 z + b1);  f2 = tanh(W2 z + b2)
a   = Wa z + ba;        b  = Wb z + bb
g   = sigmoid(a * dt + b)
h'  = f1 * (1 - g) + f2 * g
y   = Wo h' + bo
~~~

定点实现中 `a` 与 `b` 各自先舍入饱和为 Q12，再计算 `gate_pre = ROUND(a*655 + (b<<12))`，
由同一份查表得到门控；4 个新状态在同一时刻提交，且同一步四个分支读同一份旧状态快照。

## 5. 时序与资源（本次构建）

| 项 | 值 |
|---|---|
| 应用时钟 | 300 MHz（单时钟域；无 false_path / clock group 掩盖） |
| 64 步片上周期 | RUN_CYCLES = 9088（≈30.3 µs；约 142 周期/步） |
| setup / hold | WNS +0.226 ns / WHS +0.011 ns（0 失败端点） |
| 资源 | CLB LUT 25,221；FF 32,890；DSP48E2 4；BRAM Tile 28.5（CfC 核单独：LUT 2189 / FF 3280 / DSP 1） |

## 6. 板端 `summary.json` 关键字段

`schema=lnn-cfc-board-run-1`、`source`（`hardware` / `simulation-model`）、`is_hardware_measurement`、
`sample_id`、`mode`/`mode_bits`、`n_steps`、`delta_t_int`、`status`、`error_code`、
`output_match_count`/`output_total`/`outputs_bit_exact`/`mismatch_indices`、`final_state`/`final_state_match`、
`step_count`/`step_count_ok`、`task_mse`（`task_mse_warmup_skip`）、`fixed_point_mse_int`、
`run_cycles`/`run_cycles_source`/`clock_hz`、`hardware_time_s`/`e2e_time_s`/`write_time_s`/`wait_time_s`/`readback_time_s`、
`bit_sha256`/`hwh_sha256`/`xsa_sha256`/`manifest_sha256`。
