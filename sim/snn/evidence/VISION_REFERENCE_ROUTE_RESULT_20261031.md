# 图像 native 参考工程路线：资源修复实施返回交接（2026-10-31）

状态：A 轮 P0 完成，P1 进行中（small 参考式 SPS：conv1/pool1/conv2-tile/psum 已 bit-exact，prow2 落盘/drain 行对齐仍有 53 nibble 不一致，未达门）。P2 native 整链、P3 OOC、B 轮、P4/板测均为 not_run。

任务书：D:/dl_snn/docs/VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md。本文只记录实测，不把计划写成 PASS。
旧 small PASS 与参考旧实现报告仍是历史证据，不作为移植后 PASS。

## 1. 门状态总览

- P0 基线冻结/哈希/run/布局协议：PASS（build/spikformer/vision_ref_20261031/identity + vectors meta）。
- P0 bias_beats 口径统一：PASS（代码已改，未跑全回归）。config.py bias_beats=max(c1,c2)，native=384 beats 21-bit pair；旧 (c1+c2+15)//16=30 保留为 deprecated 字段。
- P0 阶段 golden（small+native）：PASS。small 与 legacy exp.hex 16/16 全等；native 向量已生成（ld1=324 bias=384 uniq=41472 pix=1024 exp=3072）。
- P1 参考式 SPS small 全等（16 beats）：FAIL（53/256 nibbles）。日志见 build/.../sim/tb_sps_ref_small.log。
- P1 native SPS 综合：not_run（被 small 门阻塞）。
- P2 native 整链 bit-exact（3072 beats + 384 sums）：not_run（RTL 已写，未仿真）。
- P3 同源 OOC 综合/实现（5ns routed）：not_run。
- B 轮 block 分块供权：not_run（按任务书条件触发）。
- P4 平台/板测：not_run。

## 2. 真正复用的参考模块与适配

- conv_layer1/Multi_add_unit -> spiking_rgb_encoder.sv：27 数值乘法单元、行移位窗口、输出通道分时；C1=96，signed pixel/bias/LIF 不变。权重 bank 用寄存器（ reference r_weight_array 风格），帧存直写窗口寄存器。
- simple_eyeriss_pe_unit -> spiking_eyeriss_pe.sv：修了一个真 bug——原 pad+shift 使单拍使用丢最后一列 tap；行复用上移到阵列定序器后改精确单窗语义。该 PE 只被新阵列使用，不影响旧回归。
- simple_eyeriss_array -> spiking_eyeriss_array.sv：64 PE（8 lane x 8 ch，K=3），3 拍行序列、输入通道分组归约、lane 累加。无全窗口单拍动态数组表达式。
- psum_callback/top/ram -> spiking_sps_psum.sv：8 lane bank（256x84b，ram_style=block）、同步 RMW、FIRST/MID/LAST、bias 只加一次。
- SpikesTmpRam -> spiking_feature_bram.sv x2：128x8192，Lmem=1（地址寄存+组合读数）。bank0=f1→prow2，bank1=pooled。
- maxpool row 结构借用到控制器 pool1/drain（2x2 OR per timestep）。
- block 本轮未动（spiking_block_top.sv 保持原哈希），A 轮保留单份权重 RAM。

## 3. 实际修改清单

新增 RTL（已注册进 rtl/ops/spiking/source_manifest.json v20261031.1）：
sps/spiking_feature_bram.sv(39行)、sps/spiking_rgb_encoder.sv(211)、sps/spiking_eyeriss_array.sv(206)、
sps/spiking_sps_psum.sv(152)、sps/spiking_sps_ref_ctrl.sv(~430，含 C_C2W/C_C2FL/DRAIN)、
sps/spiking_sps_token_adapter.sv(47，2深skid+1bit→RW4展宽0/1)、sps/spiking_sps_ref_top.sv(128)、
control/spikformer_vis_native_top.sv(122，去w2rom/blkrom/tbuf，block先装权重、SPS直喂、握手累加sums)。
旧 spikformer_vis_top.sv 保留作 legacy 基线。

工具链/向量：config.py 见§1；scripts/spikformer/gen_sps_param.py（新，与 gen_sps_replay 同方程向量化实现，--out-dir 不覆盖旧证据，输出 w_ld1/w_uniq/w_sched/pix/exp/meta）；
native 向量见 build/.../vectors/sps_native。

## 4. small 回归现状（P1 未过门，53 nibble fails）

TB: verification/sim/spikformer/tb_sps_ref_small.v（H8/C1-16/C2-32/T2，16 beats 全比+last+done+watchdog）。
编译：sim_work/vis_sps/compile_ref_small.do。日志：build/.../sim/tb_sps_ref_small*.log。

已用实测定位为 exact 的链路：conv1（f1 bank 字与 python golden 逐字全等）、pool1（bank1 低64b全等）、
conv2 tile 和（ARRAY 输出与 golden 全等，如 pos0tile0 得 44,-31,80,33,92,-123,-205,-83）、
psum beat（抽查 pos0-3/g0/g3 与 golden prow2 一致； Beat 重构 token(n0,g3)=0x333c 与 golden exp beat3 一致）、beat0 与 golden 30333300 全等。

仍 FAIL：53 nibbles，over-fire 随 g 增大（g0:9,g1:14?,g2:?,g3:27 中 over 占多数）。bank 抽查：g3y0k0 期望 0xC0 实测 0x3F（行内位置错位）。
CAP 追踪证实 4 个 beat 本身正确且 idx=0..3 有序，但 C_C2W 入口时 p2row 已错—— flush 用的 p2row 与刚捕获的不一致。
当前最可能根因（未证实，卡住点）：某次 flush 用了 stale p2row，或 p2cnt 在 tile/group 交界与 beat 到达竞态（beat 在 flush 窗口期间到达被计入 pushed 但未落盘，导致后续行整体错一位）。
已尝试的修复（每项有独立证据）：comb 转发对齐（修 tile 边界丢拍）、PE 单窗语义、bank Lmem=1、BEATW 统一、fifo 改直收+反压、flush 全局触发+resume 状态、C_C2FL 等收齐、sticky done、FIFO 指针位宽。
丢拍数 15→0、X 消除、tile 和全等，但行对齐问题未解决。

## 5. 同口径资源/时序/流量（实测 vs 推导严格区分）

- 实测利用率/时序：均未测（OOC 未跑）。不报 LUT/FF/BRAM/DSP，不报 WNS。
- 容量推导（仅用于 B 轮触发判断，非实测）：conv2 唯一权重 331776B（下限72 BRAM36）；block 六矩阵 1769472B（下限384）；SPS 双 bank 256KiB；psum 8x256x84b。
- 流量（调度定义，已生成不等于已实测）：SPS 每图 host 42180 beats 约337KB unique（无复制）；legacy w_tiles 复制展开约84.9MB 不再作硬件映像；block 权重流 222912 beats；SPS→block 3072 beats。旧 21.2MB/图 与 bias_beats=30 按任务书第3节作废。
- RTL cycles/板端 FPS：未测，不报。

## 6. native 完整复现命令（RTL 就绪，sim 待跑）

- small 门（当前 FAIL）：sim_work/vis_sps 下跑 compile_ref_small.do，再 vsim tb_sps_ref_small run 100ms。
- native SPS TB 待建（H32/C1-96/C2-384/T4，3072 beats，向量 sps_native 已就绪）。
- native 整链 TB 待建（block 权重 222912 beats + SPS 流 + 两图 + 反压）。
- OOC 待补脚本并走 heavy_lock（probe Tcl 需先扩展新 control 顶层与 BRAM 配置，不能拿 synthesis timing 当 routed timing）。

## 7. 明确未测项与 B 轮/P4 交接

- small 参考式 SPS：FAIL（53 nibbles）。下一步：给 p2row/flush 加断言（flush 时 p2cnt 必须为 ROWW 且 beat_pushed 连续），或把 p2row 改成按 (c_pr,c_pc) 直接寻址写入消灭行组装时序，再跑 small。预计 1-2 轮迭代。
- native SPS 整链、native 整链、OOC、实现、B 轮、平台 BD/HWH/XSA/板测：全部 not_run。
- A 轮保留 block 单份权重 RAM；若 OOC 显示放不下或无平台余量，按任务书第8节进 B 轮。
- 平台复用 spk_platform 骨架仍是计划，未动；NLP 演示包未碰。

## 8. 证据索引

- 任务书 docs/VISION_REFERENCE_ROUTE_REPAIR_PLAN_20261031.md；基线哈希 build/.../identity/baseline_hashes.json。
- 向量 build/.../vectors/{sps_small_check,sps_native}/；仿真 build/.../sim/tb_sps_ref_small*.log；编译 sim_work/vis_sps/compile_ref_small.do。
- 新 TB verification/sim/spikformer/tb_sps_ref_small.v。
- 历史证据（未覆盖）：sim_work/vis_sps 旧 small PASS 日志、参考旧实现报告 E:/study/.../impl_1/*placed.rpt。
