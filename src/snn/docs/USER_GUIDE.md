# 工具链用户指南（snn-toolchain-0.1.0）

目标：**配置文件 + 权重 → SNN 参数 / RTL 配置 / 工程 Tcl / 调度 / PYNQ 包**。
本快照内可**离线**执行 compile 与纯 Python 自检；synth/OOC/bit 需要 Vivado 且受 heavy lock 约束（本快照不跑）。

## 1. 入口

统一入口：`tools/toolchain.py`（在快照根执行）

```bat
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer <compile|build|package> ...
```

| 子命令 | 作用 | 是否要 Vivado |
|---|---|---|
| `compile` | 配置+权重 → params/生成物/manifest（**从不启动 Vivado**） | 否 |
| `build --stage synth\|ooc` | 经 `scripts/spikformer/run_physical.py` 在共享 heavy lock 下落 Vivado；默认只排队，`--execute` 才真跑 | 是 |
| `build --stage ip\|platform` | 只打印 IP/平台打包入口（不执行） | — |
| `package` | manifest → 板端 bundle（`runtime_config.json` + params 副本 + README） | 否 |

## 2. 编译（最常用）

```bat
:: fixture 开发权重（必须显式给 --fixture-seed，绝不随机默认）
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --fixture-seed 1 --out-dir build/_quickstart_check

:: 真实 checkpoint
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer compile ^
  --config examples/spikformer/config/nlp_sentiment_b1.json ^
  --weights examples/nlp_sentiment_b1/ckpt_real_real50.npz ^
  --out-dir build/_quickstart_real
```

参数：`--config <json>` `--out-dir <dir>`（已存在 manifest.json 会拒绝覆盖）、`--weights <npz>`、`--fixture-seed N`、`--params-dir`（未实现，传了会报错）。
退出码 0=成功；非 0=配置/权重/out-dir 冲突。`model_kind` 只会是 `development_fixture` 或 `real_corpus`（不冒充）。

## 3. 配置（schema spikformer-config-1）

NLP 示例见 `examples/spikformer/config/nlp_sentiment_b1.json`（seq32/vocab4000/E128/H4/HD32/MLP512/depth2/T4/2 类）。
必填：`schema,family,batch,time_steps,embed,heads,head_dim,mlp_hidden,depth,num_classes,lif,input,checkpoint,head,target`；
NLP 家族另需 `seq_len,vocab_size,frontend`；图像家族另需 `image_h,image_w,channels`。

硬约束（违反即 `ValueError`，无静默回退）：

| 约束 | 规则 |
|---|---|
| batch | 只支持 `B=1` |
| time_steps | `T ∈ {1,2,4}` |
| head_dim | 只实现 `32` |
| embed | `E = heads*head_dim`，且能被 16、32 整除 |
| mlp_hidden | 16 的倍数 |
| depth | ≥1 整数 |
| seq_len（NLP） | 1..256 整数；`frontend.kind="embed_spike"`、`mask="none"` |
| image（图像） | 方形、能被 4 整除；C1 默认 `E/4` 且需 8 的倍数 |

派生量（`derived`，进 manifest）：tokens、psum_slice=32、rw=4、lif_add_bits=33、sum_w=clog2(head_dim+1)=6、
av_max=tokens*32、res_bits_min、stage_bounds（残差上界）、stream 拍数等。

## 4. 权重导入

| 模式 | 触发 | model_kind | 说明 |
|---|---|---|---|
| fixture | `--fixture-seed N` | `development_fixture` | 确定性随机参数，仅用于打通流程；**禁止**用于精度/演示结论 |
| 真实 checkpoint | `--weights <npz>` | `real_corpus` | 严格导入（dtype/shape/range/thr≥1 校验，无静默 cast） |

checkpoint ABI（`export_abi.checkpoint_npz_required_exact`，由 `tools/spikformer_train/export_integer.py` 产出）：
`etab, ptab, thr_e(1,), block{b}.{q,k,v,proj,fc1,fc2}_{w int8,b int32,scale(1,),thr(1,)}, head_w, head_b,
meta_model_kind="real_corpus", meta_corpus_sha256, meta_vocab_sha256, meta_cfg`。

- 严格校验入口：`weights.import_checkpoint` → 参数写入 `params/*.bin`（+ `.coe`），阈值/scale/head 绑定进 manifest。
- 伴生资产：checkpoint 同名 `*.assets.json`（导出器写）会被 **自动发现**，把语料/词表/label-map/split 哈希绑进 manifest。
- 本快照真实资产：`examples/nlp_sentiment_b1/ckpt_real_real50.npz`（sha256 `e6a3a411…2448a1`，若未随包则以 docs 子代理放入为准）。

## 5. 生成物

`<out-dir>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 权威绑定：model_kind、config/derived、structure、schedule、params（含 nbytes）、rtl_sources + rtl_sha256、generated、target、head、numeric、assets、manifest_sha256 |
| `resolved_config.json` `numeric_contract.json` `schedule.json` | 展开配置 / 数值合同 / 调度 |
| `params/*.bin` `.coe` | 每 block 六个线性层的 w/b（int8/int32），.coe 供 Xilinx IP 初始化 |
| `generated/spikformer_config.vh` | RTL 配置头（E/H/HD/HIDDEN/T/TOKENS/…） |
| `generated/spikformer_nlp_target_top.sv` | 目标顶层适配器（含 1192 个 DONT_TOUCH hold-margin LUT1，8 级×149 bit） |
| `generated/spikformer.xdc` `spikformer_ooc.tcl` `spikformer_project.tcl` `sources.f` | 约束、OOC/工程 Tcl、源清单（**绝对路径 = 生成时的快照根**） |
| `generated/cache_key.txt` | 生成器缓存键 |

## 6. manifest 怎么复核

```bat
E:/miniconda/python.exe -B -c "import json,hashlib,pathlib; m=json.load(open('build/_quickstart_check/manifest.json')); \
print(m['model_kind'], len(m['params']), m['manifest_sha256']); \
print('rtl mismatch:', sum(1 for s,h in m['rtl_sha256'].items() if hashlib.sha256(pathlib.Path(s).read_bytes()).hexdigest()!=h))"
```

`rtl mismatch: 0` 表示 manifest 里的 RTL 哈希与快照 RTL 一致。`manifest_sha256` 在写入前对排序 JSON 计算，**不含** `generated` 绝对路径差异，
因此同配置+同权重在不同根目录下可复现同一 `manifest_sha256`（见 `docs/PORTABILITY.md` §5）。

## 7. 物理门重跑（**本快照不执行**）

```bat
:: 1) 只排队（默认，不启动 Vivado）
E:/miniconda/python.exe -B -m tools.toolchain --backend spikformer build ^
  --manifest build/<run>/manifest.json --stage ooc --shared-heavy-lock <abs lock path>

:: 2) 真正执行（需要锁 + Vivado 2021.1 + 遵守 heavy lock 规则）
... 追加 --execute
```

规则（不在快照内违反）：只走**共享** heavy lock（`scripts/spikformer/run_heavy.py`），不私有锁、不并发重任务、不杀进程；
5 ns 不降频、不加时序例外；负 WNS 记 fail；OOC 结果**不等于**集成/板级。
IP 源包：`scripts/spikformer/run_ip_pack.py --manifest <run>/manifest.json --out-dir <run>/ip`（同源哈希 + OOC 报告绑定）。

## 8. PYNQ 运行包

- `runtime/spikformer/run.py`：CSR 偏移表 + `SpikformerRunner`（写刺激 → start → 轮询 STATUS → 读 sums → ACK）；
  `runtime/spikformer/head.py`：PS FP32 head（`ps_div`：除以 T*N 后 Linear+argmax）；`tokenize.py`：按字分词 + pad=0。
- `package` 子命令产出板端 bundle（`runtime_config.json` 含 CSR/窗口/head 合同），bit/HWH 由平台阶段补。
- 上板包见 `board/`（r7 bit/hwh/vec/MANIFEST），运行方法见 `board/README_BOARD.md`。

## 9. 验证入口

| 命令 | 期望 | 说明 |
|---|---|---|
| `verification/sim/spikformer/test_short.py` | `FAILURES: 0` | LIF/popcount/残差/packing/生命周期/卷积语义 |
| `tools/spikformer_train/test_integer_ref.py` | `RESULT: 8/8 PASS` | 整数参考实现 vs torch 量化前向逐层一致（需 torch） |
| `tools/spikformer_train/check_ready.py` | 逐项 PASS / not_run / blocked_assets | 就绪度自查（快照内多数 `build/` 产物缺失 → not_run 属正常；已实测可运行） |
| 完整验收记录 | — | `evidence/ACCEPTANCE_CORE.md` |

## 10. 事实纪律

compile/package 成功**只**证明“生成链可用”；不证明时序/资源/板级。精度只认整数域评估（manifest `numeric.model_assets` 与
`integer_accuracy` 门），不用 fixture、不用旧结果、不把 OOC 说成集成。
