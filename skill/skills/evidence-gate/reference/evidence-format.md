# 证据格式与哈希复算

## 1 目录约定

```
sim/<lib>/evidence/            # 包内（开发树为 <LIB>/evidence/）
├── INDEX.md                   # 状态索引：一行一事实，每行带证据链接
├── ACCEPTANCE*.md             # 验收记录（命令 + 结果表 + 未执行表 + 刷新记录）
├── SHA256SUMS.txt             # 文件哈希清单（可复算，bad=0）
├── _copy_manifest.json        # 复制清单（从哪来、复制了哪些）
├── <topic>/                   # nlp/ platform/ vision/ behavior/ ref_route/ ooc_native/ ...
└── _gen_index.py / _rtlhash.py # 索引与 RTL 哈希生成脚本（可复跑）
```

## 2 状态索引（INDEX.md）格式

一行一事实，列为：主题 | 状态 | 证据。状态取值与 SKILL 的标签一致，但可带限定词：

```markdown
| 主题 | 状态 | 证据 |
|---|---|---|
| NLP 数学合同 | pass | `nlp/final_gates.json`、`nlp/math_tests_m1.log` |
| NLP OOC 5ns routed | pass：WNS +0.135 / WHS +0.023 / 0 failing | `nlp/final/reports/ooc_20261030_120804.json` |
| 平台集成 | partial（定稿口径）：BD/compile/CSR smoke pass；CDC 1 Critical 未收敛 | `platform/platform_build_20261030.json` |
```

写法要求：

- `pass` 后可以带冒号接关键数字（数字不带出处即违规）；
- `partial` 必须写"部分通过什么 + 未通过什么"；
- 历史或客方回传的结论须注明来源与日期，不得混进 pass 段；
- 证据链接写**相对 evidence/ 的路径**，便于整目录搬移。

## 3 哈希清单（SHA256SUMS.txt / COPY_SHA256.txt）

格式：`<64 位小写 hex><两个空格><相对路径>`；行首 `#` 为注释；空行忽略。

```text
# SNN snapshot 2026-10-31 17:38 CST
9fc1c267…  nlp/final/generated/spikformer.xdc
```

复算（Python，无需 sha256sum，Windows/Linux 通用）：

```python
import hashlib, pathlib
root = pathlib.Path("sim/snn/evidence")          # 或 <lib>/evidence
rows = [l.split("  ", 1) for l in (root/"SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if l.strip() and not l.startswith("#")]
bad = [p for h, p in rows if hashlib.sha256((root/p).read_bytes()).hexdigest() != h]
print("files=%d bad=%d" % (len(rows), len(bad)))
```

## 4 三类哈希的区别

| 哈希 | 算法对象 | 用途 | 例子 |
|---|---|---|---|
| 文件 sha256 | 文件字节 | 搬运、配对、索引自校验 | `lnn_cfc.bit` = `0fcd8d65…`；`lnn_cfc.hwh` = `fe2d9f0e…`；`lnn_cfc.xsa` = `f01d6a8a…` |
| `manifest_sha256` | 清单**内容规范化**后 | 绑定"本次编译的输入+配置+RTL"；重跑不变即确定性 | SNN `835ad38d…`（fixture）/ `f727e4c7…`（真实权重） |
| 集合摘要 | 一组文件的 `sha256sum ... \| sort -k2` 再 sha256 | 参数 blob 整体一致性 | SNN `params/*.bin` = `000b0b5b…`（28 个，fixture）/ `315171a2…`（54 个，真实权重） |

注意：`manifest_sha256` ≠ 清单文件自身的 sha256（同一份清单两者不同，前者是自描述哈希）。
两者都要记：前者证确定性，后者证字节未改。

## 5 配对检查（多件套）

| 库 | 配对对象 | 做法 | 通过判据 |
|---|---|---|---|
| LNN | `lnn_cfc.bit` ↔ `lnn_cfc.xsa` 内嵌 bitstream | 解 XSA 取内嵌 bitstream 算 sha256 | 与 `.bit` 相同 |
| LNN | `lnn_cfc.hwh` ↔ XSA 内 `design_1.hwh` ↔ Vivado hw_handoff | 逐字节比对 | 三者一致 |
| CNN | `platform_top.bit` ↔ `platform_top.hwh` ↔ `MANIFEST.json` | 读 MANIFEST 的 `bit_sha256`/`hwh_sha256` 与实际文件比对 | 一致（MANIFEST 内 `identity` 段） |
| SNN | `snn_platform.bit` ↔ `snn_platform.hwh` ↔ `MANIFEST.sha256` | 同上，并核对 `MANIFEST_DEMO_*.sha256` | 一致且 run 记录同批 |

## 6 确定性复跑

- 同 seed、同参数、**新输出目录**重跑一次，比较 `params/*.bin` 集合摘要与 `manifest.json` 文件 sha256；
- 两者都相同 → 本次证据可复现；不同 → 先查环境/缓存/路径，再决定是否采信；
- 复跑目录用完即删，避免污染证据目录（SNN 实践：开发树 `build/_accept_recheck` 复跑后删除）。

## 7 失效与作废

- 证据文件内应记录其绑定的 RTL/源码哈希与时间；
- 源变更后，旧证据进入 `invalidated_runs`（SNN `tb_sps_native_gate_result_*.json` 字段）；
- 失效证据不删除，标 `superseded_by` 指向新证据；索引中保留历史行并注明"历史基线"。
