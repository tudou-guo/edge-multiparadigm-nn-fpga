# 证据产物字段表

> 与 `evidence-format.md` 分工：本文给"每个产物文件应包含哪些字段/行"，格式细节与算法见前者。

## 1 `INDEX.md`（状态索引）

| 字段（列） | 必填 | 取值 | 说明 |
|---|---|---|---|
| 主题 | 是 | 自由文本 | 一句话，主语是"哪一步/哪个对象" |
| 状态 | 是 | `pass / fail / partial / not_run` + 限定词 | 冒号后可带关键数字；记录级写 `记录级…/not_verified` |
| 证据 | 是 | 相对 `evidence/` 的路径（可多个） | 每个 pass 至少一条；路径必须真实存在 |

附加段：校验方法（E1 命令）、未执行（not_run）表、更新记录（时间 + 变更）。

## 2 `SHA256SUMS.txt` / `COPY_SHA256.txt`

| 字段 | 必填 | 取值 |
|---|---|---|
| sha256 | 是 | 64 位小写 hex |
| 分隔 | 是 | 两个空格 |
| 相对路径 | 是 | 相对清单所在目录，正斜杠 |
| 注释行 | 否 | 以 `#` 开头（如快照时间、排除规则） |

排除规则必须写进记录：SNN 快照排除 `evidence/` 与 `build/`（清单只覆盖随包源码/文档/证据）。

## 3 `ACCEPTANCE*.md`（验收记录）

| 段 | 内容 | 必填 |
|---|---|---|
| 头部 | 执行人、执行根、记录时间、环境（OS/Python/工具版本） | 是 |
| A1… | 每项：命令原文 + 结果 + 关键输出片段 + 字段表 | 是 |
| 未执行 | not_run 项与原因 | 是 |
| 复跑 | 同 seed 新目录的确定性对比 | 建议 |
| 清理 | 临时目录/pycache 处置 | 建议 |
| 更新记录 | 初版/复跑/最终刷新时间线 | 是 |

## 4 `_copy_manifest.json`（复制清单）

| 字段 | 含义 |
|---|---|
| `source_root` | 复制来源根（开发树绝对路径） |
| `snapshot_time` | 复制时刻 |
| `entries[]` | 每个条目：`src`、`dst`、`sha256`、`bytes` |
| `excluded[]` | 明确不复制的大文件（routed DCP、>2MB 报告、参数二进制）与原因 |

## 5 单条结论记录（建议的 JSON 形态）

```json
{
  "claim": "NLP OOC 5ns routed",
  "label": "pass",
  "gate": "G2",
  "evidence": ["nlp/final/reports/ooc_20261030_120804.json"],
  "fields": {"wns_ns": 0.135, "whs_ns": 0.023, "failing_endpoints": 0},
  "identity": {"rtl_sha256": "<hash>", "manifest_sha256": "<hash>"},
  "reproduce": "python -c \"import json;print(json.load(open('...'))['wns_ns'])\"",
  "time": "2026-10-30T12:08:04+08:00",
  "executor": "consolidate-core",
  "qualifiers": {"record_level": false, "measured": true}
}
```

字段约束：`label` 只能是四值之一；`qualifiers.record_level=true` 时不得与 `pass` 同用；`evidence` 路径必须存在；
`reproduce` 必须能在本机 60 秒内跑完。

## 6 命名与放置

| 类型 | 命名 | 位置 |
|---|---|---|
| 状态索引 | `INDEX.md` | `sim/<lib>/evidence/` |
| 验收记录 | `ACCEPTANCE_<stage>_<YYYYMMDD>.md` | 同上 |
| 哈希清单 | `SHA256SUMS.txt`、`COPY_SHA256.txt` | 同上 |
| 主题证据 | `<topic>/<name>_<YYYYMMDD>.(json\|log\|md)` | 主题子目录 |
| 进度/根因文档 | `<TOPIC>_PROGRESS_<YYYYMMDD>.md` | 库 `docs/` 或 evidence 主题目录 |
| 模板 | 固定文件名 | `skill/skills/evidence-gate/templates/` |

规则：原始证据只增不改；新证据带日期或 run-id；覆盖旧文件视为篡改身份。
