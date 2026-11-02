# 证据状态索引（模板）

> 用法：复制到 `<lib>/evidence/INDEX.md`，逐行替换。删除本注释，但不要删除列。
> 规则：一行一事实；每个 pass 必须指向"文件 + 字段"；记录级/历史结论不得混入 pass 段。

快照时间：<YYYY-MM-DD HH:MM CST（复制时刻；来源 <开发树路径> 只读）>
规模：共 <N> 个条目 / <X> MB；其中 <M> 个文件计入 SHA256SUMS.txt
原则：只复制小文件；不复制 routed DCP/大报告/参数二进制（缺口见"未执行"段）

## 0 校验方法

```bash
cd <lib>/evidence
python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"
```

最终校验（<时间>）：覆盖 <M> 个文件，实测 `bad=0`。

## 1 状态摘要

| 主题 | 状态 | 证据 |
|---|---|---|
| <主题> | pass：<关键数字> | `<相对路径>`（相对 evidence/） |
| <主题> | partial：<通过项>；<未通过项> | `<相对路径>` |
| <主题> | not_run：<原因> | `<说明或路径>` |
| <主题> | 记录级（record_level）→ not_verified：<原始文件未随包/未复跑> | `<路径>` |

## 2 证据明细

| 证据 | 主线来源路径 | sha256 | 用途/结论 |
|---|---|---|---|
| `<相对路径>` | <开发树绝对路径> | <sha256> | <一句话结论> |

## 3 未执行（not_run）

| 项 | 原因 |
|---|---|
| <项> | <原因> |

## 4 更新记录

- <YYYY-MM-DD HH:MM>：初版。
- <YYYY-MM-DD HH:MM>：<复跑/刷新/更正内容>。
