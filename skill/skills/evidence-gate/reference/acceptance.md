# 证据门禁自检清单（本技能自身的验收）

> 每条给：判据 / 命令 / 不达标时的处置。全部满足才算"证据闭环"。

## A1 30 秒复核（抽 3 条 pass）

```bash
# 任选一条索引里的 pass，按条目给的复算命令执行；或直接用 reference/commands.md 的 E1/E6
cd sim/snn/evidence && python -c "import json;d=json.load(open('nlp/final/reports/ooc_20261030_120804.json'));print(d.get('wns_ns'), d.get('whs_ns'), d.get('failing_endpoints'))"
```

- 判据：命令输出与索引里写的数字一致（如 `0.135 0.023 0`）。
- 不达标：索引数字改回证据值，或把该条降为 `partial`。

## A2 无源数字扫描

```bash
grep -nE '[0-9]+\.[0-9]+ ?(ns|ms|%|MB)' report/*.md sim/*/evidence/*.md 2>/dev/null | head -40
```

- 判据：每个数字都能在同一行/上一行找到证据路径或"记录级/实测"标注。
- 不达标：补出处或删除；估算值必须标 `derived`。

## A3 记录级隔离

- 判据：`record_level` 条目不出现在"pass 段"，且结论不含"已通过"字样。
- 检查：`grep -n 'record_level' sim/*/evidence/INDEX.md`，逐条确认标签写的是"记录级/not_verified"。
- 不达标：改为 `not_verified`，并写原始文件未随包/未复跑。

## A4 索引自校验

```bash
cd sim/snn/evidence && python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p.strip()).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"
```

- 判据：`bad=0`。
- 不达标：见 reference/commands.md E1 的处置。

## A5 索引条目 ↔ 文件一一对应

```bash
# 索引里出现的证据路径都应在磁盘上
grep -oE '\(([a-z0-9_./-]+\.(json|log|md|txt|rpt))\)' sim/snn/evidence/INDEX.md | tr -d '()' | sort -u | while read p; do test -f "sim/snn/evidence/$p" || echo "MISS $p"; done
```

- 判据：无 `MISS` 行。
- 不达标：删除失效条目或补文件；不得保留指向不存在文件的 pass。

## A6 跨版本失效检查

- 判据：每条证据记录了其绑定的 `rtl_sha256`/`manifest_sha256`；当前源码哈希与之一致，或已标 `invalidated_runs`。
- 检查：比对 `sim/snn/evidence/nlp/rtl_freeze.json` 与 `rtl_hash_check_20261031.txt`（16 SAME / 2 DIFF 的 DIFF 项必须有说明）。
- 不达标：旧证据标失效，重跑或降级。

## A7 归档完整性

- 判据：`INDEX.md`、`SHA256SUMS.txt`、`ACCEPTANCE*.md`（如有）、`_copy_manifest.json`（如有）齐全；`build/` 与临时目录未被误收。
- 检查：`ls sim/snn/evidence`；对照 reference/artifacts.md §6 的命名表。
- 不达标：补齐或写明缺失原因（not_run）。

## A8 结论措辞

- 判据：全文无"基本通过/应该没问题/大致一致"；无把 `partial` 写成 `pass`；无把仿真写成板端。
- 检查：`grep -nE '基本通过|应该没问题|大致一致|板上通过|已上板' sim/*/evidence/*.md report/*.md`。
- 不达标：按 SKILL §4 步骤 2 重打标签。

## A9 交付前的最后三问

1. 这条结论**谁**在**哪台机器**上、**什么时候**跑出来的？（有记录）
2. 换一台机器能不能**30 秒内**复算出同一个数字？（有命令）
3. 如果源码明天变了，这条结论会不会自动失效？（有绑定）

三问有一问答不上，就不算证据闭环。
