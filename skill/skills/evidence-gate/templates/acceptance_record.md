# 验收记录（模板）

> 用法：复制为 `<lib>/evidence/ACCEPTANCE_<stage>_<YYYYMMDD>.md`。
> 只记录**实际执行**的命令与结果；未执行的写 not_run，不要留空，不要写"应该"。

执行人：<角色/智能体名>；执行根：`<包根或工作区路径>`；记录时间 <YYYY-MM-DD HH:MM–HH:MM CST>
环境：<OS> + <python 绝对路径>（Python <版本>，<关键包版本>）；<Vivado/ModelSim 版本>

## A1 <阶段名>（<spec 条款>）

```bash
<可直接粘贴的命令原文，含工作目录>
```

结果：**<pass|fail|partial|not_run>**（exit <code>）

```text
<关键输出 3~10 行，含 manifest 路径/状态行>
```

| 项 | 值 |
|---|---|
| 输出目录 | `<path>`（<大小>） |
| <关键字段> | <值> |
| `manifest_sha256` | `<hex>` |
| 文件 sha256 | `<hex>` |
| 源哈希自校验 | mismatch <n>（<总数>） |

## A2 确定性复跑

| 项 | 第一次 | 第二次（新目录） | 一致 |
|---|---|---|---|
| params 集合摘要 | `<hex>` | `<hex>` | 是/否 |
| manifest.json 文件 sha256 | `<hex>` | `<hex>` | 是/否 |

## A3 未执行（not_run）

| 项 | 原因 |
|---|---|
| <项> | <原因> |

## A4 清理

| 项 | 状态 |
|---|---|
| 临时目录 | 已删除 / 保留（原因） |
| `__pycache__`/`*.pyc` | 0 |

## 更新记录

- <YYYY-MM-DD HH:MM>：初版（<范围>）。
- <YYYY-MM-DD HH:MM>：<复跑/刷新>。
