# 复算命令矩阵（证据门禁）

> 所有命令在包根执行；`<lib>` ∈ {snn, lnn}，证据目录为 `sim/<lib>/evidence/`。
> 每条给出：用途 / 命令 / 期望 / 失败处理。Python 用本机 3.9.x 即可，无需第三方库。

## E1 复算哈希清单（最常用）

```bash
cd sim/snn/evidence
python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('SHA256SUMS.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p.strip()).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"
```

- 期望：`files=133 bad=0`（2026-11-02 包内实测）。
- 失败处理：`bad>0` → 该文件被改过或搬运时损坏；先查 git/拷贝记录，再决定是否重算清单（重算=改证据，需记录原因与时间）。

## E2 单文件 sha256 / 大小

```bash
sha256sum board/lnn/lnn_cfc.bit board/lnn/lnn_cfc.hwh board/lnn/lnn_cfc.xsa
stat -c '%n %s' board/lnn/lnn_cfc.*
```

- 期望：与 `sim/lnn/evidence/ACCEPTANCE.md` §3 的哈希与字节数逐一相同。
- 失败处理：任一不同 → 三件套不同源，按 G6 判 `fail`，不得部署。

## E3 确定性复跑比对（同 seed、新目录）

```bash
# 1) 两次编译到不同新目录后，比较清单文件字节与自描述哈希
python -c "import hashlib,pathlib; [print(p, hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()) for p in ['build/run_a/manifest.json','build/run_b/manifest.json']]"
python -c "import json; a=json.load(open('build/run_a/manifest.json')); b=json.load(open('build/run_b/manifest.json')); print(a.get('manifest_sha256'), b.get('manifest_sha256'), a.get('manifest_sha256')==b.get('manifest_sha256'))"
```

- 期望：文件 sha256 与 `manifest_sha256` 都相同。
- 失败处理：文件不同但 `manifest_sha256` 相同 → 清单里含路径/时间等易变字段，比对集合摘要；两者都不同 → 环境不确定，证据不可用。

## E4 参数 blob 集合摘要

```bash
cd <out-dir>/params && sha256sum *.bin | sort -k2 | sha256sum
```

- 期望：与记录值一致（SNN：fixture `000b0b5b…`（28 个）/ 真实权重 `315171a2…`（54 个））。
- 失败处理：数量不符先数 `ls *.bin | wc -l`；数量对但摘要不同 → 权重或量化不同批。

## E5 多件套配对（bit ↔ XSA ↔ hwh）

```bash
cd board/lnn
python -c "import hashlib,zipfile; z=zipfile.ZipFile('lnn_cfc.xsa'); print('xsa-bit ', hashlib.sha256(z.read('lnn_cfc.bit')).hexdigest()); print('xsa-hwh ', hashlib.sha256(z.read('design_1.hwh')).hexdigest())"
sha256sum lnn_cfc.bit lnn_cfc.hwh
```

- 期望：`xsa-bit` == `lnn_cfc.bit`；`xsa-hwh` == `lnn_cfc.hwh`。
- 失败处理：不等 → XSA 不是同次导出（或未 `-include_bit`），重新导出。

## E6 从证据 JSON 提取关键字段（写索引前）

```bash
python -c "import json; d=json.load(open('sim/snn/evidence/nlp/final/reports/ooc_20261030_120804.json')); print({k:d.get(k) for k in ['wns_ns','whs_ns','tns_ns','failing_endpoints','unrouted_nets']})"
python -c "import json; d=json.load(open('sim/snn/evidence/platform/platform_build_20261030.json')); print(d.get('status'), d.get('modelsim_compile'))"
```

- 期望：字段存在且数值与索引/报告一致；字段名以真实文件为准（不同 run 的 schema 可能不同）。
- 失败处理：字段缺失 → 不要猜，改读同级日志/报告，或把结论降为 `partial`。

## E7 证据目录规模与文件数

```bash
cd sim/snn/evidence && find . -type f | wc -l && du -sh .
```

- 期望：与 `INDEX.md` 开头"共 N 个条目 / X MB"相符（允许因新增索引脚本小幅偏差，但要在更新记录里写明）。
- 失败处理：偏差大 → 有文件被增删；补更新记录，必要时重算 SHA256SUMS。

## E8 复算 COPY_SHA256（复制清单）

```bash
cd sim/snn/evidence
python -c "import hashlib,pathlib; rows=[l.split('  ',1) for l in pathlib.Path('COPY_SHA256.txt').read_text(encoding='utf-8').splitlines() if l.strip() and not l.startswith('#')]; bad=[p for h,p in rows if hashlib.sha256(pathlib.Path(p.strip()).read_bytes()).hexdigest()!=h]; print('files=%d bad=%d' % (len(rows), len(bad)))"
```

- 期望：`bad=0`（开发期快照为 150 文件 + 2 行注释）。
- 失败处理：同 E1。

## E9 30 秒复核模板（对单条结论）

```text
结论：<主题> = <pass|partial|not_run>
证据：<相对 evidence/ 的路径>（字段：<字段名>=<值>）
复算：<上面 E1~E8 中的一条命令>
身份：<rtl_sha256/manifest_sha256/bit_sha256>
时间：<YYYY-MM-DD HH:MM CST>；执行者：<角色>
```

## E10 不可执行命令的标注

本机无 Vivado/ModelSim/板卡时，相关命令一律标 ，并在结论里写依赖与判据，
不得用"应该能过"替代；参考 `reference/gates.md` 的门级状态。
