# 接口契约 → 主机骨架 / 测试骨架 / 报告（通用）

> 目标：从一份机器可读的"接口契约"自动生成板端驱动骨架、测试骨架和结果报告，减少手抄寄存器表造成的错位。
> 与题目无关：契约描述任意 IP 即可。

## 1 契约文件（registers.json）

```json
{
  "ip_name": "my_core_0",
  "base_addr": "0x480000000",
  "addr_range": "0x1000",
  "clock_hz": 300000000,
  "registers": [
    {"name": "ID",         "offset": "0x000", "width": 32, "access": "ro"},
    {"name": "VERSION",    "offset": "0x004", "width": 32, "access": "ro"},
    {"name": "CONTROL",    "offset": "0x008", "width": 32, "access": "rw", "reset": 0, "fields": {"START": 0, "SOFT_RESET": 1}},
    {"name": "STATUS",     "offset": "0x00C", "width": 32, "access": "ro", "fields": {"BUSY": 0, "DONE": 1}},
    {"name": "ERROR_CODE", "offset": "0x010", "width": 32, "access": "ro"},
    {"name": "RUN_CYCLES", "offset": "0x020", "width": 32, "access": "ro"},
    {"name": "CLOCK_HZ",   "offset": "0x02C", "width": 32, "access": "ro"}
  ],
  "buffers": [
    {"name": "input",  "direction": "ps_to_pl", "dtype": "int16", "words": 64, "align_bytes": 4096},
    {"name": "output", "direction": "pl_to_ps", "dtype": "int16", "words": 64, "align_bytes": 4096}
  ],
  "dma": {"channels": ["MM2S", "S2MM"], "max_burst_bytes": 4096}
}
```

来源优先级：hwh 元数据（地址/时钟/范围）+ 驱动常量表 + BD 导出说明；三处不一致时以构建方记录为准并记入 `notes`。

## 2 生成驱动骨架（伪代码模板）

```python
# 由 registers.json 生成：寄存器常量 + 读写方法 + 状态轮询 + 超时
CSR = {r["name"]: int(r["offset"], 16) for r in CONTRACT["registers"]}
CLOCK_HZ = CONTRACT["clock_hz"]

class Core:
    def __init__(self, ol):
        self.csr = ol[CONTRACT["ip_name"]]
    def rd(self, name):  return int(self.csr.read(CSR[name])) & 0xFFFFFFFF
    def wr(self, name, v): self.csr.write(CSR[name], int(v) & 0xFFFFFFFF)
    def rmw(self, name, **fields):
        v = self.rd(name)
        for bit, on in fields.items():
            v = (v | (1 << bit)) if on else (v & ~(1 << bit))
        self.wr(name, v)
    def wait_done(self, timeout_s):
        import time
        t0 = time.perf_counter()
        while self.rd("STATUS") & (1 << 1) == 0:
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError("DONE timeout, STATUS=%#x ERROR=%#x" % (self.rd("STATUS"), self.rd("ERROR_CODE")))
```

生成器必须保证：偏移来自契约、位域来自契约、超时值显式传入（不许硬编码 5s）。

## 3 生成测试骨架

```python
def test_smoke(core):
    assert core.rd("ID") == EXPECTED_ID
    assert core.rd("VERSION") == EXPECTED_VERSION
    assert core.rd("CLOCK_HZ") == CLOCK_HZ
    core.rmw("CONTROL", START=1); core.wait_done(5.0)
    assert core.rd("ERROR_CODE") == 0

def test_golden(core, golden):
    out = run_once(core, golden["input"])          # 缓冲区 + flush/invalidate 见 C6
    bad = first_mismatch(golden["expected"], out)
    assert bad is None, "first_mismatch=%s" % (bad,)
```

骨架只生成结构与断言，不生成"期望值"；期望值必须来自 golden/vendor 参考。

## 4 自动汇总报告（golden + 性能 + 资源）

输入：板端 `summary.json`（本技能 §5）+ Vivado `utilization.rpt`/`timing_summary.rpt`（可选，PC 侧）。
输出：`report.md`，三段：

| 段 | 内容 | 口径标注 |
|---|---|---|
| 功能 | 样本数、matched、first_mismatch、error_code | measured |
| 性能 | `hw_cycles`→`hw_time_s`、`wall_s`（标明"端到端，含 PS 提交/轮询"） | measured |
| 资源/时序 | LUT/FF/BRAM/DSP、WNS/WHS | measured（来自 Vivado 报告，不是板端） |

红线：板端只出功能与耗时；资源/时序只引用实现报告，不得从板端反推。

## 5 生成后校验（必做）

1. 生成的偏移/位域与 hwh 地址窗、驱动原表逐项比对；
2. 冒烟三步（ID/VERSION/CLOCK）通过再跑 golden；
3. 生成器与契约文件一起入库（谁生成、基于哪版契约）。
