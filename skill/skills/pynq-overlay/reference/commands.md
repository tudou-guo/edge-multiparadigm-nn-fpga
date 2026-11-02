# 可复制命令矩阵（PYNQ overlay）

> 板端命令以 `xilinx@<board>` 为例；PC 侧命令在包根或任意目录执行。
> 占位符：`<ip_name>` IP 实例名、`<run>` 本次运行 id、`<board_ip>` 板卡地址。

## C1 环境自检（板端）

```bash
ssh xilinx@<board_ip>
python3 -c "import pynq, sys; print('pynq', pynq.__version__, 'python', sys.version.split()[0])"
python3 -c "from pynq import PL; print('device', PL.device_name)"
ls -l *.bit *.hwh *.xsa 2>/dev/null
```

期望：PYNQ ≥2.7、device 与 bit 器件一致（如 `xczu15eg`）、三件套文件字节数非零。

## C2 三件套配对（PC 侧）

```bash
sha256sum overlay.bit overlay.hwh
python - <<'PY'
import hashlib, zipfile
z = zipfile.ZipFile('overlay.xsa')
for name in ('overlay.bit', 'design_1.hwh'):
    print(name, hashlib.sha256(z.read(name)).hexdigest())
PY
```

判据：XSA 内嵌 bit/hwh 与外部文件哈希一致；不一致即不同源。

## C3 hwh 元数据静态核对

```bash
grep -o 'DEVICE="[^"]*" PACKAGE="[^"]*" SPEEDGRADE="[^"]*"' overlay.hwh | head -3
grep -o 'CLKFREQUENCY="[0-9]*"[^/]*NAME="[^"]*"' overlay.hwh | head -5
grep -o 'BASEVALUE="0x[0-9A-Fa-f]*"[^/]*RANGE="[^"]*"' overlay.hwh | head -10
```

判据：器件/时钟/地址窗与驱动常量、BD 记录一致；地址是 64 位（> 0xFFFFFFFF）时驱动必须按 64 位处理。

## C4 加载 overlay 并列出 IP

```python
from pynq import Overlay
ol = Overlay("/home/xilinx/overlay.bit")
for name, info in ol.ip_dict.items():
    print(name, hex(info["phys_addr"]), hex(info["addr_range"]))
```

判据：出现 `<ip_name>`，地址/范围与 hwh 一致。

## C5 寄存器读 / 读-改-写

```python
csr = ol.<ip_name>
csr_id = csr.read(0x000)
assert csr_id == EXPECTED_ID, hex(csr_id)
v = csr.read(0x008)
csr.write(0x008, (v | (1 << 0)) & 0xFFFFFFFF)     # 读-改-写，避免覆盖其它位
```

判据：写后回读一致（只读位/写 1 清零位按契约跳过）。

## C6 缓冲区分配 + DMA（带缓存一致性）

```python
import numpy as np
from pynq import allocate
buf = allocate(shape=(n_words,), dtype=np.uint32)
buf[:] = words
buf.flush()                                       # 非相干口：CPU → PL
csr.write(0x010, buf.physical_address & 0xFFFFFFFF)
csr.write(0x008, 0x1)                             # START
wait_done(csr, timeout_s=5.0)                     # 轮询 DONE 位或中断
buf.invalidate()                                  # 非相干口：PL → CPU
out = np.array(buf, copy=True)
```

要点：对齐（建议 ≥64 B / 4 KiB）、长度、跨边界；超时值要大于最坏用例。

## C7 黄金比对（带 first_mismatch）

```python
import numpy as np
exp = np.load("golden/exp.npy"); got = np.load("results/<run>/outputs.npy")
bad = np.flatnonzero(exp != got)
print("matched=%d/%d first_mismatch=%s" % (len(exp) - len(bad), len(exp), bad[:1]))
```

判据：`first_mismatch` 为空且 `error_code=0`；有差异时输出期望/实际的原始字。

## C8 性能采集（两种口径）

```python
import time
t0 = time.perf_counter(); start(); wait_done(csr); t1 = time.perf_counter()
wall = t1 - t0
cycles = csr.read(0x020)                          # RUN_CYCLES（若 IP 提供）
print("wall_s=%.6f hw_cycles=%d hw_s=%.6f" % (wall, cycles, cycles / CLOCK_HZ))
```

要点：至少 3 次取中位数；wall 与 hw 口径分开写，不混加。

## C9 证据归档

```python
import json, hashlib, pathlib
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
summary = {
    "source": "hardware", "is_hardware_measurement": True,
    "pynq_version": "2.7", "platform": "xczu15eg",
    "bit_sha256": sha("overlay.bit"), "hwh_sha256": sha("overlay.hwh"),
    "id_check": True, "version_check": True, "clock_hz_check": True,
    "samples": 1, "matched": 64, "first_mismatch": None, "error_code": 0,
    "wall_s": 0.0, "hw_cycles": 9088, "hw_time_s": 0.0,
}
pathlib.Path("results/<run>/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
```

## C10 离线自检（无板卡时）

```bash
sha256sum overlay.bit overlay.hwh && grep -c 'MEMRANGE' overlay.hwh
python -c "import json;d=json.load(open('results/<run>/summary.json'));print(d['source']=='hardware')"
```

判据：离线只能核对哈希/元数据/摘要字段，不能声明板端通过。

## C11 依赖探测（板端）

```bash
python3 -c "import pynq, numpy; print('ok', pynq.__version__, numpy.__version__)" || echo "missing pynq/numpy"
ls /usr/local/share/pynq-venv 2>/dev/null | head -3
```
