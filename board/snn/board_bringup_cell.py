# === SNN 平台上板 bring-up（一个 cell 直接跑）===
# 前提：当前目录包含 snn_platform.bit / snn_platform.hwh / vec/case_real/
import time
from pathlib import Path
import numpy as np
from pynq import Overlay, MMIO

CTRL_BASE = 0x4_8000_0000      # SNN AXI-Lite CSR 窗口
DDR_BASE  = 0x4_0000_0000      # PL DDR4 PS 映射窗口（1 GiB 中的前 64 MiB）
IN_OFF, W_OFF, OUT_OFF = 0x0100_0000, 0x0200_0000, 0x0300_0000
TAB_BYTES, TOK_BYTES, W_BYTES = 516096, 64, 402496

CSR = {'ID':0x000,'VERSION':0x004,'CONTROL':0x008,'STATUS':0x00C,'ERROR':0x010,
       'IN_LO':0x014,'IN_HI':0x018,'W_LO':0x01C,'W_HI':0x020,'OUT_LO':0x024,'OUT_HI':0x028,
       'TAB_BYTES':0x02C,'TOK_BYTES':0x030,'W_BYTES':0x034,'TIMEOUT':0x038,
       'RUN_CYCLES':0x03C,'SUMS':0x100}
START, SOFT_RESET, CLR_DONE, CLR_ERR, ACK = 1, 2, 4, 8, 16

overlay = Overlay('snn_platform.bit')
c = MMIO(CTRL_BASE, 0x1000)
d = MMIO(DDR_BASE, 0x0400_0000)

def rd(off):
    return int(c.read(off)) & 0xFFFFFFFF

def wr(off, val):
    c.write(off, int(val) & 0xFFFFFFFF)

def hexw(path, dtype):
    vals = [int(x, 16) for x in Path(path).read_text().split()]
    return np.asarray(vals, dtype=dtype)

def u32(words):
    w = np.ascontiguousarray(words)
    if w.dtype == np.uint64:
        return w.view(np.uint32)              # 64-bit → 低 32 位在前
    if w.dtype == np.uint16:
        return np.frombuffer(w.tobytes(), dtype=np.uint32)  # 两个 token 合一字
    return w.astype(np.uint32)

print('ID=0x%08X VERSION=0x%08X' % (rd(CSR['ID']), rd(CSR['VERSION'])))
assert rd(CSR['ID']) == 0x534E4E31 and rd(CSR['VERSION']) == 0x00020000

vec = Path('vec/case_real')
tab = np.concatenate([hexw(vec/'etab.hex', np.uint64), hexw(vec/'ptab.hex', np.uint64)])
assert tab.size * 8 == TAB_BYTES, tab.size
d.write(IN_OFF, u32(tab).tobytes())

for s in range(3):
    toks = hexw(vec/('toks_s%d.hex' % s), np.uint16)
    wt = np.concatenate([hexw(vec/('stream_b0_s%d.hex' % s), np.uint64),
                         hexw(vec/('stream_b1_s%d.hex' % s), np.uint64)])
    assert toks.size * 2 == TOK_BYTES and wt.size * 8 == W_BYTES
    d.write(IN_OFF + TAB_BYTES, u32(toks).tobytes())
    d.write(W_OFF, u32(wt).tobytes())

    for name, val in (('IN', IN_OFF), ('W', W_OFF), ('OUT', OUT_OFF)):
        wr(CSR[name + '_LO'], val); wr(CSR[name + '_HI'], 0)
    wr(CSR['TAB_BYTES'], TAB_BYTES); wr(CSR['TOK_BYTES'], TOK_BYTES)
    wr(CSR['W_BYTES'], W_BYTES); wr(CSR['TIMEOUT'], 12_000_000)

    wr(CSR['CONTROL'], CLR_DONE | CLR_ERR)   # 清上一帧 sticky 状态
    wr(CSR['CONTROL'], 0)
    t0 = time.time()
    wr(CSR['CONTROL'], START)
    while True:
        st = rd(CSR['STATUS'])
        if st & 0x4:
            raise RuntimeError('PL error code=%d' % (rd(CSR['ERROR']) & 0xF))
        if st & 0x2:
            break
        if time.time() - t0 > 120:
            wr(CSR['CONTROL'], SOFT_RESET)
            raise TimeoutError('PL timeout')
        time.sleep(0.002)
    cycles = rd(CSR['RUN_CYCLES'])
    wr(CSR['CONTROL'], ACK)

    sums = np.asarray([rd(CSR['SUMS'] + 4*i) for i in range(128)], dtype=np.uint32)
    exp = hexw(vec/('exp_sums_s%d.hex' % s), np.uint32)
    bad = np.where(sums != exp)[0]
    print('sample %d: match=%s cycles=%d model_ms=%.2f wall_ms=%.1f first_bad=%s'
          % (s, bad.size == 0, cycles, cycles/2e8*1e3, (time.time()-t0)*1e3, bad[:5]))

print('done')