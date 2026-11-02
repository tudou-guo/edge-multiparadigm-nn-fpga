"""Runtime dry-run with scripted fake MMIO (no board, no PYNQ)."""
import os, pathlib, sys
ROOT = pathlib.Path(os.environ.get('SNN_ROOT') or pathlib.Path(__file__).resolve().parents[3]).resolve()
sys.path.insert(0, str(ROOT / 'runtime/spikformer'))
import run as R
class FakeMMIO:
    def __init__(self):
        self.reg = {}
        self.reads = 0
    def write(self, off, v):
        self.reg[off] = v
        if off == R.CSR['CONTROL'] and v == 1:
            self.reg[R.CSR['STATUS']] = 1
    def read(self, off):
        self.reads += 1
        if off == R.CSR['STATUS'] and self.reads > 4:
            return 0b010
        return self.reg.get(off, 0)
fails = 0
def check(n, c, info=''):
    global fails
    print(('PASS ' if c else 'FAIL ') + n + (' ' + str(info) if info else ''))
    if not c:
        fails += 1
try:
    m = FakeMMIO()
    r = R.SpikformerRunner(m)
    out = r.run_sample(0x1000, 0x2000, 0x70000000, 0x100000, 0, timeout_s=5.0, tag=7)
    check('runtime.sample', out['out_base'] == 0x2000)
    check('runtime.ack', m.reg.get(R.CSR['ACK']) == 1)
    import head as H
    import numpy as np
    sums = np.array([10.0, 5.0, 5.0, 5.0] * 96)
    W = np.zeros((10, 384), dtype=np.float32)
    W[0, :] = 0.01
    W[1, :] = 0.1
    b = np.zeros(10, dtype=np.float32)
    cls, y = H.ps_head(sums, W, b, 4, 64, norm='ps_div')
    check('runtime.head', cls == 1, (cls, float(y[1])))
except Exception as e:
    check('runtime.nocrash ' + repr(e), False)
print('FAILURES: %d' % fails)
sys.exit(1 if fails else 0)
