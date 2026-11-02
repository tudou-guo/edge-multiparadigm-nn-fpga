"""Package: manifest -> board runtime bundle (no Vivado, no board needed).
Bundle: runtime_config.json (CSR/windows/head contract), params manifest copy,
README bring-up. HEADROOM: real bit/HWH are added by the platform stage."""
from __future__ import annotations
import json, pathlib, shutil
CSR = {'ID': 0x00, 'VERSION': 0x04, 'CONTROL': 0x08, 'STATUS': 0x0C,
       'ERROR': 0x10, 'IN_BASE': 0x14, 'OUT_BASE': 0x1C, 'W_BASE': 0x20,
       'W_STRIDE': 0x24, 'DESC': 0x28, 'TIMEOUT': 0x2C, 'ACK': 0x30, 'TAG': 0x34}
def run(manifest_path, out_dir):
    man = json.loads(pathlib.Path(manifest_path).read_text(encoding='utf-8-sig'))
    out = pathlib.Path(out_dir)
    if (out / 'runtime_config.json').exists():
        raise RuntimeError('refusing to overwrite package dir: ' + str(out))
    (out / 'params').mkdir(parents=True, exist_ok=True)
    src = pathlib.Path(manifest_path).parent
    n = 0
    for name, meta in man['params'].items():
        s = src / 'params' / meta['bin']
        if s.exists():
            shutil.copyfile(s, out / 'params' / meta['bin'])
            n += 1
    cfg = {
        'backend': 'spikformer',
        'model_kind': man['model_kind'],
        'manifest_sha256': man['manifest_sha256'],
        'csr': CSR,
        'derived': man['derived'],
        'schedule': man['schedule'],
        'head': man['config']['head'],
        'target': man['target'],
        'param_files_copied': n,
        'bitstream': 'NOT_BUILT (platform stage, shared heavy lock)',
        'hwh': 'NOT_BUILT (platform stage, shared heavy lock)',
    }
    (out / 'runtime_config.json').write_text(json.dumps(cfg, indent=1))
    (out / 'README.txt').write_text(
        'SNN board bundle (fixture or real, see runtime_config.json model_kind).\n'
        'Bring-up: program bitstream (platform stage) -> load params to DDR windows ->\n'
        'run runtime/spikformer/run.py via PYNQ MMIO -> PS head -> argmax.\n')
    print('packaged %d param files -> %s' % (n, out))
    return str(out / 'runtime_config.json')
