# -*- coding: utf-8 -*-
'''Environment check for the released toolchain.'''
import importlib, shutil, sys

def check_module(name, label):
    try:
        mod = importlib.import_module(name)
        ver = getattr(mod, '__version__', '?')
        print('%-10s OK   %s' % (label, ver))
        return True
    except Exception as exc:
        print('%-10s FAIL %s' % (label, exc))
        return False

def main():
    ok = True
    print('python', sys.version.split()[0])
    ok = check_module('numpy', 'numpy') and ok
    ok = check_module('onnx', 'onnx') and ok
    ok = check_module('PIL', 'pillow') and ok
    for tool in ('vsim', 'vlog', 'vlib', 'vivado', 'xvlog', 'xelab', 'xsim'):
        path = shutil.which(tool)
        print('%-10s %s' % (tool, path if path else 'not in PATH (optional)'))
    print('ENV_CHECK:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1

if __name__ == '__main__':
    sys.exit(main())
