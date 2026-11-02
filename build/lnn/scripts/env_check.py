#!/usr/bin/env python3
"""Release environment check for the LNN CfC PYNQ demo snapshot.

Checks the Python/board tooling needed by this release and, with --check-hashes,
verifies every file listed in MANIFEST.sha256.

    python scripts/env_check.py                 # tool/env report
    python scripts/env_check.py --check-hashes  # also verify MANIFEST.sha256
"""
import hashlib
import os
import platform
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return "-"


def main(argv):
    check = "--check-hashes" in argv
    print("release : LNN CfC PYNQ demo snapshot")
    print("root    :", ROOT)
    print("python  :", sys.version.split()[0], "|", platform.system(), platform.machine())
    print("numpy   :", ver("numpy"), "(required by the driver/notebook)")
    print("matplotlib:", ver("matplotlib"), "(notebook plots)")
    print("pynq    :", "present" if has("pynq") else "absent (only needed on the PYNQ board)")
    print("torch   :", ver("torch"), "(optional; only for model retraining)")
    print("bitfile :", "present" if os.path.isfile(os.path.join(ROOT, "board", "lnn_cfc.bit")) else "MISSING")
    print("export  :", "present" if os.path.isfile(os.path.join(ROOT, "export", "weights.hex")) else "MISSING")

    rc = 0
    if check:
        man = os.path.join(ROOT, "MANIFEST.sha256")
        if not os.path.isfile(man):
            print("hashes  : MANIFEST.sha256 missing")
            return 1
        bad = 0
        total = 0
        with open(man, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                digest, _, rel = line.partition(" ")
                rel = rel.lstrip("*")
                total += 1
                path = os.path.join(ROOT, rel.replace("/", os.sep))
                if not os.path.isfile(path):
                    print("  MISSING", rel)
                    bad += 1
                    continue
                h = hashlib.sha256()
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != digest:
                    print("  MISMATCH", rel)
                    bad += 1
        print("hashes  : %d files checked, %d bad" % (total, bad))
        rc = 0 if bad == 0 else 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
