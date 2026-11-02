"""Heavy runner: thin adapter over D:/dl_v0.1/verification/host/heavy_lock.py.

Discipline (48H handoff, section "build/platform shortest path"):
  * the CURRENT shared absolute lock path is acquired by this resident process;
  * the tool starts ONLY when acquire() really returned status == "acquired";
  * the same lease (job_id + lease_id) is released in finally;
  * lock busy / resources short -> record the waiting row and return 1,
    the tool is NOT started;
  * never a private per-SNN-run lock for a heavy start, never kill anything.
Usage:
  run_heavy.py --lock <abs lock> --job-id <id> [--category vivado_ooc]
               [--wait-sec N] [--poll-sec N] [--mem-mib N] -- <tool cmd...>
Without --lock only prints the queued command (does NOT start the tool).
"""
from __future__ import annotations
import argparse, json, subprocess, sys

MARKER = '--'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--lock', default=None)
    ap.add_argument('--job-id', default='spikformer-nlp-48h')
    ap.add_argument('--category', default='vivado_ooc')
    ap.add_argument('--wait-sec', type=int, default=0)
    ap.add_argument('--poll-sec', type=int, default=15)
    ap.add_argument('--mem-mib', type=int, default=None)
    ap.add_argument('--host-tools', default='D:/dl_v0.1/verification/host')
    ap.add_argument('tool', nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if not a.lock or not a.tool or a.tool[0] != MARKER or len(a.tool) < 2:
        print('queued (no heavy start): shared lock path and a tool command are required;')
        print('pass --lock <area-task absolute lock> -- <cmd...>')
        return 0
    sys.path.insert(0, a.host_tools)
    import heavy_lock
    cmd = a.tool[1:]
    lease = heavy_lock.acquire(lock_path=a.lock, job_id=a.job_id,
                               category=a.category, command=' '.join(cmd),
                               mem_mib=a.mem_mib, wait_sec=a.wait_sec,
                               poll_sec=a.poll_sec)
    if lease.get('status') != 'acquired':
        # P0: the old adapter ignored this return value and started the tool anyway.
        print('HEAVY LOCK NOT ACQUIRED: status=%r' % lease.get('status'))
        print('blockers: ' + json.dumps(lease.get('blockers', []), ensure_ascii=False))
        print('refusing to start the tool without a lease: ' + ' '.join(cmd))
        return 1
    lease_id = lease.get('lease_id')
    print('lock acquired job_id=%s lease_id=%s' % (a.job_id, lease_id))
    try:
        print('starting: ' + ' '.join(cmd))
        r = subprocess.run(cmd)
        print('tool exit code: %d' % r.returncode)
        return r.returncode
    finally:
        rel = heavy_lock.release(lock_path=a.lock, job_id=a.job_id, lease_id=lease_id)
        print('lock release: ' + json.dumps(rel, ensure_ascii=False))


if __name__ == '__main__':
    sys.exit(main())
