"""Read saved progress and the active directory lock; never control another task."""
import argparse
import fcntl
from pathlib import Path
from common import ROOT, read_json

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default=str(ROOT/'results'))
    a = p.parse_args()
    root = Path(a.out)
    if not root.exists():
        raise SystemExit('Results folder not created yet. Check background.log for startup errors.')
    with open(root/'runner.lock', 'a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = False
            fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            active = True
    print('Runner or training child active:', active)
    plan = read_json(root/'plan.json') if (root/'plan.json').exists() else {'total':36, 'jobs':[]}
    done = sum((root/j['id']/'result.json').exists() for j in plan['jobs'])
    print('Completed: %d/%d' % (done, plan['total']))
    if (root/'progress.json').exists():
        print('Saved progress:', read_json(root/'progress.json'))
    files = list(root.glob('*/status.json'))
    if files:
        latest = max(files, key=lambda q:q.stat().st_mtime)
        print('Latest job:', latest.parent.name, read_json(latest))
    if (root/'STOP').exists():
        print('STOP marker present. Remove it before resuming.')
    if (root/'last_error.json').exists():
        print('Recorded error (may predate a successful retry):', read_json(root/'last_error.json')['error'])
    archive = root/'IISE_uniform_return.zip'
    print('Return zip:', str(archive) if archive.exists() else 'not created yet')
