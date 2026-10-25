"""Verify, run/resume, summarize and automatically zip the 36 prespecified jobs."""
import argparse
from collections import defaultdict
import csv
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np
from common import ROOT, sha, read_json, write_json, fingerprint
from plan import make_plan
from uniform_preprocess import validate

STOP = False
CHILD = None


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print('Pause requested; saving the current epoch first.', flush=True)
    if CHILD is not None and CHILD.poll() is None:
        CHILD.send_signal(signal.SIGTERM)


def summarize(results, plan):
    results = Path(results)
    rows, groups = [], defaultdict(list)
    baseline = read_json(ROOT / 'baseline_reference.json')['runs']
    for job in plan['jobs']:
        path = results / job['id'] / 'result.json'
        if not path.exists():
            continue
        r = read_json(path)
        if r['job'] != job or r['epochs_completed'] != job['epochs']:
            raise ValueError('Completed result does not match frozen plan')
        row = dict(job, selected_epoch=r['selected_epoch'])
        for split, met in r['metrics'].items():
            for key in ['oa', 'ba', 'macro_f1', 'cross_entropy']:
                row[split + '_' + key] = met[key]
        for key in ['oa', 'ba', 'macro_f1']:
            old = baseline.get(job['baseline_id'])
            row['original_test_' + key] = None if old is None else old['metrics']['test'][key]
            row['delta_test_' + key] = None if old is None else row['test_' + key] - row['original_test_' + key]
        rows.append(row)
        group = {k:job[k] for k in ['dataset', 'model', 'variant', 'k', 'tau', 'protocol', 'batch_size', 'epochs']}
        groups[fingerprint(group)].append((group, row))
    summary = []
    for values in groups.values():
        group = values[0][0]
        item = dict(group, seeds_completed=[r['seed'] for _, r in values],
                    complete=sorted(r['seed'] for _, r in values) == sorted(plan['seeds']))
        for metric in ['train_oa', 'val_oa', 'test_oa', 'test_ba', 'test_macro_f1',
                       'delta_test_oa', 'delta_test_ba', 'delta_test_macro_f1']:
            data = [r[metric] for _, r in values if r[metric] is not None]
            item[metric + '_mean'] = float(np.mean(data)) if data else None
            item[metric + '_sd'] = float(np.std(data, ddof=1)) if len(data) > 1 else None
        summary.append(item)
    report = dict(completed=len(rows), planned=plan['total'], complete=len(rows) == plan['total'],
                  units='OA/BA/F1 and differences are fractions; multiply by 100 for percent/percentage points',
                  groups=summary)
    write_json(results / 'summary.json', report)
    if rows:
        with open(results / 'per_run.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return len(rows)


def package(results, plan, full=False):
    results = Path(results)
    summarize(results, plan)
    name = 'IISE_uniform_full_backup.zip' if full else 'IISE_uniform_return.zip'
    target = results / name
    temp = target.with_suffix('.zip.tmp')
    files, missing = [], []
    for p in sorted(results.rglob('*')):
        if not p.is_file() or p.suffix in ['.zip', '.tmp'] or p.name.endswith('.lock'):
            continue
        if p.suffix == '.pt' and not full:
            continue
        files.append((p, 'results/' + str(p.relative_to(results))))
    for p in sorted(ROOT.glob('*.py')):
        files.append((p, 'program/' + p.name))
    for rel in ['README_中文.md', 'assembly_manifest.json', 'protocol.json', 'experiment_plan.json',
                'baseline_reference.json', 'release_manifest.json', 'fixed_splits', 'audit']:
        p = ROOT / rel
        if p.is_file():
            files.append((p, rel))
        elif p.is_dir():
            files.extend((q, str(q.relative_to(ROOT))) for q in sorted(p.rglob('*')) if q.is_file())
    for ds in ['surface', 'real']:
        for name in ['manifest.json', 'mask_diagnostics.json']:
            p = ROOT / 'data' / ds / name
            if p.is_file():
                files.append((p, str(p.relative_to(ROOT))))
            else:
                missing.append(str(p.relative_to(ROOT)))
    checks = {}
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p, rel in files:
            z.write(p, rel)
            checks[rel] = {'sha256':sha(p), 'bytes':p.stat().st_size}
        z.writestr('archive_manifest.json', json.dumps(dict(full_weights_included=full,
                   program='IISE_uniform_preprocess_v1', missing_expected_files=missing, files=checks), indent=2))
    os.replace(temp, target)
    print('Saved:', target, flush=True)
    return target


def validate_release():
    manifest = read_json(ROOT / 'release_manifest.json')
    for rel, expected in manifest['files'].items():
        p = ROOT / rel
        if not p.is_file() or sha(p) != expected['sha256']:
            raise ValueError('Released file missing or changed: ' + rel + '. Keep this package unchanged; return diagnostic zip.')
    report = validate(ROOT / 'data')
    for ds in ['surface', 'real']:
        rows = read_json(ROOT / 'data' / ds / 'manifest.json')['records']
        actual = {r['id']:r['split'] for r in rows}
        if actual != read_json(ROOT / 'fixed_splits' / (ds + '.json')):
            raise ValueError('Original fixed split changed: ' + ds)
    return report


def main():
    global CHILD
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'smoke', 'pack', 'plan'], default='all')
    parser.add_argument('--out', default=str(ROOT / 'results'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--keep-every', type=int, default=50)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--full-backup', action='store_true')
    a = parser.parse_args()
    out = Path(a.out).resolve()
    if out == ROOT or out in [ROOT/'data', ROOT/'audit'] or ROOT/'data' in out.parents:
        raise ValueError('Results must use a separate results folder')
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / 'runner.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('This results folder already has an active runner or training child. No duplicate task was started.')
    plan = make_plan(a.batch_size, a.epochs)
    path = out / 'plan.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('Plan differs. Use a NEW --out directory; never resume with changed parameters.')
    write_json(path, plan)
    print('Uniform preprocessing check:', plan['total'], 'runs; batch', a.batch_size,
          'epochs', a.epochs, flush=True)
    if a.stage == 'plan':
        return
    if a.stage == 'pack':
        package(out, plan)
        if a.full_backup:
            package(out, plan, full=True)
        return
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        if (out / 'STOP').exists():
            raise ValueError('STOP marker exists. Remove it before resuming.')
        data_report = validate_release()
        write_json(out / 'preprocessing_check.json', {'passed':True, 'datasets':data_report})
        write_json(out / 'environment.json', {'python':sys.version, 'platform':platform.platform(),
                   'command':sys.argv, 'programs':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}})
        from smoke import smoke
        signature = fingerprint({'plan':plan, 'device':a.device, 'threads':a.threads,
                    'release':sha(ROOT/'release_manifest.json')})
        smoke_path = out / 'smoke.json'
        if not smoke_path.exists() or not read_json(smoke_path).get('complete') or read_json(smoke_path).get('signature') != signature:
            smoke(plan, ROOT/'data', smoke_path, a.device, a.threads,
                  stop_requested=lambda:STOP or (out/'STOP').exists())
            report = read_json(smoke_path)
            report['signature'] = signature
            write_json(smoke_path, report)
        if a.stage == 'smoke':
            print('SMOKE PASSED. Formal experiments have not been run.', flush=True)
            return
        for index, job in enumerate(plan['jobs'], 1):
            if STOP or (out/'STOP').exists():
                break
            folder = out / job['id']
            folder.mkdir(exist_ok=True)
            write_json(folder/'job.json', job)
            if (folder/'result.json').exists():
                cfg, r = read_json(folder/'config.json'), read_json(folder/'result.json')
                if cfg['job'] != job or r['job'] != job or cfg['run_hash'] != r['run_hash']:
                    raise ValueError('Completed job identity differs')
                if cfg['data_manifest'] != sha(ROOT/'data'/job['dataset']/'manifest.json'):
                    raise ValueError('Completed data identity differs')
                if any(sha(ROOT/n) != h for n, h in cfg['programs'].items()):
                    raise ValueError('Completed model/training code changed')
                if cfg['device_type'] != a.device.split(':')[0] or cfg['threads'] != a.threads:
                    raise ValueError('Completed device/thread settings differ')
                if sha(folder/'selected.pt') != r['selected_checkpoint_sha256']:
                    raise ValueError('Selected checkpoint missing or changed')
                print('SKIP completed:', job['id'], flush=True)
                continue
            if (folder/'failure.json').exists() and not a.retry_failed:
                raise ValueError('Previous failure found. Return zip for diagnosis; after fixing environment, use --retry-failed with unchanged code and settings.')
            print('JOB %d/%d: %s' % (index, plan['total'], job['id']), flush=True)
            write_json(out/'progress.json', {'state':'running', 'job':job['id'], 'index':index,
                       'planned':plan['total'], 'time':time.time()})
            cmd = [sys.executable, '-u', str(ROOT/'train.py'), '--job', str(folder/'job.json'),
                   '--data', str(ROOT/'data'), '--out', str(folder), '--device', a.device,
                   '--threads', str(a.threads), '--keep-every', str(a.keep_every)]
            with open(folder/'console.log', 'a', buffering=1, encoding='utf-8') as log:
                # The child holds the same lock even if the controller is killed.
                CHILD = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, bufsize=1, pass_fds=(lock.fileno(),))
                for line in CHILD.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                code = CHILD.wait()
                CHILD = None
            if code == 75:
                break
            if code != 0:
                write_json(folder/'failure.json', {'returncode':code, 'job':job, 'time':time.time()})
                raise RuntimeError('Training stopped: ' + job['id'] + '; see console.log. Batch size was not changed.')
            if (folder/'failure.json').exists():
                os.replace(folder/'failure.json', folder/'resolved_failure.json')
            package(out, plan)
        done = summarize(out, plan)
        state = 'complete' if done == plan['total'] else 'paused'
        write_json(out/'progress.json', {'state':state, 'completed':done, 'planned':plan['total'], 'time':time.time()})
        print('COMPLETE %d/%d' % (done, plan['total']) if state == 'complete' else 'PAUSED: %d/%d completed' % (done, plan['total']), flush=True)
    except BaseException as e:
        write_json(out/'last_error.json', {'type':type(e).__name__, 'error':str(e),
                   'traceback':traceback.format_exc(), 'time':time.time()})
        raise
    finally:
        package(out, plan)
        if a.full_backup:
            package(out, plan, full=True)


if __name__ == '__main__':
    main()
