#!/usr/bin/env python3
"""Evaluate a SHA256-matched selected checkpoint with the frozen experiment code.

Python 3.8 compatible. Listing and file checks use only the standard library.
This command never trains, chooses a checkpoint, or replaces retained results.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

LAYOUT = {
    'complete': ('complete_return/results', 'result.json'),
    'uniform': ('uniform_return/results', 'result.json'),
    'components': ('component_return/gpu_results', 'final_metrics.json'),
}
SPLITS = ('train', 'val', 'test')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_child(folder, relative):
    path = (Path(folder) / relative).resolve()
    try:
        path.relative_to(Path(folder).resolve())
    except ValueError:
        raise ValueError('A manifest path leaves its data directory: ' + str(relative))
    return path


def inventory(root):
    """Use only formal run keys already included in the statistics audit."""
    rows = read_json(root / 'recalculated_statistics/runs_recalculated.json')
    found = {}
    for row in rows:
        family, run_id = row['family'], row['run_id']
        if family not in LAYOUT or Path(run_id).name != run_id:
            raise ValueError('Invalid audited run key')
        key = family + '/' + run_id
        if key in found:
            raise ValueError('Duplicate audited run key: ' + key)
        found[key] = row
    return found


def load_run(root, family, run_id):
    key = family + '/' + run_id
    if key not in inventory(root):
        raise ValueError('Run is not in the audited results index: ' + key)
    sub, name = LAYOUT[family]
    folder = root / 'retained_results' / sub / run_id
    config, result = read_json(folder / 'config.json'), read_json(folder / name)
    if family != 'components':
        if config['job'] != result['job'] or result['job']['id'] != run_id:
            raise ValueError('Job identity differs between config and result')
        if config['run_hash'] != result['run_hash']:
            raise ValueError('Run hash differs between config and result')
        if result['epochs_completed'] != config['job']['epochs']:
            raise ValueError('Training was not completed')
    audited = inventory(root)[key]
    if audited['selected_epoch'] != result['selected_epoch']:
        raise ValueError('Selected epoch differs from the audited results index')
    return folder, config, result


def dataset_name(root, family, config):
    if family != 'components':
        return config['job']['dataset']
    old_hash = config['data_manifest_sha256']
    names = []
    for entry in read_json(root / 'staging_manifest.json')['files']:
        p = Path(entry['candidate_path'])
        if (len(p.parts) == 5 and p.parts[:3] == ('code', 'components', 'data')
                and p.name == 'manifest.json' and entry['source_sha256'] == old_hash):
            names.append(p.parent.name)
    if len(names) != 1:
        raise ValueError('Cannot uniquely map the component dataset from its recorded hash')
    return names[0]


def selected_records(manifest, family, config, split):
    if family == 'components':
        controlled = manifest.get('study_type') != 'original_cloud_classification'
        rho = config['rho']
    else:
        controlled = config['job']['dataset'] == 'controlled'
        rho = config['job']['rho']
    return [r for r in manifest['records'] if r['split'] == split and
            (not controlled or r['category'] == 'normal' or r['rho'] == rho)]


def inspect_inputs(root, family, run_id, config, result, data_root, splits):
    staged = {r['candidate_path']: r for r in
              read_json(root / 'staging_manifest.json')['files']}
    name = dataset_name(root, family, config)
    dataset = data_root / name
    prefix = 'code/' + family + '/data/' + name + '/'
    checks, issues = [], []

    def check_file(path, expected, label):
        row = {'file': label, 'expected_sha256': expected}
        if not path.is_file():
            row['status'] = 'missing'
        else:
            row['actual_sha256'] = sha(path)
            row['status'] = 'matched' if row['actual_sha256'] == expected else 'mismatch'
        checks.append(row)
        if row['status'] != 'matched':
            issues.append(label + ': ' + row['status'])
        return row['status'] == 'matched'

    sources = config['program_sha256'] if family == 'components' else config['programs']
    for name_, expected in sorted(sources.items()):
        check_file(safe_child(root / 'code' / family, name_), expected,
                   'code/' + family + '/' + name_)
    # The loader imports common.py as well; verify this frozen helper when present.
    helper = 'code/' + family + '/common.py'
    if helper in staged and 'common.py' not in sources:
        check_file(root / helper, staged[helper]['candidate_sha256'], helper)

    old_hash = (config['data_manifest_sha256'] if family == 'components'
                else config['data_manifest'])
    manifest_path = dataset / 'manifest.json'
    actual = sha(manifest_path) if manifest_path.is_file() else None
    bridge = staged.get(prefix + 'manifest.json', {})
    transformed = (actual != old_hash and bridge.get('source_sha256') == old_hash
                   and bridge.get('candidate_sha256') == actual and actual is not None)
    manifest_ok = actual == old_hash or transformed
    checks.append({'file': prefix + 'manifest.json', 'status': 'matched' if manifest_ok else 'mismatch',
                   'historical_sha256': old_hash, 'actual_sha256': actual,
                   'accepted_path_sanitization': transformed})
    if not manifest_ok:
        issues.append('Dataset manifest does not match the historical hash or audited path-only conversion')
        return {'ok': False, 'dataset': name, 'checks': checks, 'issues': issues}, None
    manifest = read_json(manifest_path)
    recorded_classes = config['classes'] if family == 'components' else config['class_names']
    if manifest['class_names'] != recorded_classes:
        issues.append('Class order differs from the training configuration')

    metrics = result['splits'] if family == 'components' else result['metrics']
    counts = {}
    records = []
    for split in splits:
        selected = selected_records(manifest, family, config, split)
        counts[split] = len(selected)
        if len(selected) != metrics[split]['n']:
            issues.append('Sample count differs for split ' + split)
        records.extend(selected)
    if family == 'components' or name == 'controlled':
        # These files do not have per-file hashes inside the old data manifest.
        # Their exact input bytes were audited in the release staging inventory.
        for r in records:
            label = prefix + r['file']
            if label not in staged:
                issues.append('No audited input hash: ' + label)
            else:
                check_file(safe_child(dataset, r['file']), staged[label]['candidate_sha256'], label)
    else:
        check_file(dataset / 'inputs.npz', manifest['inputs_sha256'], prefix + 'inputs.npz')
        if config['job']['variant'].endswith('geo'):
            lookup = {r['id']: r for r in manifest['geodesic']}
            for r in records:
                geo = lookup.get(r['id'])
                if geo is None:
                    issues.append('Missing geodesic manifest record for ' + r['id'])
                else:
                    check_file(safe_child(dataset, geo['file']), geo['cache_sha256'], prefix + geo['file'])
    return {'ok': not issues, 'dataset': name, 'split_counts': counts,
            'checks': checks, 'issues': issues}, manifest


def checkpoint_path(args, folder):
    if args.checkpoint:
        return Path(args.checkpoint).expanduser().resolve()
    if args.weights_root:
        # This is the exact layout produced by collect_reproducibility_assets.py.
        return Path(args.weights_root).expanduser().resolve() / args.family / 'runs' / args.run_id / 'selected.pt'
    return folder / 'selected.pt'


def inspect_checkpoint(path, family, result):
    field = 'checkpoint_sha256' if family == 'components' else 'selected_checkpoint_sha256'
    expected = result[field]
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Invalid recorded selected-checkpoint SHA256')
    row = {'expected_sha256': expected, 'status': 'missing', 'deserialized': False}
    if path.is_file():
        row['actual_sha256'] = sha(path)
        row['status'] = 'matched' if row['actual_sha256'] == expected else 'mismatch'
    return row


def import_frozen(root, family):
    source = root / 'code' / family
    # One family per process prevents the two modules named train.py from mixing.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    return importlib.import_module('train_revised' if family == 'components' else 'train')


def compare_rows(reference, current):
    if isinstance(reference, dict):
        reference = reference['rows']
    ref = {r['id']: r for r in reference}
    now = {r['id']: r for r in current}
    same_ids = len(ref) == len(reference) and len(now) == len(current) and set(ref) == set(now)
    result = {'same_sample_ids': same_ids}
    if not same_ids:
        return result
    result['truth_mismatches'] = sum(int(ref[k]['truth']) != int(now[k]['truth']) for k in ref)
    result['prediction_mismatches'] = sum(int(ref[k]['prediction']) != int(now[k]['prediction']) for k in ref)
    result['maximum_probability_absolute_difference'] = max(
        (abs(float(a) - float(b)) for k in ref
         for a, b in zip(ref[k]['probabilities'], now[k]['probabilities'])), default=0.0)
    result['same_probability_dimensions'] = all(
        len(ref[k]['probabilities']) == len(now[k]['probabilities']) for k in ref)
    return result


def evaluate(args, root, folder, config, result, data_root, manifest, checks, checkpoint, cp_check, splits):
    if not checks['ok']:
        raise ValueError('Input checks failed; no checkpoint was loaded')
    if cp_check['status'] != 'matched':
        raise ValueError('The selected checkpoint is ' + cp_check['status'] + '; no model was constructed')
    if not args.output:
        raise ValueError('--output is required for inference')
    out = Path(args.output).expanduser().resolve()
    if out.exists():
        raise ValueError('Output already exists; select a new directory')
    for protected in [root / 'code', root / 'retained_results', data_root, checkpoint.parent]:
        try:
            out.relative_to(protected.resolve())
        except ValueError:
            continue
        raise ValueError('Output must be outside data, checkpoints, frozen code and retained results')

    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    import random
    module = import_frozen(root, args.family)
    seed = config['seed'] if args.family == 'components' else config['job']['seed']
    threads = config['threads']
    if args.family == 'components':
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        module.seed_all(seed, threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu for a separate CPU evaluation')
    # The matching SHA256 is required before any deserialization. No unsafe fallback.
    selected = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(selected, dict) or 'model' not in selected:
        raise ValueError('Selected checkpoint does not contain model state')
    if selected.get('epoch') != result['selected_epoch']:
        raise ValueError('Checkpoint epoch differs from retained result')
    if args.family != 'components' and selected.get('run_hash') != result['run_hash']:
        raise ValueError('Checkpoint run hash differs from retained result')
    classes = len(manifest['class_names'])
    if args.family == 'components':
        model = module.RevisedGraphClassifier(classes, config['operator'], config['fusion'],
                  config['k'], tuple(config['widths']), config['embedding']).to(device)
        size = config['batch_size']
    else:
        model = module.Model(config['job'], classes).to(device)
        size = config['job']['batch_size']
    model.load_state_dict(selected['model'], strict=True)
    cp_check['deserialized'] = True
    out.mkdir(parents=True)
    report = {'family': args.family, 'run_id': args.run_id, 'selected_epoch': result['selected_epoch'],
        'checkpoint': cp_check, 'input_checks': checks, 'training_executed': False,
        'evaluation_function': 'frozen ' + ('train_revised.evaluate' if args.family == 'components' else 'train.evaluate'),
        'batch_size': size, 'threads': threads, 'environment': {'python': sys.version.split()[0],
            'numpy': np.__version__, 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
            'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None},
        'historical_torch': config.get('torch_version', config.get('torch')),
        'metrics': {}, 'comparison_with_retained_predictions': {}, 'seconds': {}}
    try:
        for split in splits:
            started = time.monotonic()
            if args.family == 'components':
                data = module.Clouds(data_root / checks['dataset'], split, config['rho'])
                metrics, rows = module.evaluate(model, data, size, device, classes)
            else:
                data = module.Clouds(data_root, config['job'], split)
                metrics, rows = module.evaluate(model, data, size, device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            report['seconds'][split] = time.monotonic() - started
            report['metrics'][split] = metrics
            reference = read_json(folder / (split + '_predictions.json'))
            report['comparison_with_retained_predictions'][split] = compare_rows(reference, rows)
            write_json(out / (split + '_predictions.json'), {'class_names': manifest['class_names'], 'rows': rows})
        report['completed'] = True
    except Exception as exc:
        report.update(completed=False, error_type=type(exc).__name__, error=str(exc))
        write_json(out / 'evaluation_report.json', report)
        raise
    write_json(out / 'evaluation_report.json', report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--family', choices=sorted(LAYOUT))
    p.add_argument('--run-id')
    p.add_argument('--list-runs', action='store_true', help='Print audited run keys; no NumPy/Torch import')
    p.add_argument('--check-inputs', action='store_true', help='Verify code/data and optional weights; never load weights')
    p.add_argument('--data-root', help='Directory containing dataset folders (default: code/FAMILY/data)')
    group = p.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', help='Exact selected.pt file')
    group.add_argument('--weights-root', help='Extracted collection directory containing FAMILY/runs/RUN_ID/selected.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--split', choices=list(SPLITS) + ['all'], default='test')
    p.add_argument('--output', help='New inference output directory; original results are never overwritten')
    args = p.parse_args(argv)
    root = Path(args.release_root).expanduser().resolve()
    if args.list_runs:
        found = inventory(root)
        keys = sorted(k for k in found if not args.family or k.startswith(args.family + '/'))
        for key in keys:
            print(key)
        print('Runs: ' + str(len(keys)), file=sys.stderr)
        return 0
    if not args.family or not args.run_id:
        p.error('--family and --run-id are required unless --list-runs is used')
    folder, config, result = load_run(root, args.family, args.run_id)
    data_root = (Path(args.data_root).expanduser().resolve() if args.data_root
                 else root / 'code' / args.family / 'data')
    splits = list(SPLITS) if args.split == 'all' else [args.split]
    checks, manifest = inspect_inputs(root, args.family, args.run_id, config, result, data_root, splits)
    checkpoint = checkpoint_path(args, folder)
    cp_check = inspect_checkpoint(checkpoint, args.family, result)
    if args.check_inputs:
        report = {'family': args.family, 'run_id': args.run_id, 'input_checks': checks,
                  'checkpoint': cp_check, 'ready_for_inference': checks['ok'] and cp_check['status'] == 'matched',
                  'training_executed': False, 'inference_executed': False}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if checks['ok'] and cp_check['status'] != 'mismatch' else 2
    report = evaluate(args, root, folder, config, result, data_root, manifest, checks,
                      checkpoint, cp_check, splits)
    print(json.dumps({'completed': report['completed'], 'metrics': report['metrics'],
                      'comparison': report['comparison_with_retained_predictions']}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(2)
