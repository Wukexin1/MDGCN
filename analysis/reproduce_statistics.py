#!/usr/bin/env python3
"""Recalculate manuscript statistics from retained prediction records.

Python 3.8+ and NumPy are required. This program does not train a model,
load a checkpoint, perform inference, or alter the source result folders.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np


SPLITS = ('train', 'val', 'test')
METRICS = ('oa', 'ba', 'macro_f1')


def load_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def save_json(path, obj):
    with path.open('w', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result_directory(root, family):
    child = root / ('gpu_results' if family == 'components' else 'results')
    return child if child.is_dir() else root


def point_cloud_metrics(rows, class_count):
    """OA, mean recall on supported classes, and macro F1 on all classes."""
    if not rows:
        raise ValueError('Prediction records are empty.')
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in rows:
        truth, prediction = row['truth'], row['prediction']
        if not isinstance(truth, int) or not isinstance(prediction, int):
            raise ValueError('Class indices must be integers.')
        if not (0 <= truth < class_count and 0 <= prediction < class_count):
            raise ValueError('Class index is outside the configured classes.')
        probability = np.asarray(row['probabilities'], dtype=float)
        if probability.shape != (class_count,) or not np.isfinite(probability).all():
            raise ValueError('Probability array has an invalid shape or non-finite value.')
        if np.any(probability < 0) or np.any(probability > 1) or abs(float(probability.sum()) - 1) > 1e-5:
            raise ValueError('Probabilities are outside [0, 1] or do not sum to one.')
        if int(np.argmax(probability)) != prediction:
            raise ValueError('Prediction differs from probability argmax.')
        matrix[truth, prediction] += 1
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    recall = np.divide(true_positive, support, out=np.zeros(class_count), where=support > 0)
    f1 = np.divide(2 * true_positive, support + predicted_support,
                   out=np.zeros(class_count), where=support + predicted_support > 0)
    return dict(n=len(rows), oa=float(true_positive.sum() / matrix.sum()),
                ba=float(recall[support > 0].mean()), macro_f1=float(f1.mean()),
                confusion_matrix=matrix.tolist(), class_support=support.tolist())


def configuration_for_run(config, expected, group):
    """Read only public protocol fields; never export a full original config."""
    family = expected['family']
    p = group['configuration']
    if family == 'components':
        actual = {k: config[k] for k in ('operator', 'fusion', 'k', 'batch_size', 'epochs')}
        actual.update(family=family, dataset='surface' if expected['run_id'].startswith('original_synthetic') else 'real',
                      model=config['operator'], variant='xyz', tau='calibrated', rho=None)
        names, seed = config['classes'], config['seed']
    else:
        actual = {k: config['job'][k] for k in ('dataset', 'model', 'variant', 'k', 'tau', 'rho', 'batch_size', 'epochs')}
        actual['family'] = family
        names, seed = config['class_names'], config['job']['seed']
    if actual != p or names != expected['class_names'] or seed != expected['seed']:
        raise ValueError('Config does not match the expected protocol, class order, or seed.')
    return actual


def verify_run(folder, expected, group):
    family = expected['family']
    result_name = 'final_metrics.json' if family == 'components' else 'result.json'
    required = [result_name, 'config.json', 'history.json'] + [s + '_predictions.json' for s in SPLITS]
    for name in required:
        if not (folder / name).is_file():
            raise ValueError('Required retained file is missing: ' + name)
    result = load_json(folder / result_name)
    config = load_json(folder / 'config.json')
    history = load_json(folder / 'history.json')
    actual_config = configuration_for_run(config, expected, group)
    reported = result['splits' if family == 'components' else 'metrics']
    recalculated = {}
    for split in SPLITS:
        raw = load_json(folder / (split + '_predictions.json'))
        if isinstance(raw, list):
            rows = raw
        else:
            rows = raw['rows']
            if raw.get('class_names', expected['class_names']) != expected['class_names']:
                raise ValueError('Prediction class names differ from the config.')
        values = point_cloud_metrics(rows, len(expected['class_names']))
        for metric in METRICS:
            reported_value = float(reported[split][metric])
            if not math.isfinite(reported_value) or abs(values[metric] - reported_value) > 1e-12:
                raise ValueError('Saved summary differs from predictions: ' + split + '/' + metric)
        for field in ('n', 'confusion_matrix', 'class_support'):
            if field in reported[split] and values[field] != reported[split][field]:
                raise ValueError('Saved summary differs from predictions: ' + split + '/' + field)
        recalculated[split] = values
    if [h['epoch'] for h in history] != list(range(1, actual_config['epochs'] + 1)):
        raise ValueError('Epoch history is missing, unordered, or duplicated.')
    criterion = 'ba' if family == 'components' else 'oa'
    for h in history:
        if not all(math.isfinite(float(h['validation'][m])) for m in (criterion, 'cross_entropy')):
            raise ValueError('Validation selection values are not finite.')
    best = max(history, key=lambda h: (h['validation'][criterion], -h['validation']['cross_entropy']))
    if result['selected_epoch'] != best['epoch']:
        raise ValueError('Selected epoch does not follow the validation selection rule.')
    if result['selected_epoch'] != expected['expected_selected_epoch']:
        raise ValueError('Selected epoch differs from the retained manuscript run.')
    return dict(key=expected['key'], run_id=expected['run_id'], family=family,
                group_id=expected['group_id'], seed=expected['seed'],
                class_names=expected['class_names'], configuration=actual_config,
                selected_epoch=result['selected_epoch'], epochs_recorded=len(history),
                checkpoint_selection='validation ' + criterion.upper() + ', lower cross entropy, earliest exact tie',
                checkpoint_present=any(folder.glob('*.pt')) or any(folder.glob('*.pth')),
                metrics=recalculated,
                retained_file_sha256={name: file_hash(folder / name) for name in required})


def aggregate_group(group, runs):
    ordered = [runs[key] for key in group['run_keys']]
    if [r['seed'] for r in ordered] != group['seeds']:
        raise ValueError('Group membership does not have the expected ordered seeds.')
    statistics = {}
    for split in SPLITS:
        for metric in METRICS:
            values = np.array([r['metrics'][split][metric] for r in ordered], dtype=float)
            statistics[split + '_' + metric] = dict(mean_percent=float(values.mean() * 100),
                                                    sample_sd_percent=float(values.std(ddof=1) * 100),
                                                    seed_values_percent=(values * 100).tolist())
    return dict(group_id=group['group_id'], configuration=group['configuration'],
                seeds=group['seeds'], run_keys=group['run_keys'], statistics=statistics)


def display_equal(value, expected, decimals):
    if decimals is None:
        return abs(value - expected) < 1e-10
    return float(format(value, '.' + str(decimals) + 'f')) == expected


def candidate_rows(root, expected):
    """Aggregate retained per-point-cloud counts without loading point arrays."""
    directory = root.parent if root.name == 'results' else root
    summaries = {}
    for dataset in sorted(set(c['dataset'] for c in expected)):
        path = directory / 'data' / dataset / 'manifest.json'
        if not path.is_file():
            raise ValueError('Uniform data/' + dataset + '/manifest.json is missing.')
        manifest = load_json(path)
        categories = sorted(set(c['category'] for c in expected if c['dataset'] == dataset))
        for category in categories:
            npoints = next(c['points_per_point_cloud'] for c in expected if c['dataset'] == dataset and c['category'] == category)
            records = [r for r in manifest['records'] if r['category'] == category]
            if not records:
                raise ValueError('A required abnormal category has no point-cloud records.')
            provided = np.array([r['original_group_count'] / npoints for r in records], dtype=float)
            calibrated = np.array([r['candidate_count'] / npoints for r in records], dtype=float)
            if not np.all((provided >= 0) & (provided <= 1)) or not np.all((calibrated >= 0) & (calibrated <= 1)) or calibrated.sum() == 0:
                raise ValueError('Retained candidate counts are outside the permitted range.')
            summaries[(dataset, category)] = dict(dataset=dataset, category=category, M=len(records),
                points_per_point_cloud=npoints, provided_percent=float(provided.mean() * 100),
                calibrated_percent=float(calibrated.mean() * 100),
                calibrated_min_percent=float(calibrated.min() * 100),
                calibrated_max_percent=float(calibrated.max() * 100),
                pooled_background_candidate_ratio=float((1 - calibrated.mean()) / calibrated.mean()),
                manifest_sha256=file_hash(path))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', required=True, type=Path, help='Complete return root, or its results directory.')
    parser.add_argument('--uniform', required=True, type=Path, help='Uniform return root with results and data manifests, or its results directory.')
    parser.add_argument('--components', required=True, type=Path, help='Complete component return root, or its gpu_results directory (batches 25 and 32).')
    parser.add_argument('--output', required=True, type=Path, help='New or existing output directory for recalculated statistics.')
    parser.add_argument('--expected', type=Path, default=Path(__file__).with_name('expected_display_values.json'), help='Expected display values and fixed run membership.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = load_json(args.expected)
    if expected.get('schema_version') != 1:
        parser.error('Unsupported expected-manifest schema.')
    roots = {f: result_directory(getattr(args, f), f) for f in ('complete', 'uniform', 'components')}
    group_spec = {g['group_id']: g for g in expected['groups']}
    issues = []
    runs = {}
    for run in expected['runs']:
        if Path(run['run_id']).name != run['run_id']:
            parser.error('Invalid run identifier in expected manifest.')
        try:
            runs[run['key']] = verify_run(roots[run['family']] / run['run_id'], run, group_spec[run['group_id']])
        except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
            # Avoid exposing private config paths through exception messages.
            detail = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else type(error).__name__
            issues.append(dict(stage='run', key=run['key'], detail=detail))
    groups = {}
    for spec in expected['groups']:
        if not all(k in runs for k in spec['run_keys']):
            issues.append(dict(stage='group', key=spec['group_id'], detail='One or more required runs were not verified.'))
            continue
        groups[spec['group_id']] = aggregate_group(spec, runs)
    checks = []
    for check in expected['performance_checks']:
        result = dict(check)
        group = groups.get(check['group_id'])
        if group is None:
            result.update(status='missing', matches=False)
        else:
            values = group['statistics'][check['metric']]
            ok = display_equal(values['mean_percent'], check['expected_mean'], check['decimals']) and display_equal(values['sample_sd_percent'], check['expected_sample_sd'], check['decimals'])
            result.update(actual_mean=values['mean_percent'], actual_sample_sd=values['sample_sd_percent'], status='matched' if ok else 'mismatch', matches=ok)
        checks.append(result)
        if not result['matches']:
            issues.append(dict(stage='display', key='{} row {} {}'.format(check['item'], check['row'], check['metric']), detail=result['status']))
    candidate_statistics = {}
    try:
        candidate_statistics = candidate_rows(args.uniform, expected['candidate_checks'])
    except (OSError, KeyError, TypeError, ValueError) as error:
        detail = str(error) if isinstance(error, ValueError) and not isinstance(error, json.JSONDecodeError) else type(error).__name__
        issues.append(dict(stage='candidate_counts', key='Tables 2 and 5', detail=detail))
    candidate_checks = []
    for check in expected['candidate_checks']:
        result = dict(check)
        values = candidate_statistics.get((check['dataset'], check['category']))
        if values is None:
            result.update(status='missing', matches=False)
        else:
            actual = values[check['metric']]
            ok = display_equal(actual, check['expected_value'], check['decimals'])
            result.update(actual_value=actual, status='matched' if ok else 'mismatch', matches=ok)
        candidate_checks.append(result)
        if not result['matches']:
            issues.append(dict(stage='candidate_display', key=check['item'] + '/' + check['category'] + '/' + check['metric'], detail=result['status']))

    # Output contains protocol fields, metrics and content hashes only, not input paths,
    # personal configuration fields, or individual sample identifiers.
    save_json(args.output / 'runs_recalculated.json', list(runs.values()))
    run_csv = []
    for run in runs.values():
        for split, values in run['metrics'].items():
            run_csv.append(dict(key=run['key'], family=run['family'], group_id=run['group_id'], seed=run['seed'], selected_epoch=run['selected_epoch'], split=split, n=values['n'], **{m: values[m] for m in METRICS}))
    save_csv(args.output / 'run_metrics.csv', run_csv, ['key','family','group_id','seed','selected_epoch','split','n','oa','ba','macro_f1'])
    save_json(args.output / 'group_summary.json', list(groups.values()))
    group_csv = []
    for group in groups.values():
        for metric, values in group['statistics'].items():
            row = dict(group_id=group['group_id'], metric=metric, mean_percent=values['mean_percent'], sample_sd_percent=values['sample_sd_percent'])
            row.update({'seed_' + str(s): v for s, v in zip(group['seeds'], values['seed_values_percent'])})
            group_csv.append(row)
    save_csv(args.output / 'group_summary.csv', group_csv, ['group_id','metric','mean_percent','sample_sd_percent','seed_24','seed_42','seed_2026'])
    save_json(args.output / 'display_value_checks.json', checks)
    save_csv(args.output / 'display_value_checks.csv', checks, ['item','row','group_id','metric','decimals','expected_mean','expected_sample_sd','actual_mean','actual_sample_sd','status','matches'])
    save_json(args.output / 'candidate_proportions.json', list(candidate_statistics.values()))
    save_csv(args.output / 'candidate_proportions.csv', list(candidate_statistics.values()), ['dataset','category','M','points_per_point_cloud','provided_percent','calibrated_percent','calibrated_min_percent','calibrated_max_percent','pooled_background_candidate_ratio','manifest_sha256'])
    save_json(args.output / 'candidate_display_checks.json', candidate_checks)
    save_csv(args.output / 'candidate_display_checks.csv', candidate_checks, ['item','dataset','category','metric','decimals','expected_value','actual_value','status','matches'])
    prediction_count = sum(v['n'] for r in runs.values() for v in r['metrics'].values())
    if len(runs) != expected['expected_run_count'] or len(groups) != expected['expected_group_count'] or prediction_count != expected['expected_prediction_rows']:
        issues.append(dict(stage='coverage', key='totals', detail='Run, group, or prediction coverage differs from the manuscript manifest.'))
    report = dict(status='passed' if not issues else 'failed',
        verification='Metrics and display values recalculated from retained predictions, history and configuration.',
        training_executed=False, inference_executed=False, figure_rendering_executed=False,
        python_version=platform.python_version(), numpy_version=np.__version__,
        expected_manifest_sha256=file_hash(args.expected),
        counts=dict(expected_runs=expected['expected_run_count'], verified_runs=len(runs),
                    expected_groups=expected['expected_group_count'], verified_groups=len(groups),
                    expected_prediction_rows=expected['expected_prediction_rows'], verified_prediction_rows=prediction_count,
                    performance_display_checks=len(checks), performance_display_matches=sum(c['matches'] for c in checks),
                    candidate_display_checks=len(candidate_checks), candidate_display_matches=sum(c['matches'] for c in candidate_checks),
                    runs_with_checkpoint=sum(r['checkpoint_present'] for r in runs.values()), issues=len(issues)),
        interpretation=dict(metric_scale='run_metrics.csv uses fractions; group summaries and displayed mean/SD use percentages.',
                            standard_deviation='Sample SD across the three training seeds, ddof=1, on the same fixed data split.',
                            candidate_counts='Saved per-point-cloud counts only; masks and covariance matrices were not recomputed.'),
        limits=[
            'This statistics check does not execute training, inference or figure rendering.',
            'runs_with_checkpoint counts files inside retained result directories; separately distributed checkpoints are not counted.',
            'Current checkpoint coverage and separate inference/rendering entry points are documented in docs/RELEASE_STATUS.txt.'
        ], issues=issues)
    save_json(args.output / 'verification_report.json', report)
    print(json.dumps(dict(status=report['status'], counts=report['counts']), indent=2))
    return 0 if not issues else 1


if __name__ == '__main__':
    sys.exit(main())
