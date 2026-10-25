"""Frozen geometry-only grouping; calibration uses current training normal clouds.

The bundled caches are ready for training. Rebuilding requires scipy; running the
bundled experiments only requires numpy and torch. All outputs are new files.
"""
from pathlib import Path
import argparse
import hashlib
import numpy as np
from common import read_json, write_json, sha, fingerprint

PROTOCOL = 'uniform_trainnormal_k20_a1p282_v1'
K0 = 20
ALPHA = 1.282


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def local_variation(xyz, k=K0):
    from scipy.spatial import cKDTree
    x = np.asarray(xyz, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 3 coordinates')
    if not 2 <= k < len(x):
        raise ValueError('Invalid geometric neighborhood')
    # Same formula as the completed controlled-support experiment: center + k neighbors.
    _, idx = cKDTree(x).query(x, k=k + 1, workers=1)
    p = x[idx]
    p -= p.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', p, p) / (k + 1)
    eig = np.maximum(np.linalg.eigh(cov)[0], 0.)
    total = eig.sum(axis=1)
    if np.any(total == 0):
        raise ValueError('Degenerate neighborhoods require inspection')
    return eig[:, 0] / total


def fit_reference(variation, records):
    # Labels identify reference normal clouds only during training calibration.
    ids = [i for i, r in enumerate(records)
           if r['split'] == 'train' and r['category'].lower() == 'normal']
    if not ids:
        raise ValueError('No training normal reference clouds')
    values = np.asarray(variation[ids], dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite training reference')
    mu, sd = float(values.mean()), float(values.std(ddof=0))
    return dict(normal_mean=mu, normal_std_population=sd, threshold=mu + ALPHA * sd,
                reference_ids=[records[i]['id'] for i in ids],
                reference_cloud_count=len(ids), reference_point_count=len(values))


def apply_rule(variation, threshold):
    # Deliberately has no class/ID/split argument. No class-dependent branch.
    return np.asarray(variation, dtype=np.float64) > float(threshold)


def build(source, dest):
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if dest.exists():
        raise ValueError('Use a new output data directory; existing caches are immutable')
    if source == dest or source in dest.parents:
        raise ValueError('Output must be separate from source data')
    for ds in ['surface', 'real']:
        m = read_json(source / ds / 'manifest.json')
        src = source / ds / 'inputs.npz'
        if sha(src) != m['inputs_sha256']:
            raise ValueError('Original input hash mismatch: ' + ds)
        with np.load(src, allow_pickle=False) as z:
            xyz, norm = z['xyz'].copy(), z['norm'].copy()
            labels, old = z['labels'].copy(), z['mask_base'].copy()
        records = [{k: r[k] for k in ['id', 'split', 'category', 'label',
                                      'source_cloud_id', 'physical_source_id', 'source_sha256']}
                   for r in m['records']]
        if xyz.shape != (len(records), 2000, 3):
            raise ValueError('Expected original 2000-point clouds')
        variation = np.stack([local_variation(x) for x in xyz])
        calibration = fit_reference(variation, records)
        masks = apply_rule(variation, calibration['threshold'])
        for i, r in enumerate(records):
            r.update(candidate_count=int(masks[i].sum()),
                     candidate_fraction=float(masks[i].mean()),
                     candidate_mask_sha256=array_sha(masks[i]),
                     original_group_count=int(old[i].sum()),
                     changed_group_point_count=int(np.count_nonzero(masks[i] != old[i])))
        folder = dest / ds
        folder.mkdir(parents=True)
        np.savez_compressed(folder / 'inputs.npz', xyz=xyz, norm=norm, labels=labels,
                            mask_uniform=masks, surface_variation=variation)
        meta = dict(dataset=ds, protocol=PROTOCOL, class_names=m['class_names'],
                    records=records, inputs_sha256=sha(folder / 'inputs.npz'),
                    counts=m['counts'], geodesic=[], calibration=calibration,
                    geometry_neighbors=K0, covariance_points=K0 + 1, coefficient=ALPHA,
                    formula='lambda_min / sum(lambda); candidate iff variation > mu + 1.282*sigma',
                    reference='pooled points of CURRENT TRAIN normal clouds only; population standard deviation',
                    application='same fixed threshold for all train/val/test clouds; no class-dependent grouping',
                    source_cache_sha256=sha(src), source_manifest_sha256=sha(source / ds / 'manifest.json'),
                    xyz_array_sha256=array_sha(xyz), normals_array_sha256=array_sha(norm),
                    labels_array_sha256=array_sha(labels),
                    source_normalization=m['normalization'],
                    graph_neighbors=15, graph_small_group_policy='unchanged original row-min masking and top-k; no forced extra candidates',
                    interpretation='New candidate partition protocol, not independently annotated point truth or recovered historical parameters',
                    physical_source_grouping='unchanged file-level split; independent physical specimens not established')
        write_json(folder / 'manifest.json', meta)
        summaries = []
        for split in ['train', 'val', 'test']:
            for category in m['class_names']:
                rr = [r for r in records if r['split'] == split and r['category'] == category]
                cc = [r['candidate_count'] for r in rr]
                summaries.append(dict(split=split, category=category, clouds=len(rr),
                    min_candidates=min(cc), max_candidates=max(cc),
                    mean_candidate_fraction=float(np.mean(cc) / 2000),
                    clouds_with_fewer_than_k_candidates=sum(c < 15 for c in cc),
                    clouds_with_fewer_than_k_background=sum(2000 - c < 15 for c in cc),
                    empty_candidate_clouds=sum(c == 0 for c in cc),
                    changed_point_fraction=float(np.mean([r['changed_group_point_count'] for r in rr]) / 2000)))
        write_json(folder / 'mask_diagnostics.json', {'protocol':PROTOCOL, 'calibration':calibration,
                   'summaries':summaries, 'normal_note':'Normal candidates are algorithm outputs, not inherited dummy labels or verified false positives.'})
        print(ds, calibration, 'candidate range', int(masks.sum(1).min()), int(masks.sum(1).max()), flush=True)


def validate(data):
    """Validate bundled geometry statistics without scipy or original Data access."""
    data = Path(data)
    reports = []
    for ds in ['surface', 'real']:
        folder = data / ds
        m = read_json(folder / 'manifest.json')
        if m['protocol'] != PROTOCOL or sha(folder / 'inputs.npz') != m['inputs_sha256']:
            raise ValueError('Uniform cache identity mismatch: ' + ds)
        rows = m['records']
        with np.load(folder / 'inputs.npz', allow_pickle=False) as z:
            x, n, labels, mask, variation = [z[k] for k in
                ['xyz', 'norm', 'labels', 'mask_uniform', 'surface_variation']]
        if x.shape != (len(rows), 2000, 3) or n.shape != x.shape or mask.shape != x.shape[:2]:
            raise ValueError('Cache shapes differ')
        if mask.dtype != np.bool_ or not all(np.isfinite(a).all() for a in [x, n, variation]):
            raise ValueError('Invalid cache values')
        if not np.array_equal(labels, [r['label'] for r in rows]):
            raise ValueError('Cloud label mismatch')
        if any(r['category'] != m['class_names'][r['label']] for r in rows):
            raise ValueError('Class mapping mismatch')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate cloud ID')
        if any(r['split'] not in ['train', 'val', 'test'] for r in rows):
            raise ValueError('Unknown split')
        calibration = fit_reference(variation, rows)
        for key in ['normal_mean', 'normal_std_population', 'threshold']:
            if not np.isclose(calibration[key], m['calibration'][key], rtol=1e-12, atol=1e-15):
                raise ValueError('Training reference mismatch: ' + key)
        if calibration['reference_ids'] != m['calibration']['reference_ids']:
            raise ValueError('Training reference IDs differ')
        # Always use the frozen threshold saved in the release, including at numeric ties.
        if not np.array_equal(mask, apply_rule(variation, m['calibration']['threshold'])):
            raise ValueError('Masks do not follow one common rule')
        for i, r in enumerate(rows):
            if r['candidate_mask_sha256'] != array_sha(mask[i]) or r['candidate_count'] != int(mask[i].sum()):
                raise ValueError('Per-cloud candidate hash/count mismatch')
        for key, arr in [('xyz_array_sha256', x), ('normals_array_sha256', n), ('labels_array_sha256', labels)]:
            if m[key] != array_sha(arr):
                raise ValueError('Frozen input field changed')
        reports.append(dict(dataset=ds, clouds=len(rows), counts=m['counts'],
                            manifest_sha256=sha(folder/'manifest.json'), calibration=calibration))
    return reports


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    build(a.source, a.out)
    validate(a.out)
