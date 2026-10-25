"""New controlled perturbations of existing normal clouds; never overwrite originals.

The imposed-displacement support is known by construction. It is not a recovered
annotation of the historical dataset. Candidate masks use training-normal geometry
only; the classifier must never read gt_mask or displacement as input features.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree

LEVELS = [.005, .01, .02, .05, .10]
ALPHAS = [0., .5, 1., 1.282, 1.5, 2.]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_geometry(xyz, k=20):
    x = np.asarray(xyz, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite N x 3 coordinates')
    if not 2 <= k < len(x):
        raise ValueError('Invalid neighborhood size')
    tree = cKDTree(x)
    _, idx = tree.query(x, k=k + 1, workers=1)
    # The query point plus k neighbors form each covariance neighborhood.
    p = x[idx]
    p = p - p.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', p, p) / (k + 1)
    eig, vectors = np.linalg.eigh(cov)
    eig = np.maximum(eig, 0.)
    sums = eig.sum(axis=1)
    if np.any(sums == 0):
        raise ValueError('Degenerate point neighborhoods require inspection')
    return eig[:, 0] / sums, vectors[:, :, 0]


def normal_splits(root, seed):
    outer = {}
    for split in ['train', 'test']:
        entries = json.loads((root / f'{split}_files.txt').read_text())
        paths = [Path(*Path(p).parts[-2:]) for p in entries]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate source entries')
        outer[split] = sorted(str(p) for p in paths if p.parent.name.lower() == 'normal')
    if set(outer['train']) & set(outer['test']):
        raise ValueError('Outer split overlap')
    train = np.array(outer['train'])
    perm = np.random.default_rng(seed).permutation(len(train))
    nval = max(1, round(.2 * len(train)))
    splits = {'train': sorted(train[perm[nval:]].tolist()),
              'val': sorted(train[perm[:nval]].tolist()), 'test': outer['test']}
    if min(map(len, splits.values())) < 2:
        raise ValueError('Insufficient source clouds')
    return splits


def source_frame(xyz):
    mean = xyz.mean(axis=0)
    _, v = np.linalg.eigh(np.cov(xyz.T))
    normal = v[:, 0]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    tangent = v[:, 1:]
    uv = (xyz - mean) @ tangent
    return mean, normal, tangent, uv


def injection_design(xyz, seed, max_fraction=.10):
    _, global_normal, tangent, uv = source_frame(xyz)
    # Keep the footprint inside the convex hull in the reference tangent plane.
    from scipy.spatial import ConvexHull
    hull = ConvexHull(uv)
    boundary_distance = -(uv @ hull.equations[:, :2].T + hull.equations[:, 2]).max(axis=1)
    tree = cKDTree(uv)
    qmax = round(max_fraction * len(xyz))
    radii = tree.query(uv, k=qmax + 1)[0][:, -1]
    eligible = np.flatnonzero(boundary_distance > 1.05 * radii)
    if not len(eligible):
        raise ValueError('No interior center supports the largest target footprint')
    center = int(np.random.default_rng(seed).choice(eligible))
    d = np.linalg.norm(uv - uv[center], axis=1)
    order = np.argsort(d, kind='stable')
    _, normals = local_geometry(xyz, k=20)
    normal = normals[center]
    if normal @ global_normal < 0:
        normal = -normal
    return {'center': center, 'distance': d, 'order': order, 'normal': normal,
            'tangent': tangent, 'uv': uv}


def inject(xyz, design, fraction, amplitude, sign):
    q = round(fraction * len(xyz))
    order, d = design['order'], design['distance']
    if not 0 < q < len(xyz) or not d[order[q - 1]] < d[order[q]]:
        raise ValueError('Cannot define exact support at tied footprint boundary')
    radius = float((d[order[q - 1]] + d[order[q]]) / 2)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[order[:q]] = True
    profile = np.zeros(len(xyz))
    profile[mask] = (1 - (d[mask] / radius) ** 2) ** 2
    displacement = sign * amplitude * profile[:, None] * design['normal']
    out = xyz + displacement
    if not np.array_equal(np.any(out != xyz, axis=1), mask):
        raise AssertionError('Stored coordinates must retain every imposed displacement')
    if not np.array_equal(out[~mask], xyz[~mask]):
        raise AssertionError('Background coordinates changed')
    return out, mask, displacement, radius


def run(root, out, seed=20260914, k0=20, alpha=1.282):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    if root == out or root in out.parents:
        raise ValueError('Output must be separate from original data')
    splits = normal_splits(root, seed)
    xyzs = {p: np.loadtxt(root / p, delimiter=',', usecols=(0, 1, 2))
            for paths in splits.values() for p in paths}
    if any(x.shape != (2000, 3) for x in xyzs.values()):
        raise ValueError('This study prespecifies 2000 points per cloud')
    hashes = {p: sha256(root / p) for p in xyzs}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Duplicate normal source files require grouped splitting')
    reference = np.concatenate([local_geometry(xyzs[p], k0)[0] for p in splits['train']])
    spacing = np.concatenate([cKDTree(xyzs[p]).query(xyzs[p], k=2)[0][:, 1]
                              for p in splits['train']])
    mu, sd = float(reference.mean()), float(reference.std(ddof=0))
    amplitude = float(5 * np.median(spacing))
    threshold = mu + alpha * sd
    out.mkdir(parents=True)
    records = []
    for split, paths in splits.items():
        for p in paths:
            source_id = Path(p).stem
            x = xyzs[p]
            design_seed = seed + int(hashlib.sha256(p.encode()).hexdigest()[:8], 16)
            design = injection_design(x, design_seed)
            cases = [('normal', 0., x, np.zeros(2000, bool), np.zeros_like(x), 0.)]
            for rho in LEVELS:
                for category, sign in [('dent', -1), ('raised', 1)]:
                    y, gt, delta, radius = inject(x, design, rho, amplitude, sign)
                    cases.append((category, rho, y, gt, delta, radius))
            for category, rho, y, gt, delta, radius in cases:
                var, _ = local_geometry(y, k0)
                mask = var > threshold
                sample_id = f'{source_id}_{category}_{round(rho * 2000):04d}'
                rel = Path('clouds') / split / (sample_id + '.npz')
                dest = out / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, xyz=y, candidate_mask=mask, gt_mask=gt,
                                    displacement=delta, surface_variation=var,
                                    cloud_label=np.array(['normal', 'dent', 'raised'].index(category)))
                records.append(dict(id=sample_id, source_id=source_id, source_file=p,
                                    split=split, category=category, rho=rho, file=str(rel),
                                    n=2000, gt_count=int(gt.sum()), candidate_count=int(mask.sum()),
                                    effective_count_5pct_amplitude=int((np.linalg.norm(delta, axis=1) >= .05 * amplitude).sum()),
                                    candidate_tp=int((mask & gt).sum()), candidate_fp=int((mask & ~gt).sum()),
                                    center_index=design['center'], footprint_radius=radius,
                                    amplitude=amplitude if category != 'normal' else 0.,
                                    orientation='local normal aligned with dominant-positive PCA normal'))
    report = dict(study='NEW controlled local displacement simulation',
                  historical_truth_reconstruction=False, seed=seed, points_per_cloud=2000,
                  class_names=['normal', 'dent', 'raised'], fractions=LEVELS,
                  source_root=str(root), source_splits=splits, source_sha256=hashes,
                  original_split_sha256={s:sha256(root / f'{s}_files.txt') for s in ['train','test']},
                  k0=k0, metric='euclidean', alpha=alpha, normal_mean=mu,
                  normal_std_population=sd, threshold=threshold,
                  reference_point_count=len(reference), amplitude=amplitude,
                  amplitude_rule='5 times pooled median nearest-neighbor spacing of training normal clouds',
                  support_rule='rho*N closest points to a fixed interior center in the source PCA tangent plane',
                  profile='delta_i = sign*h*(1-(d_i/R)^2)^2*n_center for d_i<R; zero otherwise',
                  normal_reuse='Each unchanged normal cloud is reused once in each separate fraction experiment.',
                  limitations=['Source-cloud IDs are not verified independent physical specimen IDs.',
                               'One placement seed; footprint size and slope vary with rho at fixed height.',
                               'Ground truth means imposed displacement support, including small boundary displacements.',
                               'Synthetic dents/raised regions are not labels for the historical industrial clouds.'],
                  records=records)
    (out / 'manifest.json').write_text(json.dumps(report, indent=2))
    analyze(out, report)
    print(json.dumps({k:report[k] for k in ['reference_point_count','normal_mean','normal_std_population','threshold','amplitude']}, indent=2))
    print('Source split counts:', {s:len(p) for s,p in splits.items()}, 'stored clouds:',len(records))
    return report


def analyze(out, report):
    stats = []
    for split in ['train', 'val', 'test']:
        items = [r for r in report['records'] if r['split'] == split]
        loaded = {r['id']:dict(np.load(out / r['file'], allow_pickle=False)) for r in items}
        for alpha in ALPHAS:
            t = report['normal_mean'] + alpha * report['normal_std_population']
            for category in ['normal', 'dent', 'raised']:
                for rho in ([0.] if category == 'normal' else LEVELS):
                    selected = [r for r in items if r['category'] == category and r['rho'] == rho]
                    counts = np.zeros(4, dtype=np.int64)
                    recalls, selected_fracs = [], []
                    for r in selected:
                        z = loaded[r['id']]
                        pred, gt = z['surface_variation'] > t, z['gt_mask']
                        tp, fp = int((pred & gt).sum()), int((pred & ~gt).sum())
                        fn, tn = int((~pred & gt).sum()), int((~pred & ~gt).sum())
                        counts += [tp, fp, fn, tn]
                        if gt.any():
                            recalls.append(tp / gt.sum())
                        selected_fracs.append(float(pred.mean()))
                    tp, fp, fn, tn = map(int, counts)
                    ci = None
                    if recalls:
                        boot_rng = np.random.default_rng(report['seed'])
                        boot = boot_rng.choice(recalls, size=(2000, len(recalls)), replace=True).mean(1)
                        ci = np.quantile(boot, [.025, .975]).tolist()
                    stats.append(dict(split=split, alpha=alpha, category=category, rho=rho,
                                      source_clouds=len(selected), tp=tp, fp=fp, fn=fn, tn=tn,
                                      recall=tp/(tp+fn) if tp+fn else None,
                                      precision=tp/(tp+fp) if tp+fp else None,
                                      background_fpr=fp/(fp+tn),
                                      mean_candidate_fraction=float(np.mean(selected_fracs)),
                                      recall_ci95_source_cloud_bootstrap=ci))
    (out / 'candidate_diagnostics.json').write_text(json.dumps(stats, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20260914)
    a = p.parse_args()
    run(a.source, a.out, a.seed)
