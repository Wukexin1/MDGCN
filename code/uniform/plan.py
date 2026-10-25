"""Prespecified 36-run uniform-preprocessing check; no outcome-based selection."""
from uniform_preprocess import PROTOCOL

SEEDS = [24, 42, 2026]


def make_plan(batch_size=32, epochs=250):
    if batch_size < 2 or epochs < 1:
        raise ValueError('Invalid batch size or epoch count')
    jobs = []
    for ds in ['surface', 'real']:
        for variant in ['xyz', 'norm']:
            for model in ['mvgcn', 'hard', 'soft']:
                for seed in SEEDS:
                    baseline = '{}_{}_{}_k15_base_rNA_s{}_b{}_e{}'.format(ds, model, variant, seed, batch_size, epochs)
                    jobs.append(dict(dataset=ds, model=model, variant=variant, k=15,
                        tau='uniform', rho=None, seed=seed, batch_size=batch_size,
                        epochs=epochs, lr=.1, stage='uniform', protocol=PROTOCOL,
                        baseline_id=baseline, id=baseline.replace('_base_', '_uniform_')))
    return dict(version=1, protocol=PROTOCOL, jobs=jobs, total=len(jobs),
                counts={'uniform':len(jobs)}, seeds=SEEDS, batch_size=batch_size, epochs=epochs,
                selection='maximum validation OA; tie: lower ordinary validation CE; earliest exact tie',
                purpose='check one automatic grouping rule for unknown classes, using only current training normal reference',
                unchanged='model implementations, cloud splits, normalized XYZ, normals, optimizer, augmentation, selection',
                comparison='Original saved partitions remain separate. The new protocol changes calibration and grouping for all classes; differences cannot be attributed only to normal dummy labels.')
