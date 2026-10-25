"""Local correctness checks; these small CPU fixtures are NOT research results."""
import ast
import copy
import fcntl
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
import json
import numpy as np
import torch
from common import ROOT, sha, read_json, write_json
from data_io import Clouds
from plan import make_plan
from models import Model
from uniform_preprocess import fit_reference, apply_rule, local_variation, validate
from train import run, seed_all, load_checkpoint
from run_uniform import package, validate_release


def fixture(root):
    folder = Path(root)/'surface'
    folder.mkdir(parents=True)
    rng = np.random.RandomState(83)
    x = rng.normal(size=(18, 64, 3)).astype('float32')
    mask = rng.rand(18, 64) < .2
    np.savez_compressed(folder/'inputs.npz', xyz=x, norm=x, mask_uniform=mask,
                        mask_base=~mask, labels=np.arange(18) % 3)
    rows = [dict(id=str(i), split='train' if i < 9 else 'val' if i < 12 else 'test', label=i % 3)
            for i in range(18)]
    write_json(folder/'manifest.json', {'records':rows, 'class_names':['a', 'b', 'c'],
               'inputs_sha256':sha(folder/'inputs.npz'), 'geodesic':[]})


class Checks(unittest.TestCase):
    def test_python38_plan_and_baseline_identity(self):
        for path in ROOT.glob('*.py'):
            ast.parse(path.read_text(), feature_version=(3, 8))
        plan = make_plan()
        self.assertEqual(plan['total'], 36)
        self.assertEqual(len({j['id'] for j in plan['jobs']}), 36)
        self.assertEqual(set(j['model'] for j in plan['jobs']), {'mvgcn', 'hard', 'soft'})
        refs = read_json(ROOT/'baseline_reference.json')['runs']
        for job in plan['jobs']:
            old = refs[job['baseline_id']]['job']
            for k in ['dataset', 'model', 'variant', 'k', 'rho', 'seed', 'batch_size', 'epochs', 'lr']:
                self.assertEqual(job[k], old[k])
            self.assertEqual(job['tau'], 'uniform')

    def test_current_train_normal_only_and_label_blind_application(self):
        rows = [{'id':str(i), 'category':c, 'split':s} for i, (c, s) in enumerate([
            ('Normal', 'train'), ('Dent', 'train'), ('Normal', 'val'), ('Normal', 'test')])]
        v = np.array([[1., 2., 3.], [5., 7., 9.], [20., 30., 40.], [70., 80., 90.]])
        fitted = fit_reference(v, rows)
        changed = v.copy()
        changed[1:] = 1e9
        self.assertEqual(fitted, fit_reference(changed, rows))
        self.assertEqual(fitted['reference_ids'], ['0'])
        rows2 = copy.deepcopy(rows)
        for r in rows2[1:]:
            r['category'] = 'UnseenCategory'
        self.assertEqual(fitted, fit_reference(v, rows2))
        predicted = apply_rule(v, fitted['threshold'])
        self.assertTrue(np.array_equal(predicted, v > fitted['threshold']))
        x = np.random.RandomState(9).normal(size=(90, 3))
        self.assertTrue(np.allclose(local_variation(x), local_variation(x*3+5), rtol=1e-10, atol=1e-12))

    def test_released_data_masks_and_loader(self):
        self.assertEqual(len(validate_release()), 2)
        for job in make_plan()['jobs']:
            if job['model'] != 'hard' or job['seed'] != 24:
                continue
            for split in ['train', 'val', 'test']:
                ds = Clouds(ROOT/'data', job, split)
                manifest = read_json(ROOT/'data'/job['dataset']/'manifest.json')
                with np.load(ROOT/'data'/job['dataset']/'inputs.npz', allow_pickle=False) as z:
                    wanted = [i for i, r in enumerate(manifest['records']) if r['split'] == split]
                    for i, src in enumerate(wanted):
                        self.assertTrue(np.array_equal(ds.mask[i], z['mask_uniform'][src]))
                        expected = z['xyz'][src] if job['variant'] == 'xyz' else np.concatenate([z['xyz'][src], z['norm'][src]], axis=1)
                        self.assertTrue(np.array_equal(ds.x[i], expected.T))

    def test_empty_sparse_and_full_groups_all_six_implementations(self):
        seed_all(7, 2)
        for model in ['mvgcn', 'hard', 'soft']:
            for variant in ['xyz', 'norm']:
                with self.subTest(model=model, variant=variant):
                    job = dict(model=model, variant=variant, k=15)
                    net = Model(job, 3)
                    x = torch.randn(4, 6 if variant == 'norm' else 3, 64)
                    mask = torch.zeros(4, 64, dtype=torch.bool)
                    mask[1, 0] = True
                    mask[2, :20] = True
                    mask[3] = True
                    logits = net(x, mask)
                    self.assertTrue(torch.isfinite(logits).all())
                    logits.square().mean().backward()
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
                    net.eval()
                    with torch.inference_mode():
                        a, b = net(x[:1], mask[:1]), net(x[:1], mask[:1])
                    self.assertTrue(torch.equal(a, b))

    def test_resume_fallback_and_return_archives(self):
        for model in ['mvgcn', 'hard', 'soft']:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture(root/'data')
                job = dict(make_plan(batch_size=4, epochs=3)['jobs'][0], model=model,
                           variant='xyz', k=5, id='fixture_'+model, baseline_id='fixture')
                direct, resumed = root/'direct', root/'resume'
                run(job, root/'data', direct, 'cpu', 2, 0)
                self.assertFalse(run(job, root/'data', resumed, 'cpu', 2, 0, stop_after=2))
                self.assertFalse((resumed/'test_predictions.json').exists())
                (resumed/'last.pt').write_bytes(b'intentionally damaged CPU fixture checkpoint')
                run(job, root/'data', resumed, 'cpu', 2, 0)
                aa, bb = load_checkpoint(direct/'last.pt'), load_checkpoint(resumed/'last.pt')
                for key, value in aa['model'].items():
                    self.assertTrue(torch.equal(value, bb['model'][key]), key)
                self.assertEqual(read_json(direct/'test_predictions.json'), read_json(resumed/'test_predictions.json'))
                self.assertEqual(aa['best_epoch'], bb['best_epoch'])
                plan = dict(make_plan(4, 3), total=1, jobs=[job], seeds=[job['seed']])
                result_root = root/'return'
                # Use a separate results root so package layout matches real training.
                result_root.mkdir(exist_ok=True)
                resumed.rename(result_root/job['id'])
                for full in [False, True]:
                    archive = package(result_root, plan, full=full)
                    with zipfile.ZipFile(archive) as z:
                        manifest = json.loads(z.read('archive_manifest.json'))
                        for name, meta in manifest['files'].items():
                            self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(), meta['sha256'])
                        self.assertEqual(any(n.endswith('.pt') for n in z.namelist()), full)
                        summary = json.loads(z.read('results/summary.json'))
                        self.assertTrue(summary['complete'])

    def test_duplicate_runner_is_rejected_before_plan_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with open(root/'runner.lock', 'a+') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = subprocess.run([sys.executable, str(ROOT/'run_uniform.py'),
                    '--stage', 'plan', '--out', str(root)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('active runner', result.stdout)
                self.assertFalse((root/'plan.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
