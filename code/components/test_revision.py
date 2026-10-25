"""Integrity, mathematical consistency and training-protocol regression checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from controlled_data import local_geometry, injection_design, inject
from revised_graph import neighbors, gather, edge_features, MaskedBatchNorm, RevisedGraphClassifier
from train_revised import metrics, batches, run


class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_neighborhood_exclusion_and_distance_immutability(self):
        x = torch.randn(3,3,19)
        mask = torch.zeros(3,19,dtype=torch.bool)
        mask[1,5] = True; mask[2,[0,3,7]] = True
        distances = torch.cdist(x.transpose(1,2),x.transpose(1,2))
        before = distances.clone()
        idx, valid = neighbors(x,mask,8,distances)
        self.assertTrue(torch.equal(before,distances))
        self.assertEqual(valid.sum((1,2)).tolist(),[0,19,57])
        for b in range(3):
            self.assertTrue(mask[b,idx[b][valid[b]]].all())

    def test_masked_batchnorm_matches_standard_without_padding(self):
        torch.manual_seed(4)
        x = torch.randn(3,7,17,5)
        a,b = MaskedBatchNorm(7),torch.nn.BatchNorm2d(7)
        self.assertTrue(torch.allclose(a(x,torch.ones(3,17,5,dtype=torch.bool)),b(x),atol=1e-6))
        self.assertTrue(torch.allclose(a.running_var,b.running_var,atol=1e-6))
        self.assertTrue(torch.allclose(a.running_mean,b.running_mean,atol=1e-6))

    def test_padding_does_not_change_normalization(self):
        torch.manual_seed(7)
        x = torch.randn(2,3,11,2)
        padded = torch.cat([x,torch.ones(2,3,11,6)*1e6],dim=-1)
        valid = torch.zeros(2,11,8,dtype=torch.bool); valid[:,:,:2]=True
        a,b=MaskedBatchNorm(3),MaskedBatchNorm(3)
        self.assertTrue(torch.allclose(a(x,torch.ones(2,11,2,dtype=torch.bool)),b(padded,valid)[:,:,:,:2],atol=1e-6))

    def test_soft_and_hard_match_explicit_formulas(self):
        torch.manual_seed(8)
        x=torch.randn(2,3,13)
        idx,v=neighbors(x,torch.ones(2,13,dtype=torch.bool),5)
        q=gather(x,idx); center=x.transpose(1,2)[:,:,None,:].expand_as(q)
        attended=(q@q.transpose(-1,-2)/np.sqrt(3)).softmax(-1)@q
        soft=torch.cat([q-center,attended],-1).permute(0,3,1,2)
        self.assertTrue(torch.allclose(soft,edge_features(x,idx,v,'soft'),atol=1e-6))
        m=torch.linalg.vector_norm(q,dim=-1)
        keep=m>=m.mean(-1,keepdim=True)
        pooled=(q*keep[...,None]).sum(2)/keep.sum(2)[...,None]
        hard=torch.cat([gather(pooled.transpose(1,2),idx)-center,center],-1).permute(0,3,1,2)
        self.assertTrue(torch.allclose(hard,edge_features(x,idx,v,'hard'),atol=1e-6))

    def test_empty_full_singleton_views_have_finite_gradients(self):
        for operator in ['standard','hard','soft']:
            model=RevisedGraphClassifier(operator=operator,widths=(8,8,16,32),embedding=32)
            x=torch.randn(3,3,31,requires_grad=True)
            mask=torch.zeros(3,31,dtype=torch.bool)
            mask[1,:]=True; mask[2,4]=True
            y=model(x,mask)
            self.assertTrue(torch.isfinite(y).all())
            y.square().mean().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_equal_capacity_and_permutation_invariant_cloud_prediction(self):
        counts=[]
        for op in ['standard','hard','soft']:
            for fusion in ['add','replace']:
                model=RevisedGraphClassifier(operator=op,fusion=fusion,widths=(8,8,16,32),embedding=32).eval()
                counts.append(sum(p.numel() for p in model.parameters()))
                x=torch.randn(2,3,31); mask=torch.rand(2,31)>.5; order=torch.randperm(31)
                with torch.no_grad():
                    self.assertTrue(torch.allclose(model(x,mask),model(x[:,:,order],mask[:,order]),atol=1e-5))
        self.assertEqual(len(set(counts)),1)

    def test_geometry_rigid_motion_invariance(self):
        rng=np.random.default_rng(40)
        x=rng.normal(size=(300,3))
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        a,_=local_geometry(x);b,_=local_geometry(x@q+np.array([20.,-5.,7.]))
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_exact_support_paired_geometry_and_unchanged_background(self):
        rng=np.random.default_rng(2)
        x=np.column_stack([rng.uniform(-4,4,(2000,2)),rng.normal(0,.001,2000)])
        design=injection_design(x,5)
        for rho in [.005,.01,.02,.05,.1]:
            dent,m,delta,r=inject(x,design,rho,.2,-1)
            bump,m2,delta2,r2=inject(x,design,rho,.2,1)
            self.assertEqual(int(m.sum()),round(rho*2000))
            np.testing.assert_array_equal(m,m2)
            np.testing.assert_array_equal(dent[~m],x[~m])
            np.testing.assert_allclose(delta,-delta2,atol=0)
            np.testing.assert_allclose(dent+bump,2*x,atol=1e-14)

    def test_counts_and_batches_use_clouds_not_epochs(self):
        m=metrics(np.array([0,0,1,2]),np.array([0,1,1,2]),3)
        self.assertEqual(m['n'],4);self.assertEqual(m['oa'],.75)
        self.assertAlmostEqual(m['ba'],5/6)
        ids=[i for group in batches(9,4,True) for i in group]
        self.assertEqual(sorted(ids),list(range(9)))

    def test_end_to_end_fixture_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'data';root.mkdir()
            records=[]; rng=np.random.default_rng(2)
            for split in ['train','val','test']:
                for source in range(3):
                    for label,category in enumerate(['normal','dent','raised']):
                        name=f'{split}_{source}_{category}'
                        x=rng.normal(size=(32,3))
                        np.savez(root/f'{name}.npz',xyz=x,candidate_mask=rng.random(32)>.7,
                                 cloud_label=label,gt_mask=np.zeros(32,bool))
                        records.append(dict(id=name,source_id=f'{split}_{source}',split=split,category=category,
                                            rho=0. if label==0 else .02,file=f'{name}.npz'))
            (root/'manifest.json').write_text(json.dumps(dict(fractions=[.02],class_names=['normal','dent','raised'],records=records)))
            out=Path(temp)/'run'
            args=argparse.Namespace(data=str(root),out=str(out),rho=.02,seed=4,threads=2,device='cpu',
                                    operator='soft',fusion='add',k=5,widths=[4,4,8,16],embedding=32,
                                    lr=.01,epochs=2,batch_size=4)
            result=run(args)
            self.assertTrue(result['test_loaded_after_checkpoint_selection'])
            self.assertIn(result['selected_epoch'],[1,2])
            for split in ['train','val','test']:
                self.assertEqual(result['splits'][split]['n'],9)
                rows=json.loads((out/f'{split}_predictions.json').read_text())
                self.assertEqual(len({r['id'] for r in rows}),9)


if __name__=='__main__':
    unittest.main(verbosity=2)
