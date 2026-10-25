"""Meaningful local checks, not research results. Run with Python >=3.8 + torch/numpy."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from common import ROOT,sha,write_json,read_json
from models import Model
from plan import make_plan
from train import run,load_checkpoint,seed_all
import assembled_models as base

def fixture(root):
    folder=Path(root)/'surface';folder.mkdir(parents=True)
    rng=np.random.RandomState(17);x=rng.normal(size=(18,128,3)).astype('float32');mask=rng.rand(18,128)<.2
    z=folder/'inputs.npz';np.savez_compressed(z,xyz=x,norm=x,geo_xyz=x,geo_norm=x,mask_base=mask,labels=np.arange(18)%3)
    rows=[{'id':str(i),'split':'train' if i<9 else 'val' if i<12 else 'test','label':i%3} for i in range(18)]
    write_json(folder/'manifest.json',{'records':rows,'class_names':['a','b','c'],'inputs_sha256':sha(z),'geodesic':[]})

class Checks(unittest.TestCase):
    def test_python38_syntax(self):
        for p in ROOT.glob('*.py'):ast.parse(p.read_text(),feature_version=(3,8))

    def test_plan_deduplication(self):
        p=make_plan();self.assertEqual(p['counts'],{'core':36,'fraction':45,'tables':114,'neighbors':120,'threshold':48})
        self.assertEqual(p['total'],363);self.assertEqual(len({j['id'] for j in p['jobs']}),363)
        self.assertTrue(all(j['model']!='pointnet2' for j in p['jobs'] if j['stage']=='neighbors'))

    def test_geo_mask_no_contamination(self):
        seed_all(12,2);x=torch.randn(2,3,64);mask=torch.arange(64).repeat(2,1)%4==0
        g=-torch.cdist(x.transpose(1,2),x.transpose(1,2));before=g.clone()
        for local in [True,False]:
            idx=base.knn_HGCNN_geo(x,5,mask,g,local)
            scores=g.clone();excluded=~mask if local else mask
            expected=torch.where(excluded[:,None,:],g.min(-1,keepdim=True)[0],g).topk(5,dim=-1)[1]
            self.assertTrue(torch.equal(idx,expected));self.assertTrue(torch.equal(g,before))
        centers=torch.tensor([[2,5],[1,7]])
        idx=base.query_geodesic_knn(5,g,centers)
        expected=base.index_points_pt2(g,centers).topk(5,dim=-1,largest=True)[1]
        self.assertTrue(torch.equal(idx,expected))

    def test_all_model_backward_and_singleton_eval(self):
        seed_all(12,2);seen=set()
        for job in make_plan()['jobs']:
            key=(job['model'],job['variant'])
            if key in seen:continue
            seen.add(key)
            with self.subTest(key=key):
                m=Model(job,3);n=128;c=6 if job['variant'] in ['norm','norm_geo'] else 3
                x=torch.randn(2,c,n);mask=torch.arange(n).repeat(2,1)%5==0;g=-torch.cdist(x[:,:3].transpose(1,2),x[:,:3].transpose(1,2));before=g.clone()
                y=m(x,mask,g);y.sum().backward();self.assertEqual(tuple(y.shape),(2,3))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
                self.assertTrue(torch.equal(g,before));m.eval()
                with torch.inference_mode():a=m(x[:1],mask[:1],g[:1]);b=m(x[:1],mask[:1],g[:1])
                self.assertEqual(tuple(a.shape),(1,3));self.assertTrue(torch.equal(a,b))
        self.assertEqual(len(seen),29)

    def test_resume_equals_uninterrupted_and_corrupt_fallback(self):
        for name in ['pointnet','hard']:
            with self.subTest(model=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);fixture(root/'data')
                job=dict(id='fixture_'+name,dataset='surface',model=name,variant='xyz',k=5,tau='base',rho=None,
                         seed=24,batch_size=4,epochs=4,lr=.1,stage='fixture')
                a=root/'a';b=root/'b';c=root/'c'
                run(job,root/'data',a,'cpu',2,0)
                self.assertFalse(run(job,root/'data',b,'cpu',2,0,stop_after=2));self.assertFalse((b/'test_predictions.json').exists())
                run(job,root/'data',b,'cpu',2,0)
                run(job,root/'data',c,'cpu',2,0,stop_after=2);(c/'last.pt').write_bytes(b'interrupted write fixture')
                run(job,root/'data',c,'cpu',2,0)
                aa=load_checkpoint(a/'last.pt')
                for folder in [b,c]:
                    bb=load_checkpoint(folder/'last.pt')
                    for key,value in aa['model'].items():self.assertTrue(torch.equal(value,bb['model'][key]),key)
                    self.assertEqual(aa['best_epoch'],bb['best_epoch'])
                    self.assertEqual(read_json(a/'test_predictions.json'),read_json(folder/'test_predictions.json'))

if __name__=='__main__':unittest.main(verbosity=2)
