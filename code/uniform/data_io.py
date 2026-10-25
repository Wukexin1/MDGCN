from pathlib import Path
import math
import numpy as np
import torch
from common import read_json,sha

class Clouds:
    def __init__(self,root,job,split):
        self.root=Path(root)/job['dataset'];self.job=job
        m=read_json(self.root/'manifest.json');self.classes=m['class_names']
        self.records=[];self.x=[];self.mask=[];self.y=[];self.geo=[]
        if job['dataset']=='controlled':
            for r in m['records']:
                if r['split']!=split or not (r['category']=='normal' or r['rho']==job['rho']):continue
                # Strict input allow-list; gt_mask/displacement never enter the network.
                with np.load(self.root/r['file'],allow_pickle=False) as z:
                    x=np.array(z['xyz'],dtype=np.float64,copy=True);x-=x.mean(0);x/=np.linalg.norm(x,axis=1).max()
                    self.x.append(x.T.astype(np.float32));self.mask.append(np.array(z['candidate_mask'],dtype=bool));self.y.append(int(z['cloud_label']))
                self.records.append(r);self.geo.append(None)
        else:
            if sha(self.root/'inputs.npz')!=m['inputs_sha256']:raise ValueError('Input cache hash mismatch')
            geo=job['variant'].endswith('geo');norm=job['variant'] in ['norm','norm_geo']
            glookup={r['id']:r['file'] for r in m['geodesic']}
            with np.load(self.root/'inputs.npz',allow_pickle=False) as z:
                x=z['geo_xyz' if geo else 'xyz'];n=z['geo_norm' if geo else 'norm'];labels=z['labels']
                masks=z['mask_geod' if geo else 'mask_'+job['tau']]
                for i,r in enumerate(m['records']):
                    if r['split']!=split:continue
                    self.x.append(np.concatenate([x[i],n[i]],axis=1).T if norm else x[i].T.copy())
                    self.mask.append(masks[i].copy());self.y.append(int(labels[i]));self.records.append(r)
                    self.geo.append(self.root/glookup[r['id']] if geo else None)
        if not self.records:raise ValueError('Empty split')

    def __len__(self):return len(self.records)

    def batch(self,ids,device,augment=False):
        x=torch.from_numpy(np.stack([self.x[i] for i in ids])).to(device)
        mask=torch.from_numpy(np.stack([self.mask[i] for i in ids])).to(device)
        y=torch.tensor([self.y[i] for i in ids],dtype=torch.long,device=device)
        g=None if self.geo[ids[0]] is None else torch.from_numpy(np.stack([np.load(self.geo[i],mmap_mode='r',allow_pickle=False) for i in ids])).to(device)
        if augment:
            angle=torch.rand(len(ids),device=device)*2*math.pi;c=angle.cos()[:,None];s=angle.sin()[:,None]
            for a,b in ([(0,2),(3,5)] if x.shape[1]==6 else [(0,2)]):
                xx=x[:,a].clone();zz=x[:,b].clone();x[:,a]=c*xx+s*zz;x[:,b]=-s*xx+c*zz
        return x,mask,y,g

def batches(n,size,shuffle=False):
    ids=np.random.permutation(n).tolist() if shuffle else list(range(n))
    groups=[ids[i:i+size] for i in range(0,n,size)]
    if shuffle and len(groups[-1])==1 and len(groups)>1:groups[-2].extend(groups.pop())
    return groups

