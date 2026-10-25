"""Explicit model identities and signatures for prospective training."""
from types import SimpleNamespace
import torch
from torch import nn
import assembled_models as base

NAMES = {'pointnet':'PointNet','pointnet2':'PointNet++','dgcnn':'DGCNN',
         'mvgcn':'MVGCN (supplied HGCNN)','hard':'MDGCN1 hard + DSRF',
         'soft':'MDGCN2 soft + DSRF','dsrf':'DSRF-only (same MDGCN backbone)'}

class StandardLocal(nn.Module):
    def forward(self,x,k,mask,local=True,idx=None):
        return base.get_graph_feature_HGCNN(x,k,mask,local=local,idx=None)

class Model(nn.Module):
    def __init__(self,job,classes):
        super().__init__()
        self.job=job
        a=SimpleNamespace(k=job['k'],kl=job['k'],output_C=classes,emb_dims=1024,dropout=.5)
        setattr(a,'self',job['model']=='soft')
        name=job['model'];variant=job['variant']
        suffix={'xyz':'','norm':'_norm','cov':'_cov','geo':'_geo','norm_geo':'_norm_geo'}[variant]
        self.cov_adapter=name in ['pointnet','dgcnn'] and variant=='cov'
        clsname={'pointnet':'PointNet','pointnet2':'PointNet2','dgcnn':'DGCNN',
                 'mvgcn':'HGCNN','hard':'Graphcnn','soft':'Graphcnn','dsrf':'Graphcnn'}[name]
        cls=getattr(base,clsname+('' if self.cov_adapter else suffix))
        self.net=cls(a,output_channels=classes) if name.startswith('pointnet') else cls(a)
        if self.cov_adapter:
            if name=='pointnet':self.net.conv1=nn.Conv1d(12,64,1,bias=False)
            else:self.net.conv1[0]=nn.Conv2d(24,64,1,bias=False)
        if name=='dsrf':
            if variant!='xyz':raise ValueError('DSRF-only fraction study prespecifies xyz')
            for i in range(1,6):setattr(self.net,'graph_layer'+str(i),StandardLocal())

    def forward(self,x,mask,geo=None):
        name=self.job['model'];v=self.job['variant']
        if self.cov_adapter:x=torch.cat([x,base.get_cov_features(x,self.job['k'])],dim=1)
        if v.endswith('geo'):
            if name in ['hard','soft']:return self.net(x,mask,geo,False)
            if name=='mvgcn':return self.net(x,mask,geo)
            return self.net(x,geo,False)
        if name in ['hard','soft','dsrf','mvgcn']:return self.net(x,mask,False)
        return self.net(x,False)
