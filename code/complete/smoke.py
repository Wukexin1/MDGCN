"""One real-size training step per implementation, separate from formal results."""
import gc
import time
import torch
from common import ROOT,write_json,read_json,fingerprint,sha
from models import Model
from data_io import Clouds
from train import seed_all,smooth_loss

def smoke(plan,data,out,device='cuda',threads=4):
    seen=set();report=[];seed_all(123,threads)
    for job in plan['jobs']:
        key=(job['model'],job['variant'])
        if key in seen:continue
        seen.add(key);ds=Clouds(data,job,'train')
        ids=list(range(min(job['batch_size'],len(ds))))
        if len(ids)<2:raise ValueError('Insufficient training smoke samples')
        if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        tick=time.monotonic();model=Model(job,len(ds.classes)).to(device).train()
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9)
        x,mask,y,g=ds.batch(ids,device,True)
        logits=model(x,mask,g);loss=smooth_loss(logits,y);loss.backward();opt.step()
        if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('Nonfinite gradient')
        model.eval()
        with torch.inference_mode():
            a=model(x[:1],mask[:1],None if g is None else g[:1])
            b=model(x[:1],mask[:1],None if g is None else g[:1])
        if not torch.equal(a,b):raise ValueError('Evaluation changes without input changes')
        if str(device).startswith('cuda'):torch.cuda.synchronize()
        item={'model':job['model'],'variant':job['variant'],'batch':len(ids),'points':int(x.shape[-1]),
              'loss':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),
              'seconds':time.monotonic()-tick,'peak_cuda_bytes':torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None}
        report.append(item);write_json(out,{'complete':False,'checks':report});print('SMOKE',item,flush=True)
        del model,opt,x,mask,y,g,loss,logits,a,b,ds;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result={'complete':True,'checks':report,'device':str(device),'torch':str(torch.__version__),
            'batch_size':plan['batch_size'],'plan_sha256':fingerprint(plan),
            'programs':{p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}}
    write_json(out,result);return result

