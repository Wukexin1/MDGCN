"""Validation-only selection, full cloud evaluation and atomic epoch resume (Python 3.8)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import copy
import platform
import random
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import ROOT,sha,read_json,write_json,fingerprint
from models import Model,NAMES
from data_io import Clouds,batches

STOP=False
def stop_handler(signum,frame):
    global STOP
    STOP=True
    print('Stop requested: will save at the next completed epoch.',flush=True)

def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])

def atomic_save(obj,path,previous=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f:
        torch.save(obj,f);f.flush();os.fsync(f.fileno())
    if previous and path.exists():os.replace(path,path.with_name('previous.pt'))
    os.replace(tmp,path)

def load_checkpoint(path):
    # Only locally generated trusted checkpoint files contain RNG/optimizer Python objects.
    return torch.load(path,map_location='cpu',weights_only=False)

def scores(y,p,classes):
    cm=np.zeros((classes,classes),dtype=np.int64);np.add.at(cm,(y,p),1)
    tp=cm.diagonal();support=cm.sum(1)
    recall=np.divide(tp,support,out=np.zeros(classes,float),where=support>0)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(classes,float),where=cm.sum(0)>0)
    den=support+cm.sum(0);f1=np.divide(2*tp,den,out=np.zeros(classes,float),where=den>0)
    return {'n':int(cm.sum()),'oa':float(tp.sum()/cm.sum()),'ba':float(recall[support>0].mean()),
        'macro_f1':float(f1.mean()),'class_recall':recall.tolist(),'class_precision':precision.tolist(),
        'class_f1':f1.tolist(),'confusion_matrix':cm.tolist(),'class_support':support.tolist()}

def evaluate(model,data,size,device):
    model.eval();ys=[];ps=[];rows=[];loss=0.
    with torch.inference_mode():
        for ids in batches(len(data),size):
            x,mask,y,g=data.batch(ids,device);logits=model(x,mask,g)
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite evaluation')
            loss+=float(nn.functional.cross_entropy(logits,y,reduction='sum'))
            prob=logits.softmax(1).cpu().numpy();pred=prob.argmax(1);ys.extend(y.cpu().tolist());ps.extend(pred.tolist())
            rows.extend({'id':data.records[i]['id'],'truth':int(data.y[i]),'prediction':int(p),
                         'probabilities':q.tolist()} for i,p,q in zip(ids,pred,prob))
    met=scores(np.array(ys),np.array(ps),len(data.classes));met['cross_entropy']=loss/len(data)
    return met,rows

def smooth_loss(logits,y):
    # Supplied util.cal_loss convention: epsilon distributed over C-1 other classes.
    classes=logits.shape[1];one=nn.functional.one_hot(y,classes).to(logits.dtype)
    target=one*.8+(1-one)*(.2/(classes-1))
    return -(target*nn.functional.log_softmax(logits,dim=1)).sum(1).mean()

def run(job,data_root,out,device='cuda',threads=4,keep_every=50,stop_after=None):
    global STOP
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seed_all(job['seed'],threads)
    train=Clouds(data_root,job,'train');val=Clouds(data_root,job,'val')
    model=Model(job,len(train.classes)).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=job['lr'],momentum=.9,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=job['epochs'],eta_min=.001)
    source={n:sha(ROOT/n) for n in ['assembled_models.py','models.py','data_io.py','train.py']}
    datahash=sha(Path(data_root)/job['dataset']/'manifest.json')
    identity={'job':job,'programs':source,'data_manifest':datahash,'torch':str(torch.__version__),
              'device_type':str(device).split(':')[0],'threads':threads}
    run_hash=fingerprint(identity)
    config=dict(identity,run_hash=run_hash,class_names=train.classes,model_display=NAMES[job['model']],
       parameters=sum(p.numel() for p in model.parameters()),python=platform.python_version(),
       cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
       selection='validation OA, then lower ordinary CE; earliest exact tie',
       training='SGD lr=0.1 momentum=0.9 wd=1e-4; cosine eta_min=.001 after each epoch; smoothing epsilon=.2 over other classes; random Y rotation; all training clouds',
       evaluation='frozen selected model, no augmentation, sample ID order; test evaluated only after all epochs',
       resume='completed epoch boundaries; model/optimizer/scheduler/RNG restored; previous.pt fallback')
    if (out/'config.json').exists() and read_json(out/'config.json')['run_hash']!=run_hash:
        raise ValueError('Run configuration/data/code/environment changed. Use a new results directory.')
    write_json(out/'config.json',config)
    if (out/'result.json').exists():
        if read_json(out/'result.json')['run_hash']!=run_hash:raise ValueError('Result identity mismatch')
        print('Already complete:',job['id'],flush=True);return True
    start_epoch=0;history=[];best_key=(-1.,-float('inf'));best_state=None;best_epoch=0;best_val=None;best_rows=[]
    checkpoint=None;errors=[]
    for path in [out/'last.pt',out/'previous.pt']:
        if not path.exists():continue
        try:checkpoint=load_checkpoint(path)
        except Exception as e:errors.append(str(path)+': '+repr(e));continue
        if checkpoint['run_hash']!=run_hash:raise ValueError('Checkpoint identity mismatch')
        break
    if checkpoint is None and errors:raise RuntimeError('No readable recovery checkpoint: '+str(errors))
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'],strict=True);optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng']);start_epoch=checkpoint['epoch'];history=checkpoint['history']
        best_key=tuple(checkpoint['best_key']);best_state=checkpoint['best_state'];best_epoch=checkpoint['best_epoch'];best_val=checkpoint['best_val'];best_rows=checkpoint['best_rows']
        print('Resuming',job['id'],'after epoch',start_epoch,flush=True)
        if errors:write_json(out/'recovery.json',{'errors':errors,'resumed_epoch':start_epoch})
    for epoch in range(start_epoch+1,job['epochs']+1):
        tick=time.monotonic();model.train();total=0.;seen=0;truth=[];pred=[]
        lr=optimizer.param_groups[0]['lr']
        for ids in batches(len(train),job['batch_size'],True):
            x,mask,y,g=train.batch(ids,device,augment=True);optimizer.zero_grad(set_to_none=True)
            logits=model(x,mask,g);loss=smooth_loss(logits,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
            truth.extend(y.cpu().tolist());pred.extend(logits.detach().argmax(1).cpu().tolist())
        vm,vr=evaluate(model,val,job['batch_size'],device);key=(vm['oa'],-vm['cross_entropy'])
        if key>best_key:
            best_key=key;best_epoch=epoch;best_val=vm;best_rows=vr
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step()
        hist={'epoch':epoch,'learning_rate':lr,'train_augmented_online':scores(np.array(truth),np.array(pred),len(train.classes)),
              'train_smoothed_loss':total/seen,'validation':vm,'seconds':time.monotonic()-tick,'selected_epoch':best_epoch}
        history.append(hist)
        state={'run_hash':run_hash,'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':rng_state(),'history':history,'best_key':best_key,
            'best_state':best_state,'best_epoch':best_epoch,'best_val':best_val,'best_rows':best_rows}
        atomic_save(state,out/'last.pt',previous=True)
        atomic_save({'model':best_state,'epoch':best_epoch,'validation':best_val,'run_hash':run_hash},out/'selected.pt')
        write_json(out/'history.json',history);write_json(out/'selected_validation_predictions.json',best_rows)
        if keep_every>0 and epoch%keep_every==0:
            atomic_save({'model':state['model'],'epoch':epoch,'run_hash':run_hash},out/'snapshots'/('epoch_%04d.pt'%epoch))
        print('%s epoch %d/%d train OA %.4f val OA %.4f val BA %.4f selected %d %.1fs' %
              (job['id'],epoch,job['epochs'],hist['train_augmented_online']['oa'],vm['oa'],vm['ba'],best_epoch,hist['seconds']),flush=True)
        write_json(out/'status.json',{'state':'training','epoch':epoch,'epochs':job['epochs'],'selected_epoch':best_epoch})
        if STOP or (out.parent/'STOP').exists() or (stop_after is not None and epoch>=stop_after):
            write_json(out/'status.json',{'state':'paused','epoch':epoch});return False
    model.load_state_dict(best_state,strict=True)
    final={}
    for split in ['train','val','test']:
        dataset=train if split=='train' else val if split=='val' else Clouds(data_root,job,'test')
        met,rows=evaluate(model,dataset,job['batch_size'],device);final[split]=met
        write_json(out/(split+'_predictions.json'),{'class_names':dataset.classes,'rows':rows})
    write_json(out/'result.json',{'run_hash':run_hash,'job':job,'selected_epoch':best_epoch,'metrics':final,
        'selected_checkpoint_sha256':sha(out/'selected.pt'),'epochs_completed':job['epochs'],
        'last_checkpoint_sha256':sha(out/'last.pt')})
    write_json(out/'status.json',{'state':'complete','epoch':job['epochs']})
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--keep-every',type=int,default=50)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    ok=run(read_json(a.job),a.data,a.out,a.device,a.threads,a.keep_every);sys.exit(0 if ok else 75)
