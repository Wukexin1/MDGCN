"""Validation-selected, full-cloud evaluation for the controlled revision data.

Default model dimensions and point count are full size. Runs require explicit
output directories, retain per-cloud predictions, and never select on test data.
Use a fixture dataset for integration checks; do not tune on test diagnostics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import time
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from revised_graph import RevisedGraphClassifier


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(y, pred, classes):
    cm = np.zeros((classes, classes), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    support = cm.sum(1)
    recall = np.divide(tp, support, out=np.zeros(classes, float), where=support > 0)
    f1 = np.divide(2*tp, cm.sum(0)+support, out=np.zeros(classes, float), where=cm.sum(0)+support > 0)
    return dict(n=int(cm.sum()), oa=float(tp.sum()/cm.sum()),
                ba=float(recall[support > 0].mean()), macro_f1=float(f1.mean()),
                confusion_matrix=cm.tolist(), class_support=support.tolist())


class Clouds:
    def __init__(self, folder, split, rho):
        self.root = Path(folder)
        manifest = json.loads((self.root/'manifest.json').read_text())
        original = manifest.get('study_type') == 'original_cloud_classification'
        self.records = [r for r in manifest['records'] if r['split'] == split and
                        (original or r['category'] == 'normal' or r['rho'] == rho)]
        self.xyz, self.mask, self.labels = [], [], []
        for r in self.records:
            with np.load(self.root/r['file'], allow_pickle=False) as z:
                # Explicit input allow-list: gt_mask and displacement are never read.
                x = np.array(z['xyz'], dtype=np.float64, copy=True)
                x -= x.mean(0)
                radius = np.linalg.norm(x, axis=1).max()
                if not np.isfinite(radius) or radius <= 0:
                    raise ValueError('Invalid normalization radius')
                self.xyz.append(torch.from_numpy((x/radius).astype(np.float32).T))
                self.mask.append(torch.from_numpy(np.array(z['candidate_mask'], dtype=bool)))
                self.labels.append(int(z['cloud_label']))

    def __len__(self):
        return len(self.records)

    def batch(self, indices, device, augment=False):
        x = torch.stack([self.xyz[i] for i in indices]).to(device)
        mask = torch.stack([self.mask[i] for i in indices]).to(device)
        y = torch.tensor([self.labels[i] for i in indices], device=device)
        if augment:
            angle = torch.rand(len(indices), device=device)*2*math.pi
            c, s = angle.cos()[:, None], angle.sin()[:, None]
            xx, zz = x[:, 0, :].clone(), x[:, 2, :].clone()
            x[:, 0, :], x[:, 2, :] = c*xx+s*zz, -s*xx+c*zz
        return x, mask, y


def batches(n, size, shuffle=False):
    indices = np.random.permutation(n).tolist() if shuffle else list(range(n))
    groups = [indices[i:i+size] for i in range(0,n,size)]
    # Include every training cloud while avoiding singleton classifier BatchNorm.
    if shuffle and len(groups[-1]) == 1 and len(groups)>1:
        groups[-2].extend(groups.pop())
    return groups


def evaluate(model, data, batch_size, device, classes):
    model.eval()
    predictions, truth, rows, loss = [], [], [], 0.
    with torch.inference_mode():
        for ii in batches(len(data), batch_size):
            x, mask, y = data.batch(ii, device)
            logits = model(x, mask)
            loss += float(nn.functional.cross_entropy(logits, y, reduction='sum'))
            prob = logits.softmax(1).cpu().numpy()
            pred = prob.argmax(1)
            predictions.extend(pred.tolist()); truth.extend(y.cpu().tolist())
            rows.extend(dict(id=data.records[i]['id'], source_id=data.records[i]['source_id'],
                             truth=int(data.labels[i]), prediction=int(p), probabilities=v.tolist())
                        for i,p,v in zip(ii,pred,prob))
    result = metrics(np.array(truth), np.array(predictions), classes)
    result['cross_entropy'] = loss/len(data)
    return result, rows


def run(args):
    out, root = Path(args.out), Path(args.data)
    if out.exists():
        raise ValueError('Use a new run directory')
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('study_type') == 'original_cloud_classification':
        args.rho = None
    elif args.rho not in manifest['fractions']:
        raise ValueError('Unknown fraction')
    sources = {s:set(r['source_id'] for r in manifest['records'] if r['split']==s) for s in ['train','val','test']}
    if any(sources[a] & sources[b] for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('Source-cloud leakage across partitions')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    classes = len(manifest['class_names'])
    widths = tuple(args.widths)
    model = RevisedGraphClassifier(classes, args.operator, args.fusion, args.k, widths, args.embedding).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/100)
    train, val = Clouds(root,'train',args.rho), Clouds(root,'val',args.rho)
    out.mkdir(parents=True)
    config = dict(vars(args), classes=manifest['class_names'], parameters=sum(p.numel() for p in model.parameters()),
                  data_manifest_sha256=digest(root/'manifest.json'),
                  program_sha256={n:digest(Path(__file__).parent/n) for n in ['train_revised.py','revised_graph.py']},
                  checkpoint_selection='maximum validation BA, then minimum validation cross-entropy; earliest exact tie',
                  candidate_input='training-normal calibrated geometric mask; no ground truth features',
                  training_clouds=len(train), validation_clouds=len(val), torch_version=torch.__version__)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    best, history = (-float('inf'), -float('inf')), []
    start = time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_sum, seen = 0., 0
        for ii in batches(len(train), args.batch_size, True):
            x, mask, y = train.batch(ii,device,augment=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x,mask),y,label_smoothing=.2)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite loss')
            loss.backward(); optimizer.step()
            seen += len(ii); loss_sum += float(loss.detach())*len(ii)
        vm, _ = evaluate(model,val,args.batch_size,device,classes)
        key = (vm['ba'], -vm['cross_entropy'])
        if key > best:
            best = key
            torch.save({'model':model.state_dict(), 'epoch':epoch, 'validation':vm},out/'selected.pt')
        history.append(dict(epoch=epoch, training_seen=seen, online_training_loss=loss_sum/seen,
                            validation=vm, lr=optimizer.param_groups[0]['lr'], elapsed_seconds=time.monotonic()-start))
        (out/'history.json').write_text(json.dumps(history,indent=2))
        scheduler.step()
        print(f'epoch {epoch}/{args.epochs}: validation BA={vm["ba"]:.4f}; all {seen} training clouds used',flush=True)
    selected = torch.load(out/'selected.pt',map_location=device,weights_only=True)
    model.load_state_dict(selected['model'])
    final = dict(selected_epoch=selected['epoch'], checkpoint_sha256=digest(out/'selected.pt'),
                 elapsed_training_seconds=time.monotonic()-start,
                 test_loaded_after_checkpoint_selection=True, splits={})
    # Test arrays are opened only after training and checkpoint selection finish.
    for split,data in [('train',train),('val',val),('test',Clouds(root,'test',args.rho))]:
        m, rows = evaluate(model,data,args.batch_size,device,classes)
        final['splits'][split] = m
        (out/f'{split}_predictions.json').write_text(json.dumps(rows,indent=2))
    (out/'final_metrics.json').write_text(json.dumps(final,indent=2))
    return final


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True); p.add_argument('--out',required=True)
    p.add_argument('--rho',type=float,default=.02)
    p.add_argument('--operator',choices=['standard','hard','soft'],default='hard')
    p.add_argument('--fusion',choices=['add','replace'],default='add')
    p.add_argument('--epochs',type=int,default=250); p.add_argument('--seed',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=4); p.add_argument('--k',type=int,default=15)
    p.add_argument('--widths',type=int,nargs=4,default=[64,64,128,256])
    p.add_argument('--embedding',type=int,default=1024); p.add_argument('--lr',type=float,default=.1)
    p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=4)
    args = p.parse_args()
    if args.batch_size < 2 or args.epochs < 1:
        p.error('Require batch size >= 2 and epochs >= 1')
    run(args)
