"""Full-point-size GPU check before formal training, for both datasets and input types."""
import gc
import time
import torch
from models import Model
from data_io import Clouds
from train import seed_all, smooth_loss
from common import write_json


def smoke(plan, data, out, device='cuda', threads=4, stop_requested=lambda:False):
    seen, report = set(), []
    seed_all(123, threads)
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the active Python environment. No automatic CPU fallback.')
    for job in plan['jobs']:
        key = (job['dataset'], job['model'], job['variant'])
        if key in seen:
            continue
        if stop_requested():
            raise InterruptedError('Paused during smoke; rerun the original command to repeat incomplete smoke checks')
        seen.add(key)
        dataset = Clouds(data, job, 'train')
        ids = list(range(min(job['batch_size'], len(dataset))))
        if len(ids) < 2:
            raise ValueError('Insufficient samples for batch normalization')
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        tick = time.monotonic()
        model = Model(job, len(dataset.classes)).to(device).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        x, mask, y, g = dataset.batch(ids, device, augment=True)
        logits = model(x, mask, g)
        loss = smooth_loss(logits, y)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite smoke loss')
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite smoke gradient')
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            a, b = model(x[:1], mask[:1]), model(x[:1], mask[:1])
        if not torch.isfinite(a).all() or not torch.equal(a, b):
            raise ValueError('Invalid or unstable fixed-model evaluation')
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        item = dict(dataset=job['dataset'], model=job['model'], variant=job['variant'],
                    batch=len(ids), points=int(x.shape[-1]), loss=float(loss.detach()),
                    seconds=time.monotonic()-tick,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)
        report.append(item)
        write_json(out, {'complete':False, 'checks':report})
        print('SMOKE', item, flush=True)
        del model, optimizer, x, mask, y, g, loss, logits, a, b, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = dict(complete=True, checks=report, device=device, torch=str(torch.__version__),
                  batch_size=plan['batch_size'], determinism='same warn_only=True setting as completed GPU suite')
    write_json(out, result)
    return result
