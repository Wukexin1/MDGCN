"""Remote GPU entry point. Run smoke first; formal studies can resume by run ID."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def package_results():
    output=ROOT/'gpu_results';output.mkdir(exist_ok=True)
    archive=ROOT/'IISE_results_to_return.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            # Keep checkpoints on the GPU computer; avoid multi-GB return files.
            if p.is_file() and p.suffix in ['.json','.log','.txt']:
                z.write(p,p.relative_to(ROOT))
    return archive


def smoke(batch_size):
    import numpy as np
    import torch
    from revised_graph import RevisedGraphClassifier
    from train_revised import Clouds
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is unavailable in this Python environment. Activate the GPU PyTorch environment in VS Code and rerun.')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report={'scope':'GPU environment and full-size training-step check, not scientific accuracy',
            'python':sys.version,'python_executable':sys.executable,'torch':str(torch.__version__),
            'numpy':np.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
            'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
            'batch_size':batch_size,'points_per_cloud':2000,'checks':[]}
    data=Clouds(ROOT/'data/controlled_v1','train',.02)
    for operator in ['standard','hard','soft']:
        torch.manual_seed(24);torch.cuda.manual_seed_all(24)
        model=RevisedGraphClassifier(operator=operator).cuda()
        x,mask,y=data.batch(list(range(batch_size)),'cuda')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x,mask),y)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Non-finite loss/gradient in '+operator)
            optimizer.step()
        torch.cuda.synchronize()
        report['checks'].append({'operator':operator,'two_steps_seconds':time.monotonic()-start,
                                 'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                                 'parameters':sum(p.numel() for p in model.parameters()),'finite_gradients':True})
        del model,optimizer,x,mask,y
        torch.cuda.empty_cache()
    (ROOT/'gpu_results/smoke_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def jobs(stage):
    seeds=[24,42,2026]
    grid=[]
    if stage in ['main','all']:
        for dataset in ['original_synthetic_v1','original_industrial_v1']:
            for operator in ['standard','hard','soft']:
                for fusion in ['replace','add']:
                    for seed in seeds:
                        grid.append((dataset,None,operator,fusion,seed))
    if stage in ['imbalance','all']:
        for rho in [.005,.01,.02,.05,.1]:
            for operator,fusion in [('standard','replace'),('hard','add'),('soft','add')]:
                for seed in seeds:
                    grid.append(('controlled_v1',rho,operator,fusion,seed))
    return grid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['smoke','main','imbalance','all','pack'],default='smoke')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--epochs',type=int,default=250)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--resume-partial',action='store_true',help='Restart a partially trained run from epoch 1 after preserving its folder')
    a=p.parse_args()
    output=ROOT/'gpu_results';output.mkdir(exist_ok=True)
    if a.stage=='pack':
        print(package_results());return
    if a.batch_size<2:
        p.error('Batch size must be at least 2')
    stage_log=output/f'{a.stage}_status.json'
    try:
        if a.stage=='smoke':
            smoke(a.batch_size)
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; activate the correct remote Python environment.')
            if not (output/'smoke_report.json').exists():
                raise RuntimeError('Run --stage smoke successfully before formal training.')
            grid=jobs(a.stage)
            settings=dict(stage=a.stage,epochs=a.epochs,batch_size=a.batch_size,threads=a.threads,
                          seeds=[24,42,2026],planned_runs=len(grid),
                          scope='Revised matched coordinate/Euclidean model; not a historical or normal/geodesic reproduction',
                          smoke=json.loads((output/'smoke_report.json').read_text()),
                          code_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['run_gpu.py','train_revised.py','revised_graph.py']})
            (output/f'{a.stage}_plan.json').write_text(json.dumps(settings,indent=2))
            for i,(dataset,rho,operator,fusion,seed) in enumerate(grid,1):
                ratio='original' if rho is None else f'rho{rho:g}'
                fingerprint=hashlib.sha256((ROOT/'data'/dataset/'manifest.json').read_bytes()+
                                          (ROOT/'train_revised.py').read_bytes()+
                                          (ROOT/'revised_graph.py').read_bytes()).hexdigest()[:8]
                name=f'{dataset}_{ratio}_{operator}_{fusion}_s{seed}_e{a.epochs}_b{a.batch_size}_{fingerprint}'
                run_out=output/name
                if (run_out/'final_metrics.json').exists():
                    print(f'[{i}/{len(grid)}] Completed run preserved: {name}',flush=True)
                    continue
                if run_out.exists():
                    if not a.resume_partial:
                        raise RuntimeError(f'Partial run found: {name}. Add --resume-partial to preserve it and restart that run from epoch 1.')
                    run_out.rename(output/(name+'_partial_'+time.strftime('%Y%m%d_%H%M%S')))
                command=[sys.executable,str(ROOT/'train_revised.py'),'--data',str(ROOT/'data'/dataset),'--out',str(run_out),
                         '--operator',operator,'--fusion',fusion,'--seed',str(seed),'--epochs',str(a.epochs),
                         '--batch-size',str(a.batch_size),'--device','cuda','--threads',str(a.threads)]
                if rho is not None: command+=['--rho',str(rho)]
                print(f'[{i}/{len(grid)}] {name}',flush=True)
                stage_log.write_text(json.dumps(dict(status='running',run=name,index=i,total=len(grid)),indent=2))
                with (output/(name+'.log')).open('w') as log:
                    proc=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    for line in proc.stdout:
                        log.write(line);log.flush();print(line,end='',flush=True)
                    status=proc.wait()
                if status:
                    raise RuntimeError(f'Run failed with exit {status}: {name}. Return the result archive for diagnosis.')
                package_results()
        stage_log.write_text(json.dumps(dict(status='complete',stage=a.stage),indent=2))
    except Exception as e:
        stage_log.write_text(json.dumps(dict(status='failed',stage=a.stage,error=str(e)),indent=2))
        print('ERROR:',e,file=sys.stderr)
        print('Return this diagnostic archive:',package_results(),flush=True)
        raise
    print('Return this result archive:',package_results(),flush=True)


if __name__=='__main__':
    main()
