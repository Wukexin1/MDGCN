"""One entry point: verify inputs, smoke, run/resume, summarize and pack."""
import argparse
from collections import defaultdict
import csv
import fcntl
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
from common import ROOT,sha,read_json,write_json,fingerprint
from plan import make_plan

STOP=False;CHILD=None
def stop_handler(signum,frame):
    global STOP
    STOP=True
    print('Stopping after current epoch is saved...',flush=True)
    if CHILD is not None and CHILD.poll() is None:CHILD.send_signal(signal.SIGTERM)

def summarize(results,plan):
    rows=[];groups=defaultdict(list)
    for job in plan['jobs']:
        p=results/job['id']/'result.json'
        if not p.exists():continue
        r=read_json(p);row=dict(job,selected_epoch=r['selected_epoch'])
        for split,met in r['metrics'].items():
            for key in ['oa','ba','macro_f1','cross_entropy']:row[split+'_'+key]=met[key]
        rows.append(row)
        group={k:job[k] for k in ['dataset','model','variant','k','tau','rho','batch_size','epochs','stage']}
        groups[fingerprint(group)].append((group,row))
    summary=[]
    for vals in groups.values():
        group=vals[0][0];item=dict(group,seeds_completed=[r['seed'] for _,r in vals],complete=len(vals)==3)
        for metric in ['train_oa','val_oa','test_oa','test_ba','test_macro_f1']:
            data=[r[metric] for _,r in vals];item[metric+'_mean']=float(np.mean(data));item[metric+'_sd']=float(np.std(data,ddof=1)) if len(data)>1 else None
        summary.append(item)
    write_json(results/'summary.json',{'completed':len(rows),'planned':len(plan['jobs']),'groups':summary})
    if rows:
        with open(results/'per_run.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return len(rows)

def package(results,plan,full=False):
    results=Path(results);summarize(results,plan)
    name='IISE_complete_full_backup.zip' if full else 'IISE_complete_return.zip'
    target=results/name;tmp=target.with_suffix('.zip.tmp')
    files=[]
    for p in sorted(results.rglob('*')):
        if not p.is_file() or p.suffix in ['.zip','.tmp'] or p.name=='runner.lock':continue
        if p.suffix=='.pt' and not full:continue
        files.append((p,'results/'+str(p.relative_to(results))))
    for p in sorted(ROOT.glob('*.py')):files.append((p,'program/'+p.name))
    for rel in ['assembly_manifest.json','README_中文.md','experiment_plan.json','audit','author_information','fixed_splits']:
        p=ROOT/rel
        if p.is_file():files.append((p,rel))
        elif p.is_dir():files.extend((q,str(q.relative_to(ROOT))) for q in sorted(p.rglob('*')) if q.is_file())
    for p in sorted((ROOT/'data').rglob('manifest.json')):files.append((p,str(p.relative_to(ROOT))))
    checks={}
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,allowZip64=True) as z:
        for p,rel in files:
            z.write(p,rel);checks[rel]={'sha256':sha(p),'bytes':p.stat().st_size}
        import json
        z.writestr('archive_manifest.json',json.dumps({'full_weights_included':full,'files':checks},indent=2))
    os.replace(tmp,target);print('Saved:',target,flush=True);return target

def validate_data(data,jobs):
    for name in sorted(set(j['dataset'] for j in jobs)):
        folder=data/name;m=read_json(folder/'manifest.json')
        if name=='controlled':
            sources={s:{r['source_id'] for r in m['records'] if r['split']==s} for s in ['train','val','test']}
            if any(sources[a]&sources[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Controlled source leakage')
            hashes=read_json(folder/'file_hashes.json')
            for rel,h in hashes.items():
                if sha(folder/rel)!=h:raise ValueError('Controlled file changed: '+rel)
        elif sha(folder/'inputs.npz')!=m['inputs_sha256']:raise ValueError('Cache hash mismatch: '+name)
        if any(j['dataset']==name and j['variant'].endswith('geo') for j in jobs):
            if len(m['geodesic'])!=len(m['records']):raise ValueError('Geodesic cache missing. Run --stage prepare --data-root ../Data first.')
            for r in m['geodesic']:
                if sha(folder/r['file'])!=r['cache_sha256']:raise ValueError('Geodesic hash mismatch: '+r['id'])

def main():
    global CHILD
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','priority','core','fraction','tables','neighbors','threshold','prepare','smoke','pack','plan'],default='all')
    p.add_argument('--data-root');p.add_argument('--runs-root');p.add_argument('--data',default=str(ROOT/'data'))
    p.add_argument('--out',default=str(ROOT/'results'));p.add_argument('--batch-size',type=int,default=32);p.add_argument('--epochs',type=int,default=250)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4);p.add_argument('--keep-every',type=int,default=50)
    p.add_argument('--full-backup',action='store_true');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args();out=Path(a.out).resolve();data=Path(a.data).resolve();out.mkdir(parents=True,exist_ok=True)
    # OS lock is released even after SIGKILL; no stale PID guesses or duplicate training.
    lock=open(out/'runner.lock','a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another runner already owns this results folder.')
    plan=make_plan(a.batch_size,a.epochs)
    planpath=out/'plan.json'
    if planpath.exists() and read_json(planpath)!=plan:raise ValueError('Plan changed: choose another --out folder')
    write_json(planpath,plan)
    print('Plan:',plan['counts'],'total',plan['total'],flush=True)
    if a.stage=='plan':return
    if a.stage=='pack':package(out,plan,a.full_backup);return
    if a.stage=='prepare':
        if not a.data_root:raise ValueError('--data-root is required')
        from prepare import prepare,complete_tau80,audit_runs,templates
        prepare(a.data_root,data,geo=True)
        complete_tau80(a.data_root,data)
        if a.runs_root:audit_runs(a.runs_root,ROOT/'audit/historical_runs.json')
        templates(data,ROOT/'author_information');return
    if (out/'STOP').exists():raise ValueError('STOP file exists. Remove it to resume.')
    jobs=plan['jobs'] if a.stage in ['all','smoke'] else [j for j in plan['jobs'] if j['stage'] in (['core','fraction'] if a.stage=='priority' else [a.stage])]
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    try:
        # Self-contained delivery includes all caches. Data-root can prepare missing geodesic inputs.
        if a.data_root and any(j['variant'].endswith('geo') for j in jobs) and (not read_json(data/'surface/manifest.json')['geodesic'] or not (data/'surface/geo').exists()):
            from prepare import import_geo
            import_geo(a.data_root,data)
        validate_data(data,jobs)
        write_json(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'command':sys.argv,
                    'programs':{q.name:sha(q) for q in sorted(ROOT.glob('*.py'))}})
        from smoke import smoke
        smokeplan=dict(plan,jobs=jobs)
        signature=fingerprint({'plan':smokeplan,'programs':{q.name:sha(q) for q in sorted(ROOT.glob('*.py'))},'device':a.device})
        smokeout=out/'smoke.json'
        if not smokeout.exists() or read_json(smokeout).get('signature')!=signature:
            report=smoke(smokeplan,data,smokeout,a.device,a.threads);report['signature']=signature;write_json(smokeout,report)
        if a.stage=='smoke':return
        for index,job in enumerate(jobs,1):
            if STOP or (out/'STOP').exists():break
            folder=out/job['id'];folder.mkdir(exist_ok=True);write_json(folder/'job.json',job)
            if (folder/'result.json').exists():
                cfg=read_json(folder/'config.json');r=read_json(folder/'result.json')
                if cfg['job']!=job or r['job']!=job:raise ValueError('Completed job configuration differs')
                if cfg['data_manifest']!=sha(data/job['dataset']/'manifest.json'):raise ValueError('Completed job data changed')
                if any(sha(ROOT/n)!=h for n,h in cfg['programs'].items()):raise ValueError('Completed job code changed')
                if sha(folder/'selected.pt')!=r['selected_checkpoint_sha256']:raise ValueError('Selected checkpoint missing or changed')
                continue
            if (folder/'failure.json').exists() and not a.retry_failed:raise ValueError('A job failed previously. Inspect failure, then use --retry-failed to retry with unchanged settings.')
            print('JOB %d/%d: %s'%(index,len(jobs),job['id']),flush=True)
            write_json(out/'progress.json',{'job':job['id'],'index':index,'stage_total':len(jobs),'state':'running'})
            cmd=[sys.executable,'-u',str(ROOT/'train.py'),'--job',str(folder/'job.json'),'--data',str(data),
                 '--out',str(folder),'--device',a.device,'--threads',str(a.threads),'--keep-every',str(a.keep_every)]
            # Child output appended, never overwritten on resume.
            with open(folder/'console.log','a',buffering=1) as log:
                CHILD=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in CHILD.stdout:
                    log.write(line);print(line,end='',flush=True)
                code=CHILD.wait();CHILD=None
            if code==75:break
            if code!=0:
                write_json(folder/'failure.json',{'returncode':code,'job':job,'log':'console.log'})
                raise RuntimeError('Training stopped: '+job['id']+'; see console.log. Batch size is not changed automatically.')
            package(out,plan)
        done=summarize(out,plan)
        write_json(out/'progress.json',{'state':'complete' if done==plan['total'] else 'partial_or_paused','completed':done,'planned':plan['total']})
    except BaseException as e:
        write_json(out/'last_error.json',{'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()})
        raise
    finally:package(out,plan,a.full_backup)

if __name__=='__main__':main()
