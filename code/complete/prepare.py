"""Read-only audit of author data, immutable compact caches and provenance templates."""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from common import ROOT,sha,write_json,read_json,fingerprint

BASE={'surface':'Surface_Defects_pcd_extend_2000_estnorm_noise0001','real':'real_data_new_2000'}
GEO={'surface':'Surface_Defects_pcd_extend_2000_geod_estnorm_noise0001','real':'real_data_new_2000_geod'}

def normalize(raw):
    x=np.array(raw,dtype=np.float32,copy=True)
    x-=x.mean(axis=0,keepdims=True)
    radius=np.sqrt(np.sum(x*x,axis=1)).max()
    if not np.isfinite(radius) or radius<=0:raise ValueError('Invalid radius')
    return x/radius

def prepare(data_root,out,geo=False):
    data_root=Path(data_root).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    report={'data_root':str(data_root),'datasets':{},'physical_group_identity':'unknown, not inferred from filenames'}
    for ds,dirname in BASE.items():
        root=data_root/dirname;dest=out/ds;dest.mkdir(exist_ok=True)
        cat=dict(line.split() for line in (root/'pointset2category.txt').read_text().splitlines() if line.strip())
        classes=sorted(cat);lookup={d:classes.index(c) for c,d in cat.items()}
        original={s:[str(Path(*Path(p).parts[-2:])) for p in read_json(root/(s+'_files.txt'))] for s in ['train','test']}
        if set(original['train'])&set(original['test']):raise ValueError('Original split overlap')
        if sum(len(v) for v in original.values())!=len(set(sum(original.values(),[]))):raise ValueError('Duplicate IDs')
        # Reuse prior fixed inner validation IDs when shipped; otherwise fixed stratification.
        fixed=ROOT/'fixed_splits'/ (ds+'.json')
        if fixed.exists():
            splitmap=read_json(fixed)
            if set(splitmap)!=set(original['train']+original['test']):raise ValueError('Fixed split IDs differ')
            if {k for k,v in splitmap.items() if v=='test'}!=set(original['test']):raise ValueError('Outer test differs')
        else:
            splitmap={r:'train' for r in original['train']};splitmap.update({r:'test' for r in original['test']})
            rng=np.random.RandomState(20260914)
            for d in cat.values():
                rows=sorted(r for r in original['train'] if Path(r).parent.name==d)
                for i in rng.permutation(len(rows))[:max(1,round(.2*len(rows)))]:splitmap[rows[int(i)]]='val'
        rows=[];xyz=[];norm=[];geo_xyz=[];geo_norm=[];ys=[];masks={t:[] for t in ['base','tau60','tau70','tau90','geod']};georecords=[]
        for rel in sorted(splitmap):
            p=root/rel;raw=np.loadtxt(p,delimiter=',').astype(np.float32)
            if raw.shape!=(2000,9) or not np.isfinite(raw).all():raise ValueError('Invalid cloud '+str(p))
            x=raw[:,:3];xyz.append(normalize(x));norm.append(raw[:,6:9]);ys.append(lookup[Path(rel).parent.name])
            rec={'id':rel,'split':splitmap[rel],'category':classes[ys[-1]],'label':ys[-1],
                 'source_cloud_id':rel,'physical_source_id':None,'source_sha256':sha(p),
                 'xyz_sha256':fingerprint(x.tolist()),'masks':{}}
            for tag in masks:
                r=(data_root/GEO[ds]) if tag=='geod' else (root if tag=='base' else data_root/(dirname+'_'+tag))
                txt=r/rel
                if tag!='base':
                    other=np.loadtxt(txt,delimiter=',').astype(np.float32)
                    if tag=='geod':
                        if other.shape!=raw.shape or not np.isfinite(other).all():raise ValueError('Invalid geodesic cloud')
                        geo_xyz.append(normalize(other[:,:3]));geo_norm.append(other[:,6:9])
                        rec['geodesic_xyz_max_delta']=float(np.max(np.abs(other[:,:3]-raw[:,:3])))
                        rec['geodesic_source_sha256']=sha(txt)
                    elif not np.array_equal(other,raw):raise ValueError('Threshold variant point/features differ: '+str(txt))
                mf=txt.with_suffix('.npz')
                with np.load(mf,allow_pickle=False) as z:mark=np.array(z['data'],copy=True)
                if mark.shape!=(2000,) or not np.isin(mark,[0,1]).all():raise ValueError('Invalid candidate mask '+str(mf))
                masks[tag].append(mark.astype(bool));rec['masks'][tag]={'sha256':sha(mf),'candidate_fraction':float(mark.mean())}
            if geo:
                gp=(data_root/GEO[ds]/rel).with_name(Path(rel).stem+'_geodesic_dis.npz')
                gd=dest/'geo'/ (Path(rel).parent.name+'_'+Path(rel).stem+'.npy');gd.parent.mkdir(exist_ok=True)
                sourcehash=sha(gp)
                with np.load(gp,allow_pickle=False) as z:d=np.array(z['data'],dtype=np.float32)
                if d.shape!=(2000,2000) or not np.isfinite(d).all():raise ValueError('Invalid geodesic matrix '+str(gp))
                nonpositive=int((d<=0).sum());maxd=float(d.max())
                if maxd<=0:raise ValueError('No positive geodesic distance')
                # Match supplied loader: nonpositive (including diagonal) -> max; input negative distance.
                d=np.where(d<=0,maxd,d);d=-d
                np.save(gd,d)
                georecords.append({'id':rel,'file':str(gd.relative_to(dest)),'source_sha256':sourcehash,
                                   'cache_sha256':sha(gd),'nonpositive_replaced':nonpositive})
            rows.append(rec)
        if len(set(r['xyz_sha256'] for r in rows))!=len(rows):raise ValueError('Duplicate XYZ cloud content requires grouping')
        target=dest/'inputs.npz';temp=dest/'inputs.tmp.npz'
        np.savez_compressed(temp,xyz=np.stack(xyz),norm=np.stack(norm),geo_xyz=np.stack(geo_xyz),geo_norm=np.stack(geo_norm),labels=np.array(ys,dtype=np.int64),
                            **{'mask_'+k:np.stack(v) for k,v in masks.items()})
        temp.replace(target)
        manifest={'dataset':ds,'class_names':classes,'records':rows,'inputs_sha256':sha(target),
            'counts':{s:sum(r['split']==s for r in rows) for s in ['train','val','test']},
            'original_splits_sha256':{s:sha(root/(s+'_files.txt')) for s in ['train','test']},
            'normalization':'float32 XYZ centered and divided by maximum radius; original normals retained',
            'candidate_provenance':'Saved algorithm candidates; not assumed independent point truth',
            'tau_labels':'Directory names preserved as supplied; numeric percentile interpretation not verified',
            'geodesic':georecords,'geodesic_policy':'float32 negative source distance; nonpositive replaced by matrix maximum as original loader; no in-place graph masking',
            'source_groups':'file-level split, physical source grouping unknown',
            'metric_comparison_note':'Geodesic inputs use their own supplied XYZ/normals and masks. Coordinates differ slightly from Euclidean version; do not claim graph metric is the only changing quantity.'}
        write_json(dest/'manifest.json',manifest)
        report['datasets'][ds]={'clouds':len(rows),'splits':manifest['counts'],'classes':classes,
            'candidate_fraction_summary':{tag:{c:float(np.mean([r['masks'][tag]['candidate_fraction'] for r in rows if r['category']==c])) for c in classes} for tag in masks},
            'geodesic_matrices':len(georecords)}
        print(ds,manifest['counts'],'geodesic',len(georecords),flush=True)
    write_json(out/'data_audit.json',report)
    return report

def audit_runs(runs_root,out):
    root=Path(runs_root);rows=[]
    if not root.exists():return []
    for p in sorted(root.rglob('log.txt')):
        text=p.read_text(errors='replace');lines=text.splitlines();header=lines[0] if lines else ''
        tests=re.findall(r'Test\s+(\d+),\s*loss:\s*([\d.]+),\s*test acc:\s*([\d.]+),\s*test avg acc:\s*([\d.]+)',text)
        sf=re.search(r'Self:\s*(True|False)',header)
        rec={'run':str(p.parent.relative_to(root)),'log_header':header,'log_sha256':sha(p),
             'operator_from_Self':None if sf is None else ('soft' if sf[1]=='True' else 'hard'),
             'test_epochs_logged':len(tests),'peak_test_oa':max((float(t[2]) for t in tests),default=None),
             'checkpoint_files':[{'name':q.name,'bytes':q.stat().st_size,'sha256':sha(q)} for q in sorted(p.parent.iterdir()) if q.suffix in ['.t7','.pth','.pt']],
             'formal_selection':'Historical logs are not evidence of validation selection; no automatic replacement for prospective jobs'}
        rows.append(rec)
    write_json(out,{'runs':rows,'count':len(rows),'all_checkpoint_count':sum(1 for _ in root.rglob('*.t7'))})
    return rows

def templates(cache,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=out/'source_and_label_provenance.csv'
    if p.exists():return # never overwrite author-entered information
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        fields=['dataset','cloud_id','category','physical_scan_or_specimen_id','relationship_to_other_clouds',
                'point_labels_origin','reference_label_file','annotation_method','annotator','reviewer','notes']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for ds in BASE:
            for r in read_json(Path(cache)/ds/'manifest.json')['records']:
                writer.writerow({'dataset':ds,'cloud_id':r['id'],'category':r['category'],
                                 'point_labels_origin':'unknown; saved candidate masks are separate'})

def complete_tau80(data_root,cache):
    from sklearn.neighbors import NearestNeighbors
    report=[]
    for ds,dirname in BASE.items():
        dest=Path(cache)/ds;m=read_json(dest/'manifest.json')
        with np.load(dest/'inputs.npz',allow_pickle=False) as z:arrays={k:np.array(z[k],copy=True) for k in z.files}
        masks={t:[] for t in [60,70,80,90]};checks=[]
        for i,r in enumerate(m['records']):
            x=np.loadtxt(Path(data_root)/dirname/r['id'],delimiter=',')[:,:3]
            idx=NearestNeighbors(n_neighbors=30,algorithm='kd_tree').fit(x).kneighbors(x,return_distance=False)
            a=x[idx];a-=a.mean(axis=1,keepdims=True);ev=np.linalg.eigh(a.transpose(0,2,1)@a/30)[0]
            v=ev[:,0]/(ev.sum(1)+1e-8)
            differences={str(t):int(np.sum(arrays['mask_tau'+str(t)][i]!=(v>=np.percentile(v,t)))) for t in [60,70,90]}
            for t in masks:
                mask=v>=np.percentile(v,t);masks[t].append(mask)
                r['masks']['pct'+str(t)]={'candidate_fraction':float(mask.mean()),'origin':'new percentile mask using supplied k=30 minimum-eigenvalue formula'}
            checks.append({'id':r['id'],'existing_mask_differences':differences,'new_pct80_candidate_fraction':float(masks[80][-1].mean())})
        for t in masks:arrays['mask_pct'+str(t)]=np.stack(masks[t])
        temp=dest/'inputs.tmp.npz';np.savez_compressed(temp,**arrays);temp.replace(dest/'inputs.npz')
        m['inputs_sha256']=sha(dest/'inputs.npz')
        m['tau_labels']='Original tau60/70/90 retained. Prospective pct60/70/80/90 all newly generated consistently: k0=30 including center, minimum eigenvalue / (sum +1e-8), per-cloud percentile, from supplied generate_tau_data.py formula. Differences against saved masks recorded in threshold_audit.json. Base is a separate saved mask, not pct80.'
        write_json(dest/'manifest.json',m);report.append({'dataset':ds,'checked_clouds':len(checks),'checks':checks})
    write_json(Path(cache)/'threshold_audit.json',report)

def import_geo(data_root,cache):
    """Populate missing matrices only, retaining frozen compact inputs and manifests."""
    for ds,dirname in GEO.items():
        folder=Path(cache)/ds;m=read_json(folder/'manifest.json');records={r['id']:r for r in m['records']}
        for r in m['geodesic']:
            target=folder/r['file']
            if target.exists() and sha(target)==r['cache_sha256']:continue
            txt=Path(data_root)/dirname/r['id'];p=txt.with_name(txt.stem+'_geodesic_dis.npz')
            if sha(txt)!=records[r['id']]['geodesic_source_sha256'] or sha(p)!=r['source_sha256']:
                raise ValueError('Different source version for geodesic input: '+str(p))
            with np.load(p,allow_pickle=False) as z:d=np.array(z['data'],dtype=np.float32)
            if d.shape!=(2000,2000) or not np.isfinite(d).all():raise ValueError('Invalid distance array')
            d=-np.where(d<=0,float(d.max()),d)
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.stem+'.tmp.npy');np.save(tmp,d)
            if sha(tmp)!=r['cache_sha256']:raise ValueError('Geodesic conversion hash differs: '+str(p))
            tmp.replace(target)
        print('Geodesic matrices ready:',ds,len(m['geodesic']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--runs-root')
    p.add_argument('--out',default=str(ROOT/'data'));p.add_argument('--geo',action='store_true');a=p.parse_args()
    prepare(a.data_root,a.out,a.geo)
    complete_tau80(a.data_root,a.out)
    if a.runs_root:audit_runs(a.runs_root,ROOT/'audit/historical_runs.json')
    templates(a.out,ROOT/'author_information')
