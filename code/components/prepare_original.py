"""Preserve historical cloud-category labels; regenerate a common candidate rule.

The saved outer test partition is unchanged. Validation is split from outer
training, and normal-reference calibration excludes both validation and test.
No independent point-level ground truth is asserted or generated here.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from controlled_data import sha256, local_geometry


def run(root, out, seed=20260914, k0=20, alpha=1.282):
    root,out=Path(root).resolve(),Path(out).resolve()
    if out.exists() or out==root or root in out.parents:
        raise ValueError('Use a new output directory outside the source')
    splits={}
    outer={s:[str(Path(*Path(p).parts[-2:])) for p in json.loads((root/f'{s}_files.txt').read_text())]
           for s in ['train','test']}
    if len(set(outer['train']))!=len(outer['train']) or len(set(outer['test']))!=len(outer['test']) or set(outer['train'])&set(outer['test']):
        raise ValueError('Duplicate or overlapping outer split entries')
    classes=sorted({Path(p).parent.name.lower() for v in outer.values() for p in v})
    splits['train'],splits['val'],splits['test']=[],[],sorted(outer['test'])
    # A separate deterministic permutation per class makes the normal split
    # identical to the controlled study for the same seed and source dataset.
    for category in classes:
        entries=np.array(sorted(p for p in outer['train'] if Path(p).parent.name.lower()==category))
        order=np.random.default_rng(seed).permutation(len(entries))
        nv=max(1,round(.2*len(entries)))
        splits['val'].extend(entries[order[:nv]].tolist())
        splits['train'].extend(entries[order[nv:]].tolist())
    arrays={p:np.loadtxt(root/p,delimiter=',',usecols=(0,1,2)) for v in splits.values() for p in v}
    hashes={p:sha256(root/p) for p in arrays}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError('Exact duplicate source clouds require grouped splitting')
    variation={p:local_geometry(x,k0)[0] for p,x in arrays.items()}
    reference=[p for p in splits['train'] if Path(p).parent.name.lower()=='normal']
    values=np.concatenate([variation[p] for p in reference])
    mu,sd=float(values.mean()),float(values.std(ddof=0));threshold=mu+alpha*sd
    out.mkdir(parents=True)
    records=[]
    for split,paths in splits.items():
        for p in paths:
            category=Path(p).parent.name.lower()
            sample_id=category+'_'+Path(p).stem
            rel=f'clouds/{split}/{sample_id}.npz';dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            mask=variation[p]>threshold
            np.savez_compressed(dest,xyz=arrays[p],candidate_mask=mask,surface_variation=variation[p],
                                cloud_label=np.array(classes.index(category)))
            records.append(dict(id=sample_id,source_id=sample_id,source_file=p,file=rel,split=split,
                                category=category,rho=None,n=len(mask),candidate_count=int(mask.sum()),
                                candidate_fraction=float(mask.mean())))
    manifest=dict(study_type='original_cloud_classification',class_names=classes,fractions=[],
                  source_root=str(root),source_splits=splits,source_sha256=hashes,seed=seed,
                  original_split_sha256={s:sha256(root/f'{s}_files.txt') for s in outer},
                  candidate_definition='algorithmic mask, not independent ground truth',
                  independent_point_truth_available=False,
                  normal_reference_clouds=reference,reference_point_count=len(values),
                  k0=k0,metric='euclidean',alpha=alpha,normal_mean=mu,normal_std_population=sd,threshold=threshold,
                  limitations=['Source cloud identity is not verified physical specimen identity.',
                               'Only coordinates are used by the revision training entry.',
                               'This protocol changes calibration and checkpoint selection; historical results cannot be reused.'],
                  records=records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(out=str(out),splits={s:len(v) for s,v in splits.items()},reference_clouds=len(reference),threshold=threshold)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.source,a.out)
