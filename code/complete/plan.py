"""Frozen, deduplicated experimental grid. No outcomes influence the plan."""
from collections import Counter
from common import fingerprint

MODELS=['pointnet','pointnet2','dgcnn','mvgcn','hard','soft']
SEEDS=[24,42,2026]

def make_plan(batch_size=32,epochs=250):
    jobs=[];seen=set()
    def add(stage,dataset,model,variant='xyz',k=15,tau='base',rho=None):
        for seed in SEEDS:
            cfg=dict(dataset=dataset,model=model,variant=variant,k=k,tau=tau,rho=rho,
                     seed=seed,batch_size=batch_size,epochs=epochs,lr=.1)
            key=fingerprint(cfg)
            if key in seen:continue
            seen.add(key)
            name='{}_{}_{}_k{}_{}_r{}_s{}_b{}_e{}'.format(dataset,model,variant,k,tau,
                    'NA' if rho is None else str(rho).replace('.','p'),seed,batch_size,epochs)
            jobs.append(dict(cfg,id=name,stage=stage))
    for ds in ['surface','real']:
        for model in MODELS:add('core',ds,model)
    for rho in [.005,.01,.02,.05,.10]:
        for model in ['dsrf','hard','soft']:add('fraction','controlled',model,rho=rho)
    for ds in ['surface','real']:
        for variant in ['norm','cov','geo','norm_geo']:
            if ds=='real' and variant=='cov':continue
            for model in MODELS:
                if model=='pointnet' and variant.endswith('geo'):continue
                add('tables',ds,model,variant)
    for ds in ['surface','real']:
        for k in [5,10,13,20,25]:
            for model in ['dgcnn','mvgcn','hard','soft']:add('neighbors',ds,model,'norm',k=k)
        for tau in ['pct60','pct70','pct80','pct90']:
            for model in ['hard','soft']:add('threshold',ds,model,'norm',tau=tau)
    return {'version':1,'jobs':jobs,'counts':dict(Counter(j['stage'] for j in jobs)),
        'total':len(jobs),'seeds':SEEDS,'batch_size':batch_size,'epochs':epochs,
        'selection':'maximum validation OA; tie: minimum validation ordinary cross-entropy; earliest exact tie',
        'duplicates_omitted':['k=15 normals reuses tables jobs; original base mask is NOT percentile tau80',
            'PointNet++ normals ignores args.k in supplied code (ball query nsample 32/64); reuse its fixed baseline, do not fabricate k-dependent runs'],
        'fraction_design':'3 models x 5 known support fractions x 3 seeds = 45; DSRF-only uses same Graphcnn backbone',
        'note':'All runs are new validation-selected estimates. Existing 72 component runs are not rerun or overwritten.'}
