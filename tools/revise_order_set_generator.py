"""Cross-fit both 2017 control halves and validate permutation field transfer.

The observed estimator, masks, covariance, bootstrap and original experiment
are unchanged. This isolated audit changes the noise generator only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import logging
from pathlib import Path
import time
import numpy as np
from scipy.stats import norm
import run_order_set_pilot as R
from glacier_offsets.velocity_basis import harmonic,utc_days
from glacier_offsets.velocity_inspect import M2_PERIOD
from glacier_offsets.velocity_quality import quiet_median,moving_indices
from glacier_offsets.velocity_systematics import (OrderRegionalDesign,order_screen,
    robust_background,calendar_times,holm)
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_noise import crossfit_control_bank,seasonal_time_indices,transfer_control_field
from glacier_offsets.velocity_io import digest,write_json

MODELS=['primary_azimuth','no_order','azimuth_along','residual_vertical']
CASES=[('null',0.,0.,0.),('benchmark',.1,0.,0.),('null_vertical20',0.,.2,0.),
       ('benchmark_vertical20',.1,.2,0.),('benchmark_M2_003',.1,.2,.03),
       ('benchmark_M2_020',.1,.2,.2),('null_M2_020',0.,.2,.2)]
SCENARIOS=['crossfit_permutation','crossfit_replacement','H2_permutation','H2_replacement']


def rng_for(seed,component,draw,region=0):
    return np.random.default_rng(np.random.SeedSequence([seed,component,draw,region]))


def initialize(args):
    root=args.root;old=root/'2017-order-set';out=root/'2017-order-set-generator';out.mkdir(exist_ok=True)
    source_files=['src/glacier_offsets/velocity_noise.py','src/glacier_offsets/velocity_systematics.py',
        'src/glacier_offsets/velocity_quality.py','src/glacier_offsets/velocity_vertical.py',
        'tools/run_order_set_pilot.py','tools/revise_order_set_generator.py']
    recipe={'status':'exploratory generator audit; no 2018 data',
        'original_experiment':str(old.resolve()),'original_observed_sha256':digest(old/'observed.json'),
        'corrections_sha256':digest(old/'corrections.json'),'source_cache_sha256':digest(old/'corrected-inputs.npz'),
        'covariance_sha256':digest(old/'clean_noise.json'),'original_recipe_sha256':digest(old/'recipe.json'),
        'generator':'opposite-half acquisition-purged secular/order prediction; source centering within held-out half; defined source training baseline, >=8 epochs for shared order/harmonic training; harmonics retained',
        'space':'random donor-block permutations and donor-source permutations; exhaust pool before recycling',
        'time':'common circular 5-pair moving blocks within matching H1/H2; donor crossing July 1 excluded; destination crossing pair uses H2',
        'primary':'crossfit_permutation; direct satellite-azimuth Msf; unchanged observed paired time x space uncertainty',
        'ablation_nulls':SCENARIOS,'models':MODELS,'cases':CASES,'seed':args.seed,
        'draws_per_case':args.simulations,'nested_null_per_region':args.nested_null,
        'nested_benchmark_per_region':args.nested_benchmark,'nested_resamples':args.nested_bootstrap,
        'code_sha256':{p:digest(p) for p in source_files}}
    target=out/'recipe.json'
    if target.exists() and json.loads(target.read_text())!=json.loads(json.dumps(recipe)):
        raise ValueError('Recorded generator recipe or source changed; use a new experiment directory')
    if not target.exists():write_json(target,recipe)
    fields=R.load(root,old);cov=R.calibrate(fields,old);state=R.build_state(fields,cov)
    for row in json.loads((old/'observed.json').read_text())['rows']:state[row['region']][row['variant']]=row
    # Preserve original artifacts byte-for-byte; comparison records reference them.
    baseline={p.name:digest(p) for p in old.iterdir() if p.is_file()}
    path=out/'original-artifacts.json'
    if path.exists() and json.loads(path.read_text())!=baseline:raise ValueError('Original experiment changed')
    if not path.exists():write_json(path,baseline)
    return old,out,fields,cov,state,recipe


def banks_for(fields,cov,state,out):
    crossfit=[];legacy=[];records=[];arrays={}
    for f in fields:
        if f['role']!='control':continue
        bank,folds=crossfit_control_bank(f,cov);crossfit.append(bank)
        records.append({'region':f['id'],'folds':folds,'valid_pairs':int(bank['valid'].sum()),
            'valid_sources_by_half':[r['supported_training_sources'] for r in folds]})
        arrays.update({f['id']+'__'+k:bank[k] for k in ('residual','valid','peak','snr')})
        # Reproduce the old bank exactly as a retained factorial stress case.
        half=f['bank_half'];first=calendar_times(f)[1]<utc_days('2017-07-01')
        fit=state[f['id']]['designs']['primary_azimuth'].fit(order_screen(f,allowed=first)&first[None],details=True)
        pred=fit['prediction'][...,0]-np.einsum('stj,j->st',np.nan_to_num(harmonic(f['early'],f['late'])),fit['cs'])
        residual=f['bank_residual'].copy();residual[...,1]=f['data'][...,1]-pred
        center=quiet_median(np.where(f['valid'][:,half,None],residual[:,half],np.nan),axis=1)
        legacy.append(({**f,'residual':residual-center[:,None]},half))
    if not (out/'banks.npz').exists():
        np.savez_compressed(out/'banks.npz',**arrays);write_json(out/'bank-folds.json',{'rows':records})
    else:
        with np.load(out/'banks.npz') as saved:
            for key,value in arrays.items():
                if not np.array_equal(value,saved[key],equal_nan=True):raise ValueError('Crossfit bank changed')
    return crossfit,legacy


def prepare_targets(fields,state):
    targets=[]
    for f in fields:
        nt=f['valid'].shape[1];u=state[f['id']]['axis'];prediction,bias=robust_background(f,np.ones(nt,bool))
        prediction[...,1]-=f['order']*bias[:,None]
        d=state[f['id']]['designs']['primary_azimuth'];fit=state[f['id']]['fits']['primary_azimuth']
        mapping={burst:b for burst,b in zip(d.bursts,fit['order_m'])}
        prediction[...,1]+=f['order']*np.array([mapping[x] for x in f['burst']])[:,None]
        targets.append({'secular_order':prediction,'h':np.nan_to_num(harmonic(f['early'],f['late'])),
            'm2':np.nan_to_num(harmonic(f['early'],f['late'],M2_PERIOD)),
            'vertical':vertical_columns(f['early'],f['late']),'projection':f['geometry'][...,:2]@u})
    return targets


def synthetic_fields(fields,targets,banks,legacy,scenario,case,draw,seed,total_draws):
    _,amplitude,vfraction,m2=case
    j=int(rng_for(seed,10,draw).integers(len(banks)));clock=rng_for(seed,11,draw)
    if scenario.startswith('crossfit'):
        donor=banks[j];ti=seasonal_time_indices(clock,*calendar_times(fields[0]))
    else:
        donor,half=legacy[j];nt=fields[0]['valid'].shape[1]
        ti=np.resize(np.r_[half[moving_indices(clock,len(half),5)],half[moving_indices(clock,len(half),5)]],nt)
    phase_rng=rng_for(seed,12,draw);phi=2*np.pi*(draw+phase_rng.random())/total_draws
    phasor=np.array([np.cos(phi),np.sin(phi)]);vp=phase_rng.uniform(0,2*np.pi,2);mp=phase_rng.uniform(0,2*np.pi)
    for fi,(field,target) in enumerate(zip(fields,targets)):
        base,diag=transfer_control_field(rng_for(seed,13,draw,fi),donor,field,ti,replacement=scenario.endswith('replacement'))
        msf=np.einsum('stj,j->st',target['h'],amplitude*phasor)[...,None]*target['projection'][:,None]
        z=np.r_[.6*vfraction*np.array([np.cos(vp[0]),np.sin(vp[0])]),.2*vfraction*np.array([np.cos(vp[1]),np.sin(vp[1])])]
        vertical=np.einsum('stj,j->st',target['vertical'],z)[...,None]*field['geometry'][:,None,:,2]
        m2signal=np.einsum('stj,j->st',target['m2'],m2*np.array([np.cos(mp),np.sin(mp)]))[...,None]*target['projection'][:,None]
        yield fi,{**base,'data':base['data']+target['secular_order']+msf+vertical+m2signal},msf,diag,j


def run_case(fields,targets,banks,legacy,state,out,scenario,case,args):
    name=case[0];path=out/(scenario+'--'+name+'.npz')
    if path.exists():logging.info('Reusing completed %s %s',scenario,name);return
    models=MODELS if scenario=='crossfit_permutation' else ['primary_azimuth']
    results={f['id']+'__'+m:np.full((args.simulations,9),np.nan) for f in fields for m in models}
    retention=np.full((args.simulations,len(fields),2),np.nan);diagnostics=np.zeros((args.simulations,len(fields),5),int)
    split=float(utc_days('2017-07-01'));e,l=calendar_times(fields[0]);halves=[(e+l)/2<split,(e+l)/2>=split]
    for b in range(args.simulations):
        for fi,synthetic,msf,diag,donor_index in synthetic_fields(fields,targets,banks,legacy,scenario,case,b,args.seed,args.simulations):
            f=fields[fi];good=order_screen(synthetic);u=R.axis(synthetic,good) if f['role']=='target' else state[f['id']]['axis']
            for m in models:
                d=state[f['id']]['designs'][m]
                try:
                    cs=d.fit(good,u,data=synthetic['data'])['cs'];truth=d.fit(good,u,data=msf)['cs']
                    delta=cs-truth;unc=state[f['id']][m];inv=np.linalg.inv(unc['covariance'])
                    ae=abs(np.linalg.norm(cs)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-15)
                    pe=abs(np.degrees(np.angle(np.exp(1j*(np.arctan2(cs[1],cs[0])-np.arctan2(truth[1],truth[0]))))))
                    results[f['id']+'__'+m][b]=[*cs,*truth,ae,pe,float(delta@inv@delta<=unc['cutoff95']),
                        float(cs@inv@cs>unc['cutoff95']),good.sum()/max(f['valid'].sum(),1)]
                except (ValueError,np.linalg.LinAlgError):pass
            retention[b,fi]=[good[:,h].sum()/max(f['valid'][:,h].sum(),1) for h in halves]
            diagnostics[b,fi]=[donor_index,diag['unique_donor_sources'],diag['source_assignments'],diag['unique_donor_blocks'],diag['block_assignments']]
        if (b+1)%50==0:logging.info('%s %s %d/%d',scenario,name,b+1,args.simulations)
    np.savez_compressed(path,**results,retention_by_half=retention,donor_diagnostics=diagnostics)


def nested_case(fields,targets,banks,legacy,cov,state,out,case,args):
    name=case[0];count=args.nested_null if name=='null' else args.nested_benchmark
    indices=list(range(count)) if name=='null' else np.floor(np.linspace(0,args.simulations,count,endpoint=False)).astype(int).tolist()
    destination=out/('nested-'+name);destination.mkdir(exist_ok=True);jobs=[]
    keys=('id','role','data','valid','peak','early','late','geometry','order','burst','group','calendar_early','calendar_late')
    for b in indices:
        for fi,synthetic,msf,_,_ in synthetic_fields(fields,targets,banks,legacy,'crossfit_permutation',case,b,args.seed,args.simulations):
            f=fields[fi];file=destination/(f['id']+'--'+str(b)+'.json')
            if file.exists():continue
            compact={k:synthetic[k] for k in keys}
            unc=state[f['id']]['primary_azimuth']
            jobs.append((compact,cov,msf,args.nested_bootstrap,args.seed,b,fi,case[1],unc['covariance'],unc['cutoff95'],name))
    logging.info('Nested %s: %d remaining datasets (%d per region)',name,len(jobs),count)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(R.nested_job,job):job for job in jobs}
        for n,future in enumerate(as_completed(pending),1):
            record,values=future.result();stem=record['region']+'--'+str(record['draw'])
            np.savez_compressed(destination/(stem+'.npz'),cs=values)
            write_json(destination/(stem+'.json'),record)
            if n%25==0:logging.info('Nested %s finished %d/%d',name,n,len(jobs))
    records=[json.loads((destination/(f['id']+'--'+str(b)+'.json')).read_text()) for b in indices for f in fields]
    write_json(out/('nested-'+name+'.json'),{'rows':records,'datasets_per_region':count,'resamples_per_dataset':args.nested_bootstrap,
        'scheme':'unchanged paired 30-day calendar x 3-km spatial bootstrap, frozen screen algorithm rerun',
        'indices':indices,'MC_limit':'per-region binomial uncertainty; regions share field draws'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--stage',choices=['prepare','recovery','nested','all'],default='all')
    p.add_argument('--simulations',type=int,default=200);p.add_argument('--nested-null',type=int,default=100)
    p.add_argument('--nested-benchmark',type=int,default=20);p.add_argument('--nested-bootstrap',type=int,default=199)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=12345)
    args=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');start=time.time()
    if args.nested_null>args.simulations:raise ValueError('Nested subset exceeds independent synthetic fields')
    old,out,fields,cov,state,recipe=initialize(args);banks,legacy=banks_for(fields,cov,state,out);targets=prepare_targets(fields,state)
    if args.stage in ('recovery','all'):
        for case in CASES:run_case(fields,targets,banks,legacy,state,out,'crossfit_permutation',case,args)
        for scenario in SCENARIOS[1:]:run_case(fields,targets,banks,legacy,state,out,scenario,CASES[0],args)
    if args.stage in ('nested','all'):
        for case in CASES[:2]:nested_case(fields,targets,banks,legacy,cov,state,out,case,args)
    original=json.loads((out/'original-artifacts.json').read_text())
    assert all(digest(old/name)==sha for name,sha in original.items()),'Original experiment changed'
    logging.info('Completed %s in %.1f seconds',args.stage,time.time()-start)

if __name__=='__main__':main()
