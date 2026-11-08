"""Matched Evans baseline/order/solid-Earth/both pilot on existing observations.

Prepare with prepare_evans_nuisance_pilot.py first. Runs in ice, PYTHONPATH=src.
The original 12 points and full four target/two control patches are reported.
This retrospective pilot provides conditional uncertainty, not map eligibility.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from glacier_offsets.velocity_basis import PERIOD, EPOCH_SECONDS
from glacier_offsets.velocity_fit import calibrate_noise
from glacier_offsets.velocity_io import digest, write_json
from glacier_offsets.velocity_quality import cluster_weights, flow_axis, temporal_counts
from glacier_offsets.velocity_systematics import (OrderRegionalDesign, OrderScreenCache,
    robust_background, order_screen, purged_folds, calendar_times, weighted_median)


VARIANTS={'baseline':(False,False),'order':(True,False),
          'solid_earth':(False,True),'both':(True,True)}
ESTIMATORS={'azimuth':'azimuth_direct','along_flow':'fixed_vertical'}
KEYS=[(e,v) for e in ESTIMATORS for v in VARIANTS]


def load(out):
    for filename,datafile in [('observations.json','observations.h5'),('solid_earth.json','solid_earth.h5')]:
        record=json.loads((out/filename).read_text())
        if record['sha256']!=digest(out/datafile):raise ValueError(f'Changed {datafile}')
    with h5py.File(out/'observations.h5') as h,h5py.File(out/'solid_earth.h5') as s:
        obs={k:h[k][:] for k in ['source_id','row','col','owner','early','late','data','valid',
             'peak','snr','order','calendar_early','calendar_late','pilot_target','x','y']}
        obs.update({k:h[k].asstr()[:] for k in ['regions','group','pair_id']})
        obs['geometry']=h['geometry_by_file'][0]
        obs['set_ra_m']=s['set_ra_m'][:]
    obs['valid'] &= np.isfinite(obs['set_ra_m']).all(-1)
    return obs


def axis(field,good,tc=None,sw=None,train=None):
    if train is None:train=np.ones(good.shape[1],bool)
    _,bias=robust_background(field,train,tc,threshold=.1,order_range=True)
    y=field['data']-field['order'][...,None]*bias[:,None]
    return flow_axis({**field,'data':y},good,tc,sw)


def fields(obs,threshold,min_pairs):
    regions=sorted(set(';'.join(obs['regions']).split(';')))
    selections=[(r,'region',r,np.array([r in m.split(';') for m in obs['regions']]))
                for r in regions if r.startswith(('target_','control_'))]
    selections += [(f'point_{sid}','point',next(r for r in regions if r.startswith('target_') and
                   r in obs['regions'][i].split(';')),obs['source_id']==sid)
                   for i,sid in enumerate(obs['source_id']) if obs['pilot_target'][i]]
    result=[];support=[]
    for label,cohort,region,take in selections:
        f={k:obs[k][take] for k in ['early','late','valid','peak','snr','geometry','group','source_id']}
        f['burst']=obs['owner'][take]+1
        f['order']=np.broadcast_to(obs['order'],f['valid'].shape).copy()
        f['original']=obs['data'][take]
        f['data']=f['original']-obs['set_ra_m'][take]
        f['calendar_early']=obs['calendar_early'];f['calendar_late']=obs['calendar_late']
        good=order_screen(f,threshold,order_range=True)
        keep=good.sum(1)>=min_pairs
        s={'label':label,'cohort':cohort,'region':region,'role':region.split('_')[0],
           'initial_sources':int(take.sum()),'supported_sources':int(keep.sum()),
           'valid_source_pairs':int(f['valid'].sum()),'screened_source_pairs':int(good.sum()),
           'supported_source_pairs':int(good[keep].sum()),
           'spatial_blocks':int(len(np.unique(f['group'][keep]))),
           'excluded_source_ids':f['source_id'][~keep].tolist()}
        support.append(s)
        if not keep.any():continue
        f={k:v[keep] if k not in ('calendar_early','calendar_late') else v for k,v in f.items()}
        f.update(id=label,cohort=cohort,region=region,role=s['role'],good=good[keep],threshold=threshold)
        f['axis']=axis(f,f['good'])
        result.append(f)
    # Controls have no physical flow direction: use the target_1 axis solely
    # as a fixed projection convention for their along-flow comparison.
    target_axis=next(f['axis'] for f in result if f['id']=='target_1')
    for f in result:
        if f['role']=='control':f['axis']=target_axis
    return result,support


def designs(field,cov):
    return {(e,v):OrderRegionalDesign(field,cov,ESTIMATORS[e],VARIANTS[v][0],order_range=True)
            for e,v in KEYS}


def variant_data(field,variant):
    return field['data'] if VARIANTS[variant][1] else field['original']


def rms(field,good,residual):
    w=cluster_weights(field['group'])[:,None]*good
    if not w.sum():return [None]*residual.shape[-1]
    return np.sqrt(np.sum(w[...,None]*np.nan_to_num(residual)**2,axis=(0,1))/w.sum()).tolist()


def background(field,train,order):
    if order:return robust_background(field,train,threshold=field['threshold'],order_range=True)[0]
    dt=field['late']-field['early']
    w=(field['valid']&train[None]&np.isfinite(field['peak'])&(field['peak']>=field['threshold'])).astype(float)
    v=np.stack([weighted_median(field['data'][...,ch]/dt,w) for ch in (0,1)],axis=-1)
    return np.nan_to_num(dt)[...,None]*v[:,None]


def predictive(field,ds):
    rows=[];calibration={v:np.full_like(field['data'],np.nan) for v in VARIANTS}
    calibration_good=np.zeros_like(field['valid'])
    fold_records=[]
    for fold,(train,test) in enumerate(purged_folds(*calendar_times(field))):
        train_good=order_screen(field,field['threshold'],allowed=train,order_range=True)&train[None]
        prediction=background(field,train,True)
        residual=field['data']-prediction
        enough=train_good.sum(1)>=8
        test_good=(field['valid']&test[None]&(field['peak']>=field['threshold'])&
                   np.isfinite(residual).all(-1)&(abs(residual[...,0])<=2)&
                   (abs(residual[...,1])<=5)&enough[:,None])
        train_good &= enough[:,None]
        e,l=calendar_times(field)
        train_dates=np.unique(np.floor(np.r_[e[train],l[train]]))
        test_dates=np.unique(np.floor(np.r_[e[test],l[test]]))
        assert not np.intersect1d(train_dates,test_dates).size
        fold_records.append({'fold':fold,'train':np.flatnonzero(train),'test':np.flatnonzero(test),
                             'train_acquisition_days':train_dates,'test_acquisition_days':test_dates})
        u=axis(field,train_good,train=train) if field['role']=='target' else field['axis']
        for ekey,v in KEYS:
            d=ds[ekey,v]
            try:
                fit=d.fit(train_good,u,data=variant_data(field,v),details=True)
                r=variant_data(field,v)[:,:,d.channels]-fit['prediction']
                value=rms(field,test_good,r)
                rows.append({'label':field['id'],'cohort':field['cohort'],'region':field['region'],
                    'estimator':ekey,'variant':v,'fold':fold,'n_test':int(test_good.sum()),
                    'rms_range_m':value[0] if ekey=='along_flow' else None,
                    'rms_azimuth_m':value[-1],'status':'ok'})
            except (ValueError,np.linalg.LinAlgError):
                rows.append({'label':field['id'],'cohort':field['cohort'],'region':field['region'],
                    'estimator':ekey,'variant':v,'fold':fold,'n_test':int(test_good.sum()),'status':'unsupported'})
        if field['role']=='control' and field['cohort']=='region':
            # No harmonic is removed from this calibration bank. Peak-only
            # support retains tails excluded by the supplementary fit screen.
            take=field['valid']&test[None]&(field['peak']>=field['threshold'])&enough[:,None]
            calibration_good |= take
            for v,(has_order,_) in VARIANTS.items():
                vf={**field,'data':variant_data(field,v)}
                p=background(vf,train,has_order)
                calibration[v][:,test]=vf['data'][:,test]-p[:,test]
    return rows,calibration,calibration_good,fold_records


def bootstrap_job(args):
    field,cov,counts,seed,path=args
    draws=len(counts)
    if path.exists():
        with np.load(path) as h:
            samples=h['samples']
        if samples.shape!=(draws,len(KEYS),2):raise ValueError('Changed bootstrap cache')
        return field['id'],samples
    rng=np.random.default_rng(seed);ds=designs(field,cov)
    cache=OrderScreenCache(field,field['threshold'],order_range=True)
    samples=np.full((draws,len(KEYS),2),np.nan)
    for b,tc in enumerate(counts):
        sw=cluster_weights(field['group'],rng)
        good=cache.mask_for(tc)
        u=axis(field,good,tc,sw) if field['role']=='target' else field['axis']
        for j,(e,v) in enumerate(KEYS):
            try:samples[b,j]=ds[e,v].fit(good,u,data=variant_data(field,v),temporal=tc,spatial=sw)['cs']
            except (ValueError,np.linalg.LinAlgError):pass
        if (b+1)%100==0:logging.info('%s bootstrap %d/%d',field['id'],b+1,draws)
    np.savez_compressed(path,samples=samples)
    return field['id'],samples


def summarize(out,analysis):
    observed=pd.read_csv(analysis/'observed.csv')
    cv=pd.read_csv(analysis/'predictive.csv')
    support=json.loads((analysis/'support.json').read_text())
    noise=json.loads((analysis/'control_noise.json').read_text())
    regional=observed[observed.cohort=='region']
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    targets=sorted(regional[regional.role=='target'].region.unique())
    colors=plt.get_cmap('tab10').colors
    for j,v in enumerate(VARIANTS):
        for ax,est in zip(axes[0],ESTIMATORS):
            a=regional[(regional.variant==v)&(regional.estimator==est)&(regional.role=='target')].set_index('region').reindex(targets)
            ax.errorbar(np.arange(len(targets))+(j-1.5)*.13,a.amplitude_m_day,
                yerr=a.sigma_amplitude_m_day,fmt='o',capsize=2,label=v,color=colors[j])
            ax.set_xticks(np.arange(len(targets)),targets)
            ax.set(title=est+' Msf: amplitude ± bootstrap SD',ylabel='Velocity amplitude (m/day)')
        sig=np.sqrt(np.diag(noise['variants'][v]['covariance_ra_m2']))
        axes[1,0].bar(np.array([0,1])+(j-1.5)*.18,sig,width=.18,label=v,color=colors[j])
    axes[1,0].set(xticks=[0,1],xticklabels=['Range','Azimuth'],ylabel='Empirical residual SD (m)',
        title='Controls: cross-fitted secular/order residual covariance')
    for i,r in enumerate(targets):
        a=regional[(regional.region==r)&(regional.estimator=='azimuth')].set_index('variant').loc[list(VARIANTS)]
        axes[1,1].plot(list(VARIANTS),a.sigma_amplitude_m_day,'o-',label=r)
    axes[1,1].set(title='Conditional azimuth amplitude uncertainty',ylabel='Bootstrap SD (m/day)')
    for ax in axes.ravel():ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Evans nuisance pilot — matched support/weights; uncertainty calibration pending')
    fig.savefig(analysis/'summary.png',dpi=160);plt.close(fig)
    lines=['# Evans spacecraft-order and solid-Earth nuisance pilot','',
      'Retrospective four-way comparison on the existing 117-pair record. These are exploratory fits with conditional bootstrap uncertainty; no production mask, correction or velocity fit is replaced. M2/Msf attribution and absolute recovery remain unresolved.','',
      '## Design and verification','',
      '- Native RA observations, exact owning-burst UTC endpoints and original source identities are retained. Existing CATS range correction remains applied. Full ENU solid-Earth endpoint differences are subtracted exactly once for the appropriate variants; ocean loading is not applied.',
      '- Baseline, order only, solid-Earth only and both use identical source support and fixed original pooled RA covariance. Order coefficients are shared per represented burst and radar component; source secular velocities and regional harmonic coefficients are fitted jointly.',
      '- Common observed support uses peak >=0.1 and acquisition-purged, two-component order-aware residual limits of2/5m. Sources require20 retained pairs. These are provisional diagnostic choices. Whole-record source eligibility is retrospective.',
      '- Primary results are direct satellite-azimuth velocity. Joint RA along-flow estimates assume zero remaining vertical displacement after supplied corrections and a data-derived flow axis. The amplitudes are different projections.',
      '- Observed bootstrap resamples whole30-day calendar blocks and3km physical blocks within each patch. Screening and target flow axes are recalculated in every draw, with the same draw across ablations. Temporal draws are shared between patches; cross-patch spatial covariance is not estimated.',
      '- Held-out predictions use60-day folds with shared acquisition dates purged. All variants share test support within each fold. Control calibration diagnostics remove only training-estimated secular/order motion, retain any control harmonic, and use peak-only held-out support without2/5m tail clipping.',
      '- The bootstrap is a dependence sensitivity method, not an explicit per-acquisition random-effect model or established coverage calibration. Historical GPS was not used to constrain fitted amplitude or phase.','',
      '## Regional results','',
      '| Region | Variant | Azimuth amplitude | Conditional SD | Joint along-flow amplitude | Conditional SD |',
      '|---|---|---:|---:|---:|---:|']
    for r in targets:
        for v in VARIANTS:
            a=regional[(regional.region==r)&(regional.variant==v)].set_index('estimator')
            lines.append(f"| {r} | {v} | {a.loc['azimuth','amplitude_m_day']:.4f} | {a.loc['azimuth','sigma_amplitude_m_day']:.4f} | {a.loc['along_flow','amplitude_m_day']:.4f} | {a.loc['along_flow','sigma_amplitude_m_day']:.4f} |")
    lines += ['', 'Amplitudes and standard deviations are m/day. Phase is not interpreted where the harmonic is poorly constrained. Coefficient vectors, covariance, phase sensitivity, order terms and bootstrap failures are retained in the data products.','',
      '## Control residual covariance diagnostics','',
      '| Variant | Range SD, m | Azimuth SD, m | RA correlation | Held-out samples | Spatial groups |',
      '|---|---:|---:|---:|---:|---:|']
    for v in VARIANTS:
        n=noise['variants'][v];s=np.sqrt(np.diag(n['covariance_ra_m2']))
        lines.append(f"| {v} | {s[0]:.4f} | {s[1]:.4f} | {n['correlation']:.3f} | {n['samples']} | {n['spatial_groups']} |")
    lines += ['', 'These diagnostics use cross-fitted residuals but have not been tested on an independent calibration holdout. They are not substituted into the matched fits. More sources in the same two control regions do not establish transferability to the ice targets.','',
      '## Source support','', '| Cohort | Label | Initial sources | Supported sources | Supported source-pairs | Physical blocks |',
      '|---|---|---:|---:|---:|---:|']
    for s in support:
        lines.append(f"| {s['cohort']} | {s['label']} | {s['initial_sources']} | {s['supported_sources']} | {s['supported_source_pairs']} | {s['spatial_blocks']} |")
    lines += ['', '## Next inference checks','',
      'Use the matched ablations to decide which nuisance structure belongs in the residual ledger. Then validate control-to-target covariance transfer, model acquisition-shared errors, test isolated/combined GPS-scale constituent injections, and assess interval coverage/null rejection. Smaller scatter or narrower bootstrap intervals alone do not demonstrate recovery or authorize mapping.','',
      'Files: observed.csv, predictive.csv, support.json, control_noise.json, fits.json, fitted_coefficients.npz, fold_records.json, per-field bootstrap files and recipe.json. Correction accuracy/provenance is in the parent solid_earth.json.']
    text='\n'.join(lines)+'\n'
    for a,b in [('limits of2/5m','limits of 2/5 m'),('require20','require 20'),('whole30-day','whole 30-day'),('and3km','and 3 km'),('use60-day','use 60-day'),('without2/5m','without 2/5 m')]:text=text.replace(a,b)
    (analysis/'REPORT.md').write_text(text)
    shutil.copy2(analysis/'REPORT.md',out/'REPORT.md')
    shutil.copy2(analysis/'summary.png',out/'summary.png')
    write_json(out/'latest_analysis.json',{'directory':str(analysis.relative_to(out)),
        'status':'complete','scope':'matched nuisance pilot; no scientific calibration or mapping pass'})


def run(root,out,draws,workers,seed,threshold,min_pairs,stage):
    obs=load(out)
    modules=['velocity_systematics','velocity_quality','velocity_basis','velocity_temporal','velocity_fit']
    import glacier_offsets
    src=Path(glacier_offsets.__file__).parent
    recipe={'schema':'evans-nuisance-pilot-v1','observations_sha256':digest(out/'observations.h5'),
        'solid_earth_sha256':digest(out/'solid_earth.h5'),'draws':draws,'seed':seed,
        'threshold':threshold,'minimum_pairs_per_source':min_pairs,
        'variants':VARIANTS,'estimators':ESTIMATORS,
        'code_sha256':{k:digest(src/(k+'.py')) for k in modules},'runner_sha256':digest(Path(__file__)),
        'fixed_noise_sha256':digest(root/'work/stack/packages/velocity_fit_robust/noise.json')}
    import hashlib
    rid=hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()[:12]
    analysis=out/'analysis'/rid;analysis.mkdir(parents=True,exist_ok=True)
    write_json(analysis/'recipe.json',recipe)
    if stage=='report':summarize(out,analysis);return
    if (analysis/'complete.json').exists():summarize(out,analysis);return
    fs,support=fields(obs,threshold,min_pairs);write_json(analysis/'support.json',support)
    cov=np.array(json.loads((root/'work/stack/packages/velocity_fit_robust/noise.json').read_text())['covariance_ra_m2'])
    state={};cv=[];folds={};control={v:[] for v in VARIANTS};control_groups=[];control_snr=[];control_peak=[]
    coeff={};records=[]
    for f in fs:
        ds=designs(f,cov);fits={}
        for e,v in KEYS:
            fit=ds[e,v].fit(f['good'],f['axis'],data=variant_data(f,v),details=True)
            fits[e,v]=fit
            for key in ['cs','order_m','nuisance']:coeff[f['id']+'__'+e+'__'+v+'__'+key]=fit[key]
            records.append({'label':f['id'],'cohort':f['cohort'],'region':f['region'],'role':f['role'],
                'estimator':e,'variant':v,'axis_EN':f['axis'],'source_ids':f['source_id'],
                'order_terms':ds[e,v].order_terms,'cs':fit['cs'],'order_m':fit['order_m'],
                'condition':fit['condition'],'conditional_information':fit['information'],
                'observed_residual_rms_m':rms(f,f['good'],variant_data(f,v)[:,:,ds[e,v].channels]-fit['prediction'])})
        state[f['id']]=fits
        rows,cal,good,fold_records=predictive(f,ds);cv.extend(rows);folds[f['id']]=fold_records
        if f['role']=='control' and f['cohort']=='region':
            common=good&np.logical_and.reduce([np.isfinite(cal[v]).all(-1) for v in VARIANTS])
            for v in VARIANTS:control[v].extend(cal[v][common])
            control_snr.extend(f['snr'][common]);control_peak.extend(f['peak'][common])
            groups=np.broadcast_to((f['id']+':'+f['group'].astype(object))[:,None],common.shape)
            control_groups.extend(groups[common])
            np.savez_compressed(analysis/(f['id']+'_heldout_residuals.npz'),good=common,
                                 **cal,pair_id=obs['pair_id'],group=f['group'])
        logging.info('Observed fits / held-out predictions: %s',f['id'])
    pd.DataFrame(cv).to_csv(analysis/'predictive.csv',index=False)
    write_json(analysis/'fold_records.json',folds)
    write_json(analysis/'fits.json',{'rows':records,'fixed_covariance_ra_m2':cov})
    np.savez_compressed(analysis/'fitted_coefficients.npz',**coeff)
    noise={v:calibrate_noise(np.array(control[v]),np.array(control_snr),np.array(control_peak),
                            np.array(control_groups,dtype=object)) for v in VARIANTS}
    write_json(analysis/'control_noise.json',{'variants':noise,
        'scope':'cross-fitted control residual diagnostics only; not independently validated or used to reweight ablations'})
    rng=np.random.default_rng(seed)
    counts=np.stack([temporal_counts(rng,obs['calendar_early'],obs['calendar_late']) for _ in range(draws)])
    np.save(analysis/'temporal_counts.npy',counts)
    tasks=[(f,cov,counts,seed+100+i,analysis/(f['id']+'_bootstrap.npz')) for i,f in enumerate(fs)]
    samples={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(bootstrap_job,t) for t in tasks]
        for future in as_completed(futures):
            label,values=future.result();samples[label]=values
            logging.info('Completed conditional uncertainty: %s',label)
    rows=[]
    for f in fs:
        for j,(e,v) in enumerate(KEYS):
            fit=state[f['id']][e,v];cs=fit['cs'];value=samples[f['id']][:,j]
            value=value[np.isfinite(value).all(1)]
            amp=float(np.linalg.norm(cs));sigma=np.cov(value.T) if len(value)>=10 else np.full((2,2),np.nan)
            grad=cs/max(amp,1e-15);phase_grad=np.array([-cs[1],cs[0]])/max(amp*amp,1e-30)
            rows.append({'label':f['id'],'cohort':f['cohort'],'region':f['region'],'role':f['role'],
                'estimator':e,'variant':v,'C_m_day':cs[0],'S_m_day':cs[1],'amplitude_m_day':amp,
                'phase_deg':np.degrees(np.arctan2(cs[1],cs[0])),
                'sigma_amplitude_m_day':np.sqrt(max(float(grad@sigma@grad),0)),
                'phase_std_deg':np.degrees(np.sqrt(max(float(phase_grad@sigma@phase_grad),0))),
                'C_variance':sigma[0,0],'CS_covariance':sigma[0,1],'S_variance':sigma[1,1],
                'successful_draws':len(value),'attempted_draws':draws,'condition':fit['condition']})
    pd.DataFrame(rows).to_csv(analysis/'observed.csv',index=False)
    write_json(analysis/'complete.json',{'status':'complete','draws':draws,'fields':len(fs),
        'scope':'matched ablation fits; conditional intervals; absolute recovery and calibration pending'})
    summarize(out,analysis)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=500)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261108)
    p.add_argument('--threshold',type=float,default=.1);p.add_argument('--min-pairs',type=int,default=20)
    p.add_argument('--stage',choices=['fit','report'],default='fit')
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    run(a.root.resolve(),a.output.resolve(),a.draws,a.workers,a.seed,a.threshold,a.min_pairs,a.stage)
