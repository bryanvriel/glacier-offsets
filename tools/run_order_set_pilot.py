"""Isolated 2017 spacecraft-order / solid-Earth tide regional experiment.

Run with PYTHONPATH=src in ice after prepare_order_set_experiment.py in isce3.
Every output is exploratory; the independent 2018 record remains untouched.
"""
import argparse
import json
import logging
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
from scipy.stats import f as f_distribution
from glacier_offsets.velocity_basis import harmonic,PERIOD,utc_days,EPOCH_SECONDS
from glacier_offsets.velocity_inspect import M2_PERIOD,O1_PERIOD
from glacier_offsets.velocity_io import write_json,digest
from glacier_offsets.velocity_fit import calibrate_noise
from glacier_offsets.velocity_quality import (reference_times,purged_folds,cluster_weights,flow_axis,
    temporal_counts,paired_field_draw,moving_indices,quiet_median)
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_systematics import OrderRegionalDesign,order_screen,order_residual,robust_background,holm,OrderScreenCache,calendar_times

VARIANTS={
 'primary_azimuth':('azimuth_direct',True,True),
 'no_order':('azimuth_direct',False,True),
 'no_SET':('azimuth_direct',True,False),
 'azimuth_along':('azimuth_along',True,True),
 'fixed_vertical':('fixed_vertical',True,True),
 'residual_vertical':('residual_vertical',True,True)}


def load(root,out):
    meta=json.loads((out/'corrections.json').read_text())
    if digest(out/'corrected-inputs.npz')!=meta['output_sha256']:raise ValueError('Changed tide cache')
    if digest(meta['input']['path'])!=meta['input']['sha256']:raise ValueError('Changed frozen input')
    fields=[];summary=[]
    with h5py.File(meta['input']['path']) as source:
        calendar_early=(source['earlier_time'][:]-EPOCH_SECONDS)/86400.
        calendar_late=(source['later_time'][:]-EPOCH_SECONDS)/86400.
    with np.load(out/'corrected-inputs.npz') as h:
        for r in json.loads((root/'regions.reviewed.json').read_text())['regions']:
            f={k:h[r['id']+'__'+k] for k in meta['keys']}
            f['uncorrected']=f['data'].copy();f['data']=f['data']-f['set_ra_m']
            f.update(calendar_early=calendar_early,calendar_late=calendar_late)
            clean=order_screen(f);take=clean.sum(1)>=20
            summary.append({'region':r['id'],'initial_sources':len(take),'supported_sources':int(take.sum()),
                'retained_observations':int(clean[take].sum()),'valid_observations':int(f['valid'][take].sum()),
                'spatial_blocks':len(np.unique(f['group'][take]))})
            f={k:v[take] if k not in ('calendar_early','calendar_late') else v for k,v in f.items()};f.update(id=r['id'],role=r['role'],clean=clean[take]);fields.append(f)
    if not (out/'support.json').exists():write_json(out/'support.json',summary)
    return fields


def axis(field,good,tc=None,sw=None):
    pred,bias=robust_background(field,np.ones(good.shape[1],bool),tc)
    y=field['data'].copy();y[...,1]-=field['order']*bias[:,None]
    return flow_axis({**field,'data':y},good,tc,sw)


def calibrate(fields,out):
    residual=[];snr=[];peak=[];groups=[];heldout=[]
    for f in fields:
        if f['role']!='control':continue
        e,l=calendar_times(f);first=l<utc_days('2017-07-01');last=e>=utc_days('2017-07-01')
        acq=np.unique(np.floor(np.r_[e[last],l[last]]))
        first &= ~np.isin(np.floor(e),acq)&~np.isin(np.floor(l),acq)
        r=order_residual(f,allowed=first);good=order_screen(f,allowed=first)
        residual.extend(r[good]);snr.extend(f['snr'][good]);peak.extend(f['peak'][good])
        groups.extend(np.broadcast_to((f['id']+':'+f['group'].astype(object))[:,None],good.shape)[good])
        pred,_=robust_background(f,first);r=f['data']-pred
        mask=f['valid']&last[None]&np.isfinite(f['peak'])&(f['peak']>=.08)&np.isfinite(r).all(-1)&(abs(r[...,0])<=2)&(abs(r[...,1])<=5)
        heldout.append({'region':f['id'],'train_acquisition_days':np.unique(np.floor(np.r_[e[first],l[first]])),
            'test_acquisition_days':acq,'heldout_clean_pairs':int(mask.sum()),
            'heldout_rms_ra_m':np.sqrt(np.nanmean(r[mask]**2,axis=0))})
        f['bank_residual']=r # H1 secular and order prediction applied to all H2, including failures.
        center=quiet_median(np.where(f['valid'][:,last,None],r[:,last],np.nan),axis=1)
        f['bank_residual']=r-center[:,None]
        f['bank_half']=np.flatnonzero(last)
    noise=calibrate_noise(np.array(residual),np.array(snr),np.array(peak),np.array(groups))
    if not (out/'clean_noise.json').exists():write_json(out/'clean_noise.json',{'training':'H1 controls only; 60-day acquisition-purged robust secular and order predictions, peak>=.08, residual 2/5 m',
        'pooled':noise,'heldout_H2':heldout,'bank':'H2 paired quality/failure fields minus H1 secular+order; source median centered in H2; Msf not removed'})
    return np.array(noise['covariance_ra_m2'])


def build_state(fields,cov):
    state={};target_axis=None
    for f in fields:
        u=axis(f,f['clean']) if f['role']=='target' else target_axis
        if f['id']=='target_1':target_axis=u
        designs={name:OrderRegionalDesign(f,cov,model,order) for name,(model,order,_) in VARIANTS.items()}
        fits={name:d.fit(f['clean'],u,data=f['data'] if VARIANTS[name][2] else f['uncorrected'],details=True) for name,d in designs.items()}
        state[f['id']]={'axis':u,'designs':designs,'fits':fits}
    return state


def score(field,fit,good,channels,data=None):
    y=field['data'] if data is None else data
    residual=y[:,:,channels]-fit['prediction'];w=cluster_weights(field['group'])[:,None]*good
    return {('range','azimuth')[ch]+'_rms_m':float(np.sqrt(np.sum(w*np.nan_to_num(residual[...,j])**2)/np.sum(w))) for j,ch in enumerate(channels)}


def bootstrap(field,design,draws,rng,direction=None,width=30.,models=None):
    values=[];fail=0;screen_cache=OrderScreenCache(field)
    for b in range(draws):
        tc=temporal_counts(rng,*calendar_times(field),width=width);sw=cluster_weights(field['group'],rng)
        good=screen_cache.mask_for(tc)
        try:values.append(design.fit(good,direction,temporal=tc,spatial=sw)['cs'])
        except (ValueError,np.linalg.LinAlgError):fail+=1
    return np.asarray(values),fail


def uncertainty(cs,values):
    values=np.asarray(values);values=values[np.isfinite(values).all(1)];delta=values-cs
    cov=np.cov(delta.T);inv=np.linalg.inv(cov);stat=float(cs@inv@cs)
    null=np.einsum('ni,ij,nj->n',delta,inv,delta)
    a=np.linalg.norm(cs);grad=cs/max(a,1e-15)
    return {'covariance':cov,'wald':stat,'p':float((1+np.sum(null>=stat))/(len(values)+1)),
        'cutoff95':float(np.quantile(null,.95)),'sigma_amplitude':float(np.sqrt(grad@cov@grad)),
        'amplitude_percentiles':np.quantile(np.linalg.norm(values,axis=1),[.025,.5,.975]),'successful_draws':len(values)}


def observed(fields,cov,state,out,draws,seed):
    names=list(VARIANTS);samples={f['id']:np.full((draws,len(names),2),np.nan) for f in fields}
    screen_caches={f['id']:OrderScreenCache(f) for f in fields}
    rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
    for b in range(draws):
        tc=temporal_counts(rng,*calendar_times(fields[0]));target_axis=None
        for f in fields:
            sw=cluster_weights(f['group'],rng);good=screen_caches[f['id']].mask_for(tc)
            u=axis(f,good,tc,sw) if f['role']=='target' else target_axis
            if f['id']=='target_1':target_axis=u
            for j,name in enumerate(names):
                try:samples[f['id']][b,j]=state[f['id']]['designs'][name].fit(good,u,data=f['data'] if VARIANTS[name][2] else f['uncorrected'],temporal=tc,spatial=sw)['cs']
                except (ValueError,np.linalg.LinAlgError):pass
        if (b+1)%100==0:logging.info('Observed paired rescreened bootstrap %d/%d',b+1,draws)
    rows=[]
    for f in fields:
        st=state[f['id']]
        for j,name in enumerate(names):
            fit=st['fits'][name];cs=fit['cs'];u=uncertainty(cs,samples[f['id']][:,j]);st[name]=u
            row={'region':f['id'],'role':f['role'],'variant':name,'cs_m_per_day':cs,'amplitude_m_per_day':float(np.linalg.norm(cs)),
                'phase_deg':float(np.degrees(np.arctan2(cs[1],cs[0]))),'order_m':fit['order_m'],'burst_indices':st['designs'][name].bursts,
                'axis_EN':st['axis'],'condition':fit['condition'],**u,
                **score(f,fit,f['clean'],st['designs'][name].channels,f['data'] if VARIANTS[name][2] else f['uncorrected'])}
            if name=='residual_vertical':
                mean=np.average(fit['nuisance'][:,2:],axis=0,weights=cluster_weights(f['group']))
                row['vertical_mean_cs_m']=mean;row['vertical_M2_mean_amplitude_m']=float(np.linalg.norm(mean[:2]));row['vertical_O1_mean_amplitude_m']=float(np.linalg.norm(mean[2:]))
            rows.append(row)
    primary=[r for r in rows if r['variant']=='primary_azimuth' and r['role']=='target']
    for r,p in zip(primary,holm([r['p'] for r in primary])):r['holm_p_four_targets']=float(p)
    write_json(out/'observed.json',{'rows':rows,'primary':'satellite-azimuth Msf, four originally selected ice targets; controls are negative controls',
        'uncertainty':'paired whole-field 30-day temporal and physical 3-km spatial block resampling; frozen screen algorithm rerun using training counts in every replicate; flow axis reassessed for comparison fits',
        'p':'centered empirical two-coefficient bootstrap, +1 Monte Carlo correction; calibrated against synthetic draws separately; no exact F claim',
        'status':'retrospective exploratory 2017 family, not preregistration; freeze recipe before opening 2018'})
    np.savez_compressed(out/'bootstrap.npz',**samples)
    # Source coefficients make the fitted predictions independently reconstructable.
    np.savez_compressed(out/'fitted-coefficients.npz',**{f['id']+'__'+name+'__'+key:state[f['id']]['fits'][name][key]
        for f in fields for name in names for key in ('cs','order_m','nuisance','prediction')})
    return rows


def predictive(fields,cov,state,out):
    rows=[]
    for fold,(train,test) in enumerate(purged_folds(*calendar_times(fields[0]))):
        target_axis=None
        for f in fields:
            # Holdout residual screen predicts only from this fold's training data.
            train_good=order_screen(f,allowed=train)&train[None]
            pred,_=robust_background(f,train)
            residual=f['data']-pred
            test_good=(f['valid']&test[None]&np.isfinite(f['peak'])&(f['peak']>=.08)
                &np.isfinite(residual).all(-1)&(abs(residual[...,0])<=2)&(abs(residual[...,1])<=5))
            available=train_good.sum(1)>=12;train_good &= available[:,None];test_good &= available[:,None]
            u=axis({**f,'valid':f['valid']&train[None]},train_good) if f['role']=='target' else target_axis
            if f['id']=='target_1':target_axis=u
            for name,d in state[f['id']]['designs'].items():
                y=f['data'] if VARIANTS[name][2] else f['uncorrected']
                fit=d.fit(train_good,u,data=y,details=True)
                rows.append({'region':f['id'],'fold':fold,'variant':name,'test_observations':int(test_good.sum()),**score(f,fit,test_good,d.channels,y)})
    write_json(out/'predictive.json',{'rows':rows,'method':'six acquisition-purged 60-day folds; shared corrected-data support across ablations, >=12 train epochs/source; train-only secular axis and screen'})
    return rows


def block_sensitivity(fields,cov,out,draws,seed):
    rows=[]
    for f in fields:
        for width in (15.,30.,60.):
            d=OrderRegionalDesign(f,cov);cs=d.fit(f['clean'])['cs']
            values,fail=bootstrap(f,d,draws,np.random.default_rng(np.random.SeedSequence([seed,3,int(width)])),width=width)
            rows.append({'region':f['id'],'block_days':width,'failures':fail,**uncertainty(cs,values)})
        logging.info('Completed block sensitivity %s',f['id'])
    write_json(out/'block-sensitivity.json',{'rows':rows})


def scans(fields,cov,out,seed):
    # Include the 12-day neighborhood and exact named aliases; no scan selection feeds primary fits.
    periods=np.unique(np.r_[1/np.linspace(1/45,1/12.05,400),PERIOD,14.19,15.95])
    rows=[];arrays={};rng=np.random.default_rng(np.random.SeedSequence([seed,4]))
    for f in fields:
        e,l=calendar_times(f);labels=np.floor(((e+l)/2-np.nanmin((e+l)/2))/30).astype(int)
        blocks=np.unique(labels);good=f['clean'];sw=cluster_weights(f['group'])
        # Null prediction uses source secular and burst order only, independently of frequency.
        zero={**f,'data':f['data'].copy()};d0=OrderRegionalDesign(f,cov)
        fit=d0.fit(good,details=True);h=harmonic(f['early'],f['late'])
        null_prediction=fit['prediction']-np.einsum('stj,j->st',h,fit['cs'])[...,None]
        # Refit null nuisance by projecting the null residual against its order/secular columns.
        b=d0.b[...,0];oc=d0.q[...,d0.nh:]
        y=f['data'][:,:,[1]]/d0.factor[0,0]
        nn=np.sum(good*b[...,0]**2,axis=1);ny=np.sum(good*b[...,0]*np.nan_to_num(y[...,0]),axis=1)
        no=np.einsum('st,st,stj->sj',good,b[...,0],oc[...,0,:]);oo=np.einsum('s,st,sti,stj->ij',sw,good,oc[...,0,:],oc[...,0,:])
        oy=np.einsum('s,st,sti,st->i',sw,good,oc[...,0,:],np.nan_to_num(y[...,0]))
        order=np.linalg.solve(oo-np.einsum('s,si,sj,s->ij',sw,no,no,1/nn),oy-np.einsum('s,si,s,s->i',sw,no,ny,1/nn))
        secular=(ny-no@order)/nn
        null_prediction=(b[...,0]*secular[:,None]+np.einsum('sti,i->st',oc[...,0,:],order))*d0.factor[0,0]
        residual=(f['data'][...,1]-null_prediction)[...,None]
        coefficients=[];scores=[];conditions=[]
        for period in periods:
            d=OrderRegionalDesign(f,cov,period=period)
            fitted=d.fit(good,operator=True);op=fitted['operator']
            scores.append(np.stack([np.einsum('istc,stc->i',op[:, :,labels==block],np.nan_to_num(residual[:,labels==block])) for block in blocks]))
            coefficients.append(fitted['cs']);conditions.append(fitted['condition'])
        coefficients=np.array(coefficients);scores=np.array(scores)
        covariance=np.einsum('pbi,pbj->pij',scores,scores)*len(blocks)/(len(blocks)-1)
        inv=np.linalg.pinv(covariance,hermitian=True)
        stat=np.einsum('pi,pij,pj->p',coefficients,inv,coefficients)
        webb=np.array([-np.sqrt(1.5),-1.,-np.sqrt(.5),np.sqrt(.5),1.,np.sqrt(1.5)])
        weights=rng.choice(webb,(1999,len(blocks)))
        null=np.einsum('nb,pbi->npi',weights,scores)
        nullstat=np.einsum('npi,pij,npj->np',null,inv,null)
        maxnull=np.max(nullstat,axis=1);i=np.argmin(abs(periods-PERIOD));best=int(np.argmax(stat))
        rows.append({'region':f['id'],'Msf_statistic':float(stat[i]),'Msf_scan_max_p':float((1+np.sum(maxnull>=stat[i]))/2000),
            'largest_period_days':float(periods[best]),'largest_statistic':float(stat[best]),
            'largest_away_from_Msf_period_days':float(periods[np.argmax(np.where(abs(periods-PERIOD)>.6,stat,-1))])})
        arrays.update({f['id']+'__cs':coefficients,f['id']+'__statistic':stat,f['id']+'__condition':conditions})
        logging.info('Completed scan %s',f['id'])
    np.savez_compressed(out/'scans.npz',period_days=periods,**arrays)
    write_json(out/'scans.json',{'rows':rows,'method':'azimuth direct, order included, matched fixed observed screen; 30-day null-residual cluster scores and 1999 Webb multipliers; max over 403 periods; conditional scan diagnostic, screen not resampled',
        'calibration':'null residual fitted without Msf; no exact F reference or white-noise independent-source claim'})


def nested_job(args):
    synthetic,cov,msf,draws,seed,b,fi,amplitude,frozen_cov,frozen_cutoff,name=args
    d=OrderRegionalDesign(synthetic,cov);good=order_screen(synthetic)
    cs=d.fit(good)['cs'];truth=d.fit(good,data=msf)['cs']
    values,fail=bootstrap(synthetic,d,draws,np.random.default_rng(np.random.SeedSequence([seed,6,b,fi,int(amplitude>0)])))
    unc=uncertainty(cs,values);delta=cs-truth;inv=np.linalg.inv(unc['covariance'])
    e,l=calendar_times(synthetic);mid=(e+l)/2
    nb=len(np.unique(np.floor((mid-np.nanmin(mid))/30)))
    fp=float(f_distribution.sf(unc['wald']*(nb-2)/(2*(nb-1)),2,nb-2))
    record={'case':name,'region':synthetic['id'],'draw':b,'nested_p':unc['p'],'F_reference_p':fp,
        'coverage':bool(delta@inv@delta<=unc['cutoff95']),
        'frozen_coverage':bool(delta@np.linalg.inv(frozen_cov)@delta<=frozen_cutoff),
        'cov_trace_ratio_to_observed':float(np.trace(unc['covariance'])/np.trace(frozen_cov)),
        'bootstrap_failures':fail,'successful_draws':len(values)}
    return record,values


def recovery(fields,cov,state,out,draws,nested_datasets,nested_draws,seed):
    stamp={'recipe_sha256':digest(out/'recipe.json'),'observed_sha256':digest(out/'observed.json'),
           'correction_cache_sha256':digest(out/'corrected-inputs.npz')}
    checkpoint=out/'recovery-inputs.json'
    if checkpoint.exists() and json.loads(checkpoint.read_text())!=stamp:
        raise ValueError('Recovery inputs changed; cannot reuse simulation checkpoints')
    if not checkpoint.exists():write_json(checkpoint,stamp)
    cases=[('null',0.,0.,0.),('benchmark',.1,0.,0.),('null_vertical20',0.,.2,0.),
           ('benchmark_vertical20',.1,.2,0.),('benchmark_M2_003',.1,.2,.03),
           ('benchmark_M2_020',.1,.2,.2),('null_M2_020',0.,.2,.2)]
    models=['primary_azimuth','no_order','azimuth_along','residual_vertical']
    banks=[]
    for f in fields:
        if f['role']!='control':continue
        half=f['bank_half'];first=calendar_times(f)[1]<utc_days('2017-07-01')
        # The H1 science fit estimates a common burst-order term; keep any control
        # Msf in the bank, rather than fitting it away as random noise.
        good=order_screen(f,allowed=first)&first[None]
        fitted=state[f['id']]['designs']['primary_azimuth'].fit(good,details=True)
        secular_order=fitted['prediction'][...,0]-np.einsum('stj,j->st',harmonic(f['early'],f['late']),fitted['cs'])
        residual=f['bank_residual'].copy();residual[...,1]=f['data'][...,1]-secular_order
        center=quiet_median(np.where(f['valid'][:,half,None],residual[:,half],np.nan),axis=1)
        banks.append(({**f,'residual':residual-center[:,None]},half))
    rows=json.loads((out/'recovery.json').read_text())['rows'] if (out/'recovery.json').exists() else []
    nested=json.loads((out/'nested-calibration.json').read_text())['rows'] if (out/'nested-calibration.json').exists() else []
    completed={r['case'] for r in rows}
    benchmark_nested=set(np.floor(np.linspace(0,draws,nested_datasets,endpoint=False)).astype(int))
    for name,amplitude,vfraction,m2 in cases:
        if name in completed:
            logging.info('Reusing completed simulation case %s',name);continue
        rng=np.random.default_rng(np.random.SeedSequence([seed,5]))
        results={f['id']:{m:[] for m in models} for f in fields};nested_cs={};nested_jobs=[]
        for b in range(draws):
            bank,half=banks[int(rng.integers(len(banks)))];nt=fields[0]['valid'].shape[1]
            ti=np.resize(np.r_[half[moving_indices(rng,len(half),5)],half[moving_indices(rng,len(half),5)]],nt)
            phi=2*np.pi*(b+rng.random())/draws;phasor=np.array([np.cos(phi),np.sin(phi)])
            verticalphase=rng.uniform(0,2*np.pi,2);m2phase=rng.uniform(0,2*np.pi)
            for fi,f in enumerate(fields):
                base=paired_field_draw(rng,bank,f,'paired_quality',time_index=ti)
                u=state[f['id']]['axis'];hb=harmonic(f['early'],f['late'])
                projection=f['geometry'][...,:2]@u
                msf=np.einsum('stj,j->st',hb,amplitude*phasor)[...,None]*projection[:,None]
                z=np.r_[.6*vfraction*np.array([np.cos(verticalphase[0]),np.sin(verticalphase[0])]),
                        .2*vfraction*np.array([np.cos(verticalphase[1]),np.sin(verticalphase[1])])]
                vertical=np.einsum('stj,j->st',vertical_columns(f['early'],f['late']),z)[...,None]*f['geometry'][:,None,:,2]
                m2signal=np.einsum('stj,j->st',harmonic(f['early'],f['late'],M2_PERIOD),m2*np.array([np.cos(m2phase),np.sin(m2phase)]))[...,None]*projection[:,None]
                truth_prediction,_=robust_background(f,np.ones(nt,bool))
                # Replace source screen offsets by the known regional burst-order coefficients.
                _,oldbias=robust_background(f,np.ones(nt,bool));truth_prediction[...,1]-=f['order']*oldbias[:,None]
                orderfit=state[f['id']]['fits']['primary_azimuth']['order_m'];bursts=state[f['id']]['designs']['primary_azimuth'].bursts
                ordercoef=np.array([orderfit[np.flatnonzero(bursts==x)[0]] for x in f['burst']])
                truth_prediction[...,1]+=f['order']*ordercoef[:,None]
                synthetic={**base,'data':base['data']+truth_prediction+msf+vertical+m2signal}
                good=order_screen(synthetic);fitaxis=axis(synthetic,good) if f['role']=='target' else u
                for model in models:
                    d=state[f['id']]['designs'][model]
                    try:
                        cs=d.fit(good,fitaxis,data=synthetic['data'])['cs']
                        truth=d.fit(good,fitaxis,data=msf)['cs']
                        delta=cs-truth;unc=state[f['id']][model];inv=np.linalg.inv(unc['covariance'])
                        ae=abs(np.linalg.norm(cs)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-15)
                        pe=abs(np.degrees(np.angle(np.exp(1j*(np.arctan2(cs[1],cs[0])-np.arctan2(truth[1],truth[0]))))))
                        results[f['id']][model].append([*cs,*truth,ae,pe,float(delta@inv@delta<=unc['cutoff95']),float(cs@inv@cs>unc['cutoff95']),good.sum()/max(f['valid'].sum(),1)])
                    except (ValueError,np.linalg.LinAlgError):continue
                if (name=='null' and b<nested_datasets) or (name=='benchmark' and b in benchmark_nested):
                    nested_jobs.append((synthetic,cov,msf,nested_draws,seed,b,fi,amplitude,
                        state[f['id']]['primary_azimuth']['covariance'],state[f['id']]['primary_azimuth']['cutoff95'],name))
            if (b+1)%10==0:logging.info('Recovery %s %d/%d (nested subset: %d datasets per region)',name,b+1,draws,nested_datasets)
        if nested_jobs:
            logging.info('Running %d nested synthetic calibrations with four numerical workers',len(nested_jobs))
            with ProcessPoolExecutor(max_workers=4) as pool:
                for record,values in pool.map(nested_job,nested_jobs,chunksize=1):
                    nested.append(record);nested_cs[record['region']+'__'+str(record['draw'])]=values
            logging.info('Completed nested calibrations for %s',name)
        np.savez_compressed(out/(name+'-recovery.npz'),**{rid+'__'+m:np.asarray(v) for rid,models_ in results.items() for m,v in models_.items()})
        if nested_cs:np.savez_compressed(out/(name+'-nested-bootstrap.npz'),**nested_cs)
        for f in fields:
            for model in models:
                a=np.asarray(results[f['id']][model])
                rows.append({'case':name,'region':f['id'],'variant':model,'draws':len(a),'failures':draws-len(a),
                    'along_flow_Msf_injected_m_day':amplitude,'M2_horizontal_velocity_m_day':m2,
                    'M2_residual_height_m':.6*vfraction,'O1_residual_height_m':.2*vfraction,
                    'median_amplitude_error':float(np.median(a[:,4])) if amplitude else None,
                    'median_phase_error_deg':float(np.median(a[:,5])) if amplitude else None,
                    'frozen_ellipse_coverage':float(np.mean(a[:,6])),
                    'frozen_ellipse_null_FPR':float(np.mean(a[:,7])) if amplitude==0 else None,
                    'median_retention':float(np.median(a[:,8]))})
        write_json(out/'recovery.json',{'rows':rows,'draws_per_case':draws,
            'generator':'H2 quality-paired control fields; shared calendar moving blocks and spatial block donors; H1 shared burst order and source secular removed, deterministic target order reinserted at target order; screen rerun after injection',
            'coverage':'frozen observed coefficient ellipses; nested synthetic uncertainty separately reported',
            'primary_truth':'exact noiseless Msf-only regional projection at that draw support, avoiding a varying-geometry approximation',
            'M2_alias':'horizontal velocity amplitudes .03/.20 m/day, not displacement; exact six-day alias about /28.53'})
        write_json(out/'nested-calibration.json',{'rows':nested,'datasets_per_case_per_region':nested_datasets,'resamples_per_dataset':nested_draws,
            'method':'recompute covariance and centered bootstrap radius inside each synthetic dataset, rerunning screen in each temporal/spatial resample; primary azimuth estimator only',
            'F_reference':'Hotelling-style F(2,B-2) recorded for empirical comparison, no exact distribution claim',
            'selection':'null first draws; benchmark equally spaced draw indices across the full phase cycle',
            'limitation':'small subset: per-region null rate and coverage have substantial binomial uncertainty'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--phase',choices=['preflight','observed','scan','recovery','sensitivity','all'],default='all')
    p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--simulations',type=int,default=200)
    p.add_argument('--nested-datasets',type=int,default=20);p.add_argument('--nested-bootstrap',type=int,default=199)
    p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args();out=a.root/'2017-order-set';out.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    recipe={'primary':'direct satellite-azimuth Msf (m/day), four original targets; Holm .05; controls negative controls',
        'variants':VARIANTS,'screen':'peak .08 and acquisition-purged order-aware secular residual <=2/5 m; source support >=20 observed, >=8 per resample; rerun in resamples',
        'order':'one azimuth displacement per owning burst, sign B_later - B_earlier',
        'phase_origin':'2000-01-01 UTC; C cos + S sin; phase atan2(S,C)',
        'calendar':'complete 60-pair scene endpoints for labels only; owning-burst endpoints for all fits',
        'period_days':PERIOD,'noise':'H1 control covariance after SET and order; H2 paired quality bank',
        'seed':a.seed,'observed_resamples':a.bootstrap,'simulations_per_case':a.simulations,
        'nested_datasets_per_case_region':a.nested_datasets,'nested_resamples':a.nested_bootstrap,
        'corrections_sha256':digest(out/'corrections.json'),'status':'2017 exploratory; prospective 2018 freeze candidate, no 2018 data examined'}
    recipe_file=out/'recipe.json'
    if recipe_file.exists() and json.loads(recipe_file.read_text())!=json.loads(json.dumps(recipe)):
        raise ValueError('Recipe changed: use a separate experiment directory')
    if not recipe_file.exists():write_json(recipe_file,recipe)
    start=time.time();fields=load(a.root,out);cov=calibrate(fields,out);state=build_state(fields,cov)
    if a.phase=='preflight':
        f=fields[0];d=state[f['id']]['designs']['primary_azimuth'];s=time.time()
        for _ in range(100):order_screen(f);d.fit(f['clean'])
        logging.info('100 primary screen+fit seconds %.3f',time.time()-s)
        for f in fields:
            logging.info('%s primary cs %s, order %s',f['id'],state[f['id']]['fits']['primary_azimuth']['cs'],state[f['id']]['fits']['primary_azimuth']['order_m'])
    if a.phase in ('observed','all'):observed(fields,cov,state,out,a.bootstrap,a.seed);predictive(fields,cov,state,out)
    if a.phase in ('scan','all'):scans(fields,cov,out,a.seed)
    if a.phase in ('sensitivity','all'):block_sensitivity(fields,cov,out,500,a.seed)
    if a.phase in ('recovery','all'):
        records=json.loads((out/'observed.json').read_text())['rows']
        for r in records:state[r['region']][r['variant']]=r
        recovery(fields,cov,state,out,a.simulations,a.nested_datasets,a.nested_bootstrap,a.seed)
    logging.info('Finished %s in %.1f seconds',a.phase,time.time()-start)

if __name__=='__main__':main()
