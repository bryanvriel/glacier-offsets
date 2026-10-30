"""Additional 50%-reference vertical stress on the completed observed fit.

M2=0.30 m and O1=0.10 m exceed the regional mean residual amplitudes inferred in
this experiment. These are stress scenarios, not independent evidence that the
fitted residuals are physical vertical tides.
"""
import json
import logging
from pathlib import Path
import numpy as np
from glacier_offsets.velocity_basis import harmonic,utc_days
from glacier_offsets.velocity_io import write_json,digest
from glacier_offsets.velocity_quality import (reference_times,secular_residual,quiet_median,
    moving_indices,paired_field_draw,screen,flow_axis)
from glacier_offsets.velocity_vertical import VerticalRegionalDesign,vertical_columns
from tools.run_residual_vertical_pilot import load_fields,MODELS


def run(root,draws=200):
    parent=root/'2017-residual-vertical';out=parent/'stress50';out.mkdir(exist_ok=True);fields,cov,threshold=load_fields(root)
    records=json.loads((parent/'observed.json').read_text())['rows'];states={};axis=None
    for f in fields:
        u=flow_axis(f,f['clean']) if f['role']=='target' else axis
        if f['id']=='target_1':axis=u
        states[f['id']]={'direction':u,'designs':{m:VerticalRegionalDesign(f,cov,m) for m in MODELS},
            'stats':{m:next(v for v in records if v['region']==f['id'] and v['model']==m) for m in MODELS}}
    banks=[];split=float(utc_days('2017-07-01'))
    for f in fields:
        if f['role']!='control':continue
        e,l=reference_times(f);first=l<split;half=np.flatnonzero(e>=split)
        endpoints=np.unique(np.floor(np.r_[e[half],l[half]]));endpoints=endpoints[np.isfinite(endpoints)]
        first &= ~np.isin(np.floor(e),endpoints)&~np.isin(np.floor(l),endpoints)
        residual=secular_residual(f['data'],f['late']-f['early'],f['valid'],first)
        center=quiet_median(np.where(f['valid'][...,None],residual,np.nan)[:,half],axis=1)
        banks.append(({**f,'residual':residual-center[:,None]},half))
    rows=[]
    for amplitude in (0.,.1):
        rng=np.random.default_rng(np.random.SeedSequence([12345,1]));values=np.full((draws,len(fields),3,4),np.nan)
        for b in range(draws):
            bank,half=banks[int(rng.integers(len(banks)))];nt=fields[0]['valid'].shape[1]
            ti=np.r_[half[moving_indices(rng,len(half),5)],half[moving_indices(rng,len(half),5)]];ti=np.resize(ti,nt)
            for fi,f in enumerate(fields):
                base=paired_field_draw(rng,bank,f,'paired_quality',time_index=ti)
                phi=2*np.pi*(b+rng.random())/draws;vp=rng.uniform(0,2*np.pi,size=2)
                st=states[f['id']];u=st['direction'];cs=amplitude*np.array([np.cos(phi),np.sin(phi)])
                horizontal=np.einsum('stj,j->st',harmonic(f['early'],f['late']),cs)[...,None]*np.einsum('sci,i->sc',f['geometry'][...,:2],u)[:,None]
                z=np.r_[.3*np.array([np.cos(vp[0]),np.sin(vp[0])]),.1*np.array([np.cos(vp[1]),np.sin(vp[1])])]
                vertical=np.einsum('stj,j->st',vertical_columns(f['early'],f['late']),z)[...,None]*f['geometry'][:,None,:,2]
                synthetic={**base,'data':base['data']+f['secular_truth']+horizontal+vertical}
                good=screen(synthetic,threshold);good &=(good.sum(1)>=8)[:,None]
                fit_axis=flow_axis(synthetic,good) if f['role']=='target' else u;truth=cs*float(fit_axis@u)
                for mi,m in enumerate(MODELS):
                    try:
                        fit=st['designs'][m].fit(good,fit_axis,data=synthetic['data'])['cs'];stats=st['stats'][m]
                        inv=np.linalg.inv(stats['coefficient_covariance']);cut=stats['null_statistic_95'];delta=fit-truth
                        ae=abs(np.linalg.norm(fit)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-12)
                        pe=abs(np.degrees(np.angle(np.exp(1j*(np.arctan2(fit[1],fit[0])-np.arctan2(truth[1],truth[0]))))))
                        values[b,fi,mi]=[ae,pe,float(delta@inv@delta<=cut),float(fit@inv@fit>cut)]
                    except (ValueError,np.linalg.LinAlgError):pass
            if (b+1)%50==0:logging.info('50%% vertical stress, horizontal %.1f: %d/%d',amplitude,b+1,draws)
        np.savez_compressed(out/f'stress50_horizontal{amplitude}.npz',metrics=values)
        for fi,f in enumerate(fields):
            for mi,m in enumerate(MODELS):
                v=values[:,fi,mi];v=v[np.isfinite(v).all(1)]
                rows.append({'region':f['id'],'model':m,'horizontal_amplitude':amplitude,'draws':len(v),
                    'median_amplitude_error':float(np.median(v[:,0])) if amplitude else None,
                    'median_phase_error_deg':float(np.median(v[:,1])) if amplitude else None,
                    'coefficient_coverage':float(np.mean(v[:,2])),
                    'null_false_positive_rate':float(np.mean(v[:,3])) if not amplitude else None})
    write_json(out/'vertical_stress50.json',{'rows':rows,'M2_height_amplitude_m':.3,'O1_height_amplitude_m':.1,
        'source_sha256':digest(__file__),'observed_input_sha256':digest(parent/'observed.json'),
        'scope':'same matched-support, quality-paired noise, screening and frozen bootstrap ellipse tests; larger vertical stress',
        'seed_sequence':[12345,1]})

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    run(Path('.runtime/velocity-recovery-larsen').resolve())
