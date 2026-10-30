"""2017 radar-space along-flow Msf plus residual vertical M2/O1 experiment.

The ocean-tide and swath corrections stay applied. Observed and synthetic fits
share masks, sources, geometry, noise and random draws across model comparisons.
This isolated experiment changes no production map gates or original outputs.
"""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from glacier_offsets.velocity_basis import harmonic,utc_days
from glacier_offsets.velocity_io import digest,write_json,revision
from glacier_offsets.velocity_fit import false_discovery
from glacier_offsets.velocity_quality import (screen,crossfit_residual,reference_times,quiet_median,
    temporal_counts,cluster_weights,flow_axis,paired_field_draw,moving_indices,secular_residual,purged_folds)
from glacier_offsets.velocity_vertical import VerticalRegionalDesign,vertical_columns

MODELS=('fixed_vertical','azimuth_only','residual_vertical')


def load_fields(root):
    previous=root/'2017-quality-regional';meta=json.loads((previous/'observations.json').read_text())
    if digest(previous/'observations.npz')!=meta['cache_sha256']:raise ValueError('Observation cache changed')
    if list(revision(meta['input']['path']))!=meta['stamp']['revision']:raise ValueError('Frozen input revision changed')
    reviewed=json.loads((root/'regions.reviewed.json').read_text())['regions'];fields=[]
    threshold=json.loads((previous/'screen_validation.json').read_text())['threshold']
    with np.load(previous/'observations.npz') as h:
        for r in reviewed:
            f={k:h[r['id']+'__'+k] for k in meta['keys']}
            clean=screen(f,threshold,crossfit_residual(f))
            supported=clean.sum(1)>=20
            f={k:v[supported] for k,v in f.items()}
            f.update(id=r['id'],role=r['role'],clean=clean[supported])
            # Source-specific secular truth and H2 control banks are frozen once.
            velocity=quiet_median(np.where(f['clean'][...,None],f['data']/(f['late']-f['early'])[...,None],np.nan),axis=1)
            f['secular_truth']=(f['late']-f['early'])[...,None]*velocity[:,None]
            fields.append(f)
    noise=json.loads((previous/'clean_noise.json').read_text())['pooled']
    return fields,np.array(noise['covariance_ra_m2']),threshold


def prediction_score(field,fit,good,channels):
    residual=field['data'][:,:,channels]-fit['prediction']
    weights=cluster_weights(field['group'])[:,None]*good
    result={}
    for j,channel in enumerate(channels):
        result[('range','azimuth')[channel]+'_rms_m']=float(np.sqrt(np.sum(weights*np.nan_to_num(residual[...,j])**2)/np.sum(weights)))
    return result


def observed(fields,cov,draws,out,seed):
    state={};all_rows=[];samples={};vertical_samples={}
    for f in fields:
        u=flow_axis(f,f['clean']) if f['role']=='target' else state[fields[0]['id']]['direction']
        designs={m:VerticalRegionalDesign(f,cov,m) for m in MODELS}
        fits={m:d.fit(f['clean'],u,details=True) for m,d in designs.items()}
        state[f['id']]={'direction':u,'designs':designs,'fits':fits}
        samples[f['id']]={m:[] for m in MODELS};vertical_samples[f['id']]=[]
    rng=np.random.default_rng(np.random.SeedSequence([seed,2]))
    for b in range(draws):
        tc=temporal_counts(rng,*reference_times(fields[0]));axis=None
        for f in fields:
            sw=cluster_weights(f['group'],rng)
            u=flow_axis(f,f['clean'],tc,sw) if f['role']=='target' else axis
            if f['id']=='target_1':axis=u
            for m in MODELS:
                try:
                    fit=state[f['id']]['designs'][m].fit(f['clean'],u,temporal=tc,spatial=sw,details=(m=='residual_vertical'))
                    samples[f['id']][m].append(fit['cs'])
                    if m=='residual_vertical':vertical_samples[f['id']].append(fit['vertical_cs_mean'])
                except (ValueError,np.linalg.LinAlgError):
                    samples[f['id']][m].append([np.nan,np.nan])
                    if m=='residual_vertical':vertical_samples[f['id']].append([np.nan]*4)
        if (b+1)%100==0:logging.info('Paired observed bootstrap %d/%d',b+1,draws)
    for f in fields:
        st=state[f['id']]
        for m in MODELS:
            fit=st['fits'][m];cs=fit['cs'];values=np.array(samples[f['id']][m]);values=values[np.isfinite(values).all(1)]
            delta=values-cs;covariance=np.cov(delta.T);inv=np.linalg.inv(covariance)
            nullstat=np.einsum('ni,ij,nj->n',delta,inv,delta);stat=cs@inv@cs
            row={'region':f['id'],'role':f['role'],'model':m,'sources':len(f['group']),
                 'cs_m_per_day':cs,'amplitude_m_per_day':float(np.linalg.norm(cs)),
                 'phase_deg':float(np.degrees(np.arctan2(cs[1],cs[0]))),
                 'target_condition':fit['condition'],'coefficient_covariance':covariance,
                 'centered_bootstrap_p':float((1+np.sum(nullstat>=stat))/(len(nullstat)+1)),
                 'null_statistic_95':float(np.quantile(nullstat,.95)),
                 'bootstrap_draws':len(values),'amplitude_percentiles':np.quantile(np.linalg.norm(values,axis=1),[.025,.5,.975]),
                 **prediction_score(f,fit,f['clean'],st['designs'][m].channels)}
            if m=='residual_vertical':
                v=fit['vertical_cs_mean'];vs=np.array(vertical_samples[f['id']]);vs=vs[np.isfinite(vs).all(1)]
                row['regional_mean_vertical_cs_m']=v
                row['regional_mean_M2_amplitude_m']=float(np.linalg.norm(v[:2]));row['regional_mean_O1_amplitude_m']=float(np.linalg.norm(v[2:]))
                row['vertical_amplitude_percentiles_m']={name:np.quantile(np.linalg.norm(vs[:,sl],axis=1),[.025,.5,.975]) for name,sl in [('M2',slice(0,2)),('O1',slice(2,4))]}
            st[m]={'cov':covariance,'cutoff':row['null_statistic_95'],'row':row}
            all_rows.append(row)
    flags=false_discovery(np.array([r['centered_bootstrap_p'] for r in all_rows]))
    for r,flag in zip(all_rows,flags):r['BY_q05']=bool(flag)
    comparison=[]
    for f in fields:
        a=np.array(samples[f['id']]['residual_vertical']);b=np.array(samples[f['id']]['azimuth_only'])
        delta=a-b;delta=delta[np.isfinite(delta).all(1)]
        comparison.append({'region':f['id'],'residual_vertical_minus_azimuth_cs_95':np.quantile(delta,[.025,.975],axis=0),
            'amplitude_difference_95':np.quantile(np.linalg.norm(a,axis=1)-np.linalg.norm(b,axis=1),[.025,.975])})
    write_json(out/'observed.json',{'rows':all_rows,'paired_comparisons':comparison,
        'bootstrap':'whole-field 30-day calendar blocks, physical 3-km spatial blocks, reassessed flow axis; screen fixed',
        'multiplicity':'BY q=.05 over all six regions and three model variants',
        'vertical_interpretation':'amplitude of spatially block-balanced mean source-specific coefficients, in residual height metres; percentile ranges are descriptive'})
    np.savez_compressed(out/'bootstrap.npz',**{rid+'__'+m:np.array(v) for rid,records in samples.items() for m,v in records.items()},
        **{rid+'__vertical':np.array(v) for rid,v in vertical_samples.items()})
    return state,all_rows


def predictive(fields,cov,out):
    records=[]
    for fold,(train,test) in enumerate(purged_folds(*reference_times(fields[0]))):
        axis=None
        for f in fields:
            trainmask=f['clean']&train[None]
            # Exactly the same source/test support for all models.
            available=(trainmask.sum(1)>=12)
            trainmask &= available[:,None]
            testmask=f['clean']&test[None]&available[:,None]
            if not testmask.any():continue
            u=flow_axis(f,trainmask) if f['role']=='target' else axis
            if f['id']=='target_1':axis=u
            for m in MODELS:
                d=VerticalRegionalDesign(f,cov,m)
                fit=d.fit(trainmask,u,details=True)
                records.append({'fold':fold,'region':f['id'],'model':m,'test_observations':int(testmask.sum()),
                    **prediction_score(f,fit,testmask,d.channels)})
    write_json(out/'predictive.json',{'rows':records,'method':'six acquisition-purged ~60-day folds; matched screened test observations, >=12 train epochs/source; secular axis trained within fold',
        'noise':'frozen H1 clean control covariance; this is temporal prediction, not a fresh leave-control-region-out calibration'})
    return records


def simulations(fields,cov,threshold,state,draws,out,seed):
    cases=[('null',0,0,0,0),('benchmark',.1,0,0,0),
           ('null_vertical10',0,.1,0,0),('benchmark_vertical10',.1,.1,0,0),
           ('null_vertical20',0,.2,0,0),('benchmark_vertical20',.1,.2,0,0),
           ('mismatch_minus10',.1,.2,-10,0),('mismatch_plus10',.1,.2,10,0),
           ('mismatch_minus20',.1,.2,-20,0),('mismatch_plus20',.1,.2,20,0),
           ('crossflow_quadrature20',.1,.2,0,.2)]
    controls=[f for f in fields if f['role']=='control'];banks=[]
    split=float(utc_days('2017-07-01'))
    for f in controls:
        early,late=reference_times(f);first=late<split;half=np.flatnonzero(early>=split)
        endpoints=np.unique(np.floor(np.r_[early[half],late[half]]));endpoints=endpoints[np.isfinite(endpoints)]
        first &= ~np.isin(np.floor(early),endpoints)&~np.isin(np.floor(late),endpoints)
        residual=secular_residual(f['data'],f['late']-f['early'],f['valid'],first)
        center=quiet_median(np.where(f['valid'][...,None],residual,np.nan)[:,half],axis=1)
        banks.append(({**f,'residual':residual-center[:,None]},half))
    all_rows=[]
    for name,amplitude,vertical_fraction,mismatch,crossfraction in cases:
        file=out/(name+'.npz')
        if file.exists():
            logging.info('Reusing completed simulation case %s',name)
            with np.load(file) as a:values=a['metrics'];retention=a['retention']
        else:
            values=np.full((draws,len(fields),len(MODELS),4),np.nan);retention=np.full((draws,len(fields)),np.nan)
            rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
            for b in range(draws):
                bank,half=banks[int(rng.integers(len(banks)))];nt=fields[0]['valid'].shape[1]
                ti=half[moving_indices(rng,len(half),5)]
                ti=np.r_[ti,half[moving_indices(rng,len(half),5)]]
                ti=np.resize(ti,nt)
                for fi,f in enumerate(fields):
                    base=paired_field_draw(rng,bank,f,'paired_quality',time_index=ti)
                    phi=2*np.pi*(b+rng.random())/draws
                    vp=rng.uniform(0,2*np.pi,size=2)
                    cs=np.array([np.cos(phi),np.sin(phi)])
                    u=state[f['id']]['direction'];normal=np.array([-u[1],u[0]])
                    angle=np.radians(mismatch);direction=u*np.cos(angle)+normal*np.sin(angle)
                    hcoef=amplitude*(np.outer(direction,cs)+crossfraction*np.outer(normal,[-cs[1],cs[0]]))
                    en=np.einsum('stj,ij->sti',harmonic(f['early'],f['late']),hcoef)
                    hs=np.einsum('sci,sti->stc',f['geometry'][...,:2],en)
                    vertical=np.r_[.6*vertical_fraction*np.array([np.cos(vp[0]),np.sin(vp[0])]),
                                   .2*vertical_fraction*np.array([np.cos(vp[1]),np.sin(vp[1])])]
                    vs=np.einsum('stj,j->st',vertical_columns(f['early'],f['late']),vertical)[...,None]*f['geometry'][:,None,:,2]
                    synthetic={**base,'data':base['data']+f['secular_truth']+hs+vs}
                    good=screen(synthetic,threshold)
                    good &= (good.sum(1)>=8)[:,None]
                    retention[b,fi]=good.sum()/max(f['valid'].sum(),1)
                    fit_axis=flow_axis(synthetic,good) if f['role']=='target' else u
                    truth=fit_axis@hcoef
                    for mi,m in enumerate(MODELS):
                        try:
                            fitted=state[f['id']]['designs'][m].fit(good,fit_axis,data=synthetic['data'])['cs']
                            inv=np.linalg.inv(state[f['id']][m]['cov']);cutoff=state[f['id']][m]['cutoff']
                            delta=fitted-truth
                            ae=abs(np.linalg.norm(fitted)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-12)
                            phase_error=abs(np.degrees(np.angle(np.exp(1j*(np.arctan2(fitted[1],fitted[0])-np.arctan2(truth[1],truth[0]))))))
                            values[b,fi,mi]=[ae,phase_error,float(delta@inv@delta<=cutoff),float(fitted@inv@fitted>cutoff)]
                        except (ValueError,np.linalg.LinAlgError):pass
                if (b+1)%50==0:logging.info('Simulation %s: %d/%d',name,b+1,draws)
            np.savez_compressed(file,metrics=values,retention=retention)
        for fi,f in enumerate(fields):
            for mi,m in enumerate(MODELS):
                data=values[:,fi,mi];data=data[np.isfinite(data).all(1)]
                all_rows.append({'case':name,'horizontal_amplitude':amplitude,'vertical_fraction':vertical_fraction,
                    'M2_height_amplitude_m':.6*vertical_fraction,'O1_height_amplitude_m':.2*vertical_fraction,
                    'mismatch_deg':mismatch,'crossflow_fraction':crossfraction,'region':f['id'],'model':m,
                    'draws':len(data),'median_amplitude_error':float(np.median(data[:,0])) if amplitude else None,
                    'median_phase_error_deg':float(np.median(data[:,1])) if amplitude else None,
                    'coefficient_coverage':float(np.mean(data[:,2])) if len(data) else None,
                    'null_false_positive_rate':float(np.mean(data[:,3])) if not amplitude and len(data) else None,
                    'median_retention':float(np.nanmedian(retention[:,fi]))})
        write_json(out/'recovery.json',{'rows':all_rows,'noise':'quality-paired H2 control fields with H1 secular predictions; shared field temporal/spatial resampling',
            'uncertainty':'frozen observed-data bootstrap coefficient ellipses; independent simulation stream',
            'phases':'stratified uniform Msf, random vertical constituent phases; paired across models and cases',
            'screen':'repeated after injection, including the vertical displacement; no injected vertical signal removed during noise generation'})
    return all_rows


def report(out,rows,cv,recovery):
    lines=['# 2017 residual-vertical M2/O1 plus along-flow Msf experiment','',
        'The swath and ocean-tide corrections remain applied. Free vertical coefficients are residual displacement-height harmonics, not total ocean-tide amplitudes. '
        'Each source has independent EN secular terms and M2/O1 vertical coefficients; horizontal Msf is shared regionally along a robust secular axis. '
        'No secular vertical velocity or free horizontal M2/O1 is fitted.','',
        'All three models use matched screened sources with >=20 accepted observed pairs. In resampling/fitting, sources with fewer than eight distinct selected epochs contribute no weight. '
        'The 30-day field-wide temporal and 3-km spatial bootstrap reassesses flow direction; the observed-data screen is fixed. '
        'Synthetic tests repeat screening after signal/noise injection.','',
        '## Observed horizontal coefficients','',
        '| Region | Fixed vertical joint A | Azimuth-only A | Residual vertical joint A | New-model p | Residual M2 height | Residual O1 height |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for rid in dict.fromkeys(r['region'] for r in rows):
        old,az,new=[next(r for r in rows if r['region']==rid and r['model']==m) for m in MODELS]
        lines.append(f"| {rid} | {old['amplitude_m_per_day']:.3f} | {az['amplitude_m_per_day']:.3f} | {new['amplitude_m_per_day']:.3f} | {new['centered_bootstrap_p']:.4f} | {new['regional_mean_M2_amplitude_m']:.3f} | {new['regional_mean_O1_amplitude_m']:.3f} |")
    lines+=['','Horizontal amplitudes are m/day; vertical heights are metres. Vertical amplitudes are norms of spatially balanced mean C/S coefficients, not averages of nonnegative per-source amplitudes. '
        'Observed p-values use centered coefficient bootstrap draws. BY q=.05 adjustment includes all 18 model/region tests. '
        'Bootstrap percentile amplitude ranges are descriptive and do not by themselves establish detection.','',
        '## Predictive checks','',
        '| Region | Fixed vertical held-out range RMS | Residual vertical held-out range RMS |',
        '|---|---:|---:|']
    for rid in dict.fromkeys(r['region'] for r in rows):
        vals=[]
        for m in ('fixed_vertical','residual_vertical'):
            v=[r['range_rms_m']**2 for r in cv if r['region']==rid and r['model']==m]
            vals.append(np.sqrt(np.mean(v)))
        lines.append(f'| {rid} | {vals[0]:.3f} m | {vals[1]:.3f} m |')
    lines+=['','Predictive values equally weight usable temporal folds. The temporal folds purge shared acquisitions and use the same screened test support. '
        'Noise covariance stays frozen from the earlier H1 clean-control calibration; this check does not claim a fresh independent calibration.','',
        '## Recovery','',
        '| Region | Case | Model | Amplitude error | Phase error | Coverage | Null FPR |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in recovery:
        if r['case'] not in ('benchmark','benchmark_vertical20','null_vertical20') or not r['region'].startswith('target'):continue
        ae='—' if r['median_amplitude_error'] is None else f"{100*r['median_amplitude_error']:.1f}%"
        pe='—' if r['median_phase_error_deg'] is None else f"{r['median_phase_error_deg']:.1f}°"
        fp='—' if r['null_false_positive_rate'] is None else f"{100*r['null_false_positive_rate']:.1f}%"
        lines.append(f"| {r['region']} | {r['case']} | {r['model']} | {ae} | {pe} | {100*r['coefficient_coverage']:.1f}% | {fp} |")
    lines+=['','The 20% vertical case adds M2 height amplitude 0.12 m and O1 height amplitude 0.04 m. '
        'All 11 cases, including +/-10/20-degree directional mismatch and a 20% quadrature cross-flow component, are in recovery.json. '
        'These tests assess empirical-noise transfer and model sensitivity. Two controls cannot establish all ice noise/systematics, and this already-inspected record remains exploratory.','',
        '## Limits','',
        'Freely estimating residual vertical motion can reduce horizontal bias from an inaccurate correction, but cannot add horizontal information relative to a perfectly known correction. '
        'The range M2/Msf alias makes the new horizontal estimate rely heavily on azimuth. '
        'Residual vertical harmonics can absorb other errors with matching sampled patterns; their interpretation as physical vertical tides is conditional. '
        'Cross-flow horizontal Msf, horizontal M2/O1, missing tidal constituents and calibration errors remain possible biases. '
        'Correction validation and production-map eligibility are unchanged. No 2018 observations, calibration coefficients or original input files were modified.','',
        '![Horizontal and residual vertical coefficients](coefficients.png)','',
        'Files: [observed fits](observed.json), [predictive checks](predictive.json), [recovery](recovery.json), [recipe](recipe.json), [verification](verification.json).','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def plots(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    names=list(dict.fromkeys(r['region'] for r in rows));colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for j,rid in enumerate(names):
        for mi,m in enumerate(MODELS):
            r=next(r for r in rows if r['region']==rid and r['model']==m);cs=np.array(r['cs_m_per_day'])
            axes[0].plot(*cs,marker=('o','x','s')[mi],color=colors[j],ls='none',label=rid+' '+m if j==0 else None)
        r=next(r for r in rows if r['region']==rid and r['model']=='residual_vertical')
        cs=np.array(r['cs_m_per_day']);value,vector=np.linalg.eigh(r['coefficient_covariance']);angle=np.degrees(np.arctan2(vector[1,-1],vector[0,-1]))
        axes[0].add_patch(Ellipse(cs,2*np.sqrt(value[-1]),2*np.sqrt(value[0]),angle=angle,fill=False,color=colors[j],alpha=.7))
        v=np.array(r['regional_mean_vertical_cs_m'])
        axes[1].plot(*v[:2],'o',color=colors[j],label=rid);axes[2].plot(*v[2:],'o',color=colors[j])
    for ax,title,units in zip(axes,('Horizontal Msf; new-model 1-SD ellipses','Residual vertical M2','Residual vertical O1'),('m/day','m','m')):
        ax.axhline(0,color='.7',lw=.6);ax.axvline(0,color='.7',lw=.6);ax.set_title(title,fontsize=10)
        ax.set_xlabel('Cosine coefficient ('+units+')');ax.set_ylabel('Sine coefficient ('+units+')')
    axes[0].legend(fontsize=7);axes[1].legend(fontsize=7)
    fig.savefig(out/'coefficients.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--output',type=Path);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--validation',type=int,default=200);p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args();root=a.root.resolve();out=(a.output or root/'2017-residual-vertical').resolve();out.mkdir(exist_ok=True,parents=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    recipe={'bootstrap':a.bootstrap,'validation':a.validation,'seed':a.seed,'scope':'2017 only, residual vertical motion on tide-corrected radar observations',
        'models':MODELS,'production_maps_allowed':False,'software':{str(v):digest(v) for v in (Path(__file__),Path('src/glacier_offsets/velocity_vertical.py'),Path('src/glacier_offsets/velocity_quality.py'))},
        'inputs':{str(v):digest(v) for v in (root/'2017-quality-regional/observations.npz',root/'2017-quality-regional/clean_noise.json',root/'2017-quality-regional/screen_validation.json')}}
    recipe=json.loads(json.dumps(recipe))
    if (out/'recipe.json').exists() and json.loads((out/'recipe.json').read_text())!=recipe:raise ValueError('Recipe changed; use a new output directory')
    write_json(out/'recipe.json',recipe);start=time.monotonic()
    fields,cov,threshold=load_fields(root)
    state,rows=observed(fields,cov,a.bootstrap,out,a.seed)
    cv=predictive(fields,cov,out)
    recovery=simulations(fields,cov,threshold,state,a.validation,out,a.seed)
    plots(out,rows);report(out,rows,cv,recovery)
    success=all(r['draws']==a.validation for r in recovery) and all(r['bootstrap_draws']==a.bootstrap for r in rows)
    write_json(out/'verification.json',{'status':'passed' if success else 'incomplete','validation_rows':len(recovery),
        'draws_per_row':sorted({r['draws'] for r in recovery}),'bootstrap_rows':len(rows),
        'bootstrap_draws':sorted({r['bootstrap_draws'] for r in rows}),'production_maps_generated':False,
        'source_fingerprints_current':all(digest(path)==sha for path,sha in recipe['software'].items())})
    write_json(out/'completion.json',{'status':'complete' if success else 'incomplete','elapsed_seconds':time.monotonic()-start,
        'products':{v.name:digest(v) for v in out.iterdir() if v.suffix in ('.json','.npz','.md','.png') and v.name!='completion.json'}})
    logging.info('Finished residual-vertical experiment in %.1f seconds',time.monotonic()-start)

if __name__=='__main__':main()
