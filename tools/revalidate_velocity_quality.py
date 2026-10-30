"""Run the isolated 2017 quality, dependent-noise and regional validation audit.

Run with PYTHONPATH=src and the existing ice interpreter. No processing products
or original pilot products are modified. All choices are recorded before fits.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import time
import warnings
import numpy as np
from glacier_offsets.velocity_basis import PERIOD, harmonic, utc_days
from glacier_offsets.velocity_io import Collection, digest, fingerprint, write_json, revision
from glacier_offsets.velocity_fit import calibrate_noise, false_discovery
from glacier_offsets.velocity_quality import (quiet_median, purged_folds, secular_residual,
    crossfit_residual, screen, temporal_counts, cluster_weights, flow_axis, RegionalDesign,
    paired_field_draw, moving_indices, reference_times)

CHANNELS=('joint','range','azimuth')


def extract(input_path,regions,out):
    cache=out/'observations.npz';meta=out/'observations.json'
    stamp={'revision':revision(input_path),'regions':fingerprint(regions)}
    if cache.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved['stamp']==json.loads(json.dumps(stamp)):
            if digest(cache)!=saved['cache_sha256']:
                raise ValueError('Observation cache checksum mismatch')
            with np.load(cache) as h:
                fields=[{key:h[r['id']+'__'+key] for key in saved['keys']}|{'id':r['id'],'role':r['role']} for r in regions]
            return fields,saved['input']
        raise ValueError('Observation cache is stale; use a new output directory')
    fields=[];used=set()
    with Collection([input_path],start='2017-01-01',end='2018-01-01') as c:
        for r in regions:
            w,s,e,n=r['bounds']
            rows=np.flatnonzero((c.y>=s)&(c.y<=n));cols=np.flatnonzero((c.x>=w)&(c.x<=e))
            section=(slice(rows.min(),rows.max()+1),slice(cols.min(),cols.max()+1))
            tile=c.tile(section)
            ids,first=np.unique(c.association[section].ravel(),return_index=True)
            take=(ids>=0)&~np.isin(ids,list(used));ids,first=ids[take],first[take]
            used.update(ids.tolist())
            shape=tile['valid'].shape;nt=shape[0]
            a={k:tile[k].reshape(nt,-1)[:,first].T for k in ('early','late','valid','snr','peak')}
            geometry=tile['geometry'].reshape(-1,2,3)[first]
            en=tile['data'].reshape(nt,-1,2)[:,first].transpose(1,0,2)
            a['data']=np.einsum('sij,stj->sti',geometry[...,:2],en)
            a['geometry']=geometry;a['source_id']=ids
            rr,cc=np.unravel_index(first,shape[1:]);rr+=rows.min();cc+=cols.min()
            a['group']=np.array([c.spatial_block(i,j) for i,j in zip(rr,cc)])
            a.update(row=rr,col=cc)
            a['valid'] &= np.isfinite(a['data']).all(-1)
            # Keep all deduplicated sources with at least 20 available observations,
            # independently of quality/Msf. Masks govern all subsequent use.
            support=a['valid'].sum(1)>=20
            a={k:v[support] for k,v in a.items()}
            fields.append(a|{'id':r['id'],'role':r['role']})
            logging.info('Extracted %s: %d unique supported sources, %d spatial blocks',r['id'],len(a['group']),len(np.unique(a['group'])))
        source=c.sources[0];c.check_revisions()
    keys=list(a)
    np.savez_compressed(cache,**{f['id']+'__'+k:f[k] for f in fields for k in keys})
    write_json(meta,{'stamp':stamp,'keys':keys,'input':source,'cache_sha256':digest(cache)})
    return fields,source


def screen_statistics(field,residual,mask):
    available=field['valid'] & np.isfinite(residual).all(-1)
    proxy=available & ((abs(residual[...,0])>2)|(abs(residual[...,1])>5))
    normal=available & ~proxy
    def fraction(a,b):return float(np.sum(a)/max(np.sum(b),1))
    accepted=mask & available
    scatter=[]
    for axis in (0,1):
        x=residual[...,axis][accepted]
        scatter.append(float(1.4826*np.median(abs(x-np.median(x)))) if len(x) else None)
    return {'available':int(available.sum()),'retained':int(accepted.sum()),
            'retention':fraction(accepted,available), 'gross_proxy_fraction':fraction(proxy,available),
            'gross_proxy_recall':fraction(proxy & ~mask,proxy),
            'clean_proxy_retention':fraction(normal & mask,normal),
            'retained_gross_proxy_count':int(np.sum(proxy & mask)), 'robust_scatter_ra_m':scatter}


def select_screen(controls,exclude=None):
    rows=[]
    for threshold in (.04,.06,.08,.10,.15):
        values=[]
        for f in controls:
            if f['id']==exclude:continue
            first=reference_times(f)[1]<float(utc_days('2017-07-01'))
            residual=crossfit_residual(f,allowed=first)
            quality=screen(f,threshold,supplement=False)
            stats=screen_statistics(f,residual,quality & first[None])
            # Proxy diagnostic; no claim of classifier ground truth. Equal ROI weight.
            risk=1-stats['clean_proxy_retention']+5*(1-stats['gross_proxy_recall'])
            values.append(risk)
        rows.append({'threshold':threshold,'mean_control_proxy_risk':float(np.mean(values))})
    chosen=min(rows,key=lambda row:(row['mean_control_proxy_risk'],row['threshold']))['threshold']
    return chosen,rows


def assess_screen(fields,out):
    controls=[f for f in fields if f['role']=='control']
    threshold,selection=select_screen(controls)
    report={'threshold':threshold,'selection':selection,'selection_data':'first-half controls only, acquisition-purged residuals',
            'risk':'clean-proxy rejection + 5 * missed-gross-proxy rate, equally weighted control regions',
            'proxy_definition':'absolute cross-fitted radar residual >2 m range OR >5 m azimuth; not ground-truth matches',
            'supplement_limits_ra_m':[2,5],'heldout':[],'full_year':[]}
    for f in fields:
        f['residual']=crossfit_residual(f)
        f['clean']=screen(f,threshold,residual=f['residual'])
        report['full_year'].append({'region':f['id'],**screen_statistics(f,f['residual'],f['clean']),
            'quality_only':screen_statistics(f,f['residual'],screen(f,threshold,supplement=False)),
            'unique_sources':len(f['group']),'spatial_blocks':len(np.unique(f['group'])),
            'retained_pairs_source_quantiles':np.quantile(f['clean'].sum(1),[0,.25,.5,.75,1]),
            'sources_at_least_40_pairs':int(np.sum(f['clean'].sum(1)>=40))})
    for f in controls:
        first=reference_times(f)[1]<float(utc_days('2017-07-01'))
        last=reference_times(f)[0]>=float(utc_days('2017-07-01'))
        # A source's secular baseline uses H1 only; no H2 endpoint in training.
        endpoints=np.unique(np.floor(np.r_[reference_times(f)[0][last],reference_times(f)[1][last]]))
        first &= ~np.isin(np.floor(reference_times(f)[0]),endpoints)&~np.isin(np.floor(reference_times(f)[1]),endpoints)
        residual=secular_residual(f['data'],f['late']-f['early'],f['valid'],first)
        heldout=np.where(last[None,:,None],residual,np.nan)
        for excluded in (None,f['id']):
            chosen,_=select_screen(controls,exclude=excluded)
            quality=screen(f,chosen,supplement=False)
            supplemented=screen(f,chosen,residual=residual)
            report['heldout'].append({'region':f['id'],'excluded_control_region':excluded,'threshold':chosen,
                'unscreened':screen_statistics(f,heldout,f['valid']),
                'quality_only':screen_statistics(f,heldout,quality),
                'supplemented':screen_statistics(f,heldout,supplemented),
                'training_acquisition_days':np.unique(np.floor(np.r_[reference_times(f)[0][first],reference_times(f)[1][first]])),
                'test_acquisition_days':endpoints})
        f['validation_residual']=residual
    write_json(out/'screen_validation.json',report)
    return threshold,report


def noise_calibration(fields,threshold,out):
    def learn(excluded=None):
        residual=[];snr=[];peak=[];group=[]
        for f in fields:
            if f['role']!='control' or f['id']==excluded:continue
            first=reference_times(f)[1]<float(utc_days('2017-07-01'))
            r=crossfit_residual(f,allowed=first)
            take=screen(f,threshold,residual=r)
            residual.extend(r[take]);snr.extend(f['snr'][take]);peak.extend(f['peak'][take])
            group.extend(np.broadcast_to((f['id']+':'+f['group'].astype(object))[:,None],take.shape)[take])
        return calibrate_noise(np.array(residual),np.array(snr),np.array(peak),np.array(group))
    model=learn();report={'training':'first-half screened cross-fitted controls only','pooled':model,
                         'leave_region_out':{f['id']:learn(f['id']) for f in fields if f['role']=='control'}}
    write_json(out/'clean_noise.json',report)
    return np.asarray(model['covariance_ra_m2']),report


def regional_fits(fields,sigma,draws,out,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,2]));rows=[];state={}
    bootstrap={f['id']:{c:[] for c in CHANNELS} for f in fields}
    for f in fields:
        # Controls use the target_1 axis as a declared projection convention,
        # not a direction inferred from near-zero bedrock motion.
        direction=flow_axis(f,f['clean']) if f['role']=='target' else flow_axis(fields[0],fields[0]['clean'])
        designs={c:RegionalDesign(f,f['clean'],sigma,c) for c in CHANNELS}
        fits={c:d.fit(direction) for c,d in designs.items()}
        state[f['id']]={'direction':direction,'designs':designs,'fits':fits}
    for b in range(draws):
        tc=temporal_counts(rng,*reference_times(fields[0]))
        target_axis=None
        for f in fields:
            sw=cluster_weights(f['group'],rng)
            if f['role']=='target':
                direction=flow_axis(f,f['clean'],tc,sw)
                if f['id']=='target_1':target_axis=direction
            else:direction=target_axis
            for c in CHANNELS:
                try:cs=state[f['id']]['designs'][c].fit(direction,tc,sw)['cs']
                except (ValueError,np.linalg.LinAlgError):cs=[np.nan,np.nan]
                bootstrap[f['id']][c].append(cs)
        if (b+1)%100==0:logging.info('Regional paired bootstrap %d/%d',b+1,draws)
    for f in fields:
        for c in CHANNELS:
            fit=state[f['id']]['fits'][c];cs=fit['cs'];samples=np.asarray(bootstrap[f['id']][c])
            samples=samples[np.isfinite(samples).all(1)]
            delta=samples-cs
            cov=np.cov(delta.T);inv=np.linalg.inv(cov)
            nullstat=np.einsum('ni,ij,nj->n',delta,inv,delta)
            stat=float(cs@inv@cs)
            p=float((1+np.sum(nullstat>=stat))/(len(nullstat)+1))
            row={'region':f['id'],'role':f['role'],'channel':c,'cs_m_per_day':cs,'amplitude_m_per_day':float(np.linalg.norm(cs)),
                 'phase_deg':float(np.degrees(np.arctan2(cs[1],cs[0]))),'bootstrap_draws':len(samples),
                 'coefficient_covariance':cov,'amplitude_percentiles':np.quantile(np.linalg.norm(samples,axis=1),[.025,.5,.975]),
                 'centered_bootstrap_p':p,'wald_statistic':stat,'null_statistic_95':float(np.quantile(nullstat,.95)),
                 'direction_en':fit['direction'],'n_sources':fit['n_sources'],'n_epochs':fit['n_epochs'],
                 'spatial_blocks':len(np.unique(f['group'])),'target_rank':fit['rank']}
            rows.append(row)
            state[f['id']][c]={'cov':cov,'cutoff':row['null_statistic_95'],'row':row}
    # One family includes all regions/channels, including controls, rather than
    # claiming separate favorable channels without a multiplicity adjustment.
    discoveries=false_discovery(np.array([r['centered_bootstrap_p'] for r in rows]))
    for r,flag in zip(rows,discoveries):r['BY_q05']=bool(flag)
    write_json(out/'regional_fits.json',{'rows':rows,'bootstrap_scope':'paired 30-day calendar blocks and physical 3-km blocks; flow axis reestimated; fixed cross-fitted screen',
        'limitation':'exploratory conditional bootstrap; residual-screen selection and unmodeled systematic corrections are not fully captured',
        'multiplicity_family':'all six regions and three channels, Benjamini-Yekutieli q=.05'})
    np.savez_compressed(out/'regional_bootstrap.npz',**{rid+'__'+c:np.asarray(v) for rid,a in bootstrap.items() for c,v in a.items()})
    return rows,state


def period_scans(fields,sigma,state,draws,out,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,3]))
    periods=np.unique(np.r_[np.arange(12.,20.001,.05),PERIOD]);rows=[]
    for f in fields:
        st=state[f['id']];u=st['direction'];groups=f['group'];unique=np.unique(groups)
        mid=(reference_times(f)[0]+reference_times(f)[1])/2
        labels=np.where(np.isfinite(mid),np.floor((mid-np.nanmin(mid))/30),-1).astype(int);labs=np.unique(labels[labels>=0])
        # Same draw signs across the field and channels; resample spatial blocks.
        signs=rng.choice([-1.,1.],size=(draws,len(labs)))
        spatial=np.array([np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique)) for _ in range(draws)])
        multiplier=(signs[:,:,None]*spatial[:,None,:]).reshape(draws,-1)
        for channel in CHANNELS:
            fit=st['fits'][channel];des=st['designs'][channel]
            residual=np.nan_to_num(des.yd-fit['prediction_scaled'])
            blocks=[];coeff=[]
            for p in periods:
                d=RegionalDesign(f,f['clean'],sigma,channel,p)
                op=d.operator(u)
                cs=np.einsum('istc,stc->i',op,np.nan_to_num(d.yd));coeff.append(cs)
                contribution=np.einsum('istc,stc->ist',op,residual)
                grouped=np.zeros((2,len(labs),len(unique)))
                for j,g in enumerate(unique):
                    x=contribution[:,groups==g].sum(1)
                    for k,l in enumerate(labs):grouped[:,k,j]=x[:,labels==l].sum(1)
                blocks.append(grouped.reshape(2,-1))
            null=np.einsum('db,pib->dpi',multiplier,np.asarray(blocks))
            cov=np.array([np.cov(null[:,j].T) for j in range(len(periods))]);inverse=np.linalg.pinv(cov,hermitian=True)
            observed=np.einsum('pi,pij,pj->p',coeff,inverse,coeff)
            nullstat=np.einsum('dpi,pij,dpj->dp',null,inverse,null)
            maxima=np.max(nullstat,axis=1);ix=int(np.where(periods==PERIOD)[0][0]);peak=int(np.argmax(observed))
            rows.append({'region':f['id'],'channel':channel,'periods_days':periods,'amplitudes_m_per_day':np.linalg.norm(coeff,axis=1),
                'statistics':observed,'peak_period_days':float(periods[peak]),'Msf_statistic':float(observed[ix]),
                'Msf_scan_adjusted_p':float((1+np.sum(maxima>=observed[ix]))/(draws+1)),
                'peak_scan_adjusted_p':float((1+np.sum(maxima>=observed[peak]))/(draws+1)),
                'scan_max_null_95':float(np.quantile(maxima,.95))})
        logging.info('Finished dependent period scans: %s',f['id'])
    write_json(out/'period_scans.json',{'method':'30-day whole-field wild signs and 3-km cluster resampling of full-Msf residuals; fixed screen/direction',
        'scope':'maximum over 12–20-day grid within each region/channel; no further across-region/channel adjustment',
        'draws':draws,'rows':rows})
    return rows


def simulator_validation(fields,sigma,threshold,state,legacy_noise,draws,out,seed):
    controls=[f for f in fields if f['role']=='control'];rows=[]
    for mode in ('legacy','paired_quality','clean'):
        # Identical donor, spatial/time and phase streams make generator
        # comparisons paired rather than Monte Carlo differences.
        rng=np.random.default_rng(np.random.SeedSequence([seed,1]))
        results={(f['id'],c,a):[] for f in fields for c in CHANNELS for a in (0.,.1)}
        retained={f['id']:[] for f in fields};failure={f['id']:0 for f in fields}
        for b in range(draws):
            donor=controls[int(rng.integers(len(controls)))]
            half=np.flatnonzero(reference_times(donor)[0]>=float(utc_days('2017-07-01')))
            time_index=half[moving_indices(rng,len(half),5)[:]]
            # Tile the held-out half-year block stream to the target record length.
            time_index=np.r_[time_index,half[moving_indices(rng,len(half),5)]][:fields[0]['valid'].shape[1]]
            if len(time_index)<fields[0]['valid'].shape[1]:
                time_index=np.resize(time_index,fields[0]['valid'].shape[1])
            bank={**donor,'residual':donor['validation_residual'].copy()}
            bank['clean']=screen(donor,threshold,residual=donor['validation_residual'])
            # Center each source's H2 noise, without fitting/removing Msf.
            center=quiet_median(np.where(donor['valid'][...,None],bank['residual'],np.nan)[:,half],axis=1)
            bank['residual']-=center[:,None]
            for f in fields:
                base=paired_field_draw(rng,bank,f,mode,legacy_noise,time_index)
                u=state[f['id']]['direction'];h=harmonic(f['early'],f['late'])
                # Uniform stratified phases; same signal/noise for the two amplitude cases.
                phase=2*np.pi*(b+rng.random())/draws
                truth_cs=np.array([np.cos(phase),np.sin(phase)])
                projection=np.einsum('sci,i->sc',f['geometry'][...,:2],u)
                signal=np.einsum('stj,j->st',h,truth_cs)[...,None]*projection[:,None]
                for amplitude in (0.,.1):
                    synthetic={**base,'data':base['data']+amplitude*signal}
                    good=screen(synthetic,threshold)  # repeat response-based screening after injection
                    if amplitude==0:retained[f['id']].append(float(good.sum()/max(f['valid'].sum(),1)))
                    direction=flow_axis({**synthetic,'data':synthetic['data']+f['secular_truth']},good) if f['role']=='target' else u
                    for channel in CHANNELS:
                        try:
                            d=RegionalDesign(synthetic,good,sigma,channel)
                            fit=d.fit(direction);cs=fit['cs']
                            truth=amplitude*truth_cs*float(u@direction)
                            covariance=state[f['id']][channel]['cov'];inverse=np.linalg.inv(covariance)
                            delta=cs-truth;cover=float(delta@inverse@delta<=state[f['id']][channel]['cutoff'])
                            reject=float(cs@inverse@cs>state[f['id']][channel]['cutoff'])
                            ae=abs(np.linalg.norm(cs)-np.linalg.norm(truth))/max(np.linalg.norm(truth),1e-12)
                            pe=abs(np.degrees(np.angle(np.exp(1j*(np.arctan2(cs[1],cs[0])-phase)))))
                            results[(f['id'],channel,amplitude)].append([ae,pe,cover,reject])
                        except (ValueError,np.linalg.LinAlgError):failure[f['id']]+=1
            if (b+1)%25==0:logging.info('Held-out simulator %s: %d/%d',mode,b+1,draws)
        for (rid,channel,amplitude),v in results.items():
            v=np.asarray(v)
            rows.append({'generator':mode,'region':rid,'channel':channel,'amplitude':amplitude,'successful_draws':len(v),
                'median_relative_amplitude_error':float(np.median(v[:,0])) if len(v) and amplitude else None,
                'median_phase_error_deg':float(np.median(v[:,1])) if len(v) and amplitude else None,
                'coverage_of_frozen_observed_bootstrap_ellipse':float(np.mean(v[:,2])) if len(v) else None,
                'null_false_positive_rate':float(np.mean(v[:,3])) if len(v) and amplitude==0 else None,
                'median_retention':float(np.median(retained[rid])),'failed_fits_region':failure[rid]})
        write_json(out/'simulator_validation.json',{'draws':draws,'rows':rows,
            'noise_source':'second-half controls with first-half secular predictions; centered per source, no Msf removal',
            'generators':{'legacy':'quality-standardized residuals with fixed target quality labels',
                'paired_quality':'raw residual, validity, SNR and peak transferred together',
                'clean':'paired_quality restricted to donor clean support'},
            'uncertainty':'tests frozen regional observed-data bootstrap ellipses; not original per-source analytic ellipses',
            'screen':'same cross-fitted screen rerun after every injection; secular motion restored for direction estimation',
            'limitation':'finite two-control empirical noise transfer, held-out dates are exploratory; not physical correction validation'})
    return rows


def make_report(out,recipe,screen_report,sigma,regional,simulations,scans):
    sigma=np.sqrt(np.diag(sigma))
    lines=['# Revised 2017 quality and regional validation','',
        'This is an exploratory validation of observation screening, noise transfer and regional inference. '
        'The original pilot products and production inputs are unchanged. This experiment does not authorize maps.','',
        '## Design','',f"First-half control data selected peak >= {screen_report['threshold']:.2f}. "
        'The supplementary cross-fitted gross-residual limits are 2 m in range and 5 m in azimuth. '
        'They are diagnostic choices, not independently labeled failed matches. '
        'Selection uses no target Msf amplitude. Second-half controls test the screen, with acquisition purging and leave-control-region-out checks.','',
        'All regional fits retain unique sources, physical 3-km blocks, source-specific secular motion, exact source-burst times and local radar geometry. '
        'A shared regional harmonic is fitted after eliminating source secular coefficients. '
        'Controls use the target_1 flow axis as an explicit projection convention.','',
        f'Clean H1 control-calibrated radar sigmas: range {sigma[0]:.3f} m; azimuth {sigma[1]:.3f} m.','',
        '## Screening and retained support','',
        '| Region | Sources | Spatial blocks | Retained observations | Sources with >=40 pairs |',
        '|---|---:|---:|---:|---:|']
    for r in screen_report['full_year']:
        lines.append(f"| {r['region']} | {r['unique_sources']} | {r['spatial_blocks']} | {100*r['retention']:.1f}% | {r['sources_at_least_40_pairs']} |")
    lines+=['','The supplementary residual screen is evaluated as a procedure, not as independent confirmation of its own gross-residual labels. '
        'The quality-only held-out checks provide the less circular check of the quality/failure association.','',
        '## Regional coefficients and conditional uncertainty','',
        '| Region | Channel | Amplitude m/day | Bootstrap percentile range | Centered-bootstrap p | BY q=.05 |',
        '|---|---|---:|---|---:|---|']
    for r in regional:
        lo,_,hi=r['amplitude_percentiles']
        lines.append(f"| {r['region']} | {r['channel']} | {r['amplitude_m_per_day']:.3f} | {lo:.3f}–{hi:.3f} | {r['centered_bootstrap_p']:.4f} | {r['BY_q05']} |")
    lines+=['','Percentile ranges are not calibrated amplitude confidence intervals. Positive lower bounds alone are not detection evidence. '
        'The p-values use centered coefficient bootstrap draws, with 30-day blocks shared by every source and channel and resampled spatial blocks. '
        'The screen is fixed for this observed-data bootstrap; injections below rerun the screen. '
        'BY adjustment includes all 18 region/channel tests. These are exploratory resampling results, not proof that tidal/ionospheric systematics are absent.','',
        '## Held-out control-field injections','',
        '| Generator | Region | Channel | Null FPR | Coefficient coverage | 0.10 amplitude error | Phase error |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in simulations:
        if r['amplitude']!=0 or r['channel']!='azimuth':continue
        benchmark=next(v for v in simulations if v['generator']==r['generator'] and v['region']==r['region'] and v['channel']==r['channel'] and v['amplitude']==.1)
        lines.append(f"| {r['generator']} | {r['region']} | {r['channel']} | {100*r['null_false_positive_rate']:.1f}% | {100*r['coverage_of_frozen_observed_bootstrap_ellipse']:.1f}% | {100*benchmark['median_relative_amplitude_error']:.1f}% | {benchmark['median_phase_error_deg']:.1f}° |")
    lines+=['','All channels and positive-injection coverage are in simulator_validation.json. '
        'Each generator uses H2 control fields, common temporal block indices, and donor spatial blocks. '
        'The paired generator carries quality and failures together; the legacy generator deliberately breaks that association. '
        'The clean generator restricts to accepted donor residuals. '
        'Validation tests the observed regional bootstrap coefficient ellipses, rather than claiming to reproduce the original per-source experiment.','',
        '## Off-frequency checks','',
        '| Region | Channel | Highest-scoring period (days) | Msf p adjusted over scan |',
        '|---|---|---:|---:|']
    for r in scans:
        lines.append(f"| {r['region']} | {r['channel']} | {r['peak_period_days']:.3f} | {r['Msf_scan_adjusted_p']:.4f} |")
    lines+=['','The scan spans 12–20 days at 0.05-day spacing plus exact Msf. '
        'Whole-field 30-day wild signs and spatial-block resampling supply maximum-over-frequency null statistics. '
        'This adjusts the scan within each region/channel; it does not additionally adjust all scans together.','',
        '## Limits','',
        'This study has only two independent control regions. Their transfer to floating ice, seasonal variation, residual-screen selection, '
        'the chosen temporal/spatial block scales, and incomplete tidal corrections remain limitations. '
        'The existing range-only ocean-tide correction still omits solid-Earth and load tides. '
        'Azimuth-only fits reduce vertical-range sensitivity but do not remove horizontal tides or coherent azimuth errors. '
        'No correction coefficients, original validity masks, input netCDFs, or 2018 products were changed. '
        'The original pilot remains a failed experiment under its original noise assumptions; this regional experiment is a distinct estimand.','',
        'Files: [recipe](recipe.json), [screen checks](screen_validation.json), [noise calibration](clean_noise.json), '
        '[regional fits](regional_fits.json), [simulations](simulator_validation.json), [period scans](period_scans.json).','']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--output',type=Path,default=None);p.add_argument('--bootstrap',type=int,default=1000)
    p.add_argument('--validation',type=int,default=200);p.add_argument('--seed',type=int,default=12345)
    args=p.parse_args();root=args.root.resolve();out=args.output or root/'2017-quality-regional'
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',handlers=[logging.StreamHandler(),logging.FileHandler(out/'run.log')])
    input_path=root/'frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc'
    regions=json.loads((root/'regions.reviewed.json').read_text())['regions']
    recipe={'input':str(input_path),'regions_sha256':digest(root/'regions.reviewed.json'),'seed':args.seed,
        'bootstrap_draws':args.bootstrap,'validation_draws':args.validation,'scope':'2017 only; exploratory regional experiment',
        'screen_candidates':[.04,.06,.08,.10,.15],'residual_limits_ra_m':[2,5],
        'split':'2017-07-01','temporal_blocks_days':30,'spatial_blocks_m':3000,
        'software':{str(path):digest(path) for path in [Path(__file__),*[Path('src/glacier_offsets')/name for name in
            ('velocity_quality.py','velocity_basis.py','velocity_fit.py','velocity_io.py')]]},
        'production_maps_allowed':False}
    if (out/'recipe.json').exists() and json.loads((out/'recipe.json').read_text())!=recipe:
        raise ValueError('Changed recipe; use a new output directory')
    write_json(out/'recipe.json',recipe)
    start=time.monotonic();fields,source=extract(input_path,regions,out)
    write_json(out/'input.json',source)
    threshold,screen_report=assess_screen(fields,out)
    sigma,noise=noise_calibration(fields,threshold,out)
    for f in fields:
        velocity=quiet_median(np.where(f['clean'][...,None],f['data']/(f['late']-f['early'])[...,None],np.nan),axis=1)
        f['secular_truth']=(f['late']-f['early'])[...,None]*velocity[:,None]
    regional,state=regional_fits(fields,sigma,args.bootstrap,out,args.seed)
    scans=period_scans(fields,sigma,state,args.bootstrap,out,args.seed)
    legacy_noise=json.loads((root/'2017/noise.json').read_text())
    simulations=simulator_validation(fields,sigma,threshold,state,legacy_noise,args.validation,out,args.seed)
    make_report(out,recipe,screen_report,sigma,regional,simulations,scans)
    write_json(out/'completion.json',{'status':'complete','elapsed_seconds':time.monotonic()-start,
        'recipe_sha256':digest(out/'recipe.json'),'production_maps_allowed':False,
        'products':{f.name:digest(f) for f in out.iterdir() if f.suffix in ('.json','.npz','.md') and f.name!='completion.json'}})
    logging.info('Completed revised quality/regional validation in %.1fs',time.monotonic()-start)

if __name__=='__main__':main()
