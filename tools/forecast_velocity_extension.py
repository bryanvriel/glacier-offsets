"""Conditional two-year forecasts from 2017 regional covariance and planned times.

No 2018 displacement values are read or invented. Power calculations assume a
fixed harmonic equal to the observed 2017 coefficient vector and Gaussian
coefficient errors with the measured bootstrap covariance. These are planning
scenarios, not validated detection probabilities or forecasts of fitted p-values.
"""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.stats import chi2,ncx2
from glacier_offsets.velocity_basis import harmonic,utc_days,PERIOD
from glacier_offsets.velocity_inspect import M2_PERIOD,O1_PERIOD
from glacier_offsets.velocity_io import write_json,digest
from glacier_offsets.velocity_quality import RegionalDesign,screen,crossfit_residual


def temporal_alias(e,l,mask,period):
    a=harmonic(e[mask],l[mask]);b=harmonic(e[mask],l[mask],period)
    # Remove secular displacement independently of the harmonic being assessed.
    dt=l[mask]-e[mask];q=dt/np.linalg.norm(dt)
    a-=q[:,None]*(q@a)[None];b-=q[:,None]*(q@b)[None]
    qa=np.linalg.qr(a)[0];qb=np.linalg.qr(b)[0]
    corr=np.linalg.svd(qa.T@qb,compute_uv=False).max()
    ab=np.c_[a/np.linalg.norm(a,axis=0),b/np.linalg.norm(b,axis=0)]
    return [float(corr),float(np.linalg.cond(ab))]


def forecast(root):
    prior=root/'2017-quality-regional';out=root/'2018-forecast';out.mkdir(exist_ok=True)
    regions=json.loads((prior/'regional_fits.json').read_text())['rows']
    validation=json.loads((prior/'conditional_recovery_gates.json').read_text())['rows']
    plan=json.loads((root/'extended-stack-plan.json').read_text())
    future=[v for k,v in sorted(plan['pairs'].items()) if v['later']>='2018-01-01']
    assert len(future)==60
    meta=json.loads((prior/'observations.json').read_text())
    fields=[]
    with np.load(prior/'observations.npz') as h:
        for rid in dict.fromkeys(r['region'] for r in regions):
            fields.append({k:h[rid+'__'+k] for k in meta['keys']}|{'id':rid})
    cov=np.array(json.loads((prior/'clean_noise.json').read_text())['pooled']['covariance_ra_m2'])
    threshold=json.loads((prior/'screen_validation.json').read_text())['threshold']
    forecast_rows=[];design_rows=[];alias_rows=[];coverage=[]
    family=len(regions);first_by_alpha=.05/(family*np.sum(1/np.arange(1,family+1)))
    with h5py.File(root/'frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc') as nc:
        owner=nc['source_burst_index'][:];names=nc['burst_id'][:].astype(str)
    for f in fields:
        good=screen(f,threshold,crossfit_residual(f));n,t=good.shape
        bursts=names[owner[f['row'],f['col']]-1]
        e2=np.array([[float(utc_days(pair['burst_intervals'][burst]['acquisition_times'][0])) for pair in future] for burst in bursts])
        l2=np.array([[float(utc_days(pair['burst_intervals'][burst]['acquisition_times'][1])) for pair in future] for burst in bursts])
        # Preserve the 2017 source-specific clean/missing mask pattern in the
        # added year. This is one explicit similar-coverage scenario, not a
        # prediction of which 2018 correlations will succeed.
        full={**f,'early':np.c_[f['early'],e2],'late':np.c_[f['late'],l2],
              'valid':np.c_[good,good],'data':np.zeros((n,2*t,2))}
        for ch in ('joint','azimuth'):
            r=next(v for v in regions if v['region']==f['id'] and v['channel']==ch)
            u=np.array(r['direction_en']);cs=np.array(r['cs_m_per_day']);sigma=np.array(r['coefficient_covariance'])
            a=np.linalg.inv(RegionalDesign(f,good,cov,ch).fit(u)['information'])
            b=np.linalg.inv(RegionalDesign(full,full['valid'],cov,ch).fit(u)['information'])
            j=cs/np.linalg.norm(cs)
            design_rows.append({'region':f['id'],'channel':ch,'formal_amplitude_sd_ratio':float(np.sqrt(j@b@j/(j@a@j))),
                'formal_component_sd_ratios':np.sqrt(np.diag(b)/np.diag(a))})
            amp=float(np.linalg.norm(cs));amp_sd=float(np.sqrt(j@sigma@j))
            phase_j=np.array([-cs[1],cs[0]])/(amp*amp)
            phase_sd=float(np.degrees(np.sqrt(phase_j@sigma@phase_j)))
            for rho in (0.,.5,1.):
                factor=np.sqrt((1+rho)/2);sigma2=sigma*factor*factor
                noncentral=float(cs@np.linalg.inv(sigma2)@cs)
                bench=next(v for v in validation if v['generator']=='paired_quality' and v['region']==f['id'] and v['channel']==ch)
                forecast_rows.append({'region':f['id'],'channel':ch,'year_error_correlation':rho,'sd_ratio':factor,
                    'assumed_persistent_amplitude':amp,'amplitude_sd_2017_delta_method':amp_sd,
                    'amplitude_sd_two_year_delta_method':amp_sd*factor,
                    'phase_sd_2017_deg_delta_method':phase_sd,'phase_sd_two_year_deg_delta_method':phase_sd*factor,
                    'plug_in_power_unadjusted_alpha05':float(ncx2.sf(chi2.isf(.05,2),2,noncentral)),
                    'plug_in_power_first_BY_rejection':float(ncx2.sf(chi2.isf(first_by_alpha,2),2,noncentral)),
                    'scaled_benchmark_median_amplitude_error':bench['amplitude_error']*factor,
                    'scaled_benchmark_median_phase_error_deg':bench['phase_error_deg']*factor})
        for period,name in ((M2_PERIOD,'M2'),(O1_PERIOD,'O1')):
            old=[];new=[]
            for i in range(n):
                if good[i].sum()<20:continue
                old.append(temporal_alias(f['early'][i],f['late'][i],good[i],period))
                new.append(temporal_alias(full['early'][i],full['late'][i],full['valid'][i],period))
            old,new=np.array(old),np.array(new)
            alias_rows.append({'region':f['id'],'n_sources':len(old),'constituent':name,
                'year1_median_max_canonical_correlation':float(np.median(old[:,0])),
                'two_year_median_max_canonical_correlation':float(np.median(new[:,0])),
                'year1_median_normalized_joint_condition':float(np.median(old[:,1])),
                'two_year_median_normalized_joint_condition':float(np.median(new[:,1]))})
        spans=np.array([np.max(l[g])-np.min(e[g]) if g.any() else 0 for e,l,g in zip(full['early'],full['late'],full['valid'])])
        bins=np.array([len(np.unique(np.floor(np.mod((e[g]+l[g])/2,PERIOD)/PERIOD*12))) for e,l,g in zip(full['early'],full['late'],full['valid'])])
        counts=full['valid'].sum(1)
        coverage.append({'region':f['id'],'sources':n,'median_clean_pairs_two_year':float(np.median(counts)),
            'sources_at_least_60_pairs_540_days_9_bins':int(np.sum((counts>=60)&(spans>=540)&(bins>=9)))})
    o1alias=abs(1/O1_PERIOD-round(6/O1_PERIOD)/6)
    recipe={'scope':'conditional planning forecast; no 2018 displacement data used',
        'future_pairs':len(future),'future_dates':[future[0]['earlier'],future[-1]['later']],
        'assumptions':['same geometry and per-source clean-mask pattern in added year',
            'constant true Msf coefficients across years for covariance/power projection',
            'Gaussian coefficient covariance approximation to the 2017 block bootstrap',
            'year error correlations 0, .5 and 1 are scenarios, not estimated correlations'],
        'sources':{str(path):digest(path) for path in (Path(__file__),prior/'regional_fits.json',prior/'conditional_recovery_gates.json',root/'extended-stack-plan.json',prior/'observations.npz')},
        'first_BY_threshold_18_tests':first_by_alpha,'O1_Msf_beat_days_at_exact_6d':float(1/abs(o1alias-1/PERIOD))}
    write_json(out/'forecast.json',{'recipe':recipe,'scenarios':forecast_rows,'planned_design':design_rows,
        'temporal_aliases':alias_rows,'hypothetical_coverage':coverage})
    for r in forecast_rows:
        if r['channel']=='azimuth' and r['year_error_correlation']==0 and r['region'].startswith('target'):
            print(r)
    print('design ratios',[(r['region'],r['channel'],r['formal_amplitude_sd_ratio']) for r in design_rows])
    print('aliases',alias_rows)
    print('coverage',coverage)
    print('O1 beat',recipe['O1_Msf_beat_days_at_exact_6d'])

if __name__=='__main__':forecast(Path('.runtime/velocity-recovery-larsen').resolve())
