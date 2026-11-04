"""Supplement a completed quality/regional run with sensitivity and audit checks."""
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from glacier_offsets.velocity_io import write_json, digest
from glacier_offsets.velocity_basis import utc_days
from glacier_offsets.velocity_quality import (crossfit_residual, screen, reference_times,
    flow_axis, RegionalDesign, cluster_weights, temporal_counts, quiet_median)
from tools.revalidate_velocity_quality import extract, screen_statistics, select_screen


def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n
    a=(p+z*z/(2*n))/d;b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,a-b),min(1.,a+b)]


def supplemental(root,out,draws):
    fields,_=extract(root/'frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc',
        json.loads((root/'regions.reviewed.json').read_text())['regions'],out)
    sr=json.loads((out/'screen_validation.json').read_text());threshold=sr['threshold']
    covariance=np.array(json.loads((out/'clean_noise.json').read_text())['pooled']['covariance_ra_m2'])
    for f in fields:
        f['residual']=crossfit_residual(f);f['clean']=screen(f,threshold,f['residual'])
    region=json.loads((out/'regional_fits.json').read_text())['rows']
    controls=[f for f in fields if f['role']=='control'];rows=[]
    for th in (.04,.06,.08,.10,.15):
        for supplement in (False,True):
            axis=None
            for f in fields:
                good=screen(f,th,f['residual'],supplement=supplement)
                direction=flow_axis(f,good) if f['role']=='target' else axis
                if f['id']=='target_1':axis=direction
                for channel in ('joint','range','azimuth'):
                    fit=RegionalDesign(f,good,covariance,channel).fit(direction)
                    rows.append({'threshold':th,'residual_supplement':supplement,'region':f['id'],'channel':channel,
                        'amplitude':float(np.linalg.norm(fit['cs'])),'cs':fit['cs'],'retention':float(good.sum()/f['valid'].sum())})
    write_json(out/'screen_sensitivity.json',{'interpretation':'predeclared threshold alternatives; no target-based reselection','rows':rows})
    # Vary temporal or spatial dependence scale one at a time. Avoid interpreting
    # hundreds of overlapping measurement windows as independent spatial samples.
    rng=np.random.default_rng(np.random.SeedSequence([12345,4]));uncertainty=[]
    for width,spatial_scale in ((15,3000),(60,3000),(30,6000)):
        alternate=[]
        for f in fields:
            a={**f}
            if spatial_scale==6000:
                a['group']=np.array([':'.join(str(int(x)//2) for x in g.split(':')) for g in f['group']])
            alternate.append(a)
        cache={f['id']:{c:RegionalDesign(f,f['clean'],covariance,c) for c in ('joint','azimuth')} for f in alternate}
        centers={};axis=None
        for f in alternate:
            u=flow_axis(f,f['clean']) if f['role']=='target' else axis
            if f['id']=='target_1':axis=u
            centers[f['id']]={c:cache[f['id']][c].fit(u)['cs'] for c in ('joint','azimuth')}
        samples={f['id']:{c:[] for c in ('joint','azimuth')} for f in alternate}
        for b in range(draws):
            tc=temporal_counts(rng,*reference_times(alternate[0]),width=width);axis=None
            for f in alternate:
                sw=cluster_weights(f['group'],rng)
                u=flow_axis(f,f['clean'],tc,sw) if f['role']=='target' else axis
                if f['id']=='target_1':axis=u
                for channel in ('joint','azimuth'):
                    try:cs=cache[f['id']][channel].fit(u,tc,sw)['cs']
                    except (ValueError,np.linalg.LinAlgError):cs=[np.nan,np.nan]
                    samples[f['id']][channel].append(cs)
        for f in alternate:
            for channel in ('joint','azimuth'):
                orig=next(v for v in region if v['region']==f['id'] and v['channel']==channel)
                values=np.array(samples[f['id']][channel]);values=values[np.isfinite(values).all(1)]
                # Coarser spatial blocks change equal-block weights and can
                # change the point estimate. Center at that design's own fit.
                cs=centers[f['id']][channel];delta=values-cs
                cov=np.cov(delta.T);inv=np.linalg.inv(cov)
                stats=np.einsum('ni,ij,nj->n',delta,inv,delta);stat=cs@inv@cs
                uncertainty.append({'temporal_days':width,'spatial_m':spatial_scale,'region':f['id'],'channel':channel,
                    'draws':len(values),'coefficient_std':np.sqrt(np.diag(cov)),
                    'point_cs':cs,'point_amplitude':float(np.linalg.norm(cs)),
                    'conditional_centered_bootstrap_p':float((1+np.sum(stats>=stat))/(len(values)+1)),
                    'spatial_blocks':len(np.unique(f['group']))})
        logging.info('Finished block sensitivity %d days / %d m',width,spatial_scale)
    write_json(out/'dependence_sensitivity.json',{'draws':draws,'rows':uncertainty,
        'limitation':'each spatial weighting centered at its own observed fit; exploratory scale sensitivity, not additional independent tests'})
    # Confidence ranges for quality-only screening performance on H2. The gross
    # labels are proxies, so report retained clean-proxy observations as well.
    intervals=[]
    for f in controls:
        e,l=reference_times(f);first=l<float(utc_days('2017-07-01'))
        last=e>=float(utc_days('2017-07-01'))
        endpoints=np.unique(np.floor(np.r_[e[last],l[last]]))
        first &= ~np.isin(np.floor(e),endpoints)&~np.isin(np.floor(l),endpoints)
        from glacier_offsets.velocity_quality import secular_residual
        residual=secular_residual(f['data'],f['late']-f['early'],f['valid'],first)
        avail=f['valid']&last[None]&np.isfinite(residual).all(-1)
        bad=avail&((abs(residual[...,0])>2)|(abs(residual[...,1])>5));normal=avail&~bad
        accept=screen(f,threshold,supplement=False)
        values=[]
        for b in range(draws):
            tc=temporal_counts(rng,e[last],l[last]);tw=np.zeros(len(e));tw[last]=tc
            sw=cluster_weights(f['group'],rng);weight=sw[:,None]*tw[None]
            recall=np.sum(weight*bad*~accept)/max(np.sum(weight*bad),1e-12)
            retain=np.sum(weight*normal*accept)/max(np.sum(weight*normal),1e-12)
            values.append([recall,retain])
        intervals.append({'region':f['id'],'quality_only_threshold':threshold,
            'recall_and_clean_retention_bootstrap_95':np.quantile(values,[.025,.975],axis=0),
            'scope':'second-half whole-field 30-day and 3-km block bootstrap; labels are residual proxies'})
    write_json(out/'heldout_screen_intervals.json',{'rows':intervals})
    # Recompute post-screen coverage explicitly, including source-specific phase
    # bins and span, rather than carrying forward the original valid-pair counts.
    coverage=[]
    for f in fields:
        spans=[];bins=[]
        for e,l,g in zip(f['early'],f['late'],f['clean']):
            spans.append(float(np.max(l[g])-np.min(e[g])) if g.any() else 0.)
            bins.append(len(np.unique(np.floor(np.mod((e[g]+l[g])/2,14.765294)/14.765294*12))))
        adequate=(f['clean'].sum(1)>=40)&(np.array(spans)>=270)&(np.array(bins)>=9)
        coverage.append({'region':f['id'],'unique_sources':len(f['group']),'adequate_original_per_source_coverage_count':int(adequate.sum()),
            'span_quantiles_days':np.quantile(spans,[0,.5,1]),'phase_bin_quantiles':np.quantile(bins,[0,.5,1])})
    write_json(out/'post_screen_coverage.json',{'rows':coverage,'note':'regional coefficients are distinct from per-source map eligibility'})
    simulation=json.loads((out/'simulator_validation.json').read_text())['rows'];gates=[]
    for r in simulation:
        if r['amplitude']!=0:continue
        n=r['successful_draws'];fpr=r['null_false_positive_rate']
        bench=next(v for v in simulation if v['generator']==r['generator'] and v['region']==r['region'] and v['channel']==r['channel'] and v['amplitude']==.1)
        passes=(n>=200 and fpr<=.05 and .90<=r['coverage_of_frozen_observed_bootstrap_ellipse']<=.99 and
            .90<=bench['coverage_of_frozen_observed_bootstrap_ellipse']<=.99 and
            bench['median_relative_amplitude_error']<=.2 and bench['median_phase_error_deg']<=20)
        gates.append({'generator':r['generator'],'region':r['region'],'channel':r['channel'],'null_FPR':fpr,
            'null_FPR_MC_Wilson_95':wilson(round(n*fpr),n),'null_coverage':r['coverage_of_frozen_observed_bootstrap_ellipse'],
            'positive_coverage':bench['coverage_of_frozen_observed_bootstrap_ellipse'],
            'amplitude_error':bench['median_relative_amplitude_error'],'phase_error_deg':bench['median_phase_error_deg'],
            'passes_conditional_regional_benchmark':bool(passes)})
    write_json(out/'conditional_recovery_gates.json',{'rows':gates,'scope':'exploratory frozen regional ellipse transfer checks; no production calibration gate is changed'})
    plots(out,region,simulation)
    write_json(out/'supplement_completion.json',{'status':'complete','draws':draws,'source_sha256':digest(__file__),
        'products':{name:digest(out/name) for name in ('screen_sensitivity.json','dependence_sensitivity.json',
            'heldout_screen_intervals.json','post_screen_coverage.json','conditional_recovery_gates.json','coefficient_checks.png','simulator_checks.png')}})
    return gates


def plots(out,regional,simulation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
    names=list(dict.fromkeys(r['region'] for r in regional))
    for ax,ch in zip(axes,('joint','range','azimuth')):
        for j,rid in enumerate(names):
            r=next(v for v in regional if v['region']==rid and v['channel']==ch)
            cs=np.array(r['cs_m_per_day']);cov=np.array(r['coefficient_covariance'])
            val,vec=np.linalg.eigh(cov);angle=np.degrees(np.arctan2(vec[1,-1],vec[0,-1]))
            # Display one-standard-deviation coefficient ellipses; no confidence claim.
            ax.add_patch(Ellipse(cs,2*np.sqrt(val[-1]),2*np.sqrt(val[0]),angle=angle,fill=False,color=colors[j],alpha=.8))
            ax.plot(*cs,'o' if r['role']=='target' else 'x',color=colors[j],label=rid)
        ax.axhline(0,color='.7',lw=.7);ax.axvline(0,color='.7',lw=.7)
        ax.autoscale_view();ax.set_aspect('equal',adjustable='datalim');ax.set_title(ch)
        ax.set_xlabel('Cosine coefficient (m/day)');ax.set_ylabel('Sine coefficient (m/day)')
    axes[0].legend(fontsize=8);fig.suptitle('Regional Msf coefficients: bootstrap 1-SD ellipses, exploratory')
    fig.savefig(out/'coefficient_checks.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    targets=names[:4];generators=('legacy','paired_quality','clean')
    for ax,ch in zip(axes,('joint','range','azimuth')):
        for j,g in enumerate(generators):
            rows=[next(v for v in simulation if v['generator']==g and v['region']==rid and v['channel']==ch and v['amplitude']==0) for rid in targets]
            rates=np.array([v['null_false_positive_rate'] for v in rows]);ns=np.array([v['successful_draws'] for v in rows])
            ci=np.array([wilson(round(n*p),n) for p,n in zip(rates,ns)])
            ax.errorbar(np.arange(4)+(j-1)*.15,rates*100,yerr=np.stack([rates-ci[:,0],ci[:,1]-rates])*100,
                marker='o',ls='none',capsize=2,label=g)
        ax.axhline(5,color='.4',ls='--');ax.set_xticks(range(4),targets,rotation=25);ax.set_title(ch);ax.set_ylim(bottom=0)
        ax.set_ylabel('Null false positives (%)')
    axes[0].legend(fontsize=8);fig.suptitle('Held-out control-field injections: frozen regional uncertainty, 200 draws')
    fig.savefig(out/'simulator_checks.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--output',type=Path);p.add_argument('--draws',type=int,default=1000);a=p.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    out=(a.output or a.root/'2017-quality-regional').resolve();root=a.root.resolve()
    gates=supplemental(root,out,a.draws)
    print('Passing conditional checks:',[r for r in gates if r['passes_conditional_regional_benchmark']])

if __name__=='__main__':main()
