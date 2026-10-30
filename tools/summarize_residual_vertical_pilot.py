"""Preserve source coefficients and audit the completed residual-vertical run."""
import json
from pathlib import Path
import numpy as np
from glacier_offsets.velocity_basis import harmonic,ORIGIN
from glacier_offsets.velocity_fit import false_discovery
from glacier_offsets.velocity_io import write_json,digest
from glacier_offsets.velocity_quality import flow_axis
from glacier_offsets.velocity_vertical import VerticalRegionalDesign,vertical_columns
from tools.run_residual_vertical_pilot import load_fields


def wilson(k,n):
    z=1.95996398454;p=k/n;d=1+z*z/n
    a=(p+z*z/(2*n))/d;b=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,a-b),min(1.,a+b)]


def run(root):
    out=root/'2017-residual-vertical';fields,cov,threshold=load_fields(root)
    saved=json.loads((out/'observed.json').read_text())['rows'];recovery=json.loads((out/'recovery.json').read_text())['rows']
    stress=json.loads((out/'stress50/vertical_stress50.json').read_text())['rows']
    arrays={};axis=None;maximum=0;vertical_tests=[]
    with np.load(out/'bootstrap.npz') as bootstrap:
        for f in fields:
            u=flow_axis(f,f['clean']) if f['role']=='target' else axis
            if f['id']=='target_1':axis=u
            fit=VerticalRegionalDesign(f,cov).fit(f['clean'],u,details=True)
            rid=f['id'];v=fit['nuisance'][:,2:]
            arrays.update({rid+'__'+k:value for k,value in {'source_id':f['source_id'],'row':f['row'],'col':f['col'],
                'geometry':f['geometry'],'secular_en':fit['nuisance'][:,:2],'vertical_cs_m':v,
                'horizontal_cs_m_per_day':fit['cs'],'direction_en':u}.items()})
            prediction=np.einsum('sci,si->sc',f['geometry'][...,:2],fit['nuisance'][:,:2])[:,None]*(f['late']-f['early'])[...,None]
            prediction+=np.einsum('stj,j->st',harmonic(f['early'],f['late']),fit['cs'])[...,None]*np.einsum('sci,i->sc',f['geometry'][...,:2],u)[:,None]
            prediction+=np.einsum('stj,sj->st',vertical_columns(f['early'],f['late']),v)[...,None]*f['geometry'][:,None,:,2]
            maximum=max(maximum,float(np.nanmax(abs(prediction-fit['prediction']))))
            samples=bootstrap[rid+'__vertical'];point=fit['vertical_cs_mean']
            for constituent,sl in [('M2',slice(0,2)),('O1',slice(2,4))]:
                x=samples[:,sl];x=x[np.isfinite(x).all(1)];delta=x-point[sl]
                covariance=np.cov(delta.T);inv=np.linalg.inv(covariance)
                null=np.einsum('ni,ij,nj->n',delta,inv,delta);stat=point[sl]@inv@point[sl]
                vertical_tests.append({'region':rid,'constituent':constituent,'cs_m':point[sl],
                    'coefficient_covariance':covariance,'centered_bootstrap_p':float((1+np.sum(null>=stat))/(len(null)+1))})
    flags=false_discovery(np.array([v['centered_bootstrap_p'] for v in vertical_tests]))
    for v,flag in zip(vertical_tests,flags):v['BY_q05_vertical_family']=bool(flag)
    np.savez_compressed(out/'source_coefficients.npz',**arrays)
    write_json(out/'source_coefficients.json',{'phase_origin':ORIGIN,'horizontal':'velocity C*cos+S*sin, instantaneous m/day along direction_en',
        'vertical':'height C*cos+S*sin for M2 then O1; endpoint differences produce radar displacement; residual after existing correction',
        'secular':'per-source free EN velocity in m/day; no U secular velocity',
        'source_sha256':digest(__file__),'standalone_prediction_max_difference_m':maximum})
    assert maximum<1e-8
    write_json(out/'vertical_diagnostics.json',{'rows':vertical_tests,
        'scope':'conditional exploratory residual vertical mean C/S tests; no attribution to a unique physical tide mechanism',
        'multiplicity':'BY q=.05 across 12 vertical constituent/region tests, separate from horizontal family'})
    nulls=[]
    for r in recovery:
        if r['horizontal_amplitude']!=0:continue
        nulls.append({**r,'null_FPR_MC_Wilson_95':wilson(round(r['draws']*r['null_false_positive_rate']),r['draws'])})
    write_json(out/'null_intervals.json',{'rows':nulls,'scope':'Monte Carlo intervals conditional on the empirical noise generator, not uncertainty in the control population'})
    # Preserve the completed main-run manifest; audit it before adding a final
    # report/supplement manifest that references these additional products.
    for name,sha in json.loads((out/'completion.json').read_text())['products'].items():
        assert digest(out/name)==sha, name
    recipe=json.loads((out/'recipe.json').read_text())
    assert all(digest(path)==sha for path,sha in recipe['software'].items())
    assert all(digest(path)==sha for path,sha in recipe['inputs'].items())
    assert all(r['draws']==200 for r in recovery+stress)
    assert all(r['bootstrap_draws']==1000 for r in saved)
    assert len(recovery)==198 and len(saved)==18 and len(stress)==36
    stressfile=out/'stress50/vertical_stress50.json'
    assert json.loads(stressfile.read_text())['source_sha256']==digest('tools/stress_residual_vertical_pilot.py')
    notes(out,saved,recovery,stress)
    plot(out,recovery,stress)
    write_json(out/'final_verification.json',{'status':'passed','main_validation_rows':198,'stress_validation_rows':36,
        'draws_per_validation_row':200,'observed_bootstrap_rows':18,'draws_per_bootstrap':1000,
        'source_coefficients_prediction_max_error_m':maximum,'new_numerical_tests_passed':5,
        'main_products_checksums_verified':True,'source_and_input_fingerprints_current':True,
        'maps_generated':False,'source_sha256':digest(__file__),
        'products':{p.name:digest(p) for p in out.iterdir() if p.suffix in ('.json','.npz','.md','.png') and p.name!='final_verification.json'},
        'stress_report_sha256':digest(stressfile)})


def notes(out,observed,recovery,stress):
    lines=['# Residual vertical tide experiment: interpretation','',
        'The joint residual-vertical model leaves the horizontal Msf estimates almost identical to azimuth-only fitting. '
        'It improves held-out range prediction, especially at targets 2 and 3, while accommodating residual M2/O1 patterns. '
        'It does not yield an observed horizontal detection after the primary multiple-testing adjustment.','',
        'The ocean-tide correction stays applied. Per-source residual vertical coefficients are nuisance parameters; their amplitudes are not total tidal heights. '
        'Both horizontal secular components are source-specific and free; horizontal Msf is a shared regional along-flow harmonic. '
        'Matched observations/sources/noise and paired resampling are used across three models. '
        'Sources need >=20 clean observed pairs; resampled sources with fewer than eight distinct selected epochs receive zero weight.','',
        '| Target | Fixed-vertical joint Msf | Azimuth-only Msf | Residual-vertical Msf | New-model p | Residual M2 / O1 height |',
        '|---|---:|---:|---:|---:|---:|']
    for rid in ['target_1','target_2','target_3','target_4']:
        old,az,new=[next(v for v in observed if v['region']==rid and v['model']==m) for m in ('fixed_vertical','azimuth_only','residual_vertical')]
        lines.append(f"| {rid} | {old['amplitude_m_per_day']:.3f} | {az['amplitude_m_per_day']:.3f} | {new['amplitude_m_per_day']:.3f} | {new['centered_bootstrap_p']:.3f} | {new['regional_mean_M2_amplitude_m']:.3f} / {new['regional_mean_O1_amplitude_m']:.3f} m |")
    lines+=['','Horizontal amplitudes are m/day. Reported vertical heights are magnitudes of the spatially balanced mean coefficients. '
        'Observed horizontal tests use 1,000 paired 30-day temporal / 3-km spatial bootstrap draws with reassessed flow direction. '
        'None of the 18 horizontal region/model tests passes BY q=.05. '
        'The observed screen is fixed during bootstrap; injection tests rerun it. '
        'Conditional amplitude percentile ranges are not validated confidence intervals.','',
        '## Recovery under added residual vertical motion','',
        '| Target | Added M2 / O1 height | Fixed-vertical amplitude error | New-model amplitude error | New-model phase error | Fixed-vertical null FPR | New-model null FPR |',
        '|---|---|---:|---:|---:|---:|---:|']
    for rid in ['target_1','target_2','target_3','target_4']:
        for frac in (0,.2,.5):
            if frac<.5:
                bench=[v for v in recovery if v['region']==rid and v['horizontal_amplitude']==.1 and v['vertical_fraction']==frac and v['mismatch_deg']==0 and v['crossflow_fraction']==0]
                null=[v for v in recovery if v['region']==rid and v['horizontal_amplitude']==0 and v['vertical_fraction']==frac]
            else:
                bench=[v for v in stress if v['region']==rid and v['horizontal_amplitude']==.1]
                null=[v for v in stress if v['region']==rid and v['horizontal_amplitude']==0]
            old,new=[next(v for v in bench if v['model']==m) for m in ('fixed_vertical','residual_vertical')]
            nold,nnew=[next(v for v in null if v['model']==m) for m in ('fixed_vertical','residual_vertical')]
            lines.append(f"| {rid} | {.6*frac:.2f} / {.2*frac:.2f} m | {100*old['median_amplitude_error']:.1f}% | {100*new['median_amplitude_error']:.1f}% | {new['median_phase_error_deg']:.1f}° | {100*nold['null_false_positive_rate']:.1f}% | {100*nnew['null_false_positive_rate']:.1f}% |")
    lines+=['','Each cell summarizes 200 draws under the empirical held-out control-field generator. The 50%-reference stress exceeds the largest regional-mean fitted residual heights. '
        'These are matched model comparisons within this experiment. Differences from earlier reported benchmark percentages also reflect matched-source selection and a different realized Monte Carlo stream; they are not a clean causal estimate of the effect of one code change.','',
        'The new model protects horizontal Msf against vertical contamination under its along-flow assumption. '
        'Its precision largely follows azimuth, so it is not a way to gain an additional independent horizontal observation direction. '
        'Recovery gates should be assessed separately: low null rates do not guarantee the amplitude benchmark or calibrated 95% coverage, and several intervals overcover.','',
        '## Structural sensitivity and interpretation','',
        'The main recovery file includes +/-10/20-degree flow-direction mismatch and a 20% cross-flow harmonic in quadrature. '
        'These test violations of the spatial restriction rather than vertical correction alone. '
        'The residual-vertical model cannot in general protect against horizontal cross-flow/M2/O1 contributions that its basis omits. '
        'Target_4 is outside the fast-flow regime proposed from Rutford; it remains a useful stress case, not a validation of transferring that restriction everywhere.','',
        'The residual heights can absorb calibration, projection, unmodeled tide, or other errors with similar sampled patterns. '
        'Bedrock residual harmonics and conditional vertical tests therefore do not uniquely identify an ocean-tide error. '
        'Additional years, observation directions, or independent corrections remain useful for physical attribution.','',
        'All 198 main and 36 larger-stress report rows completed 200 draws; all 18 observed bootstrap rows completed 1,000 draws. '
        'Five new numerical tests passed, including noisy dense-design agreement, source-specific tide recovery, comparison with the existing estimator, and invariance to restoring a known vertical correction. '
        'Source-level coefficients are preserved in source_coefficients.npz with units/epoch metadata in source_coefficients.json; standalone prediction agrees with the fitter to numerical precision. '
        'Input/correction files and previous experiment products are unchanged. No 2018 data or production maps were generated.','',
        '![Null-test robustness to vertical contamination](vertical_leakage_checks.png)','',
        'See [full main report](REPORT.md), [predictive checks](predictive.json), [recovery cases](recovery.json), '
        '[larger stress](stress50/vertical_stress50.json), [vertical diagnostics](vertical_diagnostics.json), '
        '[final verification](final_verification.json).','']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines))


def plot(out,recovery,stress):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,3.7),layout='constrained')
    for ax,rid in zip(axes,['target_1','target_2','target_3','target_4']):
        for model,label,marker in [('fixed_vertical','Fixed vertical','o'),('azimuth_only','Azimuth only','x'),('residual_vertical','Fit residual U','s')]:
            vals=[];errs=[]
            for frac in (0,.1,.2,.5):
                bank=recovery if frac<.5 else stress
                selected=[v for v in bank if v['region']==rid and v['model']==model and v['horizontal_amplitude']==0]
                v=next(v for v in selected if frac==.5 or v['vertical_fraction']==frac)
                p=v['null_false_positive_rate'];n=v['draws'];lo,hi=wilson(round(n*p),n)
                vals.append(100*p);errs.append([100*(p-lo),100*(hi-p)])
            ax.errorbar([0,.06,.12,.3],vals,yerr=np.array(errs).T,marker=marker,capsize=2,label=label)
        ax.axhline(5,ls='--',color='.5');ax.set_title(rid);ax.set_ylim(bottom=0)
        ax.set_xlabel('Injected M2 height amplitude (m)');ax.set_ylabel('Null false positives (%)')
    axes[0].legend(fontsize=7);fig.suptitle('Vertical contamination test: O1 height = M2 / 3; 200 draws, Monte Carlo intervals')
    fig.savefig(out/'vertical_leakage_checks.png',dpi=160);plt.close(fig)

if __name__=='__main__':run(Path('.runtime/velocity-recovery-larsen').resolve())
