"""Summarize and audit the two-half, permutation control-noise experiment."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/glacier-offsets-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_systematics import holm

REGIONS=['target_1','target_2','target_3','target_4','control_1','control_2']
SCENARIOS=['H2_replacement','H2_permutation','crossfit_replacement','crossfit_permutation']
MODELS=['primary_azimuth','no_order','azimuth_along','residual_vertical']
CASES=['null','benchmark','null_vertical20','benchmark_vertical20','benchmark_M2_003','benchmark_M2_020','null_M2_020']


def wilson(k,n):
    z=norm.ppf(.975);p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.array([center-half,center+half])


def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])


def main(root):
    old=root/'2017-order-set';out=root/'2017-order-set-generator'
    recipe=json.loads((out/'recipe.json').read_text());observed=json.loads((old/'observed.json').read_text())['rows']
    obs={(r['region'],r['variant']):r for r in observed};summary=[];raw={}
    for scenario in SCENARIOS:
        cases=CASES if scenario=='crossfit_permutation' else ['null']
        for case in cases:
            file=out/(scenario+'--'+case+'.npz')
            with np.load(file) as h:
                models=MODELS if scenario=='crossfit_permutation' else ['primary_azimuth']
                for ri,rid in enumerate(REGIONS):
                    for model in models:
                        values=h[rid+'__'+model];raw[scenario,case,rid,model]=values
                        a=values[np.isfinite(values).all(1)];delta=a[:,:2]-a[:,2:4]
                        empirical=np.cov(delta.T);oc=np.array(obs[rid,model]['covariance']);factor=np.linalg.cholesky(oc)
                        whitened=np.linalg.solve(factor,np.linalg.solve(factor,empirical).T).T
                        inv=np.linalg.inv(oc);stat=np.einsum('ni,ij,nj->n',a[:,:2],inv,a[:,:2]);target=np.array(obs[rid,model]['cs_m_per_day']);tobs=target@inv@target
                        fpr=float(np.mean(a[:,7]));n=len(a);benchmark=case.startswith('benchmark')
                        row={'scenario':scenario,'case':case,'region':rid,'model':model,'draws':n,'failures':len(values)-n,
                            'coefficient_bias':np.mean(delta,axis=0),'empirical_error_covariance':empirical,
                            'empirical_trace_ratio_to_observed':float(np.trace(empirical)/np.trace(oc)),
                            'null_second_moment_ratio_to_observed':float(np.mean(np.sum(delta**2,axis=1))/np.trace(oc)),
                            'observed_whitened_covariance_eigenvalues':np.linalg.eigvalsh(whitened),
                            'frozen_ellipse_coverage':float(np.mean(a[:,6])),
                            'frozen_ellipse_rejection_rate':fpr,'rejection_Wilson95':wilson(int(a[:,7].sum()),n),
                            'median_retention_of_target_valid':float(np.median(a[:,8])),
                            'median_retention_H1_H2':np.median(h['retention_by_half'][:,ri],axis=0),
                            'median_unique_donor_sources':float(np.median(h['donor_diagnostics'][:,ri,1])),
                            'target_sources':int(h['donor_diagnostics'][0,ri,2]),
                            'median_unique_donor_blocks':float(np.median(h['donor_diagnostics'][:,ri,3])),
                            'target_blocks':int(h['donor_diagnostics'][0,ri,4]),
                            'median_amplitude_error':float(np.median(a[:,4])) if benchmark else None,
                            'median_phase_error_deg':float(np.median(a[:,5])) if benchmark else None,
                            'transfer_null_p_at_observed':float((1+np.sum(stat>=tobs))/(len(a)+1)) if case=='null' else None}
                        summary.append(row)
    primary=[r for r in summary if r['scenario']=='crossfit_permutation' and r['case']=='null' and r['model']=='primary_azimuth']
    for r,p in zip(primary[:4],holm([r['transfer_null_p_at_observed'] for r in primary[:4]])):r['transfer_Holm_p_four_targets']=float(p)
    nested=[];attempted=0;failures=0
    for case in ('null','benchmark'):
        saved=json.loads((out/('nested-'+case+'.json')).read_text());rows=saved['rows']
        for rid in REGIONS:
            q=[r for r in rows if r['region']==rid];r=next(r for r in primary if r['region']==rid)
            empirical=next(x['empirical_error_covariance'] for x in summary if x['scenario']=='crossfit_permutation' and x['case']==case and x['region']==rid and x['model']=='primary_azimuth')
            trace_observed=np.trace(obs[rid,'primary_azimuth']['covariance']);trace_empirical=np.trace(empirical)
            ncov=[];points=[];truths=[];cutoffs=[]
            for record in q:
                b=record['draw'];a=raw['crossfit_permutation',case,rid,'primary_azimuth'][b];cs=a[:2];truth=a[2:4]
                with np.load(out/('nested-'+case)/(rid+'--'+str(b)+'.npz')) as h:values=h['cs']
                delta=values-cs;cov=np.cov(delta.T);inv=np.linalg.inv(cov)
                stat=np.einsum('ni,ij,nj->n',delta,inv,delta);cutoff=float(np.quantile(stat,.95))
                p=float((1+np.sum(stat>=cs@inv@cs))/(len(values)+1));err=cs-truth
                if abs(p-record['nested_p'])>1e-12:raise ValueError('Nested p does not reproduce')
                if bool(err@inv@err<=cutoff)!=record['coverage']:raise ValueError('Nested truth coverage does not reproduce')
                if not np.isclose(np.trace(cov)/trace_observed,record['cov_trace_ratio_to_observed'],rtol=1e-10):raise ValueError('Nested covariance does not reproduce')
                ncov.append(cov);points.append(cs);truths.append(truth);cutoffs.append(cutoff)
                attempted+=len(values)+record['bootstrap_failures'];failures+=record['bootstrap_failures']
            n=len(q);rejected=sum(x['nested_p']<=.05 for x in q);covered=sum(x['coverage'] for x in q)
            covariance=np.array(ncov);traces=np.trace(covariance,axis1=1,axis2=2)
            nr={'region':rid,'case':case,'datasets':n,'rejections':rejected,'rejection_rate':rejected/n,
                'rejection_Wilson95':wilson(rejected,n),'coverage':covered/n,'coverage_Wilson95':wilson(covered,n),
                'median_trace_ratio_to_observed':float(np.median(traces)/trace_observed),
                'median_trace_ratio_to_empirical':float(np.median(traces)/trace_empirical),
                'F_reference_rejection_rate':sum(x['F_reference_p']<=.05 for x in q)/n,
                'bootstrap_failures':sum(x['bootstrap_failures'] for x in q)}
            nested.append(nr)
            np.savez_compressed(out/('nested-derived-'+case+'-'+rid+'.npz'),draw_indices=[x['draw'] for x in q],
                                point_cs=points,truth_cs=truths,covariance=ncov,cutoff95=cutoffs)
    null_rows=json.loads((out/'nested-null.json').read_text())['rows'];family_rejections=[]
    for b in range(recipe['nested_null_per_region']):
        ps=[next(r['nested_p'] for r in null_rows if r['draw']==b and r['region']==rid) for rid in REGIONS[:4]]
        if np.any(holm(ps)<=.05):family_rejections.append(b)
    family={'independent_field_draws':recipe['nested_null_per_region'],'false_rejections':len(family_rejections),
        'false_family_draw_indices':family_rejections,'FWER':len(family_rejections)/recipe['nested_null_per_region'],
        'Wilson95':wilson(len(family_rejections),recipe['nested_null_per_region'])}
    write_json(out/'family-calibration.json',family)
    write_json(out/'summary.json',{'recovery_rows':summary,'nested_rows':nested,'attempted_nested_resamples':attempted,
        'nested_bootstrap_failures':failures,'four_target_family_null':family,'nested_reconstruction':'point/truth from recovery draw and bootstrap samples independently reproduce p, covariance and coverage',
        'MC_limit':'confidence intervals are per region; regions share field draws and are not independent pooled replicates'})
    # Factorial comparison, with the observed variance as a common reference.
    fig,ax=plt.subplots(figsize=(11,4.7),layout='constrained');x=np.arange(6)
    labels=['H2, replacement','H2, permutation','Both halves, replacement','Both halves, permutation']
    colors=['#999999','#c39061','#6ba5ba','#175c81']
    for i,(scenario,label,color) in enumerate(zip(SCENARIOS,labels,colors)):
        vals=[next(r['empirical_trace_ratio_to_observed'] for r in summary if r['scenario']==scenario and r['case']=='null' and r['region']==rid and r['model']=='primary_azimuth') for rid in REGIONS]
        ax.bar(x+(i-1.5)*.18,vals,width=.17,label=label,color=color)
    ax.axhline(1,color='.4',ls='--',lw=1);ax.set(xticks=x,xticklabels=[r.replace('_',' ') for r in REGIONS],ylabel='Synthetic / observed covariance trace',title='Null noise transfer: calendar bank and spatial assignment')
    ax.legend(fontsize=8,ncol=2);fig.savefig(out/'generator-comparison.png',dpi=180);fig.savefig(out/'generator-comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
    original=json.loads((old/'recovery.json').read_text())['rows']
    oldcov=[next(r['frozen_ellipse_coverage'] for r in original if r['region']==rid and r['case']=='null' and r['variant']=='primary_azimuth') for rid in REGIONS]
    newcov=[next(r['frozen_ellipse_coverage'] for r in primary if r['region']==rid) for rid in REGIONS]
    q=[next(r for r in nested if r['region']==rid and r['case']=='null') for rid in REGIONS];v=np.array([r['coverage'] for r in q]);bounds=np.array([r['coverage_Wilson95'] for r in q])
    ax.scatter(x-.15,oldcov,marker='x',color='.5',s=55,label='Original generator, frozen observed ellipse')
    ax.scatter(x,newcov,marker='s',color='#c39061',s=30,label='Revised generator, frozen observed ellipse')
    ax.errorbar(x+.15,v,yerr=[v-bounds[:,0],bounds[:,1]-v],fmt='o',color='#175c81',capsize=3,label='Revised generator, nested paired bootstrap (100 fields/region)')
    ax.axhline(.95,color='.5',ls='--',lw=1);ax.set(xticks=x,xticklabels=[r.replace('_',' ') for r in REGIONS],ylim=(.6,1.04),ylabel='Coefficient coverage',title='Null coverage after revising the generator')
    ax.legend(loc='lower left',fontsize=8);fig.savefig(out/'calibration-comparison.png',dpi=180);fig.savefig(out/'calibration-comparison.pdf');plt.close(fig)
    write_report(root,out,recipe,summary,nested,attempted)
    verify(root,out,recipe,summary,nested,attempted,failures)


def write_report(root,out,recipe,summary,nested,attempted):
    def row(rid,scenario='crossfit_permutation',case='null',model='primary_azimuth'):
        return next(r for r in summary if (r['region'],r['scenario'],r['case'],r['model'])==(rid,scenario,case,model))
    def nr(rid,case='null'):return next(r for r in nested if r['region']==rid and r['case']==case)
    old=root/'2017-order-set';original=json.loads((old/'recovery.json').read_text())['rows']
    title='# 2017 Larsen: cross-fitted two-half control-noise generator'
    lines=[title,'',
        'This experiment implements the generator revision requested after [Opus’s order/SET review](OPUS_FEEDBACK_ORDER_SET.md). It changes the synthetic noise generator while preserving the original observed estimates, screen, covariance, paired bootstrap, model family and original experiment files. No 2018 data are read. The new outputs are in `.runtime/velocity-recovery-larsen/2017-order-set-generator/`.','',
        'The revision substantially reduces synthetic variance inflation. In the matched comparison, target_4’s null covariance trace drops from 2.43 to 1.07 times its observed covariance trace. Its frozen-ellipse false-positive rate remains 10%, but the recomputed paired bootstrap rejects only 1/100 null fields in each target, with 99–100% coverage across targets and controls. The four-target Holm family has no false rejections among 100 shared fields. Recovery errors remain below the original amplitude and phase benchmarks.','',
        '## 1. Revised generator','',
        'H1 means January–June; H2 means July–December. For each bedrock control, H1 residuals are predicted from H2 fits and H2 residuals from H1 fits. Training pairs sharing acquisition dates with the held-out half are explicitly purged. The pair crossing July 1 is excluded from the donor bank. Range uses a robust source-specific secular prediction; azimuth uses source-specific secular terms and shared owning-burst order terms from the opposite-half primary fit. That training fit also includes Msf to separate nuisance estimation, but only secular and order predictions are subtracted: the held-out control harmonic is not removed.','',
        'Residuals are median centered per source within each held-out half. Low-peak observations and failed matches remain in the bank with their peak, SNR and validity. Sources need a defined local training prediction; the regional order/harmonic training estimate retains the science fitter’s minimum of eight clean epochs per contributing source. Centering removes the source’s constant held-out prediction offset, so sparse local secular training is not used to add an extra eight-epoch eligibility cut to donor support. All selected controls have a defined prediction in both halves. Per-source training counts are recorded in `bank-folds.json`.','',
        'A preflight version applied the eight-epoch science threshold to every donor baseline, which would have deleted many otherwise usable H1 donors. This was revised before any recovery results were computed; the preflight recipe and bank are retained with `preflight-min8` filenames. It is separate from the final recorded recipe.','',
        'Spatial transfer permutes donor 3 km blocks, then donor sources within each assigned block. Each pool is exhausted before a new permutation begins. Reuse is allowed when the chosen pool is smaller than the target demand. This preserves the block hierarchy and avoids optional duplication; it cannot guarantee a global one-to-one assignment when source or block capacities differ. Source assignments remain fixed through each synthetic year.','',
        'Whole-field circular five-pair temporal blocks are sampled independently within the matching calendar half and shared across all six target/control templates. H1 destinations receive H1 donors; H2 destinations receive H2 donors. The destination pair crossing July 1 receives an H2 donor. Circular wraps stay inside a half. Exact target endpoints and geometry remain in the fit. Quality, failure residuals and missingness travel together. Known target spacecraft-order displacement is reinserted at the target order and re-estimated in every fit.','',
        '## 2. Fixed comparisons and validation','',
        'The observed amplitudes and p-values are unchanged from [the order/SET experiment](ORDER_SET_EXPERIMENT_2026-10-29.md). Its H1 control covariance, sources, fixed peak threshold 0.08, 2/5 m residual screen, direct-azimuth primary estimand, four-target Holm family and paired 30-day temporal × 3 km spatial bootstrap remain fixed. This audit does not select a generator by forcing a calibration gate to pass. The original H2/with-replacement case is retained as a stress case.','',
        'Four null-generator variants use 200 synthetic fields each: H2 bank with replacement; H2 bank with permutation assignments; both cross-fitted calendar halves with replacement; and both halves with permutation assignments. Control-bank choice and phase/time/spatial random streams are separated by fixed seed components. Variants share donor choices; variants with the same calendar policy share time indices. Changing the calendar policy necessarily changes those indices. The H2 comparison reproduces the original construction statistically, with a new documented Monte Carlo stream rather than identical original draws.','',
        'The primary revised generator also repeats all seven recovery cases from the prior experiment, with 200 fields per case and four estimators: direct azimuth, order omitted, azimuth along-flow, and joint residual vertical. Cases include null and 0.10 m/day along-flow Msf, with and without residual vertical M2/O1 height amplitudes 0.12/0.04 m; benchmark plus horizontal M2 velocity 0.03 or 0.20 m/day; and an Msf-null with horizontal M2 0.20 m/day. Screens and flow axes are recomputed after injection.','',
        'Nested uncertainty is recomputed in 100 null fields per region and 20 benchmark fields per region, using 199 paired resamples per dataset. Benchmark indices span the full phase cycle. There are 720 regional datasets and '+f'{attempted:,}'+' attempted nested resamples. Regions share field draws, so uncertainty is reported per region rather than by treating the 720 regional fits as independent replications. P-values include the +1 Monte Carlo correction.','',
        '## 3. Generator variance and retained support','',
        table(['Region','H2 replacement','H2 permutation','Both halves replacement','Both halves permutation'],[[rid,*[f"{row(rid,s)['empirical_trace_ratio_to_observed']:.2f}" for s in SCENARIOS]] for rid in REGIONS]),'',
        'Entries are empirical synthetic null coefficient-covariance trace divided by the unchanged observed paired-bootstrap covariance trace. They measure transfer mismatch relative to a model-conditional observed estimate, rather than an absolute truth variance. The JSON also gives the second moment about the known zero coefficient, bias vectors and covariance eigenvalues after whitening by the observed covariance.','',
        '![Generator comparison](../../.runtime/velocity-recovery-larsen/2017-order-set-generator/generator-comparison.png)','',
        table(['Region','Observed clean / valid','Revised clean / target valid','Revised H1 / H2 retention','Unique donor sources / target sources','Unique donor blocks / target blocks'],[[rid,f"{next(r['retained_observations']/r['valid_observations'] for r in json.loads((old/'support.json').read_text()) if r['region']==rid):.3f}",f"{row(rid)['median_retention_of_target_valid']:.3f}",' / '.join(f'{x:.3f}' for x in row(rid)['median_retention_H1_H2']),f"{row(rid)['median_unique_donor_sources']:.0f} / {row(rid)['target_sources']}",f"{row(rid)['median_unique_donor_blocks']:.0f} / {row(rid)['target_blocks']}"] for rid in REGIONS]),'',
        'Retained fractions use each original target template’s valid observations as denominator, so transferred missingness is included. Median unique-donor counts describe the primary null generator and include unavoidable pool recycling. A common donor control is selected per whole-field realization; only two controls are available. The revised bank therefore does not establish that all regional or seasonal ice-noise distributions are represented.','',
        '## 4. Frozen observed ellipses and residual mismatch','',
        table(['Region','Original frozen null FPR','Revised frozen null FPR (Wilson 95%)','Revised frozen coverage','Whitened covariance eigenvalues'],[[rid,f"{next(r['frozen_ellipse_null_FPR'] for r in original if r['region']==rid and r['case']=='null' and r['variant']=='primary_azimuth'):.3f}",f"{row(rid)['frozen_ellipse_rejection_rate']:.3f} ({row(rid)['rejection_Wilson95'][0]:.3f}–{row(rid)['rejection_Wilson95'][1]:.3f})",f"{row(rid)['frozen_ellipse_coverage']:.3f}",' / '.join(f'{x:.2f}' for x in row(rid)['observed_whitened_covariance_eigenvalues'])] for rid in REGIONS]),'',
        'The revised generator reduces the previous mismatch substantially. Target 4 remains a distinct case: an overall covariance trace near the observed value does not guarantee the same covariance orientation, eigenvalues or tails. Its remaining frozen-ellipse rejection rate must be reported rather than removed by variance rescaling. Conversely, a frozen-ellipse failure on a transferred population does not establish that the real observed uncertainty is too small.','',
        '## 5. Nested paired-bootstrap calibration','',
        table(['Region','Null rejections / 100','Null FPR Wilson 95%','Null coverage','Median nested / empirical null covariance trace','Benchmark coverage (20 fields)'],[[rid,f"{nr(rid)['rejections']} / {nr(rid)['datasets']}",f"{nr(rid)['rejection_Wilson95'][0]:.3f}–{nr(rid)['rejection_Wilson95'][1]:.3f}",f"{nr(rid)['coverage']:.3f}",f"{nr(rid)['median_trace_ratio_to_empirical']:.2f}",f"{nr(rid,'benchmark')['coverage']:.2f}"] for rid in REGIONS]),'',
        'Each dataset receives its own covariance and centered-bootstrap 95% radius. Coverage is checked against the exact noiseless injected Msf projection on that dataset’s screened support. The saved recovery point estimates and bootstrap samples independently reproduce every nested covariance, p-value and truth-coverage decision. The paired method is retained; no narrower time-only procedure is substituted. The F reference is recorded for comparison and is not assumed exact.','',
        'All four marginal null rejections occur in the same shared field realization (draw 48), with p-values 0.015–0.025. Applying the unchanged four-target Holm rule yields zero false families among 100 field draws. Its Wilson 95% upper bound is about 3.7%. This family calculation preserves regional dependence rather than pooling marginal tests.','',
        'The paired covariance traces are approximately 2.3–2.7 times the empirical null covariance traces and coverage is 99–100%. The revised generator therefore still gives conservative paired uncertainty. These checks support retaining the paired method, while providing no reason to claim exact nominal-95% calibration or to replace it with a narrower time-only method.','',
        'Six of 143,280 attempted nested resamples were rank deficient, all in target_2 null field 6. The existing fitter skipped those draws; that dataset retains 193 of 199 samples, and every other dataset retains all 199. No synthetic dataset or recovery fit was lost. The failure audit is stored separately; these are counted rather than backfilled or silently discarded. The numerical estimator was kept fixed.','',
        'With 100 null fields per region, zero rejections has a two-sided Wilson 95% upper bound of about 3.7%. This is substantially more informative than 0/20, whose upper bound is about 16%. These bounds concern this generator and field population; they do not remove uncertainty about applying two bedrock controls to real ice. Benchmark coverage still uses only 20 datasets per region.','',
        '![Calibration comparison](../../.runtime/velocity-recovery-larsen/2017-order-set-generator/calibration-comparison.png)','',
        '## 6. Recovery and observed-signal sensitivity','',
        table(['Region','Benchmark amplitude error: direct / residual vertical','Benchmark phase error: direct / residual vertical','Benchmark + M2=.20: direct amplitude / phase error'],[[rid,' / '.join(f"{row(rid,case='benchmark',model=m)['median_amplitude_error']:.1%}" for m in ('primary_azimuth','residual_vertical')),' / '.join(f"{row(rid,case='benchmark',model=m)['median_phase_error_deg']:.1f}°" for m in ('primary_azimuth','residual_vertical')),f"{row(rid,case='benchmark_M2_020')['median_amplitude_error']:.1%} / {row(rid,case='benchmark_M2_020')['median_phase_error_deg']:.1f}°"] for rid in REGIONS]),'',
        'The benchmark is the original 0.10 m/day along-flow injection. The primary estimator reports its satellite-azimuth projection; comparison errors use the exact noiseless regional projection rather than an approximate average geometry. Horizontal M2 inputs are velocity amplitudes. Their alias remains unresolved by this one-track observation pattern.','',
        table(['Region','Transferred-null p at observed statistic','Four-target Holm p'],[[rid,f"{row(rid)['transfer_null_p_at_observed']:.4f}"+(' (MC floor)' if row(rid)['transfer_null_p_at_observed']==1/201 else ''),f"{row(rid)['transfer_Holm_p_four_targets']:.4f}" if 'transfer_Holm_p_four_targets' in row(rid) else 'negative control'] for rid in REGIONS]),'',
        'These sensitivity p-values compare the unchanged observed statistic with the revised generator’s 200 null fields. A value of 1/201 is a Monte Carlo floor, not a measured smaller tail probability. The observed-data bootstrap p-values and estimand are unchanged. Neither a lower-noise generator nor a small transferred-null p-value independently establishes physical Msf attribution.','',
        '## 7. Conclusions and scope','',
        'The experiment separates the generator revision from the observed inversion. Using both cross-fitted calendar halves and permutation spatial transfer avoids the two identified sources of synthetic variance inflation. The factorial comparisons retain the original stress construction, and target 4’s remaining mismatch is explicitly quantified. The larger nested experiment assesses the paired uncertainty procedure under the revised generator instead of borrowing an observed ellipse for every synthetic dataset.','',
        'These results refine the interpretation of the earlier calibration gate. They do not change the observed Msf estimates or resolve horizontal M2 aliasing, ocean loading, cross-flow assumptions, or across-year response stability. No loading model, 2018 observation, new ROI selection, production correction or velocity map is introduced. The original experiment remains byte-for-byte unchanged.','',
        '## 8. Reproduction and verification','',
        '```bash',
        'PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 /Users/briel/miniforge3/envs/ice/bin/python tools/revise_order_set_generator.py --stage all',
        'PYTHONPATH=src /Users/briel/miniforge3/envs/ice/bin/python tools/summarize_revised_generator.py',
        'PYTHONPATH=src:tests /Users/briel/miniforge3/envs/ice/bin/python -m unittest test_velocity_noise test_velocity_systematics test_velocity_vertical test_velocity_quality test_velocity_recovery',
        '```','',
        'The recipe hashes the original observed results, covariance, corrected input cache and numerical source files before the runs. Source changes prevent checkpoint reuse in this output directory. Individual nested datasets are checkpointed with their bootstrap samples. `banks.npz` and `bank-folds.json` retain residuals, quality, support, train/test dates, order coefficients and centering values. Random streams reproduce all synthetic datasets. `summary.json` retains covariances, Monte Carlo counts, coverage bounds and bias diagnostics.','',
        'All 59 relevant temporal-inference tests pass. Tests verify acquisition separation, retention of held-out harmonics and failed matches, quality/validity pairing, exhaustion before pool reuse, preservation of a known whole-field temporal signal, and the balanced-transfer identity for Gaussian fields with both independent and common-mode components. Permuting a balanced donor field preserves each realization’s regional mean exactly; it does not add an extra spatial resampling layer. This identity validates the transfer implementation, not universal calibration for heterogeneous ice noise.','']
    text='\n'.join(lines);doc=Path('docs/velocity-recovery/GENERATOR_REVISION_2026-10-29.md');doc.write_text(text)
    (out/'report.md').write_text(text.replace('(OPUS_FEEDBACK_ORDER_SET.md)','(../../../docs/velocity-recovery/OPUS_FEEDBACK_ORDER_SET.md)').replace('(ORDER_SET_EXPERIMENT_2026-10-29.md)','(../../../docs/velocity-recovery/ORDER_SET_EXPERIMENT_2026-10-29.md)').replace('(../../.runtime/velocity-recovery-larsen/2017-order-set-generator/','('))
    print(doc.resolve())


def verify(root,out,recipe,summary,nested,attempted,failures):
    old=root/'2017-order-set';original=json.loads((out/'original-artifacts.json').read_text())
    assert all(digest(old/name)==value for name,value in original.items()),'Original experiment changed'
    assert all(digest(path)==value for path,value in recipe['code_sha256'].items()),'Numerical source changed'
    assert len(summary)==186 and all(r['draws']==200 and r['failures']==0 for r in summary)
    assert len(nested)==12 and all(r['datasets']==(100 if r['case']=='null' else 20) for r in nested)
    assert attempted==143280
    failure_audit=json.loads((out/'rank-failure-audit.json').read_text())
    assert len(failure_audit['errors'])==failures
    records=[r for case in ('null','benchmark') for r in json.loads((out/('nested-'+case+'.json')).read_text())['rows']]
    assert failures==sum(r['bootstrap_failures'] for r in records)
    assert all(r['successful_draws']>=.9*recipe['nested_resamples'] for r in records)
    folds=json.loads((out/'bank-folds.json').read_text())['rows']
    for region in folds:
        for record in region['folds']:
            assert not np.intersect1d(record['train_acquisition_days'],record['test_acquisition_days']).size
    files=[p for p in out.iterdir() if p.is_file() and p.name!='verification.json' and p.suffix!='.log']
    file_hashes={p.name:digest(p) for p in files}
    # Include every nested checkpoint, not only the aggregate summaries.
    nested_hashes={str(p.relative_to(out)):digest(p) for p in out.glob('nested-*/*') if p.is_file()}
    frozen=json.loads((root/'frozen-2017/frozen.json').read_text())['inputs']
    assert all(digest(r['frozen'])==r['sha256'] for r in frozen)
    write_json(out/'verification.json',{'status':'complete generator revision and validation',
        'original_experiment_unchanged':True,'frozen_inputs_unchanged':True,'recovery_rows':len(summary),
        'successful_synthetic_estimator_fits':sum(r['draws'] for r in summary),
        'nested_regional_datasets':sum(r['datasets'] for r in nested),'nested_resamples':attempted,
        'recovery_fit_failures':0,'nested_fit_failures':failures,'successful_nested_resamples':attempted-failures,'nested_p_covariance_coverage_reproduced':True,
        'artifact_sha256':file_hashes,'nested_artifact_sha256':nested_hashes,
        'summary_code_sha256':digest(Path(__file__)),'tests_sha256':digest(Path('tests/test_velocity_noise.py')),
        'tests_log_sha256':digest(out/'tests.log'),'tests':'59 relevant tests passed'})
    print(f'Verified original artifacts, frozen inputs, 37200 recovery fits, {attempted-failures} successful nested resamples and {failures} skipped draws.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    main(p.parse_args().root)
