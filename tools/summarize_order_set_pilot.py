"""Write the review packet and scientific figures for the order/SET experiment."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/glacier-offsets-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import norm
from glacier_offsets.velocity_io import write_json,digest

ROOT=Path('.runtime/velocity-recovery-larsen');OUT=ROOT/'2017-order-set'
REGIONS=['target_1','target_2','target_3','target_4','control_1','control_2']


def read(name):return json.loads((OUT/name).read_text())

def table(headers,records):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in records])


def wilson(k,n):
    z=norm.ppf(.975);p=k/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-half),float(center+half)]


def main():
    rows=read('observed.json')['rows'];observed={(r['region'],r['variant']):r for r in rows}
    recovery=read('recovery.json')['rows'];nested=read('nested-calibration.json')['rows'];correction=read('corrections.json')
    if len(set(r['case'] for r in recovery))!=7:raise ValueError('Recovery run is incomplete')
    nr=[]
    for region in REGIONS:
        for case in ('null','benchmark'):
            q=[r for r in nested if r['region']==region and r['case']==case];n=len(q)
            failures=sum(r['nested_p']<=.05 for r in q);coverage=sum(r['coverage'] for r in q)
            nr.append({'region':region,'case':case,'datasets':n,'nested_rejection_rate':failures/n,
                'nested_rejection_Wilson95':wilson(failures,n),'nested_coverage':coverage/n,
                'coverage_Wilson95':wilson(coverage,n),
                'median_covariance_trace_ratio_to_observed':float(np.median([r['cov_trace_ratio_to_observed'] for r in q])),
                'F_reference_rejection_rate':sum(r['F_reference_p']<=.05 for r in q)/n})
    for row in nr:
        with np.load(OUT/(row['case']+'-recovery.npz')) as saved:
            a=saved[row['region']+'__primary_azimuth'];delta=a[:,:2]-a[:,2:4]
        empirical=np.cov(delta.T);observed_cov=np.array(observed[row['region'],'primary_azimuth']['covariance'])
        row['empirical_error_cs_bias']=np.mean(delta,axis=0)
        row['empirical_error_covariance']=empirical
        row['empirical_trace_ratio_to_observed']=float(np.trace(empirical)/np.trace(observed_cov))
        row['nested_trace_ratio_to_empirical']=row['median_covariance_trace_ratio_to_observed']/row['empirical_trace_ratio_to_observed']
    # Null-transfer calibration is a sensitivity under that donor population.
    transfer=[]
    with np.load(OUT/'null-recovery.npz') as saved:
        for rid in REGIONS:
            r=observed[rid,'primary_azimuth'];inv=np.linalg.inv(r['covariance']);a=saved[rid+'__primary_azimuth'][:,:2]
            t=np.einsum('ni,ij,nj->n',a,inv,a);tobs=np.array(r['cs_m_per_day'])@inv@np.array(r['cs_m_per_day'])
            transfer.append({'region':rid,'observed_statistic':float(tobs),'null_transfer_p':float((1+np.sum(t>=tobs))/(len(t)+1)),'draws':len(t)})
    from glacier_offsets.velocity_systematics import holm
    for row,p in zip(transfer[:4],holm([r['null_transfer_p'] for r in transfer[:4]])):row['Holm_p_four_targets']=float(p)
    write_json(OUT/'calibration-summary.json',{'nested_rows':nr,'transfer_sensitivity':transfer,
        'limits':'Wilson bounds are per region; regions share synthetic field draws and are not pooled as independent replications; transfer p assumes the donor generator is applicable to ice'})
    # Observed conditional coefficient ellipses, with no amplitude/phase shortcut near zero.
    fig,axes=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for ax,rid in zip(axes.ravel(),REGIONS):
        for variant,color,label in [('no_order','#aaaaaa','SET, no order term'),('primary_azimuth','#14618a','SET + order term')]:
            r=observed[rid,variant];cov=np.array(r['covariance']);eig,vec=np.linalg.eigh(cov);j=np.argsort(eig)[::-1];eig=eig[j];vec=vec[:,j]
            angle=np.degrees(np.arctan2(vec[1,0],vec[0,0]));cs=r['cs_m_per_day'];width,height=2*np.sqrt(eig*r['cutoff95'])
            ax.add_patch(Ellipse(cs,width,height,angle=angle,edgecolor=color,facecolor=color,alpha=.18,lw=1.5))
            ax.plot(*cs,'o',color=color,ms=4,label=label)
        ax.axhline(0,color='.65',lw=.7);ax.axvline(0,color='.65',lw=.7);ax.set(xlim=(-.12,.17),ylim=(-.16,.12),title=rid.replace('_',' '),xlabel='C (m/day)',ylabel='S (m/day)');ax.set_aspect('equal')
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Satellite-azimuth Msf: observed-data 95% bootstrap ellipses\nModel conditional; transferred-control calibration fails for frozen ellipses',fontsize=12)
    fig.savefig(OUT/'observed-ellipses.png',dpi=180);fig.savefig(OUT/'observed-ellipses.pdf');plt.close(fig)
    # Studentized scans use refitted wild residuals, conditional on the observed screen.
    scan=read('studentized-scans.json')['rows'];scan_by={r['region']:r for r in scan}
    with np.load(OUT/'studentized-scans.npz') as saved:
        fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
        for ax,rid in zip(axes.ravel(),REGIONS):
            periods=saved['period_days'];ax.plot(periods,saved[rid+'__statistic'],color='#14618a',lw=1.2)
            ax.axvline(14.765294,color='#ba6335',ls='--',lw=1,label='Msf')
            for p in (14.19,15.95):ax.axvline(p,color='.6',ls=':',lw=.8)
            ax.set(xlim=(12.05,45),yscale='log',ylim=(.03,250),xlabel='Period (days)',ylabel='Cluster Wald statistic',title=rid.replace('_',' '))
            ax.text(.98,.96,'Msf scan p = '+str(scan_by[rid]['Msf_max_scan_p']),ha='right',va='top',transform=ax.transAxes,fontsize=8)
        axes[0,0].legend(fontsize=8,loc='lower right');fig.suptitle('Conditional frequency scans: order + full ENU solid-Earth correction\n1999 studentized Webb draws; maximum over 403 periods within each region',fontsize=12)
        fig.savefig(OUT/'frequency-scans.png',dpi=180);fig.savefig(OUT/'frequency-scans.pdf');plt.close(fig)
    # Coverage comparison with explicit small-subset uncertainty.
    fig,ax=plt.subplots(figsize=(10,4.6),layout='constrained');x=np.arange(6)
    for offset,case,color in [(-.12,'null','#14618a'),(.12,'benchmark','#a75728')]:
        q=[next(r for r in nr if r['region']==rid and r['case']==case) for rid in REGIONS];v=np.array([r['nested_coverage'] for r in q]);bounds=np.array([r['coverage_Wilson95'] for r in q])
        ax.errorbar(x+offset,v,yerr=np.array([v-bounds[:,0],bounds[:,1]-v]),fmt='o',color=color,capsize=3,label='Nested '+case+' (20 datasets/region)')
    frozen=[next(r['frozen_ellipse_coverage'] for r in recovery if r['region']==rid and r['case']=='null' and r['variant']=='primary_azimuth') for rid in REGIONS]
    ax.scatter(x,frozen,marker='x',s=65,color='.3',label='Frozen observed ellipse, null (200 draws)');ax.axhline(.95,color='.5',ls='--',lw=1)
    ax.set(xticks=x,xticklabels=[r.replace('_',' ') for r in REGIONS],ylim=(.4,1.04),ylabel='Coefficient coverage',title='Transferred-control simulations: frozen versus recomputed uncertainty')
    ax.legend(fontsize=8,loc='lower left');fig.savefig(OUT/'uncertainty-calibration.png',dpi=180);fig.savefig(OUT/'uncertainty-calibration.pdf');plt.close(fig)
    # Review packet with results derived directly from output files.
    lines=['# 2017 Larsen: spacecraft order and solid-Earth tide experiment','',
        'This follow-up implements the next experiment recommended in [Opus feedback](OPUS_FEEDBACK_2026-10-29.md). It uses the frozen 2017 observations only. All new model and family choices are retrospective. The 2018 record was neither read nor processed. The outputs are under `.runtime/velocity-recovery-larsen/2017-order-set/`.','',
        'The spacecraft-order term substantially reduces the conditional Msf uncertainty, and all four ice targets reject the specified-frequency null in the observed-data four-target Holm family. Both bedrock controls remain consistent with zero. These are conditional statistical results: transferring corrected control fields into the targets fails the frozen-ellipse calibration gate, particularly at target 4. Recomputing uncertainty within synthetic datasets produces much wider intervals. Ocean loading remains uncorrected, and an Msf-frequency horizontal signal cannot be separated from aliased horizontal M2 with this acquisition pattern.','',
        '## 1. Inputs and exact deterministic correction','',
        f'The source file is `frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc`, SHA-256 `{correction["input"]["sha256"]}`. Its source-observation cache is checksum verified. The existing CATS2008_v2023 vertical ocean-height correction and empirical swath correction remain applied. Existing outputs and production products are preserved.','',
        'PySolid runs in `isce3`; inference runs in `ice`, with no dependency overlay. The tide kernel is evaluated at each unique source longitude/latitude and each owning-burst endpoint, using bracketing UTC seconds with linear interpolation for the fractional second. The later-minus-earlier full ENU displacement is projected through each source’s stored 2×3 geometry and subtracted in both radar channels. The ocean flexure factor is not applied to body tides. No minute rounding, scene-time substitution in the fit, or spatial tide-grid approximation is used. The independent point-kernel comparison differs by at most '+f'{correction["point_kernel_max_difference_m"]:.2g} m'+'. The correction RMS is about 0.047 m in range and 0.028 m in azimuth.','',
        'The displacement convention follows PySolid. Its compatibility with upstream geolocation and the vertical reference of the ocean-height model has not been independently established for floating ice. The bedrock diagnostics are useful checks, rather than a complete convention validation.','',
        'No local ocean-load coefficient set was found in the searched geodesy, glacier, InSAR and MintPy data trees. The installed pyTMD software alone does not supply those model data. Ocean loading is therefore **not applied**, and no loading amplitude is inferred from the ocean-height CATS file. References: [PySolid ENU documentation](https://github.com/insarlab/PySolid), [pyTMD model-data requirements](https://pytmd.readthedocs.io/en/latest/getting_started/Getting-Started.html).','',
        '## 2. Observation model and primary estimand','',
        'The primary fit uses satellite-azimuth displacement:','',
        '`d_az(s,k) = v_az(s) Δt(s,k) + H_Msf(s,k)·[C,S] + b_burst(s) q_k + error(s,k)`','',
        '`q_k = 1[later=S1B] - 1[earlier=S1B]`. Thus the fitted order coefficient is a displacement in metres, with +b on A→B and −b on B→A. Each region has one coefficient per owning burst represented in that region. It is estimated jointly with Msf; no empirical order correction is subtracted using a fit to the test data. Same-spacecraft pairs would have q=0. The actual 2017 record has 30 pairs of each cross-spacecraft order.','',
        'The two Msf coefficients are shared within a region, while secular velocity is source specific. The velocity basis is analytically interval integrated with P=14.765294 days. The phase origin is 2000-01-01 00:00 UTC; velocity is C cos(ωt)+S sin(ωt), amplitude is hypot(C,S), and phase is atan2(S,C). Amplitudes are instantaneous m/day; no second attenuation correction is applied. Azimuth phase differs by approximately 180° from along-flow phase where the flow projects negatively into azimuth. Phase is not interpreted for controls with amplitudes consistent with zero.','',
        'Comparison fits include azimuth converted to along-flow, joint RA with fixed vertical correction, and joint RA with four independent residual vertical coefficients per source (M2 cosine/sine and O1 cosine/sine, in height metres). Joint fits retain source-specific EN secular motion. The full RA covariance is Cholesky whitened. Source backgrounds are eliminated by a normalized Schur complement; burst order is then marginalized from the two-coefficient harmonic information. No free secular U or horizontal M2/O1 is fitted. Direct azimuth avoids a flow restriction, but it is still a radar projection rather than an independently resolved horizontal vector.','',
        'The primary family comprises the four originally selected ice targets, with Holm at 0.05. The controls are negative controls and the alternative estimators are sensitivity checks. This replaces the earlier mixed 18-test family **for this exploratory follow-up**; it does not retroactively preregister the choice.','',
        '## 3. Screening, support, noise and resampling','',
        'The peak threshold remains 0.08, with no SNR bypass. The supplementary screen rejects absolute range residuals >2 m or azimuth residuals >5 m. Its secular/order baseline is trained in acquisition-purged 60-day folds, using robust positive- and negative-order medians. It allows source-specific order amplitudes to keep screening tolerant of local offsets; the science estimator uses shared burst coefficients. No harmonic amplitude enters screening. Sources need at least 20 retained observed pairs; bootstrap fits require at least eight distinct selected epochs per contributing source.','',
        table(['Region','Sources','3 km blocks','Clean / valid source-pairs'],[[r['region'],r['supported_sources'],r['spatial_blocks'],str(r['retained_observations'])+' / '+str(r['valid_observations'])] for r in read('support.json')]),'',
        'The shared bootstrap calendar now uses all 60 scene pairs for block labels. Missing burst endpoints remain missing in the fits and masks. An initial scan used a region-specific incomplete calendar; it is retained only in files labelled `initial-calendar`. All final observed, recovery, sensitivity and studentized-scan results use the complete calendar. This prevents one region’s gaps from deleting valid dates in another region.','',
        'Control covariance is recalibrated on H1 controls after SET and train-only order/secular removal, using the existing spatially balanced, winsorized covariance procedure. The resulting range/azimuth standard deviations are '+', '.join(f'{v:.3f}' for v in np.sqrt(np.diag(read('clean_noise.json')['pooled']['covariance_ra_m2'])))+' m. H2 control fields form the noise-transfer bank. H1 shared burst order and source secular predictions are removed; any H2 control Msf is retained. H2 source medians are centered, but low-quality residuals and failures are retained together with validity, SNR and peak. Donor spatial blocks and common calendar moving blocks are transferred jointly. Known target order offsets are reinserted at the target pair order before refitting.','',
        'Observed uncertainty uses 1000 paired temporal (30-day) and physical spatial (3 km) block resamples. The frozen screening algorithm is rerun with each draw’s training counts; flow axes are reassessed for the along-flow comparisons. Cached sorting accelerates medians without freezing screen decisions. P-values use centered two-coefficient bootstrap statistics with a +1 Monte Carlo correction. Zero exceedances means p=1/1001, not a known p below 10⁻³.','',
        '## 4. Observed estimates and correction sensitivity','',
        table(['Region','Azimuth A ± σ, m/day','Along-flow A, m/day','Residual-vertical A, m/day','Azimuth p','Holm p'],[[rid,f"{observed[rid,'primary_azimuth']['amplitude_m_per_day']:.4f} ± {observed[rid,'primary_azimuth']['sigma_amplitude']:.4f}",f"{observed[rid,'azimuth_along']['amplitude_m_per_day']:.4f}",f"{observed[rid,'residual_vertical']['amplitude_m_per_day']:.4f}",f"{observed[rid,'primary_azimuth']['p']:.4f}",f"{observed[rid,'primary_azimuth'].get('holm_p_four_targets',float('nan')):.4f}" if rid.startswith('target') else 'control'] for rid in REGIONS]),'',
        'σ is the delta-method amplitude standard deviation from the joint coefficient covariance, not a symmetric 95% amplitude interval. The NPZ and JSON retain the joint distributions and percentile intervals. The direct azimuth and along-flow columns have different units of projection and should not be compared as if they estimated the same component.','',
        table(['Region','Order omitted: az A / σ','Order fitted: az A / σ','SET-induced coefficient-vector change, m/day'],[[rid,f"{observed[rid,'no_order']['amplitude_m_per_day']:.4f} / {observed[rid,'no_order']['sigma_amplitude']:.4f}",f"{observed[rid,'primary_azimuth']['amplitude_m_per_day']:.4f} / {observed[rid,'primary_azimuth']['sigma_amplitude']:.4f}",f"{np.linalg.norm(np.array(observed[rid,'no_SET']['cs_m_per_day'])-observed[rid,'primary_azimuth']['cs_m_per_day']):.5f}"] for rid in REGIONS]),'',
        'The order term reduces target azimuth σ from roughly 0.032–0.034 to 0.006–0.009 m/day. SET changes the azimuth coefficient vector by about 0.008 m/day, consistent with the scale identified in the independent review. Its effect on scalar amplitude depends on relative phase, so subtracting amplitudes is not a valid correction. Residual-vertical and azimuth-along estimates agree closely after order and SET are included.','',
        table(['Region','Burst indices','Order coefficients, m'],[[rid,observed[rid,'primary_azimuth']['burst_indices'],', '.join(f'{b:.3f}' for b in observed[rid,'primary_azimuth']['order_m'])] for rid in REGIONS]),'',
        table(['Region','Residual M2 / O1 height amplitude of mean coefficients, m','Held-out range RMS: fixed / residual vertical, m'],[[rid,' / '.join(f"{observed[rid,'residual_vertical'][key]:.3f}" for key in ('vertical_M2_mean_amplitude_m','vertical_O1_mean_amplitude_m')),' / '.join(f"{np.sqrt(np.average([r['range_rms_m']**2 for r in read('predictive.json')['rows'] if r['region']==rid and r['variant']==m],weights=[r['test_observations'] for r in read('predictive.json')['rows'] if r['region']==rid and r['variant']==m])):.3f}" for m in ('fixed_vertical','residual_vertical'))] for rid in REGIONS]),'',
        'Residual height amplitudes are norms of block-balanced mean source coefficients, not means of source amplitudes or measured total ocean tides. The fitted vertical terms can also absorb other errors with the same sampled patterns; nonzero control coefficients do not uniquely identify vertical deformation or ocean-tide error.','',
        'The actual raw-minus-swath-corrected azimuth data were restored on the identical screened support. The largest resulting Msf coefficient change is about 1.6×10⁻⁵ m/day. This is an empirical restoration check, including real spacing variation, rather than an assertion that every possible correction lies exactly in the burst-order basis. A separate mathematical test verifies exact invariance for a correction that does lie in that basis.','',
        '![Observed ellipses](../../.runtime/velocity-recovery-larsen/2017-order-set/observed-ellipses.png)','',
        '## 5. Prediction, calendar-block sensitivity and midyear changes','',
        'Six acquisition-purged 60-day folds use train-only screen predictions and axes, with matched test support across variants and at least 12 train epochs per source. These RMS values refer to individual source observations; they are not comparable to regional-median RMS values in the independent review.','']
    cv=read('predictive.json')['rows'];records=[]
    for rid in REGIONS:
        record=[rid]
        for m in ('no_order','primary_azimuth'):
            q=[r for r in cv if r['region']==rid and r['variant']==m]
            record.append(f"{np.sqrt(np.average([r['azimuth_rms_m']**2 for r in q],weights=[r['test_observations'] for r in q])):.3f}")
        records.append(record)
    lines += [table(['Region','Held-out az RMS without order, m','With order, m'],records),'',
        'The reduction appears in held-out observations as well as in fit uncertainty. Residual-vertical fits retain the range-channel improvement from the earlier experiment, while azimuth Msf remains close to the direct-channel comparison. This is prediction under the frozen H1 noise covariance, rather than fresh leave-control-out validation of the complete calibration pipeline.','',
        'Block-length checks use 500 rescreened draws each at 15, 30 and 60 days. Target_2/3/4 amplitude σ changes little; target_1 retains more sensitivity.','',
        table(['Region','σ at 15 / 30 / 60 days, az m/day'],[[rid,' / '.join(f"{next(r['sigma_amplitude'] for r in read('block-sensitivity.json')['rows'] if r['region']==rid and r['block_days']==d):.4f}" for d in (15,30,60))] for rid in REGIONS]),'',
        'A separate source-specific secular rate in each half-year, integrated exactly through pairs crossing July 1, changes the azimuth Msf coefficient vector by only 0.0005–0.0013 m/day on the ice targets. This particular background change therefore does not explain the fitted Msf. No new rift/imagery intersection check was performed; the split is a calendar sensitivity, not a causal calving attribution.','',
        table(['Region','H1 / H2 retained fraction of valid observations'],[[r['region'],' / '.join(f'{v:.3f}' for v in r['H1_H2_retained_fraction_of_valid'])] for r in read('correction-secular-sensitivity.json')['rows']]),'',
        'The H2 retention decline in targets 1–3 and control 1 shows that observation quality is not stationary across regions or seasons. Target 4 remains comparatively steady. Screening fractions use valid observations as their denominator, so missing burst support is a separate limitation.','',
        '## 6. Frequency scans','',
        'Two conditional diagnostics are retained. `scans.json` uses null-residual cluster-score covariance and is conservative under a strong alternative because the null residual contains the signal. `studentized-scans.json` uses an unrestricted cluster covariance at each period, and recomputes that covariance after refitting every wild replicate. The latter is the main displayed scan. Both retain order nuisance terms. Neither scan is used to choose the primary period, screen or ROI family.','',
        'The studentized scan covers 403 periods from 12.05 to 45 days, including exact Msf and approximate O1/N2 alias periods. It uses 1999 Webb multiplier vectors shared across frequencies and about 12 calendar blocks. Source secular and burst order are refitted in each replicate; CR1 is B/(B−1). The implementation uses linear residual projections to reproduce explicit refits, checked by an independent numerical test. Screens are held fixed in this secondary scan. Its maximum-period p-value is corrected within a region only, without a further across-region scan-family adjustment.','',
        table(['Region','Largest period, days','Msf pointwise wild p','Msf max-scan p'],[[r['region'],f"{r['largest_period_days']:.3f}",f"{r['Msf_pointwise_p']:.4f}",f"{r['Msf_max_scan_p']:.4f}"] for r in scan]),'',
        'All ice-region maxima lie close to Msf, within the approximate annual-record Fourier width P²/T ≈0.6 day, rather than establishing an exact constituent period from the scan. Target 3 does not pass the 0.05 within-region maximum-scan test, despite passing the specified-Msf test. The two controls have no comparable Msf support. This distinction prevents a fixed-frequency test from being presented as an unconstrained discovery.','',
        '![Frequency scans](../../.runtime/velocity-recovery-larsen/2017-order-set/frequency-scans.png)','',
        '## 7. Recovery and synthetic uncertainty calibration','',
        'Seven cases use 200 independent synthetic fields each, paired across four estimators and all six regions: null; 0.10 m/day along-flow Msf; each with 20% residual vertical tides (M2/O1 height amplitudes 0.12/0.04 m); benchmark plus horizontal M2 velocity 0.03 or 0.20 m/day; and Msf-null plus horizontal M2 velocity 0.20 m/day. Msf phases span the full cycle. Target geometry, burst endpoints, masks and deterministic target order are retained. Every injected dataset passes the same screen. The primary truth is the exact noiseless Msf-only regional projection on that draw’s support, which avoids approximating varying source geometry by one average projection.','',
        'The original recovery metrics reuse observed-data ellipses. They intentionally diagnose the transfer mismatch and are not treated as a calibrated confidence procedure for the donor population.','',
        table(['Region','Frozen ellipse null FPR','Frozen null coverage','Benchmark amplitude error, primary / residual-vertical','Benchmark phase error, primary / residual-vertical'],[[rid,f"{next(r['frozen_ellipse_null_FPR'] for r in recovery if r['region']==rid and r['case']=='null' and r['variant']=='primary_azimuth'):.3f}",f"{next(r['frozen_ellipse_coverage'] for r in recovery if r['region']==rid and r['case']=='null' and r['variant']=='primary_azimuth'):.3f}",' / '.join(f"{next(r['median_amplitude_error'] for r in recovery if r['region']==rid and r['case']=='benchmark' and r['variant']==m):.1%}" for m in ('primary_azimuth','residual_vertical')),' / '.join(f"{next(r['median_phase_error_deg'] for r in recovery if r['region']==rid and r['case']=='benchmark' and r['variant']==m):.1f}°" for m in ('primary_azimuth','residual_vertical'))] for rid in REGIONS]),'',
        'Frozen-null FPR is above the 5% gate, especially target 4 and control 1. Removing order therefore did not make the control-noise bank exchangeable with every target’s observed residuals. This does not by itself measure the real-ice false-positive rate: the donor and target noise populations differ. It does show that an observed ellipse cannot be used as the uncertainty for every transferred dataset.','',
        'The nested check recomputes covariance and bootstrap radius in 20 null and 20 benchmark datasets per region, with 199 resamples per dataset. The null subset uses the first 20 independently generated fields; benchmark indices are equally spaced across the full 200-draw phase cycle. That is 240 synthetic regional datasets and 47,760 attempted nested resamples. Region draws share the synthetic field construction, so they are not pooled as 240 independent replicates. The frozen screen is rerun in every resample. Along-flow direction is not needed by the primary estimator.','',
        table(['Region','Nested null FPR','Nested null / benchmark coverage','Median covariance trace ratio: null / benchmark'],[[rid,f"{next(r['nested_rejection_rate'] for r in nr if r['region']==rid and r['case']=='null'):.2f}",' / '.join(f"{next(r['nested_coverage'] for r in nr if r['region']==rid and r['case']==case):.2f}" for case in ('null','benchmark')),' / '.join(f"{next(r['median_covariance_trace_ratio_to_observed'] for r in nr if r['region']==rid and r['case']==case):.2f}" for case in ('null','benchmark'))] for rid in REGIONS]),'',
        table(['Region','Null empirical covariance / observed','Null nested covariance / empirical'],[[rid,*[f"{next(r[key] for r in nr if r['region']==rid and r['case']=='null'):.2f}" for key in ('empirical_trace_ratio_to_observed','nested_trace_ratio_to_empirical')]] for rid in REGIONS]),'',
        'The second comparison uses coefficient-error covariance across all 200 synthetic fields, rather than the observed ellipse. Null donor variance is close to the observed trace at targets 1–2, moderately higher at target 3, and about 2.70 times higher at target 4. Median nested-bootstrap variance is approximately 1.5–2.1 times the empirical null variance across regions. Thus the nested experiment indicates both target-dependent noise mismatch and a conservative bootstrap construction under this generator. The 100% nested coverage should not be called a successful nominal-95% calibration. These ratios compare traces; covariance orientation and tail shape also affect coverage.','',
        'Per-region nested rates have only 20 datasets: 0/20 rejections has a Wilson 95% interval of approximately 0–16%. These checks can separate severe frozen-ellipse mismatch from recomputed uncertainty, but cannot establish a 5% error rate. The Hotelling-style F(2,B−2) reference is also recorded and compared empirically; the bootstrap covariance is not assumed to obey an exact Hotelling distribution. No small F-tail probability is substituted for the finite Monte Carlo result.','',
        'As another sensitivity, the empirical transferred-null distribution can be evaluated at the observed coefficient statistic. This is conditional on applicability of that donor generator to ice, and is not used to replace the primary p-value after inspecting results.','',
        table(['Region','Transferred-null p (200 fields)','Four-target Holm p'],[[r['region'],f"{r['null_transfer_p']:.4f}",f"{r['Holm_p_four_targets']:.4f}" if 'Holm_p_four_targets' in r else 'control'] for r in transfer]),'',
        '![Uncertainty calibration](../../.runtime/velocity-recovery-larsen/2017-order-set/uncertainty-calibration.png)','',
        '## 8. Aliases, remaining limits and takeaways','',
        'The horizontal M2 stress uses instantaneous velocity amplitudes, not height/displacement amplitudes. With regular six-day sampling the apparent Msf velocity is approximately M2/28.53: 0.03 and 0.20 m/day M2 correspond to about 0.0011 and 0.0070 m/day apparent Msf before projection. Exact burst-time perturbations provide little practical separation. A period peak near Msf cannot distinguish these components, and an astronomically locked systematic can repeat in 2018. The Rutford horizontal-Msf model did not independently measure the absence of horizontal M2. Any weak-M2 assumption remains a physical prior.','',
        table(['Region','Benchmark + M2=.03: amplitude / phase error','Benchmark + M2=.20: amplitude / phase error'],[[rid,*[f"{next(r['median_amplitude_error'] for r in recovery if r['region']==rid and r['case']==case and r['variant']=='primary_azimuth'):.1%} / {next(r['median_phase_error_deg'] for r in recovery if r['region']==rid and r['case']==case and r['variant']=='primary_azimuth'):.1f}°" for case in ('benchmark_M2_003','benchmark_M2_020')]] for rid in REGIONS]),'',
        'The main supported change is the treatment of spacecraft order as a joint nuisance: it explains much of the previous bootstrap inflation, improves held-out prediction, and makes the along-flow and residual-vertical fits agree closely. Full ENU SET is measurable at this precision and changes the estimated harmonic. Direct azimuth provides a useful primary estimand without a directional restriction.','',
        'The observed record contains stronger, frequency-specific Msf evidence after those corrections, but the full acceptance gate is not met. Frozen-ellipse error rates fail under control transfer, the nested subset is small, ocean loading is missing, the body/ocean/geolocation convention needs further validation, and aliased horizontal M2 remains unresolved. The older directional and cross-flow sensitivity results still limit physical along-flow interpretation, particularly in slower ice. No production velocity map or physical tide-modulation detection is certified by this run.','',
        'A prospective 2018 analysis should state a final primary family, nuisance/correction conventions, year-specific secular/order terms, quality handling and uncertainty procedure before inspecting its Msf results. The present recipe is a recorded candidate; its failed calibration gate means it is not yet a fully validated confirmatory procedure. The eventual 2017/2018 record-length comparison should use compatible extended-grid products.','',
        '## 9. Reproduction and verification','',
        '```bash',
        '/Users/briel/miniforge3/envs/isce3/bin/python tools/prepare_order_set_experiment.py',
        'PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 /Users/briel/miniforge3/envs/ice/bin/python tools/run_order_set_pilot.py --phase all',
        'PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 /Users/briel/miniforge3/envs/ice/bin/python tools/scan_order_set_studentized.py',
        'PYTHONPATH=src VECLIB_MAXIMUM_THREADS=1 /Users/briel/miniforge3/envs/ice/bin/python tools/check_order_set_sensitivities.py',
        'PYTHONPATH=src /Users/briel/miniforge3/envs/ice/bin/python tools/summarize_order_set_pilot.py',
        'PYTHONPATH=src:tests /Users/briel/miniforge3/envs/ice/bin/python -m unittest test_velocity_systematics test_velocity_vertical test_velocity_quality test_velocity_recovery',
        'PYTHONPATH=src /Users/briel/miniforge3/envs/ice/bin/python tools/verify_order_set_pilot.py',
        '```','',
        'The tide preparation is separate so native/Fortran dependencies do not enter the fitting environment. Simulation checkpoints are resumed only within the recorded recipe. Final provenance records code and artifact hashes. `fitted-support.npz` records exact source IDs, row/column, masks, geometry, endpoints, blocks and axes; `fitted-coefficients.npz` records all source nuisance coefficients and predictions. Independent reconstruction reproduces all 36 fitted predictions to floating-point precision.','',
        'Tests cover recovery of a known order+vertical+Msf signal, agreement with dense weighted least squares under missingness and multiplicities, exact calibration-basis invariance, screen recovery of a large order term and rejection of failed matches, cached/uncached screen equivalence with burst gaps, and agreement between the optimized wild residual projection and explicit refitting. The relevant 52-test temporal velocity suite passes, including the new systematics tests. A broader `test_velocity*.py` discovery also included seven native-matcher tests that cannot run in `ice` because ISCE3 is absent there; those are unrelated to temporal inference.','']
    doc=Path('docs/velocity-recovery/ORDER_SET_EXPERIMENT_2026-10-29.md');doc.write_text('\n'.join(lines))
    (OUT/'report.md').write_text('\n'.join(lines).replace('(OPUS_FEEDBACK_2026-10-29.md)','(../../../docs/velocity-recovery/OPUS_FEEDBACK_2026-10-29.md)').replace('(../../.runtime/velocity-recovery-larsen/2017-order-set/','('))
    print(doc.resolve())

if __name__=='__main__':main()
