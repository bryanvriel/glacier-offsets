"""Summarize a finished pilot without reopening its displacement inputs."""
import argparse
import json
from pathlib import Path

import h5netcdf
import numpy as np
from glacier_offsets.velocity_io import write_json


def summarize(directory):
    directory=Path(directory)
    pilot=json.loads((directory/'pilot.json').read_text())
    validation=json.loads((directory/'validation_recovery.json').read_text())
    tuning=json.loads((directory/'tuning_recovery.json').read_text())
    uncertainty=json.loads((directory/'paired_intervals.json').read_text())
    noise=json.loads((directory/'noise.json').read_text())
    cv=json.loads((directory/'cross_validation.json').read_text())
    calibration_path=directory/'calibration-validation.json'
    calibration=json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    regions=sorted({p['region'] for p in pilot['sampling'] if p['role']=='target'})
    weights=[]
    for attempt in tuning['attempts']:
        if attempt['candidate']['background']!='constant':
            continue
        null=next((r for r in attempt['rows'] if r.get('amplitude')==0 and r.get('draws',0)>0),None)
        if null:
            weights.append({'model':attempt['model'],'weighting':attempt['candidate']['weighting'],
                            'region':null['region'],'source_id':null['source_id'],'draws':null['draws'],
                            'tuning_null_false_positive_rate':null['null_false_positive_rate'],
                            'tuning_coverage_95':null['coverage_95']})
    rows=[]
    for model in ('free_en','along_flow'):
        with h5netcdf.File(directory/f'pilot_{model}.nc') as h:
            for region in regions:
                points=[p for p in pilot['sampling'] if p['region']==region]
                indices=h['target_parameter_index'][:].astype(int)
                cs=[]
                for point in points:
                    r,c=point['row'],point['col'];m=h['coefficients'][:,r,c][indices]
                    direction=h['flow_direction'][:,r,c]
                    cs.append(direction @ m.reshape(2,2) if model=='free_en' else m)
                amplitude=float(np.linalg.norm(np.mean(cs,axis=0)))
                recovery=[r for r in validation['rows'] if r.get('region')==region and r.get('model')==model
                          and r.get('period')=='joint' and r.get('case')=='baseline' and r.get('draws',0)>0]
                benchmark=[r for r in recovery if r['amplitude']==.1]
                null=[r for r in recovery if r['amplitude']==0]
                passed_amplitudes=[]
                for a in (.025,.05,.1,.2):
                    group=[r for r in recovery if r['amplitude']==a]
                    if len(group)==len(points) and all(r['median_amplitude_error']<=.2 and r['median_phase_error_degrees']<=20 for r in group):
                        passed_amplitudes.append(a)
                interval=uncertainty.get(region,{}).get(model,{})
                row={'region':region,'model':model,'unique_source_blocks':len(points),
                     'conditional_mean_along_amplitude_m_per_day':amplitude,
                     'regional_bootstrap_amplitude_95_m_per_day':interval.get('amplitude_95'),
                     'benchmark_median_relative_amplitude_error':float(np.median([r['median_amplitude_error'] for r in benchmark])) if benchmark else None,
                     'benchmark_median_circular_phase_error_deg':float(np.median([r['median_phase_error_degrees'] for r in benchmark])) if benchmark else None,
                     'null_false_positive_rate_median':float(np.median([r['null_false_positive_rate'] for r in null])) if null else None,
                     'null_false_positive_rate_range':[min(r['null_false_positive_rate'] for r in null),max(r['null_false_positive_rate'] for r in null)] if null else None,
                     'coefficient_ellipse_coverage_95_range':[min(r['coverage_95'] for r in recovery),max(r['coverage_95'] for r in recovery)] if recovery else None,
                     'smallest_tested_amplitude_meeting_error_criteria':min(passed_amplitudes) if passed_amplitudes else None,
                     'error_criteria_are_not_full_detection_validation':True}
                rows.append(row)
    output={'scope':'2017 only','map_allowed':pilot['map_allowed'],'tuning_passed':tuning['passed'],
            'acceptance':pilot['acceptance'],'rows':rows,'constant_background_weighting_checks':weights,
            'constant_control_sigma_ra_m':np.sqrt(np.diag(noise['covariance_ra_m2'])),
            'conditional_quality_floors_ra_m':noise['floors_m'],'calibration':calibration,
            'numerical_fit_failures':{m:len(r.get('failed_fits',[])) for m,r in cv['models'].items()},
            'interpretation':'Conditional Msf-frequency diagnostics; no detection claim when recovery gates fail'}
    write_json(directory/'summary.json',output)
    _plot(directory,rows,regions,pilot['map_allowed'])
    if weights:
        status='no model passed' if not any(a['passed'] for a in tuning['attempts']) else ('tuning gates passed' if tuning['passed'] else 'paired tuning gates not passed')
        _weight_plot(directory,weights,status)
    lines=['# 2017 Msf velocity pilot','',
           'Production mapping is '+('allowed by the pilot gates.' if pilot['map_allowed'] else '**blocked by the pilot gates**.'),'',
           'The four ice targets and two stable-bedrock controls were approved by the user on 2026-10-28. '
           'The pilot uses 18 unique sources separated by at least 3 km, exact pair endpoints, '
           'six acquisition-purged predictive folds, the full 80-candidate bank, 200 independent validation draws per case '
           'and 1,000 paired temporal/spatial bootstrap draws.','']
    if not tuning['passed']:
        lines+=['No candidate in the one-standard-error shortlist passed all tuning recovery checks. '
                'The stored fits retain the simplest predictive choice (constant baseline weights) for diagnosis, '
                'not as a recommended model or a validated detection.','',
                'The following paired tuning checks on the first ice source show that robust/quality weights help '
                'but do not reach the 5% null false-positive requirement. These use the independent tuning stream, '
                'not the final 200-draw validation stream.','',
                '| Model | Weighting | Tuning draws | Null false positives | Nominal 95% coverage |',
                '|---|---|---:|---:|---:|']
        for r in weights:
            lines.append(f'| {r["model"]} | {r["weighting"]} | {r["draws"]} | '
                         f'{100*r["tuning_null_false_positive_rate"]:.1f}% | {100*r["tuning_coverage_95"]:.1f}% |')
        lines+=['']
        lines+=['![Paired weighting checks](weighting_checks.png)','',
                'Coverage refers to the common along-flow cosine/sine coefficient ellipse for both parameterizations.','']
    lines+=['![Pilot summary](pilot_overview.png)','',
            '| Region | Model | Conditional amplitude (m/day) | Bootstrap 95% interval | 0.10 m/day amplitude error | Phase error | Null false positives |',
            '|---|---|---:|---|---:|---:|---:|']
    for r in rows:
        ci=r['regional_bootstrap_amplitude_95_m_per_day'];interval=f'{ci[0]:.3f}–{ci[1]:.3f}' if ci else 'unavailable'
        def percent(value):return 'unavailable' if value is None else f'{100*value:.1f}%'
        phase=r['benchmark_median_circular_phase_error_deg']
        lines.append(f'| {r["region"]} | {r["model"]} | {r["conditional_mean_along_amplitude_m_per_day"]:.3f} | {interval} | '
                     f'{percent(r["benchmark_median_relative_amplitude_error"])} | '+('unavailable' if phase is None else f'{phase:.1f}°')+
                     f' | {percent(r["null_false_positive_rate_median"])} |')
    sr,sa=output['constant_control_sigma_ra_m']
    lines+=['','Table recovery scores are medians over the three source blocks per region; the gate checks each block. '
            'Instantaneous amplitudes and bootstrap intervals are conditional fits. Unsupported phase interpretation is suppressed.','',
            f'The constant control-noise baseline is {sr:.3f} m range and {sa:.3f} m azimuth. '
            'Quality-dependent covariance and paired Huber weighting are empirical estimates; SNR/peak are not inverse variances.','']
    if calibration:
        temporal=calibration['held_out_acquisitions'];ci=temporal['mae_change_95_px']
        lines += [f'The published spacecraft coefficient was checked on {temporal["test_links"]} links with no shared training acquisitions. '
                  f'The change in mean absolute relative-swath error is {temporal["mae_change_px"]:.6f} px '
                  f'(dependence interval {ci[0]:.6f} to {ci[1]:.6f} px). '
                  'The coefficient and split-period estimates are compatible; improvement remains inconclusive when the interval includes zero. '
                  'The original calibration and input netCDF were not modified.','']
    lines+=['Files: [full validation](validation_recovery.json), [tuning](tuning_recovery.json), '
            '[paired intervals](paired_intervals.json), [cross-validation](cross_validation.json), '
            '[calibration assessment](calibration-validation.json), '
            '[free EN coefficients](pilot_free_en.nc), [along-flow coefficients](pilot_along_flow.nc).','',
            'Daily source curves are in `daily_*.png` and `daily_*.npz`. Full-grid daily cubes and production maps were not generated. '
            'Extension processing is being handled separately by the user.','',
            'The per-block acceptance failures remain in `pilot.json` and `summary.json`. '
            'Results describe Msf-frequency motion conditional on correction and nuisance-signal assumptions.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return output


def _plot(directory,rows,regions,passed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for model,color,delta in [('free_en','#305c9b',-.12),('along_flow','#b65b22',.12)]:
        group=[next(r for r in rows if r['region']==region and r['model']==model) for region in regions]
        x=np.arange(len(regions))+delta
        axes[0].scatter(x,[r['conditional_mean_along_amplitude_m_per_day'] for r in group],c=color,label=model)
        for xx,r in zip(x,group):
            if r['regional_bootstrap_amplitude_95_m_per_day']:
                axes[0].vlines(xx,*r['regional_bootstrap_amplitude_95_m_per_day'],color=color)
        axes[1].scatter(x,[100*r['benchmark_median_relative_amplitude_error'] if r['benchmark_median_relative_amplitude_error'] is not None else np.nan for r in group],c=color)
        axes[2].scatter(x,[100*r['null_false_positive_rate_median'] if r['null_false_positive_rate_median'] is not None else np.nan for r in group],c=color)
    axes[0].set(title='Conditional along-flow amplitude',ylabel='Instantaneous velocity (m/day)')
    axes[1].set(title='Recovery of a 0.10 m/day signal',ylabel='Median relative amplitude error (%)',yscale='symlog')
    axes[1].axhline(20,color='.45',ls='--',lw=1)
    axes[2].set(title='Null false positives',ylabel='False-positive rate (%)',ylim=(-2,102))
    axes[2].axhline(5,color='.45',ls='--',lw=1)
    for ax in axes:
        ax.set_xticks(range(len(regions)),[r.replace('target_','Region ') for r in regions],rotation=25)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.suptitle('2017 Msf pilot — '+('pilot gates passed' if passed else 'diagnostic fits; pilot gates not passed'))
    fig.savefig(directory/'pilot_overview.png',dpi=160)
    plt.close(fig)


def _weight_plot(directory,rows,status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    order=['constant','constant_robust','quality_robust','quality_rule']
    names=['Constant','Constant + robust','Quality + robust','Quality rule']
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for model,color,offset,label in [('free_en','#305c9b',-.18,'Free EN'),('along_flow','#b65b22',.18,'Along flow')]:
        group=[next((r for r in rows if r['model']==model and r['weighting']==weight),None) for weight in order]
        x=np.arange(4)+offset
        axes[0].bar(x,[100*r['tuning_null_false_positive_rate'] if r else np.nan for r in group],width=.34,color=color,label=label)
        axes[1].bar(x,[100*r['tuning_coverage_95'] if r else np.nan for r in group],width=.34,color=color)
    axes[0].axhline(5,color='#9a3333',ls='--',lw=1,label='5% requirement')
    axes[1].axhspan(90,99,color='.5',alpha=.15)
    axes[1].axhline(95,color='.4',ls='--',lw=1)
    axes[0].set_title('Null false positives');axes[1].set_title('Nominal 95% interval coverage')
    for ax in axes:
        ax.set(ylim=(0,102),ylabel='Percent')
        ax.set_xticks(range(4),names,rotation=18,ha='right')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Paired tuning checks: first ice source, {rows[0]["draws"]} draws; {status}')
    fig.savefig(directory/'weighting_checks.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory')
    print(json.dumps({'summary_rows':len(summarize(parser.parse_args().directory)['rows'])}))
