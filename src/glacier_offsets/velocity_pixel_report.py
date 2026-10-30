"""Write the two-year pixel-map interpretation and reproduction notes."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from glacier_offsets.velocity_io import write_json
from glacier_offsets.velocity_pixel_config import parameters_from_recipe


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])


def run(out):
    read=lambda name:json.loads((out/name).read_text())
    summary=read('map-summary.json');inputs=read('inputs.json');earth=read('solid_earth_tides.json');noise=read('noise.json');regions=read('regional-year-check.json')['rows'] if (out/'regional-year-check.json').exists() else []
    recipe=read('recipe.json');parameters=parameters_from_recipe(recipe);display=parameters['display']
    with h5py.File(out/'source_coefficients.h5') as h:
        masks={'moving display':h['standard_coverage'][:]&h['residual_vertical/estimable'][:]
              &(h['mean_flow_speed'][:]>=display['min_speed_m_day'])
              &(np.min(abs(h['azimuth_flow_projection_by_year'][:]),axis=1)>=display['min_abs_azimuth_projection'])}
        mask=masks['moving display'];phase=h['residual_vertical/phase_deg'][:];amp=h['residual_vertical/amplitude'][:]
        resultant=np.mean(np.exp(1j*np.radians(phase[mask])))
        stats={'circular_mean_phase_deg':float(np.degrees(np.angle(resultant))),'circular_resultant_length':float(abs(resultant)),
            'amplitude_above_plot_limit_fraction':float(np.mean(amp[mask]>.2)),
            'fit_support_sources':int(h['fit_support'][:].sum()),'estimable_sources':int(h['residual_vertical/estimable'][:].sum())}
        # Compare the estimands after accounting for signed azimuth projection.
        q=np.mean(h['azimuth_flow_projection_by_year'][:],axis=1)
        other=h['azimuth_direct/cs'][:];joint=h['residual_vertical/cs'][:]
        strong=mask&(abs(q)>=.5)&np.isfinite(other).all(1)
        delta=joint[strong]-other[strong]/q[strong,None]
        stats['median_joint_minus_azimuth_converted_coefficient_norm_m_day']=float(np.median(np.linalg.norm(delta,axis=1)))
    write_json(out/'interpretation-statistics.json',stats)
    # New configurable experiments get their own factual report. Preserve the
    # historical checkpoint narrative instead of overwriting it with a rerun.
    lines=['# Configured 2017/2018 exploratory pixel inversion','',
        '![Along-flow Msf amplitude and phase](msf_quicklook.png)','',
        recipe['model']+'.',recipe['comparator']+'.','',
        'These are exploratory coefficients. Pixelwise uncertainty and display filters are diagnostics, not calibrated detections.','',
        '## Effective settings','',
        'Screen: '+recipe['screen']+'.','',
        'Remove mixed windows: '+str(parameters['screening']['remove_mixed_windows']).lower()+'.','',
        'Fit support: '+recipe['min_fit']+'.','',
        'Standard map coverage: '+recipe['standard_coverage']+'.','',
        'Uncertainty: '+recipe['diagnostic_uncertainty']+'.','',
        'Display filters: `'+json.dumps(display,sort_keys=True)+'`.','',
        'The prepared observations, CATS/body-tide corrections and frozen yearly H1 control covariances are reused. Changing screening does not retrain those noise weights.','',
        '## Results','',
        table(['Selection','Unique sources','Native cells'],[[name,summary[name]['unique_sources'],summary[name]['map_cells']]
            for name in ('source_coverage','along_display','phase_quality_display','fast_stable_flow_proxy')]),'',
        'Amplitude percentiles (10, 25, 50, 75, 90, 95, 99), m/day: `'+json.dumps(summary['along_amplitude_percentiles_m_day'])+'`.','',
        'Interpretation statistics: `'+json.dumps(stats,sort_keys=True)+'`.','',
        ('Separate-year regional comparisons are in `regional-year-check.json`; these descriptive checks have no new bootstrap or significance test.' if regions else 'No separate-year regional comparisons were run for this output.'),'',
        '## Reproduction','',
        'The installed command works from any directory; the resolved YAML records absolute paths.','',
        '```bash',
        (f'glacier-offsets velocity-pixels fit "{out.resolve()}/inversion-config.resolved.yaml"' if (out/'inversion-config.resolved.yaml').exists() else f'python -m glacier_offsets.velocity_pixel_inversion "{out.resolve()}"'),
        f'glacier-offsets velocity-pixels export "{out.resolve()}"',
        '```','',
        'The recipe binds the effective scientific settings, numerical code, prepared inputs and noise weights. Verification results are recorded separately in `verification.json`.','']
    (out/'report.md').write_text('\n'.join(lines));print((out/'report.md').resolve());return

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);run(p.parse_args(argv).output.resolve())


if __name__=='__main__':main()
