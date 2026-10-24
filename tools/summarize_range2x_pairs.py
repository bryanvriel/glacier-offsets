#!/usr/bin/env python3
"""Summarize range-2x versus deramp=0 across completed pair experiments.

Run as python -m tools.summarize_range2x_pairs SPEC.json --output REPORT.json.
SPEC lists, per pair: the deramp=0 baseline pair directory, the compare_swath_overlap
--compare-deramp output, its compare_swath_range_oversampling output, and the
compare_merged_preprocessing output. Statistics are descriptive; spatially
overlapping windows are grouped by burst pair for confidence intervals.
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from glacier_offsets.products import write_json

RULES = {'peak>=0.3': lambda pk, snr: pk >= .3, 'SNR>=8': lambda pk, snr: snr >= 8,
         'peak>=0.3 & SNR>=8': lambda pk, snr: (pk >= .3) & (snr >= 8),
         '(peak>=0.3 & SNR>=8) | SNR>=20': lambda pk, snr: ((pk >= .3) & (snr >= 8)) | (snr >= 20)}


def frac(x):
    return x-np.round(x)


def sine_fit(y, phase):
    good=np.isfinite(y)&np.isfinite(phase);y=np.asarray(y)[good];phase=np.asarray(phase)[good]
    x = np.sin(2*np.pi*phase);design = np.c_[np.ones_like(x), x]
    if y.size<3 or np.ptp(x)==0:
        return {'n':int(y.size),'amplitude_px':None,'intercept_px':None,'pearson_r':None}
    (intercept, amplitude), *_ = np.linalg.lstsq(design, y, rcond=None)
    return {'n': int(y.size), 'amplitude_px': float(amplitude), 'intercept_px': float(intercept),
            'pearson_r': float(np.corrcoef(y, x)[0, 1]) if np.ptp(y)>0 else None}


def phase_fit_report(data,swath,cohort):
    """Conditional correction fits, with phase coverage and exact per-swath counts."""
    result={}
    for treatment,other in (('deramp1','range2x'),('range2x','deramp1')):
        result[treatment]={}
        for s in (1,2):
            selected=cohort&(swath==s)
            phase=frac(data[other]['total/range_px'][selected])
            change=(data[treatment]['total/range_px']-data['deramp0']['total/range_px'])[selected]
            fit=sine_fit(change,phase)
            fit['predictor_treatment']=other
            fit['phase_bin_edges']=np.linspace(-.5,.5,9).tolist()
            fit['phase_bin_counts']=np.histogram(phase,np.linspace(-.5,.5,9))[0].tolist()
            result[treatment][f'iw{s}']=fit
    return result


def phase_fit_cohorts(data,swath):
    """Keep every selection step explicit; no IW1/IW2 agreement gate."""
    common=np.logical_and.reduce([d['valid'] for d in data.values()])
    strong=common&np.logical_and.reduce([d['correlation_peak']>=.3 for d in data.values()])
    spread=np.maximum(*[np.ptp(np.stack([d[f'total/{axis}_px'] for d in data.values()]),axis=0)
                         for axis in ('azimuth','range')])
    steady=strong&(spread<=.25)
    high=steady&np.logical_and.reduce([d['snr']>=20 for d in data.values()])
    masks={'common_valid':common,'all_peaks_ge_0.3':strong,
           'all_peaks_ge_0.3_spread_le_0.25':steady,
           'all_peaks_ge_0.3_spread_le_0.25_min_snr_ge_20':high}
    definitions={'common_valid':'Native-valid in all three treatments',
        'all_peaks_ge_0.3':'Common-valid and refined peak >=0.3 in all three treatments',
        'all_peaks_ge_0.3_spread_le_0.25':'Additionally max treatment spread <=0.25 original pixels in BOTH axes',
        'all_peaks_ge_0.3_spread_le_0.25_min_snr_ge_20':'Additionally SNR >=20 in all three treatments'}
    result={}
    previous=None
    for name,mask in masks.items():
        n=int(mask.sum());entry={'selection':definitions[name],'n':n,
            'n_by_swath':{f'iw{s}':int((mask&(swath==s)).sum()) for s in (1,2)}}
        if previous is not None:entry['removed_from_previous_stage_n']=previous-n
        if name!='common_valid':entry['fits']=phase_fit_report(data,swath,mask)
        result[name]=entry;previous=n
    return {'quantity':'Treatment minus deramp0 range correction, not independent absolute bias',
        'predictor':'sin(2*pi*frac(other treatment range estimate)); same original images, correlated errors remain',
        'selection_warning':'Treatment-spread trimming conditions on estimator outcomes and excludes peak jumps; report untrimmed quality cohort as well',
        'selection_stage_order':list(masks),'cohorts':result}


def grouped_mean(values, groups, rng, draws=4000):
    """Mean with a burst-pair block bootstrap interval (neighbouring windows overlap)."""
    keys = np.unique(groups);parts = [values[groups == k] for k in keys]
    boot = [np.concatenate([parts[i] for i in rng.integers(0, len(parts), len(parts))]).mean() for _ in range(draws)]
    return {'n': int(values.size), 'mean_px': float(values.mean()), 'median_px': float(np.median(values)),
            'ci95_px': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            'group_medians_px': [float(np.median(p)) for p in parts]}


def overlap_summary(overlap, range2x, rng):
    views = {'deramp0': np.load(overlap/'deramp-0/observations.npz'), 'deramp1': np.load(overlap/'deramp-1/observations.npz'),
             'range2x': np.load(range2x/'observations.npz')}
    quality = np.load(overlap/'comparison_masks.npz')['deramp0_both_peak_ge_0.3_burst_interiors']
    # Peak jumps between treatments dominate least squares; exclude them per window from
    # estimator stability alone, never from IW1/IW2 agreement.
    spread = np.max([np.ptp(np.stack([v[f'{s}_total'] for v in views.values()]), axis=0).max(axis=-1)
                     for s in ('iw1', 'iw2')], axis=0)
    cohort = quality & (spread <= .25)
    base = views['deramp0'];groups = (base['iw1_owner'][cohort].astype(int)*1000+base['iw2_owner'][cohort])
    result = {'cohort': 'deramp0 both peaks >=0.3, burst interiors; excluding windows whose estimate moves >0.25 px '
                        'between treatments in either swath', 'quality_n': int(quality.sum()), 'n': int(cohort.sum()),
              'per_swath_range_change_vs_subpixel_phase': {}, 'iw2_minus_iw1': {}}
    # The other treatment avoids direct reuse of the response estimate, but all
    # treatments share images and can retain correlated errors.
    for treatment, other in (('deramp1', 'range2x'), ('range2x', 'deramp1')):
        result['per_swath_range_change_vs_subpixel_phase'][treatment] = {
            swath: sine_fit((views[treatment][f'{swath}_total'][..., 1]-base[f'{swath}_total'][..., 1])[cohort],
                            frac(views[other][f'{swath}_total'][..., 1][cohort])) for swath in ('iw1', 'iw2')}
    phase = frac((base['iw1_total'][..., 1]+base['iw2_total'][..., 1])/2)[cohort]
    for name, v in views.items():
        delta = (v['iw2_total']-v['iw1_total'])[cohort]
        result['iw2_minus_iw1'][name] = {'range': grouped_mean(delta[:, 1], groups, rng),
                                         'azimuth': grouped_mean(delta[:, 0], groups, rng),
                                         'range_vs_baseline_phase': sine_fit(delta[:, 1], phase)}
    return result


def read_pair(directory):
    with h5py.File(directory/'radar/offsets.h5') as h:
        data = {k: h[k][:] for k in ('valid', 'correlation_peak', 'snr', 'source_burst_index', 'seam_crossing',
                                     'total/azimuth_px', 'total/range_px')}
    matching = json.loads((directory/'manifest.json').read_text())['matching']
    native = matching.get('native_parameters', {})
    segments = native.get('segments', [])
    data['cost'] = {'native_pool_wall_seconds': native.get('wall_seconds'), 'workers': native.get('workers'),
                    'max_worker_rss_GiB': max(s['worker_peak_rss_bytes'] for s in segments)/1024**3 if segments else None}
    return data


def join_distance(source, swath_of):
    """Rows from the nearest along-track change between two sources of the same swath."""
    join = np.zeros(source.shape, bool)
    a, b = source[1:], source[:-1]
    same = (a > 0) & (b > 0) & (a != b) & (swath_of[a] == swath_of[b])
    join[1:] |= same;join[:-1] |= same
    distance = np.full(source.shape, 99)
    rows = np.arange(source.shape[0])
    for col in range(source.shape[1]):
        hits = np.flatnonzero(join[:, col])
        if hits.size:distance[:, col] = np.abs(rows[:, None]-hits[None, :]).min(axis=1)
    return distance


def merged_summary(baseline, merged):
    data = {'deramp0': read_pair(baseline), 'deramp1': read_pair(merged/'deramp1'), 'range2x': read_pair(merged/'range2x')}
    ids = json.loads((merged/'comparison.json').read_text())['geocoded']['deramp1']['source_burst_ids']
    swath_of = np.array([0]+[int(i[-1]) for i in ids])
    common = np.logical_and.reduce([d['valid'] for d in data.values()])
    strong = common & np.logical_and.reduce([d['correlation_peak'] >= .3 for d in data.values()])
    source = data['deramp0']['source_burst_index'].astype(int)
    distance = join_distance(source, swath_of);swath = swath_of[np.clip(source, 0, len(ids))]
    joins = {}
    for a, b in (('deramp1', 'range2x'), ('deramp0', 'range2x'), ('deramp0', 'deramp1')):
        delta = np.abs(data[b]['total/azimuth_px']-data[a]['total/azimuth_px'])
        joins[f'{b}_minus_{a}'] = {f'iw{s}': {label: float(delta[strong & (swath == s) & mask].mean())
                                              for label, mask in (('join_row', distance == 0), ('adjacent_row', distance == 1),
                                                                  ('interior', distance >= 2))} for s in (1, 2)}
    stack = lambda axis: np.stack([d[f'total/{axis}_px'] for d in data.values()])
    spread = np.maximum(np.ptp(stack('range'), axis=0), np.ptp(stack('azimuth'), axis=0))
    unstable = common & (spread > 1)
    peak_min = np.minimum.reduce([d['correlation_peak'] for d in data.values()])
    snr_min = np.minimum.reduce([d['snr'] for d in data.values()])
    stability = {'common_valid_n': int(common.sum()), 'unstable_n': int(unstable.sum()),
                 'definition': 'any two of deramp0, deramp1, range2x differ by >1 px in range or azimuth', 'rules': {}}
    for source_name, (pk, snr) in {'minimum_across_treatments': (peak_min, snr_min),
                                   'range2x_alone': (data['range2x']['correlation_peak'], data['range2x']['snr']),
                                   'deramp0_alone': (data['deramp0']['correlation_peak'], data['deramp0']['snr'])}.items():
        stability['rules'][source_name] = {name: {'kept_fraction': float((common & rule(pk, snr)).sum()/common.sum()),
                                                  'unstable_kept': int((unstable & rule(pk, snr)).sum())}
                                           for name, rule in RULES.items()}
    # Whole-scene version of the overlap fit: every owned window, all sub-pixel phases.
    steady = strong & (spread <= .25)
    stability['per_swath_range_change_vs_subpixel_phase'] = {
        treatment: {f'iw{s}': sine_fit((data[treatment]['total/range_px']-data['deramp0']['total/range_px'])[steady & (swath == s)],
                                       frac(data[other]['total/range_px'][steady & (swath == s)])) for s in (1, 2)}
        for treatment, other in (('deramp1', 'range2x'), ('range2x', 'deramp1'))}
    stability['phase_correction_fit_report']=phase_fit_cohorts(data,swath)
    changes = {}
    for b in ('deramp1', 'range2x'):
        delta = {axis: (data[b][f'total/{axis}_px']-data['deramp0'][f'total/{axis}_px'])[strong] for axis in ('range', 'azimuth')}
        changes[f'{b}_minus_deramp0'] = {axis: {'median_abs_px': float(np.median(abs(v))), 'p90_abs_px': float(np.percentile(abs(v), 90))}
                                         for axis, v in delta.items()}
    return {'cohort_all_peaks_ge_0.3_n': int(strong.sum()), 'azimuth_abs_change_by_join_distance': joins,
            'changes_on_strong_cohort': changes, 'stability': stability, 'cost': {k: d['cost'] for k, d in data.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path);parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv);rng = np.random.default_rng(0)
    spec = json.loads(args.spec.read_text());report = {'pairs': {}}
    for name, entry in spec.items():
        paths = {k: Path(v) for k, v in entry.items() if k not in ('missions', 'note')}
        report['pairs'][name] = {'missions': entry.get('missions'), 'note': entry.get('note'),
                                 'overlap': overlap_summary(paths['overlap'], paths['overlap_range2x'], rng),
                                 'merged': merged_summary(paths['baseline'], paths['merged'])}
    write_json(args.output, report)
    return 0


if __name__ == '__main__':raise SystemExit(main())
