"""Split-period replication of a block-level Msf map, for any site.

The record of a velocity-pixel result is split at a calendar day (default: the
median pair midpoint, rounded down to UTC midnight) into two
acquisition-disjoint halves. Pairs are assigned by midpoint, and pairs sharing
an acquisition day with the other half are dropped. Each half is refitted with
the run's own pixel model and settings (its own calendar-year secular/order
regimes, the stored covariance restricted to the half, and the run's robust
weighting) and aggregated onto the same blocks as the full map from
``tools/aggregate_msf_blocks.py`` (``--blocks``).

Because the halves share no acquisitions, a block difference has covariance
Sigma_A + Sigma_B. Block differences therefore test, without any synthetic noise
model:

- whether the two halves agree (temporal stability plus replication);
- whether the shared-cluster block SE is calibrated: under stability, the
  median of Delta' (Sigma_A + Sigma_B)^-1 Delta should be chi2_2's 1.386, so
  sqrt(median / 1.386) is an SE scale factor. Real Msf change between the
  halves also inflates it.

Grounded slow blocks give the same test where no horizontal Msf is expected.
The shelf-wide mean over floating full-map discoveries eligible in both halves
is compared with its shared-cluster covariance (velocity_blocks.combine).

Each half spans only part of the record, so coverage is relaxed per half
(``--min-*``; the defaults suit halves of about one year). The source set is
the full map's mapped sources. Block CRS, model, inflation, minimum sources and
the grounded slow/floating rules default to the full map's settings, read from
its summary.json. Inputs are read only.

Example:
    PYTHONPATH=src python tools/split_period_msf_blocks.py \\
        /path/site/work/velocity_pixels/run --blocks /path/site/work/velocity_pixels/msf_blocks \\
        --output /path/site/work/velocity_pixels/split_period
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS
from scipy.stats import chi2

from glacier_offsets import velocity_block_workflow, velocity_blocks
from glacier_offsets.velocity_block_workflow import (CLASS_DEFAULTS, artifact_paths, assign_blocks, classify,
                                                     fit_pairs, load_sources, median_split_date, per_block,
                                                     result_parameters, split_day_number, split_halves,
                                                     stored_covariances, subset_covariances, subset_pairs)
from glacier_offsets.velocity_blocks import BlockAccumulator, combine, statistics
from glacier_offsets.velocity_io import digest, write_json
from glacier_offsets.velocity_pixel_general import Layout, load_field


def worker(paths, result_file, params, model, index, columns, labels_by_size, nblocks):
    with h5py.File(result_file) as h:
        cov = stored_covariances(h)
    with h5py.File(paths['observations']) as h, h5py.File(paths['solid_earth']) as body:
        layout = Layout.from_observations(h)
        field = load_field(h, body, index, params['screening']['remove_mixed_windows'])
    sub, sublayout, present = subset_pairs(field, layout, columns)
    fit, use = fit_pairs(sub, sublayout, subset_covariances(cov, present, columns), params, model)
    accs = {}
    for size, labels in labels_by_size.items():
        acc = BlockAccumulator(nblocks[size], fit['cluster_influence'].shape[1])
        acc.add(labels[use], fit['cs'][use], fit['cluster_influence'][use], fit['cluster_cr1'][use],
                fit['coefficient_covariance'][use])
        accs[size] = acc
    return accs, int(use.sum()), fit['cluster_ids'].tolist(), list(sublayout.years)


def run_half(name, columns, paths, result_file, params, model, selected, labels, nblocks, a, t0):
    chunks = np.array_split(np.arange(len(selected)), max(1, len(selected)//a.batch))
    totals = dict.fromkeys(labels)
    used, clusters, years = 0, None, None
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(worker, paths, result_file, params, model, selected[c], columns,
                            {s: labels[s][c] for s in labels}, nblocks) for c in chunks]
        for job in jobs:
            accs, n, ids, years = job.result()
            used += n
            if clusters is None:
                clusters = ids
            elif ids != clusters:
                raise ValueError('Calendar clusters differ between batches')
            for s in labels:
                totals[s] = accs[s] if totals[s] is None else totals[s].merge(accs[s])
    print(f'half {name}: {len(columns)} pairs, years {years}, {used} sources mapped, {len(clusters)} clusters '
          f'({time.time()-t0:.0f}s)', flush=True)
    return {s: totals[s].finish() for s in labels}, used, len(clusters), years


def full_map(blocks, result, sizes):
    """Settings and per-size tables/CRS of the full-map block run, checked against this result."""
    summary_file = blocks/'summary.json'
    summary = json.loads(summary_file.read_text())
    if Path(summary['result']).resolve() != result:
        raise ValueError(f"{summary_file} was made from {summary['result']}, not {result}")
    tables, crs = {}, None
    for size in sizes:
        label = f'{size/1000:g}km'
        if not (blocks/f'blocks_{label}.csv').exists():
            raise FileNotFoundError(f'{blocks} has no {label} blocks; rerun aggregate_msf_blocks.py with --sizes')
        tables[size] = pd.read_csv(blocks/f'blocks_{label}.csv').set_index(['ix', 'iy'])
        with h5py.File(blocks/f'blocks_{label}.h5') as h:
            size_crs = CRS.from_wkt(h.attrs['crs_wkt'])
        if crs is not None and size_crs != crs:
            raise ValueError(f'{blocks} block sizes use different CRS')
        crs = size_crs
    return summary, tables, crs


def setting(cli, summary, key, default):
    """CLI value, else the full map's recorded value, else the default."""
    if cli is not None:
        return cli
    return summary.get(key, default)


def half_frame(r, inflation):
    st = statistics(r['mean'], r['covariance'], r['clusters'], inflation=inflation)
    return pd.DataFrame({'sources': r['sources'], 'C': r['mean'][:, 0], 'S': r['mean'][:, 1],
                         'cov_CC': r['covariance'][:, 0, 0], 'cov_CS': r['covariance'][:, 0, 1],
                         'cov_SS': r['covariance'][:, 1, 1], 'clusters': r['clusters'],
                         'amplitude_debiased': st['amplitude_debiased'], 'amplitude_sigma': st['amplitude_sigma'],
                         'phase': st['phase_deg'], 'phase_sigma': st['phase_sigma_deg'], 'p_value': st['p_value']})


def split_table(results, keys, labels, size, full, src, rules, a):
    n = len(keys)
    t = pd.DataFrame({'ix': keys[:, 0], 'iy': keys[:, 1], 'x_center': (keys[:, 0]+.5)*size,
                      'y_center': (keys[:, 1]+.5)*size})
    for name in ('A', 'B'):
        t = t.join(half_frame(results[name][size], a.inflation).add_suffix(f'_{name}'))
    t['max_response'] = per_block(src['response'], labels, n, 'max')
    t['median_response'] = per_block(src['response'], labels, n, 'median')
    t['median_speed'] = per_block(src['speed'], labels, n, 'median')
    joined = full.reindex(list(zip(t.ix, t.iy)))
    t['discovery_full'] = joined['discovery'].eq(True).to_numpy()
    t['phase_full'] = joined['phase_deg'].to_numpy()
    t['both_eligible'] = (t.sources_A >= a.min_sources) & (t.sources_B >= a.min_sources)
    grounded_slow, floating = classify(t.max_response, t.median_response, t.median_speed, **rules)
    t['floating'], t['grounded_slow'] = floating, grounded_slow  # column order of the pilot tables
    d = t[['C_A', 'S_A']].to_numpy()-t[['C_B', 'S_B']].to_numpy()
    cov = np.stack([np.stack([t.cov_CC_A+t.cov_CC_B, t.cov_CS_A+t.cov_CS_B], -1),
                    np.stack([t.cov_CS_A+t.cov_CS_B, t.cov_SS_A+t.cov_SS_B], -1)], -2)
    t['T_difference'] = np.einsum('bi,bij,bj->b', d, np.linalg.pinv(np.nan_to_num(cov), hermitian=True), d)
    t['p_difference_chi2'] = chi2.sf(t.T_difference, 2)
    t['p_difference_chi2_inflated'] = chi2.sf(t.T_difference/a.inflation**2, 2)
    t['dphase'] = (t.phase_B-t.phase_A+180) % 360-180
    t['amplitude_ratio_BA'] = t.amplitude_debiased_B/t.amplitude_debiased_A.replace(0, np.nan)
    return t


def class_stats(t):
    both = t[t.both_eligible]
    stats = {}
    for cls in ('floating', 'grounded_slow'):
        c = both[both[cls]]
        det = c[c.discovery_full] if cls == 'floating' else c
        empty = not len(c)
        stats[cls] = {'blocks': int(len(c)),
                      'median_T': None if empty else float(c.T_difference.median()),
                      'se_scale_factor': None if empty else float(np.sqrt(c.T_difference.median()/chi2.median(2))),
                      'reject_0.05': None if empty else float(np.mean(c.p_difference_chi2 < .05)),
                      'reject_0.05_inflated': None if empty else float(np.mean(c.p_difference_chi2_inflated < .05)),
                      'median_abs_dphase_deg': float(det.dphase.abs().median()) if len(det) else None,
                      'mean_dphase_deg': float(det.dphase.mean()) if len(det) else None,
                      'median_amplitude_ratio_BA': float(det.amplitude_ratio_BA.median()) if len(det) else None,
                      'blocks_for_phase_amplitude': int(len(det))}
    return stats


def shelf_mean(t, results, size, inflation):
    """Mean over floating full-map discoveries eligible in both halves, and its half difference.

    Blocks within a half share acquisitions, so the mean's covariance combines
    block influences first; the halves are acquisition-disjoint, so the
    difference covariance is their sum.
    """
    pick = (t.both_eligible & t.floating & t.discovery_full).to_numpy()
    if not pick.any():
        return {'blocks': 0}
    w = pick/pick.sum()
    shelf, parts = {}, {}
    for name in ('A', 'B'):
        r = results[name][size]
        m = w@np.nan_to_num(r['mean'])
        c = combine(np.nan_to_num(r['influence']), np.nan_to_num(r['cr1']), w)*inflation**2
        amp, ph = float(np.hypot(*m)), float(np.degrees(np.arctan2(m[1], m[0])))
        g = m/amp
        n = np.array([-m[1], m[0]])/amp**2
        shelf[name] = {'C': float(m[0]), 'S': float(m[1]), 'amplitude': amp, 'phase_deg': ph,
                       'amplitude_sigma': float(np.sqrt(g@c@g)), 'phase_sigma_deg': float(np.degrees(np.sqrt(n@c@n)))}
        parts[name] = (m, c)
    d = parts['B'][0]-parts['A'][0]
    tt = float(d@np.linalg.solve(parts['A'][1]+parts['B'][1], d))
    return {'blocks': int(pick.sum()), **shelf, 'T_difference': tt, 'p_difference_chi2_inflated': float(chi2.sf(tt, 2)),
            'dphase_deg': float((shelf['B']['phase_deg']-shelf['A']['phase_deg']+180) % 360-180),
            'amplitude_ratio_BA': shelf['B']['amplitude']/shelf['A']['amplitude']}


def figure(path, tables, crs_name, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(tables), 4, figsize=(19, 4.6*len(tables)), constrained_layout=True, squeeze=False)
    for row, (size, t) in zip(axes, tables.items()):
        label = f'{size/1000:g} km'
        both = t[t.both_eligible]
        fl = both[both.floating]
        amax = a.amplitude_max or (float(np.nanpercentile(fl[['amplitude_debiased_A', 'amplitude_debiased_B']], 99))
                                   if len(fl) else 1.)
        row[0].errorbar(fl.amplitude_debiased_A, fl.amplitude_debiased_B, xerr=fl.amplitude_sigma_A,
                        yerr=fl.amplitude_sigma_B, fmt='.', ms=3, alpha=.4, elinewidth=.4)
        row[0].plot([0, amax], [0, amax], 'k-', lw=.8)
        row[0].set(xlabel='half A debiased amplitude (m/day)', ylabel='half B', title=f'{label} floating blocks ({len(fl)})',
                   xlim=(0, amax), ylim=(0, amax))
        det = fl[fl.discovery_full]
        if len(det):
            # Centre on the circular mean of half A and unwrap half B next to half A.
            centre = float(np.degrees(np.angle(np.exp(1j*np.radians(det.phase_A)).mean())))
            pa = (det.phase_A-centre+180) % 360-180+centre
            pb = pa+det.dphase
            row[1].scatter(pa, pb, s=4, alpha=.5)
            row[1].plot([centre-90, centre+90], [centre-90, centre+90], 'k-', lw=.8)
            row[1].set(xlim=(centre-90, centre+90), ylim=(centre-90, centre+90))
        row[1].set(xlabel='half A phase (deg)', ylabel='half B phase (deg)',
                   title=f'phase, full-map floating discoveries ({len(det)})')
        x = np.linspace(0, 20, 200)
        for cls, color in (('floating', 'C0'), ('grounded_slow', '0.4')):
            v = both[both[cls]].T_difference
            if len(v):
                row[2].hist(v.clip(upper=20), bins=x[::8], density=True, histtype='step', color=color,
                            label=f'{cls} ({len(v)})')
        row[2].plot(x, chi2.pdf(x, 2), 'r-', lw=.8, label='chi2_2')
        row[2].set(xlabel="T = D' (S_A+S_B)^-1 D", title='block difference statistic', yscale='log')
        row[2].legend(fontsize=8)
        sc = row[3].scatter(both.x_center/1e3, both.y_center/1e3, c=np.sqrt(both.T_difference).clip(upper=5),
                            s=(size/1000)**2*1.6, marker='s', cmap='magma', vmin=0, vmax=5)
        fig.colorbar(sc, ax=row[3], shrink=.8, label='sqrt(T)')
        row[3].set(aspect='equal', title='difference significance', xlabel=f'x (km, {crs_name})')
    fig.savefig(path, dpi=130)


def main(a):
    t0 = time.time()
    result, out, blocks = a.result.resolve(), a.output.resolve(), a.blocks.resolve()
    if out in (result, blocks):
        raise ValueError('--output must differ from the result and full-map block directories')
    sizes = [float(s) for s in a.sizes.split(',')]
    full_summary, full_tables, crs = full_map(blocks, result, sizes)
    model = full_summary.get('model', 'residual_vertical')
    a.inflation = setting(a.inflation, full_summary, 'inflation', 1.)
    a.min_sources = setting(a.min_sources, full_summary, 'min_sources', 8)
    recorded = full_summary.get('null_rule', {})
    rules = {'null_max_response': setting(a.null_max_response, recorded, 'max_response', CLASS_DEFAULTS['null_max_response']),
             'null_max_speed': setting(a.null_max_speed, recorded, 'max_median_speed_m_day', CLASS_DEFAULTS['null_max_speed']),
             'floating_min_response': setting(a.floating_min_response, full_summary, 'floating_min_median_response',
                                              CLASS_DEFAULTS['floating_min_response'])}
    recipe, paths = artifact_paths(result, a.cache)
    params = result_parameters(recipe)
    params['coverage'].update(min_pairs_per_year=a.min_pairs_per_year, min_pairs_total=a.min_pairs_total,
                              min_pairs_standard=a.min_pairs_total, min_span_days=a.min_span_days)
    with h5py.File(paths['observations']) as h:
        ce, cl = h['calendar_early'][:], h['calendar_late'][:]
        pair_ids = h['pair_id'][:].astype(str)
    split = a.split or median_split_date(ce, cl)
    parts, shared = split_halves(ce, cl, split_day_number(split))
    for name, columns in parts.items():
        if not len(columns):
            raise ValueError(f'Half {name} is empty for split {split}')
    result_file = str(result/'source_coefficients.h5')
    src = load_sources(result_file, model, crs.to_wkt())
    blocks_by_size = assign_blocks(src['x'], src['y'], sizes)
    keys = {s: k for s, (k, _) in blocks_by_size.items()}
    labels = {s: lab for s, (_, lab) in blocks_by_size.items()}
    nblocks = {s: len(k) for s, k in keys.items()}
    for size in sizes:
        if set(map(tuple, keys[size].tolist())) != set(full_tables[size].index):
            raise ValueError(f'{size/1000:g} km blocks differ from the full map; was it made with another CRS or source set?')
    out.mkdir(parents=True, exist_ok=True)
    results, meta = {}, {}
    for name, columns in parts.items():
        res, used, ng, years = run_half(name, columns, paths, result_file, params, model, src['selected'], labels,
                                        nblocks, a, t0)
        results[name] = res
        meta[name] = {'pairs': int(len(columns)), 'first_pair': pair_ids[columns[0]], 'last_pair': pair_ids[columns[-1]],
                      'years': years, 'sources_mapped': used, 'clusters': ng}
    summary = {'split_day': split, 'split_default_median_midpoint': a.split is None,
               'dropped_shared_acquisition_days': len(shared),
               'dropped_pairs': int(len(ce)-sum(len(v) for v in parts.values())), 'halves': meta,
               'coverage_per_half': {k: params['coverage'][k] for k in ('min_pairs_per_year', 'min_pairs_total',
                                                                        'min_span_days', 'min_phase_bins', 'phase_bins')},
               'inflation': a.inflation, 'min_sources': a.min_sources, 'model': model, 'crs': crs.name,
               'class_rules': rules, 'sizes': {}}
    tables = {}
    for size in sizes:
        label = f'{size/1000:g}km'
        t = split_table(results, keys[size], labels[size], size, full_tables[size], src, rules, a)
        t.to_csv(out/f'split_blocks_{label}.csv', index=False)
        tables[size] = t
        stats = class_stats(t)
        stats['shelf_mean_floating_discoveries'] = shelf_mean(t, results, size, a.inflation)
        summary['sizes'][label] = stats
    figure(out/'split_blocks.png', tables, crs.name, a)
    summary.update(result=str(result), blocks=str(blocks), recipe_sha256=digest(result/'recipe.json'),
                   tool_sha256=digest(Path(__file__)), module_sha256=digest(Path(velocity_blocks.__file__)),
                   workflow_module_sha256=digest(Path(velocity_block_workflow.__file__)), elapsed_seconds=time.time()-t0)
    write_json(out/'summary.json', summary)
    print(json.dumps(summary['sizes'], indent=1))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('result', type=Path, help='velocity-pixel result directory (contains recipe.json)')
    p.add_argument('--blocks', type=Path, required=True, help='full-map output of aggregate_msf_blocks.py for this result')
    p.add_argument('--output', type=Path, required=True, help='new or existing output directory')
    p.add_argument('--split', help='split date YYYY-MM-DD (default: median pair midpoint)')
    p.add_argument('--sizes', default='3000,5000', help='comma-separated block sizes in metres (must exist in --blocks)')
    p.add_argument('--min-sources', type=int, help='sources required in each half (default: the full map)')
    p.add_argument('--inflation', type=float, help='standard-error multiplier (default: the full map)')
    p.add_argument('--null-max-response', type=float, help='grounded slow rule (default: the full map)')
    p.add_argument('--null-max-speed', type=float, help='grounded slow rule, m/day (default: the full map)')
    p.add_argument('--floating-min-response', type=float, help='floating rule (default: the full map)')
    p.add_argument('--min-pairs-per-year', type=int, default=5, help='per-half coverage: clean pairs in each year')
    p.add_argument('--min-pairs-total', type=int, default=20, help='per-half coverage: clean pairs in total')
    p.add_argument('--min-span-days', type=float, default=270., help='per-half coverage: first to last retained day')
    p.add_argument('--amplitude-max', type=float, help='amplitude axis limit, m/day (default: 99th percentile)')
    p.add_argument('--cache', type=Path, help='velocity cache directory if it moved since the run')
    p.add_argument('--batch', type=int, default=512, help='sources per worker batch; computational only')
    p.add_argument('--workers', type=int, default=8)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
