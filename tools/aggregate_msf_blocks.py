"""Block-level Msf C/S, amplitude and phase from a velocity-pixel result.

Works for any velocity-pixel result directory (``velocity-pixel`` workflow
output with ``recipe.json``, ``inversion-config.resolved.yaml`` and
``source_coefficients.h5``). Per-source C/S coefficients, calendar-cluster
influences and small-sample factors are recomputed with the exact pixel
workflow (``fit_sources`` with the run's recipe, cached observations and
solid-Earth artifacts, and the covariance stored with the coefficients). The
recomputed C/S must reproduce the stored coefficients, so the blocks are built
from exactly the published fit.

Each unique source with standard coverage and an estimable fit is assigned to
one square block by its location in a projected CRS (the result's own CRS
unless ``--crs`` is given). The block C/S is the equal-weight source mean. Its
covariance averages source influences before the outer product, so acquisition
errors shared by the block's sources are counted once (see
``glacier_offsets.velocity_blocks``). Amplitude and phase are derived from the
block C/S, with a debiased amplitude sqrt(max(0, |m|^2 - tr S)).

Detection uses the small-sample F(2, G-2) reference after multiplying standard
errors by ``--inflation`` and Benjamini-Hochberg FDR at ``--q`` over blocks with
at least ``--min-sources`` sources. The inflation is a site calibration (for
example the worst regional under-dispersion of a bootstrap calibration; 1.15 at
Evans and Larsen); the default 1 applies none. Stored HDF5 covariances are never
inflated.

Where the result carries ``tidal_response_factor``, blocks are also classed as
grounded slow (every source response <= ``--null-max-response`` and median
speed <= ``--null-max-speed``) or floating (median response >=
``--floating-min-response``). Grounded slow blocks have no expected horizontal
Msf and give an empirical null for the test. Optional reference points
(``--points``: CSV with name, longitude, latitude and, optionally, amplitude in
m/day and phase_deg) are matched to their blocks, for example GPS along-flow
Msf for comparison.

Outputs per block size: blocks_<size>km.csv (table), .h5 (mean C/S, covariance,
block-mean cluster influences and cr1, so any linear combination of blocks can
be given the shared-cluster covariance with ``velocity_blocks.combine``) and
.tif (eligible blocks); plus blocks_msf.png and summary.json. Inputs are read
only.

Example:
    PYTHONPATH=src python tools/aggregate_msf_blocks.py \\
        /path/site/work/velocity_pixels/run --output /path/site/work/velocity_pixels/msf_blocks \\
        --sizes 3000,5000 --inflation 1.15
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import yaml

from glacier_offsets import velocity_blocks
from glacier_offsets.velocity_blocks import BlockAccumulator, benjamini_hochberg, block_keys, statistics
from glacier_offsets.velocity_io import digest, write_json
from glacier_offsets.velocity_pixel_config import validate_parameters
from glacier_offsets.velocity_pixel_general import MODELS, Layout, fit_sources

TABLE_BANDS = ['sources', 'C', 'S', 'amplitude_debiased', 'amplitude_sigma', 'phase_deg', 'phase_sigma_deg',
               'p_value', 'discovery', 'design_effect']


def artifact_paths(result, cache=None):
    """Observation and solid-Earth artifacts the run used, from its recipe and resolved configuration."""
    recipe = json.loads((result/'recipe.json').read_text())
    if cache is None:
        config = yaml.safe_load((result/'inversion-config.resolved.yaml').read_text())
        cache = Path(config['cache_directory'])
    art = recipe['artifacts']
    paths = {'observations': cache/'observations'/art['observations']['id']/'observations.h5',
             'solid_earth': cache/'solid_earth'/art['solid_earth']['id']/'solid_earth.h5'}
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f'{name} artifact {path} is missing; pass --cache if the cache moved')
    return recipe, {k: str(v) for k, v in paths.items()}


def stored_covariances(h):
    """The covariance the stored fit used (annual array, or annual plus per-pair)."""
    cov = h['covariance_by_year'][:]
    if 'covariance_by_pair' in h:
        return {'by_year': cov, 'by_pair': h['covariance_by_pair'][:]}
    return cov


def worker(paths, result_file, params, model, index, labels_by_size, nblocks, stored_cs):
    with h5py.File(result_file) as h:
        cov = stored_covariances(h)
    with h5py.File(paths['observations']) as h, h5py.File(paths['solid_earth']) as body:
        layout = Layout.from_observations(h)
        _, _, _, fits = fit_sources(h, body, layout, cov, params, index)
    fit = fits[model]
    if not fit['estimable'].all():
        raise ValueError(f"{(~fit['estimable']).sum()} selected sources are no longer estimable")
    diff = float(np.max(abs(fit['cs']-stored_cs)))
    accs = {}
    for size, labels in labels_by_size.items():
        acc = BlockAccumulator(nblocks[size], fit['cluster_influence'].shape[1])
        acc.add(labels, fit['cs'], fit['cluster_influence'], fit['cluster_cr1'], fit['coefficient_covariance'])
        accs[size] = acc
    return accs, diff, fit['cluster_ids'].tolist()


def merge(total, part):
    if total is None:
        return part
    for name in ('count', 'cs', 'influence', 'cr1', 'pointwise'):
        setattr(total, name, getattr(total, name)+getattr(part, name))
    return total


def block_crs(h, override):
    crs = CRS.from_user_input(override) if override else CRS.from_wkt(h.attrs.get('crs_wkt') or h['crs'].attrs['crs_wkt'])
    if not crs.is_projected:
        raise ValueError(f'Blocks need a projected CRS in metres; {crs.name} is not. Pass --crs')
    return crs


def read_points(path):
    if path is None:
        return pd.DataFrame(columns=['name', 'longitude', 'latitude'])
    table = pd.read_csv(path)
    missing = {'name', 'longitude', 'latitude'} - set(table.columns)
    if missing:
        raise ValueError(f'{path} lacks columns {sorted(missing)}')
    return table


def write_raster(path, table, size, crs_wkt, bands):
    import rasterio
    from rasterio.transform import from_origin
    ix0, ix1, iy0, iy1 = table.ix.min(), table.ix.max(), table.iy.min(), table.iy.max()
    width, height = ix1-ix0+1, iy1-iy0+1
    data = np.full((len(bands), height, width), np.nan, np.float32)
    r, c = (iy1-table.iy).to_numpy(), (table.ix-ix0).to_numpy()
    for k, name in enumerate(bands):
        data[k, r, c] = table[name].to_numpy(np.float32)
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height, count=len(bands), dtype='float32',
                       crs=crs_wkt, transform=from_origin(ix0*size, (iy1+1)*size, size, size), nodata=np.nan,
                       compress='deflate') as dst:
        dst.write(data)
        for k, name in enumerate(bands):
            dst.set_band_description(k+1, name)


def figure(path, tables, points, crs_name, a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(tables), 4, figsize=(19, 4.6*len(tables)), constrained_layout=True, squeeze=False)
    km = 1e-3
    for row, (size, t) in zip(axes, tables.items()):
        use = t[t.eligible]
        vmax = a.amplitude_max or float(np.nanpercentile(use.amplitude_debiased, 99)) if len(use) else 1.

        def scatter(ax, sel, value, title, **kw):
            sc = ax.scatter(sel.x_center*km, sel.y_center*km, c=value, s=(size/1000)**2*1.6, marker='s', **kw)
            fig.colorbar(sc, ax=ax, shrink=.8)
            ax.set_title(title, fontsize=10)
            ax.set_aspect('equal')
            for p in points.itertuples():
                ax.plot(p.x*km, p.y*km, 'k^', ms=5)
                ax.annotate(p.name, (p.x*km, p.y*km), fontsize=7, xytext=(3, 3), textcoords='offset points')
            ax.set_xlabel(f'x (km, {crs_name})')
        scatter(row[0], use, use.amplitude_debiased, f'{size/1000:g} km debiased amplitude (m/day)',
                cmap='viridis', vmin=0, vmax=vmax)
        det = use[use.discovery]
        scatter(row[1], det, det.phase_deg, f'phase (deg), BH q={a.q} discoveries ({len(det)}/{len(use)})',
                cmap='twilight', vmin=-180, vmax=180)
        scatter(row[2], use, use.phase_sigma_deg.clip(upper=60), 'phase SE (deg, inflated, clipped at 60)',
                cmap='magma_r', vmin=0, vmax=60)
        null = use[use.grounded_slow]
        if len(null):
            bins = np.linspace(0, 1, 21)
            row[3].hist(null.p_value, bins=bins, color='0.5', label=f'grounded slow ({len(null)})', density=True)
            if a.inflation != 1:
                row[3].hist(null.p_value_uninflated, bins=bins, histtype='step', color='k', density=True,
                            label='uninflated SE')
            row[3].axhline(1, color='r', lw=.8)
            row[3].legend(fontsize=8)
        else:
            row[3].text(.5, .5, 'no grounded slow blocks', ha='center', va='center', transform=row[3].transAxes)
        row[3].set_title(f'{size/1000:g} km empirical-null p-values', fontsize=10)
        row[3].set_xlabel('p')
    fig.savefig(path, dpi=130)


def block_table(res, keys, labels, size, speed, response, a):
    raw = statistics(res['mean'], res['covariance'], res['clusters'])
    inf = statistics(res['mean'], res['covariance'], res['clusters'], inflation=a.inflation)
    n = len(keys)

    def by_block(values, how):
        return getattr(pd.Series(values).groupby(labels), how)().reindex(range(n)).to_numpy()
    frame = pd.DataFrame({
        'ix': keys[:, 0], 'iy': keys[:, 1], 'x_center': (keys[:, 0]+.5)*size, 'y_center': (keys[:, 1]+.5)*size,
        'sources': res['sources'], 'clusters': res['clusters'], 'C': res['mean'][:, 0], 'S': res['mean'][:, 1],
        'cov_CC': res['covariance'][:, 0, 0], 'cov_CS': res['covariance'][:, 0, 1], 'cov_SS': res['covariance'][:, 1, 1],
        'amplitude': inf['amplitude'], 'amplitude_debiased': inf['amplitude_debiased'],
        'amplitude_sigma': inf['amplitude_sigma'], 'phase_deg': inf['phase_deg'],
        'phase_sigma_deg': inf['phase_sigma_deg'], 'component_sigma': inf['component_sigma'],
        'wald': inf['wald'], 'p_value': inf['p_value'], 'p_value_uninflated': raw['p_value'],
        'design_effect': res['design_effect'], 'effective_sources': res['effective_sources'],
        'median_speed': by_block(speed, 'median'), 'max_response': by_block(response, 'max'),
        'median_response': by_block(response, 'median')})
    frame['eligible'] = frame.sources >= a.min_sources
    # NaN responses (no tide mask) never satisfy these comparisons.
    frame['grounded_slow'] = (frame.eligible & (frame.max_response <= a.null_max_response)
                              & (frame.median_speed <= a.null_max_speed))
    frame['floating'] = frame.eligible & (frame.median_response >= a.floating_min_response)
    frame['discovery'] = False
    frame.loc[frame.eligible, 'discovery'] = benjamini_hochberg(frame.loc[frame.eligible, 'p_value'].to_numpy(), a.q)
    return frame


def point_matches(frame, points, size):
    out = {}
    if not len(points):
        return out
    ix, iy = block_keys(points.x, points.y, size)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(frame.ix, frame.iy))}
    for p, i, j in zip(points.itertuples(), ix, iy):
        k = index.get((int(i), int(j)))
        if k is None:
            out[p.name] = None
            continue
        r = frame.iloc[k]
        out[p.name] = {'block_ix': int(r.ix), 'block_iy': int(r.iy), 'block_sources': int(r.sources),
                       'eligible': bool(r.eligible), 'amplitude': float(r.amplitude),
                       'amplitude_debiased': float(r.amplitude_debiased), 'amplitude_sigma': float(r.amplitude_sigma),
                       'phase_deg': float(r.phase_deg), 'phase_sigma_deg': float(r.phase_sigma_deg),
                       'p_value': float(r.p_value), 'discovery': bool(r.discovery)}
        reference = {name: float(getattr(p, name)) for name in ('amplitude', 'phase_deg') if hasattr(p, name)}
        if reference:
            out[p.name]['reference'] = reference
    return out


def size_summary(frame, matches):
    el = frame[frame.eligible]
    null = el[el.grounded_slow]

    def median(values):
        return float(values.median()) if len(values) else None
    return {'blocks': int(len(frame)), 'eligible_blocks': int(len(el)), 'sources_in_eligible': int(el.sources.sum()),
            'discoveries': int(el.discovery.sum()), 'floating_eligible': int(el.floating.sum()),
            'floating_discoveries': int((el.floating & el.discovery).sum()),
            'median_design_effect': median(el.design_effect), 'median_effective_sources': median(el.effective_sources),
            'median_sources': median(el.sources), 'median_component_sigma_inflated': median(el.component_sigma),
            'grounded_slow_null': {'blocks': int(len(null)),
                                   'reject_0.05_inflated': float(np.mean(null.p_value < .05)) if len(null) else None,
                                   'reject_0.05_uninflated': float(np.mean(null.p_value_uninflated < .05)) if len(null) else None,
                                   'discoveries': int(null.discovery.sum()),
                                   'median_debiased_amplitude': median(null.amplitude_debiased),
                                   'median_raw_amplitude': median(null.amplitude)},
            'points': matches}


def main(a):
    t0 = time.time()
    result, out = a.result.resolve(), a.output.resolve()
    if out == result:
        raise ValueError('--output must differ from the result directory')
    if a.model not in MODELS:
        raise ValueError(f'--model must be one of {list(MODELS)}')
    recipe, paths = artifact_paths(result, a.cache)
    # Older recipes predate some screening keys; their defaults reproduce the legacy fit.
    params = validate_parameters(recipe['parameters'])
    result_file = str(result/'source_coefficients.h5')
    with h5py.File(result_file) as h:
        if h.attrs.get('status') not in (None, 'complete'):
            raise ValueError(f"{result_file} is not complete (status {h.attrs['status']})")
        selected = np.flatnonzero(h['standard_coverage'][:].astype(bool) & h[f'{a.model}/estimable'][:].astype(bool))
        crs = block_crs(h, a.crs)
        to_map = Transformer.from_crs(4326, crs, always_xy=True)
        xs, ys = to_map.transform(h['longitude'][:][selected], h['latitude'][:][selected])
        stored = h[f'{a.model}/cs'][:][selected]
        speed = h['mean_flow_speed'][:][selected]
        response = (h['tidal_response_factor'][:][selected] if 'tidal_response_factor' in h
                    else np.full(len(selected), np.nan))
    if not len(selected):
        raise ValueError('No sources have standard coverage and an estimable fit')
    out.mkdir(parents=True, exist_ok=True)
    points = read_points(a.points)
    points['x'], points['y'] = to_map.transform(points.longitude.to_numpy(float), points.latitude.to_numpy(float))
    sizes = [float(s) for s in a.sizes.split(',')]
    keys, labels, nblocks = {}, {}, {}
    for size in sizes:
        ix, iy = block_keys(xs, ys, size)
        k, inv = np.unique(np.stack([ix, iy], 1), axis=0, return_inverse=True)
        keys[size], labels[size], nblocks[size] = k, inv.ravel(), len(k)
    chunks = np.array_split(np.arange(len(selected)), max(1, len(selected)//a.batch))
    totals = dict.fromkeys(sizes)
    diffs, clusters = [], None
    with ProcessPoolExecutor(a.workers) as pool:
        jobs = [pool.submit(worker, paths, result_file, params, a.model, selected[c],
                            {s: labels[s][c] for s in sizes}, nblocks, stored[c]) for c in chunks]
        for n, job in enumerate(jobs):
            accs, diff, ids = job.result()
            diffs.append(diff)
            if clusters is None:
                clusters = ids
            elif ids != clusters:
                raise ValueError('Calendar clusters differ between batches')
            for s in sizes:
                totals[s] = merge(totals[s], accs[s])
            if n % 40 == 0:
                print(f'{n+1}/{len(jobs)} batches ({time.time()-t0:.0f}s)', flush=True)
    max_diff = float(max(diffs))
    if max_diff > a.tolerance:
        raise ValueError(f'Recomputed C/S do not reproduce the stored fit ({max_diff:.3g})')
    crs_wkt = crs.to_wkt()
    tables, summary = {}, {'sizes': {}}
    for size in sizes:
        res = totals[size].finish()
        frame = block_table(res, keys[size], labels[size], size, speed, response, a)
        label = f'{size/1000:g}km'
        frame.to_csv(out/f'blocks_{label}.csv', index=False)
        with h5py.File(out/f'blocks_{label}.h5', 'w') as h:
            h['block_ix'], h['block_iy'] = keys[size][:, 0], keys[size][:, 1]
            h['mean_cs'], h['covariance'] = res['mean'], res['covariance']
            h['cluster_influence'], h['cr1'] = res['influence'], res['cr1']
            h['cluster_ids'] = np.asarray(clusters)
            h.attrs.update(block_size_m=size, crs_wkt=crs_wkt, model=a.model, inflation_not_applied=True,
                           note='cov of sum_b w_b mean_b = mean-|w|-weighted cr1 * (sum_b w_b I_b)^T (sum_b w_b I_b); '
                                'see velocity_blocks.combine')
        eligible = frame[frame.eligible]
        if len(eligible):
            write_raster(out/f'blocks_{label}.tif', eligible.assign(discovery=eligible.discovery.astype(float)), size,
                         crs_wkt, TABLE_BANDS)
        tables[size] = frame
        summary['sizes'][label] = size_summary(frame, point_matches(frame, points, size))
    figure(out/'blocks_msf.png', tables, points, crs.name, a)
    summary.update(result=str(result), model=a.model, crs=crs.name, recipe_sha256=digest(result/'recipe.json'),
                   max_reproduction_difference=max_diff, sources=int(len(selected)), clusters=len(clusters),
                   inflation=a.inflation, q=a.q, min_sources=a.min_sources,
                   null_rule={'max_response': a.null_max_response, 'max_median_speed_m_day': a.null_max_speed},
                   floating_min_median_response=a.floating_min_response,
                   tide_response_available=bool(np.isfinite(response).any()),
                   points_file=str(a.points.resolve()) if a.points else None,
                   tool_sha256=digest(Path(__file__)), module_sha256=digest(Path(velocity_blocks.__file__)),
                   elapsed_seconds=time.time()-t0)
    write_json(out/'summary.json', summary)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'points'} for k, v in summary['sizes'].items()},
                     indent=1))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('result', type=Path, help='velocity-pixel result directory (contains recipe.json)')
    p.add_argument('--output', type=Path, required=True, help='new or existing output directory')
    p.add_argument('--model', default='residual_vertical', help=f'fit model, one of {list(MODELS)}')
    p.add_argument('--sizes', default='3000,5000', help='comma-separated block sizes in metres')
    p.add_argument('--crs', help='projected CRS for blocks (default: the result CRS), e.g. EPSG:3413')
    p.add_argument('--min-sources', type=int, default=8, help='sources required for a block to be tested')
    p.add_argument('--inflation', type=float, default=1., help='standard-error multiplier from a site calibration')
    p.add_argument('--q', type=float, default=.05, help='Benjamini-Hochberg false-discovery rate')
    p.add_argument('--points', type=Path, help='CSV of name, longitude, latitude[, amplitude, phase_deg]')
    p.add_argument('--null-max-response', type=float, default=.02, help='max tide response of grounded slow sources')
    p.add_argument('--null-max-speed', type=float, default=.1, help='max median speed of grounded slow blocks, m/day')
    p.add_argument('--floating-min-response', type=float, default=.5, help='min median tide response of floating blocks')
    p.add_argument('--amplitude-max', type=float, help='amplitude colour limit, m/day (default: 99th percentile)')
    p.add_argument('--cache', type=Path, help='velocity cache directory if it moved since the run')
    p.add_argument('--tolerance', type=float, default=1e-9, help='max allowed |recomputed - stored| C/S')
    p.add_argument('--batch', type=int, default=512, help='sources per worker batch; computational only')
    p.add_argument('--workers', type=int, default=8)
    return p


if __name__ == '__main__':
    main(parser().parse_args())
