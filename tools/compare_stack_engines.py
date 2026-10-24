#!/usr/bin/env python3
"""Test a selected matcher on a saved pair's exact stack layout, outside production.

Reassemble released SLCs from retained aligned bursts, then match, geocode and
write diagnostics. No SAFE discovery, COMPASS calls or production mutations.
"""
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import match
from glacier_offsets.diagnostics import diagnostics
from glacier_offsets.geocode import geocode
from glacier_offsets.products import ChecksumCache, write_json
from glacier_offsets.stack_config import load_stack_config
from glacier_offsets.stack_storage import assemble_date, read_manifest, release_date


def observations(path):
    with h5py.File(path) as h:
        return {k: h[k][:] for k in ('center_line', 'center_sample', 'source_burst_index',
            'seam_crossing', 'input_valid', 'valid', 'total/azimuth_px', 'total/range_px')}


def compare(pair, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    saved, selected = [observations(p/'radar/offsets.h5') for p in (pair, output)]
    for key in ('center_line', 'center_sample', 'source_burst_index', 'seam_crossing', 'input_valid'):
        np.testing.assert_array_equal(saved[key], selected[key], err_msg=f'Changed {key}')
    common = saved['valid'] & selected['valid']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    report = {'windows': int(common.size), 'baseline_valid': int(saved['valid'].sum()),
              'selected_valid': int(selected['valid'].sum()), 'common_valid': int(common.sum()),
              'identical_grid_ownership_support': True, 'differences_px': {}}
    for row, axis in enumerate(('azimuth', 'range')):
        key = f'total/{axis}_px'
        a, b = saved[key], selected[key]
        delta = b-a
        report['differences_px'][axis] = {}
        for label, mask in [('all', common), ('seam_crossing', common & saved['seam_crossing']),
                            ('interior', common & ~saved['seam_crossing'])]:
            x = delta[mask]
            report['differences_px'][axis][label] = {'n': int(x.size),
                'median_signed': float(np.median(x)) if x.size else None,
                'p90_absolute': float(np.quantile(abs(x), .9)) if x.size else None}
        values = np.r_[a[common], b[common]]
        limit = max(float(np.quantile(abs(values), .95)), .1) if values.size else .1
        dlimit = max(float(np.quantile(abs(delta[common]), .95)), 1/32) if common.any() else 1/32
        for col, (data, title, scale) in enumerate(((a, 'ISCE2 baseline', limit),
                (b, 'ISCE3 magnitude', limit), (delta, 'ISCE3 minus ISCE2', dlimit))):
            image = axes[row,col].imshow(np.where(common, data, np.nan), aspect='auto',
                cmap='RdBu_r', vmin=-scale, vmax=scale)
            axes[row,col].set_title(f'{title}: {axis} [SLC pixels]')
            axes[row,col].set_xlabel('Matching-grid range column')
            axes[row,col].set_ylabel('Matching-grid azimuth row')
            fig.colorbar(image, ax=axes[row,col], shrink=.7)
    fig.suptitle(f'{pair.name}: same common-valid overwrite assembly; common accepted windows')
    fig.savefig(output/'matcher_comparison.png', dpi=150)
    plt.close(fig)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    gdal.UseExceptions()
    pair, output = args.pair.resolve(), args.output.resolve()
    stack = pair.parent.parent
    if output.exists() or output == stack or stack in output.parents or output in stack.parents:
        raise ValueError('Choose a new comparison directory outside the production stack')
    cfg = load_stack_config(args.config)
    if cfg['matching']['engine'] != 'isce3' or cfg['matching']['deramp'] != 0:
        raise ValueError('This comparison requires matching.engine=isce3 and deramp=0')
    original = read_manifest(pair)
    if original['matching']['matching'].get('engine') != 'isce2':
        raise ValueError('Expected an ISCE2 baseline pair')
    if cfg['stack']['processing'] != 'isce2_merge_first':
        raise ValueError('Keep stack.processing=isce2_merge_first')
    # Matching chip, search and stride must remain identical to the saved run.
    for key in ('window', 'search', 'stride', 'first_center', 'raw_oversampling', 'oversampling'):
        if cfg['matching'].get(key) != original['matching']['matching'].get(key):
            raise ValueError(f'Changed matching.{key}')
    if cfg['gross_offsets'] != original['matching']['gross_offsets'] or cfg['gross_offsets']['mode'] == 'velocity':
        raise ValueError('Use the saved zero/constant gross offsets for this comparison')
    frame = read_manifest(stack/'geometry')
    days = original['stack_pair']['acquisition_dates']
    dates = [read_manifest(stack/'acquisitions'/day) for day in days]
    paths = {pair/'manifest.json', pair/'radar/offsets.h5', stack/'stack_manifest.json',
             stack/'geometry/manifest.json', Path(frame['coreg']['source_index'])}
    paths.update(stack/'acquisitions'/day/'manifest.json' for day in days)
    paths.update(Path(p) for date in dates for p in date['slcs'].values())
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
    cfg['output']['directory'] = str(output/'inputs')
    cfg['inputs']['burst_id'] = 'stack'
    cfg.pop('aoi', None)
    checksums = ChecksumCache()
    for date in dates:
        products = date.get('alignment_products', date['products'])
        for path in date['slcs'].values():
            if checksums(path) != products[str(Path(path).resolve())]:
                raise ValueError(f'Aligned source checksum mismatch: {path}')
    output.mkdir(parents=True)
    assemblies = []
    for day, date in zip(days, dates):
        logging.info('Assembling comparison input %s using saved overwrite layout', day)
        assemblies.append(assemble_date(cfg, SimpleNamespace(date=day), frame, date,
                                        force=True, checksums=checksums))
        if checksums(assemblies[-1]['slc']) != date['products'][str(Path(date['slc']).resolve())]:
            raise ValueError(f'{day}: reconstructed SLC differs from baseline checksum')
    manifest = deepcopy(original)
    for key in ('match_identity', 'geocode_identity', 'diagnostics_identity', 'matching', 'matching_summary', 'geocoding'):
        manifest.pop(key, None)
    manifest['stages'] = {'coregister': 'reused saved common-valid overwrite assemblies; no COMPASS'}
    manifest['coreg']['slcs'] = dict(zip(('reference', 'secondary'), (a['slc'] for a in assemblies)))
    logging.info('Matching full pair with ISCE3 magnitude Ampcor')
    match(cfg, output, manifest)
    logging.info('Geocoding and plotting comparison')
    geocode(cfg, output, manifest)
    diagnostics(cfg, output, manifest)
    report = compare(pair, output)
    unchanged = all((p.stat().st_size,p.stat().st_mtime_ns)==s for p,s in before.items())
    if not unchanged:
        raise ValueError('Production input size/mtime changed during comparison')
    report.update(source_pair=str(pair), status='complete', production_sizes_mtimes_unchanged=unchanged,
                  identical_assembled_slc_checksums=True,
                  matching=cfg['matching'], matching_worker_seconds=manifest['matching']['elapsed_seconds'],
                  matching_wall_seconds=manifest['matching']['native_parameters']['wall_seconds'],
                  limitation='Engine comparison on identical inputs; no independent displacement truth.')
    write_json(output/'comparison.json', report)
    for day, assembly in zip(days, assemblies):
        release_date(cfg, SimpleNamespace(date=day), assembly)
    manifest['slc_previews'] = dict(zip(('reference', 'secondary'), (a['slc_preview'] for a in assemblies)))
    write_json(output/'manifest.json', manifest)
    logging.info('Completed comparison: %s', output)


if __name__ == '__main__':
    main()
