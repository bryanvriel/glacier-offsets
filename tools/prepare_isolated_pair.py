#!/usr/bin/env python3
"""Isolated deramp=0 baseline for any two dates of an existing stack.

Run as python -m tools.prepare_isolated_pair CONFIG STACK TEMPLATE EARLIER LATER --output NEW.
TEMPLATE is a completed pair in STACK that supplies the fixed geometry, ownership,
matching grid and center geometry. NEW/stack holds copies of the geometry, stack and
date manifests only; merged SLCs are assembled there from the retained aligned bursts
and released afterwards. NEW/stack/pairs/EARLIER_LATER can then be passed to
compare_swath_overlap.py and compare_merged_preprocessing.py. Production is read-only.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import match
from glacier_offsets.diagnostics import diagnostics
from glacier_offsets.geocode import geocode
from glacier_offsets.products import ChecksumCache, identity, write_json
from glacier_offsets.stack_config import load_stack_config
from glacier_offsets.stack_storage import assemble_date, read_manifest

DERIVED = ('match_identity', 'geocode_identity', 'diagnostics_identity', 'matching', 'matching_summary',
           'geocoding', 'slc_previews', 'workflow_source_at_validation', 'diagnostics_software')


def derive_pair(template, days, dates, scene_starts, shim):
    """Template pair manifest re-keyed to new dates; geometry and ownership stay fixed."""
    manifest = deepcopy(template)
    for key in DERIVED:
        manifest.pop(key, None)
    sources = manifest['assembly']['sources']
    for entry in sources:
        times = [date['burst_times'].get(entry['burst_id']) for date in dates]
        interval = None
        if all(times):
            interval = (datetime.fromisoformat(times[1])-datetime.fromisoformat(times[0])).total_seconds()
            if interval <= 0:
                raise ValueError(f"{entry['burst_id']}: nonpositive burst interval")
        entry.update(acquisition_times=times, interval_seconds=interval)
    manifest['coreg']['assembly_sources'] = deepcopy(sources)
    manifest['coreg']['slcs'] = dict(zip(('reference', 'secondary'), (date['slc'] for date in dates)))
    manifest['coreg_specification']['inputs'] = {day: date['specification'] for day, date in zip(days, dates)}
    manifest['stack_pair'].update(acquisition_dates=list(days), geometry_manifest=str(shim/'geometry/manifest.json'))
    manifest['acquisition_times'] = list(scene_starts)
    manifest['coreg_identity'] = identity({'template': template['coreg_identity'], 'dates': list(days),
        'assemblies': [date.get('assembly_identity') for date in dates],
        'burst_intervals': [[e['burst_id'], e['acquisition_times'], e['interval_seconds']] for e in sources]})
    manifest['assembly']['identity'] = manifest['coreg_identity']
    manifest['stages'] = {'coregister': 'reused aligned acquisitions; isolated reassembly; no COMPASS'}
    return manifest


def saved_merged_checksum(date):
    """Production checksum of the merged SLC, whether retained or journaled as released."""
    path = str(Path(date['slc']).resolve())
    return date.get('products', {}).get(path) or date.get('slc_cleanup', {}).get('files', {}).get(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path);parser.add_argument('stack', type=Path)
    parser.add_argument('template');parser.add_argument('earlier');parser.add_argument('later')
    parser.add_argument('--output', type=Path, required=True);parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args(argv);stack = args.stack.resolve();out = args.output.resolve()
    if out.exists() or out == stack or stack in out.parents or out in stack.parents:
        raise ValueError('Choose a new directory outside the production stack')
    if args.workers < 1:raise ValueError('workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s');gdal.UseExceptions()
    days = (args.earlier, args.later)
    if days[0] >= days[1]:raise ValueError('earlier must precede later')
    template_dir = stack/'pairs'/args.template;template = read_manifest(template_dir)
    frame = read_manifest(stack/'geometry')
    if template['matching']['gross_offsets']['mode'] != 'zero':raise ValueError('Template must use zero gross offsets')
    if frame['geometry_identity'] != template['stack_pair']['geometry_identity']:
        raise ValueError('Template pair no longer matches the stack geometry')
    dates = [read_manifest(stack/'acquisitions'/day) for day in days]
    for day, date in zip(days, dates):
        if date.get('status') != 'complete' or not date.get('slcs'):raise ValueError(f'{day}: aligned acquisition incomplete')
    plan = json.loads((stack/'plan.json').read_text())
    inventory = {e['date']: e for e in plan['inventory']}
    shim = out/'stack';pair = shim/'pairs'/f'{days[0]}_{days[1]}'
    watched = {template_dir/'manifest.json', template_dir/'geo/center_geometry.h5', stack/'geometry/manifest.json',
               stack/'stack_manifest.json', stack/'plan.json', *(stack/'acquisitions'/day/'manifest.json' for day in days)}
    watched.update(Path(p) for date in dates for p in date['alignment_products'])
    before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
    pair.mkdir(parents=True);(shim/'geometry').mkdir()
    write_json(out/'experiment.json', {'status': 'preparing', 'stack': str(stack), 'template': args.template, 'dates': days})
    write_json(shim/'geometry/manifest.json', frame)
    write_json(shim/'stack_manifest.json', read_manifest(stack))
    for day, date in zip(days, dates):write_json(shim/'acquisitions'/day/'manifest.json', date)
    cfg = load_stack_config(args.config)
    cfg['matching'] = dict(deepcopy(template['matching']['matching']), deramp=0, engine='isce3', backend='cpu')
    cfg['gross_offsets'] = deepcopy(template['matching']['gross_offsets'])
    cfg['inputs'].update(burst_id='stack', polarization=template['coreg_specification']['polarization'])
    cfg.pop('aoi', None);cfg['output']['directory'] = str(shim)
    cfg['geocoding'] = {k: template['geocoding'][k] for k in ('epsg', 'spacing_m', 'resampling')}
    cfg['geocoding']['mask'] = template['geocoding']['source_mask']
    os.environ['OMP_NUM_THREADS'] = str(args.workers)
    checksums = ChecksumCache();assembled = [];verification = {}
    for day, date in zip(days, dates):
        for path in date['slcs'].values():
            if checksums(path) != date['alignment_products'][str(Path(path).resolve())]:
                raise ValueError(f'Changed aligned source: {path}')
        logging.info('Assembling merged SLC in isolation: %s', day)
        result = assemble_date(cfg, SimpleNamespace(date=day), frame, date, force=True, checksums=checksums)
        expected = saved_merged_checksum(date);actual = checksums(result['slc'])
        if expected is not None and actual != expected:
            raise ValueError(f'{day}: isolated merged SLC differs from the production record')
        verification[day] = 'matches production merged SLC checksum' if expected else 'no production checksum recorded'
        assembled.append(result)
    manifest = derive_pair(template, days, assembled, [inventory[day]['start'] for day in days], shim)
    with h5py.File(template_dir/'geo/center_geometry.h5') as h:
        geometry = tuple(h[k][:] for k in ('longitude_deg', 'latitude_deg', 'height_m',
                                            'azimuth_ground_spacing_m_per_slc_line', 'geometry_valid'))
    logging.info('Matching deramp=0 baseline: %s', pair.name)
    match(cfg, pair, manifest)
    with h5py.File(pair/'radar/offsets.h5') as h, h5py.File(template_dir/'radar/offsets.h5') as t:
        for key in ('center_line', 'center_sample', 'source_burst_index', 'seam_crossing'):
            np.testing.assert_array_equal(h[key][:], t[key][:], err_msg=f'Template grid changed: {key}')
    logging.info('Geocoding and diagnostics: %s', pair.name)
    with patch('glacier_offsets.geocode.center_geometry', return_value=geometry):geocode(cfg, pair, manifest)
    diagnostics(cfg, pair, manifest)
    for day, result in zip(days, assembled):
        for path in (Path(result['slc']), Path(result['slc']+'.msk'), Path(result['validity_mask'])):
            path.unlink(missing_ok=True)
    unchanged = all((p.stat().st_size, p.stat().st_mtime_ns) == stamp for p, stamp in before.items())
    report = {'status': 'complete' if unchanged else 'incomplete', 'stack': str(stack), 'template': args.template,
              'pair': str(pair), 'dates': days, 'missions': [inventory[day]['mission'] for day in days],
              'merged_slc_verification': verification, 'workers': args.workers, 'matching': cfg['matching'],
              'production_sizes_mtimes_unchanged': unchanged, 'temporary_merged_inputs_released': True}
    if args.template == pair.name:
        with h5py.File(pair/'radar/offsets.h5') as h, h5py.File(template_dir/'radar/offsets.h5') as t:
            both = h['valid'][:] & t['valid'][:]
            report['production_replay'] = {'valid_changed': int((h['valid'][:] != t['valid'][:]).sum()),
                **{f'{axis}_max_abs_px': float(abs(h[f'total/{axis}_px'][:]-t[f'total/{axis}_px'][:])[both].max())
                   for axis in ('azimuth', 'range')},
                **{f'{axis}_changed_n': int((h[f'total/{axis}_px'][:] != t[f'total/{axis}_px'][:])[both].sum())
                   for axis in ('azimuth', 'range')}}
    write_json(out/'experiment.json', report)
    if not unchanged:raise ValueError('Production files changed during experiment')
    logging.info('Completed isolated baseline: %s', pair)
    return 0


if __name__ == '__main__':raise SystemExit(main())
