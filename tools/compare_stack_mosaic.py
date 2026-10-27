#!/usr/bin/env python3
"""Real-data parity gate for the ISCE3 merge-first compositor, outside production.

From a saved merge-first stack, rebuild both ownership policies with the frozen
GDAL reference and the ISCE3 compositor on the same retained aligned bursts.
Compare every layout and date raster block by block, including unsupported
regions. Then match selected pairs on each implementation's SLCs, plus a
reference replay control, with identical worker and matching settings.
Production files are only read; sizes, timestamps and layout checksums are
verified afterwards. Each composition runs in its own process for timing,
block I/O and peak-RSS measurements.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import logging
import multiprocessing
from pathlib import Path
import resource
import shutil
import sys
import time

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets import stack_mosaic as mosaic
from glacier_offsets.ampcor import match
from glacier_offsets.geocode import geocode
from glacier_offsets.products import ChecksumCache, checksum, write_json
from glacier_offsets.raster_io import bounded_raster_cache, require_native_mmap
from glacier_offsets.stack_config import load_stack_config
from glacier_offsets.stack_storage import read_manifest
from tools import reference_stack_merge as frozen

POLICIES = {'overwrite': mosaic.POLICY, 'midpoint': mosaic.SWATH_MIDPOINT_POLICY}
IMPLEMENTATIONS = ('reference', 'isce3')
REFINEMENT_STEP_PX = 1/64
BLOCK = 64
GEO_FIELDS = ('azimuth_displacement_m.tif', 'range_displacement_m.tif', 'valid_mask.tif',
              'quality_valid_mask.tif', 'source_burst_index.tif')


def usage(started):
    scale = 1 if sys.platform == 'darwin' else 1024
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {'wall_seconds': time.perf_counter()-started, 'peak_rss_bytes': int(r.ru_maxrss*scale),
            'block_reads': int(r.ru_inblock), 'block_writes': int(r.ru_oublock)}


def layout_job(kind, root, sources, policy):
    gdal.UseExceptions()
    started = time.perf_counter()
    writer = frozen.write_layout if kind == 'reference' else mosaic.write_layout
    layout = writer(root, sources, ownership_policy=policy)
    return layout, usage(started)


def compose_job(kind, frame, slcs, slc_path, valid_path):
    gdal.UseExceptions()
    started = time.perf_counter()
    if kind == 'reference':
        count = frozen.compose(frame, slcs, slc_path, valid_path)
    else:
        count = mosaic.compose(mosaic.MergePlan.from_frame(frame), slcs, slc_path, valid_path)
    return count, usage(started)


def run(pool, function, *args):
    return pool.submit(function, *args).result()


def compare_rasters(first, second, mask=False):
    """Count differing samples block by block; NaNs compare equal to NaNs."""
    a, b = gdal.Open(str(first)), gdal.Open(str(second))
    if (a.RasterYSize, a.RasterXSize) != (b.RasterYSize, b.RasterXSize):
        return {'identical': False, 'error': 'shape differs'}
    bands = [d.GetRasterBand(1) for d in (a, b)]
    if mask:
        bands = [band.GetMaskBand() for band in bands]
    if bands[0].DataType != bands[1].DataType:
        return {'identical': False, 'error': 'data type differs'}
    different, largest = 0, 0.
    with bounded_raster_cache():
        for row in range(0, a.RasterYSize, BLOCK):
            h = min(BLOCK, a.RasterYSize-row)
            x, y = (band.ReadAsArray(0, row, a.RasterXSize, h) for band in bands)
            if np.iscomplexobj(x):
                x, y = x.view(np.float32 if x.dtype == np.complex64 else np.float64), \
                       y.view(np.float32 if y.dtype == np.complex64 else np.float64)
            changed = (x != y) & ~(np.isnan(x) & np.isnan(y)) if x.dtype.kind == 'f' else x != y
            different += int(changed.sum())
            if changed.any():
                with np.errstate(invalid='ignore'):
                    largest = max(largest, float(np.nanmax(np.abs(x[changed].astype(float)-y[changed].astype(float)))))
    return {'identical': different == 0, 'shape': [a.RasterYSize, a.RasterXSize],
            'data_type': gdal.GetDataTypeName(bands[0].DataType), 'different_samples': different,
            'max_abs_difference': largest}


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def compare_layouts(reference, layout, production=None):
    result = {'sources_identical': same(reference['sources'], layout['sources']),
              'ownership_identical': same(reference['ownership'], layout['ownership']),
              'grid_identical': same(reference['coreg']['reference_grid'], layout['coreg']['reference_grid']),
              'source_index': compare_rasters(reference['coreg']['source_index'], layout['coreg']['source_index']),
              'geometry': {f: compare_rasters(reference['coreg']['geometry'][f], layout['coreg']['geometry'][f])
                           for f in mosaic.GEOMETRY_FIELDS},
              'source_index_bytes_identical': checksum(reference['coreg']['source_index']) ==
                                              checksum(layout['coreg']['source_index']),
              'layout_identity_identical': reference['layout_identity'] == layout['layout_identity']}
    if production is not None:
        result['production'] = {
            'ownership_identical': same(production['ownership'], layout['ownership']),
            'windows_identical': same([e['ownership_windows'] for e in production['sources']],
                                      [e['ownership_windows'] for e in layout['sources']]),
            'source_index': compare_rasters(production['coreg']['source_index'], layout['coreg']['source_index']),
            'geometry': {f: compare_rasters(production['coreg']['geometry'][f], layout['coreg']['geometry'][f])
                         for f in mosaic.GEOMETRY_FIELDS},
            'layout_identity_identical': production['layout_identity'] == layout['layout_identity']}
    result['identical'] = (result['sources_identical'] and result['ownership_identical'] and result['grid_identical']
                           and result['source_index']['identical']
                           and all(v['identical'] for v in result['geometry'].values()))
    return result


def compare_offsets(first, second):
    """Every dataset of two offset products, with validity transitions."""
    report = {}
    with h5py.File(first) as a, h5py.File(second) as b:
        names = []
        a.visit(lambda name: names.append(name) if isinstance(a[name], h5py.Dataset) else None)
        for name in names:
            x, y = a[name][()], b[name][()] if name in b else None
            if y is None or x.shape != y.shape:
                report[name] = {'identical': False, 'error': 'missing or shape differs'}
                continue
            if x.dtype.kind == 'b':
                report[name] = {'identical': bool(np.array_equal(x, y)),
                                'gained': int((~x & y).sum()), 'lost': int((x & ~y).sum())}
                continue
            if x.dtype.kind not in 'fiuc':
                continue
            x, y = x.astype(float), y.astype(float)
            changed = (x != y) & ~(np.isnan(x) & np.isnan(y))
            delta = np.abs(x-y)[changed]
            report[name] = {'identical': not changed.any(), 'different_samples': int(changed.sum()),
                            'nan_transitions': int((np.isnan(x) != np.isnan(y)).sum()),
                            'max_abs_difference': float(np.nanmax(delta)) if delta.size and np.isfinite(delta).any() else 0.,
                            'p99_abs_difference': float(np.nanquantile(delta, .99)) if delta.size and np.isfinite(delta).any() else 0.}
    return report


def max_offset_difference(report):
    keys = [k for k in report if k.startswith(('total/', 'residual/'))]
    return max((report[k].get('max_abs_difference', 0.) for k in keys), default=0.)


def _vanished(function, path, error):
    """exFAT AppleDouble sidecars disappear with their file; ignore only that."""
    if not isinstance(error, FileNotFoundError):
        raise error


def snapshot(paths):
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path, help='Production merge-first stack configuration (read only)')
    parser.add_argument('--output', type=Path, required=True, help='New comparison directory outside the stack')
    parser.add_argument('--pairs', nargs='*', default=['2017-01-01_2017-01-07', '2017-01-19_2017-01-25'])
    parser.add_argument('--policies', nargs='+', choices=sorted(POLICIES), default=['overwrite', 'midpoint'])
    parser.add_argument('--match-policy', choices=sorted(POLICIES), default=None,
                        help='Ownership policy used for full-pair matching (default: the production policy)')
    parser.add_argument('--keep-slcs', action='store_true', help='Retain composed SLCs after completion')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    gdal.UseExceptions()
    cfg = load_stack_config(args.config)
    stack, output = Path(cfg['output']['directory']).resolve(), args.output.resolve()
    if output.exists() or output == stack or stack in output.parents or output in stack.parents:
        raise ValueError('Choose a new comparison directory outside the production stack')
    if cfg['matching'].get('engine', 'isce3') != 'isce3':
        raise ValueError('The matching gate uses the unchanged ISCE3 matcher')
    frame = read_manifest(stack/'geometry')
    production_policy = next(name for name, p in POLICIES.items() if p['method'] == frame['ownership']['method'])
    match_policy = args.match_policy or production_policy
    days = sorted(p.parent.name for p in (stack/'acquisitions').glob('*/manifest.json'))
    dates = {day: read_manifest(stack/'acquisitions'/day) for day in days}
    pairs = {key: read_manifest(stack/'pairs'/key) for key in args.pairs}
    for key, pair in pairs.items():
        saved = pair['matching']['matching']
        for name in ('window', 'search', 'stride', 'batch', 'zoom', 'deramp', 'oversampling', 'raw_oversampling', 'backend'):
            if cfg['matching'].get(name) != saved.get(name):
                raise ValueError(f'{key}: matching.{name} differs from the saved production pair')
        if cfg['gross_offsets'] != pair['matching']['gross_offsets']:
            raise ValueError(f'{key}: gross offsets differ from the saved production pair')

    layout_files = [stack/'geometry/manifest.json', Path(frame['coreg']['source_index']),
                    *map(Path, frame['coreg']['geometry'].values())]
    watched = [*layout_files, stack/'stack_manifest.json',
               *(stack/'acquisitions'/d/'manifest.json' for d in days),
               *(Path(p) for d in dates.values() for p in d['slcs'].values()),
               *(stack/'pairs'/k/f for k in pairs for f in ('manifest.json', 'radar/offsets.h5'))]
    before, layout_checksums = snapshot(watched), {str(p): checksum(p) for p in layout_files}

    logging.info('Verifying retained reference and aligned burst products')
    checksums = ChecksumCache()
    for entry in frame['sources']:
        for name, digest in read_manifest(Path(entry['manifest']).parent)['products'].items():
            if checksums(name) != digest:
                raise ValueError(f'Reference product checksum mismatch: {name}')
    for day, date in dates.items():
        products = date.get('alignment_products', date['products'])
        for path in date['slcs'].values():
            if checksums(path) != products[str(Path(path).resolve())]:
                raise ValueError(f'{day}: aligned source checksum mismatch: {path}')
    output.mkdir(parents=True)
    report = {'config': str(args.config.resolve()), 'stack': str(stack), 'production_policy': production_policy,
              'match_policy': match_policy, 'implementation': mosaic.IMPLEMENTATION, 'dates': days,
              'pairs': args.pairs, 'layouts': {}, 'assemblies': {}, 'matching': {}, 'status': 'running'}
    context = multiprocessing.get_context('spawn')
    layouts, composed = {}, {}
    with ProcessPoolExecutor(1, mp_context=context, max_tasks_per_child=1) as pool:
        for name in args.policies:
            policy = POLICIES[name]
            layouts[name] = {}
            for kind in IMPLEMENTATIONS:
                logging.info('Writing %s layout with the %s implementation', name, kind)
                layout, cost = run(pool, layout_job, kind, output/name/kind/'geometry', frame['sources'], policy)
                layouts[name][kind] = layout
                report['layouts'].setdefault(name, {})[f'{kind}_cost'] = cost
            report['layouts'][name].update(compare_layouts(*layouts[name].values(),
                                                           frame if name == production_policy else None))
            write_json(output/'comparison.json', report)
            for day in days:
                entry = report['assemblies'].setdefault(name, {}).setdefault(day, {})
                for kind in IMPLEMENTATIONS:
                    directory = output/name/kind/'acquisitions'/day
                    directory.mkdir(parents=True)
                    logging.info('Composing %s %s with the %s implementation', name, day, kind)
                    count, cost = run(pool, compose_job, kind, layouts[name][kind], dates[day]['slcs'],
                                      directory/'slc.tif', directory/'valid.tif')
                    composed[(name, kind, day)] = directory
                    entry[kind] = {'valid_samples': count, 'cost': cost}
                a, b = composed[(name, 'reference', day)], composed[(name, 'isce3', day)]
                require_native_mmap(gdal.Open(str(b/'slc.tif')))
                entry.update(slc=compare_rasters(a/'slc.tif', b/'slc.tif'),
                             external_mask=compare_rasters(a/'slc.tif', b/'slc.tif', mask=True),
                             validity=compare_rasters(a/'valid.tif', b/'valid.tif'),
                             counts_identical=entry['reference']['valid_samples'] == entry['isce3']['valid_samples'])
                if name == production_policy:
                    saved = {**dates[day].get('slc_cleanup', {}).get('files', {}), **dates[day]['products']}
                    recorded = saved.get(str(Path(dates[day]['slc']).resolve()))
                    entry['production_slc_sha256_matches'] = {
                        kind: recorded is not None and checksum(composed[(name, kind, day)]/'slc.tif') == recorded
                        for kind in IMPLEMENTATIONS}
                entry['identical'] = all(entry[k]['identical'] for k in ('slc', 'external_mask', 'validity')) \
                    and entry['counts_identical']
                needed = name == match_policy and any(day in key.split('_') for key in args.pairs)
                if not needed and not args.keep_slcs:
                    for kind in IMPLEMENTATIONS:
                        for path in composed[(name, kind, day)].iterdir():
                            path.unlink(missing_ok=True)  # exFAT sidecars vanish with their file.
                write_json(output/'comparison.json', report)

    effective = deepcopy(cfg)
    effective.pop('aoi', None)
    effective['inputs']['burst_id'] = 'stack'
    for key, original in pairs.items():
        earlier, later = original['stack_pair']['acquisition_dates']
        runs = {}
        for label, kind in (('reference', 'reference'), ('isce3', 'isce3'), ('replay', 'reference')):
            directory = output/'matching'/key/label
            directory.mkdir(parents=True)
            effective['output']['directory'] = str(directory/'inputs')
            manifest = deepcopy(original)
            for name in ('match_identity', 'geocode_identity', 'diagnostics_identity', 'matching',
                         'matching_summary', 'geocoding'):
                manifest.pop(name, None)
            layout = layouts[match_policy][kind]
            if not same([e['ownership_windows'] for e in manifest['coreg']['assembly_sources']],
                        [e['ownership_windows'] for e in layout['sources']]):
                raise ValueError(f'{key}: saved pair ownership differs from the {match_policy} layout')
            manifest['stages'] = {'coregister': f'{kind} merge-first compositor on retained aligned bursts'}
            manifest['coreg'].update(source_index=layout['coreg']['source_index'],
                                     geometry=layout['coreg']['geometry'],
                                     slcs={side: str(composed[(match_policy, kind, day)]/'slc.tif')
                                           for side, day in (('reference', earlier), ('secondary', later))})
            logging.info('Matching %s on %s SLCs (%s)', key, kind, label)
            started = time.perf_counter()
            match(effective, directory, manifest)
            geocode(effective, directory, manifest)
            runs[label] = {'wall_seconds': time.perf_counter()-started,
                           'native_wall_seconds': manifest['matching']['native_parameters']['wall_seconds'],
                           'workers': manifest['matching']['native_parameters'].get('workers'),
                           'matching_summary': manifest['matching_summary']}
        result = {'runs': runs}
        root = output/'matching'/key
        for label, (a, b) in {'isce3_vs_reference': ('reference', 'isce3'),
                              'replay_vs_reference': ('reference', 'replay')}.items():
            result[label] = compare_offsets(root/a/'radar/offsets.h5', root/b/'radar/offsets.h5')
            result[f'{label}_geo'] = {f: compare_rasters(root/a/'geo'/f, root/b/'geo'/f)
                                      for f in GEO_FIELDS if (root/a/'geo'/f).is_file()}
        result['production_vs_reference'] = compare_offsets(stack/'pairs'/key/'radar/offsets.h5',
                                                            root/'reference/radar/offsets.h5')
        masks = ('input_valid', 'source_burst_index', 'seam_crossing', 'center_line', 'center_sample')
        bound = max(REFINEMENT_STEP_PX + 1e-9, max_offset_difference(result['replay_vs_reference']))
        result['gate'] = {'masks_identical': all(result['isce3_vs_reference'].get(k, {}).get('identical', False)
                                                 for k in masks),
                          'max_offset_difference_px': max_offset_difference(result['isce3_vs_reference']),
                          'replay_max_offset_difference_px': max_offset_difference(result['replay_vs_reference']),
                          'bound_px': bound}
        result['gate']['passed'] = result['gate']['masks_identical'] and \
            result['gate']['max_offset_difference_px'] <= bound
        report['matching'][key] = result
        write_json(output/'comparison.json', report)

    after = snapshot(watched)
    report['production_unchanged'] = {
        'sizes_mtimes': after == before,
        'layout_checksums': all(checksum(Path(p)) == digest for p, digest in layout_checksums.items())}
    report['gates'] = {
        'layouts_identical': all(v['identical'] for v in report['layouts'].values()),
        'production_layout_reproduced': all(
            v['production']['source_index']['identical'] and v['production']['windows_identical'] and
            all(g['identical'] for g in v['production']['geometry'].values())
            for v in report['layouts'].values() if 'production' in v),
        'assemblies_identical': all(e['identical'] for p in report['assemblies'].values() for e in p.values()),
        'matching_within_bounds': all(v['gate']['passed'] for v in report['matching'].values()),
        'production_unchanged': all(report['production_unchanged'].values())}
    report['status'] = 'passed' if all(report['gates'].values()) else 'failed'
    write_json(output/'comparison.json', report)
    if not args.keep_slcs:
        for directory in composed.values():
            shutil.rmtree(directory, onexc=_vanished)
    logging.info('Comparison %s: %s', report['status'], output/'comparison.json')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
