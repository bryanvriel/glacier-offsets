"""Frozen merge-first stack merger, retained only as a parity reference.

These functions are verbatim copies of the GDAL implementation that produced
the existing merge-first stacks (commit 7625aa5): radar placement, ordered
overwrite and swath-midpoint ownership, ISCE2 validity adjustment, the
ownership raster and date composition. They must not be edited to follow the
production code; the parity tests compare the ISCE3 compositor against them.
Support detection (`invalid_samples`) is the shared matching contract and is
imported rather than frozen.
"""
from copy import deepcopy
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import numpy as np
from osgeo import gdal

from glacier_offsets.matching_grid import invalid_samples
from glacier_offsets.products import checksum, identity
from glacier_offsets.raster_io import bounded_raster_cache

POLICY = {'method': 'isce2-stack-common-valid-last-source-v1',
          'validity': 'resampling-offset-adjusted secondary rectangles intersected across complete dates',
          'precedence': 'ascending swath then radar time; later rectangle overwrites',
          'missing_support': 'mask; no source substitution'}

SWATH_MIDPOINT_POLICY = {
    'method': 'isce2-common-valid-burst-overwrite-swath-midpoint-v1',
    'validity': POLICY['validity'],
    'precedence': 'later bursts overwrite within swaths; interior midpoint between swaths',
    'edge_margin_samples': 8, 'narrow_overlap': 'error',
    'missing_support': POLICY['missing_support']}

EDGE_MARGIN_SAMPLES = 8


def subtract_window(window, excluded):
    """Return disjoint rectangles after removing an intersecting rectangle."""
    y0, y1, x0, x1 = window
    a, b = max(y0, excluded[0]), min(y1, excluded[1])
    c, d = max(x0, excluded[2]), min(x1, excluded[3])
    if a >= b or c >= d:
        return [window]
    return [w for w in ([y0, a, x0, x1], [b, y1, x0, x1],
                         [a, b, x0, c], [a, b, d, x1]) if w[0] < w[1] and w[2] < w[3]]


def _swath(entry):
    match = re.search(r'(?:^|_)(iw[123])$', entry['burst_id'].lower())
    return match.group(1) if match else 'unlabelled'


def _union(intervals):
    result = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(stop, result[-1][1])
        else:
            result.append([start, stop])
    return result


def _range_ownership(bounds, regions, swaths, *, edge_margin_samples=EDGE_MARGIN_SAMPLES):
    edges = sorted({y for windows in regions for w in windows for y in w[:2]})
    selected = [[] for _ in regions]
    previous = {}
    cuts = []
    previous_cuts = {}
    for y0, y1 in zip(edges, edges[1:]):
        active = [(i, w[2], w[3]) for i, windows in enumerate(regions)
                  for w in windows if w[0] <= y0 and y1 <= w[1]]
        support = {swath: _union((a, b) for i, a, b in active if swaths[i] == swath)
                   for swath in sorted(set(swaths))}
        components = sorted((a, b, swath) for swath, intervals in support.items() for a, b in intervals)
        excluded = {swath: [] for swath in support}
        for n, (a, b, near) in enumerate(components):
            for m, (c, d, far) in enumerate(components[n+1:], n+1):
                if near == far or c >= b:
                    continue
                if c <= a or d <= b:
                    raise ValueError(f'{near} / {far}: Interior ownership requires '
                                     'distinct range starts and non-nested swath support')
                if m != n+1:
                    continue
                if b-c < 2*edge_margin_samples:
                    raise ValueError(f'{near} / {far}: overlap has {b-c} samples; interior ownership '
                                     f"requires at least {2*edge_margin_samples} for the edge margin")
                cut = (c+b)//2
                excluded[near].append([cut, b])
                excluded[far].append([c, cut])
                key = (near, far, c, b, cut)
                old = previous_cuts.get(key)
                if old is not None and old['overlap_global'][1] == y0:
                    old['overlap_global'][1] = y1
                else:
                    old = {'near_swath': near, 'far_swath': far,
                           'overlap_global': [y0, y1, c, b], 'cut_global_sample': cut}
                    cuts.append(old)
                    previous_cuts[key] = old
        strip = []
        for i, a, b in active:
            windows = [[y0, y1, a, b]]
            for c, d in excluded[swaths[i]]:
                windows = [part for w in windows for part in subtract_window(w, [y0, y1, c, d])]
            strip.extend((i, w[2], w[3]) for w in windows)
        ordered = sorted((a, b) for _, a, b in strip)
        union = _union((x0, x1) for a, b, x0, x1 in bounds if a <= y0 and y1 <= b)
        if any(b > c for (_, b), (c, _) in zip(ordered, ordered[1:])) or _union(ordered) != union:
            raise ValueError('Interior ownership cannot partition the reference coverage without holes or overlaps')
        for i, a, b in strip:
            key = (i, a, b)
            old = previous.get(key)
            if old is not None and old[1] == y0:
                old[1] = y1
            else:
                old = [y0, y1, a, b]
                selected[i].append(old)
                previous[key] = old
    return selected, cuts


def overwrite_ownership(sources, *, swath_midpoint=False):
    sources = deepcopy(sources)
    sources.sort(key=lambda e: (int(re.search(r'iw([123])$', e['burst_id'])[1])
                                if re.search(r'iw([123])$', e['burst_id']) else 0,
                                e['offset'][0], e['burst_id']))
    bounds = []
    for e in sources:
        dy, dx = e['offset']
        a, b, c, d = e['valid_window']
        bounds.append([a+dy, b+dy, c+dx, d+dx])
    swaths = [_swath(e) for e in sources]
    for i, e in enumerate(sources):
        regions = [bounds[i]]
        for j in range(i+1, len(sources)):
            if not swath_midpoint or swaths[i] == swaths[j]:
                regions = [part for w in regions for part in subtract_window(w, bounds[j])]
        dy, dx = e['offset']
        e['ownership_windows'] = [[a-dy, b-dy, c-dx, d-dx] for a,b,c,d in regions]
    if not swath_midpoint:
        return sources, deepcopy(POLICY)
    regions = [[[a+e['offset'][0], b+e['offset'][0], c+e['offset'][1], d+e['offset'][1]]
                for a,b,c,d in e['ownership_windows']] for e in sources]
    regions, cuts = _range_ownership(bounds, regions, swaths,
                                    edge_margin_samples=SWATH_MIDPOINT_POLICY['edge_margin_samples'])
    for e, windows in zip(sources, regions):
        dy, dx = e['offset']
        e['ownership_windows'] = [[a-dy,b-dy,c-dx,d-dx] for a,b,c,d in windows]
    return sources, dict(SWATH_MIDPOINT_POLICY, cuts=cuts)


def adjusted_window(secondary, shape, extrema, reference=None):
    result = []
    for axis in (0, 1):
        first, last = secondary[2*axis:2*axis+2]
        count = last-first
        low, high = extrema[axis]
        if low > 0 and high > 0:
            first = first - (math.floor(high)-4)
            count = count-8 if first+count-8 < shape[axis] else shape[axis]-first
        elif low < 0 and high < 0:
            first = first - (math.floor(low)-4)
            count = count-8 if first+count-8 < shape[axis] else shape[axis]-first
        elif low < 0 and high > 0:
            first = first - (math.floor(low)-4)
            count += math.floor(low)-8-math.ceil(high)
            count = count if first+count < shape[axis] else shape[axis]-first
        elif reference is not None:
            first, last = reference[2*axis:2*axis+2]
            count = last-first
        first = max(0, first)
        last = min(shape[axis], first+count)
        if first >= last:
            raise ValueError('ISCE2 resampling adjustment leaves no valid burst rectangle')
        result.extend([first, last])
    return result


def offset_extrema(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise ValueError(f'Missing COMPASS resampling offsets: {path}')
    lo, hi = math.inf, -math.inf
    for row in range(0, ds.RasterYSize, 64):
        a = ds.ReadAsArray(0, row, ds.RasterXSize, min(64, ds.RasterYSize-row))
        good = a[np.isfinite(a) & (a >= -10000)]
        if good.size:
            lo, hi = min(lo, float(good.min())), max(hi, float(good.max()))
    if not math.isfinite(lo):
        raise ValueError(f'No usable resampling offsets: {path}')
    return [lo, hi]


def common_windows(sources, dates):
    """Stack-common rectangles, from `prepare_common_layout`."""
    expected = {s['burst_id'] for s in sources}
    complete = {day: data for day, data in dates.items() if set(data['slcs']) == expected}
    windows = {}
    for e in sources:
        key = e['burst_id']
        base = e.get('reference_valid_window', e['valid_window'])
        rectangles = [base, *(data['valid_windows'][key]['window'] for data in complete.values())]
        common = [max(w[0] for w in rectangles), min(w[1] for w in rectangles),
                  max(w[2] for w in rectangles), min(w[3] for w in rectangles)]
        if common[0] >= common[1] or common[2] >= common[3]:
            raise ValueError(f'{key}: stack-common valid rectangle is empty')
        windows[key] = common
    return windows, sorted(complete), sorted(set(dates)-set(complete))


def radar_layout(grids, tolerance=.001):
    import isce3
    if not grids:raise ValueError('No selected radar grids to assemble')
    for grid in grids:
        if any(not np.isfinite(grid[key]) or grid[key]<=0 for key in
               ('prf','range_pixel_spacing','wavelength','length','width','starting_range')) or not np.isfinite(grid['sensing_start']):
            raise ValueError('Radar sampling and dimensions must be finite and positive')
    base = deepcopy(grids[0])
    epoch = isce3.core.DateTime(base['ref_epoch'])
    times = [(isce3.core.DateTime(g['ref_epoch'])-epoch).total_seconds() + g['sensing_start'] for g in grids]
    base['sensing_start'] = min(times)
    base['starting_range'] = min(g['starting_range'] for g in grids)
    offsets=[]
    for index,(g,time) in enumerate(zip(grids,times)):
        for key in ('prf', 'range_pixel_spacing', 'wavelength'):
            if not math.isclose(base[key], g[key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'Radar source {index+1}: incompatible {key}; a common radar resampling is required')
        if base['lookside'] != g['lookside']:
            raise ValueError('Cannot merge different look directions')
        offset = np.array([(time-base['sensing_start'])*base['prf'],
                           (g['starting_range']-base['starting_range'])/base['range_pixel_spacing']])
        rounded = np.rint(offset).astype(int)
        residual = offset-rounded
        if np.max(np.abs(residual)) > tolerance:
            raise ValueError(f'Radar source {index+1}: fractional radar placement {residual.tolist()} exceeds '
                             f'{tolerance} pixels; resample SLCs onto a common radar grid first')
        offsets.append({'offset':rounded.tolist(),'placement_residual_px':residual.tolist()})
    base['length']=max(p['offset'][0]+g['length'] for p,g in zip(offsets,grids))
    base['width']=max(p['offset'][1]+g['width'] for p,g in zip(offsets,grids))
    return base,offsets


def write_vrt(path, shape, entries, field):
    height, width = shape
    tree = ET.Element('VRTDataset', rasterXSize=str(width), rasterYSize=str(height))
    dtype = 'CFloat32' if field in ('reference', 'secondary') else 'UInt16' if field == 'source' else 'Byte' if field == 'layover_shadow_mask' else 'Float64'
    band = ET.SubElement(tree, 'VRTRasterBand', dataType=dtype, band='1')
    ET.SubElement(band, 'NoDataValue').text = '0' if field in ('reference', 'secondary', 'source') else '255' if field == 'layover_shadow_mask' else 'nan'
    for index, entry in enumerate(entries, 1):
        dy, dx = entry['offset']
        source_path = (entry['coreg']['slcs'][field] if field in ('reference', 'secondary')
                       else entry['coreg']['geometry']['layover_shadow_mask' if field == 'source' else field])
        for y0, y1, x0, x1 in entry.get('ownership_windows', [entry['valid_window']]):
            source = ET.SubElement(band, 'ComplexSource' if field == 'source' else 'SimpleSource')
            ET.SubElement(source, 'SourceFilename', relativeToVRT='0').text = str(Path(source_path).resolve())
            ET.SubElement(source, 'SourceBand').text = '1'
            ET.SubElement(source, 'SrcRect', xOff=str(x0), yOff=str(y0), xSize=str(x1-x0), ySize=str(y1-y0))
            ET.SubElement(source, 'DstRect', xOff=str(x0+dx), yOff=str(y0+dy), xSize=str(x1-x0), ySize=str(y1-y0))
            if field == 'source':
                ET.SubElement(source, 'ScaleOffset').text = str(index)
                ET.SubElement(source, 'ScaleRatio').text = '0'
    ET.indent(tree)
    ET.ElementTree(tree).write(path, encoding='utf-8', xml_declaration=True)


def create_raster(path, shape, dtype):
    return gdal.GetDriverByName('GTiff').Create(str(path), shape[1], shape[0], 1, dtype,
                                              options=['TILED=NO', 'BIGTIFF=IF_SAFER'])


def write_layout(root, sources, *, ownership_policy):
    """The merge-first branch of `write_reference_layout`."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = deepcopy(sources)
    grid, placements = radar_layout([e['coreg']['reference_grid'] for e in sources], .001)
    for entry, placement in zip(sources, placements):
        entry.update(placement)
    sources, ownership = overwrite_ownership(sources, swath_midpoint=ownership_policy == SWATH_MIDPOINT_POLICY)
    shape = grid['length'], grid['width']
    for field in ('x', 'y', 'z', 'layover_shadow_mask'):
        write_vrt(root / f'{field}.vrt', shape, sources, field)
    owner_path = root / 'source.tif'
    owner = create_raster(owner_path, shape, gdal.GDT_UInt16)
    owner.GetRasterBand(1).Fill(0)
    with bounded_raster_cache():
        for index, entry in enumerate(sources, 1):
            dy, dx = entry['offset']
            for y0, y1, x0, x1 in entry['ownership_windows']:
                for row in range(y0, y1, 64):
                    h = min(64, y1-row)
                    valid = np.ones((h, x1-x0), bool)
                    band = owner.GetRasterBand(1)
                    block = band.ReadAsArray(x0+dx, row+dy, x1-x0, h)
                    block[valid] = index
                    band.WriteArray(block, x0+dx, row+dy)
    owner = None
    layout_identity = identity({'reference_grid': grid, 'source_index_sha256': checksum(owner_path),
                               'ownership': ownership,
                               'placements': [{k: e[k] for k in ('burst_id', 'offset', 'valid_window', 'ownership_windows')}
                                              for e in sources]})
    coreg = {'reference_grid': grid, 'source_index': str(owner_path.resolve()),
             'geometry': {field: str((root / f'{field}.vrt').resolve()) for field in ('x', 'y', 'z', 'layover_shadow_mask')},
             'geometry_crs': sources[0]['coreg']['geometry_crs'], 'assembly_sources': sources}
    return {'layout_identity': layout_identity, 'coreg': coreg, 'sources': sources, 'ownership': ownership}


def compose(frame, slcs, slc_path, mask_path):
    """The composition loop of `assemble_date`; returns the valid-sample count."""
    shape = tuple(frame['coreg']['reference_grid'][k] for k in ('length', 'width'))
    owner = gdal.Open(frame['coreg']['source_index'])
    count = 0
    with bounded_raster_cache():
        slc = create_raster(slc_path, shape, gdal.GDT_CFloat32)
        mask = create_raster(mask_path, shape, gdal.GDT_Byte)
        with gdal.config_option('GDAL_TIFF_INTERNAL_MASK', 'NO'):
            slc.CreateMaskBand(gdal.GMF_PER_DATASET)
        datasets = {key: gdal.Open(path) for key, path in slcs.items()}
        support = {key: ~invalid_samples(ds) for key, ds in datasets.items()}
        for row in range(0, shape[0], 64):
            h = min(64, shape[0]-row)
            owners = owner.ReadAsArray(0, row, shape[1], h)
            values = np.zeros((h, shape[1]), 'complex64')
            valid = np.zeros(values.shape, bool)
            for index, entry in enumerate(frame['sources'], 1):
                source = datasets.get(entry['burst_id'])
                if source is None:
                    continue
                dy, dx = entry['offset']
                y0, y1 = max(row, dy), min(row+h, dy+source.RasterYSize)
                if y0 >= y1:
                    continue
                block = source.ReadAsArray(0, y0-dy, source.RasterXSize, y1-y0)
                section = (slice(y0-row, y1-row), slice(dx, dx+source.RasterXSize))
                take = (owners[section] == index) & support[entry['burst_id']][y0-dy:y1-dy]
                values[section][take] = block[take]
                valid[section][take] = True
            slc.GetRasterBand(1).WriteArray(values, 0, row)
            mask.GetRasterBand(1).WriteArray(valid.astype('uint8'), 0, row)
            slc.GetRasterBand(1).GetMaskBand().WriteArray(valid.astype('uint8')*255, 0, row)
            count += int(valid.sum())
        slc = mask = None
    return count
