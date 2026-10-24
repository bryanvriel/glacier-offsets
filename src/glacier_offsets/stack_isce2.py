"""ISCE2 stack-common validity and ordered overwrite layout preparation."""
from copy import deepcopy
import math
from pathlib import Path
import re

import numpy as np
from osgeo import gdal

from .products import identity, write_json
from .stack_ownership import _range_ownership, _swath, subtract_window

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


def enabled(cfg):
    return cfg.get('stack', {}).get('processing') == 'isce2_merge_first'


def policy(cfg):
    from .stack_ownership import POLICY as legacy
    if not enabled(cfg):
        return legacy
    return SWATH_MIDPOINT_POLICY if cfg['stack'].get('swath_merge') == 'midpoint' else POLICY


def overwrite_ownership(sources, *, swath_midpoint=False):
    sources = deepcopy(sources)
    # Source indices must follow this same order in SLCs, geometry and intervals.
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
    """Port of ISCE2 adjustValidSampleLine_V2, in local COMPASS coordinates.

    COMPASS geo2rdr offsets already map reference indices to secondary-local
    indices, so no additional relative-burst shift or misregistration is added.
    Sign cases not covered by upstream (including both-zero extrema) retain
    the initial reference window. Bounds are clipped to the
    physical output extent before their stack intersection.
    """
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
        # The native routine clamps firstValidSample only. Both axes are also
        # clipped here to the actual output raster; empty results are rejected.
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
        good = a[np.isfinite(a) & (a >= -10000)]  # ISCE2 getValidLines sentinel rule.
        if good.size:
            lo, hi = min(lo, float(good.min())), max(hi, float(good.max()))
    if not math.isfinite(lo):
        raise ValueError(f'No usable resampling offsets: {path}')
    return [lo, hi]


def date_validity(burst, slc, shape, reference=None):
    from .assembly import _annotation_window
    window = _annotation_window((burst, burst))
    if window is None:
        raise ValueError(f'{burst.id}: missing secondary annotation validity')
    paths = [Path(slc).parent/name for name in ('azimuth.off', 'range.off')]
    products = set()
    for path in paths:
        ds = gdal.Open(str(path))
        if ds is None or (ds.RasterYSize, ds.RasterXSize) != shape:
            raise ValueError(f'Resampling offset shape differs from the reference grid: {path}')
        products.update(ds.GetFileList())
    extrema = [offset_extrema(p) for p in paths]
    return {'window': adjusted_window(window, shape, extrema, reference),
            'secondary_annotation': window, 'offset_extrema': extrema,
            'offset_paths': list(map(str, paths)), 'offset_products': sorted(products),
            'method': 'isce2-adjustValidSampleLine-V2'}


def prepare_common_layout(cfg, plan, frame, dates, checksums=None):
    from .stack_pairs import common_map_grid
    from .stack_storage import product_hashes, write_reference_layout
    root = Path(cfg['output']['directory'])/'geometry'
    sources = deepcopy(frame['sources'])
    expected = {s['burst_id'] for s in sources}
    complete = {day: data for day, data in dates.items() if set(data['slcs']) == expected}
    excluded = sorted(set(dates)-set(complete))
    windows = {}
    for e in sources:
        key = e['burst_id']
        base = e.get('reference_valid_window', e['valid_window'])
        e['reference_valid_window'] = list(base)
        rectangles = [base, *(data['valid_windows'][key]['window'] for data in complete.values())]
        common = [max(w[0] for w in rectangles), min(w[1] for w in rectangles),
                  max(w[2] for w in rectangles), min(w[3] for w in rectangles)]
        if common[0] >= common[1] or common[2] >= common[3]:
            raise ValueError(f'{key}: stack-common valid rectangle is empty')
        e['valid_window'] = windows[key] = common
    selected_policy = policy(cfg)
    same_policy = frame.get('ownership', {}).get('method') == selected_policy['method']
    spec = {'method': selected_policy['method'], 'dates': {d: m['alignment_identity'] for d,m in dates.items()},
            'complete_dates': sorted(complete), 'excluded_partial_dates': excluded, 'windows': windows}
    common_identity = identity(spec)
    if same_policy and frame.get('common_validity_identity') == common_identity:
        return frame
    if same_policy and frame.get('common_validity', {}).get('windows') == windows:
        result = dict(frame, common_validity=spec, common_validity_identity=common_identity)
        write_json(root/'manifest.json', result)
        return result
    # Fix the map extent on first preparation, independently of later contraction
    # of stack-common validity. Geometry/offset ownership inside it still agrees.
    target = frame.get('map_grid') or common_map_grid(cfg, frame)
    layout = write_reference_layout(root, sources, checksums=checksums, ownership_policy=selected_policy)
    result = dict(frame, **layout, common_validity=spec, common_validity_identity=common_identity, map_grid=target)
    layout_files = [layout['coreg']['source_index'], *layout['coreg']['geometry'].values()]
    products = dict(frame['products'])
    products.update(product_hashes(layout_files, checksums=checksums))
    result['products'] = products
    write_json(root/'manifest.json', result)
    return result
