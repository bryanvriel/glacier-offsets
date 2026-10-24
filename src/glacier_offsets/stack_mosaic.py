"""ISCE3-backed merge-first stack planning and bounded raster composition.

Planning fixes the common radar grid, integer source placements, stack-common
valid rectangles and ordered ownership once. The ownership raster, geometry
VRTs, date SLC composition and pair intervals all consume that one plan.
Composition copies supported owned complex samples in 64-line ISCE3 raster
blocks; it never interpolates, blends or substitutes another source.

These rules reproduce the earlier GDAL merger exactly (its frozen copy is
`tools/reference_stack_merge.py`). Its policy records remain the
compatibility identifiers of existing stacks; the implementation version is
recorded separately and enters no cache identity. Both merge-first selectors
use this module; `isce2_merge_first` is a deprecated alias.
"""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np
from osgeo import gdal

from .products import identity, write_json
from .stack_ownership import _range_ownership, _swath, subtract_window

IMPLEMENTATION = {'name': 'isce3-stack-mosaic', 'version': 1,
                  'raster_io': 'isce3.io.Raster block reads/writes', 'block_lines': 64}
SELECTORS = ('isce3_merge_first', 'isce2_merge_first')
BLOCK_LINES = 64

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

GEOMETRY_FIELDS = ('x', 'y', 'z', 'layover_shadow_mask')


def enabled(cfg):
    return cfg.get('stack', {}).get('processing') in SELECTORS


def policy(cfg):
    """Both selectors share policy records, so switching keeps cache identities."""
    from .stack_ownership import POLICY as legacy
    if not enabled(cfg):
        return legacy
    return SWATH_MIDPOINT_POLICY if cfg['stack'].get('swath_merge') == 'midpoint' else POLICY


def radar_grid(record):
    """ISCE3 grid for a manifest radar-grid record."""
    import isce3
    return isce3.product.RadarGridParameters(
        record['sensing_start'], record['wavelength'], record['prf'], record['starting_range'],
        record['range_pixel_spacing'], record['lookside'], record['length'], record['width'],
        isce3.core.DateTime(record['ref_epoch']))


def place_sources(records, tolerance=.001):
    """Place source grids on one common grid; reject incompatible or fractional grids.

    Starts are reconciled onto the first grid's epoch. Index arithmetic matches
    the earlier layout bit for bit, so existing placements and residuals agree.
    """
    if not records:
        raise ValueError('No selected radar grids to assemble')
    for record in records:
        if any(not np.isfinite(record[key]) or record[key] <= 0 for key in
               ('prf', 'range_pixel_spacing', 'wavelength', 'length', 'width', 'starting_range')) \
                or not np.isfinite(record['sensing_start']):
            raise ValueError('Radar sampling and dimensions must be finite and positive')
    grids = [radar_grid(r) for r in records]
    first = grids[0]
    times = [(g.ref_epoch-first.ref_epoch).total_seconds() + g.sensing_start for g in grids]
    start, near = min(times), min(g.starting_range for g in grids)
    placements = []
    for index, (record, g, time) in enumerate(zip(records, grids, times)):
        for key in ('prf', 'range_pixel_spacing', 'wavelength'):
            if not math.isclose(getattr(first, key), getattr(g, key), rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f'Radar source {index+1}: incompatible {key}; a common radar resampling is required')
        if records[0]['lookside'] != record['lookside']:
            raise ValueError('Cannot merge different look directions')
        offset = np.array([(time-start)*first.prf, (g.starting_range-near)/first.range_pixel_spacing])
        rounded = np.rint(offset).astype(int)
        residual = offset-rounded
        if np.max(np.abs(residual)) > tolerance:
            raise ValueError(f'Radar source {index+1}: fractional radar placement {residual.tolist()} exceeds '
                             f'{tolerance} pixels; resample SLCs onto a common radar grid first')
        placements.append({'offset': rounded.tolist(), 'placement_residual_px': residual.tolist()})
    common = deepcopy(records[0])
    common.update(sensing_start=start, starting_range=near,
                  length=max(p['offset'][0]+g.length for p, g in zip(placements, grids)),
                  width=max(p['offset'][1]+g.width for p, g in zip(placements, grids)))
    return common, placements


def _order(entry):
    match = re.search(r'iw([123])$', entry['burst_id'])
    return int(match[1]) if match else 0, entry['offset'][0], entry['burst_id']


def ownership(sources, *, swath_midpoint=False):
    """Order sources by swath then radar time, and assign half-open ownership.

    Later rectangles overwrite earlier ones (within each swath for midpoint,
    whose swath support is then cut at range midpoints). Returns the ordered
    entries with local `ownership_windows` and the policy record with cuts.
    """
    sources = sorted(deepcopy(sources), key=_order)
    bounds = []
    for e in sources:
        dy, dx = e['offset']
        a, b, c, d = e['valid_window']
        bounds.append([a+dy, b+dy, c+dx, d+dx])
    swaths = [_swath(e) for e in sources]
    regions = []
    for i in range(len(sources)):
        owned = [bounds[i]]
        for j in range(i+1, len(sources)):
            if not swath_midpoint or swaths[i] == swaths[j]:
                owned = [part for w in owned for part in subtract_window(w, bounds[j])]
        regions.append(owned)
    if swath_midpoint:
        regions, cuts = _range_ownership(bounds, regions, swaths,
                                        edge_margin_samples=SWATH_MIDPOINT_POLICY['edge_margin_samples'])
        record = dict(SWATH_MIDPOINT_POLICY, cuts=cuts)
    else:
        record = deepcopy(POLICY)
    for e, windows in zip(sources, regions):
        dy, dx = e['offset']
        e['ownership_windows'] = [[a-dy, b-dy, c-dx, d-dx] for a, b, c, d in windows]
    return sources, record


@dataclass(frozen=True)
class Placement:
    index: int  # Categorical source ID; 0 is unowned.
    burst_id: str
    offset: tuple
    valid_window: tuple
    ownership_windows: tuple


@dataclass(frozen=True)
class MergePlan:
    """Immutable common grid, placements and ownership of one stack layout."""
    grid_json: str
    placements: tuple
    ownership_json: str

    @classmethod
    def from_sources(cls, grid, sources, ownership_record):
        placements = tuple(Placement(i, e['burst_id'], tuple(e['offset']), tuple(e['valid_window']),
                                     tuple(tuple(w) for w in e['ownership_windows']))
                           for i, e in enumerate(sources, 1))
        return cls(json.dumps(grid, sort_keys=True), placements, json.dumps(ownership_record, sort_keys=True))

    @classmethod
    def from_frame(cls, frame):
        """Adapter from a saved geometry manifest."""
        return cls.from_sources(frame['coreg']['reference_grid'], frame['sources'], frame.get('ownership', {}))

    @property
    def grid(self):
        return json.loads(self.grid_json)

    @property
    def ownership(self):
        return json.loads(self.ownership_json)

    @property
    def radar_grid(self):
        return radar_grid(self.grid)

    @property
    def shape(self):
        grid = self.grid
        return grid['length'], grid['width']

    def owner_block(self, row, stop):
        """Categorical owners of global lines [row, stop)."""
        owners = np.zeros((stop-row, self.shape[1]), 'uint16')
        for p in self.placements:
            dy, dx = p.offset
            for y0, y1, x0, x1 in p.ownership_windows:
                a, b = max(row, y0+dy), min(stop, y1+dy)
                if a < b:
                    owners[a-row:b-row, x0+dx:x1+dx] = p.index
        return owners


def build_plan(sources, ownership_policy, tolerance=.001):
    """Place and order reference sources. Returns the plan and manifest entries."""
    grid, placements = place_sources([e['coreg']['reference_grid'] for e in sources], tolerance)
    entries = deepcopy(sources)
    for entry, placement in zip(entries, placements):
        entry.update(placement)
    entries, record = ownership(entries, swath_midpoint=ownership_policy == SWATH_MIDPOINT_POLICY)
    return MergePlan.from_sources(grid, entries, record), entries


def _open(path, update=False):
    import isce3
    return isce3.io.Raster(str(path), gdal.GA_Update) if update else isce3.io.Raster(str(path))


def _close(*rasters):
    for raster in rasters:
        raster.close_dataset()


def write_layout(root, sources, *, checksums=None, ownership_policy=POLICY):
    """Write the categorical ownership raster and geometry VRTs of one plan."""
    from .assembly import write_vrt
    from .products import checksum
    from .raster_io import bounded_raster_cache
    from .stack_storage import create_raster
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    plan, entries = build_plan(sources, ownership_policy)
    shape = plan.shape
    for field in GEOMETRY_FIELDS:
        write_vrt(root/f'{field}.vrt', shape, entries, field)
    owner_path = root/'source.tif'
    created = create_raster(owner_path, shape, gdal.GDT_UInt16)
    created = None  # Close the GDAL handle; every block is written below.
    with bounded_raster_cache():
        owner = _open(owner_path, update=True)
        try:
            for row in range(0, shape[0], BLOCK_LINES):
                stop = min(row+BLOCK_LINES, shape[0])
                owner.set_block((slice(row, stop), slice(0, shape[1])), plan.owner_block(row, stop), 1)
        finally:
            _close(owner)
    digest = checksum if checksums is None else checksums
    ownership_record = plan.ownership
    layout_identity = identity({'reference_grid': plan.grid, 'source_index_sha256': digest(owner_path),
                               'ownership': ownership_record,
                               'placements': [{k: e[k] for k in ('burst_id', 'offset', 'valid_window', 'ownership_windows')}
                                              for e in entries]})
    coreg = {'reference_grid': plan.grid, 'source_index': str(owner_path.resolve()),
             'geometry': {field: str((root/f'{field}.vrt').resolve()) for field in GEOMETRY_FIELDS},
             'geometry_crs': entries[0]['coreg']['geometry_crs'], 'assembly_sources': entries}
    return {'layout_identity': layout_identity, 'coreg': coreg, 'sources': entries,
            'ownership': ownership_record, 'layout_implementation': IMPLEMENTATION}


def compose(plan, slcs, slc_path, mask_path):
    """Compose one date onto the plan; return its number of valid samples.

    `slcs` maps burst IDs to aligned SLCs; missing bursts stay masked. Output
    lines are written once, in bounded blocks, to the stripped complex TIFF and
    the separate validity raster, then copied to the SLC's external GDAL mask.
    """
    from .matching_grid import invalid_samples
    from .raster_io import bounded_raster_cache
    from .stack_storage import create_assembly_rasters
    height, width = plan.shape
    with bounded_raster_cache():
        create_assembly_rasters(slc_path, mask_path, plan.shape)
        support, sources = {}, {}
        try:
            for p in plan.placements:
                if p.burst_id not in slcs:
                    continue
                ds = gdal.Open(str(slcs[p.burst_id]))
                if ds is None or not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType):
                    raise ValueError(f'{p.burst_id}: aligned SLC is missing or not complex')
                support[p.burst_id] = ~invalid_samples(ds)
                ds = None
                sources[p.burst_id] = _open(slcs[p.burst_id])
            outputs = [_open(slc_path, True), _open(mask_path, True)]
            try:
                count = 0
                for row in range(0, height, BLOCK_LINES):
                    stop = min(row+BLOCK_LINES, height)
                    owners = plan.owner_block(row, stop)
                    values = np.zeros(owners.shape, 'complex64')
                    valid = np.zeros(owners.shape, bool)
                    for p in plan.placements:
                        source = sources.get(p.burst_id)
                        if source is None:
                            continue
                        dy, dx = p.offset
                        y0, y1 = max(row, dy), min(stop, dy+source.length)
                        if y0 >= y1:
                            continue
                        section = (slice(y0-row, y1-row), slice(dx, dx+source.width))
                        take = owners[section] == p.index
                        if not take.any():
                            continue
                        take &= support[p.burst_id][y0-dy:y1-dy]
                        block = source.get_block((slice(y0-dy, y1-dy), slice(0, source.width)), 1)
                        values[section][take] = block[take]
                        valid[section][take] = True
                    key = (slice(row, stop), slice(0, width))
                    outputs[0].set_block(key, values, 1)
                    outputs[1].set_block(key, valid.astype('uint8'), 1)
                    count += int(valid.sum())
            finally:
                _close(*outputs)
        finally:
            _close(*sources.values())
        # An open SLC also owns its external mask band. Write the mask only after
        # the SLC is closed, so the file never has two concurrent writers.
        validity, external = _open(mask_path), _open(f'{slc_path}.msk', True)
        try:
            for row in range(0, height, BLOCK_LINES):
                key = (slice(row, min(row+BLOCK_LINES, height)), slice(0, width))
                external.set_block(key, validity.get_block(key, 1)*np.uint8(255), 1)
        finally:
            _close(validity, external)
    return count


def adjusted_window(secondary, shape, extrema, reference=None):
    """Port of ISCE2 adjustValidSampleLine_V2, in local COMPASS coordinates.

    COMPASS geo2rdr offsets already map reference indices to secondary-local
    indices, so no additional relative-burst shift or misregistration is added.
    Sign cases not covered by upstream (including both-zero extrema) retain
    the initial reference window. Bounds are clipped to the physical output
    extent before their stack intersection.
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
    if gdal.Open(str(path)) is None:
        raise ValueError(f'Missing COMPASS resampling offsets: {path}')
    raster = _open(path)
    lo, hi = math.inf, -math.inf
    try:
        for row in range(0, raster.length, BLOCK_LINES):
            a = raster.get_block((slice(row, min(row+BLOCK_LINES, raster.length)), slice(0, raster.width)), 1)
            good = a[np.isfinite(a) & (a >= -10000)]  # ISCE2 getValidLines sentinel rule.
            if good.size:
                lo, hi = min(lo, float(good.min())), max(hi, float(good.max()))
    finally:
        _close(raster)
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
        products.update(ds.GetFileList())  # Register ancillary headers with the offsets.
    extrema = [offset_extrema(p) for p in paths]
    return {'window': adjusted_window(window, shape, extrema, reference),
            'secondary_annotation': window, 'offset_extrema': extrema,
            'offset_paths': list(map(str, paths)), 'offset_products': sorted(products),
            'method': 'isce2-adjustValidSampleLine-V2'}


def common_windows(sources, dates):
    """Intersect each reference rectangle with every complete date's rectangle.

    Partial dates are excluded, following ISCE2's incomplete-acquisition
    handling, but keep their available imagery inside the common ownership.
    """
    expected = {s['burst_id'] for s in sources}
    complete = sorted(day for day, data in dates.items() if set(data['slcs']) == expected)
    windows = {}
    for e in sources:
        key = e['burst_id']
        rectangles = [e.get('reference_valid_window', e['valid_window']),
                      *(dates[day]['valid_windows'][key]['window'] for day in complete)]
        common = [max(w[0] for w in rectangles), min(w[1] for w in rectangles),
                  max(w[2] for w in rectangles), min(w[3] for w in rectangles)]
        if common[0] >= common[1] or common[2] >= common[3]:
            raise ValueError(f'{key}: stack-common valid rectangle is empty')
        windows[key] = common
    return windows, complete, sorted(set(dates)-set(complete))


def prepare_common_layout(cfg, plan, frame, dates, checksums=None):
    """Fix stack-common validity, rewriting the layout only when it changes."""
    from .stack_pairs import common_map_grid
    from .stack_storage import product_hashes
    root = Path(cfg['output']['directory'])/'geometry'
    sources = deepcopy(frame['sources'])
    windows, complete, excluded = common_windows(sources, dates)
    for e in sources:
        e['reference_valid_window'] = list(e.get('reference_valid_window', e['valid_window']))
        e['valid_window'] = windows[e['burst_id']]
    selected_policy = policy(cfg)
    same_policy = frame.get('ownership', {}).get('method') == selected_policy['method']
    spec = {'method': selected_policy['method'], 'dates': {d: m['alignment_identity'] for d, m in dates.items()},
            'complete_dates': complete, 'excluded_partial_dates': excluded, 'windows': windows}
    common_identity = identity(spec)
    if same_policy and frame.get('common_validity_identity') == common_identity:
        return frame
    if same_policy and frame.get('common_validity', {}).get('windows') == windows:
        # An inventory change with identical rectangles keeps the verified layout.
        result = dict(frame, common_validity=spec, common_validity_identity=common_identity)
        write_json(root/'manifest.json', result)
        return result
    # Fix the map extent on first preparation, independently of later contraction
    # of stack-common validity. Geometry/offset ownership inside it still agrees.
    target = frame.get('map_grid') or common_map_grid(cfg, frame)
    layout = write_layout(root, sources, checksums=checksums, ownership_policy=selected_policy)
    result = dict(frame, **layout, common_validity=spec, common_validity_identity=common_identity, map_grid=target)
    products = dict(frame['products'])
    products.update(product_hashes([layout['coreg']['source_index'], *layout['coreg']['geometry'].values()],
                                   checksums=checksums))
    result['products'] = products
    write_json(root/'manifest.json', result)
    return result
