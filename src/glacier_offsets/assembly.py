"""ISCE2-style radar-grid assembly of COMPASS reference/coregistered SLCs.

Sources are copied without interpolation or complex averaging. Later valid
rectangles own overlaps, identically for both acquisitions and geometry.
"""
from copy import deepcopy
import json
import logging
import math
from pathlib import Path
import re
import shutil
import tempfile
from xml.etree import ElementTree as ET

import numpy as np
from osgeo import gdal
from pyproj import CRS

from .products import checksum, identity, write_json
from .raster_io import bounded_raster_cache


METHOD = 'isce2-stack-valid-rectangles-last-source-v1'


def settings(cfg):
    supplied=cfg.get('burst_merge', {})
    if not isinstance(supplied,dict):raise ValueError('burst_merge must be a YAML mapping')
    value = {'enabled': True, 'alignment_tolerance_px': .001} | supplied
    if set(value) - {'enabled', 'alignment_tolerance_px'}:
        raise ValueError('burst_merge accepts enabled and alignment_tolerance_px')
    if type(value['enabled']) is not bool:
        raise ValueError('burst_merge.enabled must be true or false')
    tolerance = value['alignment_tolerance_px']
    if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 0 <= tolerance <= .5:
        raise ValueError('burst_merge.alignment_tolerance_px must be between 0 and 0.5')
    return value


def _source_order(item):
    root, manifest = item
    match = re.search(r'_iw([123])$', root.name)
    return (int(match[1]) if match else 0, manifest['acquisition_times'][0], root.name)


def _annotation_window(pair):
    if pair is None:
        return None
    native = pair[0].native
    names = ('first_valid_line', 'last_valid_line', 'first_valid_sample', 'last_valid_sample')
    if not all(hasattr(native, name) for name in names):
        return None
    y0, y1, x0, x1 = [int(getattr(native, name)) for name in names]
    return [y0, y1 + 1, x0, x1 + 1]


def radar_layout(grids, tolerance=.001):
    """Metadata-only common-grid preflight, also usable before coregistration."""
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


def validate_native_layout(pairs,cfg):
    grids=[]
    for pair in pairs:
        grid=pair[0].native.as_isce3_radargrid()
        data={key:getattr(grid,key) for key in ('prf','range_pixel_spacing','wavelength','length','width','starting_range','sensing_start')}
        data.update(ref_epoch=str(grid.ref_epoch),lookside=str(grid.lookside).rsplit('.',1)[-1].lower())
        grids.append(data)
    return radar_layout(grids,settings(cfg)['alignment_tolerance_px'])


def plan_sources(sources, tolerance=.001, native_pairs=None):
    """Describe a common time/range grid; reject incompatible or fractional grids."""
    if not sources:raise ValueError('No selected coregistered bursts to assemble')
    ordered=sorted(sources,key=_source_order)
    base,placements=radar_layout([m['coreg']['reference_grid'] for _,m in ordered],tolerance)
    first_crs=CRS.from_user_input(ordered[0][1]['coreg']['geometry_crs'])
    entries=[]
    for (root,m),placement in zip(ordered,placements):
        g=m['coreg']['reference_grid']
        if not m.get('coreg_identity'):
            raise ValueError(f'{root.name}: coregistration incomplete; run coregister first')
        if not first_crs.equals(CRS.from_user_input(m['coreg']['geometry_crs'])):
            raise ValueError('Source geometry CRSs differ; reproject geometry before assembly')
        if not np.isfinite(m['interval_seconds']) or m['interval_seconds']<=0:
            raise ValueError(f'{root.name}: invalid pair interval')
        files = dict(m['coreg']['slcs']) | {f'geometry_{k}': v for k, v in m['coreg']['geometry'].items()}
        checksums = {}
        for name, path in files.items():
            ds = gdal.Open(str(path))
            if ds is None or ds.RasterCount != 1 or (ds.RasterYSize, ds.RasterXSize) != (g['length'], g['width']):
                raise ValueError(f'{root.name}: invalid {name} raster dimensions')
            if name in ('reference', 'secondary') and not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType):
                raise ValueError(f'{root.name}: {name} must be complex')
            checksums[name] = {'path': str(Path(path).resolve()), 'sha256': checksum(path)}
        entries.append({'burst_id': root.name, 'manifest': str((root/'manifest.json').resolve()),
                        'coreg_identity': m['coreg_identity'], 'coreg': deepcopy(m['coreg']),
                        'interval_seconds': m['interval_seconds'], 'acquisition_times': m['acquisition_times'],
                        **placement,
                        'annotation_valid_window': _annotation_window((native_pairs or {}).get(root.name)),
                        'files': checksums})
    return base, entries


def _coverage_bounds(path):
    """Read only strips to recover the outer support of an existing COMPASS SLC.

    These are bounds, not a claim that their interior is valid. Native matching
    still rejects holes, zero-fill components and nonfinite chip/search samples.
    """
    ds = gdal.Open(str(path))
    y0, y1, x0, x1 = ds.RasterYSize, 0, ds.RasterXSize, 0
    with bounded_raster_cache():
        for row in range(0, ds.RasterYSize, 64):
            data = ds.ReadAsArray(0, row, ds.RasterXSize, min(64, ds.RasterYSize-row))
            good = np.isfinite(data) & ((data.real != 0) | (data.imag != 0))
            rows, cols = np.flatnonzero(good.any(axis=1)), np.flatnonzero(good.any(axis=0))
            if rows.size:
                y0, y1 = min(y0, row+int(rows[0])), max(y1, row+int(rows[-1])+1)
                x0, x1 = min(x0, int(cols[0])), max(x1, int(cols[-1])+1)
    if y0 >= y1 or x0 >= x1:
        raise ValueError(f'No finite nonzero SLC support: {path}')
    return [y0, y1, x0, x1]


def common_valid_window(entry):
    boxes = [_coverage_bounds(entry['coreg']['slcs'][side]) for side in ('reference', 'secondary')]
    if entry['annotation_valid_window'] is not None:
        boxes.append(entry['annotation_valid_window'])
    result = [max(b[0] for b in boxes), min(b[1] for b in boxes),
              max(b[2] for b in boxes), min(b[3] for b in boxes)]
    if result[0] >= result[1] or result[2] >= result[3]:
        raise ValueError(f"{entry['burst_id']}: no shared reference/secondary support")
    return result


def write_vrt(path, shape, entries, field):
    """Spatial SimpleSource assembly, preserving ISCE2's later-source precedence."""
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


def invalidate(root, reason):
    path = Path(root)/'manifest.json'
    if path.is_file():
        m = json.loads(path.read_text())
        for key in ('coreg_identity', 'match_identity', 'geocode_identity'):
            m.pop(key, None)
        m['stages'] = {};m['assembly_status'] = reason
        write_json(path, m)


def assemble(cfg, sources, native_pairs=None, force=False, *, allow_released=False):
    """Create/cache a dedicated merged product without changing source bursts."""
    gdal.UseExceptions()
    root = Path(cfg['output']['directory'])/'merged'
    root.mkdir(parents=True, exist_ok=True)
    common, entries = plan_sources(sources, settings(cfg)['alignment_tolerance_px'], native_pairs)
    if root.resolve() in {source.resolve() for source, _ in sources}:
        raise ValueError('Merged destination cannot be a source burst')
    specs = {json.dumps(m['coreg_specification']['inputs'], sort_keys=True) for _, m in sources}
    polarizations = {m['coreg_specification']['polarization'] for _, m in sources}
    if len(specs) != 1 or polarizations != {cfg['inputs']['polarization']}:
        raise ValueError('Source bursts must share input provenance and polarization')
    for entry in entries:
        entry['valid_window'] = common_valid_window(entry)
        entry['valid_window_policy'] = ('annotation intersect COMPASS coverage bounds' if entry['annotation_valid_window'] is not None
                                        else 'existing COMPASS coverage bounds; original annotation not reloaded')
    # Cache scientific placement/ownership, not whether equivalent bounds came
    # from original annotation or an already-completed COMPASS snapshot.
    scientific_sources=[{key:value for key,value in entry.items() if key not in
                         ('annotation_valid_window','valid_window_policy')} for entry in entries]
    spec = {'method': METHOD, 'settings': settings(cfg), 'reference_grid': common, 'sources': deepcopy(scientific_sources)}
    fingerprint = identity(spec)
    path = root/'manifest.json'
    previous = json.loads(path.read_text()) if path.is_file() else {}
    from .storage import products_valid
    products = previous.get('assembly', {}).get('products', {})
    logical = products_valid(products, previous, 'coreg_identity', fingerprint, allow_released=True)
    if not force and previous.get('coreg_identity') == fingerprint:
        if products_valid(products, previous, 'coreg_identity', fingerprint, allow_released=allow_released):
            logging.info('Reusing verified merged SLC assembly')
            return root, previous
    invalidate(root, 'rebuilding')
    shape = common['length'], common['width']
    with tempfile.TemporaryDirectory(prefix='.assembly-', dir=root) as scratch:
        staging = Path(scratch)/'coreg';staging.mkdir()
        for field in ('reference', 'secondary', 'x', 'y', 'z', 'layover_shadow_mask', 'source'):
            write_vrt(staging/f'{field}.vrt', shape, entries, field)
        with bounded_raster_cache():
            for side in ('reference', 'secondary'):
                logging.info('Materializing merged %s SLC: %d x %d', side, *shape)
                ds = gdal.Translate(str(staging/f'{side}.slc.tif'), str(staging/f'{side}.vrt'),
                                    format='GTiff', creationOptions=['TILED=NO', 'BIGTIFF=IF_SAFER'])
                if ds is None or (ds.RasterYSize, ds.RasterXSize) != shape:
                    raise ValueError(f'Failed to materialize merged {side}')
                ds = None
        old = root/'coreg.previous'
        if old.exists():shutil.rmtree(old)
        if (root/'coreg').exists():(root/'coreg').rename(old)
        staging.rename(root/'coreg')
        if old.exists():shutil.rmtree(old)
    directory = root/'coreg'
    products = {str(p.resolve()): checksum(p) for p in sorted(directory.iterdir()) if p.is_file()}
    first = sources[0][1]
    coreg = {'slcs': {side: str((directory/f'{side}.slc.tif').resolve()) for side in ('reference', 'secondary')},
             'geometry': {name: str((directory/f'{name}.vrt').resolve()) for name in ('x', 'y', 'z', 'layover_shadow_mask')},
             'geometry_crs': entries[0]['coreg']['geometry_crs'], 'reference_grid': common,
             'source_index': str((directory/'source.vrt').resolve()), 'assembly_sources': entries}
    manifest = {'coreg_identity': fingerprint, 'coreg': coreg,
                'coreg_specification': {'inputs': first['coreg_specification']['inputs'],
                                        'polarization': cfg['inputs']['polarization'], 'burst_id': 'merged', 'assembly': fingerprint},
                'software': first['software'], 'interval_seconds': None,
                'acquisition_times': None, 'stages': {'coregister': 'assembled from complete COMPASS bursts'},
                'assembly_status': 'complete',
                'assembly': {'identity': fingerprint, 'method': METHOD, 'specification': spec,
                             'sources': entries, 'products': products,
                             'source_burst_ids': [e['burst_id'] for e in entries],
                             'interval_policy': 'per matching center, from its source burst; seam crossings flagged'}}
    if logical and not force:
        # Reconstituting a deliberately released cache does not change science.
        stages = previous.get('stages', {}) | manifest['stages']
        manifest = previous | manifest
        manifest['stages'] = stages
        manifest.pop('slc_cleanup', None)
        manifest.pop('slc_previews', None)
    write_json(path, manifest)
    write_json(root/'assembly.json', manifest['assembly'])
    return root, manifest


def release_merged(cfg, root, manifest):
    from .storage import cache_preview, keep_merged_slcs, release_slcs
    if keep_merged_slcs(cfg):
        return
    saved = manifest.setdefault('slc_previews', {})
    for side, source in manifest['coreg']['slcs'].items():
        if Path(source).is_file():
            saved[side] = cache_preview(Path(root)/'previews'/side, source, manifest['coreg_identity'])
    release_slcs(root, manifest, 'coreg_identity', manifest['coreg']['slcs'].values(),
                 manifest['assembly']['products'])


def source_indices(coreg, line, sample):
    from .geocode import sample_geometry
    return sample_geometry(coreg['source_index'], line, sample, order=0).astype('uint16')


def center_geometry(coreg, line, sample):
    """Use the owning burst's original geometry/scale, including at splice edges."""
    from .geocode import center_geometry as source_geometry
    line, sample = np.broadcast_arrays(line, sample)
    owner = source_indices(coreg, line, sample)
    outputs = [np.full(line.shape, np.nan) for _ in range(4)] + [np.zeros(line.shape, bool)]
    for index, entry in enumerate(coreg['assembly_sources'], 1):
        selected = owner == index
        if not selected.any():continue
        dy, dx = entry['offset']
        values = source_geometry(entry['coreg'], line[selected]-dy, sample[selected]-dx)
        for result, value in zip(outputs, values):result[selected] = value
    return tuple(outputs)


def prepare_prior(cfg, manifest, grid, native_pairs=None):
    from .velocity_prior import prepare_prior as source_prior
    from .matching_grid import MatchingGrid
    owner = source_indices(manifest['coreg'], *grid.centers)
    result = None; metadata = []
    for index, entry in enumerate(manifest['assembly']['sources'], 1):
        selected = owner == index
        if not selected.any():continue
        child = deepcopy(cfg);child.pop('aoi', None);child['inputs']['burst_id'] = entry['burst_id']
        dy, dx = entry['offset']
        local = MatchingGrid(grid.starts_line-dy, grid.starts_sample-dx, grid.window, grid.search, grid.stride)
        source = json.loads(Path(entry['manifest']).read_text())
        prior = source_prior(child, source, local, native_pair=(native_pairs or {}).get(entry['burst_id']))
        if result is None:
            result = {name: np.full_like(array, False if array.dtype == bool else np.iinfo(np.int32).min if name == 'integer_px'
                                         else 1 if name == 'rejection_flags' else np.nan)
                      for name, array in prior.items() if isinstance(array, np.ndarray)}
        for name in result:result[name][selected] = prior[name][selected]
        metadata.append({'burst_id': entry['burst_id'], 'metadata': prior['metadata']})
    if result is None:raise ValueError('Merged matching grid has no source centers')
    result['metadata'] = {'settings': cfg['gross_offsets']['velocity'], 'sources': metadata,
                          'interval_policy': 'each source uses its own exact interval and original orbit/grid',
                          'missing_policy': 'mask; no zero fallback'}
    return result


def annotate_result(manifest, result):
    """Attach categorical source ownership, exact intervals and full-chip seam flags."""
    coreg = manifest['coreg'];grid = result['grid']
    owner = source_indices(coreg, *grid.centers)
    intervals = np.full(grid.shape, np.nan)
    for index, entry in enumerate(manifest['assembly']['sources'], 1):
        intervals[owner == index] = entry['interval_seconds']
    result['source_burst_index'] = owner
    result['interval_seconds'] = intervals
    result['input_valid'] &= owner > 0
    result['valid'] &= owner > 0
    if 'stack_pair' in manifest:
        # A burst absent from either date has no exact interval or observation.
        result['input_valid'] &= np.isfinite(intervals) & (intervals > 0)
        result['valid'] &= np.isfinite(intervals) & (intervals > 0)
    ds = gdal.Open(coreg['source_index'])
    gross = result['gross']
    known=np.isfinite(gross).all(axis=-1)&(owner>0)
    result['seam_search_known']=known
    crossing = np.zeros(grid.shape, bool)
    with bounded_raster_cache():
        for row, y in enumerate(grid.starts_line):
            # One narrow full-width strip per sparse row; no full source-index raster in RAM.
            eligible = owner[row] > 0
            if not eligible.any():continue
            gy = gross[row, known[row], 0].astype(int) if known[row].any() else np.array([0])
            lo = max(0, min(int(y), int(y)+int(gy.min())-grid.search[0]))
            hi = min(ds.RasterYSize, max(int(y)+grid.window[0], int(y)+int(gy.max())+grid.window[0]+grid.search[0]))
            # Large/variable priors must not turn an annotation pass into a
            # full-image read. Fall back to bounded per-search reads if needed.
            if hi-lo>512:lo,hi=int(y),int(y)+grid.window[0]
            block = ds.ReadAsArray(0, lo, ds.RasterXSize, hi-lo)
            for col in np.flatnonzero(eligible):
                x = int(grid.starts_sample[col])
                boxes = [(int(y), x, *grid.window)]
                if known[row,col]:
                    sy=int(y)+int(gross[row,col,0])-grid.search[0]
                    sx=x+int(gross[row,col,1])-grid.search[1]
                    boxes.append((sy,sx,grid.window[0]+2*grid.search[0],grid.window[1]+2*grid.search[1]))
                pure = True
                for yy, xx, h, w in boxes:
                    if yy<0 or yy+h>ds.RasterYSize or xx<0 or xx+w>ds.RasterXSize:
                        pure = False;break
                    data=(block[yy-lo:yy-lo+h,xx:xx+w] if lo<=yy and yy+h<=hi else ds.ReadAsArray(xx,yy,w,h))
                    pure &= bool(np.all(data == owner[row,col]))
                crossing[row,col] = not pure
    result['seam_crossing'] = crossing & (owner > 0)
