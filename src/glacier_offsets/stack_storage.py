"""Reusable stack geometry and bounded, fixed-ownership date assembly."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import numpy as np
from osgeo import gdal
from pyproj import CRS

from .assembly import _annotation_window, _coverage_bounds, radar_layout, write_vrt
from .compass_coreg import align_acquisition, prepare_reference
from .matching_grid import invalid_samples
from .products import checksum, identity, write_json
from .raster_io import bounded_raster_cache
from .stack_ownership import POLICY, interior_ownership
from .storage import cache_preview, keep_merged_slcs, products_valid, release_slcs

METHOD = 'stack-reference-interior-ownership-v3'
SLC_STORAGE = 'stripped-complex-tiff-external-mask-v1'


class SourceHashes:
    """Hash each distinct immutable input only once during an invocation."""
    def __init__(self):
        self.hashes = {}

    def __call__(self, path):
        path = str(Path(path).resolve())
        if path not in self.hashes:
            self.hashes[path] = checksum(path)
        return {'path': path, 'sha256': self.hashes[path]}


def read_manifest(root):
    path = Path(root) / 'manifest.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def verified(manifest, key, fingerprint, *, allow_released=False, checksums=None):
    products = manifest.get('products', {})
    return products_valid(products, manifest, key, fingerprint, allow_released=allow_released, checksums=checksums)


def alignment_verified(manifest, fingerprint, *, checksums=None):
    products = manifest.get('alignment_products', manifest.get('products', {}))
    return verified(manifest | {'products': products}, 'alignment_identity', fingerprint, checksums=checksums)


def product_hashes(paths, *, checksums=None):
    digest = checksum if checksums is None else checksums
    return {str(Path(p).resolve()): digest(p) for p in dict.fromkeys(map(str, paths))}


def burst_config(cfg, reference, acquisition):
    result = deepcopy(cfg)
    result.pop('aoi', None)
    result['inputs'].update(reference_safe=str(reference.scene.path), reference_orbit=str(reference.orbit.path),
                            secondary_safe=str(acquisition.scene.path), secondary_orbit=str(acquisition.orbit.path))
    return result


def reference_specification(cfg, plan, hashes, software):
    return {'method': METHOD, 'ownership_policy': POLICY, 'anchor': plan.anchor,
            'inputs': {key: hashes(path) for key, path in
                       [('safe', plan.reference.scene.path), ('orbit', plan.reference.orbit.path), ('dem', cfg['inputs']['dem'])]},
            'software': software['packages'], 'flatten': False, 'backend': 'cpu'}


def prepare_geometry(cfg, plan, specification, software, force=False, process=True, expected_layout=None, checksums=None):
    root = Path(cfg['output']['directory']) / 'geometry'
    fingerprint = identity(specification)
    previous = read_manifest(root)
    if not force and verified(previous, 'geometry_identity', fingerprint, checksums=checksums):
        return previous
    if not process:
        raise ValueError('Reference preparation missing, incomplete or damaged; run stack coregister first')
    root.mkdir(parents=True, exist_ok=True)
    # Invalidate before touching any shared reference product.
    write_json(root / 'manifest.json', {'specification': specification, 'status': 'running'})
    sources = []
    effective = burst_config(cfg, plan.reference, plan.reference)
    for burst in plan.bursts:
        directory = root / 'bursts' / burst.id
        spec = {'reference': fingerprint, 'burst_id': burst.id}
        old = read_manifest(directory)
        if not force and verified(old, 'identity', identity(spec), checksums=checksums):
            coreg = old['coreg']
        else:
            write_json(directory / 'manifest.json', {'specification': spec, 'status': 'running'})
            coreg = prepare_reference(effective, burst, directory)
            files = [*coreg['geometry'].values(), coreg['slcs']['reference']]
            if 'reference_directory' in coreg:
                files.extend(p for p in Path(coreg['reference_directory']).iterdir() if p.is_file())
            write_json(directory / 'manifest.json', {'identity': identity(spec), 'coreg': coreg,
                       'products': product_hashes(files, checksums=checksums), 'status': 'complete'})
        window = _coverage_bounds(coreg['slcs']['reference'])
        annotation = _annotation_window((burst, burst))
        if annotation:
            window = [max(window[0], annotation[0]), min(window[1], annotation[1]),
                      max(window[2], annotation[2]), min(window[3], annotation[3])]
        if window[0] >= window[1] or window[2] >= window[3]:
            raise ValueError(f'{burst.id}: reference has no annotation-valid SLC support')
        sources.append({'burst_id': burst.id, 'coreg': coreg, 'valid_window': window,
                        'geometry_reference_time': burst.start.isoformat(),
                        'manifest': str(directory / 'manifest.json')})
    if len({CRS.from_user_input(e['coreg']['geometry_crs']).to_wkt() for e in sources}) != 1:
        raise ValueError('Reference burst geometry CRSs differ')
    layout = write_reference_layout(root, sources, checksums=checksums)
    layout_identity = layout['layout_identity']
    pinned_layout = expected_layout or previous.get('layout_identity')
    if pinned_layout is not None and layout_identity != pinned_layout:
        raise ValueError('Pinned radar layout/ownership changed; use a new output root')
    files = [layout['coreg']['source_index'], *layout['coreg']['geometry'].values()]
    for entry in sources:
        files.extend(read_manifest(Path(entry['manifest']).parent)['products'])
    frame = dict(layout, geometry_identity=fingerprint, specification=specification,
                 software=software, status='complete', products=product_hashes(files, checksums=checksums))
    write_json(root / 'manifest.json', frame)
    return frame


def write_reference_layout(root, sources, *, checksums=None):
    """Write the interior ownership and matching geometry from prepared sources."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = deepcopy(sources)
    grid, placements = radar_layout([e['coreg']['reference_grid'] for e in sources], .001)
    for entry, placement in zip(sources, placements):
        entry.update(placement)
    sources, ownership = interior_ownership(sources)
    shape = grid['length'], grid['width']
    # Geometry VRTs retain original burst sources; ownership is a materialized
    # categorical raster using only actual reference support, including holes.
    for field in ('x', 'y', 'z', 'layover_shadow_mask'):
        write_vrt(root / f'{field}.vrt', shape, sources, field)
    owner_path = root / 'source.tif'
    owner = create_raster(owner_path, shape, gdal.GDT_UInt16)
    owner.GetRasterBand(1).Fill(0)
    with bounded_raster_cache():
        for index, entry in enumerate(sources, 1):
            source = gdal.Open(entry['coreg']['slcs']['reference'])
            unsupported = invalid_samples(source)
            dy, dx = entry['offset']
            for y0, y1, x0, x1 in entry['ownership_windows']:
                for row in range(y0, y1, 64):
                    h = min(64, y1-row)
                    valid = ~unsupported[row:row+h, x0:x1]
                    band = owner.GetRasterBand(1)
                    block = band.ReadAsArray(x0+dx, row+dy, x1-x0, h)
                    block[valid] = index
                    band.WriteArray(block, x0+dx, row+dy)
            del unsupported
    owner = None
    digest = checksum if checksums is None else checksums
    layout_identity = identity({'reference_grid': grid, 'source_index_sha256': digest(owner_path),
                               'ownership': ownership,
                               'placements': [{k: e[k] for k in ('burst_id', 'offset', 'valid_window', 'ownership_windows')}
                                              for e in sources]})
    coreg = {'reference_grid': grid, 'source_index': str(owner_path.resolve()),
             'geometry': {field: str((root / f'{field}.vrt').resolve()) for field in ('x', 'y', 'z', 'layover_shadow_mask')},
             'geometry_crs': sources[0]['coreg']['geometry_crs'], 'assembly_sources': sources}
    return {'layout_identity': layout_identity, 'coreg': coreg, 'sources': sources, 'ownership': ownership}


def create_raster(path, shape, dtype):
    return gdal.GetDriverByName('GTiff').Create(str(path), shape[1], shape[0], 1, dtype,
                                              options=['TILED=NO', 'BIGTIFF=IF_SAFER'])


def date_specification(acquisition, frame, hashes):
    if acquisition.error:
        raise ValueError(acquisition.error)
    return {'method': METHOD, 'geometry': frame['geometry_identity'],
            'safe': hashes(acquisition.scene.path), 'orbit': hashes(acquisition.orbit.path),
            'burst_times': {e['burst_id']: (acquisition.bursts[e['burst_id']].start.isoformat()
                                          if acquisition.bursts[e['burst_id']].start else None)
                            for e in frame['sources'] if e['burst_id'] in acquisition.bursts}}


def prepare_date(cfg, plan, acquisition, frame, specification, force=False, process=True, checksums=None):
    """Align available bursts once; cache each burst independently of pair count."""
    root = Path(cfg['output']['directory']) / 'acquisitions' / acquisition.date
    fingerprint = identity(specification)
    old = read_manifest(root)
    if not force and alignment_verified(old, fingerprint, checksums=checksums):
        return old
    if not process:
        raise ValueError(f'{acquisition.date}: alignment missing or stale; run stack coregister first')
    write_json(root / 'manifest.json', {'specification': specification, 'status': 'aligning'})
    effective = burst_config(cfg, plan.reference, acquisition)
    slcs = {}
    reference_bursts = {b.id: b for b in plan.bursts}
    for entry in frame['sources']:
        key = entry['burst_id']
        if key not in acquisition.bursts or acquisition.bursts[key].start is None:
            continue
        if acquisition.date == plan.reference.date:
            slcs[key] = entry['coreg']['slcs']['reference']
            continue
        directory = root / 'bursts' / key
        spec = {'date': fingerprint, 'burst_id': key}
        previous = read_manifest(directory)
        if not force and verified(previous, 'identity', identity(spec), checksums=checksums):
            slcs[key] = previous['slc']
        else:
            write_json(directory / 'manifest.json', {'specification': spec, 'status': 'aligning'})
            slc = align_acquisition(effective, reference_bursts[key], acquisition.bursts[key], directory, entry['coreg'])
            slcs[key] = slc
            write_json(directory / 'manifest.json', {'identity': identity(spec), 'slc': slc,
                       'products': product_hashes([slc], checksums=checksums), 'status': 'complete'})
    result = {'alignment_identity': fingerprint, 'specification': specification, 'slcs': slcs,
              'products': product_hashes(slcs.values(), checksums=checksums), 'status': 'aligned', 'date': acquisition.date,
              'burst_times': specification['burst_times']}
    # An acquisition with no selected bursts is an explicitly empty aligned date.
    if not slcs:
        result['products'] = product_hashes([Path(cfg['output']['directory']) / 'geometry/source.tif'], checksums=checksums)
    result['alignment_products'] = result['products']
    write_json(root / 'manifest.json', result)
    return result


def assembly_specification(frame, aligned):
    """Version date storage independently of the pinned COMPASS reference frame."""
    return {'method': METHOD, 'alignment': aligned['alignment_identity'],
            'geometry': frame['geometry_identity'], 'slc_storage': SLC_STORAGE}


def assemble_date(cfg, acquisition, frame, aligned, force=False, process=True, checksums=None):
    """Apply identical reference ownership; never fill gaps from another source."""
    root = Path(cfg['output']['directory']) / 'acquisitions' / acquisition.date
    fingerprint = identity(assembly_specification(frame, aligned))
    if not force and verified(aligned, 'assembly_identity', fingerprint, checksums=checksums):
        return aligned
    if not process:
        raise ValueError(f'{acquisition.date}: merged SLC missing or stale; run stack assemble first')
    result = deepcopy(aligned)
    result.pop('assembly_pending', None)
    result.pop('slc_cleanup', None)
    result.pop('slc_preview', None)
    result.pop('assembly_identity', None)
    result['status'] = 'assembling'
    write_json(root / 'manifest.json', result)
    shape = tuple(frame['coreg']['reference_grid'][k] for k in ('length', 'width'))
    owner = gdal.Open(frame['coreg']['source_index'])
    count = 0
    with tempfile.TemporaryDirectory(prefix='.merge-', dir=root) as scratch, bounded_raster_cache():
        slc_path, mask_path = Path(scratch)/'slc.tif', Path(scratch)/'valid.tif'
        slc = create_raster(slc_path, shape, gdal.GDT_CFloat32)
        mask = create_raster(mask_path, shape, gdal.GDT_Byte)
        # Internal mask blocks can interleave with SLC strips after cache flushes.
        # The CPU matcher always memory-maps its input, even with useMmap=0;
        # macOS requires contiguous image strips instead of GDAL's VM fallback.
        with gdal.config_option('GDAL_TIFF_INTERNAL_MASK', 'NO'):
            slc.CreateMaskBand(gdal.GMF_PER_DATASET)
        datasets = {key: gdal.Open(path) for key, path in aligned['slcs'].items()}
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
        final = root / 'slc.tif'
        slc_path.replace(final)
        # External masks are created by GDAL unless internal masks are enabled.
        ancillary = Path(str(slc_path) + '.msk')
        if ancillary.exists():
            ancillary.replace(Path(str(final) + '.msk'))
        elif Path(str(final) + '.msk').exists():
            Path(str(final) + '.msk').unlink()
        mask_path.replace(root / 'valid.tif')
    files = [*aligned['slcs'].values(), root/'slc.tif', root/'valid.tif']
    if Path(str(root/'slc.tif') + '.msk').exists():
        files.append(Path(str(root/'slc.tif') + '.msk'))
    result.update(assembly_identity=fingerprint, slc_storage=SLC_STORAGE,
                  status='complete', slc=str((root/'slc.tif').resolve()),
                  validity_mask=str((root/'valid.tif').resolve()), valid_samples=count,
                  coverage_fraction=count/int(np.prod(shape)), products=product_hashes(files, checksums=checksums))
    write_json(root / 'manifest.json', result)
    return result


def release_date(cfg, acquisition, manifest):
    """Release only the shared assembled TIFF/mask, retaining aligned bursts."""
    if keep_merged_slcs(cfg) or not manifest.get('assembly_identity'):
        return
    root = Path(cfg['output']['directory'])/'acquisitions'/acquisition.date
    source = Path(manifest['slc'])
    if source.is_file():
        manifest['slc_preview'] = cache_preview(root, source, manifest['assembly_identity'])
    release_slcs(root, manifest, 'assembly_identity', [source], manifest['products'])
