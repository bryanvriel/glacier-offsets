"""Sequential AOI dispatch using the existing independent burst stage engine."""
from copy import deepcopy
from dataclasses import dataclass
import gc
import json
import logging
from pathlib import Path

from .aoi import footprint, polygon, project_footprint, select_aoi, wgs84_parts
from .environment import inspect_environment
from .inputs import select_pair
from .products import input_provenance, write_json


@dataclass(frozen=True)
class InvocationContext:
    pairs: tuple
    provenance_json: str
    software_json: str
    inventory_json: str = '{}'

    @property
    def inventory(self):
        return json.loads(self.inventory_json)

    @property
    def provenance(self):
        return json.loads(self.provenance_json)

    @property
    def software(self):
        return json.loads(self.software_json)

    def pair(self, burst_id):
        return next(p for p in self.pairs if p[0].id == burst_id)


def effective_config(cfg, burst_id):
    effective = deepcopy(cfg)
    effective.pop('aoi', None)
    effective['inputs']['burst_id'] = burst_id
    return effective


def dem_extent(cfg):
    from osgeo import gdal
    from pyproj import CRS
    from shapely.geometry import Polygon
    gdal.UseExceptions()
    ds = gdal.Open(cfg['inputs']['dem'])
    if ds is None or not ds.GetProjection():
        raise ValueError('DEM must declare a CRS and ellipsoidal heights')
    gt = ds.GetGeoTransform()
    def xy(c, r):
        return gt[0] + c * gt[1] + r * gt[2], gt[3] + c * gt[4] + r * gt[5]
    extent = Polygon([xy(c, r) for c, r in [(0, 0), (ds.RasterXSize, 0),
                     (ds.RasterXSize, ds.RasterYSize), (0, ds.RasterYSize)]])
    polygon(extent, 'DEM extent')
    crs = CRS.from_wkt(ds.GetProjection()).to_2d()
    if crs.is_geographic:
        # Raster extents describe their continuous coordinate interval. Densify
        # before wrapping so a wide raster is not mistaken for a narrow crossing.
        extent = wgs84_parts(extent.segmentize(0.01), crs)
        crs = CRS.from_epsg(4326)
    return extent, crs


def validate_burst(cfg, pair, extent):
    select_pair(cfg, [pair])  # Burst-specific polarization/direction/sampling.
    geometry, crs = extent
    reference = project_footprint(footprint(pair[0]), crs)
    if not geometry.covers(reference):
        raise ValueError(f'DEM extent does not cover complete reference footprint for {pair[0].id}; provide a larger DEM')


def validate_batch(cfg):
    pairs, _ = select_aoi(cfg)
    if not pairs:
        raise ValueError('AOI selects no common bursts')
    extent = dem_extent(cfg)
    errors = []
    for pair in pairs:
        try:
            validate_burst(effective_config(cfg, pair[0].id), pair, extent)
        except (ValueError, RuntimeError) as exc:
            errors.append(f'{pair[0].id}: {exc}')
    if errors:
        raise ValueError('; '.join(errors))
    from .assembly import settings,validate_native_layout
    if settings(cfg)['enabled']:validate_native_layout(pairs,cfg)
    return pairs


def product_summary(root):
    path = root / 'manifest.json'
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {'manifest_error': str(exc)}
    if not isinstance(manifest, dict):
        return {'manifest_error': 'Expected manifest mapping'}
    match = manifest.get('matching_summary', {})
    geo = manifest.get('geocoding', {})
    if not isinstance(match, dict) or not isinstance(geo, dict):
        return {'manifest_error': 'Invalid matching/map coverage summary'}
    return {'stages': manifest.get('stages', {}), 'matching_coverage': match,
            'map_coverage': geo,
            'scientific_coverage': ('empty' if match.get('quality_valid') == 0 or
                                    geo.get('valid_output_cells') == 0 else
                                    'observations' if match else 'not_yet_matched')}


def execute_batch(cfg, command, force=False):
    from .assembly import settings
    if settings(cfg)['enabled'] and command != 'coregister':
        from .assembled_workflow import execute_assembled_batch
        return execute_assembled_batch(cfg,command,force=force)
    if command=='assemble':raise ValueError('assemble requires burst_merge.enabled: true')
    from .pipeline import execute
    pairs, selection = select_aoi(cfg)
    root = Path(cfg['output']['directory'])
    # An empty selection is recorded before returning preflight status 2, without
    # hashing imagery or initializing native processing software.
    write_json(root / 'selection.json', selection)
    entries = {p[0].id: {'status': 'pending', 'products': {
        k: str(root / p[0].id / k) for k in ('manifest.json', 'coreg', 'radar', 'geo', 'diagnostics')}}
               for p in pairs}
    report = {'command': command, 'force': force, 'selection_identity': selection['selection_identity'],
              'selected_ids': selection['selected_ids'], 'bursts': entries, 'outcome': 'running'}
    def save():
        write_json(root / 'batch_manifest.json', report)
    save()
    if not pairs:
        report['outcome'] = 'empty_selection'
        save()
        return 2
    try:
        extent = dem_extent(cfg)
        context = InvocationContext(pairs, json.dumps(input_provenance(cfg)), json.dumps(inspect_environment()),
                                    json.dumps(selection))
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        save()
        raise
    except Exception as exc:
        report.update(outcome='preflight_failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise
    failed = False
    for pair in pairs:
        burst_id = pair[0].id
        entry = entries[burst_id]
        entry['status'] = 'running'
        logging.info('Processing %s: %s', burst_id, command)
        save()
        try:
            effective = effective_config(cfg, burst_id)
            validate_burst(effective, pair, extent)
            execute(effective, command, force=force, context=context)
            entry.update(product_summary(root / burst_id))
            entry['status'] = 'complete'
        except KeyboardInterrupt:
            entry['status'] = 'interrupted'
            report['outcome'] = 'interrupted'
            save()
            raise
        except Exception as exc:
            failed = True
            entry.update(product_summary(root / burst_id))
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            logging.error('%s: %s', burst_id, exc)
        finally:
            save()
            gc.collect()  # Release per-burst Python/native stage objects before next burst.
    report['outcome'] = 'failed' if failed else 'complete'
    save()
    return 1 if failed else 0
