"""COMPASS radar mode adapter, without motion compensation or NESD."""
import json
import logging
from pathlib import Path
import subprocess
import sys
import numpy as np
import yaml
from .inputs import select_pair
from .products import identity, input_provenance, write_json
from .environment import inspect_environment


def make_runconfig(cfg, pair, side, root, ref_directory=None):
    """Use installed defaults and isolate exactly the configured orbit file."""
    import compass
    defaults = Path(compass.__file__).parent / 'defaults/s1_cslc_radar.yaml'
    document = yaml.safe_load(defaults.read_text())
    g = document['runconfig']['groups']
    orbit = Path(cfg['inputs'][f'{side}_orbit'])
    orbit_dir = root / 'orbits' / side
    orbit_dir.mkdir(parents=True, exist_ok=True)
    link = orbit_dir / orbit.name
    if link.is_symlink():
        link.unlink()
    link.symlink_to(orbit)
    g['input_file_group'].update(safe_file_path=[cfg['inputs'][f'{side}_safe']],
                                orbit_file_path=[str(link)], burst_id=[pair[0].id],
                                reference_burst={'is_reference': side == 'reference',
                                                 'file_path': str(ref_directory) if ref_directory else None})
    g['dynamic_ancillary_file_group']['dem_file'] = cfg['inputs']['dem']
    paths = g['product_path_group']
    paths.update(product_path=str(root / 'coreg' / side), scratch_path=str(root / 'scratch' / side),
                 sas_output_file=str(root / 'coreg' / side))
    g['processing']['polarization'] = 'co-pol' if cfg['inputs']['polarization'] in {'HH', 'VV'} else 'cross-pol'
    g['processing']['resample']['flatten'] = False
    # Upstream default 0.028 conflicts with int-only schema; let its loader merge this unused radar default.
    g['processing'].pop('correction_luts', None)
    g['worker'].update(internet_access=False, gpu_enabled=False)
    return document


def inspect_pair(reference, secondary, reference_directory, allow_empty=False):
    """Verify complex dtype, shared reference dimensions and finite valid coverage."""
    from osgeo import gdal
    from compass.utils.radar_grid import file_to_rdr_grid
    gdal.UseExceptions()
    grid = file_to_rdr_grid(str(Path(reference_directory) / 'radar_grid.txt'))
    coverages = {}
    for label, path in [('reference', reference), ('secondary', secondary)]:
        ds = gdal.Open(str(path))
        if ds.RasterCount != 1 or ds.GetRasterBand(1).DataType not in (gdal.GDT_CFloat32, gdal.GDT_CFloat64, gdal.GDT_CInt16, gdal.GDT_CInt32):
            raise ValueError(f'{label} must be a single-band complex SLC')
        if (ds.RasterYSize, ds.RasterXSize) != (grid.length, grid.width):
            raise ValueError(f'{label} does not match reference radar grid dimensions')
        valid = 0
        for line in range(0, ds.RasterYSize, 256):
            a = ds.ReadAsArray(0, line, ds.RasterXSize, min(256, ds.RasterYSize-line))
            valid += np.count_nonzero(np.isfinite(a) & (np.abs(a) > 0))
        coverages[label] = valid / (grid.length * grid.width)
        if not valid and not allow_empty:
            raise ValueError(f'{label} contains no valid complex samples')
    geometry = {}
    for name in ('x', 'y', 'z', 'layover_shadow_mask'):
        path = Path(reference_directory) / f'{name}.tif'
        ds = gdal.Open(str(path))
        if (ds.RasterYSize, ds.RasterXSize) != (grid.length, grid.width):
            raise ValueError(f'Geometry {name} shape differs from reference SLC')
        geometry[name] = str(path)
    from osgeo import osr
    import isce3
    geometry_epsg = isce3.io.Raster(str(Path(reference_directory) / 'topo.vrt')).get_epsg()
    crs = osr.SpatialReference(); crs.ImportFromEPSG(geometry_epsg)
    grid_metadata = {k: getattr(grid,k) for k in ('sensing_start','wavelength','prf','starting_range','range_pixel_spacing','length','width')}
    grid_metadata.update(ref_epoch=str(grid.ref_epoch), lookside='right')
    return {'slcs': {'reference': str(reference), 'secondary': str(secondary)},
            'reference_directory': str(reference_directory), 'reference_grid': grid_metadata,
            'coverage': coverages, 'geometry': geometry, 'geometry_crs': crs.ExportToWkt(), 'geometry_epsg': geometry_epsg,
            'alignment_validation': 'pending stationary-terrain inspection; dimensions alone do not establish accuracy'}


def run_compass(cfg, pair, side, root, reference_directory=None):
    """Run one isolated COMPASS stage; also used by reusable stack preparation."""
    root.mkdir(parents=True, exist_ok=True)
    document = make_runconfig(cfg, pair, side, root, reference_directory)
    yaml_path = root / f'compass_{side}.yaml'
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False))
    command = [sys.executable, '-m', 'glacier_offsets.compass_worker', str(yaml_path)]
    logging.info('Running %s', ' '.join(command))
    with (root / f'compass_{side}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def prepare_reference(cfg, burst, root):
    """Generate topology and reference SLC exactly once for a stack burst."""
    run_compass(cfg, (burst, burst), 'reference', root)
    from .package_geometry import retain_reference_orbit
    from .products import checksum
    orbit = cfg['inputs']['reference_orbit']
    retain_reference_orbit(Path(cfg['output']['directory']) / 'geometry',
                           {'path': orbit, 'sha256': checksum(orbit)})
    directory = root / 'coreg/reference' / burst.id / burst.start.strftime('%Y%m%d')
    slc = directory / f'{burst.id}_{burst.start:%Y%m%d}_{cfg["inputs"]["polarization"]}.slc.tif'
    return inspect_pair(slc, slc, directory)


def align_acquisition(cfg, reference_burst, burst, root, reference_coreg):
    """Resample an original acquisition directly onto the pinned reference grid.

    Sensing order is intentionally unrestricted: dates preceding an explicitly
    chosen geometry reference use this same geometry-only COMPASS operation.
    """
    run_compass(cfg, (reference_burst, burst), 'secondary', root,
                Path(reference_coreg['reference_directory']))
    slc = root / 'coreg/secondary' / burst.id / burst.start.strftime('%Y%m%d') / f'{burst.id}_{burst.start:%Y%m%d}.slc.tif'
    product = inspect_pair(reference_coreg['slcs']['reference'], slc,
                           reference_coreg['reference_directory'], allow_empty=True)
    return product['slcs']['secondary']


def coregister(cfg: dict, force: bool = False, context=None) -> tuple[Path, dict]:
    pair = context.pair(cfg['inputs']['burst_id']) if context else select_pair(cfg)
    root = Path(cfg['output']['directory']) / pair[0].id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    provenance = context.provenance if context else input_provenance(cfg)
    env = context.software if context else inspect_environment()
    spec = {'inputs': provenance, 'burst_id': pair[0].id, 'polarization': cfg['inputs']['polarization'],
            'flatten': False, 'backend': 'cpu', 'software': env['packages']}
    fingerprint = identity(spec)
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if not force and previous.get('coreg_identity') == fingerprint:
        product = previous['coreg']
        inspect_pair(product['slcs']['reference'], product['slcs']['secondary'], product['reference_directory'])
        logging.info('Reusing verified COMPASS coregistration')
        return root, previous
    ref_directory = root / 'coreg' / 'reference' / pair[0].id / pair[0].start.strftime('%Y%m%d')
    manifest = {'coreg_identity': None, 'coreg_specification': spec, 'software': env,
                'acquisition_times': [b.start.isoformat() for b in pair],
                'interval_seconds': (pair[1].start - pair[0].start).total_seconds(), 'stages': {}}
    write_json(manifest_path, manifest)  # invalidate downstream stages before mutation
    for side in ('reference', 'secondary'):
        run_compass(cfg, pair, side, root, ref_directory if side == 'secondary' else None)
    ref = ref_directory / f'{pair[0].id}_{pair[0].start:%Y%m%d}_{cfg["inputs"]["polarization"]}.slc.tif'
    sec = root / 'coreg/secondary' / pair[1].id / pair[1].start.strftime('%Y%m%d') / f'{pair[1].id}_{pair[1].start:%Y%m%d}.slc.tif'
    manifest['coreg'] = inspect_pair(ref, sec, ref_directory)
    manifest['coreg_identity'] = fingerprint
    manifest['stages']['coregister'] = 'complete; stationary-ground accuracy not independently established'
    write_json(manifest_path, manifest)
    return root, manifest
