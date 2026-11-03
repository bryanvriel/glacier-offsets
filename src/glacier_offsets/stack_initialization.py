"""Initialize a stack YAML from local SAFE/orbit directories and a burst selector."""
import argparse
import os
from pathlib import Path
import sys
import yaml

from .config import PATH_FIELDS, load_defaults, merge_config
from .stack import plan_stack
from .stack_config import STACK_DEFAULTS, validate_stack_config, verification_workers
from .swath_correction_config import settings as correction_settings
from .paths import absolute_path


CONFIG_HEADER = '''# Complete stack settings, ready for stack plan, validate and run.
# Matching arrays are [azimuth, range]. Null values retain automatic behavior.
# Production matching: ISCE3 CPU, deramp 0, complex range oversampling 2x.
# Runtime directories are derived from output.directory:
#   geometry/, acquisitions/<date>/, pairs/<earlier>_<later>/, packages/
# storage.scratch_directory: null keeps temporary matching inputs with the pair.
# Null matching.isce2 runtime paths use the current interpreter/environment.
'''


def make_stack_config(safe_directory, orbit_directory, dem, output_directory, polarization,
                      burst_id=None, aoi=None, bbox=None, aoi_layer=None, epsg=None,
                      spacing_m=None, start_date=None, end_date=None, reference_date=None,
                      neighbors=1, max_separation_days=12, pairs_file=None, validate_dem_coverage=False):
    if sum(v is not None for v in (burst_id, aoi, bbox)) != 1:
        raise ValueError('Choose exactly one of burst_id, aoi or bbox')
    if aoi_layer is not None and aoi is None:
        raise ValueError('aoi_layer requires aoi')
    from osgeo import gdal
    from pyproj import CRS
    dem = absolute_path(dem, strict=True)
    gdal.UseExceptions()
    ds = gdal.Open(str(dem))
    if not ds.GetProjection():
        raise ValueError('DEM must declare a CRS and ellipsoidal heights')
    if epsg is None:
        epsg = CRS.from_wkt(ds.GetProjection()).to_epsg()
    cfg = merge_config(load_defaults(), {
        'stack': merge_config(STACK_DEFAULTS, {
            'safe_directory': str(absolute_path(safe_directory)),
            'orbit_directory': str(absolute_path(orbit_directory)),
            'start_date': start_date, 'end_date': end_date, 'reference_date': reference_date,
            'pairing': {'neighbors': neighbors, 'max_separation_days': max_separation_days,
                        'pairs_file': str(absolute_path(pairs_file)) if pairs_file else None}}),
        'inputs': {'dem': str(dem), 'polarization': polarization, 'burst_id': burst_id},
        'matching': {'engine': 'isce3', 'backend': 'cpu', 'deramp': 0,
                     'range_oversampling': 2},
        'output': {'directory': str(absolute_path(output_directory))},
        'geocoding': {'epsg': epsg}})
    if spacing_m is not None:
        cfg['geocoding']['spacing_m'] = spacing_m
    if aoi is not None or bbox is not None:
        cfg['aoi'] = {'path': str(absolute_path(aoi))} if aoi else {'bbox': bbox}
        if aoi_layer:
            cfg['aoi']['layer'] = aoi_layer
    cfg = validate_stack_config(cfg)
    from .geocode import validate_geocoding
    validate_geocoding(cfg['geocoding'])
    plan = plan_stack(cfg)
    if validate_dem_coverage:
        from .batch import dem_extent, validate_burst
        extent = dem_extent(cfg)
        for burst in plan.bursts:
            effective = merge_config(cfg,{'inputs':{'burst_id':burst.id}})
            effective.pop('aoi',None)
            validate_burst(effective,(burst,burst),extent)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--safe-dir')
    parser.add_argument('--orbit-dir')
    parser.add_argument('--dem')
    parser.add_argument('--prepared-inputs',type=Path,help='Manifest written by prepare-inputs; fills SAFE/orbit/DEM paths')
    parser.add_argument('--output-directory', default='work/stack')
    parser.add_argument('--output', type=Path, default=Path('configs/stack.generated.yaml'))
    parser.add_argument('--polarization', required=True, choices=['HH', 'HV', 'VV', 'VH'])
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--burst-id')
    selector.add_argument('--aoi')
    selector.add_argument('--bbox', type=float, nargs=4)
    parser.add_argument('--aoi-layer')
    parser.add_argument('--epsg', type=int)
    parser.add_argument('--spacing-m', type=float)
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--reference-date')
    parser.add_argument('--neighbors', type=int, default=1)
    parser.add_argument('--max-separation-days', type=lambda v: None if v.lower() == 'null' else float(v), default=12)
    parser.add_argument('--pairs-file')
    parser.add_argument('--relative-paths', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    output = absolute_path(args.output)
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; use --force to replace it')
    try:
        kwargs = vars(args).copy()
        for key in ('output', 'relative_paths', 'force', 'prepared_inputs'):
            kwargs.pop(key)
        if args.prepared_inputs:
            from .prepared_inputs import prepared_paths
            kwargs['safe_dir'],kwargs['orbit_dir'],kwargs['dem'] = prepared_paths(
                args.prepared_inputs,args.safe_dir,args.orbit_dir,args.dem)
            kwargs['validate_dem_coverage'] = True
        if any(not kwargs[key] for key in ('safe_dir','orbit_dir','dem')):
            raise ValueError('--safe-dir, --orbit-dir and --dem are required unless supplied by --prepared-inputs')
        kwargs['safe_directory'] = kwargs.pop('safe_dir')
        kwargs['orbit_directory'] = kwargs.pop('orbit_dir')
        cfg = make_stack_config(**kwargs)
        # Expand operational defaults as well as the shared scientific settings.
        # Pair initialization deliberately continues to write compact overrides.
        document = merge_config(cfg, {'stack': {
            'verification_workers': verification_workers(cfg),
            'swath_correction': correction_settings(cfg)}})
        if args.relative_paths:
            for field in PATH_FIELDS:
                section = document
                for key in field[:-1]:
                    section = section.get(key) if isinstance(section, dict) else None
                if isinstance(section, dict) and section.get(field[-1]) is not None:
                    section[field[-1]] = os.path.relpath(section[field[-1]], output.parent)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w' if args.force else 'x') as stream:
            stream.write(CONFIG_HEADER)
            yaml.safe_dump(document, stream, sort_keys=False)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f'Wrote {output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
