#!/usr/bin/env python
"""Initialize a single-pair workflow YAML using local SAFE and precise-orbit metadata."""
import argparse
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import yaml
from glacier_offsets.initialization import make_pair_config
from glacier_offsets.config import PATH_FIELDS, config_overrides, load_defaults


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('safes', nargs=2, metavar='SAFE', help='Two SAFE directories or ZIP files, in either date order')
    parser.add_argument('--dem', required=True, help='Local ellipsoidal-height DEM')
    parser.add_argument('--orbit-dir', required=True, help='Local directory of precise orbit EOF/XML files (searched recursively)')
    parser.add_argument('--output', type=Path, default=Path('configs/pair.generated.yaml'), help='YAML path')
    parser.add_argument('--output-directory', help='Processing output directory; default is work/<reference>__<secondary>')
    parser.add_argument('--relative-paths', action='store_true', help='Write local paths relative to the generated YAML directory')
    parser.add_argument('--polarization', choices=['HH', 'HV', 'VV', 'VH'], help='Required if multiple polarizations are common')
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--burst-id', help='Optional explicit common geographic burst ID')
    selector.add_argument('--aoi', help='Polygon AOI file (GeoJSON, Shapefile, GeoPackage or KML) selecting common bursts')
    selector.add_argument('--bbox', nargs=4, type=float, metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'), help='WGS84 AOI bounds')
    parser.add_argument('--aoi-layer', help='Explicit layer name for --aoi')
    parser.add_argument('--epsg', type=int, help='Projected metre output CRS; defaults to the DEM EPSG')
    parser.add_argument('--spacing-m', type=float, help=f"Geocoded cell spacing in metres (default: {load_defaults()['geocoding']['spacing_m']})")
    parser.add_argument('--max-separation-days', type=float, help='Reject this pair if its exact manifest start-time gap exceeds this limit')
    parser.add_argument('--force', action='store_true', help='Replace an existing generated YAML')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; choose another --output or use --force')
    try:
        cfg = make_pair_config(args.safes, args.dem, args.orbit_dir, args.output_directory,
                               args.polarization, args.burst_id, args.epsg, args.spacing_m,
                               args.max_separation_days, aoi=args.aoi, bbox=args.bbox, aoi_layer=args.aoi_layer)
        document = config_overrides(cfg)
        if args.relative_paths:
            for field in PATH_FIELDS:
                section = document
                for key in field[:-1]:
                    section = section.get(key) if isinstance(section, dict) else None
                if isinstance(section, dict) and section.get(field[-1]) is not None:
                    section[field[-1]] = os.path.relpath(section[field[-1]], output.parent)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('w' if args.force else 'x') as stream:
            stream.write('# Single-pair config. Matching arrays are [azimuth, range].\n'
                         '# DEM heights must be above the WGS84 ellipsoid.\n'
                         '# Select one inputs.burst_id or an AOI before processing.\n')
            yaml.safe_dump({'extends': 'defaults'} | document, stream, sort_keys=False)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    print(f'Wrote {output}')
    for side in ['reference', 'secondary']:
        print(f'{side}: {cfg["inputs"][side + "_safe"]}')
        print(f'  orbit: {cfg["inputs"][side + "_orbit"]}')
    if args.burst_id:
        print(f'Validated common burst: {args.burst_id}')
    elif args.aoi is not None or args.bbox is not None:
        print('Validated AOI selection. Inspect with:')
        print(f'  bin/glacier-offsets list-bursts "{output}"')
    else:
        print('Burst not selected. Next run:')
        print(f'  bin/glacier-offsets list-bursts "{output}"')
        print('Then set inputs.burst_id to one reported geographic ID.')


if __name__ == '__main__':
    main()
