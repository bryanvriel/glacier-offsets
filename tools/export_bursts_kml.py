#!/usr/bin/env python
"""Export one SAFE/ZIP's annotation burst footprints and geographic IDs to KML."""
import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from glacier_offsets.burst_kml import burst_document, read_kml_bursts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('safe', help='Local SAFE directory or ZIP file')
    parser.add_argument('--output', type=Path, help='Output KML; default: <SAFE stem>.bursts.kml in the current directory')
    parser.add_argument('--polarization', choices=['HH', 'HV', 'VV', 'VH'], help='Required if multiple polarizations exist')
    parser.add_argument('--subswaths', nargs='+', type=int, choices=[1, 2, 3], default=[1, 2, 3])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--orbit-dir', help='Auto-detect a local precise orbit, for consistency with processing burst IDs')
    group.add_argument('--orbit', help='Use this explicit local precise orbit file')
    parser.add_argument('--force', action='store_true', help='Replace an existing KML')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    output = (args.output or Path(Path(args.safe).stem + '.bursts.kml')).expanduser().resolve()
    if output.suffix.lower() != '.kml':
        parser.error('--output must have a .kml extension')
    if output.exists() and not args.force:
        parser.error(f'{output} already exists; choose another --output or use --force')
    try:
        bursts, scene, polarization, orbit = read_kml_bursts(args.safe, args.polarization, args.subswaths,
                                                            args.orbit, args.orbit_dir)
        tree = burst_document(bursts, scene, polarization, orbit)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('wb' if args.force else 'xb') as stream:
            tree.write(stream, encoding='utf-8', xml_declaration=True)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    counts = ', '.join(f'IW{swath}: {sum(b.swath == swath for b in bursts)}' for swath in sorted(args.subswaths))
    print(f'Wrote {output}: {len(bursts)} bursts ({counts}), {polarization}')
    print(f'Orbit: {orbit}' if orbit else 'Burst IDs use the SAFE annotation ascending-node time; no orbit file needed.')


if __name__ == '__main__':
    main()
