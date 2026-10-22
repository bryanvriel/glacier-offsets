#!/usr/bin/env python3
"""Compare reconstructed burst metadata and every complex sample to full SAFEs.

This is a read-only input compatibility gate, independent of COMPASS accuracy.
The collection may be incomplete while diagnosing individual published groups;
normal preparation still requires a complete managed collection.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from osgeo import gdal

from glacier_offsets.burst_downloads import MANIFEST, validate_safe, verify_files
from glacier_offsets.initialization import OrbitCatalog, read_scene
from glacier_offsets.inputs import inventory
from glacier_offsets.products import write_json
from glacier_offsets.stack import scan_safes


def compare(collection, full_safe_directory, orbit_directory):
    collection = Path(collection).resolve()
    manifest = json.loads((collection / MANIFEST).read_text())
    originals = {}
    for path in scan_safes(full_safe_directory):
        scene = read_scene(path)
        key = scene.mission, scene.start.date().isoformat()
        if key in originals:
            raise ValueError(f'Ambiguous original acquisitions: {key}')
        originals[key] = scene
    orbits = OrbitCatalog(orbit_directory)
    report = {'collection': str(collection), 'status': 'complete', 'acquisitions': [],
              'limitations': 'Input equivalence only; processing agreement requires a separate native stack run.'}
    for group in manifest['acquisitions']:
        if group.get('status') != 'complete':
            report['status'] = 'incomplete'
            continue
        began = time.monotonic()
        safe = collection / group['safe']
        original = originals[group['mission'], group['date']]
        verify_files(safe, group['validation'])
        validate_safe(safe, group, manifest['request'])
        orbit = orbits.resolve(original).path
        reconstructed = inventory(safe, orbit, manifest['request']['polarization'], [1, 2, 3])
        full = {b.id: b for b in inventory(original.path, orbit, manifest['request']['polarization'], [1, 2, 3])}
        result = {'id': group['id'], 'safe': str(safe), 'original': str(original.path), 'bursts': [], 'passed': True}
        report['acquisitions'].append(result)
        datasets = {}
        for burst in reconstructed:
            left, right = burst.native, full[burst.id].native
            differences = {}
            for key in ('sensing_start', 'shape', 'starting_range', 'range_pixel_spacing',
                        'wavelength', 'azimuth_time_interval', 'platform_id', 'polarization',
                        'orbit_direction', 'first_valid_line', 'last_valid_line',
                        'first_valid_sample', 'last_valid_sample'):
                a, b = getattr(left, key), getattr(right, key)
                if a != b:
                    differences[key] = [str(a), str(b)]
            samples, unequal, max_difference = 0, 0, 0.
            for native in (left, right):
                if native.tiff_path not in datasets:
                    datasets[native.tiff_path] = gdal.Open(native.tiff_path, gdal.GA_ReadOnly)
            height, width = left.shape
            for y in range(0, height, 64):
                arrays = [datasets[n.tiff_path].ReadAsArray(0, n.i_burst * height + y,
                          width, min(64, height-y)) for n in (left, right)]
                a, b = arrays
                samples += a.size
                unequal += int(np.count_nonzero(a != b))
                max_difference = max(max_difference, float(np.max(np.abs(a-b))))
            borders_equal = len(left.border) == len(right.border) and all(
                a.equals_exact(b, 1e-10) for a,b in zip(left.border,right.border))
            row = {'burst_id': burst.id, 'sensing_start': burst.start.isoformat(), 'shape': list(left.shape),
                   'metadata_differences': differences, 'borders_equal': borders_equal,
                   'samples_compared': samples, 'unequal_samples': unequal,
                   'max_complex_difference': max_difference}
            row['passed'] = not differences and borders_equal and unequal == 0
            result['bursts'].append(row)
            result['passed'] &= row['passed']
        datasets.clear()
        result['elapsed_seconds'] = time.monotonic() - began
        if not result['passed']:
            report['status'] = 'failed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path)
    parser.add_argument('--full-safe-dir', required=True, type=Path)
    parser.add_argument('--orbit-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.collection, args.full_safe_dir, args.orbit_dir)
    write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
