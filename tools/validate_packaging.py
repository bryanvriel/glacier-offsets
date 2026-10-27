"""Compare every package slice with source GeoTIFFs and optionally check iceutils.

Run in the processing environment:
  python tools/validate_packaging.py CONFIG.yaml PACKAGE.nc --ice-python /path/to/ice/bin/python
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5netcdf
import numpy as np
from osgeo import gdal

from glacier_offsets.packaging import PRIMARY, CORRECTION_FLAGS, read_tile, tiles, validate_package
from glacier_offsets.stack_config import load_status_config
from glacier_offsets.swath_correction_config import output_root


def compare_sources(cfg, path):
    report = validate_package(path)
    with h5netcdf.File(path, 'r') as h:
        source = Path(cfg['output']['directory']) / 'pairs' if h.attrs['source'] == 'raw' else output_root(cfg)
        keys = PRIMARY + (CORRECTION_FLAGS if h.attrs['source'] == 'swath_corrected' else ())
        for index, pair in enumerate(h['pair_id'][:].astype(str)):
            rasters = {key: gdal.Open(str(source / pair / 'geo' / f'{key}.tif')) for key in keys}
            for section in tiles(report['shape'][1:]):
                mask = h['valid_mask'][(index, *section)].astype(bool)
                for key in keys:
                    expected = read_tile(rasters[key], section)
                    if key == 'valid_mask' or key in CORRECTION_FLAGS:
                        expected = np.where(mask, expected, 0).astype('uint8')
                    else:
                        expected = np.where(mask, expected, np.nan).astype('float32')
                    np.testing.assert_array_equal(h[key][(index, *section)], expected, err_msg=f'{pair}/{key}')
            rasters.clear()
        report['primary_variables_compared'] = list(PRIMARY)
        report['source_agreement'] = 'exact, including NaNs and integer masks'
        report['package_bytes'] = Path(path).stat().st_size
        for section in tiles(report['shape'][1:]):
            supported = h['source_observation_index'][section] >= 0
            los = h['reference_los_enu'][(slice(None), *section)][:, supported]
            sensitivity = h['range_sensitivity_enu'][(slice(None), *section)][:, supported]
            np.testing.assert_allclose(np.linalg.norm(los, axis=0), 1., atol=1e-8)
            np.testing.assert_allclose(sensitivity, -los, atol=1e-5)
        report['reference_geometry_check'] = 'unit ground-to-sensor LOS; range ENU sensitivity equals minus LOS'
    return report


def iceutils_check(path, python):
    """A fresh interpreter reads a copied package, with no stack configuration."""
    code = '''
import json, sys
import numpy as np
import iceutils as ice
stack = ice.Stack(sys.argv[1])
assert stack.ds['valid_mask'].dtype == np.dtype('uint8')
mask = stack.slice(0, key='valid_mask')
data = stack.slice(0, key='range_displacement_m')
assert mask.shape == data.shape == stack.hdr.shape
assert len(set(stack.names)) == stack.Nt
owner = stack.ds['source_burst_index'].values
intervals = stack.ds['pair_interval_seconds'].values
for i in range(stack.Nt):
    valid = stack.slice(i, key='valid_mask').astype(bool)
    assert np.all(np.isfinite(intervals[i, owner[valid]-1]))
assert np.isfinite(stack.ds['reference_orbit_position'].values).all()
good = np.argwhere(mask)
if len(good):
    row, col = good[0]
    assert len(stack.timeseries(coord=(int(row), int(col)), key='range_displacement_m')) == stack.Nt
print(json.dumps({'iceutils': ice.__file__, 'shape': list(stack.shape), 'EPSG': stack.hdr.epsg,
                  'mask_dtype': str(mask.dtype), 'standalone_copy': True, 'timing_geometry_available': True}))
stack.close()
'''
    with tempfile.TemporaryDirectory(prefix='glacier-package-iceutils-') as folder:
        standalone = Path(folder) / 'standalone.nc'
        shutil.copyfile(path, standalone)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['MPLCONFIGDIR'] = str(Path(folder) / 'matplotlib')
        result = subprocess.run([python, '-c', code, str(standalone)], cwd=folder, env=env,
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('package')
    parser.add_argument('--ice-python')
    args = parser.parse_args()
    report = compare_sources(load_status_config(args.config), args.package)
    if args.ice_python:
        report['iceutils_check'] = iceutils_check(args.package, args.ice_python)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
