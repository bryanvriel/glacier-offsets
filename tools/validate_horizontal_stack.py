"""Compare every horizontal slice to a separate closed-form radar inversion.

PYTHONPATH=src:. python tools/validate_horizontal_stack.py RADAR.nc HORIZONTAL.nc \
    --ice-python /path/to/ice/bin/python
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

from glacier_offsets.horizontal import validate_horizontal_stack
from glacier_offsets.packaging import tiles
from glacier_offsets.products import checksum
from glacier_offsets.tides import json_scalar


def audit(source, horizontal):
    report = validate_horizontal_stack(horizontal)
    source_digest = checksum(source)
    error, checked = 0., 0
    with h5netcdf.File(source) as a, h5netcdf.File(horizontal) as b:
        recipe = json_scalar(b, 'horizontal_conversion_provenance_json')['recipe']
        if source_digest != recipe['input']['sha256']:
            raise ValueError('Source differs from the conversion input')
        original, after = (json.loads(h.attrs['content_hashes_json']) for h in (a, b))
        if any(after[key] != digest for key, digest in original.items() if key != 'provenance_json'):
            raise ValueError('An original source variable changed')
        axes = a['enu'][:].astype(str).tolist()
        for section in tiles(report['shape'][1:]):
            r, z = (a[f'{axis}_sensitivity_enu'][(slice(None), *section)] for axis in ('range', 'azimuth'))
            re, rn = r[axes.index('east')], r[axes.index('north')]
            ae, an = z[axes.index('east')], z[axes.index('north')]
            determinant = re*an-rn*ae
            for i in range(report['pairs']):
                slot = (i, *section)
                take = b['horizontal_valid_mask'][slot] == 1
                dr, da = (a[f'{axis}_displacement_m'][slot].astype('float64') for axis in ('range', 'azimuth'))
                expected = [(an[take]*dr[take]-rn[take]*da[take])/determinant[take],
                            (re[take]*da[take]-ae[take]*dr[take])/determinant[take]]
                for key, truth in zip(('east_displacement_m', 'north_displacement_m'), expected):
                    actual = b[key][slot]
                    if not np.isnan(actual[~take]).all():
                        raise ValueError('Non-NaN horizontal data outside valid support')
                    np.testing.assert_allclose(actual[take], truth, rtol=2e-6, atol=2e-6)
                    if take.any():
                        error = max(error, float(np.max(abs(actual[take]-truth))))
                checked += int(take.sum())
    if checksum(source) != source_digest:
        raise ValueError('Source changed during validation')
    return report | {'checked_horizontal_observations': checked, 'max_closed_form_difference_m': error,
                     'preserved_variables': 'all original values except updated processing provenance',
                     'bytes': Path(horizontal).stat().st_size}


def iceutils_check(path, python):
    """Read a standalone copy in a fresh process outside the processing stack."""
    code = '''
import json, sys
import numpy as np
import iceutils as ice
stack = ice.Stack(sys.argv[1])
assert stack.ds['horizontal_valid_mask'].dtype == np.dtype('uint8')
assert stack.ds['horizontal_geometry_flags'].dtype == np.dtype('uint8')
for i in range(stack.Nt):
    mask = stack.slice(i, key='horizontal_valid_mask').astype(bool)
    for key in ('east_displacement_m', 'north_displacement_m'):
        value = stack.slice(i, key=key)
        assert value.shape == mask.shape == stack.hdr.shape
        assert np.isfinite(value[mask]).all() and np.isnan(value[~mask]).all()
owner = stack.ds['source_burst_index'].values
early = stack.ds['earlier_burst_time'].values
late = stack.ds['later_burst_time'].values
for i in range(stack.Nt):
    valid = stack.slice(i, key='horizontal_valid_mask').astype(bool)
    sources = owner[valid]-1
    assert np.isfinite(early[i, sources]).all() and np.isfinite(late[i, sources]).all()
    assert (late[i, sources] > early[i, sources]).all()
assert len(set(stack.names)) == stack.Nt
print(json.dumps({'iceutils': ice.__file__, 'shape': list(stack.shape), 'EPSG': stack.hdr.epsg,
                  'horizontal_mask_dtype': str(stack.ds['horizontal_valid_mask'].dtype),
                  'standalone_copy': True, 'exact_burst_endpoints_available': True}))
stack.close()
'''
    with tempfile.TemporaryDirectory(prefix='horizontal-iceutils-') as folder:
        standalone = Path(folder)/'horizontal.nc'
        shutil.copyfile(path, standalone)
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env['MPLCONFIGDIR'] = str(Path(folder)/'matplotlib')
        result = subprocess.run([python, '-c', code, str(standalone)], cwd=folder, env=env,
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('horizontal')
    parser.add_argument('--ice-python')
    args = parser.parse_args()
    report = audit(args.source, args.horizontal)
    if args.ice_python:
        report['iceutils_check'] = iceutils_check(args.horizontal, args.ice_python)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
