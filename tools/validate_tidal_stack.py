"""Audit a tide-corrected netCDF using only its parent netCDF and optional CATS.

python tools/validate_tidal_stack.py INPUT.nc CORRECTED.nc --model CATS2008_v2023.nc
"""
import argparse
import io
import json
from pathlib import Path
import tempfile

import h5netcdf
import numpy as np
from pyproj import Transformer

from glacier_offsets.products import checksum
from glacier_offsets.tides import FLOAT_FIELDS, json_scalar, validate_tidal_stack


def model_parity(path, model_file):
    """Compare sparse real observations with pyTMD's public compute driver.

    A cropped copy avoids the public driver's eager whole-Antarctic complex
    allocation. File decoding, flexure application, interpolation and prediction
    are all performed by that independent driver.
    """
    import pyTMD
    import xarray as xr
    with h5netcdf.File(path) as h:
        valid = h['tidal_correction_available'][0].astype(bool)
        candidates = np.flatnonzero(valid & (h['tidal_response_factor'][:] > .05))
        if not len(candidates):
            raise ValueError('No modeled nonzero-response observations for the independent check')
        select = candidates[np.unique(np.linspace(0, len(candidates)-1, min(12, len(candidates))).astype(int))]
        lon = h['observation_longitude'][:].ravel()[select]
        lat = h['observation_latitude'][:].ravel()[select]
        owner = h['source_burst_index'][:].ravel()[select]-1
        times = [h[k][0][owner] for k in ['earlier_burst_time', 'later_burst_time']]
        stored = [h[k][0].ravel()[select] for k in ['tide_height_earlier_m', 'tide_height_later_m']]
        record = json_scalar(h, 'tidal_correction_provenance_json')
    if checksum(model_file) != record['model_file']['sha256']:
        raise ValueError('Reference model differs from the model used for correction')
    with tempfile.TemporaryDirectory(prefix='cats-reference-') as folder:
        subset = Path(folder)/'CATS2008_v2023.nc'
        with xr.open_dataset(model_file, engine='h5netcdf') as raw:
            projection = raw['mapping'].attrs['spatial_proj4']
            x, y = Transformer.from_crs(4326, projection, always_xy=True).transform(lon, lat)
            selection = {}
            for key, coords in [('x', x), ('y', y)]:
                axis = raw[key].values
                padding = 2*abs(axis[1]-axis[0])
                ix = np.flatnonzero((axis >= min(coords)-padding) & (axis <= max(coords)+padding))
                selection[key] = slice(int(ix.min()), int(ix.max())+1)
            keep = ['hRe', 'hIm', 'mask', 'wct', 'flexure', 'mapping']
            raw[keep].isel(selection).load().to_netcdf(subset, engine='h5netcdf',
                encoding={key: {'dtype':'float64', '_FillValue':np.nan} for key in ('hRe', 'hIm')})
        definition = {'format':'TMD3', 'name':'CATS2008-v2023', 'projection':projection,
                      'z':{'model_file':str(subset),'units':'m','variable':'tide_ocean'}}
        errors = []
        for epoch, actual in zip(times, stored):
            predicted = pyTMD.compute.tide_elevations(lon, lat, epoch, directory=None,
                definition_file=io.StringIO(json.dumps(definition)), epoch=(1970,1,1,0,0,0), standard='UTC',
                type='trajectory', method='linear', apply_flexure=record['response'] == 'model', extrapolate=False)
            expected = np.asarray(predicted.values).reshape(-1)
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
            errors.extend(abs(actual-expected).tolist())
        return {'samples': len(lon), 'max_height_error_m': max(errors), 'reference': 'pyTMD.compute.tide_elevations'}


def audit(source, corrected):
    report = validate_tidal_stack(corrected)
    with h5netcdf.File(source) as a, h5netcdf.File(corrected) as b:
        original = json.loads(a.attrs['content_hashes_json'])
        after = json.loads(b.attrs['content_hashes_json'])
        for key, digest in original.items():
            if key not in ('range_displacement_m', 'provenance_json') and after[key] != digest:
                raise ValueError(f'Changed original variable: {key}')
        if after[FLOAT_FIELDS[0]] != original['range_displacement_m']:
            raise ValueError('Saved original range differs from the parent stack')
        record = json_scalar(b, 'tidal_correction_provenance_json')
        if checksum(source) != record['input']['sha256']:
            raise ValueError('Parent stack differs from correction input')
        report['preserved_variables'] = 'all original variables except corrected range and updated processing provenance'
        report['bytes'] = Path(corrected).stat().st_size
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input')
    parser.add_argument('corrected')
    parser.add_argument('--model', help='Optional original CATS2008_v2023.nc for independent pyTMD parity')
    parser.add_argument('--ice-python', help='Optional interpreter with iceutils for a standalone-copy check')
    args = parser.parse_args()
    report = audit(args.input, args.corrected)
    if args.model:
        report['independent_prediction_check'] = model_parity(args.corrected, args.model)
    if args.ice_python:
        from tools.validate_packaging import iceutils_check
        report['iceutils_check'] = iceutils_check(args.corrected, args.ice_python)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
