"""Windowed CATS2008-v2023 harmonics with pyTMD astronomical prediction."""
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer


class Cats2008:
    """Read local TMD3 data once, without loading Antarctic current fields.

    Harmonic constants are interpolated at the actual measurement locations.
    The v2023 flexure percentage multiplies constituents BEFORE interpolation,
    following pyTMD's apply_flexure convention. Values above 100% are retained.
    """

    def __init__(self, path, longitude, latitude, *, response='model'):
        try:
            import pyTMD
            import xarray as xr
        except ImportError as exc:
            raise ImportError('Tidal correction requires the optional dependencies: pip install ".[tides]"') from exc
        if not (3, 0, 9) <= tuple(map(int, pyTMD.__version__.split('.'))) < (3, 1, 0):
            raise ValueError('Tidal correction requires pyTMD>=3.0.9,<3.1')
        if response not in ('model', 'floating'):
            raise ValueError('Tidal response must be model or floating')
        self.version = pyTMD.__version__
        self.response = response
        self.path = Path(path).expanduser().resolve()
        self.metadata = {}
        with xr.open_dataset(self.path, engine='h5netcdf', decode_times=False) as raw:
            title = raw.attrs.get('Title', raw.attrs.get('title', ''))
            if 'CATS2008' not in title or not {'hRe', 'hIm', 'mask', 'mapping', 'constituents'}.issubset(raw.variables):
                raise ValueError('Use the local CATS2008-v2023 TMD3 netCDF model (CATS2008_v2023.nc)')
            if any(raw[key].attrs.get('units') != 'm' for key in ('hRe', 'hIm')):
                raise ValueError('CATS2008 height constituents must declare metre units')
            if response == 'model' and 'flexure' not in raw:
                raise ValueError('Model response requires the CATS2008-v2023 flexure field; no freely-floating fallback is applied')
            crs = CRS.from_user_input(raw['mapping'].attrs['spatial_proj4'])
            x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(longitude, latitude)
            xs, ys = raw['x'].values, raw['y'].values
            if min(len(xs), len(ys)) < 2 or not np.all(np.diff(xs) > 0) or not (np.all(np.diff(ys) > 0) or np.all(np.diff(ys) < 0)):
                raise ValueError('CATS2008 model axes must be regular monotonic coordinates')
            # Bounding slices include an interpolation neighbor at both ends.
            # Read only height constituents, flexure and mask, never currents.
            def window(axis, values):
                low, high = np.nanmin(values), np.nanmax(values)
                if axis[-1] > axis[0]:
                    lo, hi = np.searchsorted(axis, [low, high])
                else:
                    a, b = np.searchsorted(axis[::-1], [low, high])
                    lo, hi = len(axis)-b-1, len(axis)-a-1
                lo = max(0, min(int(lo)-1, len(axis)-2))
                hi = max(lo+2, min(int(hi)+2, len(axis)))
                return slice(lo, hi)
            region = {'x': window(xs, x), 'y': window(ys, y)}
            names = raw['constituents'].attrs['constituent_order'].lower().split()
            if len(set(names)) != len(names) or len(names) != raw.sizes['constituents']:
                raise ValueError('Invalid model constituent ordering')
            if not {'m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1', 'mf', 'mm'}.issubset(names):
                raise ValueError('CATS2008-v2023 major constituents are incomplete')
            # xarray decodes the int16 scale factors. TMD3 stores the positive
            # imaginary part; pyTMD complex harmonics use its negative.
            harmonic = (raw['hRe'].isel(region)-1j*raw['hIm'].isel(region)).load()
            flexure = (raw['flexure'].isel(region).astype('float64')/100).load() if 'flexure' in raw else None
            fields = {}
            for j, name in enumerate(names):
                a = harmonic.isel(constituents=j, drop=True)
                if response == 'model':
                    a = a*flexure.where(flexure >= 0)
                fields[name] = a.assign_attrs(units='m')
            fields['ocean_mask'] = raw['mask'].isel(region).astype('float64').load()
            if flexure is not None:
                fields['response_factor'] = flexure if response == 'model' else xr.ones_like(flexure)
            else:
                fields['response_factor'] = xr.ones_like(fields['ocean_mask'])
            ds = xr.Dataset(fields).sortby('y')
            self.local = ds.interp(x=xr.DataArray(x, dims='point'), y=xr.DataArray(y, dims='point'), method='linear').load()
            self.metadata = {k: str(raw.attrs[k]) for k in ('Title', 'Description', 'Data_citation', 'Author', 'license') if k in raw.attrs}
            self.metadata.update(constituents=names, model_crs=crs.to_wkt(), format='TMD3', interpolation='linear',
                                 extrapolation=False, response=response, nodal_corrections='TMD3', infer_minor=True,
                                 flexure_policy='multiply constituents before interpolation; percentage / 100; no clipping' if response == 'model' else 'freely floating')
        self.response_factor = np.asarray(self.local['response_factor'].values)
        self.ocean_mask = np.asarray(self.local['ocean_mask'].values)
        self.available = np.isfinite(self.response_factor)
        for name in names:
            self.available &= np.isfinite(self.local[name].values)
        if response == 'floating':
            self.available &= self.ocean_mask >= 1.-1e-6

    def predict(self, indices, utc_seconds):
        """Predict metres at one exact UTC endpoint for selected observations."""
        import pyTMD
        # OTIS/TMD3 use UTC days since 1992-01-01 and zero delta-T, matching
        # pyTMD.compute.tide_elevations. This path needs no IERS/network lookup.
        tide_days = np.array([(float(utc_seconds)-694224000.)/86400.])
        local = self.local[self.metadata['constituents']].isel(point=indices)
        major = pyTMD.predict.time_series(tide_days, local, corrections='TMD3', deltat=0.)
        minor = pyTMD.predict.infer_minor(tide_days, local, corrections='TMD3', deltat=0.)
        tide = major+minor
        result = np.asarray(tide.transpose('point', 'time').values[:, 0], dtype='float64')
        result[~self.available[indices]] = np.nan
        return result
