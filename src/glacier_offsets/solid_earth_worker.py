"""Self-contained full-ENU PySolid worker for cached radar-source observations.

Runs as a file under any interpreter providing numpy, h5py and pysolid (for
example the isce3 environment); it imports nothing from glacier_offsets.
Solid-Earth displacement is evaluated on a bilinear 0.1-degree grid at the
bracketing UTC seconds of each exact owning-burst endpoint, differenced, then
projected through each package's own range/azimuth ENU sensitivities. Sampled
exact one-point evaluations must agree within 10 micrometres. Pairs are
written in chunks; an interrupted output resumes from its completed count.
"""
import argparse
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import json
import sys

import h5py
import numpy as np


TOLERANCE_M = 1e-5
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def versions():
    from importlib.metadata import version
    return {'pysolid': version('pysolid'), 'numpy': np.__version__, 'h5py': h5py.__version__,
            'python': sys.version.split()[0]}


def grid_definition(lat, lon, spacing):
    finite = np.isfinite(lat+lon)
    la0 = np.floor(np.min(lat[finite])/spacing)*spacing-spacing
    lo0 = np.floor(np.min(lon[finite])/spacing)*spacing-spacing
    nlat = int(np.ceil((np.max(lat[finite])-la0)/spacing))+2
    nlon = int(np.ceil((np.max(lon[finite])-lo0)/spacing))+2
    return finite, la0, lo0, nlat, nlon


def run(observations, output, spacing, chunk=8, stop_after=None):
    from pysolid.solid import solid_grid
    with h5py.File(observations, 'r') as h:
        lat = h['latitude'][:]
        lon = h['longitude'][:]
        early = h['early'][:]
        late = h['late'][:]
        owner = h['owner'][:]
        files = h['file_index'][:]
        geometry = h['geometry_by_file'][:]
    finite, la0, lo0, nlat, nlon = grid_definition(lat, lon, spacing)

    @lru_cache(maxsize=80)
    def grid(seconds):
        d = EPOCH+timedelta(seconds=int(seconds))
        return np.stack(solid_grid(d.year, d.month, d.day, d.hour, d.minute, d.second,
                                   la0, spacing, nlat, lo0, spacing, nlon), axis=-1)

    def value(seconds, la, lo):
        lower = int(np.floor(seconds))
        fraction = seconds-lower
        a = grid(lower)*(1-fraction)+grid(lower+1)*fraction
        y = (la-la0)/spacing
        x = (lo-lo0)/spacing
        iy = np.floor(y).astype(int)
        ix = np.floor(x).astype(int)
        fy = (y-iy)[:, None]
        fx = (x-ix)[:, None]
        return ((1-fy)*(1-fx)*a[iy, ix]+fy*(1-fx)*a[iy+1, ix]
                + (1-fy)*fx*a[iy, ix+1]+fy*fx*a[iy+1, ix+1])

    def exact(seconds, la, lo):
        lower = int(np.floor(seconds))
        fraction = seconds-lower
        v = []
        for second in (lower, lower+1):
            d = EPOCH+timedelta(seconds=second)
            v.append(np.array(solid_grid(d.year, d.month, d.day, d.hour, d.minute, d.second,
                                         float(la), .01, 1, float(lo), .01, 1))[:, 0, 0])
        return v[0]*(1-fraction)+v[1]*fraction

    ns, nt = early.shape
    mode = 'r+'
    try:
        with h5py.File(output, 'r') as existing:
            usable = (existing.attrs.get('status') == 'writing' and existing['set_ra_m'].shape == (ns, nt, 2)
                      and float(existing.attrs['spacing_degrees']) == spacing)
    except (OSError, KeyError):
        usable = False
    if not usable:
        mode = 'w'
    with h5py.File(output, mode) as out:
        if mode == 'w':
            out.create_dataset('set_ra_m', (ns, nt, 2), dtype='f8', chunks=(min(2048, ns), min(16, nt), 2),
                               fillvalue=np.nan, compression='gzip', compression_opts=2)
            out.create_dataset('validation_error_m', (nt,), dtype='f8', fillvalue=np.nan)
            out.attrs.update(status='writing', completed_pairs=0, spacing_degrees=spacing,
                             grid_origin=[la0, lo0], grid_shape=[nlat, nlon],
                             formula='geometry @ (SET_later_ENU - SET_earlier_ENU)', **versions())
        start = int(out.attrs['completed_pairs'])
        for begin in range(start, nt, chunk):
            stop = min(begin+chunk, nt)
            block = np.full((ns, stop-begin, 2), np.nan)
            errors = np.full(stop-begin, np.nan)
            for t in range(begin, stop):
                worst = []
                for burst in np.unique(owner):
                    rows = np.flatnonzero((owner == burst) & finite & np.isfinite(early[:, t]+late[:, t]))
                    if not len(rows):
                        continue
                    e, l = early[rows[0], t]*86400., late[rows[0], t]*86400.
                    if not np.all(early[rows, t] == early[rows[0], t]) or not np.all(late[rows, t] == late[rows[0], t]):
                        raise ValueError('Source endpoints differ inside an owning burst')
                    delta = value(l, lat[rows], lon[rows])-value(e, lat[rows], lon[rows])
                    block[rows, t-begin] = np.einsum('sci,si->sc', geometry[files[t], rows], delta)
                    if t % 10 == 0:
                        rng = np.random.default_rng([12345, t, int(burst)])
                        s = int(rng.integers(len(rows)))
                        r = rows[s]
                        point = exact(l, lat[r], lon[r])-exact(e, lat[r], lon[r])
                        worst.append(float(np.max(abs(point-delta[s]))))
                if worst:
                    errors[t-begin] = max(worst)
                    if max(worst) > TOLERANCE_M:
                        raise ValueError(f'Solid-Earth grid interpolation error {max(worst):.3g} m exceeds 10 micrometres; '
                                         'refine execution.solid_earth_spacing_deg')
            out['set_ra_m'][:, begin:stop] = block
            out['validation_error_m'][begin:stop] = errors
            out.attrs['completed_pairs'] = stop
            out.flush()
            print(json.dumps({'completed_pairs': stop, 'pairs': nt}), flush=True)
            if stop_after is not None and stop >= stop_after and stop < nt:
                return 130
        sampled = out['validation_error_m'][:]
        out.attrs.update(status='complete', exact_validation_samples=int(np.isfinite(sampled).sum()),
                         max_ENU_endpoint_difference_error_m=float(np.nanmax(sampled)) if np.isfinite(sampled).any() else np.nan)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--versions', action='store_true', help='Print dependency versions and exit')
    parser.add_argument('--observations')
    parser.add_argument('--output')
    parser.add_argument('--spacing', type=float, default=.1)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--stop-after', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.versions:
        print(json.dumps(versions()))
        return 0
    if not args.observations or not args.output:
        parser.error('--observations and --output are required')
    try:
        return run(args.observations, args.output, args.spacing, args.chunk, args.stop_after)
    except ValueError as error:
        print(f'solid-Earth worker error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
