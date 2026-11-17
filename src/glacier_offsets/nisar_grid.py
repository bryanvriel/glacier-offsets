"""Native GOFF lattice and integer-decimated map-grid conventions."""
import numpy as np


def decimation_factor(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError('NISAR decimation factor must be a positive integer')
    return value


def axis_selection(values, factor):
    """Select native centers 40 + 80*k where k is divisible by factor."""
    keep = np.flatnonzero(np.rint((values-40.)/80.).astype('i8') % factor == 0)
    return slice(int(keep[0]), int(keep[-1])+1, factor) if len(keep) else slice(0, 0, factor)


def sampling_record(factor):
    return {'method': 'integer_decimation', 'factor': decimation_factor(factor),
            'native_posting_m': 80., 'posting_m': 80.*factor,
            'anchor_centers_m': [40., 40.], 'additional_resampling': 'none'}


def validate_grid(h, spec, shape):
    """Accept legacy native grids and explicitly declared decimated grids."""
    grid = spec['grid']; gt = grid['transform']
    sampling = spec.get('sampling')
    factor = decimation_factor(sampling['factor']) if sampling is not None else 1
    step = 80.*factor
    if sampling is not None and sampling != sampling_record(factor):
        raise ValueError('Invalid NISAR sampling provenance')
    for name, expected in [('map_posting_m', step), ('decimation_factor', factor),
                           ('sampling_method', 'integer_decimation')]:
        if name in h.attrs and h.attrs[name] != expected:
            raise ValueError('Inconsistent NISAR sampling attribute: '+name)
    if (grid['shape'] != list(shape) or int(h.attrs['EPSG']) != 3031 or grid.get('epsg') != 3031
            or gt[1:3] != [step, 0.] or gt[4:6] != [0., -step]):
        raise ValueError('Invalid package grid')
    for k, start, delta in [('x', gt[0], step), ('y', gt[3], -step)]:
        values = h[k][:]
        expected = start+(np.arange(len(values))+.5)*delta
        if (not np.array_equal(values, expected)
                or not np.allclose(np.mod(values-40., step), 0., rtol=0, atol=1e-6)):
            raise ValueError('Invalid pixel-center coordinates or sampling alignment')
    return step
