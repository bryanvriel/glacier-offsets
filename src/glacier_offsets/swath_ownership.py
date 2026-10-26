"""Classify whole matching footprints using production source ownership."""
import numpy as np
from osgeo import gdal


def pure_swath_footprints(coreg, line, sample, params, source_ids):
    """Same-swath joins qualify; crossing a swath or unowned sample does not.

    Zero-gross production matching only: the symmetric search is centered on
    the reference chip. Full original-grid footprints are checked, not owners
    at their centers alone. Reading bounded strips avoids a full owner image.
    """
    ds = gdal.Open(coreg['source_index'])
    lookup = np.array([0] + [int(key[-1]) for key in source_ids], dtype='uint8')
    pure = np.zeros(line.shape, dtype='uint8')
    height = params['window'][0] + 2 * params['search'][0]
    width = params['window'][1] + 2 * params['search'][1]
    starts_x = np.rint(sample[0] - (params['window'][1] - 1) / 2).astype(int) - params['search'][1]
    cols = (starts_x >= 0) & (starts_x + width <= ds.RasterXSize)
    x = starts_x[cols]
    for row, center in enumerate(line[:, 0]):
        y = int(round(center - (params['window'][0] - 1) / 2)) - params['search'][0]
        if y < 0 or y + height > ds.RasterYSize:
            continue
        owners = ds.ReadAsArray(0, y, ds.RasterXSize, height)
        swaths = lookup[owners]
        for swath in (1, 2):
            good = np.all(swaths == swath, axis=0)
            total = np.r_[0, np.cumsum(good)]
            selected = total[x + width] - total[x] == width
            pure[row, np.flatnonzero(cols)[selected]] = swath
    return pure
