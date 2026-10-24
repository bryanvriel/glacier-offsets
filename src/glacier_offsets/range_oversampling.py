"""Complex range-2x interpolation before native magnitude matching.

Native CPU Ampcor with `deramp: 0` detects magnitude at the original range
sampling. Detection doubles the bandwidth, so the amplitude spectrum aliases
and range estimates are pulled toward whole pixels (about 0.1 px in IW1 and
0.06 px in IW2 for Sentinel-1). Interpolating the complex SLC rows twice in
range first removes that bias without the burst-join artifacts of
`deramp: 1`.

With `matching.range_oversampling: 2`, stack CPU matching interpolates both
merged SLCs into temporary inputs and runs native Ampcor with window, search
and stride doubled in range. Output sample j represents original coordinate
(j + 0.5)/2 - 0.5, so physical chip centers and pixel-cell footprints are
unchanged. Offsets, gross values and covariance are converted back to
original reference pixels before publication; input support, ownership and
seam flags stay on the original grid. Native raw/correlation oversampling and
zoom are unchanged, so the physical range refinement extent halves.

This is the treatment validated by `tools.compare_merged_preprocessing` and
the known-shift experiment, except that complete images are interpolated
rather than retained strips.
"""
from contextlib import contextmanager
from copy import deepcopy
import logging
from pathlib import Path
import shutil
import time

import numpy as np
from osgeo import gdal
import scipy.fft
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import resample

from .matching_grid import MatchingGrid
from .raster_io import bounded_raster_cache

FACTOR = 2
METHOD = 'complex-range-2x-full-row-fourier-v1'
ROW_BLOCK = 16


def factor(params):
    return params.get('range_oversampling', 1)


def resample_range(rows, factor=FACTOR):
    """Periodic complex Fourier interpolation; no azimuth processing or deramp.

    Output sample j represents input coordinate (j+.5)/factor-.5. This preserves
    pixel-cell boundaries, physical window width AND center when W becomes f*W.
    scipy.signal.resample handles even-length Nyquist-bin splitting explicitly.
    """
    if factor != FACTOR:
        raise ValueError('Only 2x range interpolation is implemented')
    if not np.iscomplexobj(rows) or not np.isfinite(rows).all():
        raise ValueError('Range interpolation requires finite complex rows; no filling is performed')
    up = resample(rows, factor*rows.shape[-1], axis=-1)
    origin = (1/factor-1)/2
    phase = np.exp(2j*np.pi*fftfreq(up.shape[-1])*origin*factor).astype(up.dtype)
    return ifft(fft(up, axis=-1)*phase, axis=-1).astype('complex64')


def prepare(source, output, *, workers=1):
    """Interpolate a complete SLC in bounded row blocks into a stripped TIFF.

    Unsupported (zero) regions are interpolated like any other samples; window
    eligibility is decided on the original inputs.
    """
    gdal.UseExceptions()
    ds = gdal.Open(str(source))
    height, width = ds.RasterYSize, ds.RasterXSize
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed strips, so native CPU Ampcor can memory-map the input.
    target = gdal.GetDriverByName('GTiff').Create(str(output), FACTOR*width, height, 1, gdal.GDT_CFloat32,
                                                options=['TILED=NO', 'COMPRESS=NONE', 'BIGTIFF=IF_SAFER'])
    band = target.GetRasterBand(1)
    with bounded_raster_cache(), scipy.fft.set_workers(workers):
        for row in range(0, height, ROW_BLOCK):
            count = min(ROW_BLOCK, height-row)
            band.WriteArray(resample_range(ds.ReadAsArray(0, row, width, count)), 0, row)
    band = target = ds = None
    return {'source': str(source), 'output': str(output), 'original_shape_yx': [height, width],
            'factor_yx': [1, FACTOR], 'original_coordinate_of_output_sample_0_yx': [0, (1/FACTOR-1)/2]}


@contextmanager
def prepared_inputs(slcs, directory, *, workers=1):
    """Temporary range-2x copies of both merged SLCs, removed afterwards."""
    directory = Path(directory)
    shutil.rmtree(directory, ignore_errors=True)
    started = time.perf_counter()
    try:
        paths, records = {}, []
        for side in ('reference', 'secondary'):
            logging.info('Interpolating %s merged SLC 2x in range', side)
            paths[side] = directory/f'{side}.tif'
            records.append(prepare(slcs[side], paths[side], workers=workers))
        yield paths, {'method': METHOD, 'factor_yx': [1, FACTOR], 'inputs': records,
                      'preparation_seconds': time.perf_counter()-started,
                      'interpolator': 'Complete complex rows; scipy.signal.resample plus Fourier phase shift; '
                                      'periodic boundaries; no taper, deramp, fill or azimuth interpolation'}
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def native_parameters(params):
    """Native settings on the range-2x inputs."""
    native = deepcopy(params)
    native.pop('first_center', None)  # Explicit grids take precedence.
    native.update(deramp=0, range_oversampling=1,
                  **{key: [params[key][0], FACTOR*params[key][1]] for key in ('window', 'search', 'stride')})
    return native


def native_grid(grid):
    return MatchingGrid(grid.starts_line, grid.starts_sample*FACTOR,
                        (grid.window[0], FACTOR*grid.window[1]), (grid.search[0], FACTOR*grid.search[1]),
                        (grid.stride[0], FACTOR*grid.stride[1]))


def native_gross(cfg, gross):
    """Gross offsets in range-2x native pixels: constant settings or a per-window array."""
    if gross is not None:
        scaled = np.array(gross, dtype=np.int32, copy=True)
        scaled[..., 1] *= FACTOR
        return cfg, scaled
    gross_cfg = deepcopy(cfg)
    if gross_cfg.get('gross_offsets', {}).get('mode') == 'constant':
        gross_cfg['gross_offsets']['constant']['range_px'] *= FACTOR
    return gross_cfg, None


def normalize(result, grid, params):
    """Convert range-2x native results and eligibility to original pixel units."""
    for name in ('total', 'gross', 'residual'):
        result[name] = np.array(result[name], copy=True)
        result[name][..., 1] /= FACTOR
    cov = np.array(result['covariance'], copy=True)
    sentinel = np.all(cov[..., :2] == 99, axis=-1)
    cov[..., 1] /= FACTOR**2
    cov[..., 2] /= FACTOR
    cov[sentinel] = [99, 99, 0]
    result['covariance'] = cov
    # Search-boundary exclusion in original pixel units.
    boundary = np.any(np.abs(result['residual']) >= np.asarray(grid.search)-1/params['oversampling'], axis=-1)
    result['search_boundary'] = boundary
    result['valid'] = (result['input_valid'] & result['finite'] & ~boundary &
                       (result['snr'] >= params['snr_threshold']) &
                       (result['correlation_peak'] >= params['correlation_threshold']))
    result['grid'] = grid
    return result


def range_worker(arguments):
    """Match one scheduler segment on range-2x inputs, reporting original units.

    `arguments` has the `_native_worker` form, with original-unit settings,
    grid and gross offsets and the prepared range-2x input paths.
    """
    from .ampcor import _native_worker
    args, kwargs = arguments
    reference, secondary, output, params, cfg, backend = args
    grid = kwargs['_grid']
    native = native_grid(grid)
    native_cfg, gross = native_gross(cfg, kwargs.get('_gross'))
    result = _native_worker(((reference, secondary, output, native_parameters(params), native_cfg, backend),
                             dict(kwargs, _grid=native, _gross=gross)))
    line, sample = result['grid'].centers
    np.testing.assert_array_equal(line, grid.centers[0])
    np.testing.assert_array_equal(sample/FACTOR + (1/FACTOR-1)/2, grid.centers[1])
    normalize(result, grid, params)
    result['effective_parameters']['range_oversampling'] = {
        'method': METHOD, 'factor_yx': [1, FACTOR],
        'reported_offsets_and_covariance': 'original reference SLC pixel units'}
    return result
