"""Bound GDAL's process-local cache during sequential raster reads."""
from contextlib import contextmanager
from osgeo import gdal

RASTER_CACHE_BYTES = 64 * 1024**2


@contextmanager
def bounded_raster_cache():
    previous = gdal.GetCacheMax()
    gdal.SetCacheMax(min(previous, RASTER_CACHE_BYTES))
    try:
        yield
    finally:
        gdal.SetCacheMax(previous)


def require_native_mmap(dataset):
    """Reject CPU-incompatible raster layouts before the native fatal throw.

    The installed CPU controller ignores useMmap and unconditionally requests
    GDALGetVirtualMemAuto. Its bare C++ throw aborts the worker on failure.
    Probe that same mapping through GDAL's exception-safe Python interface.
    """
    mapped = None
    try:
        mapped = dataset.GetRasterBand(1).GetVirtualMemAutoArray()
        if mapped is None or not mapped.flags.c_contiguous:
            raise RuntimeError('a contiguous raster memory map is unavailable')
    except RuntimeError as exc:
        raise RuntimeError(f'Native CPU Ampcor cannot memory-map {dataset.GetDescription()}: {exc}. '
                           'Use an uncompressed stripped SLC with an external validity mask; '
                           'for a stack, rerun stack run to refresh its merged SLCs.') from exc
    finally:
        mapped = None  # Release the map while the GDAL dataset is still open.
