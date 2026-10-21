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
