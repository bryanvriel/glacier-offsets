"""Bounded raster reads and recycled native workers retain scientific semantics."""
from collections import Counter
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from osgeo import gdal
from scipy.ndimage import binary_opening, binary_propagation, map_coordinates

from glacier_offsets.ampcor import native_match, _native_executor, NATIVE_CALLS_PER_WORKER
from glacier_offsets.config import DEFAULT_MATCHING
from glacier_offsets.geocode import sample_geometry
from glacier_offsets.matching_grid import build_grid, valid_windows
from glacier_offsets.raster_io import bounded_raster_cache, RASTER_CACHE_BYTES
from test_synthetic_offsets import write_slc


class MemoryBoundsTests(unittest.TestCase):
    def test_geometry_reads_windows_and_preserves_bilinear_and_mask_dtype(self):
        with tempfile.TemporaryDirectory() as d:
            rng=np.random.default_rng(1)
            for kind in [gdal.GDT_Float32,gdal.GDT_Float64,gdal.GDT_Byte]:
                p=Path(d)/f'geometry-{kind}.tif'
                data=(rng.random((180,230))*3).astype('uint8' if kind==gdal.GDT_Byte else 'float32' if kind==gdal.GDT_Float32 else 'float64')
                ds=gdal.GetDriverByName('GTiff').Create(str(p),230,180,1,kind)
                ds.GetRasterBand(1).WriteArray(data);ds=None
                line=np.array([[0.,63.5,64.,127.75],[179.,-1.,np.nan,180.]])
                sample=np.array([[0.,40.25,100.5,229.],[30.,40.,40.,40.]])
                expected=map_coordinates(data,np.array([line.ravel(),sample.ravel()]),order=1,mode='constant',cval=np.nan,prefilter=False).reshape(line.shape)
                real=gdal.Open(str(p));reads=[]
                class WindowOnly:
                    RasterXSize=real.RasterXSize;RasterYSize=real.RasterYSize
                    def GetRasterBand(self,n):return real.GetRasterBand(n)
                    def ReadAsArray(self,*args,**kwargs):
                        reads.append(args);return real.ReadAsArray(*args,**kwargs)
                with patch('glacier_offsets.geocode.gdal.Open',return_value=WindowOnly()):
                    actual=sample_geometry(p,line,sample)
                np.testing.assert_array_equal(actual,expected)
                self.assertEqual(actual.dtype,expected.dtype)
                self.assertTrue(reads)
                self.assertTrue(all(len(r)==4 and r[3]<=33 for r in reads))

    def test_strip_validity_preserves_global_zero_connectivity_and_variable_gross(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);ref=np.ones((256,300),dtype='complex64');sec=ref.copy()
            # One 3x3 seed joins a narrow zero component crossing both read
            # boundaries; isolated quantized zero remains a valid observation.
            ref[10:14,80:84]=0;ref[12:210,81]=0;ref[90,180]=0
            sec[128:132,140:144]=0;sec[230,240]=np.nan+1j
            write_slc(d/'ref.tif',ref);write_slc(d/'sec.tif',sec)
            grid=build_grid(ref.shape,sec.shape,(32,40),(8,10),(20,30))
            gross=np.random.default_rng(5).integers(-3,4,(*grid.shape,2))
            expected=np.ones(grid.shape,bool)
            for side,data in enumerate([ref,sec]):
                zero=(data.real==0)&(data.imag==0)
                seeds=binary_opening(zero,structure=np.ones((3,3),bool))
                seeds[0,:]|=zero[0,:];seeds[-1,:]|=zero[-1,:]
                seeds[:,0]|=zero[:,0];seeds[:,-1]|=zero[:,-1]
                bad=~np.isfinite(data)|binary_propagation(seeds,mask=zero)
                for i,y in enumerate(grid.starts_line):
                    for j,x in enumerate(grid.starts_sample):
                        if side:
                            yy=int(y)+int(gross[i,j,0])-8;xx=int(x)+int(gross[i,j,1])-10;h,w=48,60
                        else:yy,xx,h,w=int(y),int(x),32,40
                        expected[i,j]&=yy>=0 and xx>=0 and yy+h<=256 and xx+w<=300 and not bad[yy:yy+h,xx:xx+w].any()
            reads=[]
            class StripOnly:
                def __init__(self,ds):self.ds=ds;self.RasterXSize=ds.RasterXSize;self.RasterYSize=ds.RasterYSize
                def GetRasterBand(self,n):return self.ds.GetRasterBand(n)
                def ReadAsArray(self,*args):
                    reads.append(args);return self.ds.ReadAsArray(*args)
            actual=valid_windows(StripOnly(gdal.Open(str(d/'ref.tif'))),StripOnly(gdal.Open(str(d/'sec.tif'))),grid,gross)
            np.testing.assert_array_equal(actual,expected)
            self.assertTrue(all(len(r)==4 and r[3]<=64 for r in reads))

    def test_cache_limit_respects_smaller_setting_and_restores_on_failure(self):
        previous=gdal.GetCacheMax()
        try:
            gdal.SetCacheMax(2*RASTER_CACHE_BYTES)
            with self.assertRaises(RuntimeError):
                with bounded_raster_cache():
                    self.assertEqual(gdal.GetCacheMax(),RASTER_CACHE_BYTES)
                    raise RuntimeError('interrupted read')
            self.assertEqual(gdal.GetCacheMax(),2*RASTER_CACHE_BYTES)
            gdal.SetCacheMax(1024)
            with bounded_raster_cache():self.assertEqual(gdal.GetCacheMax(),1024)
        finally:gdal.SetCacheMax(previous)

    def test_sequential_worker_recycling_and_native_scientific_equality(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);rng=np.random.default_rng(24)
            ref=rng.uniform(.1,1,(128,128)).astype('complex64')
            write_slc(d/'ref.tif',ref);write_slc(d/'sec.tif',ref)
            params=DEFAULT_MATCHING|{'window':[32,32],'search':[8,8],'stride':[32,32],
                                    'zoom':8,'batch':[1,2]}
            grid=build_grid(ref.shape,ref.shape,(32,32),(8,8),(32,32))
            pids=[];first=None
            with _native_executor() as worker:
                for i in range(NATIVE_CALLS_PER_WORKER+1):
                    result=native_match(d/'ref.tif',d/'sec.tif',d/str(i),params,{'gross_offsets':{'mode':'zero'}},
                                        'cpu',_grid=grid,_valid_input=np.ones(grid.shape,bool),_worker=worker)
                    pids.append(result['effective_parameters']['worker_pid'])
                    self.assertGreater(result['effective_parameters']['worker_peak_rss_bytes'],0)
                    if first is None:first=result
                    for field in ('total','gross','residual','valid','input_valid'):
                        np.testing.assert_array_equal(first[field],result[field])
                    for field in ('snr','correlation_peak','covariance'):
                        np.testing.assert_allclose(first[field],result[field],rtol=1e-5,atol=1e-6,equal_nan=True)
            self.assertNotEqual(pids[0],pids[-1])
            self.assertLessEqual(max(Counter(pids).values()),NATIVE_CALLS_PER_WORKER)

    def test_no_worker_when_all_priors_are_missing(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);write_slc(d/'r.tif',np.ones((128,128),'complex64'))
            grid=build_grid((128,128),(128,128),(32,32),(8,8),(32,32))
            prior={'grid':grid,'integer_px':np.full((*grid.shape,2),np.iinfo(np.int32).min,dtype='int32'),
                   'valid':np.zeros(grid.shape,bool),'rejection_flags':np.ones(grid.shape,'uint8')}
            with patch('glacier_offsets.ampcor._native_executor',side_effect=AssertionError('unneeded worker')):
                result=native_match(d/'r.tif',d/'r.tif',d/'out',DEFAULT_MATCHING,
                                    {'gross_offsets':{'mode':'velocity'}},'cpu',prior=prior)
            self.assertFalse(result['valid'].any())

    def test_fftw_cache_round_trip_and_atomic_write(self):
        from glacier_offsets.fftw_cache import restore, save
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wisdom'
            lib=restore(path)
            if lib is None:self.skipTest('No shared FFTW library in this native environment')
            save(lib,path)
            self.assertTrue(path.is_file())
            self.assertFalse(path.with_suffix('.tmp').exists())
            self.assertIsNotNone(restore(path))
            path.write_text('invalid wisdom')
            with self.assertRaisesRegex(RuntimeError,'FFTW plan cache'):restore(path)

    def test_native_worker_exception_reaches_parent(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);p=d/'real.tif'
            ds=gdal.GetDriverByName('GTiff').Create(str(p),64,64,1,gdal.GDT_Float32);ds=None
            with self.assertRaisesRegex(ValueError,'complex inputs'):
                native_match(p,p,d/'native',DEFAULT_MATCHING,{'gross_offsets':{'mode':'zero'}},'cpu')

    @unittest.skipUnless(sys.platform=='darwin','macOS lacks the GDAL fallback virtual memory implementation')
    def test_unmappable_slc_reports_worker_error_instead_of_native_abort(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); path=d/'compressed.tif'
            ds=gdal.GetDriverByName('GTiff').Create(str(path),128,128,1,gdal.GDT_CFloat32,
                                                   options=['COMPRESS=DEFLATE'])
            ds.GetRasterBand(1).WriteArray(np.ones((128,128),'complex64')); ds=None
            params=DEFAULT_MATCHING|{'window':[32,32],'search':[8,8],'stride':[32,32],'zoom':8}
            with self.assertRaisesRegex(RuntimeError,'cannot memory-map'):
                native_match(path,path,d/'native',params,{'gross_offsets':{'mode':'zero'}},'cpu')
