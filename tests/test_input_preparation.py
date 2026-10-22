"""Offline preparation integration, strict orbits, and real GDAL DEM checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from osgeo import gdal, osr
from shapely.geometry import box

from glacier_offsets import input_preparation as prep
from glacier_offsets.cli import main
from glacier_offsets.initialization import read_scene
from glacier_offsets.inputs import Burst
from glacier_offsets.prepared_inputs import prepared_paths
from glacier_offsets.products import checksum
from glacier_offsets.stack import scan_safes
from glacier_offsets.stack_config import load_stack_config
from test_initialization import manifest, orbit


def scene(path, time, bounds=(-100,-75,-99.99,-74.99)):
    with ZipFile(path,'w') as z:
        z.writestr('scene.SAFE/manifest.safe',manifest(time))
        w,s,e,n=bounds
        points=''.join(f'<geolocationGridPoint><latitude>{lat}</latitude><longitude>{lon}</longitude>'
                       '</geolocationGridPoint>' for lon,lat in ((w,s),(w,n),(e,n),(e,s)))
        z.writestr('scene.SAFE/annotation/s1a-iw2-slc-hh.xml',
                   f'<product><instrumentConfigurationId>17</instrumentConfigurationId>{points}</product>')
    return path


def make_dem(path, bounds, value=100, fmt='GTiff'):
    x0,y0,x1,y1=prep.rema_geometry(bounds)
    dx=dy=32.
    width,height=int(np.ceil((x1-x0)/dx))+2,int(np.ceil((y1-y0)/dy))+2
    ds=gdal.GetDriverByName(fmt).Create(str(path),width,height,1,gdal.GDT_Float32)
    srs=osr.SpatialReference();srs.ImportFromEPSG(3031)
    ds.SetProjection(srs.ExportToWkt());ds.SetGeoTransform((x0-dx,dx,0,y1+dy,0,-dy))
    ds.GetRasterBand(1).SetNoDataValue(-9999);ds.GetRasterBand(1).Fill(value)
    ds=None


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.safe=self.root/'safe';self.safe.mkdir()
        self.time=datetime(2020,1,12,4,35,56,123456)
        self.path=scene(self.safe/'scene.zip',self.time)
        self.bounds=[-100,-75,-99.99,-74.99]
        self.output=self.root/'prepared'
        self.args=argparse.Namespace(padding_km=0.,dem_source='REMA',rema_resolution=32,
            dem_format='GTiff',isce2_xml=False,overwrite_dem=False,min_dem_valid_fraction=.01,max_dem_pixels=100000)
        self.stream=io.StringIO()
        self.addCleanup(self.stream.close)

    def fake_download(self, **kwargs):
        self.assertTrue(kwargs['s1reader_compat'])
        self.assertEqual(kwargs['orbit_type'],'precise')
        self.assertTrue(kwargs['force_asf'])
        self.assertEqual(kwargs['orbit_dts'][0].microsecond,123456)
        path=Path(kwargs['save_dir'])/'S1A_OPER_AUX_POEORB_test.EOF'
        orbit(path,self.time-timedelta(hours=3))
        return [path]

    def downloader(self, function=None):
        package=ModuleType('eof');package.__path__=[]
        module=ModuleType('eof.download')
        def download_eofs(*,s1reader_compat=False,**kwargs):
            return (function or self.fake_download)(s1reader_compat=s1reader_compat,**kwargs)
        module.download_eofs=download_eofs
        return patch.dict(sys.modules,{'eof':package,'eof.download':module})

    def generator(self, path, bounds, resolution, fmt, max_pixels):
        make_dem(path,bounds,fmt=fmt)

    def test_metadata_is_authoritative_and_hidden_trees_are_pruned(self):
        self.assertEqual(prep.product_info(self.path)[1],self.time)
        hidden=self.safe/'.metadata';hidden.mkdir();scene(hidden/'hidden.zip',self.time)
        (self.safe/'._scene.zip').write_text('AppleDouble')
        nested=self.safe/'nested';nested.mkdir();scene(nested/'nested.zip',self.time+timedelta(days=6))
        self.assertEqual(prep.find_safes(self.safe,recursive=False)[0],[self.path])
        self.assertEqual(prep.find_safes(self.safe,recursive=True)[0],scan_safes(self.safe))
        self.assertEqual(len(scan_safes(self.safe)),2)

    def test_unpacked_safe_wins_over_its_identical_zip_in_shared_scanner(self):
        unpacked=self.safe/'scene.SAFE';unpacked.mkdir()
        (unpacked/'manifest.safe').write_text(manifest(self.time))
        self.assertEqual(scan_safes(self.safe),[unpacked])
        self.assertEqual(prep.find_safes(self.safe,True)[0],[unpacked])

    def test_orbit_cache_rejects_resorb_corrupt_and_insufficient_vectors(self):
        folder=self.root/'orbits';folder.mkdir();s=read_scene(self.path)
        orbit(folder/'res.EOF',self.time-timedelta(hours=3),kind='AUX_RESORB')
        (folder/'bad.EOF').write_text('not XML')
        orbit(folder/'too-late.EOF',self.time-timedelta(minutes=10))
        orbit(folder/'vectors.EOF',self.time-timedelta(hours=3),vector_start=self.time)
        self.assertIsNone(prep.cached_orbit(folder,s))
        old=orbit(folder/'old.EOF',self.time-timedelta(hours=3),created='2021-01-01T00:00:00')
        new=orbit(folder/'new.EOF',self.time-timedelta(hours=3),created='2022-01-01T00:00:00')
        self.assertEqual(prep.cached_orbit(folder,s),new)
        with self.downloader(lambda **kw:self.fail('downloaded despite valid cache')):
            self.assertEqual(Path(prep.get_orbit(self.path,folder,False,False)),new)

    def test_download_uses_strict_margins_and_atomic_promotion(self):
        folder=self.root/'orbits'
        with self.downloader(),contextlib.redirect_stdout(self.stream):
            found=Path(prep.get_orbit(self.path,folder,False,False))
        self.assertTrue(found.is_file())
        self.assertEqual(list(folder.glob('.orbit-*')),[])
        before=found.read_bytes()
        def invalid(**kwargs):
            orbit(Path(kwargs['save_dir'])/found.name,self.time-timedelta(hours=3),kind='AUX_RESORB')
        with self.downloader(invalid),self.assertRaises(ValueError):
            prep.get_orbit(self.path,folder,False,True)
        self.assertEqual(found.read_bytes(),before)

    def test_older_downloader_uses_padded_query_and_still_requires_one_strict_orbit(self):
        package=ModuleType('eof');package.__path__=[];module=ModuleType('eof.download')
        seen=[]
        def download_eofs(orbit_dts,missions,save_dir,orbit_type,force_asf):
            seen.extend(orbit_dts)
            orbit(Path(save_dir)/'precise.EOF',self.time-timedelta(hours=3))
        module.download_eofs=download_eofs
        with patch.dict(sys.modules,{'eof':package,'eof.download':module}):
            found=prep.get_orbit(self.path,self.root/'orbits',False,False)
        self.assertTrue(Path(found).is_file())
        self.assertLess(seen[0],self.time-timedelta(minutes=90))
        self.assertEqual(seen[1],read_scene(self.path).stop)

    def test_invalid_dem_caches_are_rebuilt_and_valid_cache_is_reused(self):
        folder=self.root/'dem';folder.mkdir()
        path=prep.dem_filename(folder,self.bounds,self.args);path.write_text('not a raster')
        with patch.object(prep,'generate_rema',side_effect=self.generator) as generate,contextlib.redirect_stdout(self.stream):
            self.assertEqual(prep.get_dem(self.bounds,folder,self.args),str(path))
            generate.assert_called_once()
        before=(checksum(path),path.stat().st_mtime_ns)
        with patch.object(prep,'generate_rema',side_effect=AssertionError('regenerated valid cache')),contextlib.redirect_stdout(self.stream):
            prep.get_dem(self.bounds,folder,self.args)
        self.assertEqual(before,(checksum(path),path.stat().st_mtime_ns))
        self.assertGreater(json.loads(Path(str(path)+'.preparation.json').read_text())['valid_fraction'],.9)

    def test_dem_nodata_extent_and_sparse_valid_data_are_rejected(self):
        path=self.root/'dem.tif';make_dem(path,self.bounds,value=-9999)
        with self.assertRaisesRegex(ValueError,'valid fraction'):
            prep.validate_dem(path,self.bounds)
        make_dem(path,self.bounds)
        with self.assertRaisesRegex(ValueError,'extent'):
            prep.validate_dem(path,[-101,-76,-99,-74])
        ds=gdal.Open(str(path),gdal.GA_Update);ds.GetRasterBand(1).Fill(-9999)
        ds.GetRasterBand(1).WriteArray(np.array([[20]],'float32'),1,1);ds=None
        with self.assertRaisesRegex(ValueError,'valid fraction'):
            prep.validate_dem(path,self.bounds,min_valid_fraction=.5)

    def test_failed_dem_refresh_preserves_valid_old_binary(self):
        folder=self.root/'dem';folder.mkdir();path=prep.dem_filename(folder,self.bounds,self.args)
        make_dem(path,self.bounds);before=checksum(path)
        self.args.overwrite_dem=True
        def invalid(path,bounds,*args):make_dem(path,bounds,value=-9999)
        with patch.object(prep,'generate_rema',side_effect=invalid),self.assertRaisesRegex(ValueError,'valid fraction'):
            prep.get_dem(self.bounds,folder,self.args)
        self.assertEqual(checksum(path),before)
        self.assertEqual(list(folder.glob('.dem-*')),[])

    def test_real_gdal_warp_preserves_rema_projection_and_restores_network_options(self):
        source=self.root/'source.tif';make_dem(source,self.bounds)
        original=gdal.Warp
        previous={'AWS_NO_SIGN_REQUEST':'NO','GDAL_HTTP_MAX_RETRY':'9','GDAL_HTTP_RETRY_DELAY':'7'}
        expected={'AWS_NO_SIGN_REQUEST':'YES','GDAL_HTTP_MAX_RETRY':'3','GDAL_HTTP_RETRY_DELAY':'2'}
        def worker_options():
            return {key:gdal.GetConfigOption(key) for key in expected}
        def warp(dst,remote,options):
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker_options).result(),expected)
            return original(dst,str(source),options=options)
        with gdal.config_options(previous,thread_local=False):
            with patch('osgeo.gdal.Warp',side_effect=warp):
                prep.generate_rema(self.root/'warped.tif',self.bounds,32,'GTiff',100000)
            self.assertEqual(worker_options(),previous)
        report=prep.validate_dem(self.root/'warped.tif',self.bounds)
        self.assertIn('3031',report['crs'])
        self.assertGreater(report['valid_fraction'],.5)

    def test_failed_rema_warp_restores_worker_network_options(self):
        previous={'AWS_NO_SIGN_REQUEST':'NO','GDAL_HTTP_MAX_RETRY':'9','GDAL_HTTP_RETRY_DELAY':'7'}
        expected={'AWS_NO_SIGN_REQUEST':'YES','GDAL_HTTP_MAX_RETRY':'3','GDAL_HTTP_RETRY_DELAY':'2'}
        def worker_options():
            return {key:gdal.GetConfigOption(key) for key in expected}
        def fail(*args,**kwargs):
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker_options).result(),expected)
            raise RuntimeError('simulated read failure')
        with gdal.config_options(previous,thread_local=False):
            with patch('osgeo.gdal.Warp',side_effect=fail),self.assertRaisesRegex(RuntimeError,'simulated read failure'):
                prep.generate_rema(self.root/'failed.tif',self.bounds,32,'GTiff',100000)
            with ThreadPoolExecutor(max_workers=1) as pool:
                self.assertEqual(pool.submit(worker_options).result(),previous)

    def test_non_rema_delegates_vertical_conversion_and_validates_result(self):
        self.args.dem_source='COP'
        package=ModuleType('sardem');package.__path__=[]
        module=ModuleType('sardem.dem')
        def download(**kwargs):
            self.assertFalse(kwargs['keep_egm'])
            self.assertFalse(kwargs['make_isce_xml'])
            make_dem(Path(kwargs['output_name']),kwargs['bbox'])
        module.main=download
        with patch.dict(sys.modules,{'sardem':package,'sardem.dem':module}),contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        self.assertTrue(Path(output).is_file())

    def local_source(self, value=123):
        path=self.root/'local DEM.tif'
        x0,y0,x1,y1=prep.rema_geometry(self.bounds)
        # Offset origin deliberately: cropping must preserve the source grid.
        gt=(np.floor(x0/100)*100-83,100.,0.,np.ceil(y1/100)*100+127,0.,-100.)
        width=int(np.ceil((x1-gt[0])/100))+3;height=int(np.ceil((gt[3]-y0)/100))+3
        ds=gdal.GetDriverByName('GTiff').Create(str(path),width,height,1,gdal.GDT_Float32)
        srs=osr.SpatialReference();srs.ImportFromEPSG(3031)
        ds.SetProjection(srs.ExportToWkt());ds.SetGeoTransform(gt)
        ds.GetRasterBand(1).SetNoDataValue(-32767);ds.GetRasterBand(1).Fill(value);ds=None
        return path

    def test_local_raster_crops_at_native_resolution_and_records_source(self):
        source=self.local_source();before=checksum(source)
        self.args.dem_source=str(source)
        with contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        ds=gdal.Open(output);gt=ds.GetGeoTransform();original=gdal.Open(str(source)).GetGeoTransform()
        self.assertEqual(gt[1],100.);self.assertEqual(gt[5],-100.)
        self.assertAlmostEqual((gt[0]-original[0])/100,round((gt[0]-original[0])/100))
        self.assertAlmostEqual((gt[3]-original[3])/100,round((gt[3]-original[3])/100))
        np.testing.assert_array_equal(ds.ReadAsArray(),123)
        report=json.loads(Path(output+'.preparation.json').read_text())
        self.assertEqual(report['valid_fraction'],1.)
        self.assertEqual(report['local_source']['files'][0]['sha256'],before)
        self.assertEqual(report['local_source']['path'],str(source))
        self.assertEqual(report['local_source']['nodata'],-32767)
        self.assertIn('no vertical conversion',report['height_reference_basis'])
        self.assertEqual(checksum(source),before)
        self.assertEqual(ds.GetRasterBand(1).GetUnitType(),'m')

    def test_local_raster_cache_reuse_and_changed_source_invalidation(self):
        source=self.local_source();self.args.dem_source=str(source)
        with contextlib.redirect_stdout(self.stream):
            first=prep.get_dem(self.bounds,self.root/'dem',self.args)
            before=(checksum(first),Path(first).stat().st_mtime_ns)
            with patch.object(prep,'generate_local_dem',side_effect=AssertionError('regenerated valid cache')):
                self.assertEqual(prep.get_dem(self.bounds,self.root/'dem',self.args),first)
            self.assertEqual(before,(checksum(first),Path(first).stat().st_mtime_ns))
            ds=gdal.Open(str(source),gdal.GA_Update);ds.GetRasterBand(1).Fill(456);ds=None
            second=prep.get_dem(self.bounds,self.root/'dem',self.args)
        self.assertNotEqual(first,second)
        np.testing.assert_array_equal(gdal.Open(second).ReadAsArray(),456)
        self.assertEqual(checksum(first),before[0])

    def test_local_raster_respects_nodata_and_valid_fraction(self):
        source=self.local_source();self.args.dem_source=str(source)
        ds=gdal.Open(str(source),gdal.GA_Update);ds.GetRasterBand(1).WriteArray(np.array([[-32767]],'float32'),3,3);ds=None
        with contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        ds=gdal.Open(output);data=ds.ReadAsArray()
        self.assertEqual(ds.GetRasterBand(1).GetNoDataValue(),-9999)
        self.assertTrue(np.any(data==-9999));self.assertFalse(np.any(data==-32767))
        self.args.min_dem_valid_fraction=1.
        with contextlib.redirect_stdout(self.stream),self.assertRaisesRegex(ValueError,'valid fraction'):
            prep.get_dem(self.bounds,self.root/'dem',self.args)
        self.assertTrue(Path(output).is_file())

    def test_local_raster_extent_pixel_limit_and_metadata_checked_before_warp(self):
        source=self.local_source();self.args.dem_source=str(source)
        with patch('osgeo.gdal.Warp',side_effect=AssertionError('warped invalid request')),contextlib.redirect_stdout(self.stream):
            with self.assertRaisesRegex(ValueError,'extent'):
                prep.get_dem([-101,-76,-99,-74],self.root/'dem',self.args)
            self.args.max_dem_pixels=1
            with self.assertRaisesRegex(ValueError,'limit'):
                prep.get_dem(self.bounds,self.root/'dem',self.args)
        source.write_text('not a raster')
        with contextlib.redirect_stdout(self.stream),self.assertRaises(RuntimeError):
            prep.local_dem_source(source)

    def test_local_raster_envi_uses_geographic_crs_and_retains_header(self):
        source=self.local_source();self.args.dem_source=str(source);self.args.dem_format='ENVI'
        with contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        ds=gdal.Open(output);srs=osr.SpatialReference(wkt=ds.GetProjection())
        self.assertEqual(srs.GetAuthorityCode(None),'4326')
        self.assertGreater(prep.validate_dem(output,self.bounds)['valid_fraction'],.9)
        self.assertTrue(any(p.endswith('.hdr') for p in ds.GetFileList()))

    def test_local_raster_cli_records_manifest_without_downloads(self):
        source=self.local_source()
        orbits=self.output/'orbits';orbits.mkdir(parents=True)
        orbit(orbits/'cached.EOF',self.time-timedelta(hours=3))
        with patch.object(prep,'generate_rema',side_effect=AssertionError('downloaded REMA')), \
             self.downloader(lambda **kw:self.fail('downloaded a cached orbit')),contextlib.redirect_stdout(self.stream):
            result=main(['prepare-inputs',str(self.safe),'-o',str(self.output),
                         '--dem-source',str(source),'--padding-km','0','--min-dem-valid-fraction','1'])
        self.assertEqual(result,0)
        m=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(m['dem_source'],str(source));self.assertEqual(m['status'],'complete')
        self.assertEqual(m['dems'][0]['validation']['local_source']['path'],str(source))
        self.assertEqual(prepared_paths(self.output/'manifest.json')[2],m['dem'])
        with contextlib.redirect_stderr(self.stream),self.assertRaises(SystemExit):
            main(['prepare-inputs',str(self.safe),'--dem-source',str(source)+'.missing'])

    def test_invalid_local_raster_records_failure_before_orbit_download(self):
        source=self.local_source()
        ds=gdal.Open(str(source),gdal.GA_Update);ds.GetRasterBand(1).SetUnitType('ft');ds=None
        with patch.object(prep,'get_orbit',side_effect=AssertionError('started orbit processing')), \
             contextlib.redirect_stdout(self.stream),contextlib.redirect_stderr(self.stream):
            result=main(['prepare-inputs',str(self.safe),'-o',str(self.output),'--dem-source',str(source)])
        self.assertEqual(result,1)
        m=json.loads((self.output/'manifest.json').read_text())
        self.assertEqual(m['status'],'failed');self.assertIn('metres',m['error'])
        with self.assertRaisesRegex(ValueError,'failures'):
            prepared_paths(self.output/'manifest.json')

    def test_envi_header_survives_and_xml_names_final_binary(self):
        self.args.dem_format='ENVI'
        self.args.isce2_xml=True
        package=ModuleType('sardem');package.__path__=[]
        module=ModuleType('sardem.utils')
        def xml(path,keep_egm):
            self.assertFalse(keep_egm)
            self.assertIsNotNone(gdal.Open(path))
            Path(path+'.xml').write_text(f'<image><filename>{path}</filename></image>')
        module.gdal2isce_xml=xml
        with patch.dict(sys.modules,{'sardem':package,'sardem.utils':module,'isce':ModuleType('isce'),'isceobj':ModuleType('isceobj')}), \
             patch.object(prep,'generate_rema',side_effect=self.generator),contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        import xml.etree.ElementTree as ET
        self.assertEqual(ET.parse(output+'.xml').findtext('filename'),output)
        self.assertIsNotNone(gdal.Open(output))

    def test_envi_without_isce2_xml_uses_only_gdal(self):
        self.args.dem_format='ENVI'
        with patch.dict(sys.modules,{'isce':None,'isceobj':None}), \
             patch.object(prep,'generate_rema',side_effect=self.generator),contextlib.redirect_stdout(self.stream):
            output=prep.get_dem(self.bounds,self.root/'dem',self.args)
        self.assertIsNotNone(gdal.Open(output))
        self.assertFalse(Path(output+'.xml').exists())

    def test_prepared_manifest_relative_paths_and_explicit_dem_override(self):
        folder=self.root/'handoff';folder.mkdir()
        path=folder/'manifest.json'
        m={'method':'s1-input-preparation-v1','status':'complete','failed_steps':0,'scenes':[{'status':'ok'}],
           'safe_directory':'../safe','orbit_directory':'orbits','dem':'dem/one.tif'}
        path.write_text(json.dumps(m))
        safe,orbits,dem=prepared_paths(path)
        self.assertEqual(safe,str(self.safe))
        self.assertEqual(orbits,str(folder/'orbits'))
        self.assertEqual(dem,str(folder/'dem/one.tif'))
        explicit=self.root/'override.tif'
        self.assertEqual(prepared_paths(path,dem=explicit)[2],str(explicit))
        m['dem']=None;path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError,'shared DEM'):
            prepared_paths(path)

    def test_fail_fast_does_not_download_dem_after_orbit_failure(self):
        with self.downloader(lambda **kw: (_ for _ in ()).throw(RuntimeError('offline'))), \
             patch.object(prep,'generate_rema',side_effect=AssertionError('downloaded DEM after failure')), \
             contextlib.redirect_stdout(self.stream),contextlib.redirect_stderr(self.stream):
            result=main(['prepare-inputs',str(self.safe),'-o',str(self.output),'--fail-fast'])
        self.assertEqual(result,1)
        self.assertIsNone(json.loads((self.output/'manifest.json').read_text())['dem'])

    def test_interrupted_refresh_does_not_leave_prior_success_current(self):
        with self.downloader(),patch.object(prep,'generate_rema',side_effect=self.generator), \
             contextlib.redirect_stdout(self.stream):
            self.assertEqual(main(['prepare-inputs',str(self.safe),'-o',str(self.output),'--padding-km','0']),0)
        m=json.loads((self.output/'manifest.json').read_text())
        before=checksum(m['dem'])
        with patch.object(prep,'get_orbit',side_effect=KeyboardInterrupt), \
             contextlib.redirect_stdout(self.stream),contextlib.redirect_stderr(self.stream):
            self.assertEqual(main(['prepare-inputs',str(self.safe),'-o',str(self.output),'--refresh-orbits']),130)
        self.assertEqual(json.loads((self.output/'manifest.json').read_text())['status'],'running')
        self.assertEqual(checksum(m['dem']),before)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            prepared_paths(self.output/'manifest.json')

    def test_cli_preparation_handoff_and_init_covers_complete_bursts(self):
        second=scene(self.safe/'second.zip',self.time+timedelta(days=6))
        def download(**kwargs):
            t=kwargs['orbit_dts'][0]
            path=Path(kwargs['save_dir'])/f'S1A_OPER_AUX_POEORB_{t:%Y%m%d}.EOF'
            orbit(path,t-timedelta(hours=3));return [path]
        with self.downloader(download),patch.object(prep,'generate_rema',side_effect=self.generator), \
             contextlib.redirect_stdout(self.stream):
            self.assertEqual(main(['prepare-inputs',str(self.safe),'-o',str(self.output),
                                   '--padding-km','0']),0)
        path=self.output/'manifest.json';m=json.loads(path.read_text())
        self.assertEqual(m['failed_steps'],0);self.assertEqual(len(m['scenes']),2)
        self.assertEqual(m['dems'][0]['validation']['source'],'REMA')
        self.assertEqual(prepared_paths(path),(str(self.safe),str(self.output/'orbits'),m['dem']))
        def inventory(path,*args):
            native=SimpleNamespace(polarization='HH',orbit_direction='DESCENDING',border=[box(*self.bounds)],
                                   as_isce3_radargrid=lambda:SimpleNamespace(prf=500.,range_pixel_spacing=2.))
            return [Burst('burst',2,read_scene(path).start,native)]
        config=self.root/'stack.yaml'
        with patch('glacier_offsets.stack.inventory',side_effect=inventory),contextlib.redirect_stdout(self.stream):
            self.assertEqual(main(['stack','init','--prepared-inputs',str(path),'--polarization','HH',
                '--burst-id','burst','--output',str(config),'--output-directory',str(self.root/'work')]),0)
        cfg=load_stack_config(config)
        self.assertEqual(cfg['inputs']['dem'],m['dem']);self.assertEqual(cfg['geocoding']['epsg'],3031)
        self.assertFalse((self.root/'work').exists())
        # A larger selected burst must fail before emitting a misleading config.
        def larger(path,*args):
            result=inventory(path,*args);result[0].native.border=[box(-101,-76,-99,-74)];return result
        with patch('glacier_offsets.stack.inventory',side_effect=larger),contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(SystemExit):
            main(['stack','init','--prepared-inputs',str(path),'--polarization','HH','--burst-id','burst',
                  '--output',str(self.root/'bad.yaml')])
        self.assertFalse((self.root/'bad.yaml').exists())

    def test_partial_failures_and_invalid_zip_cannot_be_consumed_as_success(self):
        (self.safe/'broken.zip').write_text('incomplete download')
        def invalid(path,bounds,*args):make_dem(path,bounds,value=-9999)
        with self.downloader(),patch.object(prep,'generate_rema',side_effect=invalid), \
             contextlib.redirect_stdout(self.stream),contextlib.redirect_stderr(self.stream):
            self.assertEqual(main(['prepare-inputs',str(self.safe),'-o',str(self.output),'--padding-km','0']),1)
        path=self.output/'manifest.json';m=json.loads(path.read_text())
        self.assertGreater(m['failed_steps'],0)
        self.assertEqual(m['scenes'][0]['status'],'error')
        with self.assertRaisesRegex(ValueError,'failures'):
            prepared_paths(path)

    def test_invalid_limits_fail_before_any_download_or_output_mutation(self):
        for args in (['--min-dem-valid-fraction','0'],['--padding-km','nan'],['--max-dem-pixels','0']):
            with contextlib.redirect_stderr(self.stream),self.assertRaises(SystemExit) as error:
                main(['prepare-inputs',str(self.safe),'-o',str(self.output),*args])
            self.assertEqual(error.exception.code,2)
        self.assertFalse(self.output.exists())


if __name__=='__main__':
    unittest.main()
