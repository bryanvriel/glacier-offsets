"""Offline ASF planning, real TIFF validation, restarts and preparation contract."""
import argparse
import contextlib
import copy
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from osgeo import gdal
from shapely.geometry import Polygon, box, mapping

from glacier_offsets import burst_downloads as bd, input_preparation as prep
from glacier_offsets.aoi import AOI, load_aoi
from glacier_offsets.cli import main
from glacier_offsets.inputs import inventory
from glacier_offsets.products import checksum
from glacier_offsets.prepared_inputs import prepared_paths
from glacier_offsets.stack import scan_safes, plan_stack
from glacier_offsets.stack_config import load_stack_config
from test_initialization import manifest, orbit
from test_input_preparation import make_dem


TIME = datetime(2018, 1, 8, 8, 2, 38)


def product(burst_id=10, swath='IW2', day=8, mission='A', orbit_number=20060,
            geometry=None, suffix='AAAA', **changes):
    start = TIME.replace(day=day) + timedelta(seconds=burst_id-10)
    name = f'S1_{burst_id:06d}_{swath}_{start:%Y%m%dT%H%M%S}_HH_{mission}{suffix}-BURST'
    props = dict(platform=f'SENTINEL-1{mission}', beamModeType='IW', polarization='HH',
                 pathNumber=38, orbit=orbit_number, startTime=start.isoformat()+'Z',
                 stopTime=(start+timedelta(seconds=3)).isoformat()+'Z', fileID=name,
                 flightDirection='DESCENDING', url='https://example.com/data',
                 additionalUrls=['https://example.com/meta'],
                 burst={'subswath': swath, 'relativeBurstID': burst_id, 'burstIndex': burst_id-10})
    props.update(changes)
    parent_start=TIME.replace(day=day)
    parent=f'S1{mission}_IW_SLC__1SSH_{parent_start:%Y%m%dT%H%M%S}_{parent_start+timedelta(seconds=30):%Y%m%dT%H%M%S}_{orbit_number:06d}_0222F2_2151-SLC'
    return SimpleNamespace(properties=props, geometry=mapping(geometry if geometry is not None else box(0,0,1,1)),
                           umm={'InputGranules': [parent]})


def synthetic_safe(scratch, group, request):
    path = Path(scratch) / 'synthetic.SAFE'
    (path/'annotation').mkdir(parents=True)
    (path/'measurement').mkdir()
    start = datetime.fromisoformat(min(r['start'] for r in group['bursts']))
    (path/'manifest.safe').write_text(manifest(start, mission=group['mission'][-1]))
    for swath in ('IW1','IW2','IW3'):
        count = sum(r['swath'] == swath for r in group['bursts'])
        points = ''.join(f'<geolocationGridPoint><longitude>{x}</longitude><latitude>{y}</latitude>'
                         '</geolocationGridPoint>' for x,y in ((0,0),(0,1),(1,1),(1,0))) if count else ''
        bursts = ''.join(f'<burst><azimuthTime>{(start+timedelta(seconds=i)).isoformat()}</azimuthTime></burst>'
                         for i in range(count))
        stem = f's1{group["mission"][-1].lower()}-{swath.lower()}-slc-hh-test'
        (path/'annotation'/f'{stem}.xml').write_text(f'''<product><adsHeader><swath>{swath}</swath>
          <polarisation>HH</polarisation></adsHeader><swathTiming><linesPerBurst>4</linesPerBurst>
          <samplesPerBurst>5</samplesPerBurst><burstList count="{count}">{bursts}</burstList></swathTiming>
          <generalAnnotation><productInformation><radarFrequency>5.4e9</radarFrequency>
          <rangeSamplingRate>6.4e7</rangeSamplingRate></productInformation></generalAnnotation>
          <imageAnnotation><imageInformation><numberOfSamples>5</numberOfSamples>
          <azimuthTimeInterval>0.002</azimuthTimeInterval><slantRangeTime>0.005</slantRangeTime>
          <ascendingNodeTime>{start.isoformat()}</ascendingNodeTime>
          <numberOfLines>{4*count}</numberOfLines></imageInformation></imageAnnotation>{points}</product>''')
        if count:
            ds = gdal.GetDriverByName('GTiff').Create(str(path/'measurement'/f'{stem}.tiff'),5,4*count,1,gdal.GDT_CInt16)
            data = np.full((4*count,5),3+4j, 'complex64'); data[1,1]=0
            ds.GetRasterBand(1).WriteArray(data); ds=None
    return path


class BurstDownloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.asf, _, cls.safe_class, create = bd.dependencies()
        except ImportError:
            raise unittest.SkipTest('Optional burst-download extra unavailable')
        cls.create = staticmethod(create)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.args = argparse.Namespace(aoi=None,aoi_layer=None,bbox=[0,0,1,1],start_date='2018-01-08',
            end_date='2018-01-20',relative_orbit=38,polarization='HH',subswaths=[2])
        self.request, self.aoi = bd.request_from_args(self.args)

    def plan(self, products, context=None, request=None, aoi=None):
        service = SimpleNamespace(constants=self.asf.constants)
        def search(**kw):
            return context if 'fullBurstID' in kw and context is not None else products
        service.search = unittest.mock.Mock(side_effect=search)
        result = bd.plan_downloads(request or self.request, aoi or self.aoi, service, self.safe_class, self.create)
        return result, service

    def download(self, groups, converter=None, request=None):
        def convert(**kw):
            self.assertTrue(kw['all_anns'])
            group = next(g for g in groups if [r['granule'] for r in g['bursts']] == kw['granules'])
            return synthetic_safe(kw['work_dir'], group, request or self.request)
        with patch('burst2safe.auth.check_earthdata_credentials',return_value='netrc'):
            return bd.download(request or self.request, copy.deepcopy(groups), self.root/'collection', converter or convert)

    def test_inclusive_days_filters_and_deduplication(self):
        a=product(); end=product(day=20,orbit_number=20235)
        wrong=[product(pathNumber=39),product(polarization='VV'),product(day=21),product(swath='IW1')]
        groups, service=self.plan([a,a,end,*wrong])
        self.assertEqual(len(groups),2)
        self.assertTrue(all(not g['errors'] for g in groups))
        self.assertEqual(service.search.call_args.kwargs['start'],'2018-01-08T00:00:00Z')
        self.assertEqual(service.search.call_args.kwargs['end'],'2018-01-21T00:00:00Z')

    def test_holes_touching_and_contiguous_context(self):
        polygon=Polygon([(0,0),(4,0),(4,4),(0,4)],holes=[[(1,1),(3,1),(3,3),(1,3)]])
        from pyproj import CRS
        aoi=AOI(polygon,CRS(4326),'{}')
        first=product(10,geometry=box(0,.1,.5,.5)); last=product(12,geometry=box(3.5,.1,4,.5))
        middle=product(11,geometry=box(1.5,1.5,2,2)); touch=product(13,geometry=box(4,0,5,1))
        groups,_=self.plan([first,last,middle,touch],context=[first,middle,last],aoi=aoi)
        self.assertEqual(len(groups[0]['requested']),2)
        self.assertEqual(groups[0]['context'],[middle.properties['fileID']])
        self.assertFalse(groups[0]['errors'])
        groups,service=self.plan([first,last],context=[first,middle,last],aoi=aoi)
        self.assertEqual(service.search.call_count,2)
        self.assertEqual(groups[0]['context'],[middle.properties['fileID']])

    def test_missing_interior_duplicate_products_and_incompatible_swaths(self):
        a,c=product(10),product(12)
        groups,_=self.plan([a,c],context=[a,c]);self.assertIn('missing interior',groups[0]['errors'][0])
        groups,_=self.plan([a,product(10,suffix='BBBB')]);self.assertIn('ambiguous duplicate',groups[0]['errors'][0])
        request=self.request|{'subswaths':[1,3]}
        groups,_=self.plan([product(10,swath='IW1'),product(15,swath='IW3')],request=request)
        self.assertIn('do not overlap',groups[0]['errors'][0])

    def test_same_day_different_groups_rejected_and_partial_dates_allowed(self):
        groups,_=self.plan([product(),product(mission='B',orbit_number=9164)])
        self.assertTrue(all(g['errors'] for g in groups))
        groups,_=self.plan([product(),product(11,day=14,mission='B',orbit_number=9164)])
        self.assertTrue(all(not g['errors'] for g in groups))

    def test_antimeridian_split_and_projected_aoi(self):
        args=copy.copy(self.args);args.bbox=[179,-1,-179,1]
        request,aoi=bd.request_from_args(args)
        groups,service=self.plan([product(geometry=box(179.1,0,179.5,.5))],request=request,aoi=aoi)
        self.assertEqual(service.search.call_count,2);self.assertEqual(len(groups),1)
        from shapely.ops import transform
        from pyproj import CRS, Transformer
        geometry=transform(Transformer.from_crs(4326,3031,always_xy=True).transform,box(0,-75,1,-74))
        projected=AOI(geometry,CRS(3031),'{}')
        groups,_=self.plan([product(geometry=box(.1,-74.9,.8,-74.2))],aoi=projected)
        self.assertEqual(len(groups),1)

    def test_dry_run_creates_no_files_and_empty_search_is_failure(self):
        argv=['download-bursts','--bbox','0','0','1','1','--start-date','2018-01-08',
              '--end-date','2018-01-20','--relative-orbit','38','--polarization','HH','--subswaths','2',
              '-o',str(self.root/'absent'),'--dry-run']
        service=SimpleNamespace(constants=self.asf.constants,search=lambda **kw:[product()])
        with patch.object(bd,'dependencies',return_value=(service,None,self.safe_class,self.create)),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv),0)
            service.search=lambda **kw:[]
            self.assertEqual(main(argv),2)
        self.assertFalse((self.root/'absent').exists())

    def test_optional_imports_authentication_guard_and_collection_lock(self):
        with patch.dict('sys.modules',{'asf_search':None}):
            with self.assertRaisesRegex(ImportError,'burst-download'):
                bd.dependencies()
        groups,_=self.plan([product()])
        with patch('burst2safe.auth.check_earthdata_credentials',side_effect=ValueError('No existing credentials')) as auth, \
                patch('burst2safe.auth.write_credentials_to_netrc_file') as write:
            self.assertEqual(bd.download(self.request,copy.deepcopy(groups),self.root/'collection',
                lambda **kw:self.fail('downloaded without credentials')),1)
            auth.assert_called_once_with(append=False);write.assert_not_called()
        self.assertEqual(self.download(groups),0)
        with patch.dict('sys.modules',{'asf_search':None,'burst2safe':None}):
            self.assertIsNotNone(bd.managed_collection(self.root/'collection'))
        with bd.collection_lock(self.root/'collection'):
            with self.assertRaisesRegex(ValueError,'Another download'):
                self.download(groups)

    def test_invalid_request_and_selection_changes(self):
        for updates in ({'start_date':'20180108'}, {'end_date':'2018-01-07'},
                        {'relative_orbit':176}, {'aoi_layer':'invalid'}):
            args=copy.copy(self.args)
            for key,value in updates.items():setattr(args,key,value)
            with self.assertRaises(ValueError):bd.request_from_args(args)
        groups,_=self.plan([product()]);self.assertEqual(self.download(groups),0)
        with self.assertRaisesRegex(ValueError,'Changed selection'):
            self.download(groups,request=self.request|{'polarization':'VV'})

    def test_publication_reuse_corruption_retry_and_provenance(self):
        groups,_=self.plan([product(),product(11)])
        self.assertEqual(self.download(groups),0)
        collection=self.root/'collection'
        provenance=bd.managed_collection(collection)
        self.assertEqual(provenance,bd.managed_collection(collection/'safe'))
        self.assertEqual(len(scan_safes(collection)),1)
        safe=scan_safes(collection)[0]
        self.assertEqual(prep.footprint(safe)[0],[0.,0.,1.,1.])
        before={str(p):(checksum(p),p.stat().st_mtime_ns) for p in safe.rglob('*') if p.is_file()}
        self.assertEqual(self.download(groups,lambda **kw:self.fail('unchanged SAFE redownloaded')),0)
        self.assertEqual(before,{str(p):(checksum(p),p.stat().st_mtime_ns) for p in safe.rglob('*') if p.is_file()})
        measurement=next((safe/'measurement').iterdir());measurement.write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError,'checksum'):
            bd.managed_collection(collection)
        self.assertEqual(self.download(groups),0)
        self.assertEqual(bd.managed_collection(collection)['download']['status'],'complete')

    def test_macos_sidecars_allow_initialization_validation_and_reuse(self):
        collection=self.root/'collection';collection.mkdir()
        (collection/'._.download.lock').write_bytes(b'AppleDouble')
        (collection/'.DS_Store').write_bytes(b'Finder metadata')
        groups,_=self.plan([product()])
        self.assertEqual(self.download(groups),0)
        safe=scan_safes(collection)[0]
        validation=bd.managed_collection(collection)['download']['acquisitions'][0]['validation']
        for p in list(safe.rglob('*')):
            if p.is_file():(p.parent/('._'+p.name)).write_bytes(b'AppleDouble, not XML or TIFF')
        (safe/'measurement'/'.DS_Store').write_bytes(b'Finder metadata')
        self.assertEqual(bd.validate_safe(safe,groups[0],self.request),validation)
        self.assertEqual(len(prep.annotation_paths(safe)),3)
        self.assertEqual(prep.footprint(safe)[0],[0.,0.,1.,1.])
        self.assertEqual(self.download(groups,lambda **kw:self.fail('redownloaded due to sidecars')),0)
        self.assertIsNotNone(bd.managed_collection(collection))

    def test_other_collection_content_without_manifest_is_still_rejected(self):
        collection=self.root/'collection';collection.mkdir()
        groups,_=self.plan([product()])
        for name in ('unrelated.txt','.staging'):
            path=collection/name;path.write_text('not filesystem metadata')
            with self.assertRaisesRegex(ValueError,'must be empty'):
                self.download(groups)
            path.unlink()

    def test_invalid_safe_tiff_xml_and_helpers(self):
        groups,_=self.plan([product()]);group=groups[0]
        safe=synthetic_safe(self.root,group,self.request)
        good=bd.validate_safe(safe,group,self.request)
        self.assertEqual(good['helper_swaths'],['IW1','IW3'])
        annotation=next((safe/'annotation').glob('*iw1*'))
        annotation.write_text('<truncated')
        with self.assertRaises(Exception):bd.validate_safe(safe,group,self.request)
        shutil.rmtree(safe)
        safe=synthetic_safe(self.root,group,self.request)
        next((safe/'measurement').iterdir()).write_text('bad tiff')
        with self.assertRaises(Exception):bd.validate_safe(safe,group,self.request)

    def test_failures_continue_and_incomplete_collections_refused(self):
        groups,_=self.plan([product(),product(day=14,mission='B',orbit_number=9164)])
        calls=[]
        def converter(**kw):
            calls.append(kw['granules'])
            if len(calls)==1:raise RuntimeError('network failure')
            return synthetic_safe(kw['work_dir'],groups[1],self.request)
        self.assertEqual(self.download(groups,converter),1);self.assertEqual(len(calls),2)
        collection=self.root/'collection'
        for path in (collection,collection/'safe'):
            with self.assertRaisesRegex(ValueError,'incomplete'):bd.managed_collection(path)
            self.assertEqual(main(['prepare-inputs',str(path),'-o',str(self.root/'prep')]),1)
        self.assertEqual(self.download(groups),0)

    def test_interruption_retry_extensions_and_inventory_consistency(self):
        groups,_=self.plan([product()])
        def interrupt(**kw):raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):self.download(groups,interrupt)
        collection=self.root/'collection'
        self.assertEqual(json.loads((collection/bd.MANIFEST).read_text())['status'],'interrupted')
        self.assertEqual(self.download(groups),0)
        extended=self.request|{'end_date':'2018-01-26'}
        more,_=self.plan([product(),product(day=26,orbit_number=20322)],request=extended)
        self.assertEqual(self.download(more,request=extended),0)
        with self.assertRaisesRegex(ValueError,'narrowed'):self.download(groups)
        with self.assertRaisesRegex(ValueError,'omitted'):self.download(groups,request=extended)
        extra=collection/'safe'/'extra.SAFE';extra.mkdir()
        with self.assertRaisesRegex(ValueError,'inconsistent'):bd.managed_collection(collection)

    def test_helpers_are_not_inventoried_or_used_for_dem(self):
        request=self.request|{'subswaths':[1,3]}
        groups,_=self.plan([product(swath='IW1'),product(swath='IW3')],request=request)
        safe=synthetic_safe(self.root,groups[0],request)
        bd.validate_safe(safe,groups[0],request)
        def loader(*args,**kw):self.fail('s1-reader called for metadata-only helper')
        loader.__module__='s1reader.s1_reader'
        self.assertEqual(inventory(safe,'unused','HH',[2],loader),[])
        helper=next((safe/'annotation').glob('*iw2*'))
        helper.write_text(helper.read_text().replace('</product>',
            '<geolocationGridPoint><longitude>100</longitude><latitude>80</latitude></geolocationGridPoint></product>'))
        self.assertEqual(prep.footprint(safe)[0],[0.,0.,1.,1.])

    def test_offline_prepare_and_stack_initialization(self):
        groups,_=self.plan([product(),product(day=14,mission='B',orbit_number=9164)])
        self.assertEqual(self.download(groups),0)
        from glacier_offsets.inputs import Burst
        native=SimpleNamespace(border=[box(0,0,1,1)])
        def fake_inventory(safe,*args,**kwargs):
            from glacier_offsets.initialization import read_scene
            return [Burst('t038_000010_iw2',2,read_scene(safe).start,native)]
        def get_orbit(safe,folder,*args):
            from glacier_offsets.initialization import read_scene
            scene=read_scene(safe)
            return str(orbit(folder/f'{scene.mission}.EOF',scene.start-timedelta(hours=3),mission=scene.mission[1:]))
        def get_dem(bounds,folder,args):
            path=folder/'dem.tif';make_dem(path,bounds)
            Path(str(path)+'.preparation.json').write_text('{}');return str(path)
        prepared=self.root/'prepared'
        with patch.object(prep,'get_orbit',side_effect=get_orbit),patch.object(prep,'get_dem',side_effect=get_dem),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['prepare-inputs',str(self.root/'collection'),'-o',str(prepared),'--padding-km','0']),0)
        manifest_path=prepared/'manifest.json'
        self.assertEqual(json.loads(manifest_path.read_text())['burst_download']['download']['status'],'complete')
        self.assertEqual(prepared_paths(manifest_path)[0],str(self.root/'collection'/'safe'))
        config=self.root/'stack.yaml'
        with patch('glacier_offsets.stack.inventory',side_effect=fake_inventory),patch('glacier_offsets.batch.validate_burst'),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['stack','init','--prepared-inputs',str(manifest_path),'--bbox','0','0','1','1',
                '--polarization','HH','--epsg','3031','--output',str(config),'--output-directory',str(self.root/'stack')]),0)
            plan=plan_stack(load_stack_config(config))
        self.assertEqual(len(plan.acquisitions),2);self.assertEqual(len(plan.pairs),1)
        with patch('glacier_offsets.stack.inventory',side_effect=fake_inventory), \
                patch('glacier_offsets.assembly.validate_native_layout'), \
                patch('glacier_offsets.batch.validate_burst'),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['stack','plan',str(config)]),0)
            self.assertEqual(main(['stack','validate',str(config),'--backend','cpu']),0)
        # A verification-only downloader restart preserves the handoff identity.
        self.assertEqual(self.download(groups),0)
        self.assertEqual(prepared_paths(manifest_path)[0],str(self.root/'collection'/'safe'))
        state=json.loads((self.root/'collection'/bd.MANIFEST).read_text())
        state['status']='running'
        (self.root/'collection'/bd.MANIFEST).write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'collection changed'):
            prepared_paths(manifest_path)


if __name__=='__main__':unittest.main()
