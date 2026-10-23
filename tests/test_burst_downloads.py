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
        bursts = ''.join(f'<burst><azimuthTime>{r["start"]}</azimuthTime></burst>'
                         for r in sorted((r for r in group['bursts'] if r['swath']==swath),
                                         key=lambda r:r['start']))
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

    def partial_plan(self, products):
        request=self.request|{'subswaths':[1,2],'allow_partial':True}
        groups,_=self.plan(products,request=request)
        return request,groups

    def test_partial_missing_interior_preserves_iw1_and_records_all_omissions(self):
        products=[product(i,swath='IW1') for i in (10,11,13,14)]+[product(i) for i in range(10,15)]
        strict,_=self.plan(products,request=self.request|{'subswaths':[1,2]})
        self.assertIn('missing interior',strict[0]['errors'][0])
        request,groups=self.partial_plan(products);group=groups[0]
        self.assertFalse(group['errors'])
        self.assertEqual(group['coverage']['retained_ranges'],{'IW1':[10,11],'IW2':[10,12]})
        self.assertEqual(group['coverage']['missing_bursts'],[
            {'swath':'IW1','burst_id':12,'reason':'unavailable_after_search'}])
        discarded=group['coverage']['discarded_bursts']
        self.assertEqual({(r['swath'],r['burst_id']) for r in discarded},
                         {('IW1',13),('IW1',14),('IW2',13),('IW2',14)})
        self.assertTrue(all(r['aoi_requested'] and r['parent'] and r['geometry'] for r in discarded))
        self.assertEqual(len(group['requested']),9)
        self.assertEqual(len(group['bursts']),5)
        self.assertEqual(self.download(groups,request=request),0)
        provenance=bd.managed_collection(self.root/'collection')
        self.assertEqual(provenance['download']['acquisitions'][0]['coverage'],group['coverage'])
        self.assertEqual(provenance['download']['status'],'complete')
        summary=bd.format_plan(request,groups)
        self.assertIn('0 full, 1 partial, 0 with issues',summary)
        self.assertIn('1 unavailable bursts, 4 available products discarded',summary)

    def test_partial_cross_swath_extent_mismatch_and_unresolvable_groups(self):
        products=[product(i,swath='IW1') for i in range(10,15)]+[product(i) for i in range(12,15)]
        request,groups=self.partial_plan(products)
        self.assertFalse(groups[0]['errors'])
        self.assertEqual(groups[0]['coverage']['retained_ranges'],{'IW1':[11,14],'IW2':[12,14]})
        self.assertEqual(groups[0]['coverage']['missing_bursts'],[])
        self.assertTrue(groups[0]['coverage']['partial'])
        self.assertEqual(len(groups[0]['coverage']['discarded_bursts']),1)
        _,unresolvable=self.partial_plan([product(10,swath='IW1'),product(15)])
        self.assertIn('every observed swath',unresolvable[0]['errors'][0])
        _,duplicate=self.partial_plan([product(10,swath='IW1'),product(10,swath='IW1',suffix='BBBB')])
        self.assertIn('ambiguous duplicate',duplicate[0]['errors'][0])
        _,inconsistent=self.partial_plan([product(10,swath='IW1'),product(10,flightDirection='ASCENDING')])
        self.assertIn('directions',inconsistent[0]['errors'][0])

    def test_partial_policy_adopts_existing_collection_reuses_good_dates_and_survives_resume(self):
        # The good date sorts after the problematic date, reproducing an
        # interruption before the already-completed acquisition is visited.
        products=[product(i,swath='IW1') for i in (10,11,13,14)]+[product(i) for i in range(10,15)]
        products += [product(i,swath=s,day=20,orbit_number=20235)
                     for s in ('IW1','IW2') for i in range(10,15)]
        strict_request=self.request|{'subswaths':[1,2]}
        strict,_=self.plan(products,request=strict_request)
        self.assertEqual(self.download(strict,request=strict_request),1)
        collection=self.root/'collection';good=scan_safes(collection)[0]
        before={str(p):(checksum(p),p.stat().st_mtime_ns) for p in good.rglob('*') if p.is_file()}
        request,groups=self.partial_plan(products)
        for _ in range(2):
            def interrupt(**kw):raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):self.download(groups,interrupt,request=request)
            state=json.loads((collection/bd.MANIFEST).read_text())
            self.assertEqual(state['acquisitions'][1]['status'],'complete')
        calls=[]
        def convert(**kw):
            calls.append(kw['granules'])
            self.assertEqual(kw['granules'],[r['granule'] for r in groups[0]['bursts']])
            return synthetic_safe(kw['work_dir'],groups[0],request)
        self.assertEqual(self.download(groups,convert,request=request),0)
        self.assertEqual(len(calls),1)
        self.assertEqual(before,{p:(checksum(p),Path(p).stat().st_mtime_ns) for p in before})
        self.assertEqual(self.download(groups,lambda **kw:self.fail('unchanged partial SAFE redownloaded'),request),0)
        self.assertIsNotNone(bd.managed_collection(collection))
        with self.assertRaisesRegex(ValueError,'repeat --allow-partial'):
            self.download(strict,request=strict_request)

    def test_partial_coverage_provenance_is_part_of_identity_and_rejects_tampering(self):
        request,groups=self.partial_plan([product(i,swath='IW1') for i in (10,11,13,14)]+
                                         [product(i) for i in range(10,15)])
        self.assertEqual(self.download(groups,request=request),0)
        collection=self.root/'collection';path=collection/bd.MANIFEST
        original=json.loads(path.read_text());previous=bd.collection_identity(original)
        changed=copy.deepcopy(original)
        changed['acquisitions'][0]['coverage']['discarded_bursts'][0]['reason']='different explanation'
        self.assertNotEqual(previous,bd.collection_identity(changed))
        for change in ('policy','counts','partial','request'):
            state=copy.deepcopy(original)
            coverage=state['acquisitions'][0]['coverage']
            if change=='policy':coverage['policy']='unknown-policy'
            if change=='counts':coverage['retained_burst_count']+=1
            if change=='partial':coverage['partial']=False
            if change=='request':state['request'].pop('allow_partial')
            path.write_text(json.dumps(state))
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'partial coverage'):
                bd.managed_collection(collection)
        path.write_text(json.dumps(original))
        self.assertIsNotNone(bd.managed_collection(collection))

    def test_partial_cli_dry_run_and_single_swath_helpers(self):
        service=SimpleNamespace(constants=self.asf.constants,
            search=lambda **kw:[product(i,swath='IW1') for i in (10,12,13,14)])
        argv=['download-bursts','--bbox','0','0','1','1','--start-date','2018-01-08',
              '--end-date','2018-01-20','--relative-orbit','38','--polarization','HH','--subswaths','1',
              '-o',str(self.root/'absent'),'--allow-partial','--dry-run']
        output=io.StringIO()
        with patch.object(bd,'dependencies',return_value=(service,None,self.safe_class,self.create)), \
                contextlib.redirect_stdout(output):
            self.assertEqual(main(argv),0)
        self.assertIn('1 partial',output.getvalue())
        self.assertFalse((self.root/'absent').exists())
        request=self.request|{'subswaths':[1],'allow_partial':True}
        groups,_=self.plan(service.search(),request=request)
        self.assertEqual(groups[0]['coverage']['retained_ranges'],{'IW1':[12,14]})
        self.assertEqual(self.download(groups,request=request),0)
        group=bd.managed_collection(self.root/'collection')['download']['acquisitions'][0]
        self.assertEqual(group['validation']['helper_swaths'],['IW2','IW3'])

    def test_partial_collection_preparation_and_three_date_stack_handoff(self):
        products=[product(i,swath=s,day=day,orbit_number=orb,mission=mission)
                  for day,orb,mission in ((8,20060,'A'),(14,9164,'B'),(20,20235,'A'))
                  for s in ('IW1','IW2') for i in range(10,15)
                  if not (day==14 and s=='IW1' and i==12)]
        request,groups=self.partial_plan(products)
        self.assertEqual(self.download(groups,request=request),0)
        from glacier_offsets.initialization import read_scene
        from glacier_offsets.inputs import Burst
        def get_orbit(safe,folder,*args):
            scene=read_scene(safe)
            return str(orbit(folder/f'{scene.mission}_{scene.start:%Y%m%d}.EOF',
                       scene.start-timedelta(hours=3),mission=scene.mission[1:]))
        def get_dem(bounds,folder,args):
            path=folder/'dem.tif';make_dem(path,bounds)
            Path(str(path)+'.preparation.json').write_text('{}');return str(path)
        native=SimpleNamespace(border=[box(0,0,1,1)])
        def inventory(safe,*args,**kwargs):
            group=next(g for g in groups if g['date']==read_scene(safe).start.date().isoformat())
            return [Burst(f't038_{r["burst_id"]:06d}_{r["swath"].lower()}',int(r['swath'][-1]),
                          datetime.fromisoformat(r['start']),native) for r in group['bursts']]
        prepared=self.root/'prepared';config=self.root/'partial-stack.yaml'
        with patch.object(prep,'get_orbit',side_effect=get_orbit),patch.object(prep,'get_dem',side_effect=get_dem), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['prepare-inputs',str(self.root/'collection'),'-o',str(prepared),
                                   '--padding-km','0']),0)
        manifest_path=prepared/'manifest.json'
        preparation=json.loads(manifest_path.read_text())
        self.assertEqual(preparation['status'],'complete')
        coverage=[g['coverage'] for g in preparation['burst_download']['download']['acquisitions']]
        self.assertEqual(sum(c['partial'] for c in coverage),1)
        with patch('glacier_offsets.stack.inventory',side_effect=inventory), \
                patch('glacier_offsets.assembly.validate_native_layout'), \
                patch('glacier_offsets.batch.validate_burst'),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['stack','init','--prepared-inputs',str(manifest_path),
                '--bbox','0','0','1','1','--polarization','HH','--epsg','3031','--neighbors','2',
                '--reference-date','2018-01-08','--output',str(config),
                '--output-directory',str(self.root/'stack')]),0)
            plan=plan_stack(load_stack_config(config))
            self.assertEqual(main(['stack','validate',str(config),'--backend','cpu']),0)
        self.assertEqual(len(plan.bursts),10)
        self.assertEqual(len(plan.pairs),3)
        partial=next(r for r in plan.report['inventory'] if r['date']=='2018-01-14')
        self.assertEqual(set(partial['missing_burst_ids']),
            {'t038_000012_iw1','t038_000013_iw1','t038_000014_iw1','t038_000013_iw2','t038_000014_iw2'})
        self.assertEqual(len(plan.report['pairs']['2018-01-14_2018-01-20']['shared_burst_ids']),5)
        # Omission provenance changes require a new preparation handoff, even
        # when the already-published retained granules are identical.
        state=json.loads((self.root/'collection'/bd.MANIFEST).read_text())
        next(g for g in state['acquisitions'] if g['coverage']['partial'])['coverage']['missing_bursts'][0]['reason']='updated provenance'
        (self.root/'collection'/bd.MANIFEST).write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError,'collection changed'):
            prepared_paths(manifest_path)

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

    def test_repeated_interruptions_preserve_unvisited_completed_acquisitions(self):
        groups,_=self.plan([product(),product(day=14,mission='B',orbit_number=9164),
                            product(day=20,orbit_number=20235)])
        def initial(**kw):
            if kw['granules']==[r['granule'] for r in groups[0]['bursts']]:
                raise RuntimeError('first acquisition unavailable')
            group=next(g for g in groups if [r['granule'] for r in g['bursts']]==kw['granules'])
            return synthetic_safe(kw['work_dir'],group,self.request)
        self.assertEqual(self.download(groups,initial),1)
        collection=self.root/'collection';manifest_path=collection/bd.MANIFEST
        saved=json.loads(manifest_path.read_text())
        completed={g['id']:g for g in saved['acquisitions'] if g['status']=='complete'}
        self.assertEqual(len(completed),2)
        # Older collections recorded upstream versions only at collection level.
        for group in completed.values():group.pop('versions')
        bd.write_json(manifest_path,saved)
        completed={key:value|{'versions':saved['versions']} for key,value in completed.items()}
        before={str(p):(checksum(p),p.stat().st_mtime_ns)
                for safe in scan_safes(collection) for p in safe.rglob('*') if p.is_file()}
        for _ in range(2):
            def interrupt(**kw):
                self.assertEqual(kw['granules'],[r['granule'] for r in groups[0]['bursts']])
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):self.download(groups,interrupt)
            state=json.loads(manifest_path.read_text())
            self.assertEqual(state['status'],'interrupted')
            for group in state['acquisitions'][1:]:
                self.assertEqual(group['status'],'complete')
                for key in ('safe','validation','versions','downloaded_bytes','download_elapsed_seconds'):
                    self.assertEqual(group.get(key),completed[group['id']].get(key))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                bd.managed_collection(collection)
        calls=[]
        def finish(**kw):
            calls.append(kw['granules'])
            self.assertEqual(kw['granules'],[r['granule'] for r in groups[0]['bursts']])
            return synthetic_safe(kw['work_dir'],groups[0],self.request)
        self.assertEqual(self.download(groups,finish),0)
        self.assertEqual(len(calls),1)
        self.assertEqual(before,{name:(checksum(name),Path(name).stat().st_mtime_ns) for name in before})
        self.assertIsNotNone(bd.managed_collection(collection))

    def test_interrupted_checksum_verification_preserves_completion_and_detects_corruption(self):
        groups,_=self.plan([product(),product(day=20,orbit_number=20235)])
        self.assertEqual(self.download(groups),0)
        collection=self.root/'collection';manifest_path=collection/bd.MANIFEST
        original=json.loads(manifest_path.read_text())
        for _ in range(2):
            with patch.object(bd,'verify_files',side_effect=KeyboardInterrupt), \
                    self.assertRaises(KeyboardInterrupt):
                self.download(groups,lambda **kw:self.fail('transfer during verification'))
            state=json.loads(manifest_path.read_text())
            self.assertEqual(state['status'],'interrupted')
            self.assertTrue(all(g['status']=='complete' for g in state['acquisitions']))
            for saved,current in zip(original['acquisitions'],state['acquisitions']):
                self.assertEqual(saved['safe'],current['safe'])
                self.assertEqual(saved['validation'],current['validation'])
        self.assertEqual(self.download(groups,lambda **kw:self.fail('redownloaded completed acquisition')),0)
        # Preserved markers must not bypass verification on the next retry.
        corrupt=collection/original['acquisitions'][0]['safe']
        next((corrupt/'measurement').iterdir()).write_bytes(b'truncated')
        def interrupt_retry(**kw):raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):self.download(groups,interrupt_retry)
        state=json.loads(manifest_path.read_text())
        self.assertEqual(state['acquisitions'][0]['status'],'interrupted')
        self.assertEqual(state['acquisitions'][1]['status'],'complete')
        self.assertEqual(self.download(groups),0)
        self.assertIsNotNone(bd.managed_collection(collection))

    def test_termination_after_initial_resume_checkpoint_retains_completed_products(self):
        groups,_=self.plan([product(),product(day=20,orbit_number=20235)])
        self.assertEqual(self.download(groups),0)
        collection=self.root/'collection';manifest_path=collection/bd.MANIFEST
        original=json.loads(manifest_path.read_text())
        write=bd.write_json
        def checkpoint_then_exit(path,manifest):
            write(path,manifest)
            raise SystemExit('simulated process termination')
        with patch.object(bd,'write_json',side_effect=checkpoint_then_exit),self.assertRaises(SystemExit):
            self.download(groups,lambda **kw:self.fail('transferred before initial checkpoint'))
        state=json.loads(manifest_path.read_text())
        self.assertEqual(state['status'],'running')
        self.assertTrue(all(g['status']=='complete' for g in state['acquisitions']))
        self.assertEqual([g['validation'] for g in original['acquisitions']],
                         [g['validation'] for g in state['acquisitions']])
        self.assertEqual(self.download(groups,lambda **kw:self.fail('redownloaded after termination')),0)

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


class BurstPlanOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.request={'start_date':'2018-01-08','end_date':'2018-01-20',
                      'relative_orbit':38,'polarization':'HH','subswaths':[2]}
        bursts=[{'granule':f'burst-{i}','swath':'IW2','burst_id':i,
                 'geometry':{'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,0]]]}}
                for i in (10,11)]
        self.groups=[{'id':'S1A_020060','mission':'S1A','date':'2018-01-08',
                      'requested':['burst-10','burst-11'],'context':[],
                      'bursts':bursts,'errors':[]},
                     {'id':'S1B_009164','mission':'S1B','date':'2018-01-14',
                      'requested':['burst-10'],'context':['burst-11'],
                      'bursts':bursts,'errors':[]}]
        self.argv=['download-bursts','--bbox','0','0','1','1','--start-date','2018-01-08',
                   '--end-date','2018-01-20','--relative-orbit','38','--polarization','HH',
                   '--subswaths','2','-o',str(self.root/'absent')]

    def run_cli(self, extra, groups=None, result=0):
        output, errors=io.StringIO(),io.StringIO()
        with patch.object(bd,'dependencies',return_value=(None,None,None,None)), \
                patch.object(bd,'plan_downloads',return_value=self.groups if groups is None else groups), \
                patch.object(bd,'download',return_value=result) as download, \
                contextlib.redirect_stdout(output),contextlib.redirect_stderr(errors):
            code=main(self.argv+extra)
        return code,output.getvalue(),errors.getvalue(),download

    def test_default_summary_counts_and_no_dry_run_files(self):
        code,output,_,download=self.run_cli(['--dry-run'])
        self.assertEqual(code,0);download.assert_not_called()
        for expected in ('Acquisitions: 2 (2 ready, 0 with issues)',
                         'Available UTC dates: 2018-01-08 to 2018-01-14 (2 dates)',
                         'Spacecraft: S1A: 1, S1B: 1',
                         'Burst products: 4 (3 AOI, 1 added for contiguity)',
                         'Unique geographic bursts: 2','Burst products per acquisition: 2',
                         'By swath: IW2: 4'):
            self.assertIn(expected,output)
        self.assertNotIn('coordinates',output)
        self.assertNotIn('burst-10',output)
        self.assertFalse((self.root/'absent').exists())

    def test_two_year_summary_and_error_output_remain_bounded(self):
        groups=[]
        for i in range(122):
            group=copy.deepcopy(self.groups[0])
            group.update(id=f'S1A_{20060+175*i:06d}',
                         date=(TIME+timedelta(days=6*i)).date().isoformat())
            groups.append(group)
        request=self.request|{'end_date':'2020-01-07'}
        summary=bd.format_plan(request,groups)
        self.assertIn('Acquisitions: 122 (122 ready, 0 with issues)',summary)
        self.assertIn('Burst products: 244',summary)
        self.assertLess(len(summary.splitlines()),15)
        for group in groups:group['errors']=['A long acquisition error '+('x'*500),'another issue']
        summary=bd.format_plan(request,groups)
        self.assertIn('0 ready, 122 with issues',summary)
        self.assertIn('117 more acquisitions with issues',summary)
        self.assertEqual(summary.count('(+1 more issues)'),5)
        self.assertLess(len(summary.splitlines()),20)
        self.assertLess(max(map(len,summary.splitlines())),300)

    def test_partial_summary_stays_compact_for_long_requests(self):
        groups=[copy.deepcopy(self.groups[0]) for _ in range(122)]
        for group in groups:
            group['coverage']={'partial':True,'retained_ranges':{'IW1':[10,11],'IW2':[10,12]},
                               'missing_bursts':[{'swath':'IW1','burst_id':12}],
                               'discarded_bursts':[{'granule':'omitted'}]}
        output=bd.format_plan(self.request|{'allow_partial':True},groups)
        self.assertIn('0 full, 122 partial, 0 with issues',output)
        self.assertIn('117 more partial acquisitions',output)
        self.assertEqual(output.count(': retained IW1 10-11, IW2 10-12'),5)
        self.assertLess(len(output.splitlines()),25)

    def test_empty_and_incompatible_searches_still_exit_nonzero(self):
        code,output,_,_=self.run_cli(['--dry-run'],groups=[])
        self.assertEqual(code,2)
        self.assertIn('Acquisitions: 0 (0 ready, 0 with issues)',output)
        self.assertIn('Burst products: 0',output)
        self.assertIn('Available UTC dates: none',output)
        self.groups[0]['errors']=['missing interior burst']
        code,output,_,_=self.run_cli(['--dry-run'])
        self.assertEqual(code,2)
        self.assertIn('1 ready, 1 with issues',output)
        self.assertIn('S1A_020060: missing interior burst',output)

    def test_full_json_preserves_complete_plan_and_is_parseable_after_download(self):
        for extra in (['--dry-run','--json'],['--json']):
            with self.subTest(extra=extra):
                code,output,errors,download=self.run_cli(extra)
                self.assertEqual(code,0)
                plan=json.loads(output)
                self.assertEqual(plan['acquisitions'],self.groups)
                self.assertEqual(plan['request']['start_date'],'2018-01-08')
                if '--dry-run' in extra:
                    download.assert_not_called()
                    self.assertFalse((self.root/'absent').exists())
                else:
                    download.assert_called_once()
                    self.assertIn('Download manifest:',errors)

    def test_normal_download_uses_same_compact_plan(self):
        code,output,_,download=self.run_cli([],result=1)
        self.assertEqual(code,1);download.assert_called_once()
        self.assertIn('Burst products: 4',output)
        self.assertIn('Download manifest:',output)
        self.assertNotIn('coordinates',output)


if __name__=='__main__':unittest.main()
