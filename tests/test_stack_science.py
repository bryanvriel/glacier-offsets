"""Three-date native stack science, fixed ownership and incremental failures."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
from threading import current_thread, main_thread
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal
from scipy.ndimage import shift

import test_assembly
from test_synthetic_offsets import write_slc
from glacier_offsets.ampcor import match, native_match
from glacier_offsets.inputs import Burst
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.stack import Acquisition, StackPlan
from glacier_offsets.stack_pairs import common_map_grid, prepared_pair
from glacier_offsets.stack_storage import (SourceHashes, assemble_date, date_specification, prepare_date,
                                           prepare_geometry, read_manifest, reference_specification,
                                           write_reference_layout)
from glacier_offsets.stack_workflow import execute_stack


class StackScienceTests(unittest.TestCase):
    def setUp(self):
        test_assembly.AssemblyTests.setUp(self)
        self.cfg['stack'] = {'reference_date': '2020-01-04'}
        self.reference_day = datetime(2020,1,4)
        self.acquisitions = []
        self.aligned_slcs = {}
        self.translations = [(1,-2), (0,0), (4,-5)]
        self.software = {'packages': {}}
        for index, day in enumerate([datetime(2020,1,1), self.reference_day, datetime(2020,1,9)]):
            safe = self.root/f'safe-{index}'; safe.write_text(f'scene-{index}')
            orbit = self.root/f'orbit-{index}'; orbit.write_text(f'orbit-{index}')
            bursts = {}
            for n, (directory, manifest) in enumerate(self.sources):
                key = directory.name
                grid = deepcopy(manifest['coreg']['reference_grid'])
                grid['ref_epoch'] = self.reference_day.isoformat()
                manifest['coreg']['reference_grid'] = grid
                native = SimpleNamespace(as_isce3_radargrid=lambda grid=grid: SimpleNamespace(**grid),
                    orbit=object(), polarization='HH', orbit_direction='DESCENDING',
                    first_valid_line=4, last_valid_line=187, first_valid_sample=4, last_valid_sample=379)
                # Deliberately unequal exact per-burst intervals on the last date.
                time = day+timedelta(seconds=n*.256 + (n if index==2 else 0))
                bursts[key] = Burst(key, 2, time, native)
                source = gdal.Open(manifest['coreg']['slcs']['reference']).ReadAsArray()
                data = shift(source,self.translations[index],order=0,mode='constant',cval=0)
                path = self.root/f'date-{index}-{key}.tif'; write_slc(path,data)
                self.aligned_slcs[(day.date().isoformat(),key)] = str(path)
            metadata = SimpleNamespace(path=safe,start=day,stop=day+timedelta(seconds=30),
                relative_orbit=65,direction='DESCENDING',polarizations=('HH',),mission='S1A')
            self.acquisitions.append(Acquisition(metadata,SimpleNamespace(path=orbit),bursts))
        self.reference = self.acquisitions[1]
        self.plan = self.make_plan(self.acquisitions)
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        self.reference_call = self.stack.enter_context(patch('glacier_offsets.stack_storage.prepare_reference', side_effect=self.reference_product))
        self.align_call = self.stack.enter_context(patch('glacier_offsets.stack_storage.align_acquisition', side_effect=self.align_product))
        self.hashes = SourceHashes()

    def make_plan(self, acquisitions, links=None):
        links = links if links is not None else [(a.date,b.date) for i,a in enumerate(acquisitions) for b in acquisitions[i+1:]]
        entries = {f'{a}_{b}': {'earlier':a,'later':b,'active':True,'status':'planned'} for a,b in links}
        anchor = {'reference_date':self.reference.date,'selection':list(self.reference.bursts)}
        report = {'pairs':entries,'inventory':[],'reference_date':self.reference.date,'anchor':anchor}
        return StackPlan(tuple(acquisitions),self.reference,tuple(self.reference.bursts.values()),tuple(links),report,anchor)

    def reference_product(self, cfg, burst, root):
        return deepcopy(next(m['coreg'] for directory,m in self.sources if directory.name==burst.id))

    def align_product(self, cfg, reference, burst, root, coreg):
        return self.aligned_slcs[(burst.start.date().isoformat(),burst.id)]

    def geometry(self, force=False):
        spec = reference_specification(self.cfg,self.plan,self.hashes,self.software)
        return prepare_geometry(self.cfg,self.plan,spec,self.software,force=force)

    def dates(self, frame, acquisitions=None):
        result = {}
        for a in acquisitions or self.acquisitions:
            spec = date_specification(a,frame,self.hashes)
            aligned = prepare_date(self.cfg,self.plan,a,frame,spec)
            result[a.date] = assemble_date(self.cfg,a,frame,aligned)
        return result

    def effective(self):
        cfg = deepcopy(self.cfg); cfg.pop('aoi',None); cfg['inputs']['burst_id']='stack'
        return cfg

    def test_parallel_restart_preserves_products_and_retries_failed_checks(self):
        self.cfg['stack']['verification_workers'] = 1
        from glacier_offsets.stack_storage import checksum as source_checksum

        def prepare(*args, **kwargs):
            self.assertIs(current_thread(), main_thread())
            return prepare_date(*args, **kwargs)

        with patch('glacier_offsets.stack_workflow.plan_stack', return_value=self.plan), \
                patch('glacier_offsets.stack_workflow.validate_stack'), \
                patch('glacier_offsets.stack_workflow.inspect_environment', return_value=self.software), \
                patch('glacier_offsets.stack_workflow.prepare_date', side_effect=prepare):
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
            root = Path(self.cfg['output']['directory'])
            paths = [p for section in ('geometry', 'acquisitions')
                     for p in (root/section).rglob('*') if p.is_file()]
            before = {p: (checksum(p), p.stat().st_mtime_ns) for p in paths}
            calls = self.align_call.call_count
            self.cfg['stack']['verification_workers'] = 4
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
            self.assertEqual(self.align_call.call_count, calls)
            self.assertEqual(before, {p: (checksum(p), p.stat().st_mtime_ns) for p in paths})

            bad = self.acquisitions[0]
            def read(path):
                if Path(path) == bad.scene.path.resolve():
                    raise OSError('injected source read failure')
                return source_checksum(path)
            with patch('glacier_offsets.stack_storage.checksum', side_effect=read):
                self.assertEqual(execute_stack(self.cfg, 'coregister'), 1)
            report = json.loads((root/'stack_manifest.json').read_text())
            self.assertEqual(report['acquisitions'][bad.date]['status'], 'failed')
            self.assertTrue(all(report['acquisitions'][a.date]['status'] == 'complete'
                                for a in self.acquisitions if a is not bad))
            self.assertEqual(execute_stack(self.cfg, 'coregister'), 0)
            self.assertEqual(self.align_call.call_count, calls)
            self.assertEqual(before, {p: (checksum(p), p.stat().st_mtime_ns) for p in paths})

    def test_swath_major_reference_layout_keeps_slc_and_geometry_source_indices(self):
        sources = []
        for key, dy, dx, original in [('a_iw1', 0, 0, 0), ('b_iw1', 128, 0, 1),
                                       ('a_iw2', 64, 32, 0), ('b_iw2', 192, 32, 1)]:
            coreg = deepcopy(self.sources[original][1]['coreg'])
            grid = coreg['reference_grid']
            grid['sensing_start'] = dy/grid['prf']
            grid['starting_range'] += dx*grid['range_pixel_spacing']
            sources.append({'burst_id': key, 'coreg': coreg, 'valid_window': [4, 188, 4, 380]})
        layout = write_reference_layout(self.root/'swath-layout', sources)
        self.assertEqual([s['burst_id'] for s in layout['sources']], [s['burst_id'] for s in sources])
        owner = gdal.Open(layout['coreg']['source_index']).ReadAsArray()
        count = np.zeros(owner.shape, 'uint8')
        union = np.zeros(owner.shape, bool)
        geometry = gdal.Open(layout['coreg']['geometry']['x']).ReadAsArray()
        for index, entry in enumerate(layout['sources'], 1):
            dy, dx = entry['offset']
            a,b,c,d = entry['valid_window']
            union[a+dy:b+dy,c+dx:d+dx] = True
            original_geometry = gdal.Open(entry['coreg']['geometry']['x']).ReadAsArray()
            for a,b,c,d in entry['ownership_windows']:
                section = (slice(a+dy,b+dy), slice(c+dx,d+dx))
                count[section] += 1
                np.testing.assert_array_equal(owner[section], index)
                np.testing.assert_array_equal(geometry[section], original_geometry[a:b,c:d])
        np.testing.assert_array_equal(count, union.astype('uint8'))

    def test_failed_ownership_layout_reuses_completed_compass_bursts_on_retry(self):
        with patch('glacier_offsets.stack_storage.interior_ownership',
                   side_effect=ValueError('ownership failure')):
            with self.assertRaisesRegex(ValueError, 'ownership failure'):
                self.geometry()
        self.assertEqual(self.reference_call.call_count, 2)
        self.assertEqual(self.geometry()['status'], 'complete')
        self.assertEqual(self.reference_call.call_count, 2)

    def test_cross_swath_interior_recovers_both_dates_without_filling_real_holes(self):
        from glacier_offsets.matching_grid import build_grid, valid_windows
        rng = np.random.default_rng(217)
        original = (rng.uniform(.1, 1, (192, 512))*np.exp(1j*rng.normal(0, .2, (192, 512)))).astype('complex64')
        secondary = shift(original, (2, -3), order=0, mode='constant', cval=0)
        sources, slcs = [], [{}, {}]
        for key, dx in [('a_iw1', 0), ('a_iw2', 128)]:
            coreg = deepcopy(self.sources[0][1]['coreg'])
            coreg['reference_grid']['starting_range'] += dx*2
            for date, data in enumerate((original, secondary)):
                crop = data[:, dx:dx+384].copy()
                if dx:
                    # A fragile near-range edge on BOTH dates, still inside
                    # the reference annotation rectangle. IW1 supports it.
                    crop[:, 4:19] = 0
                    if date == 0:
                        crop[60:63, 192:195] = 0  # reference hole at x=320
                    else:
                        crop[100:103, 172:175] = 0  # secondary hole at x=300
                path = self.root/f'swath-{date}-{key}.tif'
                write_slc(path, crop)
                slcs[date][key] = str(path)
            coreg['slcs']['reference'] = slcs[0][key]
            sources.append({'burst_id': key, 'coreg': coreg, 'valid_window': [4, 188, 4, 380]})
        frame = write_reference_layout(self.root/'range-layout', sources)
        frame['geometry_identity'] = frame['layout_identity']
        dates = []
        for i in range(2):
            a = SimpleNamespace(date=f'2020-02-0{i+1}')
            dates.append(assemble_date(self.cfg, a, frame,
                {'slcs': slcs[i], 'alignment_identity': str(i)}))
        owner = gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        self.assertEqual(owner[80, 255], 1)
        self.assertEqual(owner[80, 256], 2)
        self.assertEqual(owner[61, 321], 0)
        for result in dates:
            valid = gdal.Open(result['validity_mask']).ReadAsArray()
            self.assertTrue(valid[4:188, 132:147].all())
            self.assertFalse(valid[61, 321])
        self.assertFalse(gdal.Open(dates[1]['validity_mask']).ReadAsArray()[101, 301])
        ref, sec = (gdal.Open(d['slc']) for d in dates)
        shape = ref.RasterYSize, ref.RasterXSize
        p = self.cfg['matching']
        grid = build_grid(shape, shape, p['window'], p['search'], p['stride'])
        supported = valid_windows(ref, sec, grid, (0, 0))
        y, x = grid.centers
        seam = (abs(x-256) < 32) & supported
        self.assertTrue(seam.any())
        result = native_match(dates[0]['slc'], dates[1]['slc'], self.root/'range-native', p,
                              self.cfg, _grid=grid, _valid_input=supported)
        self.assertTrue(result['valid'][seam].all())
        np.testing.assert_allclose(result['total'][seam], np.tile([2, -3], (int(seam.sum()), 1)), atol=.18)

    def test_reference_prepared_once_dates_before_reference_and_no_pair_duplication(self):
        frame = self.geometry(); dates = self.dates(frame)
        self.assertEqual(self.reference_call.call_count,2)
        self.assertEqual(self.align_call.call_count,4)
        # A damaged merged SLC can be repaired by assemble without re-aligning
        # any acquisition or demanding a coregistration refresh.
        first=self.acquisitions[0]
        Path(dates[first.date]['slc']).unlink()
        specification=date_specification(first,frame,self.hashes)
        aligned=prepare_date(self.cfg,self.plan,first,frame,specification,process=False)
        repaired=assemble_date(self.cfg,first,frame,aligned)
        self.assertTrue(Path(repaired['slc']).is_file())
        self.assertEqual(self.align_call.call_count,4)
        target = common_map_grid(self.cfg,frame)
        for a,b in self.plan.pairs:
            by_date = {x.date:x for x in self.acquisitions}
            directory,manifest = prepared_pair(self.cfg,self.plan,frame,by_date[a],by_date[b],dates,target)
            self.assertEqual(manifest['coreg']['slcs']['reference'],dates[a]['slc'])
            self.assertFalse((directory/'coreg').exists())
            self.assertEqual(manifest['stack_pair']['geometry_reference_date'],'2020-01-04')
        paths = [Path(p) for p in frame['products']]+[Path(p) for m in dates.values() for p in m['products']]
        before = {p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
        self.dates(self.geometry())
        self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
        self.assertEqual(self.reference_call.call_count,2)
        self.assertEqual(self.align_call.call_count,4)

    def test_fixed_overlap_owner_missing_burst_and_isolated_date_hole(self):
        frame = self.geometry()
        last = self.acquisitions[-1]
        keys = list(last.bursts)
        last.bursts.pop(keys[1])
        path = self.aligned_slcs[(last.date,keys[0])]
        ds = gdal.Open(path,gdal.GA_Update)
        data = ds.ReadAsArray(); data[60,100]=0
        ds.GetRasterBand(1).WriteArray(data)
        # This is a known one-pixel hole, distinct from an isolated quantized zero.
        with gdal.config_option('GDAL_TIFF_INTERNAL_MASK','NO'):
            ds.CreateMaskBand(gdal.GMF_PER_DATASET)
        ds.GetRasterBand(1).GetMaskBand().Fill(255)
        ds.GetRasterBand(1).GetMaskBand().WriteArray(np.zeros((1,1),'uint8'),100,60)
        ds=None
        dates = self.dates(frame)
        owner = gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        slc = gdal.Open(dates[last.date]['slc']).ReadAsArray()
        # Fixed interior cut is line 160; a missing lower burst cannot change it.
        self.assertTrue((owner[140:160,20:300]==1).all())
        self.assertTrue((slc[140:160,20:300]!=0).all())
        self.assertTrue((owner[160:180,20:300]==2).all())
        self.assertTrue((slc[160:180,20:300]==0).all())
        self.assertEqual(slc[60,100],0)
        ds = gdal.Open(dates[last.date]['slc'])
        self.assertEqual(ds.GetRasterBand(1).GetMaskBand().ReadAsArray()[60,100],0)
        self.assertGreater(dates[last.date]['valid_samples'],0)
        self.assertLess(dates[last.date]['valid_samples'],dates[self.reference.date]['valid_samples'])

    def test_interior_geometry_and_reference_holes_follow_the_same_fixed_owner(self):
        lower=self.sources[1][1]['coreg']
        ds=gdal.Open(lower['geometry']['z'],gdal.GA_Update)
        ds.GetRasterBand(1).Fill(100); ds=None
        ds=gdal.Open(lower['slcs']['reference'],gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.zeros((3,3),'complex64'),100,40); ds=None
        frame=self.geometry(); dates=self.dates(frame)
        owner=gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        self.assertEqual(owner[150,110],1)
        self.assertEqual(owner[160,110],2)
        # A hole in the assigned reference stays empty despite upper support.
        self.assertEqual(owner[169,101],0)
        for p in dates.values():
            ds=gdal.Open(p['slc'])
            self.assertEqual(ds.GetRasterBand(1).GetMaskBand().ReadAsArray(101,169,1,1)[0,0],0)
        vrt=gdal.Open(frame['coreg']['geometry']['z'])
        self.assertEqual(vrt.ReadAsArray(110,150,1,1)[0,0],0)
        self.assertEqual(vrt.ReadAsArray(110,160,1,1)[0,0],100)
        self.assertEqual(frame['ownership']['cuts'][0]['cut_global_line'],160)

    def test_native_translation_survives_secondary_burst_edge_amplitude_notches(self):
        # Both matched dates have a stationary dark edge in the lower burst.
        # Keeping it outside the owned overlap prevents a false zero shift.
        key=list(self.reference.bursts)[1]
        for a in (self.acquisitions[0],self.acquisitions[-1]):
            ds=gdal.Open(self.aligned_slcs[(a.date,key)],gdal.GA_Update)
            block=ds.ReadAsArray(0,0,ds.RasterXSize,16)
            ds.GetRasterBand(1).WriteArray(block*.01); ds=None
        frame=self.geometry(); dates=self.dates(frame)
        cfg=self.effective()
        a,b=self.acquisitions[0],self.acquisitions[-1]
        directory,m=prepared_pair(cfg,self.plan,frame,a,b,dates,common_map_grid(cfg,frame))
        match(cfg,directory,m,native_pair=self.reference.bursts)
        with h5py.File(directory/'radar/offsets.h5') as h:
            valid=h['valid'][:]
            near=valid & (h['center_line'][:]>110) & (h['center_line'][:]<190)
            self.assertTrue(near.any())
            np.testing.assert_allclose(h['total/azimuth_px'][:][near],3,atol=.18)
            np.testing.assert_allclose(h['total/range_px'][:][near],-3,atol=.18)

    def test_legacy_pinned_policy_is_rejected_before_any_mutation_even_with_force(self):
        with patch('glacier_offsets.stack_workflow.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack',return_value='cpu'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software):
            self.assertEqual(execute_stack(self.cfg,'coregister'),0)
            root=Path(self.cfg['output']['directory'])
            catalog=json.loads((root/'stack_manifest.json').read_text())
            spec=reference_specification(self.cfg,self.plan,self.hashes,self.software)
            spec['method']='stack-reference-support-connected-zero-fill-v2'
            spec.pop('ownership_policy')
            catalog['reference_specification_identity']=identity(spec)
            write_json(root/'stack_manifest.json',catalog)
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            with self.assertRaisesRegex(ValueError,'Pinned reference.*new output root'):
                execute_stack(self.cfg,'run',force=True)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in before})

    def test_isolated_reference_zero_does_not_erase_other_dates_or_invent_seams(self):
        from glacier_offsets.matching_grid import valid_windows
        from glacier_offsets.assembly import annotate_result
        from glacier_offsets.stack_pairs import matching_template
        path=self.sources[0][1]['coreg']['slcs']['reference']
        ds=gdal.Open(path,gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.zeros((1,1),'complex64'),100,60)
        # A genuine connected zero-filled hole must still remain unowned.
        ds.GetRasterBand(1).WriteArray(np.zeros((3,3),'complex64'),200,60)
        ds=None
        # Also retain an isolated zero from a matched acquisition.
        last=self.acquisitions[-1]
        key=list(last.bursts)[0]
        ds=gdal.Open(self.aligned_slcs[(last.date,key)],gdal.GA_Update)
        ds.GetRasterBand(1).WriteArray(np.zeros((1,1),'complex64'),110,70); ds=None
        frame=self.geometry();dates=self.dates(frame)
        owner=gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        self.assertEqual(owner[60,100],1)
        self.assertEqual(owner[61,201],0)
        for a in (self.acquisitions[0],last):
            ds=gdal.Open(dates[a.date]['slc'])
            self.assertNotEqual(ds.ReadAsArray(100,60,1,1)[0,0],0)
            self.assertEqual(ds.GetRasterBand(1).GetMaskBand().ReadAsArray(100,60,1,1)[0,0],255)
        ds=gdal.Open(dates[last.date]['slc'])
        self.assertEqual(ds.ReadAsArray(110,70,1,1)[0,0],0)
        self.assertEqual(ds.GetRasterBand(1).GetMaskBand().ReadAsArray(110,70,1,1)[0,0],255)
        ds=None
        grid=matching_template(self.cfg,frame)
        ref=gdal.Open(dates[self.acquisitions[0].date]['slc'])
        sec=gdal.Open(dates[last.date]['slc'])
        valid=valid_windows(ref,sec,grid,(0,0))
        self.assertTrue(valid[2,1])
        directory,m=prepared_pair(self.effective(),self.plan,frame,self.acquisitions[0],last,dates,
                                   common_map_grid(self.cfg,frame))
        result={'grid':grid,'gross':np.zeros((*grid.shape,2)),
                'input_valid':valid.copy(),'valid':valid.copy()}
        annotate_result(m,result)
        self.assertFalse(result['seam_crossing'][2,1])

    def test_merged_slcs_remain_memory_mappable_after_cache_flush_and_storage_upgrade(self):
        from glacier_offsets.raster_io import require_native_mmap
        from glacier_offsets.stack_storage import METHOD, SLC_STORAGE
        frame=self.geometry()
        previous_cache=gdal.GetCacheMax()
        try:
            # Force data and mask blocks to flush while writing, as for a real
            # multi-gigabyte SLC. Small all-cached rasters hid the original bug.
            gdal.SetCacheMax(64*1024)
            with gdal.config_option('GDAL_TIFF_INTERNAL_MASK','YES'):
                dates=self.dates(frame)
            for product in dates.values():
                self.assertTrue(Path(product['slc']+'.msk').is_file())
                ds=gdal.Open(product['slc'])
                require_native_mmap(ds)
                self.assertEqual(ds.GetRasterBand(1).GetMaskFlags(),gdal.GMF_PER_DATASET)
                ds=None
            acquisition=self.acquisitions[0]
            legacy=dates[acquisition.date]
            legacy['assembly_identity']=identity({'method':METHOD,'alignment':legacy['alignment_identity'],
                                                  'geometry':frame['geometry_identity']})
            legacy.pop('slc_storage')
            write_json(Path(legacy['slc']).parent/'manifest.json',legacy)
            counts=(self.reference_call.call_count,self.align_call.call_count)
            spec=date_specification(acquisition,frame,self.hashes)
            aligned=prepare_date(self.cfg,self.plan,acquisition,frame,spec,process=False)
            upgraded=assemble_date(self.cfg,acquisition,frame,aligned)
            self.assertEqual(upgraded['slc_storage'],SLC_STORAGE)
            self.assertNotEqual(upgraded['assembly_identity'],legacy['assembly_identity'])
            self.assertEqual(counts,(self.reference_call.call_count,self.align_call.call_count))
            self.assertEqual(frame['geometry_identity'],self.geometry()['geometry_identity'])
        finally:
            gdal.SetCacheMax(previous_cache)

    def test_three_date_native_sign_gross_intervals_scales_common_maps_and_empty_support(self):
        from glacier_offsets.geocode import geocode
        frame = self.geometry(); dates = self.dates(frame)
        cfg = self.effective()
        cfg['gross_offsets']={'mode':'constant','constant':{'azimuth_px':1,'range_px':-1}}
        target = common_map_grid(cfg,frame)
        by_date = {a.date:a for a in self.acquisitions}
        map_grids = []
        for earlier,later in self.plan.pairs:
            a,b = by_date[earlier],by_date[later]
            directory,m = prepared_pair(cfg,self.plan,frame,a,b,dates,target)
            match(cfg,directory,m,native_pair=self.reference.bursts)
            expected = np.array(self.translations[self.acquisitions.index(b)])-self.translations[self.acquisitions.index(a)]
            with h5py.File(directory/'radar/offsets.h5') as h:
                good = h['valid'][:]
                self.assertTrue(good.any())
                for k,axis in enumerate(('azimuth','range')):
                    total=h[f'total/{axis}_px'][:]; gross=h[f'gross/{axis}_px'][:]; residual=h[f'residual/{axis}_px'][:]
                    np.testing.assert_allclose(total[good],expected[k],atol=.18)
                    np.testing.assert_array_equal(total,gross+residual)
                    np.testing.assert_array_equal(gross[good],(1,-1)[k])
                owner=h['source_burst_index'][:]; dt=h['interval_seconds'][:]
                for i,key in enumerate(self.reference.bursts,1):
                    exact=(b.bursts[key].start-a.bursts[key].start).total_seconds()
                    np.testing.assert_array_equal(dt[owner==i],exact)
                metadata=json.loads(h.attrs['metadata_json'])
                self.assertEqual(metadata['stack_pair']['acquisition_dates'],[earlier,later])
                # Independent native invocation on exactly the same SLCs/centers.
                from glacier_offsets.stack_pairs import matching_template
                control=native_match(m['coreg']['slcs']['reference'],m['coreg']['slcs']['secondary'],
                    self.root/f'control-{earlier}-{later}',cfg['matching'],cfg,'cpu',_grid=matching_template(cfg,frame))
                for k,axis in enumerate(('azimuth','range')):
                    np.testing.assert_allclose(h[f'total/{axis}_px'][:][good],control['total'][...,k][good],atol=1e-5)
            geocode(cfg,directory,m)
            ds=gdal.Open(str(directory/'geo/range_velocity_m_per_day.tif'))
            map_grids.append((ds.RasterYSize,ds.RasterXSize,ds.GetGeoTransform()))
            interval=gdal.Open(str(directory/'geo/pair_interval_seconds.tif')).ReadAsArray()
            displacement=gdal.Open(str(directory/'geo/range_displacement_m.tif')).ReadAsArray()
            velocity=ds.ReadAsArray()
            np.testing.assert_allclose(velocity,displacement/(interval/86400),rtol=2e-6,equal_nan=True)
            sigma=gdal.Open(str(directory/'geo/range_displacement_sigma_m.tif')).ReadAsArray()
            self.assertTrue(np.isfinite(sigma).any())
            paths=[directory/'radar/offsets.h5',directory/'geo/range_velocity_m_per_day.tif']
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
            _,again=prepared_pair(cfg,self.plan,frame,a,b,dates,target)
            with patch('glacier_offsets.ampcor.native_match',side_effect=AssertionError('cache missed')):
                match(cfg,directory,again,native_pair=self.reference.bursts); geocode(cfg,directory,again)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
        self.assertEqual(len(set(map_grids)),1)
        # With no common bursts, matching emits a complete, explicitly empty field.
        last=self.acquisitions[-1]; last.bursts.clear()
        spec=date_specification(last,frame,SourceHashes())
        aligned=prepare_date(cfg,self.plan,last,frame,spec)
        dates[last.date]=assemble_date(cfg,last,frame,aligned)
        directory,m=prepared_pair(cfg,self.plan,frame,self.acquisitions[0],last,dates,target)
        with patch('glacier_offsets.ampcor.native_match',side_effect=AssertionError('native on empty support')):
            match(cfg,directory,m,native_pair=self.reference.bursts)
        self.assertEqual(m['matching_summary']['input_valid'],0)
        self.assertEqual(m['matching_summary']['quality_valid'],0)
        geocode(cfg,directory,m)
        self.assertEqual(m['geocoding']['valid_output_cells'],0)
        self.assertEqual(tuple(m['geocoding']['shape']),map_grids[0][:2])

    def test_nonreference_velocity_prior_uses_reference_orbit_and_exact_burst_intervals(self):
        from glacier_offsets.stack_pairs import matching_template, prepare_prior
        frame=self.geometry(); dates=self.dates(frame)
        cfg=self.effective()
        path=self.root/'velocity.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(path),100,100,1,gdal.GDT_Float32)
        ds.SetProjection(frame['coreg']['geometry_crs']); ds.SetGeoTransform((-1010000,500,0,-990000,0,-500))
        ds.GetRasterBand(1).Fill(1);ds.GetRasterBand(1).SetUnitType('m/day');ds=None
        cfg['gross_offsets']={'mode':'velocity','velocity':{'vx':{'path':str(path),'band':1},'vy':{'path':str(path),'band':1},
            'basis':'east_north','units':'m/day','interpolation':'bilinear','product':'synthetic','version':'1'}}
        a,b=self.acquisitions[0],self.acquisitions[-1]
        directory,m=prepared_pair(cfg,self.plan,frame,a,b,dates,common_map_grid(cfg,frame))
        seen=[]
        def prediction(lon,lat,height,east,north,grid,orbit,valid):
            index=len(seen); key=list(self.reference.bursts)[index]
            self.assertIs(orbit,self.reference.bursts[key].native.orbit)
            expected=(b.bursts[key].start-a.bursts[key].start).total_seconds()/86400
            np.testing.assert_allclose(east[np.isfinite(east)],expected)
            seen.append(expected)
            return np.zeros((*lon.shape,2)),valid
        with patch('glacier_offsets.velocity_prior.radar_prediction',side_effect=prediction):
            prior=prepare_prior(cfg,m,matching_template(cfg,frame),self.reference.bursts)
        self.assertEqual(len(seen),2)
        self.assertNotEqual(seen[0],seen[1])
        self.assertTrue(prior['valid'].any())
        self.assertEqual(prior['metadata']['geometry_reference_date'],'2020-01-04')
        seen.clear()
        with patch('glacier_offsets.velocity_prior.radar_prediction',side_effect=prediction):
            match(cfg,directory,m,native_pair=self.reference.bursts)
        with h5py.File(directory/'radar/offsets.h5') as h:
            good=h['valid'][:]
            self.assertTrue(good.any())
            np.testing.assert_allclose(h['total/azimuth_px'][:][good],3,atol=.18)
            np.testing.assert_allclose(h['total/range_px'][:][good],-3,atol=.18)
            metadata=json.loads(h['prior'].attrs['metadata_json'])
            self.assertEqual(metadata['geometry_reference_date'],'2020-01-04')

    def test_missing_burst_timestamp_is_invalid_without_prior_or_zero_fallback(self):
        from glacier_offsets.stack_pairs import matching_template, prepare_prior
        from glacier_offsets.assembly import source_indices
        frame=self.geometry()
        last=self.acquisitions[-1]
        key=list(last.bursts)[1]
        last.bursts[key]=Burst(key,2,None,last.bursts[key].native)
        dates=self.dates(frame)
        cfg=self.effective()
        directory,m=prepared_pair(cfg,self.plan,frame,self.acquisitions[0],last,dates,common_map_grid(cfg,frame))
        entry=m['assembly']['sources'][1]
        self.assertIsNone(entry['interval_seconds'])
        self.assertIsNone(dates[last.date]['burst_times'][key])
        self.assertNotIn(key,dates[last.date]['slcs'])
        match(cfg,directory,m,native_pair=self.reference.bursts)
        with h5py.File(directory/'radar/offsets.h5') as h:
            owner=h['source_burst_index'][:]
            self.assertFalse(h['input_valid'][:][owner==2].any())
            self.assertTrue(np.isnan(h['interval_seconds'][:][owner==2]).all())

    def test_append_partial_and_insert_dates_preserve_unrelated_scientific_files(self):
        self.cfg['storage'] = {'keep_merged_slcs': True}
        self.plan=self.make_plan(self.acquisitions[1:])
        with patch('glacier_offsets.stack_workflow.plan_stack',side_effect=lambda cfg,previous:self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack',return_value='cpu'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software):
            for stage in ('coregister','assemble','match','geocode'):
                self.assertEqual(execute_stack(self.cfg,stage),0)
            root=Path(self.cfg['output']['directory'])
            old_pair=root/'pairs/2020-01-04_2020-01-09'
            paths=[old_pair/'radar/offsets.h5',old_pair/'geo/range_velocity_m_per_day.tif',
                   root/'acquisitions/2020-01-04/slc.tif',root/'acquisitions/2020-01-09/slc.tif']
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
            self.acquisitions[0].bursts.pop(list(self.reference.bursts)[1])
            self.plan=self.make_plan(self.acquisitions)
            for stage in ('coregister','assemble','match','geocode'):
                self.assertEqual(execute_stack(self.cfg,stage),0)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
            # Insert a date and recompute a nearest-one graph. The former 4->9
            # edge stays on disk as inactive, with its scientific bytes intact.
            inserted=deepcopy(self.acquisitions[-1])
            inserted.scene.path=self.root/'inserted-safe';inserted.scene.path.write_text('inserted')
            inserted.scene.start=datetime(2020,1,6);inserted.scene.stop=datetime(2020,1,6,0,0,30)
            inserted.bursts={key:Burst(key,b.swath,b.start-timedelta(days=3),b.native)
                             for key,b in inserted.bursts.items()}
            for key in inserted.bursts:
                self.aligned_slcs[(inserted.date,key)]=self.aligned_slcs[(self.acquisitions[-1].date,key)]
            acquisitions=sorted([*self.acquisitions,inserted],key=lambda a:a.date)
            links=[(a.date,b.date) for a,b in zip(acquisitions,acquisitions[1:])]
            self.plan=self.make_plan(acquisitions,links)
            for stage in ('coregister','assemble','match','geocode'):
                self.assertEqual(execute_stack(self.cfg,stage),0)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
            catalog=json.loads((root/'stack_manifest.json').read_text())
            self.assertFalse(catalog['pairs']['2020-01-04_2020-01-09']['active'])
            self.assertEqual(catalog['reference_date'],'2020-01-04')
            self.assertEqual(self.reference_call.call_count,2)
            # 2 original, 1 partial appended, 2 inserted burst alignments.
            self.assertEqual(self.align_call.call_count,5)

    def test_source_hashes_once_per_invocation_and_frame_force_rejects_changes(self):
        with patch('glacier_offsets.stack_storage.checksum',wraps=checksum) as digest:
            hashes=SourceHashes()
            for _ in range(3):
                hashes(self.reference.scene.path)
                hashes(self.reference.orbit.path)
            self.assertEqual(digest.call_count,2)
        frame=self.geometry()
        self.assertEqual(self.geometry(force=True)['layout_identity'],frame['layout_identity'])
        changed=deepcopy(self.sources)
        changed[1][1]['coreg']['reference_grid']['sensing_start']+=.001
        with patch.object(self,'sources',changed), self.assertRaisesRegex(ValueError,'fractional'):
            self.geometry(force=True)
        from glacier_offsets.stack import read_catalog
        # Restore the complete frame, then check pin recovery without a catalog.
        self.geometry(force=True)
        recovered=read_catalog(self.cfg['output']['directory'])
        self.assertEqual(recovered['anchor'],self.plan.anchor)

    def test_workflow_incremental_inactive_pairs_failures_interrupts_and_force(self):
        # Exercise real preparation/assembly with mocked downstream science only.
        def fake_match(cfg,root,m,**kwargs):
            from glacier_offsets.ampcor import matching_specification
            m['match_identity']=identity(matching_specification(cfg,m))
            m['matching_summary']={'quality_valid':1}; m['stages']['match']='complete'
            (root/'radar').mkdir(exist_ok=True);(root/'radar/offsets.h5').write_text('science')
            write_json(root/'manifest.json',m)
        def fake_geocode(cfg,root,m,**kwargs):
            from glacier_offsets.geocode import geocoding_specification
            from glacier_offsets.products import identity
            m['geocode_identity']=identity(geocoding_specification(cfg,m))
            m['stages']['geocode']='complete'
            write_json(root/'manifest.json',m)
        def fake_diagnostics(cfg,root,m):
            write_json(root/'diagnostics/report.json',{})
            m['stages']['diagnostics']='complete'
        with patch('glacier_offsets.stack_workflow.plan_stack',side_effect=lambda cfg,previous:self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack',return_value='cpu'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_workflow.match',side_effect=fake_match), \
             patch('glacier_offsets.stack_workflow.geocode',side_effect=fake_geocode), \
             patch('glacier_offsets.stack_workflow.diagnostics',side_effect=fake_diagnostics):
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            pair=self.plan.pairs[0]; path=Path(self.cfg['output']['directory'])/'pairs'/f'{pair[0]}_{pair[1]}'/'radar/offsets.h5'
            dates_root=Path(self.cfg['output']['directory'])/'acquisitions'
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in dates_root.glob('*/slc.tif')}
            # Changing the graph retains old pairs as inactive without touching date products.
            self.plan=self.make_plan(self.acquisitions,links=[pair])
            self.assertEqual(execute_stack(self.cfg,'assemble'),0)
            report=json.loads((Path(self.cfg['output']['directory'])/'stack_manifest.json').read_text())
            self.assertEqual(sum(e['active'] for e in report['pairs'].values()),1)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in before})
            # A failure on one date leaves other dates complete and blocks only dependents.
            self.plan=self.make_plan(self.acquisitions)
            original=self.acquisitions[-1].error; self.acquisitions[-1].error='injected date failure'
            self.assertEqual(execute_stack(self.cfg,'run'),1)
            report=json.loads((Path(self.cfg['output']['directory'])/'stack_manifest.json').read_text())
            self.assertEqual(report['acquisitions'][self.reference.date]['status'],'complete')
            self.assertEqual(report['pairs']['2020-01-01_2020-01-09']['status'],'blocked')
            self.assertEqual(report['pairs']['2020-01-01_2020-01-04']['status'],'complete')
            self.acquisitions[-1].error=original
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            with patch('glacier_offsets.stack_workflow.geocode',side_effect=RuntimeError('map fails')):
                self.assertEqual(execute_stack(self.cfg,'run'),1)
            partial=read_manifest(path.parent.parent)
            self.assertIn('match_identity',partial)
            self.assertNotIn('geocode_identity',partial)
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            spacing=self.cfg['geocoding']['spacing_m']
            self.cfg['geocoding']['spacing_m']=spacing*2
            self.assertEqual(execute_stack(self.cfg,'diagnostics'),1)
            self.assertIn('match_identity',read_manifest(path.parent.parent))
            self.assertNotIn('geocode_identity',read_manifest(path.parent.parent))
            self.cfg['geocoding']['spacing_m']=spacing
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            with patch('glacier_offsets.stack_workflow.common_map_grid',side_effect=RuntimeError('map grid fails')):
                self.assertEqual(execute_stack(self.cfg,'geocode'),1)
            self.assertIn('match_identity',read_manifest(path.parent.parent))
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            def failure(cfg,root,m,**kwargs):
                if m['stack_pair']['acquisition_dates']==['2020-01-01','2020-01-04']:
                    raise RuntimeError('pair fails')
                return fake_match(cfg,root,m,**kwargs)
            with patch('glacier_offsets.stack_workflow.match',side_effect=failure):
                self.assertEqual(execute_stack(self.cfg,'match'),1)
            with patch('glacier_offsets.stack_storage.align_acquisition',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute_stack(self.cfg,'coregister',force=True)
            stale=read_manifest(path.parent.parent)
            self.assertNotIn('match_identity',stale)
            self.assertNotIn('coreg_identity',stale)
            self.assertEqual(execute_stack(self.cfg,'run'),0)


if __name__=='__main__':
    unittest.main()
