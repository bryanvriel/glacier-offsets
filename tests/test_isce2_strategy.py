"""ISCE2 validity/overwrite contracts and optional actual native integration."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import matching_specification, native_match
from glacier_offsets.config import load_defaults
from glacier_offsets.isce2_matching import child_environment, validate_settings
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import checksum, identity
from glacier_offsets.stack_isce2 import (POLICY, SWATH_MIDPOINT_POLICY, adjusted_window, date_validity,
                                          overwrite_ownership, policy, prepare_common_layout)
from glacier_offsets.stack_storage import (assemble_date, assembly_specification, prepare_date,
                                           date_specification, read_manifest, write_reference_layout)
from glacier_offsets.stack_workflow import execute_stack
import test_stack_science


def runtime_settings():
    return {key: os.environ.get('GLACIER_ISCE2_'+name) for key, name in
            [('python', 'PYTHON'), ('packages', 'PACKAGES'), ('library_path', 'LIBRARY_PATH')]}


class ISCE2ValidityTests(unittest.TestCase):
    def test_swath_midpoints_keep_later_bursts_and_partition_staggered_overlap(self):
        sources = [{'burst_id': name, 'offset': [dy, dx], 'valid_window': [4,188,4,380]}
                   for name,dy,dx in [('a_iw1',0,0),('b_iw1',128,0),
                                      ('a_iw2',64,128),('b_iw2',192,128),
                                      ('a_iw3',64,256)]]
        ordered, metadata = overwrite_ownership(sources[::-1], swath_midpoint=True)
        owner = np.zeros((400,650), 'uint16');count=np.zeros(owner.shape, 'uint8')
        union=np.zeros(owner.shape,bool)
        for i,e in enumerate(ordered,1):
            dy,dx=e['offset'];a,b,c,d=e['valid_window'];union[a+dy:b+dy,c+dx:d+dx]=True
            for a,b,c,d in e['ownership_windows']:
                owner[a+dy:b+dy,c+dx:d+dx]=i
                count[a+dy:b+dy,c+dx:d+dx]+=1
        np.testing.assert_array_equal(count, union.astype('uint8'))
        self.assertEqual(owner[140,200],2)  # Later IW1 burst already owns, before its old midpoint.
        self.assertEqual(owner[140,300],3)
        self.assertEqual(owner[200,300],4)
        self.assertEqual(owner[140,450],5)
        self.assertEqual(metadata['method'],SWATH_MIDPOINT_POLICY['method'])
        self.assertTrue(all('cut_global_sample' in c and 'cut_global_line' not in c for c in metadata['cuts']))
        sources[2]['offset'][1]=370
        with self.assertRaisesRegex(ValueError,'overlap has'):
            overwrite_ownership(sources[:3],swath_midpoint=True)

    def test_upstream_v2_sign_cases_and_zero_preserves_reference(self):
        sec = [10, 190, 20, 380]
        self.assertEqual(adjusted_window(sec, (200, 400), [[1.2, 3.8], [1.2, 3.8]]), [11,183,21,373])
        self.assertEqual(adjusted_window(sec, (200, 400), [[-3.8,-1.2], [-3.8,-1.2]]), [18,190,28,380])
        self.assertEqual(adjusted_window(sec, (200, 400), [[-3.8,1.2], [-3.8,1.2]]), [18,184,28,374])
        reference = [12,188,24,376]
        self.assertEqual(adjusted_window(sec, (200,400), [[0,0],[0,0]], reference), reference)
        with self.assertRaisesRegex(ValueError, 'no valid'):
            adjusted_window(sec, (200,400), [[-300,300],[0,0]])

    def test_engine_rejects_unsupported_requests_instead_of_silent_fallback(self):
        cfg = load_defaults()
        cfg['matching'].update(engine='isce2',backend='cpu')
        validate_settings(cfg)
        for key, value in [('backend','gpu'), ('raw_oversampling',4), ('window',[31,64]), ('correlation_threshold',.2)]:
            changed = deepcopy(cfg);changed['matching'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_settings(changed)

    def test_optional_engine_settings_preserve_existing_isce3_identities(self):
        cfg=load_defaults();old=deepcopy(cfg)
        old['matching'].pop('engine');old['matching'].pop('isce2')
        manifest={'coreg_identity':'unchanged','stack_pair':{}}
        self.assertEqual(matching_specification(cfg,manifest),matching_specification(old,manifest))

    def test_external_interpreter_environment_does_not_leak_dependencies_or_modify_parent(self):
        with patch.dict(os.environ,{'PYTHONPATH':'/python312/deps','OMP_NUM_THREADS':'8'}):
            env=child_environment(('/different/python',None,None))
            self.assertNotIn('PYTHONPATH',env)
            self.assertEqual(env['OMP_NUM_THREADS'],'1')
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'8')
            self.assertEqual(os.environ['PYTHONPATH'],'/python312/deps')
            self.assertEqual(child_environment(('/different/python','/isce2/packages',None))['PYTHONPATH'],'/isce2/packages')


class ISCE2LayoutTests(unittest.TestCase):
    # Reuse the real synthetic radar/geometry fixture, without inheriting its
    # legacy-policy tests or mocking the new validity computation.
    setUp = test_stack_science.StackScienceTests.setUp
    make_plan = test_stack_science.StackScienceTests.make_plan
    reference_product = test_stack_science.StackScienceTests.reference_product
    geometry = test_stack_science.StackScienceTests.geometry
    effective = test_stack_science.StackScienceTests.effective

    def align_product(self, cfg, reference, burst, root, coreg):
        source = self.aligned_slcs[(burst.start.date().isoformat(),burst.id)]
        root.mkdir(parents=True, exist_ok=True)
        output = root/'slc.tif'
        shutil.copyfile(source, output)
        shape = coreg['reference_grid']['length'], coreg['reference_grid']['width']
        for name in ('azimuth.off','range.off'):
            ds = gdal.GetDriverByName('GTiff').Create(str(root/name),shape[1],shape[0],1,gdal.GDT_Float32)
            ds.GetRasterBand(1).Fill(.25)
            ds = None
        return str(output)

    def enable(self):
        self.cfg['stack']['processing'] = 'isce2_merge_first'
        self.cfg['matching']['engine'] = 'isce2'
        self.plan.anchor['selection'] = {'selected_ids':list(self.reference.bursts),
            'polarization':self.cfg['inputs']['polarization'], 'burst_id':self.cfg['inputs']['burst_id'],
            'subswaths':self.cfg['inputs']['subswaths'], 'aoi':{'specification':self.cfg.get('aoi')}}

    def aligned_dates(self, frame):
        return {a.date: prepare_date(self.cfg,self.plan,a,frame,date_specification(a,frame,self.hashes))
                for a in self.plan.acquisitions}

    def test_overwrite_matches_ordered_rectangle_copies_and_geometry_at_staggered_swaths(self):
        sources = []
        for name, dy, dx in [('a_iw1',0,0),('b_iw1',128,0),('a_iw2',64,32),('b_iw2',192,32)]:
            coreg = deepcopy(self.sources[0][1]['coreg'])
            coreg['reference_grid']['sensing_start'] = dy/500
            coreg['reference_grid']['starting_range'] += dx*2
            sources.append({'burst_id':name,'coreg':coreg,'valid_window':[4,188,4,380]})
        frame = write_reference_layout(self.root/'overwrite', sources[::-1], ownership_policy=POLICY)
        owner = gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        expected = np.zeros(owner.shape,'uint16')
        expected_x = np.full(owner.shape,np.nan)
        for i,e in enumerate(frame['sources'],1):
            y0,y1,x0,x1=e['valid_window'];dy,dx=e['offset']
            expected[y0+dy:y1+dy,x0+dx:x1+dx] = i
            expected_x[y0+dy:y1+dy,x0+dx:x1+dx] = gdal.Open(e['coreg']['geometry']['x']).ReadAsArray()[y0:y1,x0:x1]
        np.testing.assert_array_equal(owner,expected)
        np.testing.assert_array_equal(gdal.Open(frame['coreg']['geometry']['x']).ReadAsArray(),expected_x)
        # IW2 owns the full overlap starting at its first valid sample.
        self.assertEqual(owner[100,36],3)
        self.assertEqual(owner[140,20],2)

    def test_common_validity_is_stack_wide_and_partial_dates_do_not_fill_owners(self):
        self.enable()
        frame = self.geometry();dates = self.aligned_dates(frame)
        for d,m in dates.items():
            for key,v in m['valid_windows'].items():
                self.assertEqual(v['window'],[4,188,4,380] if d==self.reference.date else [8,184,8,376])
        partial = self.plan.acquisitions[-1]
        missing = list(dates[partial.date]['slcs'])[1]
        dates[partial.date]['slcs'].pop(missing)
        # Exclude the entire incomplete date from the common intersection, as
        # ISCE2 does; retain its available imagery for supported output regions.
        dates[partial.date]['valid_windows'][list(dates[partial.date]['slcs'])[0]]['window']=[40,150,40,300]
        frame = prepare_common_layout(self.cfg,self.plan,frame,dates)
        self.assertEqual(frame['common_validity']['excluded_partial_dates'],[partial.date])
        for e in frame['sources']:
            self.assertEqual(e['valid_window'],[8,184,8,376])
        p = assemble_date(self.cfg,partial,frame,dates[partial.date])
        ds = gdal.Open(p['slc']);slc = ds.ReadAsArray()
        owner = gdal.Open(frame['coreg']['source_index']).ReadAsArray()
        self.assertTrue((slc[owner==2]==0).all())
        self.assertTrue((owner[140:150,20:300]==2).all())
        # Recomputing with identical inputs neither writes geometry nor changes identities.
        path = Path(frame['coreg']['source_index']);before=(checksum(path),path.stat().st_mtime_ns)
        same = prepare_common_layout(self.cfg,self.plan,frame,dates)
        self.assertEqual(same['layout_identity'],frame['layout_identity'])
        self.assertEqual(before,(checksum(path),path.stat().st_mtime_ns))
        dates[partial.date]['alignment_identity']='new-partial-inventory-same-valid-region'
        same_inventory = prepare_common_layout(self.cfg,self.plan,frame,dates)
        self.assertEqual(same_inventory['layout_identity'],frame['layout_identity'])
        self.assertEqual(before,(checksum(path),path.stat().st_mtime_ns))
        self.assertEqual(assembly_specification(frame,dates[self.reference.date]),
                         assembly_specification(same_inventory,dates[self.reference.date]))
        old_id = identity(assembly_specification(frame,dates[self.reference.date]))
        dates[self.acquisitions[0].date]['valid_windows'][frame['sources'][0]['burst_id']]['window']=[12,180,12,372]
        narrowed = prepare_common_layout(self.cfg,self.plan,frame,dates)
        self.assertNotEqual(identity(assembly_specification(narrowed,dates[self.reference.date])),old_id)
        self.assertEqual(narrowed['map_grid'],frame['map_grid'])

    def test_changed_swath_policy_rebuilds_layout_even_with_identical_common_windows(self):
        self.enable();frame=self.geometry();dates=self.aligned_dates(frame)
        original=prepare_common_layout(self.cfg,self.plan,frame,dates)
        self.cfg['stack']['swath_merge']='midpoint'
        changed=prepare_common_layout(self.cfg,self.plan,original,dates)
        self.assertEqual(changed['common_validity']['windows'],original['common_validity']['windows'])
        self.assertEqual(changed['ownership']['method'],SWATH_MIDPOINT_POLICY['method'])
        self.assertNotEqual(changed['layout_identity'],original['layout_identity'])
        self.assertEqual(changed['map_grid'],original['map_grid'])

    def test_common_policy_rebuild_preserves_offsets_validity_and_restart(self):
        from glacier_offsets.stack_rebuild import rebuild_layout
        self.enable();self.cfg['matching'].update(engine='isce3',deramp=0)
        with patch('glacier_offsets.stack_workflow.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_rebuild.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack'), \
             patch('glacier_offsets.stack_rebuild.validate_stack'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_rebuild.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_workflow.select_pair'):
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            root=Path(self.cfg['output']['directory'])
            originals={a.date:read_manifest(root/'acquisitions'/a.date) for a in self.acquisitions}
            native={Path(p):(checksum(p),Path(p).stat().st_mtime_ns) for d in originals.values()
                    for p in d['alignment_products']}
            self.cfg['stack']['swath_merge']='midpoint'
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            self.assertEqual(rebuild_layout(self.cfg,dry_run=True)['status'],'ready')
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in before})
            import glacier_offsets.stack_rebuild as rebuilding
            write = rebuilding.write_json
            def interrupt(path,data):
                if str(path).endswith('/acquisitions/2020-01-04/manifest.json'):
                    raise KeyboardInterrupt()
                return write(path,data)
            with patch('glacier_offsets.stack_rebuild.write_json',side_effect=interrupt):
                for _ in range(2):
                    with self.assertRaises(KeyboardInterrupt):
                        rebuild_layout(self.cfg)
            self.assertEqual(rebuild_layout(self.cfg,dry_run=True)['status'],'resume_ready')
            self.assertEqual(rebuild_layout(self.cfg)['status'],'complete')
            for day,original in originals.items():
                rebuilt=read_manifest(root/'acquisitions'/day)
                self.assertEqual(rebuilt['alignment_products'],original['alignment_products'])
                self.assertEqual(rebuilt['valid_windows'],original['valid_windows'])
                self.assertNotIn('assembly_identity',rebuilt)
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            frame=read_manifest(root/'geometry')
            self.assertEqual(frame['ownership']['method'],SWATH_MIDPOINT_POLICY['method'])
            self.assertEqual(frame['common_validity']['method'],SWATH_MIDPOINT_POLICY['method'])
            self.assertEqual(rebuild_layout(self.cfg)['status'],'already_current')
            with patch('glacier_offsets.stack_pairs.match_prepared',side_effect=AssertionError('restart reruns matching')):
                self.assertEqual(execute_stack(self.cfg,'run'),0)
            self.assertEqual(self.reference_call.call_count,2)
            self.assertEqual(self.align_call.call_count,4)
            self.assertEqual(native,{p:(checksum(p),p.stat().st_mtime_ns) for p in native})

    def test_common_policy_rebuild_reuses_completed_bursts_of_incomplete_date(self):
        from glacier_offsets.stack_rebuild import rebuild_layout
        from glacier_offsets.products import write_json
        self.enable();self.cfg['matching']['engine']='isce3'
        with patch('glacier_offsets.stack_workflow.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_rebuild.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack'), \
             patch('glacier_offsets.stack_rebuild.validate_stack'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_rebuild.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_workflow.select_pair'):
            self.assertEqual(execute_stack(self.cfg,'coregister'),0)
            root=Path(self.cfg['output']['directory'])
            day=self.acquisitions[0].date
            date=read_manifest(root/'acquisitions'/day)
            date.pop('alignment_identity');date['status']='aligning'
            write_json(root/'acquisitions'/day/'manifest.json',date)
            self.cfg['stack']['swath_merge']='midpoint'
            self.assertEqual(rebuild_layout(self.cfg)['status'],'complete')
            self.assertEqual(execute_stack(self.cfg,'coregister'),0)
            self.assertEqual(self.reference_call.call_count,2)
            self.assertEqual(self.align_call.call_count,4)
            self.assertTrue(read_manifest(root/'acquisitions'/day)['valid_windows'])

    def test_offset_products_are_verified_and_bad_offset_rasters_fail(self):
        self.enable();frame=self.geometry();dates=self.aligned_dates(frame)
        first=self.acquisitions[0];key=frame['sources'][0]['burst_id']
        metadata=dates[first.date]['valid_windows'][key]
        self.assertEqual(metadata['offset_extrema'],[[.25,.25],[.25,.25]])
        for p in metadata['offset_paths']:
            self.assertIn(str(Path(p).resolve()),dates[first.date]['alignment_products'])
        path=Path(metadata['offset_paths'][0]);ds=gdal.Open(str(path),gdal.GA_Update)
        ds.GetRasterBand(1).Fill(-1.e6);ds=None
        with self.assertRaisesRegex(ValueError,'No usable'):
            date_validity(first.bursts[key],dates[first.date]['slcs'][key],(192,384))

    def test_failed_alignment_blocks_common_layout_without_using_smaller_inventory(self):
        self.enable()
        with patch('glacier_offsets.stack_workflow.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_workflow.select_pair'), \
             patch('glacier_offsets.isce2_matching.provenance',return_value={'test':'runtime'}), \
             patch('glacier_offsets.stack_workflow.prepare_date',side_effect=RuntimeError('alignment failed')):
            self.assertEqual(execute_stack(self.cfg,'coregister'),1)
        self.assertNotIn('common_validity_identity',read_manifest(Path(self.cfg['output']['directory'])/'geometry'))

    @unittest.skipUnless(os.environ.get('GLACIER_ISCE2_PYTHON'),'Set GLACIER_ISCE2_PYTHON for actual native ISCE2 integration')
    def test_actual_native_stack_run_geocode_and_restart(self):
        self.enable();self.cfg['matching']['isce2']=runtime_settings()
        with patch('glacier_offsets.stack_workflow.plan_stack',return_value=self.plan), \
             patch('glacier_offsets.stack_workflow.validate_stack'), \
             patch('glacier_offsets.stack_workflow.inspect_environment',return_value=self.software), \
             patch('glacier_offsets.stack_workflow.select_pair'):
            self.assertEqual(execute_stack(self.cfg,'run'),0)
            root=Path(self.cfg['output']['directory']);frame=read_manifest(root/'geometry')
            self.assertEqual(frame['ownership']['method'],POLICY['method'])
            paths=[*root.glob('pairs/*/radar/offsets.h5'),*root.glob('pairs/*/geo/*velocity*.tif')]
            before={p:(checksum(p),p.stat().st_mtime_ns) for p in paths}
            for (a,b),(ta,tb) in zip(self.plan.pairs,[(self.translations[0],self.translations[1]),
                (self.translations[0],self.translations[2]),(self.translations[1],self.translations[2])]):
                with h5py.File(root/'pairs'/f'{a}_{b}'/'radar/offsets.h5') as h:
                    good=h['valid'][:]&~h['seam_crossing'][:]
                    self.assertGreater(int(good.sum()),20)
                    for axis,k in [('azimuth',0),('range',1)]:
                        self.assertAlmostEqual(float(np.median(h[f'total/{axis}_px'][:][good])),tb[k]-ta[k],delta=.15)
                        np.testing.assert_allclose(h[f'total/{axis}_px'][:][h['valid'][:]],tb[k]-ta[k],atol=.2)
                    self.assertTrue((h['valid'][:]&h['seam_crossing'][:]).any())
                    self.assertTrue(np.isnan(h['correlation_peak'][:]).all())
            with patch('glacier_offsets.isce2_matching.invoke',side_effect=AssertionError('native rerun')):
                self.assertEqual(execute_stack(self.cfg,'run'),0)
            self.assertEqual(before,{p:(checksum(p),p.stat().st_mtime_ns) for p in paths})
            self.assertEqual(self.reference_call.call_count,2)
            self.assertEqual(self.align_call.call_count,4)
            from glacier_offsets.stack_status import inspect_stack
            with patch('glacier_offsets.isce2_matching.provenance',side_effect=AssertionError('status native probe')):
                report=inspect_stack(dict(self.cfg,matching=self.cfg['matching']|{'engine':'isce2'}))
            self.assertEqual(report['stages']['match']['status'],'complete',json.dumps(report,indent=2))
            # Switch only the matcher on this pinned stack. No native ISCE2
            # probe or COMPASS work is needed; released assemblies are rebuilt
            # from the retained aligned bursts with exactly the same ownership.
            geometry_before = {p: (checksum(p), p.stat().st_mtime_ns)
                               for p in (root/'geometry/manifest.json', Path(frame['coreg']['source_index']))}
            old_matches = {p: read_manifest(p)['match_identity'] for p in root.glob('pairs/*')}
            self.cfg['matching'].update(engine='isce3', deramp=0)
            with patch('glacier_offsets.isce2_matching.provenance',side_effect=AssertionError('ISCE2 probe after switch')):
                self.assertEqual(execute_stack(self.cfg,'run'),0)
                for p, previous in old_matches.items():
                    self.assertNotEqual(read_manifest(p)['match_identity'], previous)
                    with h5py.File(p/'radar/offsets.h5') as h:
                        self.assertTrue(np.isfinite(h['correlation_peak'][:][h['valid'][:]]).all())
                report=inspect_stack(self.cfg)
                self.assertEqual(report['stages']['match']['status'],'complete',json.dumps(report,indent=2))
                with patch('glacier_offsets.stack_pairs.match_prepared',side_effect=AssertionError('ISCE3 restart reran matcher')):
                    self.assertEqual(execute_stack(self.cfg,'run'),0)
            self.assertEqual(geometry_before,{p:(checksum(p),p.stat().st_mtime_ns) for p in geometry_before})
            self.assertEqual(self.reference_call.call_count,2)
            self.assertEqual(self.align_call.call_count,4)

    @unittest.skipUnless(os.environ.get('GLACIER_ISCE2_PYTHON'),'Set GLACIER_ISCE2_PYTHON for actual native ISCE2 integration')
    def test_actual_native_single_window_nonzero_gross_and_no_isce3_fallback(self):
        cfg=self.effective();cfg['matching'].update(engine='isce2',isce2=runtime_settings())
        cfg['gross_offsets']={'mode':'constant','constant':{'azimuth_px':1,'range_px':-2}}
        pair=self.sources[0][1]['coreg']['slcs']
        grid=MatchingGrid(np.array([64]),np.array([128]),(32,64),(8,12),(16,32))
        with patch('glacier_offsets.environment.select_backend',side_effect=AssertionError('ISCE3 fallback')):
            r=native_match(pair['reference'],pair['secondary'],self.root/'single',cfg['matching'],cfg,_grid=grid)
        self.assertTrue(r['valid'][0,0])
        np.testing.assert_allclose(r['total'][0,0],[2,-3],atol=.1)
        np.testing.assert_array_equal(r['gross'][0,0],[1,-2])
        np.testing.assert_allclose(r['gross']+r['residual'],r['total'])
        np.testing.assert_array_equal(r['grid'].centers,grid.centers)

    @unittest.skipUnless(os.environ.get('GLACIER_ISCE2_PYTHON'),'Set GLACIER_ISCE2_PYTHON for actual native ISCE2 integration')
    def test_actual_varying_prior_preserves_missing_support_and_totals(self):
        cfg=self.effective();cfg['matching'].update(engine='isce2',isce2=runtime_settings())
        cfg['gross_offsets']={'mode':'velocity'}
        pair=self.sources[0][1]['coreg']['slcs']
        grid=MatchingGrid(np.array([64,80]),np.array([96,128,160]),(32,64),(8,12),(16,32))
        gross=np.array([[[0,-1],[1,-2],[2,-3]],[[1,-2],[0,-1],[2,-3]]],dtype='int32')
        good=np.ones(grid.shape,bool);good[1,1]=False;gross[1,1]=np.iinfo('int32').min
        prior={'grid':grid,'integer_px':gross,'valid':good,'rejection_flags':np.zeros(grid.shape,'uint8')}
        r=native_match(pair['reference'],pair['secondary'],self.root/'varying',cfg['matching'],cfg,prior=prior)
        np.testing.assert_array_equal(r['valid'],good)
        np.testing.assert_allclose(r['total'][good],np.tile([2,-3],(int(good.sum()),1)),atol=.12)
        self.assertTrue(np.isnan(r['gross'][1,1]).all())
        self.assertTrue(np.isnan(r['total'][1,1]).all())


if __name__=='__main__':
    unittest.main()
