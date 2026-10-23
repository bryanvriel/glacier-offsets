"""Read-only status, stale dependencies, missing products and offline inputs."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from glacier_offsets.cli import main
from glacier_offsets.config import load_defaults
from glacier_offsets.ampcor import matching_specification
from glacier_offsets.geocode import geocoding_specification
from glacier_offsets.products import identity, write_json
from glacier_offsets.stack_config import load_status_config
from glacier_offsets.stack_ownership import POLICY
from glacier_offsets.stack_status import inspect_stack
from glacier_offsets.stack_storage import METHOD, assembly_specification
from glacier_offsets.stack_workflow import invalidate_pair


class StackStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_defaults()
        self.cfg.update(stack={'safe_directory':str(self.root/'offline-safe'),
                               'orbit_directory':str(self.root/'offline-orbits')},
                        output={'directory':str(self.root/'output')})
        self.cfg['inputs'].update(dem=str(self.root/'offline-dem.tif'),burst_id='burst')
        self.cfg['geocoding']['epsg'] = 3031
        self.output = Path(self.cfg['output']['directory'])
        self.config = self.root/'config.yaml'
        self.config.write_text(yaml.safe_dump(self.cfg))
        self.day1, self.day2 = '2020-01-01', '2020-01-07'
        self.pair = f'{self.day1}_{self.day2}'

    def files(self, *paths):
        for path in paths:
            path.parent.mkdir(parents=True,exist_ok=True); path.write_text('product')
        return {str(p):'recorded-hash' for p in paths}

    def prepare(self):
        spec = {'method':METHOD,'ownership_policy':POLICY,
                'anchor':{'reference_date':self.day1,'selection':{
                    k:self.cfg['inputs'][k] for k in ('polarization','subswaths','burst_id')}},
                'inputs':{'dem':{'path':self.cfg['inputs']['dem']}}}
        frame = {'geometry_identity':identity(spec),'specification':spec,
                 'products':self.files(self.output/'geometry/source.tif')}
        write_json(self.output/'geometry/manifest.json',frame)
        dates = {}
        for day in (self.day1,self.day2):
            root = self.output/'acquisitions'/day
            spec = {'method':METHOD,'geometry':frame['geometry_identity'],'day':day}
            m = {'alignment_identity':identity(spec),'specification':spec,
                 'alignment_products':self.files(root/'burst.tif')}
            m['products'] = dict(m['alignment_products'], **self.files(
                root/'slc.tif',root/'slc.tif.msk',root/'valid.tif'))
            m['assembly_identity'] = identity(assembly_specification(frame,m))
            dates[day] = m
            write_json(root/'manifest.json',m)
        root = self.output/'pairs'/self.pair
        m = {'coreg_identity':'pair-identity','stack_pair':{'acquisition_dates':[self.day1,self.day2]},
             'coreg_specification':{'stack':{'geometry':frame['geometry_identity'],
                  'dates':[dates[d]['assembly_identity'] for d in (self.day1,self.day2)]}},
             'stages':{k:'complete' for k in ('coregister','match','geocode','diagnostics')}}
        m['match_identity'] = identity(matching_specification(self.cfg,m))
        m['geocode_identity'] = identity(geocoding_specification(self.cfg,m))
        m['diagnostics_identity'] = identity({'match':m['match_identity'],'geocode':m['geocode_identity'],
                                            'method':'stack-diagnostics-v1'})
        self.files(root/'radar/offsets.h5',root/'diagnostics/report.json',*[root/'geo'/name for name in
            ('geocoding.json','range_displacement_m.tif','azimuth_displacement_m.tif',
             'range_velocity_m_per_day.tif','azimuth_velocity_m_per_day.tif',
             'valid_mask.tif','quality_valid_mask.tif','center_geometry.h5')])
        write_json(root/'manifest.json',m)
        write_json(self.output/'stack_manifest.json',{
            'acquisitions':{d:{'active':True} for d in dates},
            'pairs':{self.pair:{'active':True},'inactive-pair':{'active':False}}})
        return root

    def test_stages_cli_lists_processing_order_without_config(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['stack','stages']),0)
        text = out.getvalue()
        self.assertLess(text.index('coregister'),text.index('assemble'))
        self.assertLess(text.index('match'),text.index('geocode'))
        self.assertIn('diagnostics',text)

    def test_complete_status_offline_and_no_hash_discovery_native_or_writes(self):
        self.prepare()
        before = {p:p.stat().st_mtime_ns for p in self.output.rglob('*') if p.is_file()}
        with patch('glacier_offsets.products.checksum',side_effect=AssertionError('hashed inputs')), \
             patch('glacier_offsets.stack_cli.plan_stack',side_effect=AssertionError('discovered SAFE')), \
             patch('glacier_offsets.stack_workflow.execute_stack',side_effect=AssertionError('executed')), \
             contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['stack','status',str(self.config),'--json']),0)
        report = json.loads(out.getvalue())
        self.assertTrue(all(e['status']=='complete' for e in report['stages'].values()))
        self.assertEqual(report['stages']['coregister']['total'],3)
        self.assertEqual(len(report['pairs']),1)
        self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['stack','status',str(self.config)]),0)
        self.assertIn('✓ match',out.getvalue())

    def test_absent_output_is_pending_without_creating_it(self):
        report = inspect_stack(load_status_config(self.config))
        self.assertTrue(all(e['status']=='pending' for e in report['stages'].values()))
        self.assertFalse(self.output.exists())

    def test_intentionally_released_slcs_do_not_block_completed_pairs(self):
        from glacier_offsets.storage import METHOD as cleanup_method
        self.prepare()
        root = self.output/'acquisitions'/self.day1
        m = json.loads((root/'manifest.json').read_text())
        paths = [root/'slc.tif', root/'slc.tif.msk']
        m['slc_cleanup'] = {'method':cleanup_method, 'identity_key':'assembly_identity',
                            'identity':m['assembly_identity'], 'status':'released',
                            'files':{str(p):m['products'][str(p)] for p in paths}}
        for p in paths:p.unlink()
        write_json(root/'manifest.json', m)
        report = inspect_stack(self.cfg)
        self.assertTrue(all(e['status']=='complete' for e in report['stages'].values()))
        self.assertTrue(any('intentionally released' in note for note in report['notes']))
        # Cleanup evidence never excuses missing original alignment products.
        (root/'burst.tif').unlink()
        report = inspect_stack(self.cfg)
        self.assertEqual(report['pairs'][self.pair]['match']['status'], 'blocked')

    def test_matching_config_change_blocks_maps_and_diagnostics(self):
        self.prepare(); cfg=deepcopy(self.cfg);cfg['matching']['window']=[96,256]
        report=inspect_stack(cfg)
        self.assertEqual(report['stages']['coregister']['status'],'complete')
        self.assertEqual(report['pairs'][self.pair]['match']['status'],'stale')
        self.assertEqual(report['pairs'][self.pair]['geocode']['status'],'blocked')

    def test_geocoding_change_only_stales_maps_and_diagnostics(self):
        self.prepare();cfg=deepcopy(self.cfg);cfg['geocoding']['spacing_m']=100
        report=inspect_stack(cfg)
        self.assertEqual(report['pairs'][self.pair]['match']['status'],'complete')
        self.assertEqual(report['pairs'][self.pair]['geocode']['status'],'stale')
        self.assertEqual(report['pairs'][self.pair]['diagnostics']['status'],'blocked')

    def test_invalidated_match_does_not_trust_leftover_files_or_catalog(self):
        root=self.prepare(); invalidate_pair(root,'match')
        report=inspect_stack(self.cfg)
        self.assertTrue((root/'radar/offsets.h5').is_file())
        self.assertEqual(report['pairs'][self.pair]['match']['status'],'pending')
        self.assertEqual(report['pairs'][self.pair]['geocode']['status'],'blocked')

    def test_missing_and_corrupt_outputs_have_no_checkmark(self):
        root=self.prepare();(root/'geo/azimuth_displacement_m.tif').unlink()
        report=inspect_stack(self.cfg)
        self.assertEqual(report['pairs'][self.pair]['geocode']['status'],'incomplete')
        (root/'manifest.json').write_text('{interrupted JSON')
        report=inspect_stack(self.cfg)
        self.assertEqual(report['pairs'][self.pair]['match']['status'],'blocked')
        self.assertEqual(report['stages']['match']['completed'],0)

    def test_missing_date_source_blocks_all_dependent_work(self):
        self.prepare();(self.output/'acquisitions'/self.day2/'burst.tif').unlink()
        report=inspect_stack(self.cfg)
        self.assertEqual(report['acquisitions'][self.day2]['coregister']['status'],'incomplete')
        self.assertEqual(report['stages']['match']['completed'],0)

    def test_missing_catalog_recovers_saved_directories(self):
        self.prepare();(self.output/'stack_manifest.json').unlink()
        report=inspect_stack(self.cfg)
        self.assertEqual(report['pairs'][self.pair]['match']['status'],'complete')
        self.assertTrue(report['notes'])

    def test_legacy_reference_policy_and_backend_override_are_visible(self):
        self.prepare()
        path=self.output/'geometry/manifest.json'; frame=json.loads(path.read_text())
        frame['specification']['method']='legacy'
        frame['geometry_identity']=identity(frame['specification']);write_json(path,frame)
        self.assertEqual(inspect_stack(self.cfg)['reference']['status'],'stale')
        self.prepare()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(['stack','status',str(self.config),'--backend','cpu','--json']),0)
        self.assertEqual(json.loads(out.getvalue())['pairs'][self.pair]['match']['status'],'stale')

    def test_velocity_status_uses_saved_hashes_and_compares_declared_source_paths(self):
        root=self.prepare()
        cfg=deepcopy(self.cfg)
        cfg['gross_offsets']={'mode':'velocity','velocity':{
            k:{'path':str(self.root/f'offline-{k}.tif'),'units':'m/year'} for k in ('vx','vy')}}
        m=json.loads((root/'manifest.json').read_text())
        saved={k:cfg['gross_offsets']['velocity'][k] | {'sha256':f'saved-{k}'} for k in ('vx','vy')}
        m['velocity_provenance']=saved
        m['matching']={'velocity_provenance':saved}
        m['match_identity']=identity(matching_specification(cfg,m))
        m['geocode_identity']=identity(geocoding_specification(cfg,m))
        write_json(root/'manifest.json',m)
        with patch('glacier_offsets.products.checksum',side_effect=AssertionError('hashed velocity')):
            self.assertEqual(inspect_stack(cfg)['pairs'][self.pair]['match']['status'],'complete')
            cfg['gross_offsets']['velocity']['vx']['path']=str(self.root/'changed-source.tif')
            self.assertEqual(inspect_stack(cfg)['pairs'][self.pair]['match']['status'],'stale')

    def test_empty_pair_graph_does_not_show_completed_pair_stages(self):
        self.prepare()
        path=self.output/'stack_manifest.json';catalog=json.loads(path.read_text())
        catalog['pairs']={};write_json(path,catalog)
        report=inspect_stack(self.cfg)
        self.assertEqual(report['stages']['match']['status'],'not_applicable')
        self.assertEqual(report['stages']['match']['total'],0)

    def test_status_rejects_force_and_does_not_create_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(['stack','status',str(self.config),'--force'])
        self.assertEqual(error.exception.code,2)
        self.assertFalse(self.output.exists())


if __name__=='__main__':
    unittest.main()
