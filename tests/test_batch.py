"""Batch failure isolation, native-pair reuse and scientific cache preservation."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shapely.geometry import box
from pyproj import CRS

from glacier_offsets.batch import execute_batch, effective_config, validate_batch, validate_burst
from glacier_offsets.products import checksum, identity, input_provenance, write_json
from glacier_offsets.config import load_defaults
from test_aoi import pair


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root/'input'; source.write_text('shared')
        self.cfg = load_defaults()
        self.cfg['burst_merge']['enabled']=False  # These tests exercise the retained independent-burst path.
        self.cfg.update(aoi={'bbox':[0,0,3,3]}, output={'directory':str(self.root/'work')})
        self.cfg['inputs'].update({k:str(source) for k in ('reference_safe','secondary_safe','reference_orbit','secondary_orbit','dem')})
        self.cfg['inputs']['polarization']='HH'
        self.rows=(pair('a'),pair('b',swath=2))

    def selected(self, rows=None):
        rows=self.rows if rows is None else rows
        report={'selected_ids':[p[0].id for p in rows],'common_count':2,'selection_identity':identity([p[0].id for p in rows])}
        return rows,report

    def mocks(self, rows=None):
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch('glacier_offsets.batch.select_aoi',return_value=self.selected(rows)))
        stack.enter_context(patch('glacier_offsets.batch.dem_extent',return_value=(box(-1,-1,5,5),CRS.from_epsg(4326))))
        software=stack.enter_context(patch('glacier_offsets.batch.inspect_environment',return_value={'packages':{}}))
        return stack,software

    def test_failure_continues_restart_selection_change_and_force(self):
        stack,software=self.mocks()
        seen=[]
        def stage(cfg, command, force, context):
            key=cfg['inputs']['burst_id'];seen.append((key,force))
            self.assertNotIn('aoi',cfg)
            self.assertIs(context.pair(key)[0].native,self.rows[0 if key=='a' else 1][0].native)
            self.assertEqual(context.provenance,input_provenance(self.cfg))
            if key=='a' and len(seen)==1:raise RuntimeError('individual failure')
            root=Path(cfg['output']['directory'])/key
            path=root/'radar/science';path.parent.mkdir(parents=True,exist_ok=True)
            if force or not path.exists():path.write_text('science '+key)
            write_json(root/'manifest.json',{'matching_summary':{'quality_valid':0,'windows':3},'stages':{'match':'complete'}})
        with patch('glacier_offsets.pipeline.execute',side_effect=stage):
            self.assertEqual(execute_batch(self.cfg,'match'),1)
        report=json.loads((self.root/'work/batch_manifest.json').read_text())
        self.assertEqual(report['bursts']['a']['status'],'failed')
        self.assertEqual(report['bursts']['b']['status'],'complete')
        self.assertEqual(report['bursts']['b']['scientific_coverage'],'empty')
        kept=self.root/'work/b/radar/science'; old=(checksum(kept),kept.stat().st_mtime_ns)
        with patch('glacier_offsets.pipeline.execute',side_effect=stage):
            self.assertEqual(execute_batch(self.cfg,'match'),0)
        self.assertEqual((checksum(kept),kept.stat().st_mtime_ns),old)
        stack.close()
        self.mocks((self.rows[0],))
        self.cfg['aoi']['bbox']=[0,0,1,1]
        with patch('glacier_offsets.pipeline.execute',side_effect=stage):
            self.assertEqual(execute_batch(self.cfg,'match',force=True),0)
        report=json.loads((self.root/'work/batch_manifest.json').read_text())
        self.assertEqual(list(report['bursts']),['a'])
        self.assertEqual(seen[-1],('a',True))
        self.assertEqual((checksum(kept),kept.stat().st_mtime_ns),old)
        self.assertEqual(software.call_count,2)
        self.assertIn('aoi',self.cfg)

    def test_context_hashes_inputs_once_and_defends_serialized_metadata(self):
        self.mocks()
        with patch('glacier_offsets.batch.input_provenance',wraps=input_provenance) as provenance, \
                patch('glacier_offsets.pipeline.execute') as stage:
            self.assertEqual(execute_batch(self.cfg,'coregister'),0)
            provenance.assert_called_once()
            a,b=[c.kwargs['context'] for c in stage.call_args_list]
            self.assertIs(a,b)
            mutable=a.provenance;mutable.clear()
            self.assertTrue(a.provenance)

    def test_empty_preflight_and_interruption(self):
        stack,software=self.mocks(())
        with patch('glacier_offsets.batch.input_provenance') as provenance:
            self.assertEqual(execute_batch(self.cfg,'run'),2)
            provenance.assert_not_called();software.assert_not_called()
        self.assertEqual(json.loads((self.root/'work/selection.json').read_text())['selected_ids'],[])
        stack.close();self.mocks()
        with patch('glacier_offsets.pipeline.execute',side_effect=KeyboardInterrupt) as stage:
            with self.assertRaises(KeyboardInterrupt):execute_batch(self.cfg,'run')
            stage.assert_called_once()
        report=json.loads((self.root/'work/batch_manifest.json').read_text())
        self.assertEqual(report['outcome'],'interrupted')
        self.assertEqual(report['bursts']['b']['status'],'pending')

    def test_dem_and_sampling_failures_are_burst_specific(self):
        stack,_=self.mocks()
        self.rows[0][0].native.as_isce3_radargrid=lambda: type('grid',(),{'prf':100,'range_pixel_spacing':2.3})()
        with patch('glacier_offsets.pipeline.execute') as stage:
            self.assertEqual(execute_batch(self.cfg,'run'),1)
            self.assertEqual(stage.call_count,1)
            self.assertEqual(stage.call_args.args[0]['inputs']['burst_id'],'b')
        stack.close()
        # A broad geographic DEM retains its continuous raster extent.
        from glacier_offsets.batch import dem_extent
        from osgeo import gdal
        path=self.root/'wide-dem.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(path),34,16,1,gdal.GDT_Float32)
        ds.SetProjection(CRS.from_epsg(4326).to_wkt());ds.SetGeoTransform((-170,10,0,80,0,-10));ds=None
        cfg=deepcopy(self.cfg);cfg['inputs']['dem']=str(path)
        validate_burst(effective_config(cfg,'b'),self.rows[1],dem_extent(cfg))
        with self.assertRaisesRegex(ValueError,'DEM extent'):
            validate_burst(effective_config(self.cfg,'b'),self.rows[1],(box(0,0,1,1),CRS.from_epsg(4326)))

    def test_validation_read_only_and_aggregates_errors(self):
        self.mocks()
        with patch('glacier_offsets.batch.validate_burst',side_effect=[ValueError('a fail'),ValueError('b fail')]) as validate:
            with self.assertRaisesRegex(ValueError,'a: a fail; b: b fail'):validate_batch(self.cfg)
            self.assertEqual(validate.call_count,2)
        self.assertFalse((self.root/'work').exists())

    def test_existing_stage_prerequisites_fail_per_burst(self):
        self.mocks()
        self.assertEqual(execute_batch(self.cfg,'geocode'),1)
        report=json.loads((self.root/'work/batch_manifest.json').read_text())
        for entry in report['bursts'].values():
            self.assertEqual(entry['status'],'failed')
            self.assertIn('run coregister first',entry['error'])

    def test_stage_engine_reuses_context_pair_for_coreg_and_velocity_match(self):
        from glacier_offsets.pipeline import execute
        from glacier_offsets.batch import InvocationContext
        import glacier_offsets.ampcor as ampcor
        context=InvocationContext(self.rows,json.dumps(input_provenance(self.cfg)),json.dumps({'packages':{}}))
        cfg=effective_config(self.cfg,'b')
        manifest={'coreg_identity':'fixed','coreg_specification':{'inputs':context.provenance,'polarization':'HH'}}
        root=self.root/'work/b';write_json(root/'manifest.json',manifest)
        with patch('glacier_offsets.pipeline.coregister',return_value=(root,manifest)) as coreg:
            execute(cfg,'coregister',context=context)
            self.assertIs(coreg.call_args.kwargs['context'],context)
        with patch('glacier_offsets.pipeline.match') as match,patch('glacier_offsets.pipeline.input_provenance',side_effect=AssertionError('rehash')):
            execute(cfg,'match',context=context)
            self.assertIs(match.call_args.kwargs['native_pair'],self.rows[1])

    def test_cli_batch_exit_codes_and_read_only_validation(self):
        from glacier_offsets.cli import main
        from contextlib import redirect_stdout
        import io
        with patch('glacier_offsets.cli.load_config',return_value=self.cfg), \
                patch('glacier_offsets.batch.execute_batch') as execute:
            for status in (0,1,2):
                execute.return_value=status
                self.assertEqual(main(['match','fixture.yaml']),status)
            execute.side_effect=ValueError('shared preflight failure')
            self.assertEqual(main(['match','fixture.yaml']),2)
            execute.side_effect=KeyboardInterrupt
            self.assertEqual(main(['match','fixture.yaml']),130)
            execute.reset_mock();execute.side_effect=None
            self.cfg['geocoding']['spacing_m']=-1
            self.assertEqual(main(['geocode','fixture.yaml']),2)
            execute.assert_not_called()

    def test_coreg_cache_identity_is_independent_of_aoi_context(self):
        from glacier_offsets.batch import InvocationContext
        from glacier_offsets.compass_coreg import coregister
        cfg=effective_config(self.cfg,'a')
        provenance=input_provenance(cfg);env={'packages':{}}
        spec={'inputs':provenance,'burst_id':'a','polarization':'HH','flatten':False,
              'backend':'cpu','software':{}}
        previous={'coreg_identity':identity(spec),'coreg_specification':spec,
                  'coreg':{'slcs':{'reference':'r','secondary':'s'},'reference_directory':'ref'}}
        write_json(self.root/'work/a/manifest.json',previous)
        context=InvocationContext(self.rows,json.dumps(provenance),json.dumps(env))
        with patch('glacier_offsets.compass_coreg.select_pair',side_effect=AssertionError('rediscovery')), \
                patch('glacier_offsets.compass_coreg.input_provenance',side_effect=AssertionError('rehash')), \
                patch('glacier_offsets.compass_coreg.inspect_environment',side_effect=AssertionError('software')), \
                patch('glacier_offsets.compass_coreg.inspect_pair'):
            root,actual=coregister(cfg,context=context)
            self.assertEqual(actual,previous)

    def test_provenance_hashes_each_shared_source_only_once(self):
        with patch('glacier_offsets.products.checksum',wraps=checksum) as hashed:
            provenance=input_provenance(self.cfg)
            hashed.assert_called_once()
            self.assertEqual(len(provenance),5)
