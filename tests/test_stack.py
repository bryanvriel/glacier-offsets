"""Stack inventory, YAML inheritance, UTC pairing and read-only CLI controls."""
from copy import deepcopy
from datetime import datetime, timedelta
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from shapely.geometry import box

from glacier_offsets.cli import main
from glacier_offsets.config import load_defaults
from glacier_offsets.inputs import Burst
from glacier_offsets.stack import StackPreflightError, pair_graph, plan_stack, scan_safes
from glacier_offsets.stack_config import load_stack_config, load_status_config
from test_initialization import manifest, orbit, scene


class StackPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.safe = self.root/'safe'; self.safe.mkdir()
        self.orbits = self.root/'orbits'; self.orbits.mkdir()
        self.times = [datetime(2020, 1, 1, 10), datetime(2020, 1, 13, 10), datetime(2020, 1, 25, 10, 0, 1)]
        self.paths = []
        for i, time in enumerate(self.times):
            self.paths.append(scene(self.safe/f'{i}.zip', time))
            orbit(self.orbits/f'{i}.EOF', time-timedelta(hours=8))
        dem = self.root/'dem.tif'; dem.touch()
        cfg = load_defaults()
        cfg.update(stack={'safe_directory': str(self.safe), 'orbit_directory': str(self.orbits)},
                   output={'directory': str(self.root/'output')})
        cfg['inputs'].update(polarization='HH', burst_id='burst', dem=str(dem))
        cfg['geocoding']['epsg'] = 3031
        self.path = self.root/'stack.yaml'
        self.path.write_text(yaml.safe_dump(cfg))
        self.cfg = load_stack_config(self.path)
        self.loader = patch('glacier_offsets.stack.inventory', side_effect=self.bursts)
        self.loader.start(); self.addCleanup(self.loader.stop)

    def bursts(self, safe, orbit, polarization, swaths):
        i = self.paths.index(Path(safe))
        native = SimpleNamespace(border=[box(-100, -76, -98, -74)])
        return [Burst('burst', 2, self.times[i], native)]

    def test_merge_policy_honors_explicit_and_inherited_matching_engine(self):
        cfg = deepcopy(self.cfg)
        cfg['stack']['processing'] = 'isce2_merge_first'
        cfg['matching'].pop('engine')
        self.path.write_text(yaml.safe_dump(cfg))
        for loader in (load_stack_config, load_status_config):
            with self.assertLogs(level='WARNING') as logs:
                self.assertEqual(loader(self.path)['matching']['engine'], 'isce2')
            self.assertIn('deprecated alias of isce3_merge_first', logs.output[0])
        child = self.root/'isce3.yaml'
        child.write_text('extends: stack.yaml\nmatching:\n  engine: isce3\n')
        grandchild = self.root/'inherited.yaml'
        grandchild.write_text('extends: isce3.yaml\nmatching:\n  deramp: 0\n')
        with patch('glacier_offsets.isce2_matching.validate_settings', side_effect=AssertionError('ISCE2 runtime required')):
            for path in (child, grandchild):
                for loader in (load_stack_config, load_status_config):
                    selected = loader(path)
                    self.assertEqual(selected['matching']['engine'], 'isce3')
                    self.assertEqual(selected['stack']['processing'], 'isce2_merge_first')

    def test_canonical_merge_selector_defaults_to_isce3_and_honors_explicit_engines(self):
        cfg = deepcopy(self.cfg)
        cfg['stack'].update(processing='isce3_merge_first', swath_merge='midpoint')
        cfg['matching'].pop('engine')
        self.path.write_text(yaml.safe_dump(cfg))
        for loader in (load_stack_config, load_status_config):
            selected = loader(self.path)
            self.assertEqual(selected['matching']['engine'], 'isce3')
            self.assertEqual(selected['stack']['swath_merge'], 'midpoint')
        child = self.root/'isce2.yaml'
        child.write_text('extends: stack.yaml\nmatching:\n  engine: isce2\n')
        with patch('glacier_offsets.isce2_matching.validate_settings'):
            self.assertEqual(load_stack_config(child)['matching']['engine'], 'isce2')
        cfg['stack']['processing'] = 'isce3_merge'
        self.path.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError, 'isce3_merge_first or isce2_merge_first'):
            load_stack_config(self.path)

    def test_swath_midpoint_requires_common_validity_and_rejects_unknown_policy(self):
        cfg=deepcopy(self.cfg)
        cfg['stack']['swath_merge']='midpoint'
        self.path.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError,'requires stack.processing'):
            load_stack_config(self.path)
        cfg['stack']['processing']='isce2_merge_first'
        self.path.write_text(yaml.safe_dump(cfg))
        self.assertEqual(load_stack_config(self.path)['stack']['swath_merge'],'midpoint')
        cfg['stack']['swath_merge']='average'
        self.path.write_text(yaml.safe_dump(cfg))
        with self.assertRaisesRegex(ValueError,'stack.swath_merge'):
            load_stack_config(self.path)

    def test_order_exact_cap_explicit_override_and_disconnected_dates(self):
        plan = plan_stack(self.cfg)
        self.assertEqual(plan.reference.date, '2020-01-01')
        self.assertEqual(plan.pairs, (('2020-01-01', '2020-01-13'),))
        self.assertEqual(plan.report['disconnected_acquisitions'], ['2020-01-25'])
        self.assertEqual(plan.report['omitted_links'][0]['separation_seconds'], 12*86400+1)
        self.assertFalse((self.root/'output').exists())
        self.cfg['stack']['pairing'].update(neighbors=2, max_separation_days=None)
        self.assertEqual(len(plan_stack(self.cfg).pairs), 3)
        csv = self.root/'pairs.csv'
        csv.write_text('earlier,later\n2020-01-01,2020-01-25\n2020-01-01,2020-01-25\n')
        self.cfg['stack']['pairing'].update(pairs_file=str(csv), max_separation_days=1)
        self.assertEqual(plan_stack(self.cfg).pairs, (('2020-01-01', '2020-01-25'),))
        for text in ['later,earlier\n', 'earlier,later\n2020-01-25,2020-01-01\n',
                     'earlier,later\n2020-01-01,2020-01-01\n', 'earlier,later\n2020-01-01,2020-01-02\n',
                     'earlier,later\n20200101,2020-01-25\n', 'earlier,later\n2020-01-01,2020-01-25,extra\n']:
            csv.write_text(text)
            with self.assertRaises(StackPreflightError):
                plan_stack(self.cfg)

    def test_inclusive_dates_reference_override_pin_and_new_earlier_date(self):
        self.cfg['stack'].update(start_date='2020-01-13', end_date='2020-01-25', reference_date='2020-01-25')
        plan = plan_stack(self.cfg)
        self.assertEqual([a.date for a in plan.acquisitions], ['2020-01-13', '2020-01-25'])
        self.assertEqual(plan.reference.date, '2020-01-25')
        old = {'anchor': plan.anchor}
        self.cfg['stack'].update(start_date=None, reference_date=None)
        pinned = plan_stack(self.cfg, old)
        self.assertEqual(pinned.reference.date, '2020-01-25')
        self.assertEqual(len(pinned.acquisitions), 3)
        self.cfg['stack']['reference_date'] = '2020-01-01'
        with self.assertRaisesRegex(StackPreflightError, 'pinned'):
            plan_stack(self.cfg, old)

    def test_duplicates_groups_orbit_failure_and_no_bridging(self):
        duplicate = scene(self.safe/'duplicate.zip', self.times[0]+timedelta(hours=1))
        with self.assertRaisesRegex(StackPreflightError, 'duplicate acquisition date') as caught:
            plan_stack(self.cfg)
        self.assertEqual(len(caught.exception.report['inventory']), 4)
        duplicate.unlink()
        scene(self.paths[1], self.times[1], track=39)
        with self.assertRaisesRegex(StackPreflightError, 'Mixed relative orbit'):
            plan_stack(self.cfg)
        scene(self.paths[1], self.times[1])
        (self.orbits/'1.EOF').unlink()
        self.cfg['stack']['pairing']['max_separation_days'] = None
        plan = plan_stack(self.cfg)
        self.assertIsNotNone(plan.acquisitions[1].error)
        self.assertEqual(plan.pairs, (('2020-01-01','2020-01-13'), ('2020-01-13','2020-01-25')))

    def test_aoi_reference_eligibility_partial_coverage_and_kml(self):
        self.cfg['inputs']['burst_id'] = None
        kml = self.root/'area.kml'
        kml.write_text('''<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark>
          <Polygon><outerBoundaryIs><LinearRing><coordinates>
          -100,-76 -98,-76 -98,-74 -100,-74 -100,-76
          </coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>''')
        self.cfg['aoi'] = {'path': str(kml)}
        def inventory(safe, *args):
            i = self.paths.index(Path(safe))
            if i == 0:
                return [Burst('outside', 2, self.times[i], SimpleNamespace(border=[box(10,10,11,11)]))]
            rows = self.bursts(safe, *args)
            if i == 1:
                rows.append(Burst('partial', 2, self.times[i]+timedelta(seconds=2), rows[0].native))
            return rows
        with patch('glacier_offsets.stack.inventory', side_effect=inventory):
            plan = plan_stack(self.cfg)
        self.assertEqual(plan.reference.date, '2020-01-13')
        self.assertEqual(plan.report['selected_burst_ids'], ['burst','partial'])
        self.assertEqual(plan.report['inventory'][2]['missing_burst_ids'], ['partial'])

    def test_recursive_safe_scan_does_not_descend_into_safe(self):
        unpacked = self.safe/'nested'/'x.SAFE'; unpacked.mkdir(parents=True)
        (unpacked/'nested.zip').touch()
        (unpacked/'manifest.safe').write_text(manifest(self.times[0]))
        found = scan_safes(self.safe)
        self.assertIn(unpacked, found)
        self.assertNotIn(unpacked/'nested.zip', found)

    def test_hidden_safe_files_and_directories_are_ignored(self):
        for name in ('._scene.zip', '.hidden.zip'):
            (self.safe/name).write_bytes(b'macOS metadata, not a SAFE archive')
        hidden=self.safe/'.metadata'; hidden.mkdir()
        (hidden/'scene.zip').write_bytes(b'not an archive')
        (hidden/'scene.SAFE').mkdir()
        (self.safe/'.hidden.SAFE').mkdir()
        self.assertEqual(scan_safes(self.safe),self.paths)
        self.assertEqual(len(plan_stack(self.cfg).acquisitions),3)

    def test_yaml_relative_inheritance_dates_and_initializer(self):
        parent = self.root/'parent'; parent.mkdir()
        base = parent/'base.yaml'
        base.write_text('stack:\n  safe_directory: ../safe\n  orbit_directory: ../orbits\n'
                        '  start_date: 2020-01-01\ninputs:\n  dem: ../dem.tif\n'
                        '  polarization: HH\n  burst_id: burst\noutput:\n  directory: ../output\n')
        child = self.root/'child.yaml'; child.write_text('extends: parent/base.yaml\ngeocoding:\n  epsg: 3031\n')
        cfg = load_stack_config(child)
        self.assertEqual(cfg['stack']['safe_directory'], str(self.safe))
        self.assertEqual(cfg['stack']['start_date'], '2020-01-01')
        self.assertNotIn('reference_safe', cfg['inputs'])
        self.assertEqual(cfg['stack']['pairing']['max_separation_days'], 12)
        from glacier_offsets.stack_initialization import main as initialize
        with patch('glacier_offsets.stack_initialization.make_stack_config', return_value=cfg), contextlib.redirect_stdout(io.StringIO()):
            output = self.root/'generated.yaml'
            self.assertEqual(initialize(['--safe-dir',str(self.safe),'--orbit-dir',str(self.orbits),
                '--dem',str(self.root/'dem.tif'),'--burst-id','burst','--polarization','HH',
                '--output',str(output),'--relative-paths']), 0)
        generated = load_stack_config(output)
        self.assertEqual(generated, cfg)
        for bad in [{'pairing': {'neighbors':0}}, {'reference_date':'20200101'},
                    {'pairing':{'max_separation_days':float('inf')}}, {'pairing':None}, {'unexpected':True}]:
            document = yaml.safe_load(self.path.read_text())
            document['stack'].update(bad)
            child.write_text(yaml.safe_dump(document))
            with self.assertRaises(ValueError):
                load_stack_config(child)

    def test_plan_and_validate_cli_do_not_hash_or_write(self):
        with patch('glacier_offsets.products.checksum', side_effect=AssertionError('hashed SAFE')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['stack', 'plan', str(self.path)]), 0)
            self.assertEqual(json.loads(output.getvalue())['reference_date'], '2020-01-01')
            with patch('glacier_offsets.stack_cli.validate_stack', return_value='cpu') as validate:
                self.assertEqual(main(['stack','validate',str(self.path),'--backend','cpu']), 0)
                validate.assert_called_once()
        self.assertFalse((self.root/'output').exists())
        with patch('glacier_offsets.stack_workflow.execute_stack', side_effect=KeyboardInterrupt), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['stack','run',str(self.path)]), 130)

    def test_native_preflight_and_real_initializer_validate_reference_and_date_grids(self):
        from osgeo import gdal
        from pyproj import CRS
        from glacier_offsets.stack import validate_stack
        from glacier_offsets.stack_initialization import make_stack_config
        dem=self.root/'dem.tif'
        ds=gdal.GetDriverByName('GTiff').Create(str(dem),100,100,1,gdal.GDT_Float32)
        ds.SetProjection(CRS.from_epsg(3031).to_wkt())
        ds.SetGeoTransform((-4000000,80000,0,4000000,0,-80000)); ds=None
        cfg=make_stack_config(self.safe,self.orbits,dem,self.root/'output','HH',burst_id='burst',epsg=3031)
        plan=plan_stack(cfg)
        for acquisition in plan.acquisitions:
            native=acquisition.bursts['burst'].native
            native.polarization='HH';native.orbit_direction='DESCENDING'
            grid=SimpleNamespace(prf=500.,range_pixel_spacing=2.,wavelength=.05,length=192,width=384,
                starting_range=800000.,sensing_start=0.,ref_epoch='2020-01-01T00:00:00',lookside='right')
            native.as_isce3_radargrid=lambda grid=grid:grid
        self.assertEqual(validate_stack(cfg,plan),'cpu')
        plan.acquisitions[-1].bursts['burst'].native.as_isce3_radargrid().prf=510.
        with self.assertRaisesRegex(ValueError,'sampling'):
            validate_stack(cfg,plan)


if __name__ == '__main__':
    unittest.main()
