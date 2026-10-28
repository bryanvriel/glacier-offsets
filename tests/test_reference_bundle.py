"""Relocatable reference assets and reference-independent acquisition selection."""
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from osgeo import gdal
from pyproj import CRS
from shapely.geometry import box
import yaml

from glacier_offsets.config import load_defaults
from glacier_offsets.inputs import Burst
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.reference_bundle import export_reference, read_bundle, configure, install, runtime_frame, MANIFEST
from glacier_offsets.stack import plan_stack
from glacier_offsets.stack_config import load_stack_config
from glacier_offsets.stack_lock import stack_lock
from glacier_offsets.stack_mosaic import POLICY, prepare_common_layout, MergePlan, Placement, compose
from glacier_offsets.stack_rebuild import rebuild_layout
from glacier_offsets.stack_storage import reference_specification, prepare_geometry
from test_initialization import scene, orbit


class ReferenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / 'old-stack'
        self.geometry = self.source / 'geometry'
        directory = self.geometry / 'bursts/burst/coreg/reference/burst/20170101'
        directory.mkdir(parents=True)
        self.time = datetime(2017, 1, 1, 8)
        self.safe = scene(self.root / 'reference.zip', self.time)
        self.orbit = orbit(self.root / 'reference.EOF', self.time-timedelta(hours=8))
        text = self.orbit.read_text().replace('</UTC>', '</UTC><X>7078137</X><Y>0</Y><Z>0</Z><VX>0</VX><VY>7500</VY><VZ>0</VZ>')
        self.orbit.write_text(text)
        self.dem = self.root / 'dem.tif'
        self.write(self.dem, np.zeros((32, 48)))
        paths = {}
        for name in ('x', 'y', 'z', 'layover_shadow_mask'):
            path = directory / (name+'.tif')
            self.write(path, np.zeros((32, 48)))
            paths[name] = str(path)
        slc = directory / 'reference.slc.tif'
        self.write(slc, np.ones((32, 48), 'complex64'), gdal.GDT_CFloat32)
        # Ensure a sidecar is collected and remains beside the raster.
        (directory / 'reference.slc.tif.aux.xml').write_text('<PAMDataset/>')
        (directory / 'radar_grid.txt').write_text('native-grid-placeholder')
        self.owner = self.geometry / 'source.tif'
        self.write(self.owner, np.ones((32, 48)), gdal.GDT_UInt16)
        vrt = self.geometry / 'x.vrt'
        vrt.write_text(f'<VRTDataset rasterXSize="48" rasterYSize="32"><VRTRasterBand dataType="Float64" band="1"><SimpleSource><SourceFilename relativeToVRT="0">{paths["x"]}</SourceFilename><SourceBand>1</SourceBand></SimpleSource></VRTRasterBand></VRTDataset>')
        cfg = self.cfg = load_defaults()
        cfg.update(stack={'processing': 'isce3_merge_first', 'swath_merge': 'overwrite', 'reference_date': None},
                   output={'directory': str(self.source)})
        cfg['inputs'].update(polarization='HH', subswaths=[1], burst_id='burst', dem=str(self.dem))
        cfg['matching'].update(window=[8, 8], search=[2, 2], stride=[4, 4], zoom=4)
        cfg['geocoding']['epsg'] = 3031
        selection = {'aoi': None, 'burst_id': 'burst', 'selected_ids': ['burst'], 'subswaths': [1], 'polarization': 'HH'}
        anchor = {'reference_date': '2017-01-01', 'reference_safe': str(self.safe), 'reference_orbit': str(self.orbit),
                  'dem': str(self.dem), 'selection': selection, 'alignment_tolerance_px': .001}
        spec = {'method': 'stack-reference-interior-ownership-v3', 'ownership_policy': POLICY, 'anchor': anchor,
                'inputs': {k: {'path': str(p), 'sha256': checksum(p)} for k, p in [('safe', self.safe), ('orbit', self.orbit), ('dem', self.dem)]},
                'software': {}, 'flatten': False, 'backend': 'cpu'}
        grid = {'length': 32, 'width': 48, 'prf': 500., 'range_pixel_spacing': 2., 'wavelength': .05,
                'lookside': 'right', 'sensing_start': 0., 'ref_epoch': self.time.isoformat(), 'starting_range': 8e5}
        coreg = {'geometry': paths, 'reference_directory': str(directory), 'slcs': {'reference': str(slc)},
                 'reference_grid': grid, 'geometry_crs': CRS.from_epsg(4326).to_wkt()}
        burst_manifest = self.geometry / 'bursts/burst/manifest.json'
        write_json(burst_manifest, {'coreg': coreg, 'status': 'complete'})
        source = {'burst_id': 'burst', 'coreg': coreg, 'offset': [0, 0], 'valid_window': [0,32,0,48],
                  'ownership_windows': [[0,32,0,48]], 'geometry_reference_time': self.time.isoformat(), 'manifest': str(burst_manifest)}
        self.frame = {'status': 'complete', 'geometry_identity': identity(spec), 'specification': spec, 'layout_identity': 'layout',
                      'coreg': coreg | {'source_index': str(self.owner)}, 'sources': [source],
                      'map_grid': {'epsg':3031, 'shape':[8,12], 'transform':[-2e6,200.,0,1e6,0,-200.]},
                      'common_validity': {'windows': {'burst':[0,32,0,48]}}, 'common_validity_identity':'common',
                      'products': {str(p):checksum(p) for p in [self.owner, vrt, slc, *map(Path, paths.values())]}}
        write_json(self.geometry / 'manifest.json', self.frame)
        write_json(self.source / 'stack_manifest.json', {'anchor': anchor, 'map_grid':self.frame['map_grid']})
        self.bundle = self.root / 'export'

    def write(self, path, data, dtype=gdal.GDT_Float64):
        ds = gdal.GetDriverByName('GTiff').Create(str(path), data.shape[1], data.shape[0], 1, dtype)
        ds.GetRasterBand(1).WriteArray(data)
        ds = None

    def export(self):
        return export_reference(self.cfg, self.bundle)

    def destination(self):
        cfg = deepcopy(self.cfg)
        cfg['output']['directory'] = str(self.root / 'new-stack')
        cfg['stack']['reference_directory'] = str(self.bundle)
        return configure(cfg)

    def test_relocation_import_and_offline_force_preserve_identity(self):
        result = self.export()
        moved = self.root / 'moved'
        self.bundle.rename(moved)
        self.bundle = moved
        data = read_bundle(moved, verify=True)
        self.assertEqual(result['fingerprint'], data['fingerprint'])
        self.assertEqual(data['frame']['geometry_identity'], self.frame['geometry_identity'])
        cfg = self.destination()
        target = Path(cfg['output']['directory'])
        with stack_lock(target, create=True):
            install(cfg)
            before = checksum(moved)
            frame = runtime_frame(cfg)
            prepared = prepare_geometry(cfg, None, frame['specification'], {'packages':{'isce3':'destination-version'}}, force=True)
            self.assertEqual(prepared['layout_identity'], 'layout')
            self.assertEqual(prepared['software']['packages']['isce3'], 'destination-version')
            self.assertEqual(prepared['specification'], self.frame['specification'])
            self.assertEqual(prepared['reference_preparation_software'], {})
            self.assertEqual(before, checksum(moved))
        shutil.rmtree(moved)
        shutil.rmtree(self.source)
        self.safe.unlink(); self.orbit.unlink(); self.dem.unlink()
        cfg = configure(cfg)
        self.assertEqual(read_bundle(cfg['_reference_bundle']['root'], verify=True)['fingerprint'], result['fingerprint'])
        imported = runtime_frame(cfg)
        from glacier_offsets.stack_status import reference_status
        self.assertEqual(reference_status(cfg, imported, None)['status'], 'complete')
        ds = gdal.Open(str(Path(cfg['_reference_bundle']['root']) / 'geometry/x.vrt'))
        np.testing.assert_array_equal(ds.ReadAsArray(), np.zeros((32,48)))
        self.assertEqual(reference_specification(cfg, None, None, None), self.frame['specification'])
        self.assertEqual(prepare_common_layout(cfg, None, imported, {}), imported)
        with self.assertRaisesRegex(ValueError, 'immutable'):
            rebuild_layout(cfg)

    def test_source_stack_can_be_addressed_through_a_directory_symlink(self):
        alias = self.root/'source-link'
        alias.symlink_to(self.source, target_is_directory=True)
        frame = json.loads(json.dumps(self.frame).replace(str(self.source),str(alias)))
        write_json(self.geometry/'manifest.json',frame)
        self.cfg['output']['directory'] = str(alias)
        self.export()
        data = read_bundle(self.bundle,verify=True)
        self.assertEqual(data['frame']['geometry_identity'],self.frame['geometry_identity'])
        configure(self.destination())

    def test_read_only_bundle_can_be_imported_without_changes(self):
        self.export()
        before = checksum(self.bundle)
        for p in sorted(self.bundle.rglob('*'), reverse=True):
            p.chmod(0o555 if p.is_dir() else 0o444)
        self.bundle.chmod(0o555)
        try:
            cfg = self.destination()
            with stack_lock(cfg['output']['directory'], create=True):
                install(cfg)
            self.assertEqual(checksum(self.bundle), before)
        finally:
            self.bundle.chmod(0o755)
            for p in self.bundle.rglob('*'):
                p.chmod(0o755 if p.is_dir() else 0o644)

    def test_acquisition_filters_exclude_reference_from_pair_graph(self):
        self.export(); cfg = self.destination()
        safes = self.root/'safe-2018'; safes.mkdir()
        orbits = self.root/'orbits-2018'; orbits.mkdir()
        times = [datetime(2018,1,2,8), datetime(2018,1,8,8)]
        for i, t in enumerate(times):
            scene(safes/f'{i}.zip', t)
            orbit(orbits/f'{i}.EOF', t-timedelta(hours=8))
        cfg['stack'].update(safe_directory=str(safes), orbit_directory=str(orbits), start_date='2018-01-01', end_date='2018-12-31')
        path = self.root/'2018.yaml'
        portable_cfg = {k:v for k,v in cfg.items() if not k.startswith('_')}
        portable_cfg['stack']['reference_directory'] = 'export'
        path.write_text(yaml.safe_dump(portable_cfg))
        loaded = load_stack_config(path)
        self.assertEqual(loaded['stack']['reference_directory'], str(self.bundle))
        def inventory(safe, *args):
            t = self.time if Path(safe).name == self.safe.name else times[int(Path(safe).stem)]
            return [Burst('burst', 1, t, SimpleNamespace(border=[box(-64,-69,-63,-68)]))]
        with patch('glacier_offsets.stack.inventory', side_effect=inventory):
            plan = plan_stack(loaded)
        self.assertEqual(plan.reference.date, '2017-01-01')
        self.assertEqual([a.date for a in plan.acquisitions], ['2018-01-02','2018-01-08'])
        self.assertEqual(plan.pairs, (('2018-01-02','2018-01-08'),))

    def test_native_safe_reader_ignores_appledouble_annotations(self):
        from glacier_offsets.inputs import load_safe_bursts
        safe = self.root/'reader.SAFE'
        (safe/'annotation').mkdir(parents=True)
        (safe/'measurement').mkdir()
        for swath in (1,2):
            (safe/'annotation'/f'._s1a-iw{swath}-slc-hh-example.xml').write_bytes(b'opaque')
            (safe/'annotation'/f's1a-iw{swath}-slc-hh-example.xml').write_text('<xml/>')
            (safe/'measurement'/f'._s1a-iw{swath}-slc-hh-example.tiff').write_bytes(b'opaque')
            (safe/'measurement'/f's1a-iw{swath}-slc-hh-example.tiff').write_bytes(b'slc')
        with patch('s1reader.s1_reader.burst_from_xml',return_value=[SimpleNamespace(burst_id='burst')]) as parser:
            bursts = load_safe_bursts(str(safe),str(self.orbit),1,'HH',flag_apply_eap=False)
        self.assertEqual(len(bursts),1)
        for p in (parser.call_args.args[0],parser.call_args.args[2],parser.call_args.args[3]):
            self.assertFalse(Path(p).name.startswith('.'))
        self.assertFalse(parser.call_args.kwargs['flag_apply_eap'])

    def test_package_records_reference_provenance(self):
        import h5netcdf
        from test_packaging import PackagingTests
        from glacier_offsets.packaging import package_stack
        f = PackagingTests()
        self.addCleanup(f.doCleanups)
        f.setUp()
        f.frame['reference_bundle'] = {'schema':'glacier-offsets-reference-v1', 'fingerprint':'portable-reference', 'origin':{'reference_date':'2017-01-01'}}
        write_json(f.root/'geometry/manifest.json', f.frame)
        result = package_stack(f.cfg)
        with h5netcdf.File(result['path'], 'r') as h:
            self.assertEqual(h.attrs['reference_bundle_fingerprint'], 'portable-reference')
            self.assertEqual(json.loads(h.attrs['reference_bundle_provenance']), f.frame['reference_bundle'])

    def test_yaml_date_and_polarization_normalization(self):
        self.export()
        cfg = deepcopy(self.cfg)
        cfg['stack']['reference_directory'] = str(self.bundle)
        cfg['stack']['reference_date'] = self.time.date()
        cfg['inputs']['polarization'] = 'hh'
        configure(cfg)
        self.assertEqual(cfg['stack']['reference_date'], '2017-01-01')
        self.assertEqual(cfg['inputs']['polarization'], 'HH')

    def test_conflicts_and_checksum_failure(self):
        self.export()
        for section, key, value in [('stack','reference_date','2018-01-01'), ('matching','stride',[8,8]),
                                    ('inputs','polarization','VV'), ('geocoding','spacing_m',100), ('inputs','dem',str(self.safe))]:
            cfg = deepcopy(self.cfg)
            cfg['stack']['reference_directory'] = str(self.bundle)
            cfg[section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'conflict'):
                configure(cfg)
        data = read_bundle(self.bundle)
        asset = self.bundle / next(iter(data['files']))
        asset.write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            read_bundle(self.bundle, verify=True)

    def test_export_preflight_lock_and_interrupted_publication(self):
        with stack_lock(self.source):
            with self.assertRaisesRegex(ValueError, 'writer'):
                self.export()
        write_json(self.source/'.layout-rebuild/journal.json', {'status':'running'})
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.export()
        shutil.rmtree(self.source/'.layout-rebuild')
        with patch('glacier_offsets.reference_bundle.shutil.copyfile', side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                self.export()
        self.assertFalse(self.bundle.exists())
        self.assertFalse(list(self.root.glob('.reference-export-*')))
        self.export()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.export()

    def test_appledouble_and_symlink_dependencies_are_materialized(self):
        slc = Path(self.frame['sources'][0]['coreg']['slcs']['reference'])
        backing = self.root/'external-reference.tif'
        slc.rename(backing)
        slc.symlink_to(backing)
        metadata = slc.parent/'._x.tif'
        metadata.write_bytes(b'opaque AppleDouble metadata')
        self.frame['products'][str(metadata)] = checksum(metadata)
        write_json(self.geometry/'manifest.json', self.frame)
        self.export()
        data = read_bundle(self.bundle, verify=True)
        local = configure(self.destination())['_reference_bundle']['resolved']['frame']
        self.assertFalse(Path(local['sources'][0]['coreg']['slcs']['reference']).is_symlink())
        self.assertTrue(any(p.endswith('/._x.tif') for p in data['files']))

    def test_export_detects_source_changes_during_copy(self):
        from glacier_offsets.reference_bundle import _Collector
        original = _Collector.copy
        def corrupt(collector, stage):
            original(collector, stage)
            self.dem.write_bytes(b'changed during export')
        with patch.object(_Collector, 'copy', corrupt):
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.export()
        self.assertFalse(self.bundle.exists())

    def test_bad_manifest_paths_and_missing_sidecars(self):
        self.export()
        data = read_bundle(self.bundle)
        name = next(p for p in data['files'] if p.endswith('.aux.xml'))
        (self.bundle/name).unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            read_bundle(self.bundle)
        data['files']['../escape'] = 'bad'
        data['fingerprint'] = identity({k:v for k,v in data.items() if k!='fingerprint'})
        write_json(self.bundle/MANIFEST, data)
        # Remove the prior missing dependency to exercise path rejection separately.
        data['files'].pop(name)
        data['fingerprint'] = identity({k:v for k,v in data.items() if k!='fingerprint'})
        write_json(self.bundle/MANIFEST, data)
        with self.assertRaisesRegex(ValueError, 'Invalid reference bundle path'):
            read_bundle(self.bundle)

    def test_initialized_reference_replacement_and_interrupted_import(self):
        self.export(); cfg = self.destination()
        target = Path(cfg['output']['directory']); target.mkdir()
        with patch('glacier_offsets.reference_bundle.shutil.copyfile', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                install(cfg)
        self.assertFalse((target/'.reference').exists())
        self.assertFalse(list(target.glob('.reference-import-*')))
        write_json(target/'geometry/manifest.json', self.frame)
        with self.assertRaisesRegex(ValueError, 'initialized'):
            install(cfg)

    def test_frozen_layout_masks_actual_date_support_without_source_substitution(self):
        p = Placement(1,'burst',(0,0),(0,32,0,48),((0,32,0,48),))
        plan = MergePlan(json.dumps({'length':32,'width':48}), (p,), '{}')
        source = self.frame['sources'][0]['coreg']['slcs']['reference']
        slc, mask = self.root/'merged.tif', self.root/'valid.tif'
        count = compose(plan, {'burst':source}, slc, mask, valid_windows={'burst':{'window':[4,28,6,40]}})
        self.assertEqual(count,24*34)
        values = gdal.Open(str(mask)).ReadAsArray()
        expected = np.zeros((32,48),'uint8'); expected[4:28,6:40]=1
        np.testing.assert_array_equal(values,expected)
        self.assertEqual(compose(plan, {}, slc, mask, valid_windows={}),0)
        # An absent neighboring burst does not hand its pixels to the available one.
        left = Placement(1,'burst',(0,0),(0,32,0,48),((0,32,0,24),))
        right = Placement(2,'missing',(0,0),(0,32,0,48),((0,32,24,48),))
        multi = MergePlan(plan.grid_json, (left,right), '{}')
        self.assertEqual(compose(multi, {'burst':source}, slc, mask,
            valid_windows={'burst':{'window':[0,32,0,48]}}),32*24)
        self.assertEqual(int(gdal.Open(str(mask)).ReadAsArray()[:,24:].sum()),0)
