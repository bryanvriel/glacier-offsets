from datetime import datetime, timedelta
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from osgeo import gdal, osr
import yaml
from glacier_offsets.config import load_config
from glacier_offsets.initialization import OrbitCatalog, make_pair_config, order_pair, read_scene


def manifest(start, mission='A', polarizations=('HH',), track=38, product='SLC'):
    stop = start + timedelta(seconds=30)
    pol = ''.join(f'<s:transmitterReceiverPolarisation>{p}</s:transmitterReceiverPolarisation>' for p in polarizations)
    return f'''<root xmlns:s="urn:sentinel:test"><s:acquisitionPeriod>
      <s:startTime>{start.isoformat()}Z</s:startTime><s:stopTime>{stop.isoformat()}Z</s:stopTime>
      </s:acquisitionPeriod><s:platform><s:familyName>SENTINEL-1</s:familyName>
      <s:number>{mission}</s:number></s:platform><s:mode>IW</s:mode>
      <s:productType>{product}</s:productType>{pol}<s:relativeOrbitNumber>{track}</s:relativeOrbitNumber>
      <s:pass>DESCENDING</s:pass></root>'''


def scene(path, start, **kwargs):
    with ZipFile(path, 'w') as z:
        z.writestr('example.SAFE/manifest.safe', manifest(start, **kwargs))
    return path


def orbit(path, start, mission='1A', created='2021-03-16T21:13:54', kind='AUX_POEORB',
          vector_start=None, reversed_vectors=False):
    stop = start + timedelta(days=1)
    vector_start = vector_start or start
    times = [vector_start + (stop-vector_start)*i/3 for i in range(4)]
    if reversed_vectors:
        times.reverse()
    vectors = ''.join(f'<OSV><UTC>UTC={t.isoformat()}</UTC></OSV>' for t in times)
    path.write_text(f'''<Earth_Explorer_File><Earth_Explorer_Header><Fixed_Header>
      <File_Type>{kind}</File_Type><Mission>Sentinel-{mission}</Mission><File_Version>0001</File_Version>
      <Validity_Period><Validity_Start>UTC={start.isoformat()}</Validity_Start>
      <Validity_Stop>UTC={stop.isoformat()}</Validity_Stop></Validity_Period>
      <Source><Creation_Date>UTC={created}</Creation_Date></Source></Fixed_Header></Earth_Explorer_Header>
      <Data_Block><List_of_OSVs>{vectors}</List_of_OSVs></Data_Block></Earth_Explorer_File>''')
    return path


class InitializationTests(unittest.TestCase):
    def setUp(self):
        self.time = datetime(2020, 1, 10, 8, 2, 37, 123456)

    def test_manifest_zip_directory_and_ambiguous_archive(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            zipped = scene(d/'a.zip', self.time)
            unpacked = d/'a.SAFE'; unpacked.mkdir()
            (unpacked/'manifest.safe').write_text(manifest(self.time))
            a, b = read_scene(zipped), read_scene(unpacked)
            self.assertEqual((a.start, a.stop, a.mission, a.polarizations, a.relative_orbit, a.direction),
                             (b.start, b.stop, b.mission, b.polarizations, b.relative_orbit, b.direction))
            self.assertEqual(a.start, self.time)
            with ZipFile(zipped, 'a') as z:
                z.writestr('other.SAFE/manifest.safe', manifest(self.time))
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                read_scene(zipped)
            with self.assertRaisesRegex(ValueError, 'IW SLC'):
                read_scene(scene(d/'grd.zip', self.time, product='GRD'))

    def test_orbit_mission_validity_vectors_and_latest_generation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); s = read_scene(scene(d/'a.zip', self.time))
            old = orbit(d/'old.EOF', self.time-timedelta(hours=8))
            new = orbit(d/'nested.xml', self.time-timedelta(hours=8), created='2022-01-01T00:00:00')
            orbit(d/'wrong_mission.EOF', self.time-timedelta(hours=8), mission='1B', created='2023-01-01T00:00:00')
            orbit(d/'short_margin.EOF', self.time-timedelta(minutes=5), created='2023-01-01T00:00:00')
            orbit(d/'short_vectors.EOF', self.time-timedelta(hours=8), vector_start=self.time, created='2023-01-01T00:00:00')
            orbit(d/'restituted.EOF', self.time-timedelta(hours=8), kind='AUX_RESORB', created='2023-01-01T00:00:00')
            orbit(d/'invalid.EOF', self.time-timedelta(hours=8), reversed_vectors=True)
            catalog = OrbitCatalog(d)
            self.assertEqual(catalog.resolve(s).path, new.resolve())
            new.unlink()
            self.assertEqual(OrbitCatalog(d).resolve(s).path, old.resolve())
            old.unlink()
            with self.assertRaisesRegex(ValueError, 'No precise orbit'):
                OrbitCatalog(d).resolve(s)

    def test_pair_order_track_and_exact_maximum_separation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a = read_scene(scene(d/'a.zip', self.time))
            b = read_scene(scene(d/'b.zip', self.time+timedelta(days=12)))
            self.assertEqual(order_pair([b, a], 12), (a, b))
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                order_pair([a, b], 11.999)
            for limit in [0, -1, float('nan'), float('inf')]:
                with self.assertRaises(ValueError):
                    order_pair([a, b], limit)
            with self.assertRaisesRegex(ValueError, 'distinct'):
                order_pair([a, a])
            wrong = read_scene(scene(d/'wrong.zip', self.time+timedelta(days=12), track=39))
            with self.assertRaisesRegex(ValueError, 'relative orbit'):
                order_pair([a, wrong])

    def test_orbit_catalog_ignores_hidden_files_and_directories(self):
        from glacier_offsets.initialization import read_precise_orbit
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            s=read_scene(scene(d/'scene.zip',self.time))
            visible=orbit(d/'visible.EOF',self.time-timedelta(hours=8))
            (d/'._visible.EOF').write_bytes(b'macOS metadata, not orbit XML')
            orbit(d/'.hidden.xml',self.time-timedelta(hours=8),created='2025-01-01T00:00:00')
            hidden=d/'.metadata'; hidden.mkdir()
            orbit(hidden/'newer.EOF',self.time-timedelta(hours=8),created='2025-01-01T00:00:00')
            with patch('glacier_offsets.initialization.read_precise_orbit',wraps=read_precise_orbit) as read:
                catalog=OrbitCatalog(d)
            read.assert_called_once_with(visible.resolve())
            self.assertEqual(catalog.resolve(s).path,visible.resolve())

    def fixture(self, d, polarizations=('HH',), epsg=3031):
        a = scene(d/'ref.zip', self.time, polarizations=polarizations)
        b = scene(d/'sec.zip', self.time+timedelta(days=12), polarizations=polarizations)
        orbits = d/'orbits'; (orbits/'nested').mkdir(parents=True)
        orbit(orbits/'nested'/'a.EOF', self.time-timedelta(hours=8))
        orbit(orbits/'b.EOF', self.time+timedelta(days=12)-timedelta(hours=8))
        dem = d/'dem.tif'
        raster = gdal.GetDriverByName('GTiff').Create(str(dem), 2, 2, 1, gdal.GDT_Float32)
        crs = osr.SpatialReference(); crs.ImportFromEPSG(epsg)
        raster.SetProjection(crs.ExportToWkt()); raster = None
        return [b, a], dem, orbits

    def test_generated_yaml_discovery_processing_and_burst_validation(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            paths, dem, orbits = self.fixture(d)
            cfg = make_pair_config(paths, dem, orbits, d/'work', max_separation_days=12)
            self.assertEqual(cfg['inputs']['reference_safe'], str(paths[1].resolve()))
            self.assertEqual(cfg['inputs']['polarization'], 'HH')
            self.assertNotIn('burst_id', cfg['inputs'])
            self.assertNotIn('first_center', cfg['matching'])
            self.assertEqual(cfg['matching']['window'], [64, 256])
            self.assertEqual(cfg['matching']['search'], [10, 40])
            self.assertEqual(cfg['matching']['snr_threshold'], 0.0)
            self.assertEqual(cfg['matching']['correlation_threshold'], 0.0)
            self.assertEqual(cfg['geocoding']['epsg'], 3031)
            self.assertEqual(cfg['geocoding']['resampling'], 'nearest')
            self.assertEqual(cfg['geocoding']['mask'], 'input')
            f = d/'pair.yaml'; f.write_text(yaml.safe_dump(cfg))
            load_config(f, discovery=True)
            with self.assertRaisesRegex(ValueError, 'burst_id'):
                load_config(f)
            with patch('glacier_offsets.inputs.select_pair') as select:
                cfg = make_pair_config(paths, dem, orbits, burst_id='t038_080975_iw2')
                select.assert_called_once_with(cfg)
            f.write_text(yaml.safe_dump(cfg))
            load_config(f)

    def test_ambiguous_polarization_and_geographic_dem_requires_epsg(self):
        with tempfile.TemporaryDirectory() as d:
            paths, dem, orbits = self.fixture(Path(d), ('VV', 'VH'), 4326)
            with self.assertRaisesRegex(ValueError, 'Specify --polarization'):
                make_pair_config(paths, dem, orbits)
            with self.assertRaises(ValueError):
                make_pair_config(paths, dem, orbits, polarization='VV')
            cfg = make_pair_config(paths, dem, orbits, polarization='VV', epsg=3031)
            self.assertEqual(cfg['geocoding']['epsg'], 3031)
            with self.assertRaisesRegex(ValueError, 'not present'):
                make_pair_config(paths, dem, orbits, polarization='HH', epsg=3031)

    def test_cli_refuses_overwrite_and_writes_absolute_paths(self):
        spec = importlib.util.spec_from_file_location('init_pair_config', Path(__file__).resolve().parents[1]/'tools/init_pair_config.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); paths, dem, orbits = self.fixture(d)
            output = d/'pair.yaml'
            args = [str(p) for p in paths]+['--dem', str(dem), '--orbit-dir', str(orbits), '--output', str(output)]
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(args)
            cfg = yaml.safe_load(output.read_text())
            self.assertEqual(cfg['extends'],'defaults')
            self.assertNotIn('matching',cfg)
            self.assertNotIn('gross_offsets',cfg)
            self.assertEqual(cfg['geocoding'],{'epsg':3031})
            self.assertEqual(load_config(output,discovery=True)['matching']['window'],[64,256])
            self.assertTrue(Path(cfg['output']['directory']).is_absolute())
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                module.main(args)
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(args+['--force'])

    def test_cli_writes_relative_paths_that_round_trip(self):
        spec = importlib.util.spec_from_file_location('init_pair_config', Path(__file__).resolve().parents[1]/'tools/init_pair_config.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            paths, dem, orbits = self.fixture(d)
            output = d/'configs/pair.yaml'
            args = [str(p) for p in paths]+['--dem', str(dem), '--orbit-dir', str(orbits),
                '--output', str(output), '--output-directory', str(d/'work'), '--relative-paths']
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(args)
            cfg = yaml.safe_load(output.read_text())
            for key in ('reference_safe', 'secondary_safe', 'reference_orbit', 'secondary_orbit', 'dem'):
                self.assertFalse(Path(cfg['inputs'][key]).is_absolute())
            self.assertEqual(cfg['inputs']['dem'], '../dem.tif')
            self.assertEqual(cfg['output']['directory'], '../work')
            with contextlib.chdir(d.parent):
                loaded = load_config(output, discovery=True)
            expected = make_pair_config(paths, dem, orbits, d/'work')
            self.assertEqual(loaded, expected)


if __name__ == '__main__':
    unittest.main()
