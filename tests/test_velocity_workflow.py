"""Version-2 netCDF-to-results workflow on synthetic self-contained packages."""
from contextlib import chdir
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import h5py
import h5netcdf
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.cli import main as cli_main
from glacier_offsets.velocity_annual_noise import read_supplied
from glacier_offsets.velocity_cache import exclusive_lock
from glacier_offsets.velocity_collection import collect, content_identity, year_start
from glacier_offsets.velocity_io import writer_lock
from glacier_offsets.velocity_pixel_general import Layout
from glacier_offsets.velocity_solid_earth import Interrupted
from glacier_offsets.velocity_workflow import run_workflow
from glacier_offsets.velocity_workflow_config import load_workflow_config

from velocity_synthetic import Truth, config_text, pair_list, write_package


SUPPLIED = json.dumps({'covariance_ra_m2_by_year': {'2017': [[.01, .002], [.002, .04]], '2018': [[.02, .001], [.001, .05]]}})
ISCE3 = Path('/Users/briel/miniforge3/envs/isce3/bin/python')


def fits(path):
    """Every numerical dataset of a coefficient file."""
    result = {}
    with h5py.File(path) as h:
        def visit(name, item):
            if isinstance(item, h5py.Dataset) and item.dtype.kind in 'fiub?':
                result[name] = item[()]
        h.visititems(visit)
        result['names'] = json.loads(h['residual_vertical'].attrs['parameter_names'])
    return result


def assert_same_fits(test, a, b, atol=0.):
    test.assertEqual(a['names'], b['names'])
    for name in ('residual_vertical/coefficients', 'azimuth_direct/coefficients', 'flow_axis_EN',
                 'residual_vertical/rms_m', 'clean_pairs_by_year', 'residual_vertical/estimable'):
        if atol:
            assert_allclose(a[name], b[name], atol=atol, rtol=0, equal_nan=True, err_msg=name)
        else:
            assert_array_equal(a[name], b[name], err_msg=name)


class Workspace:
    def __init__(self, root):
        self.root = Path(root)
        (self.root/'noise.json').write_text(SUPPLIED)

    def config(self, inputs, output='out', noise='mode: supplied\nfiles:\n  - noise.json', min_span=300, extra=''):
        path = self.root/f'{output}.yaml'
        path.write_text(config_text(inputs, 'cache', output, noise, min_span=min_span, extra=extra))
        return path


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.truth = Truth([2017, 2018])
        self.pairs = pair_list('2017-07-01', 60)

    def tearDown(self):
        self.tmp.cleanup()

    def package(self, name, pairs=None, **options):
        return write_package(self.root/name, self.truth, self.pairs if pairs is None else pairs, **options)

    def test_order_independent_manifest_and_year_labels(self):
        a = self.package('a.nc', self.pairs[:25])
        b = self.package('b.nc', self.pairs[25:])
        one = collect([a, b], self.root/'cache')
        two = collect([b, a], self.root/'cache')
        self.assertEqual(content_identity(one), content_identity(two))
        self.assertEqual(one['years'], [2017, 2018])
        self.assertEqual(one['cross_year_pairs'], ['2017-12-28_2018-01-03'])
        cross = next(p for p in one['pairs'] if not p['within_year'])
        self.assertEqual(cross['year'], 2017)
        days = [p['calendar_early_day'] for p in one['pairs']]
        self.assertEqual(days, sorted(days))

    def test_geometry_projection_and_duplicate_checks(self):
        base = self.package('base.nc', self.pairs[:30])
        other = self.package('shifted.nc', self.pairs[30:], grid_shift=1.)
        with self.assertRaisesRegex(ValueError, 'incompatible x'):
            collect([base, other], self.root/'cache')
        within = self.package('within.nc', self.pairs[30:], perturb=5e-6)
        self.assertLess(max(collect([base, within], self.root/'cache')['geometry_max_abs_differences'][1].values()), 1e-5)
        beyond = self.package('beyond.nc', self.pairs[30:], perturb=5e-5)
        with self.assertRaisesRegex(ValueError, 'numerical-export tolerance'):
            collect([base, beyond], self.root/'cache')
        duplicate = self.package('duplicate.nc', self.pairs[29:40])
        with self.assertRaisesRegex(ValueError, 'Duplicate pair ID'):
            collect([base, duplicate], self.root/'cache')

    def test_correction_provenance(self):
        missing = self.package('missing.nc', drop=('tidal_correction_provenance_json',))
        with self.assertRaisesRegex(ValueError, 'ocean-tide correction provenance'):
            collect([missing], self.root/'cache')
        applied = self.package('applied.nc', extra_attrs={'solid_earth_correction_method': 'pysolid'})
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            collect([applied], self.root/'cache')
        tampered = self.package('tampered.nc')
        with h5py.File(tampered, 'r+') as h:
            h['range_displacement_m'][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            collect([tampered], self.root/'cache')

    def test_layout_overlap_is_exact_at_year_boundary(self):
        layout = Layout([2017, 2018])
        boundary = year_start(2018)
        overlap = layout.overlaps(np.array([boundary-2.5, boundary-8, np.nan]), np.array([boundary+3.5, boundary-2, 1.]))
        assert_allclose(overlap[:2], [[2.5, 3.5], [6., 0.]], atol=1e-12)
        self.assertTrue(np.isnan(overlap[2]).all())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.space = Workspace(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_requires_span_and_validates_noise(self):
        path = self.root/'bad.yaml'
        path.write_text('version: 2\ninputs: [a.nc]\ncache_directory: c\noutput: o\nnoise: {mode: supplied, files: [n.json]}\n')
        with self.assertRaisesRegex(ValueError, 'min_span_days'):
            load_workflow_config(path)
        path.write_text(config_text(['a.nc'], 'c', 'o', 'mode: controls'))
        with self.assertRaisesRegex(ValueError, 'noise.regions'):
            load_workflow_config(path)
        path.write_text(config_text(['a.nc'], 'c', 'o', 'mode: supplied\nfiles: [n.json]', extra='screening:\n  max_peak: 1\n'))
        with self.assertRaisesRegex(ValueError, 'Unknown screening keys'):
            load_workflow_config(path)
        config = load_workflow_config(self.space.config(['sub/a.nc']))
        self.assertEqual(config['inputs'], [str(self.root/'sub/a.nc')])
        self.assertFalse(config['screening']['remove_mixed_windows'])

    def test_impossible_settings_reported_before_preparation(self):
        truth = Truth([2017])
        write_package(self.root/'a.nc', truth, pair_list('2017-01-05', 30))
        config = self.space.config(['a.nc'], min_span=540)
        with self.assertRaisesRegex(ValueError, 'min_span_days=540'):
            run_workflow(config, 'prepare')
        self.assertFalse((self.root/'cache/observations').exists())

    def test_supplied_noise_mapping(self):
        legacy = self.root/'legacy.json'
        legacy.write_text(json.dumps({'covariance_by_year': [[[.01, 0], [0, .04]], [[.02, 0], [0, .05]]]}))
        with self.assertRaisesRegex(ValueError, 'unlabeled'):
            read_supplied([{'path': str(legacy), 'years': None}])
        by_year, _ = read_supplied([{'path': str(legacy), 'years': [2017, 2018]}])
        assert_allclose(by_year[2018], [[.02, 0], [0, .05]])
        with self.assertRaisesRegex(ValueError, 'more than one'):
            read_supplied([{'path': str(legacy), 'years': [2017, 2018]}, {'path': str(self.root/'noise.json'), 'years': None}])
        bad = self.root/'bad.json'
        bad.write_text(json.dumps({'covariance_ra_m2_by_year': {'2017': [[.01, .1], [.1, .04]]}}))
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            read_supplied([{'path': str(bad), 'years': None}])
        truth = Truth([2017, 2018])
        write_package(self.root/'a.nc', truth, pair_list('2017-07-01', 60))
        only = self.root/'only2017.json'
        only.write_text(json.dumps({'covariance_ra_m2_by_year': {'2017': [[.01, 0], [0, .04]]}}))
        config = self.space.config(['a.nc'], noise=f'mode: supplied\nfiles:\n  - {only}')
        with self.assertRaisesRegex(ValueError, r'no covariance for observed year\(s\) \[2018\]'):
            run_workflow(config, 'prepare')

    def test_single_and_multi_year_recovery_with_cross_year_pair(self):
        for years, start, count in (([2017], '2017-01-05', 55), ([2017, 2018], '2017-07-01', 60)):
            with self.subTest(years=years):
                truth = Truth(years)
                name = f'y{len(years)}'
                write_package(self.root/f'{name}.nc', truth, pair_list(start, count), dtype='f8')
                run_workflow(self.space.config([f'{name}.nc'], output=name), 'fit')
                result = fits(self.root/name/'source_coefficients.h5')
                index = {n: j for j, n in enumerate(result['names'])}
                coef = result['residual_vertical/coefficients']
                self.assertTrue(result['residual_vertical/estimable'].all())
                for year in years:
                    velocity = coef[:, [index[f'east_velocity_{year}'], index[f'north_velocity_{year}']]]
                    assert_allclose(velocity, truth.velocity[year], atol=1e-9)
                    assert_allclose(coef[:, index[f'order_{year}']], truth.order[year], atol=1e-9)
                assert_allclose(coef[:, [index['Msf_C'], index['Msf_S']]], truth.msf, atol=5e-3)
                with h5py.File(self.root/name/'source_coefficients.h5') as h:
                    if len(years) == 1:
                        assert_array_equal(h['year_direction_change_deg'][:], 0)
                    self.assertEqual(list(h.attrs['years']), years)
        manifest = json.loads((self.root/'y2/collection-manifest.json').read_text())
        cross = next(p for p in manifest['pairs'] if not p['within_year'])
        self.assertEqual((cross['year'], manifest['pairs_by_year']), (2017, {'2017': 31, '2018': 29}))
        # Every pair, including the cross-year pair, survives the screen of noise-free data.
        self.assertEqual(fits(self.root/'y2/source_coefficients.h5')['clean_pairs_by_year'].sum(), 60*12)

    def test_virtual_and_physical_concatenation_match_for_any_order(self):
        truth = Truth([2017, 2018])
        pairs = pair_list('2017-07-01', 60)
        noise = [[.01, .004], [.004, .04]]
        write_package(self.root/'whole.nc', truth, pairs, noise=noise, seed=8)
        with h5py.File(self.root/'whole.nc') as h:
            whole = {k: h[k][:] for k in ('range_displacement_m', 'azimuth_displacement_m')}
        # Split files carry exactly the same measurements.
        for name, part in (('first.nc', slice(0, 22)), ('second.nc', slice(22, 60))):
            write_package(self.root/name, truth, pairs[part], seed=8)
            with h5py.File(self.root/name, 'r+') as h:
                for key, value in whole.items():
                    h[key][:] = value[part]
            from glacier_offsets.velocity_io import variable_hashes
            with h5netcdf.File(self.root/name, 'r') as h:
                hashes = variable_hashes(h)
            with h5py.File(self.root/name, 'r+') as h:
                h.attrs['content_hashes_json'] = json.dumps(hashes)
        run_workflow(self.space.config(['whole.nc'], output='physical'), 'fit')
        run_workflow(self.space.config(['second.nc', 'first.nc'], output='virtual'), 'fit')
        run_workflow(self.space.config(['first.nc', 'second.nc'], output='virtual2'), 'fit')
        physical = fits(self.root/'physical/source_coefficients.h5')
        assert_same_fits(self, physical, fits(self.root/'virtual/source_coefficients.h5'))
        assert_same_fits(self, physical, fits(self.root/'virtual2/source_coefficients.h5'))
        # The two virtual orders share one observation artifact.
        self.assertEqual(len(list((self.root/'cache/observations').glob('[!.]*'))), 2)

    def test_control_noise_full_year_and_correlation(self):
        truth = Truth([2017, 2018], controls=range(6))
        write_package(self.root/'a.nc', truth, pair_list('2017-01-05', 115), noise=[[.01, .012], [.012, .04]], seed=5)
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [
            {'id': 'control_1', 'role': 'control', 'reviewed': True, 'bounds': [-2135100, 989750, -2133900, 990050]},
            {'id': 'control_2', 'role': 'control', 'reviewed': False, 'bounds': [-2135100, 989000, -2133900, 989700]}]}))
        manifest = run_workflow(self.space.config(['a.nc'], noise=f'mode: controls\nregions: {regions}', min_span=500), 'prepare')
        noise = json.loads((Path(manifest['artifacts']['noise']['directory'])/'noise.json').read_text())
        self.assertEqual(noise['controls'][0]['source_id'], list(range(6)))
        self.assertEqual(len(noise['controls']), 1)  # unreviewed controls are never used
        for record in noise['calibration']:
            self.assertGreater(record['samples'], 150)
            days = record['training_acquisition_days_since_2000']
            self.assertLess(min(days), year_start(record['year'])+31)
            self.assertGreater(max(days), year_start(record['year'])+300)  # full year, not January-June
        for year, d in noise['diagnostics'].items():
            self.assertAlmostEqual(d['sigma_range_m'], .1, delta=.025)
            self.assertAlmostEqual(d['sigma_azimuth_m'], .2, delta=.06)
            self.assertAlmostEqual(d['correlation'], .6, delta=.2)

    def test_insufficient_control_support_fails_by_year(self):
        truth = Truth([2017, 2018], controls=[0])
        write_package(self.root/'a.nc', truth, pair_list('2017-09-01', 50), noise=[[.01, 0], [0, .04]], seed=2)
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [
            {'id': 'control_1', 'role': 'control', 'reviewed': True, 'bounds': [-2135100, 989950, -2134700, 990050]}]}))
        config = self.space.config(['a.nc'], noise=f'mode: controls\nregions: {regions}', min_span=200)
        with self.assertRaisesRegex(ValueError, 'Year 2017 cannot be calibrated.*fewer than the required 40'):
            run_workflow(config, 'prepare')

    def test_mixed_window_policies_preserve_other_exclusions(self):
        truth = Truth([2017, 2018])
        pairs = pair_list('2017-07-01', 60)
        mixed = [(t, 4) for t in range(0, 60, 3)]
        write_package(self.root/'a.nc', truth, pairs, mixed=mixed, correction_unavailable=[(5, 6)],
                      tide_unavailable=[(9, 4), (10, 7)])
        for remove in (False, True):
            name = 'remove' if remove else 'keep'
            extra = f'screening:\n  remove_mixed_windows: {str(remove).lower()}\n'
            run_workflow(self.space.config(['a.nc'], output=name, extra=extra), 'fit')
        keep = fits(self.root/'keep/source_coefficients.h5')
        remove = fits(self.root/'remove/source_coefficients.h5')
        counts_keep, counts_remove = keep['clean_pairs_by_year'].sum(1), remove['clean_pairs_by_year'].sum(1)
        self.assertEqual(counts_keep[4], 59)          # tide exclusion kept, mixed windows retained
        self.assertEqual(counts_remove[4], 0)         # whole mixed source removed
        self.assertEqual(counts_keep[6], 59)          # unrelated correction gap still excluded
        self.assertEqual(counts_keep[7], 59)
        self.assertTrue(np.isnan(remove['residual_vertical/coefficients'][4]).all())
        others = np.r_[0:4, 5:12]
        assert_array_equal(keep['residual_vertical/coefficients'][others], remove['residual_vertical/coefficients'][others])
        with h5py.File(self.root/'keep/source_coefficients.h5') as h:
            assert_array_equal(np.flatnonzero(h['known_mixed_source'][:]), [4])

    def test_interrupted_preparation_and_fit_resume_identically(self):
        truth = Truth([2017, 2018])
        write_package(self.root/'a.nc', truth, pair_list('2017-07-01', 60), noise=[[.01, 0], [0, .04]], seed=1)
        run_workflow(self.space.config(['a.nc'], output='reference'), 'fit')
        reference = fits(self.root/'reference/source_coefficients.h5')
        shutil.rmtree(self.root/'cache')
        config = self.space.config(['a.nc'], output='resumed')
        with self.assertRaises(Interrupted):
            run_workflow(config, 'fit', interrupt={'stage': 'observations', 'after': 14})
        partial = next((self.root/'cache/observations').glob('.*.partial'))
        with h5py.File(partial/'observations.h5') as h:
            self.assertEqual(h.attrs['completed_pairs'], 14)
        with self.assertRaises(Interrupted):
            run_workflow(config, 'fit', interrupt={'stage': 'fit', 'after': 5})
        with h5py.File(self.root/'resumed/source_coefficients.partial.h5') as h:
            self.assertEqual(h.attrs['completed_sources'], 5)
        run_workflow(config, 'fit')
        assert_same_fits(self, reference, fits(self.root/'resumed/source_coefficients.h5'))
        self.assertFalse((self.root/'resumed/source_coefficients.partial.h5').exists())

    def test_stage_invalidation_and_writer_locks(self):
        truth = Truth([2017, 2018], controls=range(6))
        write_package(self.root/'a.nc', truth, pair_list('2017-01-05', 115), noise=[[.01, 0], [0, .04]], seed=6)
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [
            {'id': 'control_1', 'role': 'control', 'reviewed': True, 'bounds': [-2135100, 989750, -2133900, 990050]}]}))
        noise = f'mode: controls\nregions: {regions}'
        config = self.space.config(['a.nc'], noise=noise, min_span=500)
        first = run_workflow(config, 'run')
        fit_mtime = (self.root/'out/source_coefficients.h5').stat().st_mtime_ns
        figure = (self.root/'out/msf_quicklook.png').stat().st_mtime_ns
        # Display only: fits reused, figures redrawn.
        config = self.space.config(['a.nc'], noise=noise, min_span=500, extra='display:\n  min_speed_m_day: 0.2\n  proxy_min_speed_m_day: 0.6\n')
        second = run_workflow(config, 'run')
        self.assertEqual(second['artifacts'], first['artifacts'])
        self.assertEqual((self.root/'out/source_coefficients.h5').stat().st_mtime_ns, fit_mtime)
        self.assertNotEqual((self.root/'out/msf_quicklook.png').stat().st_mtime_ns, figure)
        self.assertIn('"min_speed_m_day": 0.2', h5py.File(self.root/'out/coefficient_maps.nc').attrs['display_settings_json'])
        # Screening: same observations, new control noise and fits.
        config = self.space.config(['a.nc'], noise=noise, min_span=500, extra='screening:\n  min_peak: 0.1\n')
        third = run_workflow(config, 'run')
        self.assertEqual(third['artifacts']['observations'], first['artifacts']['observations'])
        self.assertNotEqual(third['artifacts']['noise']['id'], first['artifacts']['noise']['id'])
        self.assertNotEqual((self.root/'out/source_coefficients.h5').stat().st_mtime_ns, fit_mtime)
        self.assertEqual(json.loads((self.root/'out/recipe.json').read_text())['parameters']['screening']['min_peak'], .1)
        # Writers are exclusive for result directories and cache artifacts.
        with writer_lock(self.root/'out'):
            with self.assertRaisesRegex(ValueError, 'Another velocity fitter'):
                run_workflow(config, 'run')
        artifact = Path(third['artifacts']['noise']['directory'])
        shutil.rmtree(self.root/'cache/noise')
        with exclusive_lock(artifact.parent/f'.{artifact.name}.lock', 'test'):
            with self.assertRaisesRegex(ValueError, 'Another process is writing noise'):
                run_workflow(config, 'prepare')

    def test_cli_run_outputs_and_postprocessing_from_elsewhere(self):
        truth = Truth([2017, 2018], controls=range(6))
        write_package(self.root/'a.nc', truth, pair_list('2017-01-05', 115), noise=[[.01, .003], [.003, .04]], seed=9)
        regions = self.root/'regions.json'
        regions.write_text(json.dumps({'EPSG': 3031, 'regions': [
            {'id': 'control_1', 'role': 'control', 'reviewed': True, 'bounds': [-2135100, 989750, -2133900, 990050]},
            {'id': 'target_1', 'role': 'target', 'reviewed': True, 'bounds': [-2135100, 989350, -2133900, 989650],
             'center_xy': [-2134500, 989500]}]}))
        config = self.space.config(['a.nc'], noise=f'mode: controls\nregions: {regions}', min_span=500)
        elsewhere = self.root/'elsewhere'
        elsewhere.mkdir()
        with chdir(elsewhere), np.errstate(all='ignore'):
            self.assertEqual(cli_main(['velocity-pixels', 'run', str(config)]), 0)
            out = self.root/'out'
            for name in ('coefficient_maps.nc', 'msf_along_amplitude.tif', 'residual_coverage_diagnostics.png',
                         'amplitude_phase_maps/msf_along_amplitude_phase.png', 'point_timeseries/T1.csv',
                         'verification.json', 'report.md', 'run-manifest.json'):
                self.assertTrue((out/name).is_file(), name)
            self.assertEqual(json.loads((out/'verification.json').read_text())['status'], 'passed')
            with h5netcdf.File(out/'coefficient_maps.nc') as h:
                self.assertIn('east_velocity_2018', h.variables)
                self.assertEqual(int(h.attrs['EPSG']), 3031)
            for command in (['verify', str(config)], ['report', str(out)], ['maps', str(config), '--component', 'all'],
                            ['export', str(out)], ['plot', str(out)]):
                self.assertEqual(cli_main(['velocity-pixels', *command]), 0, command)
            self.assertEqual(cli_main(['velocity-pixels', 'prepare', str(config)]), 0)
        # Results reference cache artifacts rather than copying them.
        self.assertFalse((self.root/'out/observations.h5').exists())

    @unittest.skipUnless(ISCE3.is_file(), 'isce3 interpreter with PySolid not available')
    def test_solid_earth_worker_matches_pysolid_and_resumes(self):
        truth = Truth([2017])
        write_package(self.root/'a.nc', truth, pair_list('2017-01-05', 24), noise=[[.01, 0], [0, .04]], seed=3)
        extra = f'execution:\n  solid_earth_python: {ISCE3}\n  preparation_chunk_pairs: 5\n'
        text = config_text(['a.nc'], 'cache', 'out', 'mode: supplied\nfiles:\n  - noise.json', min_span=100)
        text = text.replace('solid_earth: false', 'solid_earth: true').replace('execution:\n  batch_size: 5\n  preparation_chunk_pairs: 7\n', extra)
        (self.root/'set.yaml').write_text(text.replace('min_pairs_per_year: 10', 'min_pairs_per_year: 5').replace('min_pairs_total: 20', 'min_pairs_total: 10').replace('min_pairs_standard: 20', 'min_pairs_standard: 10'))
        with self.assertRaises(Interrupted):
            run_workflow(self.root/'set.yaml', 'prepare', interrupt={'stage': 'solid_earth', 'after': 10})
        manifest = run_workflow(self.root/'set.yaml', 'prepare')
        set_file = Path(manifest['artifacts']['solid_earth']['directory'])/'solid_earth.h5'
        obs_file = Path(manifest['artifacts']['observations']['directory'])/'observations.h5'
        with h5py.File(set_file) as h:
            resumed = h['set_ra_m'][:]
            self.assertLess(h.attrs['max_ENU_endpoint_difference_error_m'], 1e-5)
        uninterrupted = self.root/'uninterrupted.h5'
        from glacier_offsets.velocity_solid_earth import worker_file
        subprocess.run([str(ISCE3), '-I', str(worker_file()), '--observations', str(obs_file), '--output', str(uninterrupted)], check=True,
                       capture_output=True)
        with h5py.File(uninterrupted) as h:
            assert_array_equal(h['set_ra_m'][:], resumed)
        # Independent exact PySolid point check of one projected correction.
        script = ('import sys,json,numpy as np,h5py\nfrom datetime import datetime,timedelta,timezone\n'
                  'from pysolid.solid import solid_grid\n'
                  'h=h5py.File(sys.argv[1]);s,t=3,7;la,lo=h["latitude"][s],h["longitude"][s];g=h["geometry_by_file"][h["file_index"][t],s]\n'
                  'def tide(sec):\n  b=int(np.floor(sec));f=sec-b;v=[]\n  for o in (0,1):\n'
                  '    d=datetime(2000,1,1,tzinfo=timezone.utc)+timedelta(seconds=b+o)\n'
                  '    v.append(np.array(solid_grid(d.year,d.month,d.day,d.hour,d.minute,d.second,float(la),.01,1,float(lo),.01,1))[:,0,0])\n'
                  '  return v[0]*(1-f)+v[1]*f\n'
                  'print(json.dumps((g@(tide(h["late"][s,t]*86400.)-tide(h["early"][s,t]*86400.))).tolist()))\n')
        result = subprocess.run([str(ISCE3), '-I', '-c', script, str(obs_file)], check=True, capture_output=True, text=True)
        assert_allclose(resumed[3, 7], json.loads(result.stdout), atol=1e-5)
        self.assertGreater(np.max(abs(resumed[3, 7])), 1e-4)


if __name__ == '__main__':
    unittest.main()
