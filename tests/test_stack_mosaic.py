"""Exact parity of the ISCE3 merge-first compositor with the frozen GDAL merger."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal
from pyproj import CRS

from glacier_offsets import stack_mosaic as mosaic
from glacier_offsets.products import checksum
from glacier_offsets.raster_io import require_native_mmap
from glacier_offsets.stack_storage import read_manifest
from glacier_offsets.stack_workflow import execute_stack
from tools import reference_stack_merge as frozen
import test_isce2_strategy

PRF, SPACING = 500., 2.
SHAPE = (96, 160)
POLICIES = (mosaic.POLICY, mosaic.SWATH_MIDPOINT_POLICY)


def write(path, array, dtype, hole=None):
    ds = gdal.GetDriverByName('GTiff').Create(str(path), array.shape[1], array.shape[0], 1, dtype)
    ds.GetRasterBand(1).WriteArray(array)
    if hole is not None:
        # An explicit per-dataset mask is honored by the shared support policy.
        ds.CreateMaskBand(gdal.GMF_PER_DATASET)
        mask = np.full(array.shape, 255, 'uint8')
        mask[hole] = 0
        ds.GetRasterBand(1).GetMaskBand().WriteArray(mask)
    ds = None
    return str(path)


def slc(rng, path, *, hole=None, extra_zero=None):
    """Complex samples with connected zero fill, isolated zeros and nonfinite values."""
    data = (rng.uniform(.1, 1, SHAPE)*np.exp(1j*rng.uniform(-np.pi, np.pi, SHAPE))).astype('complex64')
    data[:2] = 0                       # Connected boundary fill.
    data[40:44, 60:64] = 0             # Interior fill (3x3-seeded component).
    data[20, 30] = data[70, 120] = 0   # Isolated zero observations remain valid.
    data[50, 80] = np.nan
    data[51, 81] = np.inf
    if extra_zero is not None:
        data[extra_zero] = 0
    return write(path, data, gdal.GDT_CFloat32, hole)


def sources(root, specs, rng):
    """Reference sources with per-source epochs: (burst ID, line, sample, epoch days, valid window)."""
    result = []
    for n, (key, dy, dx, days, window) in enumerate(specs):
        directory = Path(root)/key
        directory.mkdir(parents=True)
        epoch = datetime(2020, 1, 1, 8, 2, 39, 405820)-timedelta(days=days)
        grid = {'length': SHAPE[0], 'width': SHAPE[1], 'prf': PRF, 'range_pixel_spacing': SPACING,
                'wavelength': .05546576, 'lookside': 'right', 'ref_epoch': epoch.isoformat(),
                'sensing_start': dy/PRF + days*86400, 'starting_range': 8e5 + dx*SPACING}
        yy, xx = np.indices(SHAPE)
        geometry = {name: write(directory/f'{name}.tif', data, dtype) for name, data, dtype in [
            ('x', 1e3*n + xx + .5, gdal.GDT_Float64), ('y', -1e3*n - yy - .25, gdal.GDT_Float64),
            ('z', np.full(SHAPE, 10.*n), gdal.GDT_Float64),
            ('layover_shadow_mask', (xx+yy+n) % 3, gdal.GDT_Byte)]}
        coreg = {'reference_grid': grid, 'geometry': geometry, 'geometry_crs': CRS.from_epsg(4326).to_wkt(),
                 'slcs': {'reference': slc(rng, directory/'reference.tif')}}
        result.append({'burst_id': key, 'coreg': coreg, 'valid_window': list(window)})
    return result


def staggered(swaths, bursts, step, overlap, *, windows=None):
    """Sentinel-1-like staggered swath/burst placements."""
    specs = []
    for s in range(swaths):
        for b in range(bursts):
            window = (windows or {}).get((s, b), [3, 93, 4, 156])
            specs.append((f't001_{b:06d}_iw{s+1}', b*step + s*step//2, s*(SHAPE[1]-overlap), (s+b) % 3, window))
    return specs


def portable(value, root):
    """Replace the output root that stack records embed in their paths."""
    root = Path(root)
    return json.loads(json.dumps(value).replace(str(root.resolve()), '<root>').replace(str(root), '<root>'))


def read(path, mask=False):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    return (band.GetMaskBand() if mask else band).ReadAsArray()


class PlanParityTests(unittest.TestCase):
    def test_policy_records_and_selectors_preserve_compatibility_identities(self):
        self.assertEqual(mosaic.POLICY, frozen.POLICY)
        self.assertEqual(mosaic.SWATH_MIDPOINT_POLICY, frozen.SWATH_MIDPOINT_POLICY)
        for merge in ('overwrite', 'midpoint'):
            records = [mosaic.policy({'stack': {'processing': p, 'swath_merge': merge}}) for p in mosaic.SELECTORS]
            self.assertEqual(records[0], records[1])
        self.assertTrue(all(mosaic.enabled({'stack': {'processing': p}}) for p in mosaic.SELECTORS))
        self.assertFalse(mosaic.enabled({'stack': {'processing': 'isce3'}}))

    def test_placement_matches_epochs_rounding_tolerance_and_incompatible_grids(self):
        rng = np.random.default_rng(5)
        base = datetime(2016, 12, 30, 8, 2, 39, 405820)
        prf, spacing = 486.48631029955294, 2.329562114715323
        for trial in range(400):
            records = []
            for _ in range(rng.integers(1, 7)):
                days = int(rng.integers(0, 4))
                epoch = base + timedelta(days=days, microseconds=int(rng.integers(0, 10**6)))
                line, sample = int(rng.integers(0, 20000)), int(rng.integers(0, 60000))
                # Errors span accepted, boundary and rejected fractional placements.
                jitter = rng.choice([0, 1e-4, 9.9e-4, 2e-3, .3], 2)*rng.choice([-1, 1], 2)
                records.append({'length': int(rng.integers(1, 2000)), 'width': int(rng.integers(1, 30000)),
                                'prf': prf, 'range_pixel_spacing': spacing, 'wavelength': .05546576,
                                'lookside': 'right', 'ref_epoch': epoch.isoformat(),
                                'sensing_start': (base-epoch).total_seconds() + 172800 + (line+jitter[0])/prf,
                                'starting_range': 8e5 + (sample+jitter[1])*spacing})
            if trial % 25 == 1:
                records[-1]['prf'] *= 1+1e-6
            if trial % 25 == 2:
                records[-1]['lookside'] = 'left'
            if trial % 25 == 3:
                records[-1]['width'] = 0
            if trial % 25 == 4:
                records[-1]['sensing_start'] = np.nan
            tolerance = (.001, .01, 0.)[trial % 3]
            with self.subTest(trial=trial):
                try:
                    expected = frozen.radar_layout(deepcopy(records), tolerance)
                except ValueError as exc:
                    with self.assertRaisesRegex(ValueError, f'^{re.escape(str(exc))}$'):
                        mosaic.place_sources(deepcopy(records), tolerance)
                    continue
                self.assertEqual(mosaic.place_sources(deepcopy(records), tolerance), expected)
        with self.assertRaisesRegex(ValueError, 'No selected radar grids'):
            mosaic.place_sources([])

    def test_ownership_matches_staggered_swaths_narrow_overlaps_and_input_order(self):
        rng = np.random.default_rng(11)
        cases = 0
        for trial in range(300):
            entries = []
            swaths, bursts = int(rng.integers(1, 4)), int(rng.integers(1, 5))
            step, overlap = int(rng.integers(40, 200)), int(rng.integers(-5, 120))
            for s in range(swaths):
                for b in range(bursts):
                    y0, x0 = int(rng.integers(0, 12)), int(rng.integers(0, 12))
                    name = f'burst_{b}_iw{s+1}' if trial % 7 else f'burst_{b}_{s}'
                    entries.append({'burst_id': name,
                                    'offset': [b*step + s*int(rng.integers(0, step)), s*(400-overlap)],
                                    'valid_window': [y0, int(rng.integers(y0+1, 200)), x0, int(rng.integers(x0+1, 400))]})
            order = rng.permutation(len(entries))
            entries = [entries[i] for i in order]
            for midpoint in (False, True):
                with self.subTest(trial=trial, midpoint=midpoint):
                    try:
                        expected = frozen.overwrite_ownership(deepcopy(entries), swath_midpoint=midpoint)
                    except ValueError as exc:
                        with self.assertRaisesRegex(ValueError, f'^{re.escape(str(exc))}$'):
                            mosaic.ownership(deepcopy(entries), swath_midpoint=midpoint)
                        continue
                    cases += 1
                    self.assertEqual(mosaic.ownership(deepcopy(entries), swath_midpoint=midpoint), expected)
        self.assertGreater(cases, 300)

    def test_validity_adjustment_matches_every_sign_case_clip_and_empty_result(self):
        rng = np.random.default_rng(17)
        values = [0., .4, 1., 3.8, 7.2, 300., -.4, -1., -3.8, -7.2, -300.]
        for trial in range(3000):
            shape = tuple(int(v) for v in rng.integers(20, 400, 2))
            y0, x0 = int(rng.integers(-10, shape[0])), int(rng.integers(-10, shape[1]))
            secondary = [y0, y0+int(rng.integers(1, shape[0]+20)), x0, x0+int(rng.integers(1, shape[1]+20))]
            extrema = [sorted(rng.choice(values, 2)) for _ in range(2)]
            reference = None if trial % 2 else [2, shape[0]-2, 3, shape[1]-3]
            with self.subTest(trial=trial):
                try:
                    expected = frozen.adjusted_window(secondary, shape, extrema, reference)
                except ValueError as exc:
                    with self.assertRaisesRegex(ValueError, f'^{re.escape(str(exc))}$'):
                        mosaic.adjusted_window(secondary, shape, extrema, reference)
                    continue
                self.assertEqual(mosaic.adjusted_window(secondary, shape, extrema, reference), expected)

    def test_offset_extrema_match_sentinels_nonfinite_values_and_failures(self):
        rng = np.random.default_rng(23)
        with tempfile.TemporaryDirectory() as d:
            for n, dtype in enumerate((gdal.GDT_Float32, gdal.GDT_Float64)):
                data = rng.normal(0, 3, (150, 70))
                data[rng.random(data.shape) < .1] = np.nan
                data[rng.random(data.shape) < .1] = -1e6
                data[3, 4] = np.inf
                path = write(Path(d)/f'{n}.off', data, dtype)
                self.assertEqual(mosaic.offset_extrema(path), frozen.offset_extrema(path))
            bad = write(Path(d)/'bad.off', np.full((65, 9), -1e6), gdal.GDT_Float32)
            for function in (frozen.offset_extrema, mosaic.offset_extrema):
                with self.assertRaisesRegex(ValueError, 'No usable resampling offsets'):
                    function(bad)
            # GDAL's process-wide exception mode decides the missing-file error type.
            with self.assertRaises((ValueError, RuntimeError)) as expected:
                frozen.offset_extrema(Path(d)/'missing.off')
            with self.assertRaises(type(expected.exception)):
                mosaic.offset_extrema(Path(d)/'missing.off')

    def test_common_intersection_excludes_partial_dates_and_rejects_empty(self):
        entries = [{'burst_id': 'a', 'valid_window': [4, 90, 4, 150]},
                   {'burst_id': 'b', 'valid_window': [2, 80, 6, 140], 'reference_valid_window': [1, 95, 2, 158]}]
        dates = {'d1': {'slcs': {'a': 1, 'b': 1}, 'valid_windows': {'a': {'window': [6, 88, 1, 149]},
                                                                    'b': {'window': [0, 91, 9, 150]}}},
                 'd2': {'slcs': {'a': 1}, 'valid_windows': {'a': {'window': [50, 60, 50, 60]}}}}
        self.assertEqual(mosaic.common_windows(entries, dates), frozen.common_windows(entries, dates))
        self.assertEqual(mosaic.common_windows(entries, dates)[2], ['d2'])
        dates['d1']['valid_windows']['a']['window'] = [91, 95, 1, 149]
        for function in (frozen.common_windows, mosaic.common_windows):
            with self.assertRaisesRegex(ValueError, 'a: stack-common valid rectangle is empty'):
                function(entries, dates)


class RasterParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(29)

    def layouts(self, name, entries, policy):
        root = self.root/name
        try:
            reference = frozen.write_layout(root/'reference', entries, ownership_policy=policy)
        except ValueError as exc:
            with self.assertRaisesRegex(ValueError, f'^{re.escape(str(exc))}$'):
                mosaic.write_layout(root/'isce3', entries, ownership_policy=policy)
            return None, None
        layout = mosaic.write_layout(root/'isce3', entries, ownership_policy=policy)
        self.assertEqual(layout['sources'], reference['sources'])
        self.assertEqual(layout['ownership'], reference['ownership'])
        self.assertEqual(layout['coreg']['reference_grid'], reference['coreg']['reference_grid'])
        self.assertEqual(layout['coreg']['assembly_sources'], reference['coreg']['assembly_sources'])
        self.assertEqual(layout['layout_implementation'], mosaic.IMPLEMENTATION)
        owner = read(layout['coreg']['source_index'])
        np.testing.assert_array_equal(owner, read(reference['coreg']['source_index']))
        self.assertEqual(owner.dtype, np.uint16)
        plan = mosaic.MergePlan.from_frame(layout)
        np.testing.assert_array_equal(plan.owner_block(0, plan.shape[0]), owner)
        self.assertEqual(plan.radar_grid.shape, plan.shape)
        for field in mosaic.GEOMETRY_FIELDS:
            np.testing.assert_array_equal(read(layout['coreg']['geometry'][field]),
                                          read(reference['coreg']['geometry'][field]))
        if checksum(layout['coreg']['source_index']) == checksum(reference['coreg']['source_index']):
            self.assertEqual(layout['layout_identity'], reference['layout_identity'])
        return reference, layout

    def dates(self, entries, count=3):
        """Per-date aligned SLCs, including missing bursts and date-specific holes."""
        result = []
        for day in range(count):
            slcs = {}
            for n, e in enumerate(entries):
                if (day+n) % 4 == 3:
                    continue  # Missing date burst: never filled from another source.
                hole = (slice(10+day, 30), slice(100, 140)) if n == day else None
                extra = (slice(60, 61), slice(5+n, 6+n)) if day == 1 else None
                slcs[e['burst_id']] = slc(self.rng, self.root/f'date{day}-{e["burst_id"]}.tif', hole=hole, extra_zero=extra)
            result.append(slcs)
        result.append({e['burst_id']: e['coreg']['slcs']['reference'] for e in entries})
        result.append({})
        return result

    def compose(self, name, reference, layout, slcs):
        root = self.root/name
        root.mkdir()
        expected = frozen.compose(reference, slcs, root/'reference.tif', root/'reference-valid.tif')
        count = mosaic.compose(mosaic.MergePlan.from_frame(layout), slcs, root/'isce3.tif', root/'isce3-valid.tif')
        self.assertEqual(count, expected)
        values = read(root/'isce3.tif')
        self.assertEqual(values.dtype, np.complex64)
        np.testing.assert_array_equal(values, read(root/'reference.tif'))
        np.testing.assert_array_equal(read(root/'isce3.tif', mask=True), read(root/'reference.tif', mask=True))
        valid = read(root/'isce3-valid.tif')
        np.testing.assert_array_equal(valid, read(root/'reference-valid.tif'))
        self.assertEqual(int(valid.sum()), count)
        self.assertTrue(np.isfinite(values).all())
        self.assertFalse(values[valid == 0].any())
        ds = gdal.Open(str(root/'isce3.tif'))
        self.assertEqual(ds.GetRasterBand(1).GetMaskFlags(), gdal.GMF_PER_DATASET)
        self.assertEqual(ds.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE'), 'BAND')
        self.assertEqual(ds.GetRasterBand(1).GetBlockSize()[0], ds.RasterXSize)
        self.assertTrue(Path(f'{root}/isce3.tif.msk').is_file())
        require_native_mmap(ds)
        return values, valid

    def test_two_and_three_swath_layouts_and_dates_match_for_both_policies(self):
        composed = 0
        for swaths, bursts, step, overlap in [(2, 2, 70, 40), (3, 3, 60, 24), (1, 4, 50, 0), (3, 2, 80, 16)]:
            windows = {(0, 1): [9, 85, 2, 150], (1, 0): [3, 93, 12, 158]}
            entries = sources(self.root/f'sources-{swaths}-{bursts}', staggered(swaths, bursts, step, overlap,
                                                                             windows=windows), self.rng)
            dates = self.dates(entries)
            for policy in POLICIES:
                name = f'{swaths}-{bursts}-{policy["method"]}'
                with self.subTest(name):
                    reference, layout = self.layouts(name, entries[::-1], policy)
                    composed += layout is not None
                    for day, slcs in enumerate(dates if layout else []):
                        values, valid = self.compose(f'{name}-{day}', reference, layout, slcs)
                        if not slcs:
                            self.assertFalse(valid.any())
        self.assertEqual(composed, 6)  # Both 8-sample midpoint overlaps are rejected identically.

    def test_nonuniform_epochs_and_narrow_overlaps_fail_identically(self):
        entries = sources(self.root/'narrow', staggered(2, 1, 64, 12), self.rng)
        reference, layout = self.layouts('narrow-overwrite', entries, mosaic.POLICY)
        self.compose('narrow-overwrite-date', reference, layout, self.dates(entries, 1)[0])
        for function in (frozen.write_layout, mosaic.write_layout):
            with self.assertRaisesRegex(ValueError, 'overlap has 4 samples'):
                function(self.root/'narrow-midpoint'/function.__module__, entries,
                         ownership_policy=mosaic.SWATH_MIDPOINT_POLICY)
        entries[1]['coreg']['reference_grid']['starting_range'] += .5*SPACING
        for function in (frozen.write_layout, mosaic.write_layout):
            with self.assertRaisesRegex(ValueError, 'fractional radar placement'):
                function(self.root/'fractional'/function.__module__, entries, ownership_policy=mosaic.POLICY)

    def test_composition_bounds_memory_to_blocks_and_closes_rasters(self):
        entries = sources(self.root/'blocks', staggered(2, 3, 60, 30), self.rng)
        layout = mosaic.write_layout(self.root/'blocks-layout', entries, ownership_policy=mosaic.POLICY)
        plan = mosaic.MergePlan.from_frame(layout)
        calls = []
        original = mosaic.MergePlan.owner_block

        def record(self, row, stop):
            calls.append(stop-row)
            return original(self, row, stop)
        with patch.object(mosaic.MergePlan, 'owner_block', record), \
             patch.object(mosaic, 'BLOCK_LINES', 16):
            mosaic.compose(plan, self.dates(entries, 1)[0], self.root/'blocks.tif', self.root/'blocks-valid.tif')
        self.assertEqual(sum(calls), plan.shape[0])
        self.assertLessEqual(max(calls), 16)
        # The SLC handle owns its external mask band; they are never open together
        # for update (concurrent writers made closing take minutes on exFAT).
        opened, open_ = set(), mosaic._open

        class Tracked:
            def __init__(self, path, update=False):
                self.raster, self.path = open_(path, update), str(path)
                self.slc = self.path.removesuffix('.msk')
                if update:
                    if self.slc in opened or f'{self.slc}.msk' in opened:
                        raise AssertionError(f'Concurrent SLC/mask writers: {self.path}')
                    opened.add(self.path)

            def __getattr__(self, name):
                return getattr(self.raster, name)

            def close_dataset(self):
                opened.discard(self.path)
                self.raster.close_dataset()
        with patch.object(mosaic, '_open', Tracked):
            mosaic.compose(plan, self.dates(entries, 1)[0], self.root/'tracked.tif', self.root/'tracked-valid.tif')
        self.assertFalse(opened)
        # Closed outputs can be renamed and reopened by another process.
        for name in ('blocks.tif', 'blocks.tif.msk', 'blocks-valid.tif'):
            (self.root/name).replace(self.root/f'moved-{name}')
        self.assertEqual(gdal.Open(str(self.root/'moved-blocks.tif')).RasterYSize, plan.shape[0])


class MergeFirstWorkflowTests(unittest.TestCase):
    """End-to-end stacks on the synthetic native fixture, using the ISCE3 matcher."""
    fixture = test_isce2_strategy.ISCE2LayoutTests
    setUp = fixture.setUp
    make_plan = fixture.make_plan
    reference_product = fixture.reference_product
    geometry = fixture.geometry
    effective = fixture.effective
    align_product = fixture.align_product

    def configure(self, processing, name):
        cfg = deepcopy(self.cfg)
        cfg['stack']['processing'] = processing
        cfg['matching'].update(engine='isce3', deramp=0)
        cfg['gross_offsets'] = {'mode': 'constant', 'constant': {'azimuth_px': 1, 'range_px': -1}}
        cfg['output']['directory'] = str(self.root/name)
        self.plan.anchor['selection'] = {'selected_ids': list(self.reference.bursts),
            'polarization': cfg['inputs']['polarization'], 'burst_id': cfg['inputs']['burst_id'],
            'subswaths': cfg['inputs']['subswaths'], 'aoi': {'specification': cfg.get('aoi')}}
        return cfg

    @contextmanager
    def workflow(self):
        with ExitStack() as stack:
            for target, value in [('plan_stack', self.plan), ('inspect_environment', self.software)]:
                for module in ('stack_workflow', 'stack_rebuild'):
                    stack.enter_context(patch(f'glacier_offsets.{module}.{target}', return_value=value))
            for target in ('stack_workflow.validate_stack', 'stack_rebuild.validate_stack', 'stack_workflow.select_pair'):
                stack.enter_context(patch(f'glacier_offsets.{target}'))
            yield

    @contextmanager
    def frozen_merger(self):
        """Write layouts and compose dates with the frozen GDAL merger, as
        the stacks produced before the ISCE3 cutover were."""
        def layout(root, sources, *, checksums=None, ownership_policy):
            return frozen.write_layout(root, sources, ownership_policy=ownership_policy)
        with ExitStack() as stack:
            stack.enter_context(patch.object(mosaic, 'write_layout', layout))
            stack.enter_context(patch.object(mosaic.MergePlan, 'from_frame', staticmethod(lambda frame: frame)))
            stack.enter_context(patch.object(mosaic, 'compose', frozen.compose))
            yield

    def reference_arrays(self, root, name):
        """Frozen-reference composition of every saved date on the saved layout."""
        frame = read_manifest(root/'geometry')
        result = {}
        for a in self.acquisitions:
            date = read_manifest(root/'acquisitions'/a.date)
            directory = self.root/name/a.date
            directory.mkdir(parents=True)
            count = frozen.compose(frame, date['slcs'], directory/'slc.tif', directory/'valid.tif')
            result[a.date] = (read(directory/'slc.tif'), read(directory/'valid.tif'), count)
        return result

    def assert_dates_equal(self, root, expected):
        for day, (values, valid, count) in expected.items():
            date = read_manifest(root/'acquisitions'/day)
            np.testing.assert_array_equal(read(date['slc']), values)
            np.testing.assert_array_equal(read(date['slc'], mask=True), valid*255)
            np.testing.assert_array_equal(read(date['validity_mask']), valid)
            self.assertEqual(date['valid_samples'], count)
            require_native_mmap(gdal.Open(date['slc']))

    def test_isce3_workflow_reproduces_frozen_merger_layouts_dates_offsets_and_known_translations(self):
        roots = {}
        with self.workflow():
            # The deprecated alias selects the ISCE3 compositor too.
            for name, processing, merger in (('gdal', 'isce3_merge_first', self.frozen_merger),
                                             ('isce3', 'isce2_merge_first', ExitStack)):
                cfg = self.configure(processing, name)
                cfg['storage'] = {'keep_merged_slcs': True}
                if roots:
                    # Separate roots measure their own FFTW plans; share them to compare bitwise.
                    shutil.copytree(roots['gdal']/'.cache', Path(cfg['output']['directory'])/'.cache')
                with merger():
                    self.assertEqual(execute_stack(cfg, 'run'), 0)
                roots[name] = Path(cfg['output']['directory'])
        reference, new = roots['gdal'], roots['isce3']
        old, frame = read_manifest(reference/'geometry'), read_manifest(new/'geometry')

        for key in ('sources', 'ownership', 'common_validity', 'map_grid', 'geometry_identity'):
            self.assertEqual(portable(frame[key], new), portable(old[key], reference), key)
        self.assertEqual(frame['layout_implementation'], mosaic.IMPLEMENTATION)
        self.assertNotIn('layout_implementation', old)
        np.testing.assert_array_equal(read(frame['coreg']['source_index']), read(old['coreg']['source_index']))
        self.assertEqual(frame['layout_identity'], old['layout_identity'])  # Identical TIFF bytes here.
        for a in self.acquisitions:
            before, after = (read_manifest(r/'acquisitions'/a.date) for r in (reference, new))
            self.assertEqual(after['assembly_identity'], before['assembly_identity'])
            self.assertEqual(portable(after['valid_windows'], new), portable(before['valid_windows'], reference))
            self.assertEqual(after['assembly_implementation'], mosaic.IMPLEMENTATION)
            for field in ('slc', 'validity_mask'):
                np.testing.assert_array_equal(read(after[field]), read(before[field]))
            np.testing.assert_array_equal(read(after['slc'], mask=True), read(before['slc'], mask=True))
        by_date = {a.date: n for n, a in enumerate(self.acquisitions)}
        for earlier, later in self.plan.pairs:
            pair = f'{earlier}_{later}'
            native = read_manifest(new/'pairs'/pair)['matching']['native_parameters']
            wisdom = Path(native['fftw_wisdom']['path'])
            self.assertEqual(wisdom.parent, new/'.cache/fftw')
            self.assertEqual(checksum(wisdom), native['fftw_wisdom']['sha256'])
            self.assertEqual({s['fftw_wisdom_extended'] for s in native['segments']}, {False})
            # Pair identities name the geometry manifest, so compare their specifications.
            self.assertEqual(portable(read_manifest(new/'pairs'/pair)['coreg_specification'], new),
                             portable(read_manifest(reference/'pairs'/pair)['coreg_specification'], reference))
            with h5py.File(reference/'pairs'/pair/'radar/offsets.h5') as h, h5py.File(new/'pairs'/pair/'radar/offsets.h5') as g:
                names = []
                h.visit(lambda name: names.append(name) if isinstance(h[name], h5py.Dataset) else None)
                for name in names:
                    np.testing.assert_array_equal(g[name][()], h[name][()], err_msg=name)
                good = g['valid'][:]
                self.assertGreater(int(good.sum()), 20)
                expected = np.subtract(self.translations[by_date[later]], self.translations[by_date[earlier]])
                for k, axis in enumerate(('azimuth', 'range')):
                    total, gross = g[f'total/{axis}_px'][:], g[f'gross/{axis}_px'][:]
                    np.testing.assert_array_equal(total, gross+g[f'residual/{axis}_px'][:])
                    np.testing.assert_array_equal(gross[good], (1, -1)[k])
                    np.testing.assert_allclose(total[good], expected[k], atol=.2)
            for name in ('range_displacement_m.tif', 'azimuth_displacement_m.tif', 'valid_mask.tif'):
                np.testing.assert_array_equal(read(new/'pairs'/pair/'geo'/name), read(reference/'pairs'/pair/'geo'/name))

    def test_range2x_stack_matching_recovers_known_translations_in_original_units(self):
        from glacier_offsets import range_oversampling
        from glacier_offsets.stack_pairs import matching_template
        cfg = self.configure('isce3_merge_first', 'range2x')
        cfg['matching'].update(range_oversampling=2, backend='cpu')
        scratch = self.root/'scratch'
        scratch.mkdir()
        cfg['storage'] = {'scratch_directory': str(scratch)}
        root = Path(cfg['output']['directory'])
        with self.workflow():
            self.assertEqual(execute_stack(cfg, 'run'), 0)
        self.assertEqual(list(scratch.iterdir()), [])  # Temporary inputs are released.
        grid = matching_template(cfg, read_manifest(root/'geometry'))
        by_date = {a.date: n for n, a in enumerate(self.acquisitions)}
        for earlier, later in self.plan.pairs:
            directory = root/'pairs'/f'{earlier}_{later}'
            native = read_manifest(directory)['matching']['native_parameters']
            self.assertEqual(native['range_oversampling']['method'], range_oversampling.METHOD)
            self.assertEqual({s['range_oversampling']['factor_yx'][1] for s in native['segments']}, {2})
            self.assertEqual({s['windowSizeWidth'] for s in native['segments']}, {2*cfg['matching']['window'][1]})
            self.assertFalse((directory/'radar/native/range2x').exists())
            with h5py.File(directory/'radar/offsets.h5') as h:
                np.testing.assert_array_equal(h['center_sample'][:], grid.centers[1])
                good = h['valid'][:]
                self.assertGreater(int(good.sum()), 20)
                expected = np.subtract(self.translations[by_date[later]], self.translations[by_date[earlier]])
                for k, axis in enumerate(('azimuth', 'range')):
                    gross = h[f'gross/{axis}_px'][:]
                    np.testing.assert_array_equal(gross[good], (1, -1)[k])
                    np.testing.assert_allclose(h[f'total/{axis}_px'][:][good], expected[k], atol=.2)

    def test_cutover_reuses_frozen_merger_pairs_and_reassembles_released_dates_exactly(self):
        from glacier_offsets.stack_status import inspect_stack
        cfg = self.configure('isce2_merge_first', 'switch')
        root = Path(cfg['output']['directory'])
        with self.workflow():
            with self.frozen_merger():
                self.assertEqual(execute_stack(cfg, 'run'), 0)
            expected = self.reference_arrays(root, 'frozen')
            frame = read_manifest(root/'geometry')
            layout = [root/'geometry/manifest.json', Path(frame['coreg']['source_index'])]
            before = {p: (checksum(p), p.stat().st_mtime_ns) for p in layout}
            identities = {a.date: read_manifest(root/'acquisitions'/a.date)['assembly_identity'] for a in self.acquisitions}
            self.assertFalse((root/'acquisitions'/self.acquisitions[0].date/'slc.tif').exists())
            cfg['stack']['processing'] = 'isce3_merge_first'
            with patch('glacier_offsets.stack_pairs.match_prepared', side_effect=AssertionError('matching rerun')):
                self.assertEqual(execute_stack(cfg, 'run'), 0)
            self.assertEqual(before, {p: (checksum(p), p.stat().st_mtime_ns) for p in layout})
            report = inspect_stack(cfg)
            for stage in ('coregister', 'match', 'geocode', 'diagnostics'):
                self.assertEqual(report['stages'][stage]['status'], 'complete', stage)
            # Released dates are rebuilt by the ISCE3 compositor with unchanged identities.
            self.assertEqual(execute_stack(cfg, 'assemble'), 0)
            self.assert_dates_equal(root, expected)
            for a in self.acquisitions:
                date = read_manifest(root/'acquisitions'/a.date)
                self.assertEqual(date['assembly_identity'], identities[a.date])
                self.assertEqual(date['assembly_implementation'], mosaic.IMPLEMENTATION)
                self.assertTrue(set(map(str, [date['slc'], f'{date["slc"]}.msk', date['validity_mask']])) <= set(date['products']))
            self.assertEqual(before, {p: (checksum(p), p.stat().st_mtime_ns) for p in layout})
            self.assertEqual(self.reference_call.call_count, 2)
            self.assertEqual(self.align_call.call_count, 4)

    def test_interrupted_and_damaged_assemblies_rebuild_identically(self):
        cfg = self.configure('isce3_merge_first', 'restart')
        root = Path(cfg['output']['directory'])
        with self.workflow():
            self.assertEqual(execute_stack(cfg, 'coregister'), 0)
            self.assertEqual(execute_stack(cfg, 'assemble'), 0)
            expected = self.reference_arrays(root, 'frozen-restart')
            self.assert_dates_equal(root, expected)
            original = mosaic.MergePlan.owner_block
            calls = []

            def interrupt(plan, row, stop):
                calls.append(row)
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                return original(plan, row, stop)
            with patch.object(mosaic.MergePlan, 'owner_block', interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute_stack(cfg, 'assemble', force=True)
            day = self.acquisitions[0].date
            self.assertNotEqual(read_manifest(root/'acquisitions'/day).get('status'), 'complete')
            self.assertFalse(list((root/'acquisitions'/day).glob('.merge-*')))
            self.assertEqual(execute_stack(cfg, 'assemble'), 0)
            self.assert_dates_equal(root, expected)
            damaged = Path(read_manifest(root/'acquisitions'/day)['slc'])
            with damaged.open('r+b') as stream:
                stream.seek(-64, 2)
                stream.write(b'\x7f'*64)
            self.assertEqual(execute_stack(cfg, 'assemble'), 0)
            self.assert_dates_equal(root, expected)
            self.assertEqual(self.align_call.call_count, 4)

    def test_rebuild_layout_under_canonical_selector_matches_frozen_midpoint(self):
        from glacier_offsets.stack_rebuild import rebuild_layout
        cfg = self.configure('isce3_merge_first', 'rebuild')
        root = Path(cfg['output']['directory'])
        with self.workflow():
            self.assertEqual(execute_stack(cfg, 'coregister'), 0)
            cfg['stack']['swath_merge'] = 'midpoint'
            self.assertEqual(rebuild_layout(cfg)['status'], 'complete')
            self.assertEqual(execute_stack(cfg, 'assemble'), 0)
            frame = read_manifest(root/'geometry')
            control = frozen.write_layout(self.root/'frozen-midpoint', frame['sources'],
                                          ownership_policy=mosaic.SWATH_MIDPOINT_POLICY)
            self.assertEqual(frame['ownership'], control['ownership'])
            self.assertEqual(frame['sources'], control['sources'])
            np.testing.assert_array_equal(read(frame['coreg']['source_index']), read(control['coreg']['source_index']))
            self.assert_dates_equal(root, self.reference_arrays(root, 'frozen-rebuild'))


if __name__ == '__main__':
    unittest.main()
