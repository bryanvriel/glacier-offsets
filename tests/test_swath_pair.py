"""Adjacent-swath selection, observation signs, and unsupported-swath isolation."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.swath_pair import select_pair, recorded_pair, validate_pair
from glacier_offsets.swath_observations import overlap_grid
from glacier_offsets.swath_correction import correction_arrays
from glacier_offsets.swath_experiment_io import cohorts, save_npz, Experiment
from glacier_offsets.swath_experiment_analysis import observation_rows, closure_report
from glacier_offsets.swath_experiment_previews import symmetric_bias, corrected_preview
from glacier_offsets.swath_physics import load_native


def sources(swaths):
    return [{'burst_id': f't065_139088_iw{s}', 'offset': [0, 0],
             'valid_window': [0, 1000, 0, 1000]} for s in swaths]


class SwathPairTests(unittest.TestCase):
    def test_constructor_infers_pair_and_rejects_missing_or_ambiguous_sources_before_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stack = root/'stack'; (stack/'geometry').mkdir(parents=True)
            cfg = {'output': {'directory': str(stack)},
                   'matching': {'engine': 'isce3', 'backend': 'cpu', 'deramp': 0, 'range_oversampling': 2,
                                'window': [64, 256], 'search': [10, 40], 'stride': [32, 128]},
                   'gross_offsets': {'mode': 'zero'}, 'geocoding': {'resampling': 'nearest'}}
            frame = {'status': 'complete', 'sources': sources((2, 3)),
                     'geometry_identity': 'geometry', 'layout_identity': 'layout',
                     'coreg': {'reference_grid': {'length': 1000, 'width': 1000}}}
            path = stack/'geometry/manifest.json'
            path.write_text(json.dumps(frame))
            (stack/'stack_manifest.json').write_text('{"inventory": []}')
            with patch('glacier_offsets.stack_config.load_stack_config', return_value=cfg):
                exp = Experiment(root/'config.yaml', root/'results')
                self.assertEqual(exp.swaths, (2, 3))
                self.assertEqual(exp.spec['swaths'], [2, 3])
                frame['sources'] = sources((1, 2, 3)); path.write_text(json.dumps(frame))
                with self.assertRaisesRegex(ValueError, 'three swaths'):
                    Experiment(root/'config.yaml', root/'ambiguous')
                self.assertFalse((root/'ambiguous').exists())
                exp = Experiment(root/'config.yaml', root/'selected', swaths=(1, 2))
                self.assertEqual(exp.swaths, (1, 2))
                self.assertNotIn('swaths', exp.spec)  # Existing IW1/IW2 experiment identity.
                frame['sources'] = sources((2,)); path.write_text(json.dumps(frame))
                with self.assertRaisesRegex(ValueError, 'two adjacent swaths'):
                    Experiment(root/'config.yaml', root/'missing')

    def test_inference_explicit_selection_and_legacy_records(self):
        for pair in ((1, 2), (2, 3)):
            self.assertEqual(select_pair(sources(pair)), pair)
            self.assertEqual(select_pair(sources((1, 2, 3)), pair), pair)
            self.assertEqual(recorded_pair({'swaths': np.array(pair)}), pair)
        self.assertEqual(recorded_pair({}), (1, 2))
        for pair in ((1, 3), (3, 2), (2, 2), (True, 2), (), None):
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                validate_pair(pair)
        with self.assertRaisesRegex(ValueError, 'three swaths'):
            select_pair(sources((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, 'absent'):
            select_pair(sources((2, 3)), (1, 2))
        with self.assertRaisesRegex(ValueError, 'two adjacent swaths'):
            select_pair(sources((2,)))

    def test_identical_geometry_accepts_both_pairs_and_reports_actual_narrow_overlap(self):
        full = MatchingGrid(np.array([10, 42]), np.array([40, 168]), (64, 256), (10, 40), (32, 128))
        for pair in ((1, 2), (2, 3)):
            frame = {'sources': sources(pair)}
            with patch('glacier_offsets.swath_observations.matching_template', return_value=full):
                grid, chosen, masks = overlap_grid({}, frame)
                self.assertEqual(grid.shape, (2, 2))
                self.assertEqual(len(chosen), 2)
                self.assertTrue(all(m.all() for m in masks.values()))
                frame['sources'][1]['valid_window'][2] = 900
                with self.assertRaisesRegex(ValueError, f'IW{pair[0]}/IW{pair[1]}'):
                    overlap_grid({}, frame)

    def test_iw23_signs_common_motion_and_uncalibrated_swaths(self):
        owner = np.array([0, 1, 2, 3, 2, 3])
        pure = np.array([0, 1, 2, 3, 0, 3])
        eligible = np.array([True, True, True, True, True, False])
        for contrast in (-1, 0, 1):
            delta = contrast*.04
            total = np.full(6, 7.)
            total[2:4] += [-delta/2, delta/2]
            value, bias, available, sigma = correction_arrays(
                total, owner, pure, eligible, delta, abs(contrast)*.01, swaths=(2, 3))
            np.testing.assert_array_equal(available, [False, False, True, True, False, False])
            np.testing.assert_allclose(value[2:4], 7.)
            np.testing.assert_array_equal(value[~available], total[~available])
            np.testing.assert_allclose(sigma[available], abs(contrast)*.005)
            preview, _ = corrected_preview(total, symmetric_bias(delta, owner, swaths=(2, 3)), available)
            np.testing.assert_array_equal(preview, value)

    def test_iw23_cohorts_rows_closures_and_native_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            links = [{'pair': a+'_'+b, 'earlier': a, 'later': b}
                     for a, b in (('a', 'b'), ('b', 'c'), ('a', 'c'))]
            exp = SimpleNamespace(output=root, spec={'swaths': [2, 3]}, links=links,
                                  dates=['a', 'b', 'c'], checked_report=lambda *a, **k: None)
            for link, delta in zip(links, (.03, -.01, .02)):
                data = {'swaths': np.array([2, 3]), 'dual_valid': np.ones((1, 2), bool),
                        'dual_support': np.ones((1, 2), bool), 'burst_interior': np.ones((1, 2), bool),
                        'geometry_eligible': np.ones((1, 2), bool), 'source_ids': np.array(['x_iw2', 'y_iw3']),
                        'center_line': np.array([[10., 10.]]), 'center_sample': np.array([[20., 30.]])}
                for swath, owner, value in (('iw2', 1, 0.), ('iw3', 2, delta)):
                    data.update({f'{swath}_owner': np.full((1, 2), owner),
                                 f'{swath}_total': np.full((1, 2, 2), value),
                                 f'{swath}_peak': np.full((1, 2), .5), f'{swath}_snr': np.full((1, 2), 10.)})
                path = root/'measurements'/link['pair']; path.mkdir(parents=True)
                (path/'report.json').write_text('{"status":"complete"}')
                save_npz(path/'observations.npz', **data)
                self.assertTrue(cohorts(data)['primary'].all())
            rows, _ = observation_rows(exp, 'primary')
            np.testing.assert_allclose([r['mean'] for r in rows], [.03, -.01, .02])
            self.assertEqual(rows[0]['burst1'], 'x_iw2')
            self.assertEqual(rows[0]['burst2'], 'y_iw3')
            np.testing.assert_allclose(closure_report(exp, 'primary')['a_b_c']['azimuth_px'], 0, atol=1e-16)
            safe = root/'safe'; safe.mkdir()
            exp.cfg = {'inputs': {'polarization': 'HH'}}
            exp.inputs = SimpleNamespace(file=lambda *a: None)
            exp.acquisition = lambda _: {'specification': {'safe': {'path': str(safe)},
                'orbit': {'path': 'orbit', 'sha256': 'hash'}}}
            with patch('glacier_offsets.input_preparation.annotation_paths', return_value=[]), \
                 patch('glacier_offsets.inputs.inventory', return_value=[]) as inventory:
                load_native(exp, 'a')
                self.assertEqual(inventory.call_args.args[-1], [2, 3])
