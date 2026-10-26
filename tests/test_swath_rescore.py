"""Link targets, matched comparisons, fold isolation and frozen-file preservation."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np

from glacier_offsets.products import checksum, identity
from glacier_offsets.swath_models import evaluate
from tools.swath_experiment_io import Experiment, Inputs, COHORTS
from tools.swath_experiment_rescore import (select_rows, summarize_links, link_instances,
    score_instances, evaluate_variant, original_matched_window_scores, rescore, render_report)


def fixture():
    dates = ['a', 'b', 'c', 'd']
    metadata = {day: {'mission': 'S1B' if i % 2 else 'S1A',
                      'ipf': '2.72' if i < 2 else '2.82'} for i, day in enumerate(dates)}
    rows = []
    for a, b in (('a', 'b'), ('b', 'c'), ('c', 'd'), ('a', 'c'), ('b', 'd')):
        motion = .026*(int(metadata[b]['mission'] == 'S1B')-int(metadata[a]['mission'] == 'S1B'))
        for group in ('x', 'y'):
            values = np.array([-.1, .1, -.05, .05])+motion
            rows.append({'pair': a+'_'+b, 'earlier': a, 'later': b,
                'burst1': group+'_iw1', 'burst2': group+'_iw2', 'mean': float(values.mean()),
                'values': values.tolist(), 'physical': {c: [0.]*len(values) for c in ('bistatic', 'fm_rate', 'combined')}})
    return rows, metadata


class LinkScoringTests(unittest.TestCase):
    def test_loss_is_after_averaging_and_burst_groups_have_equal_weight(self):
        rows, _ = fixture()
        rows = rows[:2]
        rows[0]['values'] = [-1., 1.]
        rows[1]['values'] = [.2]*100
        instances, _ = link_instances(rows, [0, 1], [0., 0.], np.zeros((5, 2)), np.ones((5, 2)))
        self.assertAlmostEqual(instances[0]['observed_mean_px'], .1)
        scores = score_instances(instances)
        self.assertAlmostEqual(scores['baseline_mae_px'], .1)
        self.assertEqual(scores['links'], 1)
        self.assertEqual(scores['unique_test_windows'], 102)

    def test_support_filter_counts_whole_links_and_does_not_select_by_value(self):
        rows, _ = fixture()
        rows[0]['values'] = [999.]*1
        rows[1]['values'] = [-999.]*2
        indexes, selected = select_rows(rows, 4)
        self.assertEqual(indexes, list(range(2, 10)))
        self.assertFalse(any(r['pair'] == 'a_b' for r in selected))
        changed = deepcopy(rows)
        for row in changed:
            row['values'] = [v*1000 for v in row['values']]
        self.assertEqual(select_rows(changed, 4)[0], indexes)

    def test_same_link_gets_equal_total_weight_across_repeated_folds(self):
        rows, _ = fixture()
        a, _ = link_instances(rows[:2], [0, 1], [0., 0.], np.zeros((5, 2)), np.ones((5, 2)))
        instance = a[0]
        first = dict(instance, observed_mean_px=1., residual_px=1.)
        second = dict(instance, pair='c_d', observed_mean_px=3., residual_px=3.)
        self.assertAlmostEqual(score_instances([first, first, second])['baseline_mae_px'], 2.)

    def test_small_bias_is_visible_in_link_mean_scores(self):
        rows, metadata = fixture()
        result = evaluate_variant(rows, metadata, draws=12)
        self.assertAlmostEqual(result['coefficient']['beta_px'], .026)
        for kind in ('leave_acquisition_out', 'leave_link_out', 'leave_burst_group_out'):
            scores = result['validation'][kind]['spacecraft']
            self.assertLess(scores['model_rmse_px'], 1.e-12)
            self.assertLess(scores['mae_change_px'], -.01)
        self.assertEqual(result['validation']['leave_acquisition_out']['per_acquisition']['links'], 0)

    def test_fold_predictions_do_not_use_held_out_offsets_or_shared_bursts(self):
        rows, metadata = fixture()
        first = evaluate_variant(rows, metadata, draws=10)
        changed = deepcopy(rows)
        for row in changed:
            if row['pair'] == 'a_b':
                row['values'] = (np.array(row['values'])+10).tolist()
                row['mean'] += 10
        second = evaluate_variant(changed, metadata, draws=10)
        def held_prediction(result):
            fold = next(f for f in result['folds'] if f['kind'] == 'leave_link_out' and f['label'] == 'a_b')
            return fold['models']['spacecraft']['instances'][0]['prediction_mean_px']
        self.assertEqual(held_prediction(first), held_prediction(second))
        for fold in first['folds']:
            self.assertFalse(set(fold['train_rows']) & set(fold['test_rows']))
            if fold['kind'] == 'leave_burst_group_out':
                held = set(fold['label'].split('|'))
                for i in fold['train_rows']:
                    self.assertFalse(held & {rows[i]['burst1'], rows[i]['burst2']})

    def test_window_comparisons_always_match_baseline_to_model_availability(self):
        rows, _ = fixture()
        rows[0]['values'] = [.5, .5]
        rows[1]['values'] = [10., 10.]
        rows[0]['physical']['combined'] = [0., 0.]
        rows[1]['physical']['combined'] = [0., 0.]
        record = {'rows': rows, 'folds': [{'kind': 'leave_burst_group_out', 'test_rows': [0, 1],
            'models': {'spacecraft': {'prediction_px': [0., 0.]},
                       'per_acquisition': {'prediction_px': [.2, np.nan]}}}]}
        result = original_matched_window_scores(record)['leave_burst_group_out']['per_acquisition']
        self.assertEqual(result['common_n'], 2)
        self.assertEqual(result['models']['baseline']['n'], result['models']['model']['n'])
        self.assertEqual(result['models']['baseline']['mae_px'], .5)
        self.assertAlmostEqual(result['models']['model']['mae_px'], .3)

    def test_repeatability_and_explicit_empty_populations(self):
        rows, metadata = fixture()
        a = evaluate_variant(rows, metadata, draws=10)
        b = evaluate_variant(rows, metadata, draws=10)
        self.assertEqual(a, b)
        empty = evaluate_variant(rows, metadata, minimum=50, draws=10)
        self.assertIsNone(empty['coefficient']['beta_px'])
        self.assertEqual(empty['validation']['leave_link_out']['spacecraft']['links'], 0)


class FrozenArtifactTests(unittest.TestCase):
    def test_rescore_cli_does_not_rewrite_the_frozen_plan(self):
        from tools.swath_correction_experiment import main
        with tempfile.TemporaryDirectory() as tmp:
            experiment = SimpleNamespace(output=Path(tmp), lock=lambda: nullcontext(),
                inputs=SimpleNamespace(unchanged=lambda: None), plan=Mock())
            with patch('tools.swath_experiment_io.Experiment', return_value=experiment), \
                 patch('tools.swath_experiment_rescore.rescore', return_value={'status': 'complete'}) as run:
                self.assertEqual(main(['rescore', 'config.yaml', '--output', tmp]), 0)
            experiment.plan.assert_not_called()
            run.assert_called_once_with(experiment)

    def test_default_run_does_not_silently_enable_posthoc_selection(self):
        from tools.swath_correction_experiment import main
        with tempfile.TemporaryDirectory() as tmp:
            experiment = SimpleNamespace(output=Path(tmp), lock=lambda: nullcontext(),
                inputs=SimpleNamespace(unchanged=lambda: None), plan=lambda: {'status': 'complete'})
            done = {'status': 'complete'}
            with patch('tools.swath_experiment_io.Experiment', return_value=experiment), \
                 patch('glacier_offsets.swath_physics.audit', return_value=done), \
                 patch('glacier_offsets.swath_physics.physics', return_value=done), \
                 patch('glacier_offsets.swath_observations.measure', return_value=done), \
                 patch('tools.swath_experiment_analysis.fit', return_value=done), \
                 patch('tools.swath_experiment_analysis.report', return_value=done), \
                 patch('tools.swath_experiment_previews.previews', return_value=done), \
                 patch('tools.swath_experiment_rescore.rescore', side_effect=AssertionError('Post hoc analysis must be explicit')):
                self.assertEqual(main(['run', 'config.yaml', '--output', tmp]), 0)

    def test_rescore_preserves_frozen_files_and_detects_corrupt_analysis(self):
        rows, metadata = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root/'analysis').mkdir()
            experiment = object.__new__(Experiment)
            experiment.key, experiment.output, experiment.inputs = 'frozen-key', root, Inputs()
            (root/'plan.json').write_text(json.dumps({'identity': experiment.key}))
            (root/'summary.json').write_text('{"status":"complete"}')
            (root/'REPORT.md').write_text('Original report stays unchanged.\n')
            (root/'analysis/comparison.png').write_bytes(b'original figure')
            record = evaluate(rows, sorted(metadata), {d: m['mission'] for d, m in metadata.items()}, draws=8)
            record['rows'] = rows
            products = {}
            for cohort in COHORTS:
                path = root/'analysis'/f'{cohort}.json'
                path.write_text(json.dumps(record))
                products[str(path)] = checksum(path)
            spec = {'experiment': experiment.key}
            (root/'fit.json').write_text(json.dumps({'status': 'complete', 'specification': spec,
                'identity': identity(spec), 'metadata': metadata, 'products': products}))
            originals = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            with patch('tools.swath_experiment_rescore.plot_results', return_value=[]):
                result = rescore(experiment, minimum_support=(0,), draws=8)
            self.assertTrue(result['original_results_unchanged'])
            for path, expected in originals.items():
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
            out = root/'rescoring/posthoc-link-means-v1'
            native = json.loads((out/'native.json').read_text())
            self.assertEqual(native['role'], 'diagnostic_only')
            self.assertEqual(native['variants'], {})
            self.assertIn('post hoc', (out/'REPORT.md').read_text())
            experiment.inputs = Inputs()
            (root/'analysis/primary.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                rescore(experiment, minimum_support=(0,), draws=8)


if __name__ == '__main__':
    unittest.main()
