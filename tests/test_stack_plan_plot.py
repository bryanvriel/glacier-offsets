"""Offline temporal-network analysis and plot export from stack-plan JSON."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.plot_stack_plan import analyze_plan, main, plot_bursts, plot_overview


def sample(days=(0, 6, 12, 18), links=((0, 1), (1, 2), (2, 3))):
    epoch = datetime(2020, 1, 1, 8, tzinfo=timezone.utc)
    starts = [epoch+timedelta(days=d) for d in days]
    inventory = [{'date':dt.date().isoformat(), 'start':dt.isoformat(), 'eligible':True,
                  'mission':'S1A' if i % 2 == 0 else 'S1B', 'available_burst_ids':['a', 'b']}
                 for i, dt in enumerate(starts)]
    pairs = {}
    for i, j in links:
        a, b = inventory[i]['date'], inventory[j]['date']
        seconds = (starts[j]-starts[i]).total_seconds()
        pairs[a+'_'+b] = {'earlier':a, 'later':b, 'active':True, 'shared_burst_ids':['a', 'b'],
                          'scene_interval_seconds':seconds,
                          'burst_intervals':{key:{'interval_seconds':seconds} for key in ('a', 'b')}}
    return {'inventory':inventory, 'pairs':pairs, 'reference_date':inventory[0]['date'],
            'selected_burst_ids':['a', 'b'], 'errors':[], 'omitted_links':[], 'disconnected_acquisitions':[]}


class PlanAnalysisTests(unittest.TestCase):
    def test_components_include_disjoint_chains_without_trusting_isolated_list(self):
        plan = sample((0, 6, 24, 30, 48), ((0, 1), (2, 3)))
        result = analyze_plan(plan)
        self.assertEqual([len(c) for c in result['components']], [2, 2, 1])
        self.assertEqual(result['isolated_dates'], ['2020-02-18'])
        self.assertEqual([g['days'] for g in result['large_gaps']], [18, 18])
        self.assertEqual(result['degree']['2020-01-01'], 1)

    def test_partial_burst_disconnects_geographic_network_while_plan_is_connected(self):
        plan = sample()
        plan['inventory'][1]['available_burst_ids'] = ['a']
        for pair in list(plan['pairs'].values())[:2]:
            pair['shared_burst_ids'] = ['a']; pair['burst_intervals'].pop('b')
        result = analyze_plan(plan)
        self.assertEqual(len(result['components']), 1)
        self.assertEqual(len(result['bursts']['a']['components']), 1)
        self.assertEqual(len(result['bursts']['b']['components']), 2)
        self.assertEqual(result['bursts']['b']['largest_availability_gap_days'], 12)
        self.assertEqual(result['bursts']['b']['reference_component'], ['2020-01-01'])

    def test_empty_shared_links_and_metadata_failures_are_not_observations(self):
        plan = sample()
        plan['inventory'][1].update(error='Invalid footprint', available_burst_ids=[])
        for pair in list(plan['pairs'].values())[:2]:pair['shared_burst_ids'] = []
        result = analyze_plan(plan)
        self.assertEqual(len(result['components']), 1)
        self.assertEqual(len(result['usable_shared_components']), 3)
        self.assertEqual(result['metadata_error_dates'], ['2020-01-07'])
        self.assertTrue(any('Invalid footprint' in warning for warning in result['warnings']))
        plan = sample()
        for pair in plan['pairs'].values():
            for interval in pair['burst_intervals'].values():interval['interval_seconds'] = None
        result = analyze_plan(plan)
        self.assertEqual(len(result['usable_shared_components']), 4)
        self.assertEqual(len(result['bursts']['a']['components']), 4)

    def test_threshold_preserves_exact_utc_times_and_inactive_pairs_are_ignored(self):
        plan = sample((0, 12), ((0, 1),))
        self.assertEqual(analyze_plan(plan)['large_gaps'], [])
        plan['inventory'][1]['start'] = '2020-01-13T10:00:01+02:00'
        pair = next(iter(plan['pairs'].values()))
        pair['scene_interval_seconds'] += 1
        plan['pairs']['inactive'] = {'active':False, 'earlier':'not-a-date'}
        result = analyze_plan(plan)
        self.assertEqual(len(result['large_gaps']), 1)
        self.assertAlmostEqual(result['large_gaps'][0]['days'], 12+1/86400)
        self.assertEqual(len(result['pairs']), 1)

    def test_unknown_coverage_is_not_silently_treated_as_missing(self):
        plan = sample((0,), ())
        plan['inventory'][0].pop('available_burst_ids')
        result = analyze_plan(plan)
        self.assertEqual(result['bursts']['a']['missing_dates'], [])
        self.assertEqual(result['bursts']['a']['unknown_dates'], ['2020-01-01'])
        self.assertTrue(result['warnings'])
        plan['inventory'][0]['missing_burst_ids'] = ['b']
        result = analyze_plan(plan)
        self.assertEqual(result['bursts']['a']['available_dates'], ['2020-01-01'])
        self.assertEqual(result['bursts']['b']['missing_dates'], ['2020-01-01'])

    def test_malformed_plan_and_conflicting_metadata_fail_clearly(self):
        invalid = [{}, {'inventory':[], 'pairs':{}}, sample()]
        invalid[-1]['inventory'].append(deepcopy(invalid[-1]['inventory'][0]))
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises(ValueError):analyze_plan(plan)
        for value in (0, -1, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'gap-days'):analyze_plan(sample(), value)
        plan = sample(); first = next(iter(plan['pairs']))
        plan['pairs'][first]['later'] = '2099-01-01'
        with self.assertRaisesRegex(ValueError, 'endpoints'):analyze_plan(plan)
        plan = sample(); plan['pairs'][first]['scene_interval_seconds'] = -1
        with self.assertRaisesRegex(ValueError, 'scene interval'):analyze_plan(plan)
        plan = sample(); plan['inventory'][0]['available_burst_ids'] = []
        with self.assertRaisesRegex(ValueError, 'unavailable burst'):analyze_plan(plan)
        plan = sample(); plan['pairs'][first]['burst_intervals'] = []
        with self.assertRaisesRegex(ValueError, 'burst intervals'):analyze_plan(plan)


class PlanPlotTests(unittest.TestCase):
    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_cli_exports_figures_and_summary_without_reading_source_data(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder); source = folder/'plan.json'
            plan = sample((0, 6, 24, 30, 48), ((0, 1), (2, 3)))
            for row in plan['inventory']:row['path'] = '/not-mounted/data/scene.SAFE'
            source.write_text(json.dumps(plan)); before = source.read_bytes()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(source), '--formats', 'png', 'pdf']), 0)
            output = folder/'plan_plots'
            for name in ('overview.png', 'burst_network.png'):
                self.assertEqual((output/name).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            for name in ('overview.pdf', 'burst_network.pdf'):
                self.assertTrue((output/name).read_bytes().startswith(b'%PDF'))
            report = json.loads((output/'summary.json').read_text())
            self.assertEqual(len(report['components']), 3)
            self.assertEqual(source.read_bytes(), before)

    def test_singleton_empty_burst_and_stdin(self):
        import matplotlib
        matplotlib.use('Agg')
        plan = sample((0,), ()); plan['inventory'][0]['available_burst_ids'] = []
        report = analyze_plan(plan)
        overview, bursts = plot_overview(report), plot_bursts(report, ['a'])
        overview.canvas.draw(); bursts.canvas.draw()
        self.assertEqual(len(report['components']), 1)
        self.assertEqual(report['bursts']['a']['components'], [])
        with tempfile.TemporaryDirectory() as folder, patch('sys.stdin', io.StringIO(json.dumps(plan))), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(main(['-', '-o', folder, '--burst', 'a']), 0)
            self.assertEqual(json.loads((Path(folder)/'summary.json').read_text())['source_plan'], 'stdin')


if __name__ == '__main__':unittest.main()
