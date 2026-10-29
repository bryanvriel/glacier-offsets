"""Installed estimator orchestration and standalone/legacy calibration export."""
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml

from glacier_offsets.config import load_defaults
from glacier_offsets.products import checksum, identity, write_json
from glacier_offsets.swath_models import evaluate
from glacier_offsets.swath_calibration import main
from glacier_offsets.swath_calibration_export import export_calibration
from glacier_offsets.swath_correction_config import load_calibration
from glacier_offsets.swath_experiment_io import METHOD, Inputs, reference_metadata, result_lock
from glacier_offsets.stack_lock import stack_lock


def result_fixture(base):
    root = base/'results'
    root.mkdir()
    stack = base/'stack'
    cfg = load_defaults()
    cfg['inputs']['polarization'] = 'HH'
    cfg['matching'].update(engine='isce3', backend='cpu', deramp=0, range_oversampling=2)
    cfg['output'] = {'directory': str(stack)}
    cfg['stack'] = {'processing': 'isce3_merge_first', 'swath_merge': 'midpoint',
                    'swath_correction': {'enabled': False}}
    dates = ['2018-01-02', '2018-01-08', '2018-01-14', '2018-01-20']
    metadata = {day: {'mission': 'S1A' if i%2 else 'S1B', 'ipf': '002.84'} for i,day in enumerate(dates)}
    refspec = {'anchor': {'reference_date': '2017-01-01', 'selection': {'polarization': 'HH'}}}
    frame = {'status': 'complete', 'specification': refspec, 'geometry_identity': identity(refspec),
             'layout_identity': 'layout', 'coreg': {'reference_grid': {'prf': 500.}}}
    catalog = {'inventory': [{'date': day, 'eligible': True, 'direction': 'DESCENDING',
                             'relative_orbit': 38, 'polarizations': ['HH']} for day in dates]}
    write_json(stack/'geometry/manifest.json', frame)
    write_json(stack/'stack_manifest.json', catalog)
    spec = {'method': METHOD, 'config': cfg, 'dates': dates, 'source_hashes': {},
            'geometry_identity': frame['geometry_identity'], 'layout_identity': 'layout'}
    key = identity({k:v for k,v in spec.items() if k!='source_hashes'})
    plan = {'status': 'complete', 'identity': key, 'specification': spec,
            'reference_metadata': reference_metadata(frame, catalog), 'primary_selection': 'primary fixed cohort'}
    write_json(root/'plan.json', plan)
    rows = []
    for a,b in zip(dates[:-1],dates[1:]):
        c = int(metadata[b]['mission']=='S1B')-int(metadata[a]['mission']=='S1B')
        for group in ('x','y'):
            values = [c*.03-.01,c*.03+.01]
            rows.append({'pair':a+'_'+b,'earlier':a,'later':b,'burst1':group+'_iw1',
                         'burst2':group+'_iw2','mean':c*.03,'values':values})
    analysis = evaluate(rows, dates, {day:v['mission'] for day,v in metadata.items()}, draws=12)
    analysis['rows'] = rows
    write_json(root/'analysis/primary.json', analysis)
    write_json(root/'physics_status.json', {'status': 'complete'})
    write_json(root/'measure_stage.json', {'status': 'complete', 'pilot_only': False})
    fit_spec = {'experiment': key, 'physics_sha256': checksum(root/'physics_status.json')}
    fit = {'status': 'complete', 'specification': fit_spec, 'identity': identity(fit_spec),
           'metadata': metadata, 'products': {str(root/'analysis/primary.json'): checksum(root/'analysis/primary.json')}}
    write_json(root/'fit.json', fit)
    config = base/'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return root, cfg, config


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root, self.cfg, self.config = result_fixture(self.base)

    def read(self, name):
        return json.loads((self.root/name).read_text())

    def replace_analysis(self, analysis):
        write_json(self.root/'analysis/primary.json', analysis)
        fit = self.read('fit.json')
        fit['products'] = {str(self.root/'analysis/primary.json'): checksum(self.root/'analysis/primary.json')}
        write_json(self.root/'fit.json', fit)

    def test_export_relocated_results_offline_and_roundtrip_through_correction_loader(self):
        old = {p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        moved = self.base/'moved'
        self.root.rename(moved)
        self.root = moved
        shutil.rmtree(self.base/'stack')
        result = export_calibration(moved)
        cal = self.read('calibration.json')
        self.assertAlmostEqual(cal['beta_px'], .03)
        self.assertEqual(cal['scope']['start_date'], '2018-01-02')
        self.assertEqual(cal['scope']['end_date'], '2018-01-20')
        self.assertEqual(cal['provenance']['geometry_reference_date'], '2017-01-01')
        self.assertEqual(cal['reference_prf_hz'], 500.)
        cfg = deepcopy(self.cfg)
        cfg['stack']['swath_correction'] = result['settings']['stack']['swath_correction']
        self.assertEqual(load_calibration(cfg)[0], cal)
        self.assertEqual(result['calibration_sha256'], checksum(moved/'calibration.json'))
        self.assertEqual((moved/'calibration.json.sha256').read_text().split()[0], result['calibration_sha256'])
        self.assertEqual(yaml.safe_load((moved/'calibration-settings.yaml').read_text()), result['settings'])
        for path, contents in old.items():
            self.assertEqual((moved/path).read_bytes(), contents)
        before = (moved/'calibration.json').stat().st_mtime_ns
        self.assertEqual(export_calibration(moved), result)
        self.assertEqual((moved/'calibration.json').stat().st_mtime_ns, before)

    def test_legacy_results_export_with_config_but_without_raw_inputs(self):
        plan = self.read('plan.json')
        plan.pop('reference_metadata')
        write_json(self.root/'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'Legacy results need CONFIG'):
            export_calibration(self.root)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main([str(self.config), '--output', str(self.root), '--stage', 'export']), 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'complete')

    def test_inconclusive_validation_is_exported_honestly(self):
        analysis = self.read('analysis/primary.json')
        for record in analysis['validation_score_intervals'].values():
            record['spacecraft'].update(mae_change_vs_baseline_px=-.01, descriptive_interval95_px=[-.02,.01])
        self.replace_analysis(analysis)
        result = export_calibration(self.root)
        self.assertEqual(result['validation_status'], 'inconclusive')
        self.assertEqual(self.read('calibration.json')['provenance']['validation_status'], 'inconclusive')

    def test_corruption_unidentifiable_model_and_pilot_never_publish(self):
        primary = self.root/'analysis/primary.json'
        contents = primary.read_bytes()
        primary.write_bytes(contents+b' ')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            export_calibration(self.root)
        primary.write_bytes(contents)
        analysis = self.read('analysis/primary.json')
        analysis['models']['spacecraft']['rank'] = 0
        self.replace_analysis(analysis)
        with self.assertRaisesRegex(ValueError, 'not identifiable'):
            export_calibration(self.root)
        analysis['models']['spacecraft']['rank'] = 1
        self.replace_analysis(analysis)
        write_json(self.root/'measure_stage.json', {'status':'complete','pilot_only':True})
        with self.assertRaisesRegex(ValueError, 'all planned measurements'):
            export_calibration(self.root)
        self.assertFalse((self.root/'calibration.json').exists())

    def test_scope_mismatch_and_conflicting_output_preserve_existing_files(self):
        bad = deepcopy(self.cfg)
        bad['matching']['zoom'] += 1
        with self.assertRaisesRegex(ValueError, 'matching treatment differs'):
            export_calibration(self.root, cfg=bad)
        result = export_calibration(self.root)
        target = Path(result['calibration'])
        target.write_text('{"existing":"preserve"}')
        with self.assertRaisesRegex(ValueError, 'different contents'):
            export_calibration(self.root)
        self.assertEqual(target.read_text(), '{"existing":"preserve"}')
        export_calibration(self.root, force=True)
        self.assertAlmostEqual(self.read('calibration.json')['beta_px'], .03)
        with self.assertRaisesRegex(ValueError, 'retained result products'):
            export_calibration(self.root, self.root/'analysis/candidate.json', force=True)
        with self.assertRaisesRegex(ValueError, 'production stack'):
            export_calibration(self.root, self.base/'stack/calibration.json', cfg=self.cfg)

    def test_interrupted_publication_leaves_original_json_unchanged(self):
        export_calibration(self.root)
        target = self.root/'calibration.json'
        target.write_text('previous calibration')
        with patch('glacier_offsets.swath_calibration_export.os.replace', side_effect=OSError('interrupted')):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                export_calibration(self.root, force=True)
        self.assertEqual(target.read_text(), 'previous calibration')
        self.assertFalse(list(self.root.glob('.calibration.json*')))

    def test_active_result_and_stack_writers_are_rejected(self):
        args = [str(self.config), '--output', str(self.root), '--stage', 'export']
        with result_lock(self.root):
            self.assertEqual(main(args), 2)
        with stack_lock(self.base/'stack'):
            self.assertEqual(main(args), 2)
        self.assertFalse((self.root/'calibration.json').exists())


class CommandTests(unittest.TestCase):
    def test_full_run_exports_and_never_silently_enables_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = SimpleNamespace(output=Path(tmp), cfg={}, lock=lambda:nullcontext(),
                                  inputs=Inputs(), plan=lambda:{'status':'complete'})
            done = {'status':'complete'}
            with patch('glacier_offsets.swath_experiment_io.Experiment', return_value=exp), \
                 patch('glacier_offsets.swath_physics.audit', return_value=done), \
                 patch('glacier_offsets.swath_observations.measure', return_value=done), \
                 patch('glacier_offsets.swath_physics.physics', return_value=done), \
                 patch('glacier_offsets.swath_experiment_analysis.fit', return_value=done), \
                 patch('glacier_offsets.swath_experiment_previews.previews', return_value=done), \
                 patch('glacier_offsets.swath_experiment_analysis.report', return_value=done), \
                 patch('glacier_offsets.swath_experiment_rescore.rescore', side_effect=AssertionError('Post hoc rescore')), \
                 patch('glacier_offsets.swath_calibration_export.export_calibration', return_value=done) as export:
                self.assertEqual(main(['config.yaml','--output',tmp]),0)
                export.assert_called_once()

    def test_pilot_does_not_export_and_incomplete_measurement_stops_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = SimpleNamespace(output=Path(tmp), cfg={}, lock=lambda:nullcontext(),
                                  inputs=Inputs(), plan=lambda:{'status':'complete'})
            with patch('glacier_offsets.swath_experiment_io.Experiment', return_value=exp), \
                 patch('glacier_offsets.swath_physics.audit', return_value={'status':'complete'}), \
                 patch('glacier_offsets.swath_observations.measure', return_value={'status':'incomplete'}), \
                 patch('glacier_offsets.swath_physics.physics', side_effect=AssertionError('Must stop')), \
                 patch('glacier_offsets.swath_calibration_export.export_calibration', side_effect=AssertionError('Must not export')):
                self.assertEqual(main(['config.yaml','--output',tmp]),1)
                self.assertEqual(main(['config.yaml','--output',tmp,'--pilot-only']),1)


if __name__ == '__main__':
    unittest.main()
