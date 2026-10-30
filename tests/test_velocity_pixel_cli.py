"""Exercise the package entry point without the repository as the working directory."""
from contextlib import chdir, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from glacier_offsets.cli import main
from glacier_offsets.velocity_io import digest
from glacier_offsets.velocity_pixel_inversion import noise,stage_prepared_inputs
from glacier_offsets.velocity_pixel_runtime import module_fingerprints,verify_code


class PortablePixelTests(unittest.TestCase):
    def test_package_fit_and_reuse_from_different_directories(self):
        from test_velocity_pixel_config import RunnerTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();RunnerTests().prepared(root/'cache')
            regions=root/'regions.json';regions.write_text('{"regions": []}')
            config=root/'run.yaml'
            config.write_text(yaml.safe_dump({'input_directory':'cache','output':'result','regions':'regions.json'}))
            elsewhere=root/'unrelated';elsewhere.mkdir()
            with chdir(elsewhere):
                self.assertEqual(main(['velocity-pixels','fit',str(config)]),0)
            output=root/'result';before=digest(output/'source_coefficients.h5')
            recipe=json.loads((output/'recipe.json').read_text())
            self.assertTrue(all(key.startswith('glacier_offsets.') for key in recipe['code_sha256']))
            self.assertEqual(recipe['auxiliary_sha256']['regions.reviewed.json'],digest(regions))
            with chdir(root):
                self.assertEqual(main(['velocity-pixels','fit',str(config)]),0)
                verify_code(recipe['code_sha256'])
            self.assertEqual(digest(output/'source_coefficients.h5'),before)
            self.assertEqual((output/'regions.reviewed.json').read_bytes(),regions.read_bytes())

    def test_fingerprints_do_not_depend_on_cwd_and_reject_changes(self):
        before=module_fingerprints(('velocity_pixel',))
        with tempfile.TemporaryDirectory() as tmp,chdir(tmp):
            self.assertEqual(before,module_fingerprints(('velocity_pixel',)))
            verify_code(before)
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_code({'glacier_offsets.velocity_pixel':'bad hash'})
            with self.assertRaisesRegex(ValueError,'Historical'):
                verify_code({'tools/run_pixel_velocity_inversion.py':'bad hash'})

    def test_missing_explicit_auxiliary_fails_before_output_creation(self):
        from test_velocity_pixel_config import RunnerTests
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);RunnerTests().prepared(root/'cache')
            with self.assertRaisesRegex(ValueError,'Missing regions'):
                stage_prepared_inputs(root/'cache',root/'result',regions=root/'absent.json')
            self.assertFalse((root/'result').exists())

    def test_missing_noise_is_actionable_without_implicit_calibration(self):
        with tempfile.TemporaryDirectory() as tmp,chdir(tmp):
            with self.assertRaisesRegex(ValueError,'prepared noise.json'):
                noise(None,None,Path(tmp))

    def test_cli_help_lists_fit_export_verify_plot_and_report(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main(['velocity-pixels'])
        self.assertEqual(error.exception.code,2)


if __name__=='__main__':unittest.main()
