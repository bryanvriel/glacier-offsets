from contextlib import chdir
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from glacier_offsets.paths import absolute_path


class LogicalPathTests(unittest.TestCase):
    def test_aliases_dot_components_missing_outputs_and_strict_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            drive=root/'drive';drive.mkdir()
            alias=root/'data';alias.symlink_to(drive,target_is_directory=True)
            (drive/'input').touch()
            self.assertEqual(absolute_path(alias/'input',strict=True),alias/'input')
            self.assertEqual(absolute_path('data/./input',root,strict=True),alias/'input')
            self.assertEqual(absolute_path('configs/../data/new/output',root),alias/'new/output')
            with self.assertRaises(FileNotFoundError):absolute_path(alias/'missing',strict=True)
            self.assertFalse((drive/'new').exists())

    def test_logical_shell_cwd_is_used_only_when_it_names_current_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            drive=root/'drive';drive.mkdir()
            alias=root/'project';alias.symlink_to(drive,target_is_directory=True)
            with chdir(alias),patch.dict('os.environ',{'PWD':str(alias)}):
                self.assertEqual(absolute_path('input'),alias/'input')
            with chdir(drive),patch.dict('os.environ',{'PWD':str(root)}):
                self.assertEqual(absolute_path('input'),drive/'input')
