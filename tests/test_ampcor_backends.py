import unittest
from unittest.mock import patch
from glacier_offsets.environment import select_backend

class BackendTests(unittest.TestCase):
    def test_cpu_and_auto_instantiation(self):
        name,cls=select_backend('cpu');self.assertEqual(name,'cpu');cls()
        name,cls=select_backend('auto');self.assertIn(name,['cpu','gpu']);cls()
    def test_missing_cuda_auto_fallback_and_explicit_error(self):
        with patch('isce3.cuda',new=None,create=True):
            self.assertEqual(select_backend('auto')[0],'cpu')
            with self.assertRaisesRegex(RuntimeError,'CUDA Ampcor unavailable'):select_backend('gpu')
    def test_gpu_numerical_consistency(self):
        try: select_backend('gpu')
        except RuntimeError as exc:self.skipTest(str(exc))
        from test_synthetic_offsets import SyntheticTests
        import numpy as np
        test=SyntheticTests()
        cpu=test.measure((2.5,-1.25),backend='cpu')
        gpu=test.measure((2.5,-1.25),backend='gpu')
        mask=cpu['valid']&gpu['valid']
        self.assertTrue(mask.any())
        np.testing.assert_allclose(cpu['total'][mask],gpu['total'][mask],atol=0.25)

    def test_cuda_build_without_usable_device_falls_back(self):
        import isce3
        from types import SimpleNamespace
        cuda=SimpleNamespace(matchtemplate=SimpleNamespace(PyCuAmpcor=isce3.matchtemplate.PyCPUAmpcor))
        with patch('isce3.cuda',new=cuda,create=True),patch('isce3.core.gpu_check.cuda_device_valid',side_effect=ValueError('no device')):
            self.assertEqual(select_backend('auto')[0],'cpu')
            with self.assertRaisesRegex(RuntimeError,'no device'):select_backend('gpu')
