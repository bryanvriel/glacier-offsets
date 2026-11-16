"""Numerical and integrity checks for GOFF preparation optimizations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from glacier_offsets.nisar_velocity import purged_predictions, _purged_predictions_reference


def field(ns=12, nt=40):
    rng = np.random.default_rng(551)
    early = np.broadcast_to(8767.+np.arange(nt)*6+.43, (ns, nt)).copy()
    early += rng.uniform(-.01, .01, early.shape)
    late = early+12.000003
    geom = rng.normal(size=(ns, nt, 2, 3))
    geom[..., 0, 0] += 2.; geom[..., 1, 1] += 2.
    data = np.einsum('stci,i->stc', geom[..., :2], [.4, -.1])*(late-early)[..., None]
    data += rng.normal(scale=.05, size=data.shape)
    data[::3, ::7] += 2.
    return dict(early=early, late=late, data=data, valid=rng.uniform(size=early.shape) > .2,
                peak=np.ones(early.shape), year_index=np.arange(nt) % 2,
                reference_acquisition=np.array([f'A{i}' for i in range(nt)]),
                secondary_acquisition=np.array([f'A{i+2}' for i in range(nt)]), geometry_by_pair=geom)


class NisarPerformanceTests(unittest.TestCase):
    def test_batched_purging_matches_scalar_irls(self):
        f = field()
        f['valid'][0] = False
        f['early'][1, 3] = np.nan
        f['geometry_by_pair'][1, 3] = np.nan
        for allowed in (None, np.arange(40) % 3 != 0):
            assert_allclose(purged_predictions(f, allowed), _purged_predictions_reference(f, allowed), atol=2e-10, rtol=2e-10)
        take = f['valid'] & (f['peak'] >= .08)
        assert_allclose(purged_predictions(f, valid_only=True)[take], _purged_predictions_reference(f)[take], atol=2e-10, rtol=2e-10)

    def test_ill_conditioned_and_rank_deficient_fallback(self):
        f = field(ns=4)
        f['geometry_by_pair'][0, ..., 1] *= 1e-6
        f['geometry_by_pair'][1, ..., 1] = 0.
        f['geometry_by_pair'][2, ..., 1] = f['geometry_by_pair'][2, ..., 0]
        assert_allclose(purged_predictions(f), _purged_predictions_reference(f), atol=2e-10, rtol=2e-10)

    def test_acquisition_and_calendar_purge_has_no_target_leakage(self):
        f = field()
        before = purged_predictions(f)
        target = 10
        ref, sec = f['reference_acquisition'], f['secondary_acquisition']
        shared = [ref[target], sec[target]]
        mask = np.isin(ref, shared) | np.isin(sec, shared)
        mask |= (np.floor(f['early'][0]) == np.floor(f['early'][0, target]))
        f['data'][:, mask] += 10000
        assert_allclose(purged_predictions(f)[:, target], before[:, target], atol=1e-10, rtol=1e-10)

    def test_validation_reuse_and_changed_file(self):
        from glacier_offsets.nisar_validation import verified_nisar
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'package.nc'; path.write_bytes(b'one')
            with patch('glacier_offsets.nisar_validation.validate_nisar', return_value={'valid': True}) as validate:
                sha = verified_nisar(path, root/'cache')
                self.assertEqual(verified_nisar(path, root/'cache'), sha)
                self.assertEqual(validate.call_count, 1)
                from glacier_offsets.velocity_io import revision, digest
                expected = {'sha256': sha, 'revision': list(revision(path))}
                with patch('glacier_offsets.nisar_validation.digest', wraps=digest) as hashed:
                    self.assertEqual(verified_nisar(path, root/'cache', expected), sha)
                    self.assertFalse(any(Path(call.args[0]) == path for call in hashed.call_args_list))
                path.write_bytes(b'two')
                with self.assertRaisesRegex(ValueError, 'changed since collection'):
                    verified_nisar(path, root/'cache', expected)
                self.assertNotEqual(verified_nisar(path, root/'cache'), sha)
                self.assertEqual(validate.call_count, 2)
                path.write_bytes(b'bad')
                validate.side_effect = ValueError('invalid source')
                with self.assertRaisesRegex(ValueError, 'invalid source'): verified_nisar(path, root/'cache')

    def test_compaction_resume_and_map_identity(self):
        from test_nisar import synthetic, corrected
        from glacier_offsets.packaging import content_hashes
        from glacier_offsets.velocity_collection import collect
        from glacier_offsets.velocity_prepare import prepare_observations
        from glacier_offsets.velocity_solid_earth import Interrupted
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = corrected(synthetic(root/'raw.nc', n=8))
            with h5netcdf.File(path, 'a') as h:
                h['horizontal_valid_mask'][:, 0, 1] = 0
                h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
            manifest = collect([path], root/'cache')
            def stop(stage, progress):
                if progress == 2: raise Interrupted('checkpoint')
            with self.assertRaises(Interrupted): prepare_observations(root/'cache', manifest, chunk=2, hook=stop)
            obs, _ = prepare_observations(root/'cache', manifest, chunk=2)
            with h5py.File(obs.path('observations.h5')) as h:
                assert_array_equal(h['source_id'][:], [0, 2, 3])
                assert_array_equal(h['map_source_index'][:], [[0, -1], [1, 2]])
                self.assertEqual(h['data'].shape, (3, 8, 2))
                self.assertEqual(h.attrs['completed_pairs'], 8)
            self.assertTrue(prepare_observations(root/'cache', manifest)[0].id == obs.id)

    def test_shifted_packages_keep_pair_geometry_and_overlap_ownership(self):
        from test_nisar import synthetic, corrected
        from glacier_offsets.packaging import content_hashes
        from glacier_offsets.velocity_collection import collect
        from glacier_offsets.velocity_prepare import prepare_observations
        from glacier_offsets.nisar_collection import NisarCollection
        from glacier_offsets.products import identity
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [corrected(synthetic(root/f'raw{i}.nc', n=4)) for i in range(2)]
            with h5netcdf.File(paths[1], 'a') as h:
                h['x'][:] += 80
                h['frame_number'][:] = 145
                for k in ('pair_id', 'names'): h[k][:] = [s+'_frame145' for s in h[k][:].astype(str)]
                spec = json.loads(h['provenance_json'][()].decode())
                spec['grid']['transform'][0] += 80
                h['provenance_json'][()] = json.dumps(spec)
                h.attrs['package_identity'] = identity(spec)
                h.attrs['content_hashes_json'] = json.dumps(content_hashes(h), sort_keys=True)
            manifest = collect(paths, root/'cache')
            obs, _ = prepare_observations(root/'cache', manifest, chunk=3)
            with h5py.File(obs.path('observations.h5')) as h, NisarCollection(paths) as c:
                expected = c.tile((slice(0, 2), slice(0, 3)))
                assert_array_equal(h['source_id'][:], np.arange(6))
                for dest, source in [('early', 'early'), ('late', 'late'), ('data', 'data_ra'), ('geometry_by_pair', 'geometry'), ('valid_including_mixed', 'valid')]:
                    a = expected[source].reshape(8, 6, *expected[source].shape[3:]).swapaxes(0, 1)
                    assert_array_equal(h[dest][:], a)


if __name__ == '__main__': unittest.main()
