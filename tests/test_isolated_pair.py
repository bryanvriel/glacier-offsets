"""Isolated pairs keep template geometry and re-key only dates and burst intervals."""
from pathlib import Path
import unittest

from tools.prepare_isolated_pair import derive_pair, saved_merged_checksum


def template():
    sources = [{'burst_id': f't038_080975_iw{s}', 'offset': [0, 100*s], 'valid_window': [0, 10, 0, 10],
                'acquisition_times': ['2017-01-01T08:00:00', '2017-01-07T08:00:00'], 'interval_seconds': 518400.0}
               for s in (1, 2)]
    return {'coreg_identity': 'old', 'coreg': {'slcs': {}, 'assembly_sources': sources, 'reference_grid': {'length': 8}},
            'coreg_specification': {'inputs': {}, 'polarization': 'HH'},
            'assembly': {'identity': 'old', 'sources': sources, 'ownership': {'method': 'fixed'}},
            'stack_pair': {'acquisition_dates': ['2017-01-01', '2017-01-07'], 'geometry_identity': 'g'},
            'acquisition_times': ['a', 'b'], 'matching': {'matching': {'deramp': 0}}, 'match_identity': 'm',
            'geocode_identity': 'x', 'geocoding': {}, 'slc_previews': {}, 'stages': {'match': 'complete'}}


def date(day, times, slc):
    return {'burst_times': times, 'slc': slc, 'specification': {'day': day}, 'assembly_identity': f'a-{day}'}


class IsolatedPairTests(unittest.TestCase):
    def test_twelve_day_pair_uses_exact_burst_times_and_keeps_geometry(self):
        t = template()
        early = date('2017-01-07', {'t038_080975_iw1': '2017-01-07T08:01:57', 't038_080975_iw2': '2017-01-07T08:01:58'}, '/s/a.tif')
        late = date('2017-01-19', {'t038_080975_iw1': '2017-01-19T08:01:56', 't038_080975_iw2': '2017-01-19T08:01:58'}, '/s/b.tif')
        m = derive_pair(t, ('2017-01-07', '2017-01-19'), [early, late], ['s0', 's1'], Path('/shim'))
        self.assertEqual([e['interval_seconds'] for e in m['assembly']['sources']], [12*86400-1.0, 12*86400.0])
        self.assertEqual(m['coreg']['assembly_sources'], m['assembly']['sources'])
        self.assertEqual(m['coreg']['slcs'], {'reference': '/s/a.tif', 'secondary': '/s/b.tif'})
        self.assertEqual(m['stack_pair']['acquisition_dates'], ['2017-01-07', '2017-01-19'])
        self.assertEqual(m['stack_pair']['geometry_identity'], 'g')
        self.assertEqual(m['assembly']['ownership'], {'method': 'fixed'})
        self.assertNotEqual(m['coreg_identity'], 'old')
        for key in ('match_identity', 'geocode_identity', 'matching', 'geocoding', 'slc_previews'):
            self.assertNotIn(key, m)
        self.assertEqual(t['assembly']['sources'][0]['interval_seconds'], 518400.0)

    def test_missing_burst_time_is_invalid_and_reversed_time_is_rejected(self):
        early = date('2017-01-07', {'t038_080975_iw1': '2017-01-07T08:01:57'}, '/s/a.tif')
        late = date('2017-01-19', {'t038_080975_iw1': '2017-01-19T08:01:57', 't038_080975_iw2': '2017-01-19T08:01:58'}, '/s/b.tif')
        m = derive_pair(template(), ('2017-01-07', '2017-01-19'), [early, late], ['s0', 's1'], Path('/shim'))
        self.assertIsNone(m['assembly']['sources'][1]['interval_seconds'])
        with self.assertRaises(ValueError):
            derive_pair(template(), ('2017-01-19', '2017-01-07'), [late, early], ['s0', 's1'], Path('/shim'))

    def test_released_merged_slc_checksum_is_read_from_cleanup_journal(self):
        slc = str(Path('/tmp/slc.tif').resolve())
        self.assertEqual(saved_merged_checksum({'slc': slc, 'products': {slc: 'p'}}), 'p')
        self.assertEqual(saved_merged_checksum({'slc': slc, 'slc_cleanup': {'files': {slc: 'c'}}}), 'c')
        self.assertIsNone(saved_merged_checksum({'slc': slc}))


if __name__ == '__main__':
    unittest.main()
