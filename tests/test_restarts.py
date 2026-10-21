import unittest
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
from glacier_offsets.ampcor import match
from glacier_offsets.products import identity, write_json

class RestartTests(unittest.TestCase):
    def test_match_identity_includes_gross_and_skips_exact_restart(self):
        from glacier_offsets.config import DEFAULT_MATCHING
        cfg={'matching':DEFAULT_MATCHING,'gross_offsets':{'mode':'zero'}}
        from glacier_offsets.ampcor import matching_specification
        spec=matching_specification(cfg,{'coreg_identity':'fixed'})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'radar').mkdir();(root/'radar/offsets.h5').touch()
            m={'coreg_identity':'fixed','match_identity':identity(spec),'coreg':{'slcs':{'reference':'ref','secondary':'sec'}},'stages':{}}
            with patch('glacier_offsets.ampcor.native_match') as native:
                match(cfg,root,m);native.assert_not_called()
            cfg['gross_offsets']={'mode':'constant','constant':{'azimuth_px':2,'range_px':-3}}
            m['geocode_identity']='old'
            with patch('glacier_offsets.ampcor.native_match',side_effect=RuntimeError('test interruption')):
                with self.assertRaisesRegex(RuntimeError,'interruption'):match(cfg,root,m)
            saved=json.loads((root/'manifest.json').read_text())
            self.assertNotIn('match_identity',saved);self.assertNotIn('geocode_identity',saved)
            self.assertEqual(saved['coreg_identity'],'fixed')
    def test_later_stages_reject_changed_configuration(self):
        from glacier_offsets.pipeline import execute
        from glacier_offsets.ampcor import matching_specification
        from glacier_offsets.geocode import geocoding_specification
        from glacier_offsets.config import DEFAULT_MATCHING
        from glacier_offsets.products import input_provenance
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);dummy=p/'input';dummy.write_text('fixture')
            inp={k:str(dummy) for k in ['reference_safe','secondary_safe','reference_orbit','secondary_orbit','dem']}
            inp.update(burst_id='burst',polarization='HH')
            cfg={'inputs':inp,'output':{'directory':str(p)},'matching':dict(DEFAULT_MATCHING),'gross_offsets':{'mode':'zero'},'geocoding':{'epsg':3031,'spacing_m':200}}
            m={'coreg_identity':'fixed','coreg_specification':{'inputs':input_provenance(cfg),'polarization':'HH'}}
            m['match_identity']=identity(matching_specification(cfg,m));m['geocode_identity']=identity(geocoding_specification(cfg,m))
            write_json(p/'burst/manifest.json',m)
            cfg['geocoding']['epsg']=3413
            with self.assertRaisesRegex(ValueError,'Geocoding configuration changed'):execute(cfg,'diagnostics')
            cfg['gross_offsets']={'mode':'constant','constant':{'azimuth_px':1,'range_px':-2}}
            with self.assertRaisesRegex(ValueError,'Matching configuration changed'):execute(cfg,'geocode')
