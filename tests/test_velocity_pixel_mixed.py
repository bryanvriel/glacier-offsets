"""Mixed-window inclusion preserves every non-swath validity exclusion."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from numpy.testing import assert_array_equal,assert_allclose

from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_pixel_config import validate_parameters
from glacier_offsets.velocity_pixel_inversion import get_field,run,stage_prepared_inputs
from glacier_offsets.velocity_pixel_mixed import prepare_mixed_support,check_mixed_support,SUPPORT_FILE
from glacier_offsets.cli import main as cli_main
import test_velocity_pixel_config as pixel_config_tests


def legacy_fixture(root):
    cache=root/'cache';pixel_config_tests.RunnerTests().prepared(cache)
    pair_ids=np.array([f'pair{i}' for i in range(119)])
    expected=np.ones((5,119),bool)
    expected[0,[5,6,8,9]]=False;expected[2,7]=False
    # Source 1 lacks a swath correction on a pure footprint for half of 2017.
    expected[1,30:60]=False
    records=[]
    for year,start,end in ((2017,0,60),(2018,60,119)):
        path=root/f'package-{year}.h5';nt=end-start
        with h5py.File(path,'w') as h:
            h['source_observation_index']=np.arange(5)[None]
            h['pair_id']=np.array(pair_ids[start:end],dtype=h5py.string_dtype())
            horizontal=np.ones((nt,1,5),np.uint8);tidal=horizontal.copy()
            if year==2017:
                horizontal[5,0,0]=0;tidal[6,0,0]=0
                mixed=np.zeros_like(horizontal);mixed[:,0,0]=1;mixed[:30,0,1]=1
                correction=np.ones_like(horizontal);correction[:,0,:2]=0;correction[7,0,2]=0
                h['cross_swath_mixed']=mixed;h['correction_available']=correction
            h['horizontal_valid_mask']=horizontal;h['tidal_correction_available']=tidal
        records.append({'path':str(path),'sha256':digest(path)})
    with h5py.File(cache/'observations.h5','r+') as h:
        h['known_mixed_source']=[True,True,False,False,False]
        h['row']=np.zeros(5,int);h['col']=np.arange(5)
        h['pair_id']=np.array(pair_ids,dtype=h5py.string_dtype())
        h['data'][0,8]=np.nan;h['late'][0,9]=h['early'][0,9]
        old=expected.copy();old[:2]=False;h['valid'][:]=old
    write_json(cache/'inputs.json',{'inputs':records,'cache_sha256':digest(cache/'observations.h5')})
    return cache,expected


class MixedWindowTests(unittest.TestCase):
    def test_cli_flag_overrides_yaml_in_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,expected=legacy_fixture(root)
            with h5py.File(cache/'observations.h5','r+') as h:h['valid_including_mixed']=expected
            meta=json.loads((cache/'inputs.json').read_text());meta['cache_sha256']=digest(cache/'observations.h5');write_json(cache/'inputs.json',meta)
            for enabled,output,flag in ((True,'keep','--no-remove-mixed-windows'),(False,'remove','--remove-mixed-windows')):
                config=root/f'{output}.yaml'
                config.write_text(f'input_directory: cache\noutput: {output}\nscreening:\n  remove_mixed_windows: {str(enabled).lower()}\n')
                with np.errstate(divide='ignore',invalid='ignore'):
                    self.assertEqual(cli_main(['velocity-pixels','fit',str(config),flag]),0)
                recipe=json.loads((root/output/'recipe.json').read_text())
                self.assertEqual(recipe['parameters']['screening']['remove_mixed_windows'],not enabled)
                with h5py.File(root/output/'source_coefficients.h5') as h:
                    self.assertEqual(h['clean_pairs_by_year'][:2].any(),enabled)

    def test_legacy_recovery_retains_mixed_but_preserves_other_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,expected=legacy_fixture(root)
            before=digest(cache/'observations.h5')
            policy=prepare_mixed_support(cache)
            self.assertFalse(policy['remove_mixed_windows'])
            with h5py.File(cache/'observations.h5') as h,h5py.File(cache/'solid_earth_tides.h5') as body:
                assert_array_equal(get_field(h,body,slice(None))['valid'],expected)
                strict=get_field(h,body,slice(None),True)['valid']
                self.assertFalse(strict[:2].any());assert_array_equal(strict[2:],expected[2:])
            self.assertEqual(before,digest(cache/'observations.h5'))
            self.assertEqual(policy,prepare_mixed_support(cache))
            check_mixed_support(cache,{'mixed_window_policy':policy})
            with h5py.File(cache/SUPPORT_FILE,'r+') as h:h['valid_including_mixed'][0,0]=False
            with self.assertRaisesRegex(ValueError,'support changed'):check_mixed_support(cache,{'mixed_window_policy':policy})

    def test_new_cache_uses_preserved_masks_without_original_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,expected=legacy_fixture(root)
            with h5py.File(cache/'observations.h5','r+') as h:h['valid_including_mixed']=expected
            for path in root.glob('package-*.h5'):path.unlink()
            self.assertEqual(prepare_mixed_support(cache)['validity'],'cached')
            with h5py.File(cache/'observations.h5') as h,h5py.File(cache/'solid_earth_tides.h5') as body:
                assert_array_equal(get_field(h,body,slice(None))['valid'],expected)
                self.assertFalse(get_field(h,body,slice(None),True)['valid'][:2].any())

    def test_missing_original_is_explicit_but_exclusion_still_works(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,_=legacy_fixture(root);(root/'package-2018.h5').unlink()
            with self.assertRaisesRegex(ValueError,'original package is unavailable'):prepare_mixed_support(cache)
            self.assertFalse((cache/SUPPORT_FILE).exists())
            self.assertTrue(prepare_mixed_support(cache,True)['remove_mixed_windows'])
            with h5py.File(cache/'observations.h5') as h,h5py.File(cache/'solid_earth_tides.h5') as body:
                self.assertFalse(get_field(h,body,slice(None),True)['valid'][:2].any())

    def test_changed_original_package_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,_=legacy_fixture(root)
            with h5py.File(root/'package-2017.h5','r+') as h:h['horizontal_valid_mask'][0,0,0]=0
            with self.assertRaisesRegex(ValueError,'Original package changed'):prepare_mixed_support(cache)

    def test_flag_changes_actual_fit_support_and_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,_=legacy_fixture(root)
            # Make a self-contained modern cache; no recovery is needed for either fit.
            with h5py.File(cache/'observations.h5','r+') as h:
                valid=np.ones((5,119),bool);valid[0,[5,6,8,9]]=False
                h['valid_including_mixed']=valid
            meta=json.loads((cache/'inputs.json').read_text());meta['cache_sha256']=digest(cache/'observations.h5');write_json(cache/'inputs.json',meta)
            stage_prepared_inputs(cache,root/'keep');stage_prepared_inputs(cache,root/'remove')
            # Deliberately invalid intervals exercise validity, not numerical warnings.
            with np.errstate(divide='ignore',invalid='ignore'):
                run(root/'keep')
                run(root/'remove',parameters=validate_parameters({'screening':{'remove_mixed_windows':True}}))
            with h5py.File(root/'keep/source_coefficients.h5') as keep,h5py.File(root/'remove/source_coefficients.h5') as remove:
                self.assertTrue(keep['residual_vertical/estimable'][:2].all())
                self.assertFalse(remove['residual_vertical/estimable'][:2].any())
                self.assertTrue((keep['clean_pairs_by_year'][:2]>0).all())
                self.assertFalse(remove['clean_pairs_by_year'][:2].any())
                assert_allclose(keep['residual_vertical/coefficients'][2:],remove['residual_vertical/coefficients'][2:])
            recipe=json.loads((root/'keep/recipe.json').read_text())
            self.assertFalse(recipe['parameters']['screening']['remove_mixed_windows'])
            self.assertFalse(recipe['mixed_window_policy']['remove_mixed_windows'])
            with self.assertRaisesRegex(ValueError,'recipe changed'):
                run(root/'keep',parameters=validate_parameters({'screening':{'remove_mixed_windows':True}}))

    def test_recovered_support_is_copied_with_prepared_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cache,_=legacy_fixture(root);prepare_mixed_support(cache)
            stage_prepared_inputs(cache,root/'new')
            for path in root.glob('package-*.h5'):path.unlink()
            self.assertEqual(prepare_mixed_support(root/'new')['support_sha256'],digest(cache/SUPPORT_FILE))


if __name__=='__main__':unittest.main()
