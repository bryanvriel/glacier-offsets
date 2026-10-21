"""Exact scientific invariants and bounded ancillary comparisons are distinct."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.comparison import compare_array, compare_burst, compare_h5, RADAR_REQUIRED, GEOMETRY_REQUIRED
from glacier_offsets.products import write_json


class ComparisonTests(unittest.TestCase):
    def test_ancillary_differences_reported_and_strict_failure(self):
        a=np.array([.041250668,11.392983,.16684993],dtype='float32')
        b=np.array([.041250601,11.392949,.16685155],dtype='float32')
        result=compare_array(a,b,ancillary=True)
        self.assertEqual(result['status'],'within_tolerance')
        self.assertEqual(result['different_values'],3)
        self.assertGreater(result['max_abs_difference'],0)
        self.assertEqual(result['outside_tolerance'],0)
        self.assertEqual(compare_array(a,b,ancillary=True,strict=True)['status'],'failed')
        self.assertEqual(compare_array(a,b)['status'],'failed')

    def test_small_motion_changes_and_quality_mask_changes_fail(self):
        a=np.array([1.],dtype='float32');b=np.nextafter(a,np.inf)
        self.assertEqual(compare_array(a,b)['status'],'failed')
        self.assertEqual(compare_array(np.array([True]),np.array([False]))['status'],'failed')
        self.assertEqual(compare_array(a,a)['status'],'exact')

    def test_nonfinite_patterns_dtype_and_shape_are_exact(self):
        a=np.array([np.nan,np.inf,-np.inf,1.],dtype='float32')
        self.assertEqual(compare_array(a,a,ancillary=True)['status'],'exact')
        for b in (np.array([0,np.inf,-np.inf,1.],dtype='float32'),
                  np.array([np.nan,-np.inf,-np.inf,1.],dtype='float32'),a.astype('float64'),a.reshape(2,2)):
            self.assertEqual(compare_array(a,b,ancillary=True)['status'],'failed')

    def test_larger_ancillary_difference_still_fails(self):
        a=np.array([1.],dtype='float32');b=np.array([1.01],dtype='float32')
        result=compare_array(a,b,ancillary=True)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['outside_tolerance'],1)

    def fixture(self, directory):
        manifest={'coreg_identity':'same','match_identity':'same','geocode_identity':'same',
                  'interval_seconds':86400,'acquisition_times':['a','b'],
                  'coreg':{'reference_grid':{'length':2}},'matching':{'backend':'cpu'}}
        for root in (directory/'batch/b',directory/'standalone/b/b'):
            write_json(root/'manifest.json',manifest)
            for relative,names in [('radar/offsets.h5',RADAR_REQUIRED),
                    ('radar/matching_grid.h5',{'center_line','center_sample'}),
                    ('geo/center_geometry.h5',GEOMETRY_REQUIRED)]:
                (root/relative).parent.mkdir(parents=True,exist_ok=True)
                with h5py.File(root/relative,'w') as h:
                    for name in names:h[name]=np.ones((2,2),dtype='float32')
            for name in ('range_displacement_m.tif','snr.tif'):
                ds=gdal.GetDriverByName('GTiff').Create(str(root/'geo'/name),2,2,1,gdal.GDT_Float32)
                ds.SetGeoTransform((0,1,0,2,0,-1));ds.GetRasterBand(1).Fill(1);ds=None
        write_json(directory/'batch/selection.json',{'selected_ids':['b'],'selection_identity':'fixture'})
        return directory/'batch/b',directory/'standalone/b/b'

    def test_saved_products_pass_and_detect_schema_metadata_and_motion_changes(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.fixture(Path(d))
            self.assertEqual(compare_burst(a,b)['result'],'exact')
            with h5py.File(b/'radar/offsets.h5','r+') as h:
                h['snr'][0,0]=1.000005
            self.assertEqual(compare_burst(a,b)['result'],'within_tolerance')
            self.assertEqual(compare_burst(a,b,strict=True)['result'],'failed')
            with h5py.File(b/'radar/offsets.h5','r+') as h:
                h['total/range_px'][0,0]=np.nextafter(np.float32(1),np.float32(2))
            self.assertEqual(compare_burst(a,b)['result'],'failed')
            with h5py.File(b/'radar/offsets.h5','r+') as h:
                h['total/range_px'][0,0]=1
                del h['valid']
            self.assertEqual(compare_burst(a,b)['result'],'failed')
            manifest=json.loads((b/'manifest.json').read_text());manifest['interval_seconds']=2
            write_json(b/'manifest.json',manifest)
            result=compare_burst(a,b)
            self.assertFalse(result['manifest_checks']['interval_seconds'])

    def test_both_missing_required_dataset_and_georeferencing_fail(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=self.fixture(Path(d))
            for root in (a,b):
                with h5py.File(root/'radar/offsets.h5','r+') as h:del h['valid']
            result=compare_burst(a,b)
            self.assertEqual(result['arrays']['radar/offsets.h5:valid']['status'],'failed')
            ds=gdal.Open(str(b/'geo/snr.tif'),gdal.GA_Update)
            ds.SetGeoTransform((1,1,0,2,0,-1));ds=None
            self.assertFalse(compare_burst(a,b)['arrays']['geo/snr.tif']['georeferencing_exact'])

    def test_compare_only_avoids_sources_and_processing_and_persists_failures(self):
        spec=importlib.util.spec_from_file_location('multi_validator',Path(__file__).parents[1]/'tools/validate_multi_burst.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a,b=self.fixture(d)
            args=['unavailable-input-config.yaml','--output-directory',str(d),'--compare-only']
            with patch.object(module,'load_config',side_effect=AssertionError('source access')), \
                    patch.object(module,'workflow',side_effect=AssertionError('processing')),redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(args),0)
                report=json.loads((d/'comparison.json').read_text())
                self.assertEqual(report['scope'],'existing_products')
                self.assertEqual(report['aoi_restart_checks'],'not_checked')
                with h5py.File(b/'radar/offsets.h5','r+') as h:h['snr'][0,0]=1.000005
                self.assertEqual(module.main(args+['--strict']),1)
                report=json.loads((d/'comparison.json').read_text())
                self.assertEqual(report['outcome'],'failed')
                self.assertGreater(report['comparisons'][0]['arrays']['radar/offsets.h5:snr']['max_abs_difference'],0)
