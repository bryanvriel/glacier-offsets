"""Full-image segment adaptation preserves centers, halos and physical units."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from glacier_offsets.ampcor import _native_executor
from glacier_offsets.matching_grid import MatchingGrid
from tools.compare_merged_preprocessing import range_native_worker, refinement_settings, matching_without_zoom
from tools.compare_swath_range_oversampling import prepare_strip
from test_swath_overlap import make_pair


class MergedRangeAdapterTests(unittest.TestCase):
    def test_scalar_zoom_extents_and_native_cap_are_explicit(self):
        params={'zoom':16,'raw_oversampling':2,'search':[10,40]}
        selected,extent=refinement_settings(params,32)
        self.assertEqual(params['zoom'],16);self.assertEqual(selected['zoom'],32)
        self.assertEqual(extent['deramp1_half_width_original_px'],[8,8])
        self.assertEqual(extent['range2x_half_width_original_px'],[8,4])
        for zoom in (0,17,44):
            with self.assertRaises(ValueError):refinement_settings(params,zoom)

    def test_saved_implicit_isce3_engine_is_equivalent_but_other_changes_are_not(self):
        old={'zoom':16,'batch':[1,8],'deramp':1}
        new={'zoom':32,'batch':[1,8],'deramp':1,'engine':'isce3'}
        self.assertEqual(matching_without_zoom(old),matching_without_zoom(new))
        self.assertNotEqual(matching_without_zoom(old),matching_without_zoom(new|{'batch':[1,4]}))
        with self.assertRaises(ValueError):matching_without_zoom(new|{'engine':'isce2'})

    def test_native_zoom32_is_not_clipped_and_preserves_known_motion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();pair,_=make_pair(root);stack=pair.parent.parent
            params=json.loads((pair/'manifest.json').read_text())['matching']['matching']
            params.update(zoom=32,search=[10,16])
            grid=MatchingGrid(np.array([12,60]),np.array([24,120]),(32,64),(10,16),(48,96))
            paths=[stack/'acquisitions'/day/'t038_080975_iw1.tif' for day in ('2020-01-01','2020-01-07')]
            prepared=[]
            for i,path in enumerate(paths):
                target=root/f'prepared-{i}.tif';_,record=prepare_strip(path,target,grid);prepared.append(str(target))
            y0,_,x0,_=record['retained_original_window_yx']
            cfg={'gross_offsets':{'mode':'zero'},'_range2x_control':{'paths':prepared,'origin_yx':[y0,x0]}}
            with _native_executor(1,single_threaded=True) as pool:
                result=pool.apply(range_native_worker,(((*paths,root/'native',params,cfg,'cpu'),
                    {'_grid':grid,'_valid_input':np.ones(grid.shape,bool),'_gross':None}),))
            self.assertEqual(result['effective_parameters']['corrSurfaceZoomInWindow'],32)
            self.assertTrue(result['valid'].all())
            np.testing.assert_array_equal(result['grid'].centers,grid.centers)
            np.testing.assert_allclose(result['total'],np.broadcast_to([0,1],(*grid.shape,2)),atol=.05)

    def test_native_segments_use_same_complete_images_and_original_centers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();pair,grid=make_pair(root);stack=pair.parent.parent
            params=json.loads((pair/'manifest.json').read_text())['matching']['matching']
            paths=[stack/'acquisitions'/day/'t038_080975_iw1.tif' for day in ('2020-01-01','2020-01-07')]
            prepared=[];records=[]
            for i,path in enumerate(paths):
                target=root/f'prepared-{i}.tif';_,record=prepare_strip(path,target,grid)
                prepared.append(str(target));records.append(record)
            y0,_,x0,_=records[0]['retained_original_window_yx']
            cfg={'gross_offsets':{'mode':'zero'},'_range2x_control':{'paths':prepared,'origin_yx':[y0,x0]}}
            before=[p.read_bytes() for p in paths]
            with _native_executor(1,single_threaded=True) as pool:
                full=pool.apply(range_native_worker,(((*paths,root/'full',params,cfg,'cpu'),
                        {'_grid':grid,'_valid_input':np.ones(grid.shape,bool),'_gross':None}),))
                segment=MatchingGrid(grid.starts_line[1:],grid.starts_sample[1:],grid.window,grid.search,grid.stride)
                part=pool.apply(range_native_worker,(((*paths,root/'part',params,cfg,'cpu'),
                        {'_grid':segment,'_valid_input':np.ones(segment.shape,bool),'_gross':None}),))
            for result,expected_grid in ((full,grid),(part,segment)):
                self.assertTrue(result['valid'].all())
                np.testing.assert_array_equal(result['grid'].centers,expected_grid.centers)
                np.testing.assert_allclose(result['total'],np.broadcast_to([0,1],(*expected_grid.shape,2)),atol=.05)
                self.assertEqual(result['effective_parameters']['referenceImageName'],prepared[0])
                self.assertEqual(result['effective_parameters']['secondaryImageName'],prepared[1])
            np.testing.assert_allclose(part['total'],full['total'][1:,1:],atol=1/64)
            for path,contents in zip(paths,before):self.assertEqual(path.read_bytes(),contents)


if __name__=='__main__':unittest.main()
