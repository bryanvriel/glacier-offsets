"""Boundary joins must recover metadata without concealing conflicting grids."""
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zipfile

from s1reader.s1_reader import get_burst_centers_and_boundaries

from glacier_offsets.annotation_geometry import boundary_grid, normalize_burst_geometry
from glacier_offsets.annotation_geometry import source_boundary_row
from glacier_offsets.inputs import inventory
from glacier_offsets.products import checksum


@dataclass(frozen=True)
class Native:
    i_burst: int
    burst_id: str
    sensing_start: datetime
    center: object
    border: object
    radar_metadata: object
    orbit: object = None


def annotation(duplicate='row', dateline=False):
    root = ET.Element('product')
    info = ET.SubElement(ET.SubElement(root, 'imageAnnotation'), 'imageInformation')
    for key, value in [('productComposition', 'Assembled'), ('azimuthTimeInterval', '.002'),
                       ('azimuthPixelSpacing', '14')]:
        ET.SubElement(info, key).text = value
    timing = ET.SubElement(root, 'swathTiming')
    ET.SubElement(timing, 'linesPerBurst').text = '1000'
    ET.SubElement(timing, 'samplesPerBurst').text = '201'
    bursts = ET.SubElement(timing, 'burstList', count='3')
    start = datetime(2018, 9, 17)
    for i in range(3):
        ET.SubElement(ET.SubElement(bursts, 'burst'), 'azimuthTime').text = (start + timedelta(seconds=2*i)).isoformat()
    grid = ET.SubElement(ET.SubElement(root, 'geolocationGrid'), 'geolocationGridPointList')
    for row in range(4):
        points = []
        for pixel in (0, 100, 200):
            point = ET.Element('geolocationGridPoint')
            lon = (-65 if not dateline else 179) + pixel*.01
            if lon > 180:
                lon -= 360
            for key, value in [('azimuthTime', (start+timedelta(seconds=2*row)).isoformat()),
                               ('slantRangeTime', '.005'), ('line', row*1000), ('pixel', pixel),
                               ('latitude', -68+row*.1), ('longitude', lon)]:
                ET.SubElement(point, key).text = str(value)
            points.append(point)
        if row == 1 and duplicate:
            old = deepcopy(points if duplicate == 'row' else points[-1:])
            for point in old:
                point.find('azimuthTime').text = (start+timedelta(seconds=2, milliseconds=-2)).isoformat()
                point.find('latitude').text = str(float(point.findtext('latitude')) - .0001)
            points = old + points
        grid.extend(points)
    grid.set('count', str(len(grid)))
    return root


def natives(tree):
    centers, borders = get_burst_centers_and_boundaries(tree, 3)
    return [Native(i, f'burst{i}', datetime(2018, 9, 17)+timedelta(seconds=2*i),
                   centers[i], borders[i], object()) for i in range(3)]


def reference_surface_annotation(full_following=False):
    """Independent range-Doppler geometry, with two source reference heights."""
    import isce3
    import numpy as np
    start=datetime(2018,9,17)
    radius=7071000.;rate=.001
    states=[]
    for t in range(-60,61,10):
        states.append(isce3.core.StateVector(isce3.core.DateTime(start+timedelta(seconds=t)),
            [radius*np.cos(rate*t),radius*np.sin(rate*t),0],
            [-radius*rate*np.sin(rate*t),radius*rate*np.cos(rate*t),0]))
    orbit=isce3.core.Orbit(states,isce3.core.DateTime(start))
    tree=annotation(None)
    grid=tree.find('geolocationGrid/geolocationGridPointList');grid[:]=[]
    for index in range(4):
        surfaces=(0.,400.) if index==1 else ((0.,) if index==0 else (400.,))
        for surface in surfaces:
            pixels=(0,100,200)
            if index==1 and surface==400. and not full_following:pixels=(200,)
            for pixel in pixels:
                t=2.*index+(1e-6 if index==1 and surface==400. else 0.)
                distance=850000.+500*pixel
                llh=isce3.geometry.rdr2geo(t,distance,orbit,'right',dem=isce3.geometry.DEMInterpolator(surface))
                point=ET.SubElement(grid,'geolocationGridPoint')
                for key,value in [('azimuthTime',(start+timedelta(seconds=t)).isoformat()),
                    ('slantRangeTime',2*distance/isce3.core.speed_of_light),
                    ('line',index*1000),('pixel',pixel),('longitude',np.rad2deg(llh[0])),
                    ('latitude',np.rad2deg(llh[1])),('height',200. if index==1 else surface)]:
                    ET.SubElement(point,key).text=str(value)
    grid.set('count',str(len(grid)))
    return tree,orbit


class AnnotationGeometryTests(unittest.TestCase):
    def test_orbit_validated_source_join_keeps_a_whole_row_without_synthesizing_points(self):
        for full in (False,True):
            with self.subTest(full_following=full):
                tree,orbit=reference_surface_annotation(full)
                before=ET.tostring(tree)
                selected,report=source_boundary_row(tree,1000,orbit)
                self.assertEqual(report['source'],'following' if full else 'previous')
                self.assertAlmostEqual(report['previous']['height_m'],0.,places=5)
                self.assertAlmostEqual(report['following']['height_m'],400.,places=5)
                self.assertLess(report['selected']['height_span_m'],1e-5)
                self.assertLess(report['selected']['max_zero_doppler_distance_m'],1e-5)
                normalized,affected=boundary_grid(tree,orbit)
                self.assertEqual(affected,{0,1})
                actual=[p for p in normalized.findall('.//geolocationGridPoint') if p.findtext('line')=='1000']
                self.assertEqual([ET.tostring(p) for p in actual],[ET.tostring(p) for p in selected])
                self.assertEqual(ET.tostring(tree),before)
                original=[replace(b,orbit=orbit) for b in natives(tree)]
                fixed=normalize_burst_geometry('unused.SAFE',2,'HH',original,tree)
                expected=natives(normalized)
                for index in (0,1):
                    self.assertTrue(fixed[index].border[0].equals_exact(expected[index].border[0],0))
                    self.assertIs(fixed[index].orbit,original[index].orbit)
                    self.assertIs(fixed[index].radar_metadata,original[index].radar_metadata)
                self.assertIs(fixed[2],original[2])
                with self.assertRaisesRegex(ValueError,'coordinates disagree'):
                    boundary_grid(tree)

    def test_orbit_join_does_not_accept_arbitrary_conflicts_or_partial_rows(self):
        for problem in ('position','range','doppler','height','fragment','neighbour','time'):
            with self.subTest(problem=problem):
                tree,orbit=reference_surface_annotation()
                grid=tree.find('geolocationGrid/geolocationGridPointList')
                point=grid[3]
                if problem=='position':point.find('latitude').text='-60'
                elif problem=='range':point.find('slantRangeTime').text='.007'
                elif problem=='doppler':point.find('longitude').text='10'
                elif problem=='height':point.find('height').text='nan'
                elif problem=='fragment':
                    del grid[3:5];grid.set('count',str(len(grid)))
                elif problem=='neighbour':grid[8].find('latitude').text='-60'
                elif problem=='time':point.find('azimuthTime').text='2018-09-17T00:00:01'
                with self.assertRaises((ValueError,RuntimeError)):
                    boundary_grid(tree,orbit)

    def test_duplicate_row_and_endpoint_recover_expected_border_and_center(self):
        expected = natives(annotation(None))
        for mode in ('row', 'endpoint'):
            with self.subTest(mode=mode):
                tree = annotation(mode)
                before = ET.tostring(tree)
                original = natives(tree)
                if mode == 'row':
                    self.assertFalse(original[0].border[0].is_valid)
                fixed = normalize_burst_geometry('unused.SAFE', 2, 'HH', original, tree)
                self.assertEqual(ET.tostring(tree), before)
                for index in (0, 1):
                    self.assertTrue(fixed[index].border[0].equals_exact(expected[index].border[0], 0))
                    self.assertTrue(fixed[index].center.equals_exact(expected[index].center, 0))
                    self.assertIs(fixed[index].radar_metadata, original[index].radar_metadata)
                    self.assertEqual(fixed[index].sensing_start, original[index].sensing_start)
                    self.assertEqual(fixed[index].burst_id, original[index].burst_id)
                self.assertIs(fixed[2], original[2])
                # COMPASS supplies a filtered list, not every burst in the SAFE.
                filtered = normalize_burst_geometry('unused.SAFE', 2, 'HH', original[1:2], tree)
                self.assertTrue(filtered[0].center.equals_exact(expected[1].center, 0))

    def test_no_duplicates_is_an_exact_noop_and_dateline_parts_survive(self):
        tree = annotation(None)
        items = natives(tree)
        self.assertIs(normalize_burst_geometry('unused.SAFE', 2, 'HH', items, tree), items)
        tree = annotation(dateline=True)
        # Upstream dateline splitting cannot itself split a self-crossing raw
        # polygon. Exercise our normalized geometry with already loaded natives.
        fixed = normalize_burst_geometry('unused.SAFE', 2, 'HH',
                                         natives(annotation(None, dateline=True)), tree)
        self.assertEqual(len(fixed[0].border), 2)
        self.assertTrue(all(p.is_valid and not p.is_empty for p in fixed[0].border))

    def test_conflicting_records_and_nonboundary_duplicates_are_rejected(self):
        for problem in ('time', 'position', 'nonfinite', 'ambiguous', 'external', 'ordinary', 'count', 'rows'):
            with self.subTest(problem=problem):
                tree = annotation()
                grid = tree.find('geolocationGrid/geolocationGridPointList')
                point = grid[3]  # Older record on the first internal boundary.
                if problem == 'time':
                    point.find('azimuthTime').text = '2018-09-16T00:00:00'
                elif problem == 'position':
                    point.find('latitude').text = '-60'
                elif problem == 'nonfinite':
                    point.find('longitude').text = 'nan'
                elif problem == 'ambiguous':
                    point.find('azimuthTime').text = grid[6].findtext('azimuthTime')
                elif problem == 'external':
                    grid.append(deepcopy(grid[0])); grid.set('count', str(len(grid)))
                elif problem == 'ordinary':
                    tree.find('imageAnnotation/imageInformation/productComposition').text = 'Slice'
                elif problem == 'count':
                    grid.set('count', '999')
                elif problem == 'rows':
                    grid[-1].find('line').text = '2999'
                with self.assertRaisesRegex(ValueError, 'Conflicting annotation'):
                    boundary_grid(tree)

    def test_unrelated_invalid_outline_still_fails(self):
        tree = annotation()
        grid = tree.find('geolocationGrid/geolocationGridPointList')
        grid[1].find('latitude').text = '-67.5'
        with self.assertRaisesRegex(ValueError, 'invalid geometry'):
            normalize_burst_geometry('unused.SAFE', 2, 'HH', natives(tree), tree)

    def test_inventory_uses_normalization_for_directory_and_zip_without_writes(self):
        tree = annotation()
        def loader(*args, **kwargs):
            return natives(tree)
        loader.__module__ = 's1reader.s1_reader'
        with tempfile.TemporaryDirectory() as d:
            safe = Path(d)/'example.SAFE'
            (safe/'annotation').mkdir(parents=True)
            xml = safe/'annotation/s1a-iw2-slc-hh-test.xml'
            xml.write_bytes(ET.tostring(tree))
            archive = Path(d)/'example.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.write(xml, f'{safe.name}/annotation/{xml.name}')
            for path in (safe, archive):
                before = checksum(path)
                with patch('glacier_offsets.inputs.validate_orbit') as orbit:
                    result = inventory(path, 'orbit.EOF', 'HH', [2], loader)
                orbit.assert_called_once()
                self.assertEqual(len(result), 3)
                self.assertTrue(all(p.is_valid for b in result for p in b.native.border))
                self.assertEqual(checksum(path), before)


if __name__ == '__main__':
    unittest.main()
