from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

from shapely.geometry import Point, Polygon
from glacier_offsets.burst_kml import KML_NAMESPACE, burst_document
from glacier_offsets.initialization import SceneMetadata
from glacier_offsets.inputs import Burst


class BurstKMLTests(unittest.TestCase):
    def setUp(self):
        self.time = datetime(2020, 1, 10, 8)
        self.scene = SceneMetadata(Path('/data/a&b.zip'), 'S1A', self.time, self.time, ('HH',), 38, 'DESCENDING')
        self.ns = {'k': KML_NAMESPACE}

    def burst(self, key, swath=1, borders=None):
        if borders is None:
            # Intentionally clockwise input, geographically asymmetric axes.
            borders = [Polygon([(-65, -68), (-65, -67), (-64, -67), (-64, -68)])]
        return Burst(key, swath, self.time, SimpleNamespace(border=borders, center=Point(-64.5, -67.5)))

    def test_longitude_latitude_ring_orientation_labels_and_folders(self):
        bursts = [self.burst('z_iw2', 2), self.burst('z_iw1'), self.burst('a_iw1'), self.burst('a_iw3', 3)]
        root = ET.fromstring(ET.tostring(burst_document(bursts, self.scene, 'HH').getroot()))
        folders = root.findall('k:Document/k:Folder', self.ns)
        self.assertEqual([f.findtext('k:name', namespaces=self.ns) for f in folders], ['IW1', 'IW2', 'IW3'])
        names = [p.findtext('k:name', namespaces=self.ns) for p in root.findall('.//k:Placemark', self.ns)]
        self.assertEqual(names, ['a_iw1', 'z_iw1', 'z_iw2', 'a_iw3'])
        ring = root.find('.//k:outerBoundaryIs/k:LinearRing/k:coordinates', self.ns).text
        coords = [tuple(map(float, p.split(','))) for p in ring.split()]
        self.assertEqual(coords[0], coords[-1])
        self.assertEqual(coords[0], (-65, -68, 0))
        self.assertTrue(Polygon([(x, y) for x, y, _ in coords]).exterior.is_ccw)
        point = root.find('.//k:Point/k:coordinates', self.ns).text
        self.assertEqual(tuple(map(float, point.split(','))), (-64.5, -67.5, 0))
        self.assertEqual(root.findtext('.//k:Data[@name="safe_path"]/k:value', namespaces=self.ns), '/data/a&b.zip')
        self.assertEqual(root.findtext('.//k:Data[@name="sensing_start_UTC"]/k:value', namespaces=self.ns), '2020-01-10T08:00:00Z')

    def test_antimeridian_parts_and_holes_are_preserved(self):
        borders = [Polygon([(179, 1), (180, 1), (180, 2), (179, 2)]),
                   Polygon([(-180, 1), (-179, 1), (-179, 2), (-180, 2)])]
        burst = self.burst('dateline_iw1', borders=borders)
        burst.native.center = Point(180, 1.5)
        root = burst_document([burst], self.scene, 'HH').getroot()
        self.assertEqual(len(root.findall('.//k:Placemark', self.ns)), 1)
        self.assertEqual(len(root.findall('.//k:Polygon', self.ns)), 2)
        hole = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]])
        root = burst_document([self.burst('hole_iw1', borders=[hole])], self.scene, 'HH').getroot()
        text = root.find('.//k:innerBoundaryIs/k:LinearRing/k:coordinates', self.ns).text
        coords = [tuple(map(float, p.split(',')[:2])) for p in text.split()]
        self.assertFalse(Polygon(coords).exterior.is_ccw)

    def test_missing_invalid_footprints_and_duplicate_ids_fail(self):
        for bursts in [[], [self.burst('empty', borders=[])], [self.burst('empty', borders=[Polygon()])],
                       [self.burst('bad', borders=[Polygon([(181, 0), (182, 0), (182, 1), (181, 1)])])],
                       [self.burst('a'), self.burst('a')]]:
            with self.assertRaises(ValueError):
                burst_document(bursts, self.scene, 'HH')


if __name__ == '__main__':
    unittest.main()
