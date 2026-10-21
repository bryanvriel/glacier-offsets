"""AOI geometry, metadata-only discovery, and selector configuration contracts."""
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import yaml
from osgeo import ogr, osr
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box, mapping
from shapely.ops import transform, unary_union

from glacier_offsets.aoi import load_aoi, select_aoi, footprint, project_footprint
from glacier_offsets.cli import main
from glacier_offsets.config import load_config
from glacier_offsets.inputs import Burst, discover_validated
import test_initialization as initialization
from test_initialization import scene, orbit


def pair(key='a', ref=None, sec=None, swath=1):
    start = datetime(2020, 1, 10, 8)
    def burst(geometry, time):
        native = SimpleNamespace(border=[geometry], polarization='hh', orbit_direction='DESCENDING')
        native.as_isce3_radargrid = lambda: SimpleNamespace(prf=1000, range_pixel_spacing=2.3)
        return Burst(key, swath, time, native)
    ref = box(0, 0, 2, 2) if ref is None else ref
    return burst(ref, start), burst(ref if sec is None else sec, start + timedelta(days=12))


def vector(path, geometries, epsg=4326, layer_names=('aoi',)):
    driver = ogr.GetDriverByName({'.geojson':'GeoJSON', '.shp':'ESRI Shapefile', '.gpkg':'GPKG'}[path.suffix])
    ds = driver.CreateDataSource(str(path))
    srs = None
    if epsg:
        srs = osr.SpatialReference(); srs.ImportFromEPSG(epsg)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    for name in layer_names:
        layer = ds.CreateLayer(name, srs=srs, geom_type=ogr.wkbUnknown)
        for geometry in geometries:
            f = ogr.Feature(layer.GetLayerDefn())
            f.SetGeometry(ogr.CreateGeometryFromWkb(geometry.wkb))
            layer.CreateFeature(f)
    ds = None
    return path


def kml(path, geometries=(), folders=None):
    """Handwritten KML exercises the reader independently of its vector writer."""
    def placemarks(items):
        return ''.join('<Placemark>' + ogr.CreateGeometryFromWkb(geom.wkb).ExportToKML()
                       + '</Placemark>' for geom in items)
    content = placemarks(geometries) if folders is None else ''.join(
        f'<Folder><name>{name}</name>{placemarks(items)}</Folder>' for name, items in folders.items())
    path.write_text('<?xml version="1.0"?><kml xmlns="http://www.opengis.net/kml/2.2">'
                    '<Document><name>AOI</name>' + content + '</Document></kml>')
    return path


class AOITests(unittest.TestCase):
    def test_kml_polygons_holes_multigeometry_and_altitude(self):
        hole = Polygon([(0,0,100),(4,0,100),(4,4,100),(0,4,100)],
                       [[(1,1,100),(3,1,100),(3,3,100),(1,3,100)]])
        components = GeometryCollection([hole, GeometryCollection([box(10,10,11,11)])])
        with tempfile.TemporaryDirectory() as d:
            p = kml(Path(d)/'a.kml', [components, box(20,20,21,21)])
            aoi = load_aoi({'path':str(p)})
            self.assertTrue(aoi.crs.equals(CRS.from_epsg(4326)))
            self.assertFalse(aoi.geometry.has_z)
            expected = Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(3,1),(3,3),(1,3)]])
            self.assertTrue(aoi.geometry.equals(unary_union([expected,box(10,10,11,11),box(20,20,21,21)])))
            rows = [pair('hole',box(1.2,1.2,2,2)),pair('edge',box(4,0,5,1)),
                    pair('inside',box(0,0,.5,.5)),pair('component',box(10,10,11,11))]
            chosen, report = select_aoi({'aoi':{'path':str(p)}}, rows)
            self.assertEqual([item[0].id for item in chosen],['component','inside'])
            self.assertEqual(list(report['aoi']['source_checksums']),[str(p.resolve())])
            old = report['aoi']['source_checksums']
            p.write_text(p.read_text().replace('<name>AOI</name>','<name>Changed</name>'))
            self.assertNotEqual(old,load_aoi({'path':str(p)}).report['source_checksums'])

    def test_kml_bbox_equivalence_altitude_identity_and_dateline(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            geometry = Polygon([(170,-5,10),(-170,-5,20),(-170,5,30),(170,5,40)])
            p = kml(d/'crossing.kml',[geometry])
            aoi = load_aoi({'path':str(p)})
            bbox = load_aoi({'bbox':[170,-5,-170,5]})
            self.assertEqual(aoi.report['geometry_identity'],bbox.report['geometry_identity'])
            rows = [pair('east',box(175,-1,179,1)),pair('west',box(-179,-1,-175,1)),
                    pair('greenwich',box(0,-1,1,1))]
            self.assertEqual(select_aoi({'aoi':{'path':str(p)}},rows)[0],
                             select_aoi({'aoi':{'bbox':[170,-5,-170,5]}},rows)[0])
            flat = kml(d/'flat.kml',[box(170,-5,179,5)])
            elevated = kml(d/'elevated.kml',[Polygon([(170,-5,100),(179,-5,100),(179,5,100),(170,5,100)])])
            self.assertEqual(load_aoi({'path':str(flat)}).report['geometry_identity'],
                             load_aoi({'path':str(elevated)}).report['geometry_identity'])

    def test_kml_folder_selection_and_nonpolygon_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            p = kml(d/'folders.kml',folders={'glaciers':[box(0,0,1,1)],'markers':[Point(2,2)]})
            with self.assertRaisesRegex(ValueError,'Multiple layers.*glaciers.*markers'):
                load_aoi({'path':str(p)})
            self.assertTrue(load_aoi({'path':str(p),'layer':'glaciers'}).geometry.equals(box(0,0,1,1)))
            with self.assertRaisesRegex(ValueError,'Polygon'):
                load_aoi({'path':str(p),'layer':'markers'})
            with self.assertRaisesRegex(ValueError,'No layer'):
                load_aoi({'path':str(p),'layer':'missing'})
            for i, geometry in enumerate([Point(0,0), LineString([(0,0),(1,1)]),
                                           GeometryCollection([box(0,0,1,1),Point(2,2)]),
                                           Polygon([(0,0),(1,1),(0,1),(1,0)])]):
                p = kml(d/f'bad-{i}.kml',[geometry])
                with self.assertRaises(ValueError):load_aoi({'path':str(p)})

    def test_kml_inherited_relative_path_and_metadata_only_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d/'configs').mkdir()
            p = kml(d/'aoi.kml',[box(0,0,1,1)])
            inp = d/'input';inp.touch()
            parent = {'inputs':{key:str(inp) for key in ('reference_safe','secondary_safe','reference_orbit','secondary_orbit')},
                      'output':{'directory':str(d/'work')}, 'aoi':{'path':'../aoi.kml'}}
            (d/'configs/base.yaml').write_text(yaml.safe_dump(parent))
            child = d/'child.yaml';child.write_text('extends: configs/base.yaml\n')
            self.assertEqual(load_config(child,discovery=True)['aoi']['path'],str(p.resolve()))
            with patch('glacier_offsets.inputs.discover_validated',return_value=[pair()]), \
                    patch('glacier_offsets.pipeline.execute',side_effect=AssertionError('processing')), \
                    redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['list-bursts',str(child)]),0)
                self.assertIn('1 selected / 1 common',out.getvalue())
            self.assertFalse((d/'work').exists())

    def test_vector_bbox_equivalence_deterministic_order(self):
        rows = [pair('z', swath=2), pair('b'), pair('a'), pair('outside', box(5,5,6,6))]
        with tempfile.TemporaryDirectory() as d:
            p = vector(Path(d)/'a.geojson', [box(.2,.2,1,1)])
            first, report = select_aoi({'aoi':{'path':str(p)}}, rows)
            second, _ = select_aoi({'aoi':{'bbox':[.2,.2,1,1]}}, rows[::-1])
            self.assertEqual([p[0].id for p in first], ['a','b','z'])
            self.assertEqual(first, second)
            self.assertEqual(report['common_count'],4)
            self.assertEqual(report, select_aoi({'aoi':{'path':str(p)}}, rows[::-1])[1])
            self.assertEqual(len(report['aoi']['source_checksums']),1)

    def test_holes_components_boundary_and_three_way_overlap(self):
        hole = Polygon([(0,0),(4,0),(4,4),(0,4)], [[(1,1),(3,1),(3,3),(1,3)]])
        with tempfile.TemporaryDirectory() as d:
            path = vector(Path(d)/'a.geojson',[hole, box(10,10,11,11)])
            rows = [pair('hole',box(1.2,1.2,2,2)), pair('edge',box(4,0,5,1)),
                    pair('inside',box(0,0,.5,.5)), pair('component',box(10,10,11,11))]
            chosen, _ = select_aoi({'aoi':{'path':str(path)}}, rows)
            self.assertEqual([p[0].id for p in chosen],['component','inside'])
            path = vector(Path(d)/'b.geojson',[box(0,0,1,1), box(3,0,4,1)])
            chosen, _ = select_aoi({'aoi':{'path':str(path)}}, [pair('disjoint', box(0,0,2.5,1), box(1.5,0,4,1))])
            self.assertEqual(chosen, ())

    def test_projected_antarctic_boundaries_preserved(self):
        crs = CRS.from_epsg(3031)
        native = project_footprint(box(-105,-76,-104,-75), crs)
        center = native.representative_point()
        geom = center.buffer(1000).envelope
        with tempfile.TemporaryDirectory() as d:
            p = vector(Path(d)/'a.gpkg',[geom],3031)
            aoi = load_aoi({'path':str(p)})
            self.assertTrue(aoi.geometry.equals(geom))
            chosen, _ = select_aoi({'aoi':{'path':str(p)}},[pair('antarctic',box(-105,-76,-104,-75))])
            self.assertEqual(len(chosen),1)
        long_edge = project_footprint(box(-105,-76,-104,-75),crs)
        self.assertGreater(len(long_edge.exterior.coords),300)

    def test_dateline_bbox_vector_split_parts_and_hole(self):
        crossing = Polygon([(170,-5),(-170,-5),(-170,5),(170,5)])
        with tempfile.TemporaryDirectory() as d:
            p = vector(Path(d)/'a.geojson',[crossing])
            a = load_aoi({'path':str(p)})
            b = load_aoi({'bbox':[170,-5,-170,5]})
            self.assertTrue(a.geometry.equals(b.geometry))
            rows = [pair('east',box(175,-1,179,1)),pair('west',box(-179,-1,-175,1)),pair('greenwich',box(0,-1,1,1))]
            chosen, _ = select_aoi({'aoi':{'path':str(p)}}, rows)
            self.assertEqual([p[0].id for p in chosen],['east','west'])
            both = pair('split',box(175,-1,179,1))
            both[0].native.border.append(box(-179,-1,-175,1))
            self.assertEqual(len(footprint(both[0]).geoms),2)
            # Dateline parts become neighbors in a polar projected CRS.
            crossing_polar = Polygon([(170,-76),(-170,-76),(-170,-74),(170,-74)])
            projected = project_footprint(footprint(pair('polar',crossing_polar)[0]),CRS.from_epsg(3031))
            inside = projected.representative_point().buffer(1000).envelope
            polar_path = vector(Path(d)/'polar.gpkg',[inside],3031)
            selected, _ = select_aoi({'aoi':{'path':str(polar_path)}},[pair('polar',crossing_polar)])
            self.assertEqual(len(selected),1)

    def test_invalid_geometry_crs_layer_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            bad = Polygon([(0,0),(1,1),(0,1),(1,0)])
            for name, geoms, epsg, reason in [
                ('invalid.shp',[bad],4326,'invalid'), ('missing.shp',[box(0,0,1,1)],None,'CRS'),
                ('empty.gpkg',[],4326,'no features')]:
                p = vector(d/name,geoms,epsg)
                with self.assertRaisesRegex(ValueError,reason):load_aoi({'path':str(p)})
            p = vector(d/'multi.gpkg',[box(0,0,1,1)],layer_names=('one','two'))
            with self.assertRaisesRegex(ValueError,'Multiple layers'):load_aoi({'path':str(p)})
            with self.assertRaisesRegex(ValueError,'No layer'):load_aoi({'path':str(p),'layer':'missing'})
            self.assertFalse(load_aoi({'path':str(p),'layer':'one'}).geometry.is_empty)
            p = d/'point.geojson'
            p.write_text(json.dumps({'type':'Point','coordinates':[0,0]}))
            with self.assertRaisesRegex(ValueError,'Polygon'):load_aoi({'path':str(p)})
            broken = pair('broken');broken[0].native.border=[]
            with self.assertRaisesRegex(ValueError,'broken.*missing footprint'):
                select_aoi({'aoi':{'bbox':[0,0,1,1]}},[broken])
            broken[0].native.border=['not a polygon']
            with self.assertRaisesRegex(ValueError,'broken.*Polygon'):
                select_aoi({'aoi':{'bbox':[0,0,1,1]}},[broken])
            broken[0].native.border=box(0,0,1,1)
            with self.assertRaisesRegex(ValueError,'broken.*collection'):
                select_aoi({'aoi':{'bbox':[0,0,1,1]}},[broken])
            self.assertEqual(select_aoi({'aoi':{'bbox':[10,10,11,11]}},[pair()])[0],())

    def test_invalid_specs_and_sidecar_checksums(self):
        for spec in ({},{'path':None,'bbox':None},{'bbox':[0,0,0,1]}, {'bbox':[180,0,-180,1]},
                     {'bbox':[0,1,1,0]}, {'bbox':[0,-91,1,1]}, {'bbox':[0,0,float('inf'),1]},
                     {'bbox':[0,0,181,1]}, {'bbox':[False,0,1,1]}, {'bbox':[0,0,1,1],'layer':'x'}):
            with self.subTest(spec=spec),self.assertRaises(ValueError):load_aoi(spec)
        with tempfile.TemporaryDirectory() as d:
            p=vector(Path(d)/'a.shp',[box(0,0,1,1)])
            report=load_aoi({'path':str(p)}).report
            self.assertTrue({'.shp','.shx','.dbf','.prj'} <= {Path(p).suffix for p in report['source_checksums']})
            old=report['source_checksums']
            p.with_suffix('.cpg').write_text('UTF-8')
            self.assertNotEqual(old,load_aoi({'path':str(p)}).report['source_checksums'])

    def test_config_inheritance_selectors_and_metadata_only_cli(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); (d/'child').mkdir()
            p=vector(d/'a.geojson',[box(0,0,1,1)])
            inp=d/'input';inp.touch()
            parent={'inputs':{k:str(inp) for k in ('reference_safe','secondary_safe','reference_orbit','secondary_orbit')} | {'burst_id':'old'},
                    'output':{'directory':str(d/'work')}}
            (d/'base.yaml').write_text(yaml.safe_dump(parent))
            child=d/'child/pair.yaml'
            child.write_text(yaml.safe_dump({'extends':'../base.yaml','aoi':{'path':'../a.geojson'}}))
            with self.assertRaisesRegex(ValueError,'exclusively'):load_config(child,discovery=True)
            child.write_text(yaml.safe_dump({'extends':'../base.yaml','inputs':{'burst_id':None},'aoi':{'path':'../a.geojson'}}))
            cfg=load_config(child,discovery=True)
            self.assertEqual(cfg['aoi']['path'],str(p.resolve()))
            grand=d/'grand.yaml';grand.write_text('extends: child/pair.yaml\n')
            self.assertEqual(load_config(grand,discovery=True)['aoi'],cfg['aoi'])
            with patch('glacier_offsets.inputs.discover_validated',return_value=[pair()]), \
                    patch('glacier_offsets.products.checksum',side_effect=AssertionError('SAFE hashing')), \
                    patch('glacier_offsets.pipeline.execute',side_effect=AssertionError('processing')), \
                    redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['list-bursts',str(child)]),0)
                self.assertIn('1 selected / 1 common',out.getvalue())
            self.assertFalse((d/'work').exists())

    def test_shared_manifest_orbit_validation_inventory_once(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); t=datetime(2020,1,10,8)
            cfg={'inputs':{'reference_safe':str(scene(d/'a.zip',t)),
                'secondary_safe':str(scene(d/'b.zip',t+timedelta(days=12))),
                'reference_orbit':str(orbit(d/'a.EOF',t-timedelta(hours=8))),
                'secondary_orbit':str(orbit(d/'b.EOF',t+timedelta(days=12)-timedelta(hours=8))),
                'polarization':'HH','subswaths':[1]}}
            with patch('glacier_offsets.inputs.inventory',side_effect=[[pair()[0]],[pair()[1]]]) as inv:
                self.assertEqual(len(discover_validated(cfg)),1)
                self.assertEqual(inv.call_count,2)
            cfg['inputs']['secondary_safe']=str(scene(d/'wrong.zip',t+timedelta(days=12),track=39))
            with patch('glacier_offsets.inputs.inventory') as inv:
                with self.assertRaisesRegex(ValueError,'relative orbit'):discover_validated(cfg)
                inv.assert_not_called()

    def test_initializer_aoi_relative_paths_and_exclusive_flags(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('initializer',Path(__file__).parents[1]/'tools/init_pair_config.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            fixture=initialization.InitializationTests();fixture.setUp()
            safes,dem,orbits=fixture.fixture(d)
            p=vector(d/'a.geojson',[box(0,0,1,1)])
            output=d/'configs/pair.yaml'
            args=list(map(str,safes))+['--dem',str(dem),'--orbit-dir',str(orbits),'--output',str(output),'--relative-paths']
            with patch('glacier_offsets.aoi.select_aoi',return_value=([pair()],{})),redirect_stdout(io.StringIO()):
                module.main(args+['--aoi',str(p)])
            self.assertEqual(yaml.safe_load(output.read_text())['aoi']['path'],'../a.geojson')
            self.assertEqual(load_config(output)['aoi']['path'],str(p.resolve()))
            with self.assertRaisesRegex(ValueError,'--aoi-layer'):
                from glacier_offsets.initialization import make_pair_config
                make_pair_config(safes,dem,orbits,aoi_layer='x')
