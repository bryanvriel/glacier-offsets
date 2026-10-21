import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from osgeo import gdal
import isce3
from compass.utils.radar_grid import rdr_grid_to_file
from glacier_offsets.compass_coreg import make_runconfig, inspect_pair


class CoregTests(unittest.TestCase):
    def test_config_matches_installed_schema(self):
        import yamale
        import compass
        import yaml
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);orbit=root/'orbit.EOF';orbit.touch()
            cfg={'inputs':{'reference_orbit':str(orbit),'reference_safe':'/tmp/example.SAFE','dem':'/tmp/dem.tif','polarization':'HH'}}
            pair=[SimpleNamespace(id='t038_080971_iw1')]
            doc=make_runconfig(cfg,pair,'reference',root)
            f=root/'compass.yaml';f.write_text(yaml.safe_dump(doc))
            schema=yamale.make_schema(str(Path(compass.__file__).parent/'schemas/s1_cslc_radar.yaml'))
            yamale.validate(schema,yamale.make_data(str(f)))
            self.assertFalse(doc['runconfig']['groups']['processing']['resample']['flatten'])
            self.assertEqual(list((root/'orbits/reference').iterdir()),[root/'orbits/reference'/orbit.name])
    def test_coreg_checks_complex_coverage_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);grid=isce3.product.RadarGridParameters(0,0.055,1000,800000,2.3,'right',8,10,isce3.core.DateTime('2020-01-10'))
            rdr_grid_to_file(str(p/'radar_grid.txt'),grid)
            def write(name,a,typ):
                ds=gdal.GetDriverByName('GTiff').Create(str(p/name),a.shape[1],a.shape[0],1,typ)
                ds.GetRasterBand(1).WriteArray(a);ds=None
            for n in ['a.tif','b.tif']:write(n,np.ones((8,10),dtype=np.complex64),gdal.GDT_CFloat32)
            for n in ['x','y','z','layover_shadow_mask']:write(n+'.tif',np.ones((8,10)),gdal.GDT_Float64)
            vrt = gdal.BuildVRT(str(p/'topo.vrt'), [str(p/(n+'.tif')) for n in ['x','y','z']], separate=True)
            from osgeo import osr
            crs=osr.SpatialReference(); crs.ImportFromEPSG(4326); vrt.SetProjection(crs.ExportToWkt()); vrt=None
            out=inspect_pair(p/'a.tif',p/'b.tif',p)
            self.assertEqual(out['coverage']['secondary'],1)
            write('b.tif',np.zeros((8,10),dtype=np.complex64),gdal.GDT_CFloat32)
            with self.assertRaisesRegex(ValueError,'no valid'):inspect_pair(p/'a.tif',p/'b.tif',p)
            write('b.tif',np.ones((8,10)),gdal.GDT_Float32)
            with self.assertRaisesRegex(ValueError,'complex'):inspect_pair(p/'a.tif',p/'b.tif',p)
