"""Independent native zero-Doppler map-to-radar lookup check, following Geogrid's direction."""
import json
from pathlib import Path
import h5py
import isce3
import numpy as np
from pyproj import Transformer
from osgeo import gdal
from glacier_offsets.config import load_config
from glacier_offsets.inputs import select_pair
from glacier_offsets.geocode import RadarNearestMapper
from glacier_offsets.products import write_json

cfg=load_config('configs/pair.yaml')
root=Path(cfg['output']['directory'])/cfg['inputs']['burst_id']
m=json.loads((root/'manifest.json').read_text())
native=select_pair(cfg)[0].native
grid=native.as_isce3_radargrid()
epoch=(grid.ref_epoch-native.orbit.reference_epoch).total_seconds()
with h5py.File(root/'radar/offsets.h5') as h:
 line=h['center_line'][:];sample=h['center_sample'][:];quality=h['valid'][:]
 total_range=h['total/range_px'][:];total_az=h['total/azimuth_px'][:]
with h5py.File(root/'geo/center_geometry.h5') as h:
 lon=h['longitude_deg'][:];lat=h['latitude_deg'][:];valid=quality&h['geometry_valid'][:]
 azscale=h['azimuth_ground_spacing_m_per_slc_line'][:]
epsg=m['geocoding']['epsg'];spacing=m['geocoding']['spacing_m']
x,y=Transformer.from_crs(4326,epsg,always_xy=True).transform(lon,lat)
mapper=RadarNearestMapper(x,y,valid,spacing)
transformer=Transformer.from_crs(epsg,4326,always_xy=True)
dem=isce3.geometry.DEMInterpolator(isce3.io.Raster(cfg['inputs']['dem']))
range_map=gdal.Open(str(root/'geo/range_displacement_m.tif')).ReadAsArray()
az_map=gdal.Open(str(root/'geo/azimuth_displacement_m.tif')).ReadAsArray()
records=[]
for k in np.linspace(0,len(mapper.positions)-1,30).astype(int):
 row,col=np.unravel_index(mapper.positions[k],mapper.shape)
 xx=mapper.transform[0]+(col+.5)*spacing
 yy=mapper.transform[3]-(row+.5)*spacing
 ll=np.deg2rad(transformer.transform(xx,yy))
 height=dem.interpolate_lonlat(*ll)
 t,r=isce3.geometry.geo2rdr([ll[0],ll[1],height],isce3.core.Ellipsoid(),native.orbit,
                          isce3.core.LUT2d(),grid.wavelength,grid.lookside)
 native_line=(t-epoch-grid.sensing_start)*grid.prf
 native_sample=(r-grid.starting_range)/grid.range_pixel_spacing
 i=int(np.floor((native_line-line[0,0])/32+.5))
 j=int(np.floor((native_sample-sample[0,0])/128+.5))
 actual=np.unravel_index(mapper.source_indexes[k],line.shape)
 agrees=(i,j)==actual
 records.append({'map_row':int(row),'map_column':int(col),'native_nearest_radar_row':i,
                 'native_nearest_radar_column':j,'mesh_selected_radar_row':int(actual[0]),
                 'mesh_selected_radar_column':int(actual[1]),'agrees':bool(agrees),
                 'native_line_minus_selected_center_px':float(native_line-line[actual]),
                 'native_sample_minus_selected_center_px':float(native_sample-sample[actual])})
 np.testing.assert_allclose(range_map[row,col],total_range[actual]*grid.range_pixel_spacing,rtol=2e-6,atol=1e-6)
 np.testing.assert_allclose(az_map[row,col],total_az[actual]*azscale[actual],rtol=2e-6,atol=1e-6)
report={'samples':records,'passed':all(p['agrees'] for p in records),
        'criterion':'Exact selected sparse radar slot agrees with native geo2rdr/DEM lookup; mapped TOTAL physical fields equal that source observation',
        'method':'Map pixel center -> DEM height -> native zero-Doppler radar coordinates -> nearest original matching-grid center'}
write_json('docs/geocoding-strategies/native-map-sampling.json',report)
print(json.dumps(report,indent=2))
assert report['passed'],'Nearest radar sample disagrees with independent native geometry'
