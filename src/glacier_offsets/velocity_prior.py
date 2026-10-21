"""Local horizontal velocity samples converted to reference-radar search priors.

No download, filling, clipping or motion-compensated SLC resampling. NetCDF4
support is deliberately restricted to CF rectilinear (y,x) component grids.
"""
from pathlib import Path
import logging
import h5py
import numpy as np
from osgeo import gdal
from pyproj import CRS, Transformer, Geod
from .products import checksum

GEOD = Geod(ellps='WGS84')
REASONS = '1 invalid reference geometry; 2 missing velocity; 4 radar projection failure; 8 integer overflow; 16 secondary search outside image'


def scalar(value):
    array = np.asarray(value)
    if array.size == 1:
        value = array.ravel()[0]
        if isinstance(value, bytes): return value.decode()
        if isinstance(value, np.generic): return value.item()
        return value
    return array.tolist()


def unit_name(value):
    return {'meter/year':'m/year', 'meters/year':'m/year', 'm/yr':'m/year',
            'meter/day':'m/day', 'meters/day':'m/day', 'meter/second':'m/s'}.get(value, value)


def validate_velocity(settings):
    if not isinstance(settings, dict): raise ValueError('gross_offsets.velocity must be a mapping')
    unknown = set(settings) - {'vx','vy','basis','units','year_days','map_units','interpolation','product','version'}
    if unknown: raise ValueError(f'Unknown velocity options: {sorted(unknown)}')
    for key in ('vx','vy'):
        source = settings.get(key)
        if not isinstance(source, dict) or set(source)-{'path','variable','band'}:
            raise ValueError(f'velocity.{key} needs path and variable (NetCDF) or band (GDAL raster)')
        if not isinstance(source.get('path'),str) or not Path(source['path']).is_absolute() or not Path(source['path']).is_file():
            raise ValueError(f'velocity.{key}.path must be an existing absolute local file')
        if ('variable' in source) == ('band' in source): raise ValueError(f'velocity.{key} must declare exactly one variable or band')
        if 'variable' in source and (not isinstance(source['variable'],str) or not source['variable']): raise ValueError('Velocity variable must be named')
        if 'band' in source and (type(source['band']) is not int or source['band']<1): raise ValueError('Velocity band must be a positive integer')
    if settings.get('basis') not in ('east_north','map_xy'): raise ValueError('velocity.basis must be east_north or map_xy')
    if settings.get('units') not in ('m/year','m/day','m/s'): raise ValueError('velocity.units must be m/year, m/day or m/s')
    if settings['units']=='m/year' and (type(settings.get('year_days')) not in (int,float) or not np.isfinite(settings['year_days']) or settings['year_days']<=0):
        raise ValueError('m/year requires an explicit positive velocity.year_days')
    if settings['basis']=='map_xy' and settings.get('map_units') not in ('ground','projected'):
        raise ValueError('map_xy requires velocity.map_units: ground (physical metres along map axes) or projected (coordinate rates)')
    if settings.get('interpolation') not in ('nearest','bilinear'): raise ValueError('velocity.interpolation must be nearest or bilinear')
    for key in ('product','version'):
        if not isinstance(settings.get(key),str) or not settings[key].strip(): raise ValueError(f'velocity.{key} must be declared')
    return settings


def velocity_provenance(settings):
    hashes = {p: checksum(p) for p in sorted({settings[k]['path'] for k in ('vx','vy')})}
    return {k: settings[k] | {'sha256': hashes[settings[k]['path']]} for k in ('vx','vy')}


def sample_component(source, lon, lat, interpolation, units):
    """Read just the needed rectangular subset, honoring metadata masks/nodata.

    Bilinear sampling requires every neighbor with nonzero weight to be valid.
    Neither method extends beyond pixel-center coverage or fills nodata holes.
    """
    handle = None
    try:
        if 'variable' in source:
            handle = h5py.File(source['path'],'r')
            data = handle[source['variable']]
            x,y = handle['x'][:],handle['y'][:]
            if data.shape != (len(y),len(x)) or min(len(x),len(y))<2:
                raise ValueError('CF velocity must have rectilinear (y,x) dimensions')
            if 'DIMENSION_LIST' in data.attrs:
                dims = [handle[refs[0]].name.rsplit('/',1)[-1] for refs in data.attrs['DIMENSION_LIST']]
                if dims != ['y','x']: raise ValueError('CF velocity dimension order must be (y,x)')
            dx,dy = x[1]-x[0],y[1]-y[0]
            if dx==0 or dy==0 or not np.allclose(np.diff(x),dx) or not np.allclose(np.diff(y),dy):
                raise ValueError('CF velocity x/y must be uniformly spaced pixel-center coordinates')
            mapping = scalar(data.attrs.get('grid_mapping',''))
            if not mapping or mapping not in handle: raise ValueError('CF velocity must declare its grid_mapping CRS')
            crs = CRS.from_cf({k:scalar(v) for k,v in handle[mapping].attrs.items()})
            xx,yy = Transformer.from_crs(4326,crs,always_xy=True).transform(lon,lat)
            col,row = (xx-x[0])/dx,(yy-y[0])/dy
            attrs = {k:scalar(v) for k,v in data.attrs.items() if k in ('units','_FillValue','missing_value','scale_factor','add_offset')}
            nodata = [attrs[k] for k in ('_FillValue','missing_value') if k in attrs]
            read = lambda y0,y1,x0,x1: np.asarray(data[y0:y1,x0:x1],dtype=float)
            mask_read = None
        else:
            gdal.UseExceptions();handle = gdal.Open(source['path'])
            band = handle.GetRasterBand(source['band'])
            crs = CRS.from_wkt(handle.GetProjection())
            xx,yy = Transformer.from_crs(4326,crs,always_xy=True).transform(lon,lat)
            gt = handle.GetGeoTransform();inv = gdal.InvGeoTransform(gt)
            col = inv[0]+inv[1]*xx+inv[2]*yy-.5
            row = inv[3]+inv[4]*xx+inv[5]*yy-.5
            data = np.empty((handle.RasterYSize,0))  # shape below without loading the mosaic
            shape = (handle.RasterYSize,handle.RasterXSize)
            attrs = {'units':band.GetUnitType(),'scale_factor':band.GetScale() or 1.,'add_offset':band.GetOffset() or 0.}
            nodata = [] if band.GetNoDataValue() is None else [band.GetNoDataValue()]
            read = lambda y0,y1,x0,x1: band.ReadAsArray(x0,y0,x1-x0,y1-y0).astype(float)
            mask_read = lambda y0,y1,x0,x1: band.GetMaskBand().ReadAsArray(x0,y0,x1-x0,y1-y0)!=0
        if attrs.get('units') and unit_name(attrs['units']) != units: raise ValueError(f'Velocity units metadata {attrs["units"]!r} differs from configured {units!r}')
        if 'variable' in source: shape = data.shape
        # CRS round trips may put an exact center 1e-10 pixels across a boundary.
        # Snap numerical roundoff only; this does not fill holes or extend coverage.
        row=np.where(np.abs(row-np.rint(row))<1e-7,np.rint(row),row)
        col=np.where(np.abs(col-np.rint(col))<1e-7,np.rint(col),col)
        good = np.isfinite(row)&np.isfinite(col)&(row>=0)&(col>=0)&(row<=shape[0]-1)&(col<=shape[1]-1)
        result = np.full(lon.shape,np.nan)
        if good.any():
            rr,cc = row[good],col[good]
            y0,x0 = int(np.floor(rr.min())),int(np.floor(cc.min()))
            y1,x1 = min(shape[0],int(np.ceil(rr.max()))+1),min(shape[1],int(np.ceil(cc.max()))+1)
            block = read(y0,y1,x0,x1)
            valid = np.isfinite(block)
            for value in nodata: valid &= block!=value
            if mask_read is not None: valid &= mask_read(y0,y1,x0,x1)
            block = block*attrs.get('scale_factor',1.)+attrs.get('add_offset',0.)
            block[~valid] = np.nan
            rr,cc = rr-y0,cc-x0
            if interpolation=='nearest':
                values = block[np.floor(rr+.5).astype(int),np.floor(cc+.5).astype(int)]
            else:
                r0,c0 = np.floor(rr).astype(int),np.floor(cc).astype(int)
                fr,fc = rr-r0,cc-c0
                values = np.zeros(rr.shape)
                for dr,dc,weight in [(0,0,(1-fr)*(1-fc)),(1,0,fr*(1-fc)),(0,1,(1-fr)*fc),(1,1,fr*fc)]:
                    selected = block[np.minimum(r0+dr,block.shape[0]-1),np.minimum(c0+dc,block.shape[1]-1)]
                    values += np.where(weight>1e-12,selected*weight,0.)
            result[good] = values
        metadata = {'crs_wkt':crs.to_wkt(),'nodata':nodata,'source_units':attrs.get('units'), 'shape':list(shape)}
        return result,crs,metadata
    except (KeyError,OSError) as exc:
        raise ValueError(f'Cannot read velocity component {source}: {exc}') from exc
    finally:
        if isinstance(handle,h5py.File): handle.close()


def displacement_enu(vx,vy,lon,lat,crs,settings,interval_seconds):
    """Rotate map axes locally; explicitly distinguish ground from projected metres."""
    if not np.isfinite(interval_seconds) or interval_seconds<=0: raise ValueError('Positive exact pair interval required')
    seconds = {'m/s':1.,'m/day':86400.,'m/year':settings.get('year_days',0)*86400.}[settings['units']]
    dx,dy = vx*interval_seconds/seconds,vy*interval_seconds/seconds
    if settings['basis']=='east_north': return dx,dy
    if not crs.is_projected or any(a.unit_name!='metre' for a in crs.axis_info[:2]): raise ValueError('map_xy requires a projected CRS in metres')
    forward = Transformer.from_crs(4326,crs,always_xy=True)
    inverse = Transformer.from_crs(crs,4326,always_xy=True)
    x,y = forward.transform(lon,lat)
    if settings['map_units']=='projected':
        moved_lon,moved_lat = inverse.transform(x+dx,y+dy)
        az,_,distance = GEOD.inv(lon,lat,moved_lon,moved_lat)
        return distance*np.sin(np.deg2rad(az)),distance*np.cos(np.deg2rad(az))
    # Unit physical-ground vectors tangent to each positive map coordinate axis.
    lx,ly = inverse.transform(x+1.,y);mx,my = inverse.transform(x,y+1.)
    ax,_,_ = GEOD.inv(lon,lat,lx,ly);ay,_,_ = GEOD.inv(lon,lat,mx,my)
    ax,ay = np.deg2rad(ax),np.deg2rad(ay)
    if np.any(np.isfinite(ax)&np.isfinite(ay)&(np.abs(np.cos(ax-ay))>1e-5)):
        raise ValueError('ground map_xy requires locally orthogonal map axes; use projected rates for other CRSs')
    return dx*np.sin(ax)+dy*np.sin(ay),dx*np.cos(ax)+dy*np.cos(ay)


def radar_prediction(lon,lat,height,east,north,grid,orbit,valid):
    """Difference of native zero-Doppler geo2rdr positions, constant ellipsoid height.

    Base and moved positions use the SAME reference orbit/grid, canceling small
    inverse-geometry interpolation error. Up velocity is assumed zero.
    """
    import isce3
    moved_lon,moved_lat,_ = GEOD.fwd(lon,lat,np.rad2deg(np.arctan2(east,north)),np.hypot(east,north))
    prediction = np.full((*lon.shape,2),np.nan)
    success = valid.copy()
    for index in zip(*np.where(valid)):
        try:
            def solve(x,y):
                return isce3.geometry.geo2rdr([np.deg2rad(x),np.deg2rad(y),float(height[index])],
                    isce3.core.Ellipsoid(),orbit,isce3.core.LUT2d(),grid.wavelength,grid.lookside)
            t0,r0 = solve(lon[index],lat[index]);t1,r1 = solve(moved_lon[index],moved_lat[index])
            prediction[index] = [(t1-t0)*grid.prf,(r1-r0)/grid.range_pixel_spacing]
            success[index] = np.isfinite(prediction[index]).all()
        except RuntimeError:
            success[index] = False
    return prediction,success


def round_predictions(prediction,valid):
    """Nearest integer, half ties away from zero; never clip overflow."""
    rounded = np.copysign(np.floor(np.abs(prediction)+.5),prediction)
    good = valid&np.all(np.isfinite(rounded)&(rounded>=-(2**31)+1)&(rounded<2**31),axis=-1)
    output = np.full(prediction.shape,np.iinfo(np.int32).min,dtype=np.int32)
    output[good] = rounded[good].astype(np.int32)
    return output,good


def prepare_prior(cfg,manifest,grid,native_pair=None):
    from .geocode import center_geometry
    from .inputs import select_pair
    settings = validate_velocity(cfg['gross_offsets']['velocity'])
    lon,lat,height,_,geometry_valid = center_geometry(manifest['coreg'],*grid.centers)
    vx,crs,mx = sample_component(settings['vx'],lon,lat,settings['interpolation'],settings['units'])
    vy,crsy,my = sample_component(settings['vy'],lon,lat,settings['interpolation'],settings['units'])
    if not crs.equals(crsy): raise ValueError('Velocity components must use the same CRS')
    # Missing/layover geometry is never passed to orbit projection or native Ampcor.
    covered = np.isfinite(vx)&np.isfinite(vy)
    east,north = displacement_enu(vx,vy,lon,lat,crs,settings,manifest['interval_seconds'])
    native = (native_pair if native_pair is not None else select_pair(cfg))[0].native
    radar_grid = native.as_isce3_radargrid()
    expected = manifest['coreg']['reference_grid']
    for key in ('prf','starting_range','range_pixel_spacing','length','width','sensing_start'):
        if not np.isclose(getattr(radar_grid,key),expected[key],rtol=1e-10,atol=1e-8): raise ValueError(f'Prior reference radar grid mismatch: {key}')
    predicted,projected = radar_prediction(lon,lat,height,east,north,radar_grid,native.orbit,geometry_valid&covered)
    integer,valid = round_predictions(predicted,projected)
    reason = (~geometry_valid).astype('uint8') | ((~covered).astype('uint8')<<1)
    reason |= ((geometry_valid&covered&~projected).astype('uint8')<<2)
    reason |= ((projected&~valid).astype('uint8')<<3)
    logging.info('Velocity prior: %d/%d covered and projectable windows',valid.sum(),valid.size)
    return {'vx':vx,'vy':vy,'east_displacement_m':east,'north_displacement_m':north,
            'longitude_deg':lon,'latitude_deg':lat,'height_m':height,'predicted_px':predicted,
            'integer_px':integer,'valid':valid,'rejection_flags':reason,
            'metadata':{'settings':settings,'components':{'vx':mx,'vy':my},'interval_seconds':manifest['interval_seconds'],
                        'vertical_assumption':'zero vertical velocity; constant WGS84 ellipsoid height during horizontal motion',
                        'radar_projection':'native ISCE3 geo2rdr moved-minus-base, same reference orbit/grid, zero Doppler',
                        'rounding':'nearest integer; half ties away from zero; int32 minimum is missing sentinel',
                        'missing_policy':'mask; no zero fallback, extrapolation, clipping or prior subtraction'}}
