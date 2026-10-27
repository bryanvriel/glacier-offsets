"""Geometry-based geocoding of sparse TOTAL offsets with explicit resampling/masks."""
import json
from pathlib import Path
import h5py
import numpy as np
from osgeo import gdal, osr
from pyproj import CRS, Transformer, Geod
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay
from .products import identity, write_json
from .config import load_defaults, merge_config


def sample_geometry(path,line,sample,order=1):
    """Bilinear sampling at zero-based full-resolution radar pixel centers."""
    gdal.UseExceptions()
    from osgeo import gdal_array
    from .raster_io import bounded_raster_cache
    ds=gdal.Open(str(path))
    line,sample=np.broadcast_arrays(line,sample)
    rows,cols=line.ravel(),sample.ravel()
    dtype=gdal_array.GDALTypeCodeToNumericTypeCode(ds.GetRasterBand(1).DataType)
    # Match scipy's constant-mode conversion even for integer mask rasters.
    fill=map_coordinates(np.zeros((1,1),dtype=dtype),np.array([[-1.],[-1.]]),
                order=order,mode='constant',cval=np.nan,prefilter=False)[0]
    out=np.full(rows.shape,fill,dtype=dtype)
    good=np.isfinite(rows)&np.isfinite(cols)&(rows>=0)&(rows<=ds.RasterYSize-1)&(cols>=0)&(cols<=ds.RasterXSize-1)
    indexes=np.flatnonzero(good)
    tiles=np.floor(rows[indexes]/32).astype('int64')
    with bounded_raster_cache():
        for tile in np.unique(tiles):
            selected=indexes[tiles==tile]
            y0=int(np.floor(rows[selected].min()));y1=int(np.ceil(rows[selected].max()))+1
            x0=int(np.floor(cols[selected].min()));x1=int(np.ceil(cols[selected].max()))+1
            block=ds.ReadAsArray(x0,y0,x1-x0,y1-y0)
            coords=np.array([rows[selected]-y0,cols[selected]-x0])
            out[selected]=map_coordinates(block,coords,order=order,mode='constant',cval=np.nan,prefilter=False)
    return out.reshape(line.shape)


def center_geometry(coreg,line,sample):
    if 'assembly_sources' in coreg:
        from .assembly import center_geometry as assembled_geometry
        return assembled_geometry(coreg,line,sample)
    x=sample_geometry(coreg['geometry']['x'],line,sample)
    y=sample_geometry(coreg['geometry']['y'],line,sample)
    transformer=Transformer.from_crs(coreg['geometry_crs'],4326,always_xy=True)
    lon,lat=transformer.transform(x,y)
    # Ground spacing from neighboring zero-Doppler, constant-slant-range DEM intersections.
    xm=sample_geometry(coreg['geometry']['x'],line-0.5,sample)
    ym=sample_geometry(coreg['geometry']['y'],line-0.5,sample)
    xp=sample_geometry(coreg['geometry']['x'],line+0.5,sample)
    yp=sample_geometry(coreg['geometry']['y'],line+0.5,sample)
    lonm,latm=transformer.transform(xm,ym);lonp,latp=transformer.transform(xp,yp)
    _,_,scale=Geod(ellps='WGS84').inv(lonm,latm,lonp,latp)
    z=sample_geometry(coreg['geometry']['z'],line,sample)
    lay=sample_geometry(coreg['geometry']['layover_shadow_mask'],line,sample)
    valid=np.isfinite(lon)&np.isfinite(lat)&np.isfinite(z)&np.isfinite(scale)&(scale>0)&(lay==0)
    # Interpolation of a categorical mask can only equal zero when all weighted neighbors are zero.
    return lon,lat,z,scale,valid


def physical_fields(total_range,total_azimuth,range_spacing,azimuth_scale,interval_seconds, *, velocities=True):
    interval_seconds=np.asarray(interval_seconds)
    if range_spacing<=0 or np.any(interval_seconds<=0) or np.isinf(interval_seconds).any() or (interval_seconds.ndim==0 and not np.isfinite(interval_seconds)):
        raise ValueError('Positive reference range spacing and exact pair interval required')
    days=interval_seconds/86400
    rg=total_range*range_spacing;az=total_azimuth*azimuth_scale
    fields = {'range_displacement_m':rg,'azimuth_displacement_m':az}
    if velocities:
        fields.update(range_velocity_m_per_day=rg/days, azimuth_velocity_m_per_day=az/days)
    return fields


def formal_uncertainties(covariance, range_spacing, azimuth_scale, interval_seconds, mapper, *, velocities=True):
    """Convert native variances, then interpolate with squared independent-vertex weights.

    The caller excludes invalid/sentinel covariance vertices in mapper. Native
    component order is azimuth variance, range variance, cross covariance.
    These are formal estimates, excluding systematic bias and correlated errors.
    """
    interval_seconds=np.asarray(interval_seconds)
    if range_spacing <= 0 or np.any(interval_seconds <= 0) or np.isinf(interval_seconds).any() or (interval_seconds.ndim==0 and not np.isfinite(interval_seconds)):
        raise ValueError('Positive reference range spacing and exact pair interval required')
    fields = {}
    for axis, k, scale in [('azimuth', 0, azimuth_scale), ('range', 1, range_spacing)]:
        sigma = np.sqrt(mapper.map(covariance[..., k] * scale**2, variance=True))
        fields[f'{axis}_displacement_sigma_m'] = sigma
        # Preserve the established scalar path. Variable source intervals must
        # scale each variance BEFORE interpolation, like the velocity itself.
        if velocities:
            fields[f'{axis}_velocity_sigma_m_per_day'] = (sigma / (interval_seconds / 86400) if interval_seconds.ndim==0
                else np.sqrt(mapper.map(covariance[..., k] * scale**2 / (interval_seconds / 86400)**2, variance=True)))
    return fields


def map_grid(x, y, spacing):
    """A target grid derived from reference geometry, independent of pair validity."""
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        raise ValueError('No finite reference geometry for the common map grid')
    xmin = np.floor(x[finite].min()/spacing)*spacing
    xmax = np.ceil(x[finite].max()/spacing)*spacing
    ymin = np.floor(y[finite].min()/spacing)*spacing
    ymax = np.ceil(y[finite].max()/spacing)*spacing
    width = max(1, int(round((xmax-xmin)/spacing)))
    height = max(1, int(round((ymax-ymin)/spacing)))
    if width*height > 20_000_000:
        raise ValueError('Geogrid too large; increase spacing_m')
    return {'shape': [height, width], 'transform': [float(xmin), float(spacing), 0, float(ymax), 0, -float(spacing)]}


class ConservativeMapper:
    """Linear interpolation only inside triangles joining neighboring valid radar centers.

    Triangulate all finite geometry (including invalid observations); discard
    triangles that cross a missing row/column or have ANY invalid vertex.
    This never bridges quality holes by triangulating only good samples.
    """
    def __init__(self,x,y,valid,spacing,neighbor_only=True,target_grid=None):
        if x.shape!=y.shape or x.shape!=valid.shape or x.ndim!=2:
            raise ValueError('Geometry and validity must share a two-dimensional sparse grid')
        geometry=np.isfinite(x)&np.isfinite(y)
        indexes=np.flatnonzero(geometry)
        if indexes.size<3:raise ValueError('Insufficient finite geometry for geocoding')
        points=np.column_stack((x.ravel()[indexes],y.ravel()[indexes]))
        tri=Delaunay(points)
        self.vertices=indexes[tri.simplices]
        rows,cols=np.unravel_index(self.vertices,x.shape)
        allowed=np.all(valid.ravel()[self.vertices],axis=1)
        if neighbor_only:
            allowed&=(rows.max(1)-rows.min(1)<=1)&(cols.max(1)-cols.min(1)<=1)
        target_grid = map_grid(x,y,spacing) if target_grid is None else target_grid
        height,width=target_grid['shape']
        if width*height>20_000_000:raise ValueError('Geogrid too large; increase spacing_m')
        self.shape=(height,width);self.transform=tuple(target_grid['transform'])
        xmin,dx,_,ymax,_,dy=self.transform
        xx,yy=np.meshgrid(xmin+(np.arange(width)+0.5)*dx,ymax+(np.arange(height)+0.5)*dy)
        target=np.column_stack((xx.ravel(),yy.ravel()))
        simplices=tri.find_simplex(target)
        inside=simplices>=0
        self.positions=np.flatnonzero(inside&allowed[np.maximum(simplices,0)])
        self.simplices=simplices[self.positions]
        transform=tri.transform[self.simplices]
        delta=target[self.positions]-transform[:,2]
        weights=np.einsum('nij,nj->ni',transform[:,:2],delta)
        self.weights=np.column_stack((weights,1-weights.sum(1)))
        self.valid=np.zeros(width*height,bool);self.valid[self.positions]=True;self.valid=self.valid.reshape(self.shape)

    def map(self,array,variance=False):
        values=array.ravel()[self.vertices[self.simplices]]
        weights=self.weights**2 if variance else self.weights
        result=np.full(np.prod(self.shape),np.nan,dtype=np.float32)
        result[self.positions]=np.sum(values*weights,axis=1)
        return result.reshape(self.shape)


class RadarNearestMapper(ConservativeMapper):
    """Nearest source sample in the original sparse radar grid, then apply its mask.

    Invert the complete finite-center geometry mesh to fractional radar row/
    column. Round each coordinate to its nearest source center (half ties upward).
    Invalid samples retain their own regions; never choose the nearest GOOD sample.
    Support remains inside the finite center mesh, without edge extrapolation.
    """
    def __init__(self,x,y,valid,spacing,target_grid=None):
        # Projected-space Delaunay triangles can span more than one radar column
        # even on a regular sheared grid. Geometry inversion uses the full mesh;
        # motion is copied from the rounded original radar slot, never interpolated
        # across quality holes or selected from a tree of accepted observations.
        super().__init__(x,y,np.ones_like(valid,dtype=bool),spacing,neighbor_only=False,target_grid=target_grid)
        vertices=self.vertices[self.simplices]
        rows,cols=np.unravel_index(vertices,x.shape)
        rr=np.floor(np.sum(rows*self.weights,axis=1)+.5).astype(int)
        cc=np.floor(np.sum(cols*self.weights,axis=1)+.5).astype(int)
        indexes=np.ravel_multi_index((rr,cc),x.shape)
        selected=valid.ravel()[indexes]&np.isfinite(x.ravel()[indexes])&np.isfinite(y.ravel()[indexes])
        self.positions=self.positions[selected]
        self.source_indexes=indexes[selected]
        self.valid=np.zeros(np.prod(self.shape),bool)
        self.valid[self.positions]=True
        self.valid=self.valid.reshape(self.shape)

    def map(self,array,variance=False):
        # One source with unit weight: marginal variance is copied, not averaged.
        result=np.full(np.prod(self.shape),np.nan,dtype=np.float32)
        result[self.positions]=array.ravel()[self.source_indexes]
        return result.reshape(self.shape)


def make_mapper(x,y,valid,spacing,resampling,target_grid=None):
    return (RadarNearestMapper if resampling=='nearest' else ConservativeMapper)(x,y,valid,spacing,target_grid=target_grid)


def write_geotiff(path,data,transform,epsg,metadata):
    ds=gdal.GetDriverByName('GTiff').Create(str(path),data.shape[1],data.shape[0],1,gdal.GDT_Float32,
                                         options=['COMPRESS=DEFLATE','TILED=YES'])
    crs=osr.SpatialReference();crs.ImportFromEPSG(epsg)
    ds.SetProjection(crs.ExportToWkt());ds.SetGeoTransform(transform)
    ds.GetRasterBand(1).SetNoDataValue(float('nan'))
    ds.GetRasterBand(1).WriteArray(data)
    ds.SetMetadata({k:str(v) for k,v in metadata.items()});ds=None


def validate_geocoding(settings: dict) -> tuple[int, float]:
    if not isinstance(settings,dict):raise ValueError('geocoding must be a YAML mapping')
    unknown=set(settings)-{'epsg','spacing_m','resampling','mask'}
    if unknown:raise ValueError(f'Unknown geocoding options: {sorted(unknown)}')
    epsg=settings.get('epsg');spacing=settings.get('spacing_m')
    if type(epsg) is not int or type(spacing) not in (float,int) or not np.isfinite(spacing) or spacing<=0:
        raise ValueError('geocoding.epsg and positive spacing_m are required')
    if not CRS.from_epsg(epsg).is_projected or CRS.from_epsg(epsg).axis_info[0].unit_name!='metre':
        raise ValueError('Geocoding requires a projected output CRS with metre units')
    defaults=load_defaults()['geocoding']
    if settings.get('resampling',defaults['resampling']) not in ('nearest','linear'):
        raise ValueError('geocoding.resampling must be nearest or linear')
    if settings.get('mask',defaults['mask']) not in ('quality','input'):
        raise ValueError('geocoding.mask must be quality or input')
    return epsg,float(spacing)


def geocoding_specification(cfg: dict,manifest: dict) -> dict:
    # Cache the effective policy, so omitted defaults cannot reuse maps produced
    # under a different default. Explicit and equivalent implicit policies agree.
    settings=merge_config(load_defaults()['geocoding'],cfg.get('geocoding',{}))
    spec = {'match':manifest['match_identity'],'settings':settings,'method':'radar-grid-resampling-v5-retention-defaults'}
    if 'map_grid' in manifest:
        spec['target_grid'] = manifest['map_grid']
    return spec


def geocode(cfg: dict,root: Path,manifest: dict,force: bool = False):
    if not manifest.get('match_identity'):raise ValueError('Run matching before geocoding')
    settings=merge_config(load_defaults()['geocoding'],cfg.get('geocoding',{}))
    epsg,spacing=validate_geocoding(settings)
    resampling=settings['resampling'];mask_policy=settings['mask']
    velocities = 'stack_pair' not in manifest
    fingerprint=identity(geocoding_specification(cfg,manifest))
    directory=root/'geo';directory.mkdir(exist_ok=True)
    if not force and manifest.get('geocode_identity')==fingerprint and (directory/'geocoding.json').is_file():return
    manifest.pop('geocode_identity',None)
    for stage in ('geocode','diagnostics'):manifest.get('stages',{}).pop(stage,None)
    write_json(root/'manifest.json',manifest)
    with h5py.File(root/'radar/offsets.h5','r') as h:
        line,sample=h['center_line'][:],h['center_sample'][:]
        lon,lat,height,azscale,geometry_valid=center_geometry(manifest['coreg'],line,sample)
        quality=h['valid'][:]
        total_range,total_az=h['total/range_px'][:],h['total/azimuth_px'][:]
        # Keep input-invalid, nonfinite and search-boundary failures excluded even
        # when low SNR/peak observations are explicitly requested for inspection.
        input_eligible=h['input_valid'][:]&((h['rejection_flags'][:]&7)==0)
        finite=np.isfinite(total_range)&np.isfinite(total_az)
        valid=(quality if mask_policy=='quality' else input_eligible)&geometry_valid&finite
        interval=h['interval_seconds'][:] if 'interval_seconds' in h else manifest['interval_seconds']
        assembled = ({name:h[name][:] for name in ('source_burst_index','seam_crossing')}
                     if 'source_burst_index' in h else None)
        fields=physical_fields(total_range,total_az,manifest['coreg']['reference_grid']['range_pixel_spacing'],azscale,interval,
                               velocities=velocities)
        covariance=h['covariance'][:]
        qa={'snr':h['snr'][:],'correlation_peak':h['correlation_peak'][:]}
    x,y=Transformer.from_crs(4326,epsg,always_xy=True).transform(lon,lat)
    target_grid=manifest.get('map_grid')
    mapper=make_mapper(x,y,valid,spacing,resampling,target_grid=target_grid)
    quality_mapper=make_mapper(x,y,valid&quality,spacing,resampling,target_grid=target_grid)
    method=('nearest original sparse radar sample; mask after selection; no extrapolation' if resampling=='nearest'
            else 'linear neighboring-center triangles, all selected vertices valid, no extrapolation')
    metadata={'axis_sign':'secondary-minus-reference on geometrically coregistered reference grid',
              'motion':'TOTAL measured gross + residual; radar observables, not east/north',
              'geocoding':method,'resampling':resampling,'source_mask':mask_policy,
              'quality_mask':'quality_valid_mask.tif; passes configured matching checks, not an independent accuracy assessment',
              'matching_quality_thresholds_json':json.dumps({key:manifest.get('matching',{}).get('matching',{}).get(key)
                                                            for key in ('snr_threshold','correlation_threshold')},sort_keys=True),
              'interval_seconds':manifest['interval_seconds'],'match_identity':manifest['match_identity'],
              'burst_id':cfg['inputs']['burst_id'],'polarization':cfg['inputs']['polarization'],
              'acquisition_times_UTC':json.dumps(manifest['acquisition_times']),
              'reference_grid_json':json.dumps(manifest['coreg']['reference_grid'],sort_keys=True),
              'input_provenance_json':json.dumps(manifest['coreg_specification']['inputs'],sort_keys=True),
              'software_versions_json':json.dumps(manifest['software']['packages'],sort_keys=True),
              'provenance_manifest':str(root/'manifest.json')}
    if 'stack_pair' in manifest:
        metadata.update(geometry_reference_date=manifest['stack_pair']['geometry_reference_date'],
                        geometry_reference_burst_times_json=json.dumps(manifest['stack_pair']['geometry_reference_burst_times']),
                        geometry_reference_inputs_json=json.dumps(manifest['stack_pair']['geometry_reference_inputs'],sort_keys=True),
                        matched_acquisition_dates=json.dumps(manifest['stack_pair']['acquisition_dates']),
                        axis_sign='later-minus-earlier on the pinned stack-reference radar grid')
    if assembled is not None:
        metadata.pop('interval_seconds',None)
        metadata.pop('acquisition_times_UTC',None)
        metadata.update(source_bursts_json=json.dumps([
            {key:e[key] for key in ('burst_id','interval_seconds','acquisition_times','offset')}
            for e in manifest['assembly']['sources']],sort_keys=True),
            interval_policy='per-source interval applied before geocoding; see pair_interval_seconds.tif',
            assembly=manifest['assembly']['method'])
    metadata['gross_offsets_json']=json.dumps(manifest.get('matching',{}).get('gross_offsets',{}),sort_keys=True)
    if 'velocity_provenance' in manifest.get('matching',{}):
        metadata['velocity_prior_sources_json']=json.dumps(manifest['matching']['velocity_provenance'],sort_keys=True)
    for name,array in (fields|qa).items():
        units='m/day' if 'velocity' in name else 'm' if 'displacement' in name else 'dimensionless'
        write_geotiff(directory/f'{name}.tif',mapper.map(array),mapper.transform,epsg,metadata|{'units':units,'quantity':name})
    write_geotiff(directory/'valid_mask.tif',mapper.valid.astype('float32'),mapper.transform,epsg,
                  metadata|{'units':'0 nodata, 1 selected observation support','quantity':'selected_data_coverage'})
    write_geotiff(directory/'quality_valid_mask.tif',quality_mapper.valid.astype('float32'),mapper.transform,epsg,
                  metadata|{'units':'0 not quality accepted, 1 quality accepted','quantity':'matching_quality_coverage'})
    uncertainty_valid=valid&np.all(np.isfinite(covariance[...,:2])&(covariance[...,:2]>=0)&(covariance[...,:2]<99),axis=-1)
    umapper=make_mapper(x,y,uncertainty_valid,spacing,resampling,target_grid=target_grid)
    uncertainty_fields = formal_uncertainties(covariance, manifest['coreg']['reference_grid']['range_pixel_spacing'],
                                            azscale, interval, umapper, velocities=velocities)
    for name, array in uncertainty_fields.items():
        write_geotiff(directory/f'{name}.tif',array,umapper.transform,epsg,
                      metadata|{'quantity':name, 'units':'m/day' if 'velocity' in name else 'm',
                                'uncertainty':'native formal covariance; independent vertices assumed; excludes bias and spatial correlations'})
    if assembled is not None:
        source=assembled['source_burst_index'];seam=assembled['seam_crossing']
        if resampling=='nearest':
            mapped_source=mapper.map(source);mapped_seam=mapper.map(seam)
        else:
            vertices=mapper.vertices[mapper.simplices]
            ids=source.ravel()[vertices]
            same=np.all(ids==ids[:,:1],axis=1)
            mapped_source=np.full(mapper.shape,np.nan,dtype='float32')
            mapped_source.ravel()[mapper.positions]=np.where(same,ids[:,0],0)
            mapped_seam=np.full(mapper.shape,np.nan,dtype='float32')
            mapped_seam.ravel()[mapper.positions]=np.any(seam.ravel()[vertices],axis=1)|~same
        for name,data,units in [('source_burst_index',mapped_source,'1-based source list; 0 mixed sources under linear resampling'),
                                 ('seam_crossing_mask',mapped_seam,'1 chip/search crosses source boundary; 0 single source'),
                                 ('pair_interval_seconds',mapper.map(interval),'seconds; interpolated source intervals for linear resampling')]:
            write_geotiff(directory/f'{name}.tif',data,mapper.transform,epsg,metadata|{'quantity':name,'units':units})
    # Preserve each source center's actual geolocation and physical scale for audit.
    with h5py.File(directory/'center_geometry.h5','w') as h:
        for name,array in [('longitude_deg',lon),('latitude_deg',lat),('height_m',height),('azimuth_ground_spacing_m_per_slc_line',azscale),('geometry_valid',geometry_valid),('center_line',line),('center_sample',sample)]:h.create_dataset(name,data=array)
        if assembled is not None:
            h.create_dataset('source_burst_index',data=assembled['source_burst_index'])
            h.create_dataset('interval_seconds',data=interval)
        h.attrs['metadata_json']=json.dumps(metadata,sort_keys=True)
        h.attrs['azimuth_scale']='WGS84 geodesic separation of DEM-intersecting constant-slant-range zero-Doppler positions at center line ±0.5'
    summary={'epsg':epsg,'spacing_m':spacing,'shape':mapper.shape,'transform':mapper.transform,
             'valid_source_centers':int(valid.sum()),'valid_output_cells':int(mapper.valid.sum()),
             'method':method,'resampling':resampling,'source_mask':mask_policy,
             'quality_valid_source_centers':int((quality&geometry_valid&finite).sum()),
             'quality_valid_output_cells':int(quality_mapper.valid.sum()),
             'azimuth_scale_method':'local reference rdr2geo surface separation at fixed slant range per SLC line'}
    if assembled is not None:
        summary.update(source_burst_ids=manifest['assembly']['source_burst_ids'],
                       seam_crossing_source_centers=int((valid&assembled['seam_crossing']).sum()),
                       interval_policy='exact per-source burst-start interval; see pair_interval_seconds.tif')
    summary['product_profile'] = 'displacement-and-velocity-v1' if velocities else 'displacement-v1'
    if not velocities:
        for axis in ('range', 'azimuth'):
            for quantity in ('velocity_m_per_day', 'velocity_sigma_m_per_day'):
                (directory / f'{axis}_{quantity}.tif').unlink(missing_ok=True)
    write_json(directory/'geocoding.json',summary)
    manifest['geocode_identity']=fingerprint;manifest['geocoding']=summary;manifest['stages']['geocode']='complete'
    write_json(root/'manifest.json',manifest)
