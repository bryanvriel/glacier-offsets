"""Only native ISCE3 amplitude matching of complex radar SLC rasters."""
import json
import logging
import time
from contextlib import contextmanager
import multiprocessing
import os
import resource
import signal
import sys
import tempfile
from pathlib import Path
import h5py
import numpy as np
from osgeo import gdal
from .config import gross_values
from .environment import select_backend
from .gross_offsets import apply_gross
from .matching_grid import MatchingGrid, build_grid, valid_windows
from .products import identity, write_json

# The native CPU chunk destructor does not release its working allocations.
# Recycle a sequential worker after four calls, bounding accumulation across
# fragmented velocity priors as well as across bursts/standalone validations.
NATIVE_CALLS_PER_WORKER = 4
_plan_cache_directory = None
_worker_plan_cache = None
_worker_fftw = None
_worker_fftw_initialized = False


def _initialize_native_worker(plan_cache):
    from .raster_io import RASTER_CACHE_BYTES
    global _worker_plan_cache
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    gdal.SetCacheMax(min(gdal.GetCacheMax(), RASTER_CACHE_BYTES))
    _worker_plan_cache = plan_cache


@contextmanager
def _native_executor():
    global _plan_cache_directory
    if _plan_cache_directory is None:
        _plan_cache_directory = tempfile.TemporaryDirectory(prefix='glacier-offsets-fftw-')
    plan_cache = str(Path(_plan_cache_directory.name) / 'wisdom')
    # Spawn rather than fork: GDAL, FFTW and CUDA can own native thread state.
    with multiprocessing.get_context('spawn').Pool(
            processes=1, initializer=_initialize_native_worker,
            initargs=(plan_cache,),
            maxtasksperchild=NATIVE_CALLS_PER_WORKER) as worker:
        yield worker


def _native_worker(arguments):
    from .fftw_cache import restore, save
    global _worker_fftw, _worker_fftw_initialized
    if not _worker_fftw_initialized:
        _worker_fftw = restore(_worker_plan_cache)
        _worker_fftw_initialized = True
    args, kwargs = arguments
    result = _native_match(*args, **kwargs)
    save(_worker_fftw, _worker_plan_cache)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['effective_parameters'].update(worker_pid=os.getpid(),
        fftw_plan_cache=_worker_fftw is not None,
        worker_peak_rss_bytes=int(peak if sys.platform == 'darwin' else peak * 1024))
    return result


def native_match(reference: str | Path, secondary: str | Path, output: str | Path, params: dict, gross_cfg: dict, backend: str = 'cpu', *, prior=None, _grid=None, _gross=None, _valid_input=None, _worker=None) -> dict:
    args = (reference, secondary, output, params, gross_cfg, backend)
    kwargs = dict(prior=prior, _grid=_grid, _gross=_gross, _valid_input=_valid_input)
    if prior is not None:
        return _native_match(*args, **kwargs)  # Segment dispatch owns one recycled worker.
    if _worker is not None:
        return _worker.apply(_native_worker, ((args, kwargs),))
    with _native_executor() as worker:
        return worker.apply(_native_worker, ((args, kwargs),))


def _native_match(reference, secondary, output, params, gross_cfg, backend='cpu', *, prior=None, _grid=None, _gross=None, _valid_input=None):
    gdal.UseExceptions()
    ref,sec=(gdal.Open(str(p)) for p in (reference,secondary))
    for ds in (ref,sec):
        if ds.RasterCount!=1 or not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType):
            raise ValueError('Native CPU Ampcor requires single-band complex inputs')
    if prior is not None:
        return varying_match(reference,secondary,output,params,gross_cfg,backend,prior,ref,sec)
    gross=gross_values(gross_cfg) if _gross is None else _gross
    grid=_grid if _grid is not None else build_grid((ref.RasterYSize,ref.RasterXSize),(sec.RasterYSize,sec.RasterXSize),
                    params['window'],params['search'],params['stride'],gross,params.get('first_center'))
    valid_input=valid_windows(ref,sec,grid,gross) if _valid_input is None else _valid_input
    requested_backend=backend
    backend,cls=select_backend(backend)
    if backend == 'cpu':
        from .raster_io import require_native_mmap
        for ds in (ref,sec):
            require_native_mmap(ds)
    a=cls()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    mapping={'offsetImageName':'residual.bip','grossOffsetImageName':'gross.bip','snrImageName':'snr.bin',
             'covImageName':'covariance.bip','corrImageName':'correlation.bin'}
    for key,name in mapping.items():setattr(a,key,str(output/name))
    a.referenceImageName=str(reference);a.secondaryImageName=str(secondary)
    a.referenceImageHeight=ref.RasterYSize;a.referenceImageWidth=ref.RasterXSize
    a.secondaryImageHeight=sec.RasterYSize;a.secondaryImageWidth=sec.RasterXSize
    a.numberWindowDown,a.numberWindowAcross=grid.shape
    a.windowSizeHeight,a.windowSizeWidth=grid.window
    a.halfSearchRangeDown,a.halfSearchRangeAcross=grid.search
    a.skipSampleDown,a.skipSampleAcross=grid.stride
    a.referenceStartPixelDownStatic=int(grid.starts_line[0]);a.referenceStartPixelAcrossStatic=int(grid.starts_sample[0])
    a.numberWindowDownInChunk=min(params['batch'][0],grid.shape[0]);a.numberWindowAcrossInChunk=min(params['batch'][1],grid.shape[1])
    a.rawDataOversamplingFactor=params['raw_oversampling'];a.corrSurfaceOverSamplingFactor=params['oversampling']
    a.corrSurfaceZoomInWindow=params['zoom'];a.derampMethod=params['deramp']
    a.algorithm=0;a.corrSurfaceOverSamplingMethod=0;a.useMmap=0
    a.setupParams()
    if _gross is None:
        apply_gross(a,gross_cfg)
    else:
        a.mergeGrossOffset=0
        a.setVaryingGrossOffset(np.ascontiguousarray(gross[...,0].ravel(),dtype=np.int32),
                               np.ascontiguousarray(gross[...,1].ravel(),dtype=np.int32))
    a.checkPixelInImageRange()
    effective_parameters={name:getattr(a,name) for name in dir(a) if not name.startswith('_') and isinstance(getattr(a,name),(str,int,float))}
    start=time.perf_counter()
    try:
        a.runAmpcor()
    except RuntimeError as exc:
        if backend == 'gpu' and requested_backend == 'auto' and any(w in str(exc).lower() for w in ('cuda','device','allocation')):
            import logging
            logging.warning('CUDA execution failed (%s); rerunning native CPU Ampcor', exc)
            return _native_match(reference,secondary,output,params,gross_cfg,'cpu',_grid=grid,_gross=_gross,_valid_input=valid_input)
        raise
    elapsed=time.perf_counter()-start
    def read(name,components=1):
        data=np.fromfile(output/name,dtype=np.float32)
        expected=np.prod(grid.shape)*components
        if len(data)!=expected:raise ValueError(f'Unexpected native output length in {name}: {len(data)} != {expected}')
        return data.reshape((*grid.shape,components)) if components>1 else data.reshape(grid.shape)
    residual=read('residual.bip',2);applied=read('gross.bip',2)
    if not np.all(applied==np.array(gross)):
        raise RuntimeError('Native gross output does not match applied azimuth/range gross values')
    total=applied+residual  # add once, independently of backend
    snr=read('snr.bin');peak=read('correlation.bin');cov=read('covariance.bip',3)
    boundary=np.any(np.abs(residual)>=np.asarray(grid.search)-1/params['oversampling'],axis=-1)
    finite=np.all(np.isfinite(total),axis=-1)&np.isfinite(snr)&np.isfinite(peak)
    valid=valid_input&finite&~boundary&(snr>=params['snr_threshold'])&(peak>=params['correlation_threshold'])
    return dict(grid=grid,gross=applied,residual=residual,total=total,snr=snr,correlation_peak=peak,covariance=cov,
                input_valid=valid_input,finite=finite,search_boundary=boundary,valid=valid,backend=backend,elapsed_seconds=elapsed,effective_parameters=effective_parameters)


def varying_match(reference,secondary,output,params,cfg,backend,prior,ref,sec):
    """Run native Ampcor only on valid, bounded priors, in contiguous row segments.

    Missing priors are never replaced by zero or sent to the C++ matcher. Segments
    preserve reference chip coordinates and native row-major component ordering.
    """
    grid=prior['grid'];gross=prior['integer_px'];eligible=prior['valid'].copy()
    if gross.shape!=(*grid.shape,2) or eligible.shape!=grid.shape or gross.dtype!=np.int32:
        raise ValueError('Varying prior must contain int32 [row,column,azimuth/range] on the matching grid')
    line,sample=np.meshgrid(grid.starts_line.astype('int64'),grid.starts_sample.astype('int64'),indexing='ij')
    bounds=np.ones(grid.shape,bool)
    for axis,start,size in [(0,line,sec.RasterYSize),(1,sample,sec.RasterXSize)]:
        low=start+gross[...,axis].astype('int64')-grid.search[axis]
        bounds&=(low>=0)&(low+grid.window[axis]+2*grid.search[axis]<=size)
    prior['rejection_flags'][eligible&~bounds]|=16
    eligible&=bounds;prior['valid']=eligible
    valid_input=valid_windows(ref,sec,grid,gross)&eligible
    selected_backend=select_backend(backend)[0]
    segment_backend='auto' if backend=='auto' and selected_backend=='gpu' else selected_backend
    result={name:np.full((*grid.shape,n),np.nan,dtype='float32') if n>1 else np.full(grid.shape,np.nan,dtype='float32')
            for name,n in [('gross',2),('residual',2),('total',2),('snr',1),('correlation_peak',1),('covariance',3)]}
    result.update({key:np.zeros(grid.shape,bool) for key in ('finite','search_boundary','valid')})
    result.update(grid=grid,input_valid=valid_input,elapsed_seconds=0.,backend=selected_backend,
                  effective_parameters={'mergeGrossOffset':0,'gross_method':'setVaryingGrossOffset','segments':[]},prior=prior)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fallback=False
    if eligible.any():
        with _native_executor() as worker:
            for row in range(grid.shape[0]):
                padded=np.r_[False,eligible[row],False].astype('int8')
                starts=np.flatnonzero(np.diff(padded)==1);stops=np.flatnonzero(np.diff(padded)==-1)
                for begin,end in zip(starts,stops):
                    logging.info('Varying Ampcor row %d/%d, columns %d:%d',row+1,grid.shape[0],begin,end)
                    section=(slice(row,row+1),slice(begin,end))
                    subgrid=MatchingGrid(grid.starts_line[row:row+1],grid.starts_sample[begin:end],grid.window,grid.search,grid.stride)
                    native=native_match(reference,secondary,output/f'row-{row:04d}-col-{begin:04d}',params,cfg,segment_backend,
                                        _grid=subgrid,_gross=gross[section],_valid_input=valid_input[section],_worker=worker)
                    if selected_backend=='gpu' and native['backend']=='cpu':
                        logging.warning('CUDA segment fell back; rerunning the full varying field on CPU for consistent provenance')
                        fallback=True
                        break
                    for name in ('gross','residual','total','snr','correlation_peak','covariance','finite','search_boundary','valid'):
                        result[name][section]=native[name]
                    result['elapsed_seconds']+=native['elapsed_seconds'];result['backend']=native['backend']
                    result['effective_parameters']['segments'].append(native['effective_parameters'])
                if fallback:
                    break
    if fallback:
        return varying_match(reference,secondary,output,params,cfg,'cpu',prior,ref,sec)
    # Combined native-layout files retain missing windows as NaN, not zero priors.
    for name,file in [('gross','gross.bip'),('residual','residual.bip'),('snr','snr.bin'),('correlation_peak','correlation.bin'),('covariance','covariance.bip')]:
        result[name].tofile(output/file)
    return result


def write_offsets(path,result,metadata):
    with h5py.File(path,'w') as h:
        h.attrs['metadata_json']=json.dumps(metadata,sort_keys=True)
        h.attrs['axis_order']='azimuth/down, range/across'
        h.attrs['sign']='secondary-minus-reference feature displacement after geometric coregistration'
        if 'stack_pair' in metadata:
            h.attrs['sign']='later-minus-earlier on the pinned stack-reference radar grid'
            h.attrs['geometry_reference_date']=metadata['stack_pair']['geometry_reference_date']
            h.attrs['matched_acquisition_dates_json']=json.dumps(metadata['stack_pair']['acquisition_dates'])
        h.attrs['units']='full-resolution reference SLC pixels'
        h.attrs['mergeGrossOffset']=0
        h.attrs['total_definition']='gross + native residual, exactly once'
        for field in ('gross','residual','total'):
            for k,axis in enumerate(('azimuth','range')):
                ds=h.create_dataset(f'{field}/{axis}_px',data=result[field][...,k],compression='gzip')
                ds.attrs['units']='reference SLC pixels';ds.attrs['axis']=axis
        rejection=(~result['input_valid']).astype('uint8') | ((~result['finite']).astype('uint8')<<1) | (result['search_boundary'].astype('uint8')<<2)
        rejection |= (result['snr']<metadata['matching']['snr_threshold']).astype('uint8')<<3
        rejection |= (result['correlation_peak']<metadata['matching']['correlation_threshold']).astype('uint8')<<4
        h.create_dataset('rejection_flags',data=rejection)
        h['rejection_flags'].attrs['bits']='1 invalid input; 2 nonfinite; 4 search boundary; 8 SNR; 16 correlation peak'
        for name in ('snr','correlation_peak','covariance','input_valid','finite','search_boundary','valid'):
            h.create_dataset(name,data=result[name],compression='gzip')
        for name in ('source_burst_index','interval_seconds','seam_crossing','seam_search_known'):
            if name in result:h.create_dataset(name,data=result[name],compression='gzip')
        if 'source_burst_index' in result:
            h['source_burst_index'].attrs['source_burst_ids_json']=json.dumps(metadata['source_burst_ids'])
            h['source_burst_index'].attrs['index_origin']=1
            h['source_burst_index'].attrs['nodata']=0
            h['interval_seconds'].attrs['units']='seconds; exact interval of owning source burst'
            h['seam_crossing'].attrs['meaning']='Reference chip or secondary search contains a different source or unowned pixels'
            h['seam_search_known'].attrs['meaning']='Finite gross offsets allow secondary-search classification; otherwise seam flag covers reference chip only'
        h['snr'].attrs['meaning']='coarse correlation squared peak / surrounding mean square'
        h['correlation_peak'].attrs['meaning']='refined normalized amplitude-correlation peak'
        h['covariance'].attrs['invalid_sentinel']='99,99,0 for coarse boundary or near-singular peak curvature'
        h['covariance'].attrs['component_order']='azimuth variance, range variance, azimuth-range covariance (native estimates)'
        h['covariance'].attrs['units']='reference SLC pixels squared; native formal estimates, not absolute accuracy'
        for name,array in zip(('center_line','center_sample'),result['grid'].centers):
            ds=h.create_dataset(name,data=array);ds.attrs['index_origin']=0
        for k in ('window','search','stride'):h.attrs[k]=getattr(result['grid'],k)
        if 'prior' in result:
            from .velocity_prior import REASONS
            prior=result['prior'];group=h.create_group('prior')
            group.attrs['metadata_json']=json.dumps(prior['metadata'],sort_keys=True)
            for name in ('vx','vy','east_displacement_m','north_displacement_m','longitude_deg','latitude_deg','height_m','valid','rejection_flags'):
                group.create_dataset(name,data=prior[name],compression='gzip')
            group['rejection_flags'].attrs['bits']=REASONS
            group['vx'].attrs['units']=group['vy'].attrs['units']=prior['metadata']['settings']['units']
            for name in ('east_displacement_m','north_displacement_m','height_m'):
                group[name].attrs['units']='m'
            for name in ('longitude_deg','latitude_deg'):
                group[name].attrs['units']='degrees'
            for field in ('predicted_px','integer_px'):
                for k,axis in enumerate(('azimuth','range')):
                    ds=group.create_dataset(f'{field}/{axis}',data=prior[field][...,k],compression='gzip')
                    ds.attrs['units']='full-resolution reference SLC pixels'
                    ds.attrs['axis']=axis
                    if field=='integer_px':ds.attrs['missing_sentinel']=np.iinfo(np.int32).min
    with h5py.File(Path(path).parent/'matching_grid.h5','w') as h:
        for name,array in zip(('center_line','center_sample'),result['grid'].centers):h.create_dataset(name,data=array)
        h.attrs['metadata_json']=json.dumps(metadata,sort_keys=True)
        h.attrs['reference_grid_json']=json.dumps(metadata.get('reference_grid',{}))
        h.attrs['centers']='zero-based geometric chip centers = top-left + (window-1)/2'
        h.attrs['stride']=result['grid'].stride;h.attrs['window']=result['grid'].window


def matching_specification(cfg,manifest):
    spec={'coreg_identity':manifest['coreg_identity'],'matching':cfg['matching'],
            'gross_offsets':cfg.get('gross_offsets',{'mode':'zero'}),'implementation':'native-ampcor-v7-bounded-workers'}
    if cfg['gross_offsets']['mode']=='velocity':
        from .velocity_prior import velocity_provenance
        spec['velocity_provenance']=(manifest['velocity_provenance'] if 'velocity_provenance' in manifest
                                     else velocity_provenance(cfg['gross_offsets']['velocity']))
    return spec


def match(cfg,root,manifest,force=False,native_pair=None):
    spec=matching_specification(cfg,manifest)
    fingerprint=identity(spec)
    directory=root/'radar';directory.mkdir(exist_ok=True)
    path=directory/'offsets.h5'
    if not force and manifest.get('match_identity')==fingerprint and path.is_file():return
    manifest.pop('geocode_identity',None);manifest.pop('match_identity',None)
    for stage in ('match','geocode','diagnostics'):manifest.get('stages',{}).pop(stage,None)
    write_json(root/'manifest.json',manifest)
    prior=None
    if cfg['gross_offsets']['mode']=='velocity':
        from .velocity_prior import prepare_prior
        shape=tuple(manifest['coreg']['reference_grid'][key] for key in ('length','width'))
        p=cfg['matching'];grid=build_grid(shape,shape,p['window'],p['search'],p['stride'],first_center=p.get('first_center'))
        if 'stack_pair' in manifest:
            from .stack_pairs import prepare_prior as stack_prior
            prior=stack_prior(cfg,manifest,grid,native_pair)
        elif 'assembly' in manifest:
            from .assembly import prepare_prior as assembled_prior
            prior=assembled_prior(cfg,manifest,grid,native_pairs=native_pair)
        else:
            prior=prepare_prior(cfg,manifest,grid,native_pair=native_pair)
        prior['grid']=grid
        prior['metadata']['source_provenance']=spec['velocity_provenance']
    if 'stack_pair' in manifest:
        from .stack_pairs import match_prepared
        r=match_prepared(cfg,manifest,directory,prior)
    else:
        r=native_match(manifest['coreg']['slcs']['reference'],manifest['coreg']['slcs']['secondary'],directory/'native',
                       cfg['matching'],cfg,cfg['matching']['backend'],prior=prior)
    if 'assembly' in manifest:
        from .assembly import annotate_result
        annotate_result(manifest,r)
    metadata=spec|{'reference_grid':manifest['coreg']['reference_grid'],'acquisition_times':manifest['acquisition_times'],
                  'interval_seconds':manifest['interval_seconds'],'backend':r['backend'],'elapsed_seconds':r['elapsed_seconds'], 'native_parameters':r['effective_parameters'],
                  'burst_id':cfg['inputs']['burst_id'],'polarization':cfg['inputs']['polarization'],
                  'input_provenance':manifest['coreg_specification']['inputs'],
                  'software_versions':manifest['software']['packages'],
                  'mask_policy':'nonfinite invalid; 3x3/boundary connected complex-zero fill invalid; isolated quantized zeros retained; derived complex nodata=0 GDAL mask superseded',
                  'center_definition':'zero-based full-resolution reference chip start + (window-1)/2'}
    if 'assembly' in manifest:
        metadata.update(assembly_identity=manifest['assembly']['identity'],
                        source_burst_ids=manifest['assembly']['source_burst_ids'],
                        interval_policy=manifest['assembly']['interval_policy'])
    if 'stack_pair' in manifest:
        metadata['stack_pair']=manifest['stack_pair']
    tmp=directory/'offsets.tmp.h5';write_offsets(tmp,r,metadata);tmp.replace(path)
    manifest['match_identity']=fingerprint;manifest['matching']=metadata
    manifest['matching_summary']={'windows':int(np.prod(r['grid'].shape)),'input_valid':int(r['input_valid'].sum()),'quality_valid':int(r['valid'].sum())}
    if 'assembly' in manifest:
        manifest['matching_summary']['seam_crossing']=int(r['seam_crossing'].sum())
        manifest['matching_summary']['seam_crossing_valid']=int((r['seam_crossing']&r['valid']).sum())
    logging.info('Native %s Ampcor: %d windows, %d input-valid, %d quality-valid', r['backend'], np.prod(r['grid'].shape),r['input_valid'].sum(),r['valid'].sum())
    manifest['stages']['match']='complete'
    write_json(root/'manifest.json',manifest)
