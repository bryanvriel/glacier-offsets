#!/usr/bin/env python3
"""Range-only complex interpolation before magnitude detection, on fixed patches.

Run as python -m tools.compare_swath_range_oversampling BASELINE --output NEW.
BASELINE is a completed compare_swath_overlap.py --compare-deramp experiment.
This diagnostic reads the exact same aligned inputs and source selections and
does not modify the baseline or production products.
"""
import argparse
from copy import deepcopy
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import resample

from glacier_offsets.ampcor import _native_executor, _native_worker
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import ChecksumCache, write_json
from glacier_offsets.raster_io import bounded_raster_cache
from glacier_offsets.stack_storage import read_manifest
from tools.compare_swath_overlap import (attach_results, cohort_summary, dual_valid,
                                         select_views, statistics, write_mode_outputs)


def resample_range(rows, factor=2):
    """Periodic complex Fourier interpolation; no azimuth processing or deramp.

    Output sample j represents input coordinate (j+.5)/factor-.5. This preserves
    pixel-cell boundaries, physical window width AND center when W becomes f*W.
    scipy.signal.resample handles even-length Nyquist-bin splitting explicitly.
    """
    if factor!=2:raise ValueError('This control implements exactly 2x range sampling')
    if not np.iscomplexobj(rows) or not np.isfinite(rows).all():
        raise ValueError('Range interpolation requires finite complex rows; no filling is performed')
    up=resample(rows,factor*rows.shape[-1],axis=-1)
    origin=(1/factor-1)/2
    phase=np.exp(2j*np.pi*fftfreq(up.shape[-1])*origin*factor).astype(up.dtype)
    return ifft(fft(up,axis=-1)*phase,axis=-1).astype('complex64')


def prepared_grid(grid, origin_yx, factor=2):
    y0,x0=origin_yx
    return MatchingGrid(grid.starts_line-y0,(grid.starts_sample-x0)*factor,
                        (grid.window[0],grid.window[1]*factor),
                        (grid.search[0],grid.search[1]*factor),
                        (grid.stride[0],grid.stride[1]*factor))


def prepare_strip(source, output, grid):
    """Interpolate complete original rows, then retain only the matching strip.

    Full-row interpolation avoids a new FFT boundary near the overlap patches.
    Bounded row blocks limit RAM; TIFFs are uncompressed and stripped for native
    CPU mmap. Input support is inherited only after verifying source checksums.
    """
    ds=gdal.Open(str(source));height,width=ds.RasterYSize,ds.RasterXSize
    y0=int(grid.starts_line.min()-grid.search[0]);x0=int(grid.starts_sample.min()-grid.search[1])
    y1=int(grid.starts_line.max()+grid.window[0]+grid.search[0])
    x1=int(grid.starts_sample.max()+grid.window[1]+grid.search[1])
    if not (0<=y0<y1<=height and 0<=x0<x1<=width):
        raise ValueError('Saved matching chip/search lies outside an aligned input')
    output.parent.mkdir(parents=True,exist_ok=True)
    dst=gdal.GetDriverByName('GTiff').Create(str(output),2*(x1-x0),y1-y0,1,gdal.GDT_CFloat32,
                                            options=['TILED=NO','COMPRESS=NONE'])
    with bounded_raster_cache():
        for row in range(y0,y1,16):
            count=min(16,y1-row)
            data=ds.ReadAsArray(0,row,width,count)
            interpolated=resample_range(data)
            dst.GetRasterBand(1).WriteArray(interpolated[:,2*x0:2*x1],0,row-y0)
    dst=None;ds=None
    native_grid=prepared_grid(grid,(y0,x0))
    line,sample=native_grid.centers
    np.testing.assert_array_equal(line+y0,grid.centers[0])
    np.testing.assert_array_equal(sample/2+x0-.25,grid.centers[1])
    return native_grid,{'source':str(source),'output':str(output),'original_shape_yx':[height,width],
        'retained_original_window_yx':[y0,y1,x0,x1],'factor_yx':[1,2],
        'original_coordinate_of_output_sample_0_yx':[y0,x0-.25],
        'interpolator':'Full original complex rows; scipy.signal.resample plus Fourier phase shift; periodic boundaries; no taper, deramp, fill or azimuth interpolation',
        'physical_center_and_pixel_cell_footprint_preserved':True}


def run_range_job(job):
    key,paths,out,params,grid,support=job
    try:
        gdal.UseExceptions()
        prepared=[];metadata=[];native_grid=None
        for i,path in enumerate(paths):
            target=out/'prepared'/key/f'date-{i}.tif'
            native_grid,record=prepare_strip(path,target,grid)
            prepared.append(target);metadata.append(record)
        native_params=deepcopy(params)
        native_params.update(deramp=0,window=list(native_grid.window),search=list(native_grid.search),
                             stride=list(native_grid.stride))
        # Explicit _grid takes precedence over the saved original first_center.
        native_params.pop('first_center',None)
        write_json(out/'prepared'/key/'preparation.json',{'inputs':metadata,'matching':native_params})
        result=_native_worker(((*prepared,out/'native'/key,native_params,{'gross_offsets':{'mode':'zero'}},'cpu'),
                               {'_grid':native_grid,'_valid_input':support}))
        np.testing.assert_array_equal(result['input_valid'],support)
        y0,_,x0,_=metadata[0]['retained_original_window_yx']
        line,sample=result['grid'].centers
        np.testing.assert_array_equal(line+y0,grid.centers[0])
        np.testing.assert_array_equal(sample/2+x0-.25,grid.centers[1])
        normalize_range_result(result,grid,params)
        result['range_preparation']=metadata
        return key,result,None
    except Exception as exc:
        return key,None,f'{type(exc).__name__}: {exc}'


def normalize_range_result(result,grid,params):
    """Convert 2x-range native results and eligibility to original pixel units."""
    result['native_valid_before_unit_conversion']=result['valid'].copy()
    for name in ('total','gross','residual'):
        result[name]=np.array(result[name],copy=True);result[name][...,1]/=2
    cov=np.array(result['covariance'],copy=True)
    sentinel=np.all(cov[...,:2]==99,axis=-1)
    cov[...,1]/=4;cov[...,2]/=2;cov[sentinel]=[99,99,0]
    result['covariance']=cov
    # Apply the baseline's boundary exclusion in original pixel units.
    boundary=np.any(abs(result['residual'])>=np.asarray(grid.search)-1/params['oversampling'],axis=-1)
    result['search_boundary']=boundary
    result['valid']=result['input_valid']&result['finite']&~boundary&(
        result['snr']>=params['snr_threshold'])&(result['correlation_peak']>=params['correlation_threshold'])
    result['grid']=grid
    return result


def load_npz(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}


def summarize_control(views,cohort):
    result=cohort_summary(views,cohort)
    result['treatments']={'baseline_deramp0':result['modes'].pop('0'),
                          'range2x_then_deramp0':result['modes'].pop('1')}
    result.pop('modes')
    result['control_minus_baseline_on_common_valid_px']=result.pop('deramp1_minus_deramp0_on_common_valid_px')
    result['validity_transitions']={
        key.replace('deramp0','baseline').replace('deramp1','control'):value
        for key,value in result['validity_transitions'].items()}
    return result


def sine_fit(predictor,values):
    good=np.isfinite(predictor)&np.isfinite(values);x=predictor[good];y=values[good]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return {'n':int(len(x))}
    X=np.column_stack([np.ones(len(x)),x]);coef=np.linalg.lstsq(X,y,rcond=None)[0]
    return {'n':int(len(x)),'pearson_r':float(np.corrcoef(x,y)[0,1]),
            'intercept_px':float(coef[0]),'sine_coefficient_px':float(coef[1])}


def write_comparison(out,pair,baseline,control,deramp1,saved,interior,errors):
    support=baseline['iw1']['support']&baseline['iw2']['support']
    valid0=dual_valid(baseline);valid1=dual_valid(control)
    groups={'all_input_supported':support,'input_supported_burst_interiors':support&interior,
            'common_native_valid_burst_interiors':support&interior&valid0&valid1}
    for threshold in (.3,.5):
        groups[f'baseline_both_peak_ge_{threshold}_burst_interiors']=support&interior&valid0&(
            baseline['iw1']['peak']>=threshold)&(baseline['iw2']['peak']>=threshold)
    views={0:baseline,1:control}
    m=groups['baseline_both_peak_ge_0.3_burst_interiors']
    predictor=np.sin(2*np.pi*(baseline['iw1']['total'][...,1]+baseline['iw2']['total'][...,1])/2)
    deltas={label:(v['iw2']['total']-v['iw1']['total']) for label,v in
            [('baseline_deramp0',baseline),('range2x_then_deramp0',control)]}
    deltas['deramp1']=deramp1['iw2_total']-deramp1['iw1_total']
    common=m&valid1&deramp1['dual_valid']
    fits={label:sine_fit(predictor[common],delta[...,1][common]) for label,delta in deltas.items()}
    report={'status':'incomplete' if errors else 'complete','source_pair':str(pair),'job_errors':errors,
        'groups':{name:summarize_control(views,mask) for name,mask in groups.items()},
        'fixed_baseline_sine_predictor_fits':fits,'sine_fit_common_n':int(common.sum()),
        'sine_predictor':'sin(2*pi*average baseline IW1/IW2 range offset); same baseline-quality interior cohort, common valid across all three treatments',
        'units':'Original full-resolution reference SLC pixels; array component order [azimuth,range]',
        'range_quantization_px':{'baseline':1/64,'control':1/128},
        'limitations':'No independent displacement truth. Range sampling and detection order change together; window discretization, physical native refinement extent, SNR normalization and quantization differ. Full-row periodic Fourier interpolation can be influenced by distant edges. Native zoom/raw/refinement factors stay fixed; physical range refinement extent halves. Azimuth sampling is unchanged. The sine fit shares measured data with its predictor and is descriptive.'}
    np.savez_compressed(out/'comparison_masks.npz',center_line=saved['center_line'],center_sample=saved['center_sample'],**groups)
    write_json(out/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    line=saved['center_line'];colors=['tab:blue','tab:green','tab:orange']
    labels=['baseline_deramp0','range2x_then_deramp0','deramp1']
    for col,(axis,k) in enumerate([('range',1),('azimuth',0)]):
        for label,color in zip(labels,colors):
            y=deltas[label][...,k]
            axes[0,col].scatter(line[common],y[common],s=15,alpha=.65,label=label,color=color)
        axes[0,col].axhline(0,color='black',lw=.8)
        axes[0,col].set(title=f'{axis}: same {int(common.sum())} baseline-quality interior points',
                        xlabel='Global reference azimuth line',ylabel='IW2 minus IW1 [original px]')
        axes[0,col].legend(fontsize=8)
        for swath,color in zip(('iw1','iw2'),colors):
            change=(control[swath]['total']-baseline[swath]['total'])[...,k]
            axes[1,col].scatter(line[common],change[common],s=15,label=swath.upper(),color=color)
        axes[1,col].axhline(0,color='black',lw=.8)
        axes[1,col].set(title=f'Range-only oversampling: change in {axis}',
                        xlabel='Global reference azimuth line',ylabel='Control minus baseline [original px]')
        axes[1,col].legend(fontsize=8)
    fig.suptitle(f'{pair.name}: complex range interpolation before magnitude; fixed support and physical centers')
    fig.savefig(out/'range_control.png',dpi=160);plt.close(fig)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(argv)
    baseline=args.baseline.resolve();out=args.output.resolve()
    report0=json.loads((baseline/'report.json').read_text())
    pair=Path(report0['source_pair']).resolve();stack=pair.parent.parent
    if out.exists() or any(out==p or p in out.parents or out in p.parents for p in (stack,baseline)):
        raise ValueError('Use a new output directory outside the stack and baseline')
    if args.workers<1:raise ValueError('workers must be positive')
    if report0['status']!='complete' or not report0.get('production_sizes_mtimes_unchanged'):
        raise ValueError('A complete verified paired-deramp baseline is required')
    params=deepcopy(report0['matching_by_deramp']['0'])
    manifest=read_manifest(pair);expected=deepcopy(manifest['matching']['matching']);expected['engine']='isce3'
    if params!=expected or params['deramp']!=0 or manifest['matching']['gross_offsets']['mode']!='zero':
        raise ValueError('Saved pair matching settings no longer match the baseline')
    gdal.UseExceptions();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    saved=load_npz(baseline/'deramp-0/observations.npz');deramp1=load_npz(baseline/'deramp-1/observations.npz')
    dates=manifest['stack_pair']['acquisition_dates'];acqs=[read_manifest(stack/'acquisitions'/day) for day in dates]
    sources=manifest['assembly']['sources'];records={};jobs=[];checksums=ChecksumCache()
    watched={baseline/'report.json',pair/'manifest.json',stack/'stack_manifest.json',stack/'geometry/manifest.json',pair/'radar/offsets.h5'}
    watched.update(baseline/f'deramp-{mode}/observations.npz' for mode in (0,1))
    watched.update(stack/'acquisitions'/day/'manifest.json' for day in dates)
    for source in sources:
        watched.update(Path(p) for p in source['coreg']['geometry'].values())
        for acq in acqs:watched.update(Path(p) for p in acq['alignment_products'])
    watched.update((baseline/'deramp-0').glob('t*.npz'))
    stamps={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
    out.mkdir(parents=True);write_json(out/'report.json',{'status':'verifying','baseline':str(baseline)})
    for source in sources:
        key=source['burst_id'];raw_path=baseline/'deramp-0'/f'{key}.npz'
        if not raw_path.is_file():continue
        raw=load_npz(raw_path);rows,cols=raw['rows'],raw['cols'];dy,dx=source['offset']
        starts_y=saved['center_line'][rows,0]-(params['window'][0]-1)/2-dy
        starts_x=saved['center_sample'][0,cols]-(params['window'][1]-1)/2-dx
        if not np.all(starts_y==starts_y.astype(int)) or not np.all(starts_x==starts_x.astype(int)):
            raise ValueError('Baseline centers do not imply integer original chip starts')
        grid=MatchingGrid(starts_y.astype('int32'),starts_x.astype('int32'),
                          tuple(params['window']),tuple(params['search']),tuple(params['stride']))
        paths=[]
        logging.info('Verifying original aligned inputs: %s',key)
        for day,acq in zip(dates,acqs):
            path=Path(acq['slcs'][key]).resolve();provenance=report0['verified_aligned_inputs'][day][key]
            if str(path)!=provenance['path'] or checksums(path)!=provenance['sha256'] or (
                acq['alignment_products'][str(path)]!=provenance['sha256']):
                raise ValueError(f'Aligned input differs from verified baseline: {path}')
            paths.append(path)
        result0={name:raw[name] for name in ('total','valid','finite','search_boundary','snr')}
        result0['correlation_peak']=raw['peak']
        records[key]={'entry':source,'rows':rows,'cols':cols,'support':raw['input_valid'],
                      'geometry':raw['geometry_lonlat'],'results':{0:result0}}
        jobs.append((key,paths,out,params,grid,raw['input_valid']))
    selection,numbers=select_views(sources,records,saved['center_line'].shape)
    for swath in ('iw1','iw2'):
        np.testing.assert_array_equal(selection[swath]['owner'],saved[swath+'_owner'])
    baseline_views=attach_results(selection,records,numbers,0)
    for swath in ('iw1','iw2'):
        np.testing.assert_array_equal(baseline_views[swath]['total'],saved[swath+'_total'])
    np.testing.assert_array_equal(selection['iw1']['support']&selection['iw2']['support'],saved['dual_support'])
    if not jobs:raise ValueError('No saved per-burst observations in baseline')
    write_json(out/'report.json',{'status':'matching','baseline':str(baseline),'job_count':len(jobs)})
    errors=[];completed=[]
    with _native_executor(min(args.workers,len(jobs)),single_threaded=True) as pool:
        for key,result,error in pool.imap_unordered(run_range_job,jobs):
            if error:
                errors.append({'burst':key,'error':error});logging.error('%s: %s',key,error)
            else:
                records[key]['results'][1]=result;r=records[key]
                arrays={name:result[name] for name in ('total','gross','residual','valid','finite','input_valid',
                    'search_boundary','snr','covariance','native_valid_before_unit_conversion')}
                np.savez_compressed(out/f'{key}.npz',rows=r['rows'],cols=r['cols'],**arrays,
                    peak=result['correlation_peak'],effective_json=json.dumps(result['effective_parameters']),
                    center_line=result['grid'].centers[0]+r['entry']['offset'][0],
                    center_sample=result['grid'].centers[1]+r['entry']['offset'][1])
                logging.info('Range2x %s: %d/%d supported windows valid',key,int(result['valid'].sum()),int(r['support'].sum()))
            completed.append({'burst':key,'status':'failed' if error else 'complete'})
            write_json(out/'progress.json',{'completed_jobs':completed,'job_errors':errors,'job_count':len(jobs)})
    control=attach_results(selection,records,numbers,1)
    with h5py.File(pair/'radar/offsets.h5') as h:merged_valid=h['valid'][:]
    saved_fields={'center_line':saved['center_line'],'center_sample':saved['center_sample'],'valid':merged_valid,
                  'total/azimuth_px':saved['merged_total'][...,0],'total/range_px':saved['merged_total'][...,1]}
    # The individual mode plot uses saved merged values only as a reference.
    mode_report=write_mode_outputs(out,pair,params,saved_fields,control,saved['dual_support'],numbers,
                                   saved['burst_interior'],saved['geometry_distance_m'])
    mode_report['status']='incomplete' if errors else 'complete'
    mode_report['input_preparation']='Complex 2x range interpolation before native magnitude detection; results in original pixel units'
    write_json(out/'control_mode_report.json',mode_report)
    report=write_comparison(out,pair,baseline_views,control,deramp1,saved,saved['burst_interior'],errors)
    unchanged=all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in stamps.items())
    report.update(baseline=str(baseline),production_and_baseline_sizes_mtimes_unchanged=unchanged,
                  fixed_source_owners_support_centers=True,matching_original=params,
                  verified_aligned_inputs=report0['verified_aligned_inputs'])
    if not unchanged:report['status']='incomplete'
    write_json(out/'report.json',report)
    if not unchanged:raise ValueError('Source or baseline files changed during control')
    logging.info('Completed range-only control: %s',out)
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
