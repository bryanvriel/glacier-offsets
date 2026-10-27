#!/usr/bin/env python3
"""Full merged-pair deramp=1 versus complex range-2x/magnitude comparison.

All native workers read complete merged inputs across partition boundaries.
Production inputs are read-only. Run with python -m tools.compare_merged_preprocessing.
"""
import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import _native_worker, match
from glacier_offsets.diagnostics import diagnostics
from glacier_offsets.geocode import geocode
from glacier_offsets.products import ChecksumCache, identity, write_json
from glacier_offsets.stack_config import load_stack_config
from glacier_offsets.stack_pairs import matching_template
from glacier_offsets.stack_storage import assemble_date, read_manifest, release_date
from tools.compare_swath_overlap import statistics
from tools.compare_swath_range_oversampling import normalize_range_result, prepare_strip, prepared_grid


def refinement_settings(params,zoom=None):
    """Reject silent native clipping and record scalar-zoom physical extents."""
    selected=deepcopy(params)
    if zoom is not None:selected['zoom']=zoom
    z=selected['zoom'];raw=selected['raw_oversampling']
    if z<=0 or z%(2*raw):
        raise ValueError('zoom must be a positive multiple of twice raw_oversampling')
    if z>2*raw*min(selected['search']):
        raise ValueError('Requested zoom exceeds the native search-dependent cap')
    half=z//(2*raw)
    return selected,{'requested_zoom':z,'raw_oversampling':raw,
                     'deramp1_half_width_original_px':[half,half],
                     'range2x_half_width_original_px':[half,half/2]}


def matching_without_zoom(params):
    """Canonical ISCE3 settings, matching the production provenance policy."""
    if params.get('engine','isce3')!='isce3':raise ValueError('Expected ISCE3 matching')
    return {k:v for k,v in params.items() if k not in ('zoom','engine','isce2')}


def range_native_worker(arguments):
    """Adapt one scheduler segment without cropping its image or matching halos."""
    args,kwargs=arguments
    reference,secondary,output,params,cfg,backend=args
    control=cfg['_range2x_control'];grid=kwargs['_grid']
    native_grid=prepared_grid(grid,control['origin_yx'])
    native_params=deepcopy(params)
    native_params.update(deramp=0,window=list(native_grid.window),search=list(native_grid.search),stride=list(native_grid.stride))
    native_params.pop('first_center',None)
    native_kwargs=dict(kwargs,_grid=native_grid)
    if kwargs.get('_gross') is not None:
        raise ValueError('This experiment requires zero constant gross offsets and the full-grid matcher')
    result=_native_worker(((*control['paths'],output,native_params,cfg,backend),native_kwargs))
    line,sample=result['grid'].centers;y0,x0=control['origin_yx']
    np.testing.assert_array_equal(line+y0,grid.centers[0])
    np.testing.assert_array_equal(sample/2+x0-.25,grid.centers[1])
    normalize_range_result(result,grid,params)
    result['effective_parameters']['range2x_control']={
        'factor_yx':[1,2],'origin_original_yx':[y0,x0-.25],
        'reported_offsets_and_covariance':'Original reference SLC pixel units',
        'matching_grid':'Original global chip centers; complete merged input images across scheduling boundaries'}
    return result


def read_offsets(directory):
    with h5py.File(directory/'radar/offsets.h5') as h:
        names=('center_line','center_sample','source_burst_index','seam_crossing','input_valid','valid',
               'correlation_peak','snr','total/azimuth_px','total/range_px')
        return {name:h[name][:] for name in names}


def compare_outputs(pair,out,variants):
    """Full fields and fixed-cohort changes; no swath-agreement selection."""
    data={'baseline_deramp0':read_offsets(pair)}|{name:read_offsets(path) for name,path in variants.items()}
    baseline=data['baseline_deramp0'];common=baseline['valid'].copy()
    for name,arrays in data.items():
        for key in ('center_line','center_sample','source_burst_index','seam_crossing','input_valid'):
            np.testing.assert_array_equal(baseline[key],arrays[key],err_msg=f'{name}: changed {key}')
        common&=arrays['valid']
    report={'same_centers_ownership_seam_flags_and_input_support':True,'windows':int(common.size),
        'input_valid':int(baseline['input_valid'].sum()),'common_valid_all_three':int(common.sum()),
        'treatments':{},'pairwise_changes':{},'geocoded':{},
        'limitations':'Changed offsets and improved visual continuity do not establish accuracy. Range2x changes window discretization, SNR normalization, quantization and physical refinement extent. Periodic Fourier interpolation of merged rows includes splice discontinuities.'}
    for name,arrays in data.items():
        accepted=baseline['input_valid']&arrays['valid']
        report['treatments'][name]={'valid_n':int(arrays['valid'].sum()),
            'baseline_valid_lost':int((baseline['valid']&~arrays['valid']).sum()),
            'baseline_invalid_gained':int((~baseline['valid']&arrays['valid']).sum()),
            'snr':statistics(arrays['snr'][accepted]),'peak':statistics(arrays['correlation_peak'][accepted])}
    cohorts={'common_valid':common,'common_valid_seam_crossing':common&baseline['seam_crossing'],
             'common_valid_interior':common&~baseline['seam_crossing']}
    strong=common.copy()
    for arrays in data.values():strong&=arrays['correlation_peak']>=.3
    cohorts['all_three_peak_ge_0.3']=strong
    for a,b in [('baseline_deramp0','deramp1'),('baseline_deramp0','range2x'),('deramp1','range2x')]:
        report['pairwise_changes'][f'{b}_minus_{a}']={label:{axis:statistics(
            (data[b][f'total/{axis}_px']-data[a][f'total/{axis}_px'])[mask])
            for axis in ('azimuth','range')} for label,mask in cohorts.items()}
    for name,path in variants.items():
        report['geocoded'][name]=read_manifest(path)['geocoding']
    np.savez_compressed(out/'comparison_masks.npz',**cohorts)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['baseline_deramp0','deramp1','range2x']
    for kind in ('radar','geo'):
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        for row,axis in enumerate(('range','azimuth')):
            images=[]
            for name in labels:
                if kind=='radar':image=np.where(common,data[name][f'total/{axis}_px'],np.nan)
                else:
                    directory=pair if name=='baseline_deramp0' else variants[name]
                    ds=gdal.Open(str(directory/'geo'/f'{axis}_displacement_m.tif'))
                    reference=gdal.Open(str(pair/'geo'/f'{axis}_displacement_m.tif'))
                    if ds.GetGeoTransform()!=reference.GetGeoTransform() or ds.GetProjection()!=reference.GetProjection():
                        raise ValueError('Changed map geometry')
                    image=ds.ReadAsArray();np.testing.assert_array_equal(image.shape,reference.ReadAsArray().shape)
                images.append(image)
            finite=np.concatenate([im[np.isfinite(im)] for im in images])
            limit=max(float(np.quantile(abs(finite),.95)),.1) if finite.size else .1
            for col,(name,image) in enumerate(zip(labels,images)):
                artist=axes[row,col].imshow(image,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
                axes[row,col].set_title(f'{name}: {axis} [{"original px" if kind=="radar" else "m"}]')
                axes[row,col].set(xlabel='Matching range column' if kind=='radar' else 'Map column',
                                  ylabel='Matching azimuth row' if kind=='radar' else 'Map row')
                fig.colorbar(artist,ax=axes[row,col],shrink=.7)
        fig.suptitle(f'{pair.name}: complete merged inputs; identical display scales within each component')
        fig.savefig(out/f'{kind}_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    for row,axis in enumerate(('range','azimuth')):
        pairs=[('baseline_deramp0','deramp1'),('baseline_deramp0','range2x'),('deramp1','range2x')]
        changes=[np.where(common,data[b][f'total/{axis}_px']-data[a][f'total/{axis}_px'],np.nan) for a,b in pairs]
        finite=np.concatenate([d[np.isfinite(d)] for d in changes])
        limit=max(float(np.quantile(abs(finite),.95)),1/64) if finite.size else 1/64
        for col,((a,b),delta) in enumerate(zip(pairs,changes)):
            artist=axes[row,col].imshow(delta,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row,col].set_title(f'{b} minus {a}: {axis} [original px]')
            fig.colorbar(artist,ax=axes[row,col],shrink=.7)
    fig.suptitle('Changes on the same observations accepted by all three treatments')
    fig.savefig(out/'radar_changes.png',dpi=160);plt.close(fig)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',type=Path);parser.add_argument('pair',type=Path)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--zoom',type=int,help='Override scalar refinement zoom for both treatments')
    parser.add_argument('--zoom-baseline',type=Path,help='Completed full comparison at another zoom, for sensitivity analysis')
    args=parser.parse_args(argv);pair=args.pair.resolve();out=args.output.resolve();stack=pair.parent.parent
    if out.exists() or out==stack or stack in out.parents or out in stack.parents:
        raise ValueError('Choose a new directory outside the production stack')
    if args.workers<1:raise ValueError('workers must be positive')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');gdal.UseExceptions()
    original=read_manifest(pair);frame=read_manifest(stack/'geometry')
    if original['matching']['gross_offsets']['mode']!='zero' or original['matching']['matching']['deramp']!=0:
        raise ValueError('A zero-gross deramp=0 baseline is required')
    if frame['geometry_identity']!=original['stack_pair']['geometry_identity']:
        raise ValueError('Saved pair no longer matches the production geometry')
    cfg=load_stack_config(args.config)
    cfg['matching'],refinement=refinement_settings(original['matching']['matching'],args.zoom)
    if args.zoom_baseline:
        prior=args.zoom_baseline.resolve()
        previous=json.loads((prior/'comparison.json').read_text())
        if previous['status']!='complete' or previous['source_pair']!=str(pair):
            raise ValueError('Zoom baseline must be a completed comparison of the same saved pair')
        for name in ('deramp1','range2x'):
            old=read_manifest(prior/name)['matching']['matching']
            new=dict(cfg['matching'],engine='isce3',backend='cpu',deramp=1 if name=='deramp1' else 0)
            if matching_without_zoom(old)!=matching_without_zoom(new):
                raise ValueError('Only zoom may change relative to the saved comparison')
    cfg['matching'].update(engine='isce3',backend='cpu');cfg['gross_offsets']=deepcopy(original['matching']['gross_offsets'])
    cfg['inputs'].update(burst_id='stack',polarization=original['coreg_specification']['polarization'])
    cfg.pop('aoi',None);cfg['output']['directory']=str(out/'inputs')
    cfg['geocoding']={k:original['geocoding'][k] for k in ('epsg','spacing_m','resampling')}
    cfg['geocoding']['mask']=original['geocoding']['source_mask']
    os.environ['OMP_NUM_THREADS']=str(args.workers)
    days=original['stack_pair']['acquisition_dates'];dates=[read_manifest(stack/'acquisitions'/day) for day in days]
    watched={pair/'manifest.json',stack/'geometry/manifest.json',stack/'stack_manifest.json',pair/'radar/offsets.h5',pair/'geo/center_geometry.h5'}
    watched.update(stack/'acquisitions'/day/'manifest.json' for day in days)
    watched.update(Path(p) for date in dates for p in date['alignment_products'])
    watched.update(Path(p) for p in frame['products'])
    if args.zoom_baseline:
        watched.add(prior/'comparison.json')
        watched.update(prior/name/file for name in ('deramp1','range2x')
                       for file in ('manifest.json','radar/offsets.h5'))
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
    out.mkdir(parents=True)
    write_json(out/'experiment.json',{'status':'verifying','source_pair':str(pair),'config':str(args.config.resolve())})
    checksums=ChecksumCache()
    for date in dates:
        for path in date['slcs'].values():
            if checksums(path)!=date['alignment_products'][str(Path(path).resolve())]:
                raise ValueError(f'Changed aligned source: {path}')
    assemblies=[]
    for day,date in zip(days,dates):
        logging.info('Reconstructing exact saved merged SLC: %s',day)
        assembly=assemble_date(cfg,SimpleNamespace(date=day),frame,date,force=True,checksums=checksums)
        if checksums(assembly['slc'])!=date['products'][str(Path(date['slc']).resolve())]:
            raise ValueError(f'{day}: rebuilt merged SLC checksum differs from saved baseline')
        assemblies.append(assembly)
    grid=matching_template(cfg,{'coreg':original['coreg']})
    geometry_names=('longitude_deg','latitude_deg','height_m','azimuth_ground_spacing_m_per_slc_line','geometry_valid')
    with h5py.File(pair/'geo/center_geometry.h5') as h:geometry=tuple(h[k][:] for k in geometry_names)
    variants={};manifests={};range_inputs=[]
    for name in ('deramp1','range2x'):
        directory=out/name;directory.mkdir();variant=deepcopy(cfg)
        variant['matching']['deramp']=1 if name=='deramp1' else 0
        if name=='range2x':
            records=[]
            for index,assembly in enumerate(assemblies):
                target=out/'range2x_inputs'/f'date-{index}.tif'
                logging.info('Range-only complex oversampling of complete merged SLC: %s',days[index])
                _,record=prepare_strip(assembly['slc'],target,grid)
                records.append(record);range_inputs.append(target)
            y0,_,x0,_=records[0]['retained_original_window_yx']
            if records[0]['retained_original_window_yx']!=records[1]['retained_original_window_yx']:
                raise ValueError('Prepared merged input origins differ')
            variant['_range2x_control']={'paths':[str(p) for p in range_inputs],'origin_yx':[y0,x0]}
            write_json(out/'range2x_inputs/preparation.json',{'inputs':records,'sha256':{str(p):checksums(p) for p in range_inputs}})
        manifest=deepcopy(original)
        for key in ('match_identity','geocode_identity','diagnostics_identity','matching','matching_summary','geocoding'):
            manifest.pop(key,None)
        manifest['stages']={'coregister':'Reused verified saved merged geometry; no COMPASS'}
        manifest['coreg']['slcs']=dict(zip(('reference','secondary'),[a['slc'] for a in assemblies]))
        manifest['coreg_identity']=identity({'original':original['coreg_identity'],'experiment':str(out),'treatment':name})
        manifest['validation']={'source_pair':str(pair),'treatment':name,'merged_inputs':True,'fixed_original_grid':True}
        logging.info('Matching complete merged pair: %s',name)
        if name=='range2x':
            with patch('glacier_offsets.stack_matching._native_worker',range_native_worker):match(variant,directory,manifest)
        else:match(variant,directory,manifest)
        logging.info('Geocoding and diagnostic plots: %s',name)
        with patch('glacier_offsets.geocode.center_geometry',return_value=geometry):geocode(variant,directory,manifest)
        diagnostics(variant,directory,manifest)
        variants[name]=directory;manifests[name]=manifest
        write_json(out/'progress.json',{'completed_treatments':list(variants)})
    report=compare_outputs(pair,out,variants)
    unchanged=all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in before.items())
    report.update(status='complete' if unchanged else 'incomplete',source_pair=str(pair),
                  identical_reconstructed_merged_slc_checksums=True,production_sizes_mtimes_unchanged=unchanged,
                  matching_original=original['matching']['matching'],matching_selected=cfg['matching'],
                  refinement=refinement,workers=args.workers)
    write_json(out/'comparison.json',report)
    if not unchanged:raise ValueError('Production files changed during experiment')
    if args.zoom_baseline:
        from tools.compare_zoom_sensitivity import analyze
        analyze(prior,out)
    # Release only the temporary merged images generated inside this experiment.
    cfg.setdefault('storage',{})['keep_merged_slcs']=False
    for day,assembly in zip(days,assemblies):release_date(cfg,SimpleNamespace(date=day),assembly)
    for directory,manifest in zip(variants.values(),manifests.values()):
        manifest['slc_previews']=dict(zip(('reference','secondary'),[a['slc_preview'] for a in assemblies]))
        manifest['validation']['temporary_merged_inputs_released']=True
        write_json(directory/'manifest.json',manifest)
    for path in range_inputs:path.unlink()
    write_json(out/'experiment.json',{'status':'complete','source_pair':str(pair),'temporary_merged_inputs_released':True,
                                    'comparison':str(out/'comparison.json')})
    logging.info('Completed full merged-pair comparison: %s',out)
    return 0


if __name__=='__main__':raise SystemExit(main())
