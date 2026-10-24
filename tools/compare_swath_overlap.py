#!/usr/bin/env python3
"""Compare independent IW1/IW2 burst matches at identical saved radar centers.

Reads retained aligned bursts, never assembled SLCs. Support and the primary
burst choice are established before matching, without selecting for offset
agreement. Native observations, geometry checks and plots are diagnostic only.
Use --compare-deramp to run paired magnitude-first and complex-interpolation
treatments with the same input support, source owners, centers and settings.
"""
import argparse
from copy import deepcopy
from datetime import date as calendar_date
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Geod, Transformer

from glacier_offsets.ampcor import _native_executor, _native_worker
from glacier_offsets.geocode import sample_geometry
from glacier_offsets.matching_grid import MatchingGrid, valid_windows
from glacier_offsets.products import ChecksumCache, write_json
from glacier_offsets.stack_storage import read_manifest


def comparison_reference(params,saved,default_days,requested_days=None,paired=False):
    """Reuse a grid for new dates without presenting its old merged offsets as observations."""
    params=deepcopy(params);saved=deepcopy(saved)
    if requested_days is None:
        if params['deramp']!=0:raise ValueError('This comparison requires saved magnitude-first matching')
        return params,saved,list(default_days)
    if not paired:raise ValueError('--acquisition-dates requires --compare-deramp')
    days=list(requested_days)
    if len(days)!=2 or calendar_date.fromisoformat(days[0])>=calendar_date.fromisoformat(days[1]):
        raise ValueError('Choose two increasing acquisition dates')
    params['deramp']=0;saved['valid'][:]=False
    for axis in ('azimuth','range'):saved[f'total/{axis}_px']=np.full(saved['valid'].shape,np.nan)
    return params,saved,days


def run_job(job):
    name, args, kwargs = job
    try:
        return name, _native_worker((args, kwargs)), None
    except Exception as exc:
        return name, None, f'{type(exc).__name__}: {exc}'


def statistics(values):
    a = np.asarray(values);a = a[np.isfinite(a)]
    if not a.size:
        return {'n': 0}
    return {'n': int(a.size), 'mean': float(np.mean(a)), 'median': float(np.median(a)),
            'p10': float(np.quantile(a,.1)), 'p90': float(np.quantile(a,.9)),
            'mean_absolute': float(np.mean(abs(a))), 'median_absolute': float(np.median(abs(a))),
            'p90_absolute': float(np.quantile(abs(a),.9)), 'max_absolute': float(abs(a).max())}


def burst_interior_mask(sources, line, sample, owners, dual, window, search):
    """Exclude chip/search footprints crossing an along-track ownership change."""
    start_y=(line-(window[0]-1)/2).astype('int32')-search[0]
    start_x=(sample-(window[1]-1)/2).astype('int32')-search[1]
    height,width=[window[k]+2*search[k] for k in (0,1)]
    if not dual.any():
        return np.array(dual,copy=True)
    y0,x0=int(start_y[dual].min()),int(start_x[dual].min())
    y1,x1=int(start_y[dual].max()+height),int(start_x[dual].max()+width)
    within=np.array(dual,copy=True)
    ordered=sorted(sources,key=lambda e:(e['offset'][0],e['burst_id']))
    for swath in ('iw1','iw2'):
        raster=np.zeros((y1-y0,x1-x0),'uint16')
        for i,e in enumerate(ordered,1):
            if not e['burst_id'].endswith(swath):continue
            dy,dx=e['offset'];a,b,c,d=e['valid_window']
            a,b=max(a+dy,y0),min(b+dy,y1);c,d=max(c+dx,x0),min(d+dx,x1)
            if a<b and c<d:raster[a-y0:b-y0,c-x0:d-x0]=i
        for row,col in zip(*np.nonzero(dual)):
            a,b=int(start_y[row,col]-y0),int(start_x[row,col]-x0)
            within[row,col] &= bool((raster[a:a+height,b:b+width]==owners[swath][row,col]).all())
    return within


def select_views(sources, records, shape):
    """Freeze source ownership using support alone, before native matching."""
    views={s:{'owner':np.zeros(shape,'int16'),'support':np.zeros(shape,bool),
        'geometry':np.full((*shape,5,2),np.nan)} for s in ('iw1','iw2')}
    source_numbers={}
    for i,e in enumerate(sorted(sources,key=lambda e:(e['offset'][0],e['burst_id'])),1):
        key=e['burst_id'];source_numbers[i]=key
        if key not in records:continue
        r=records[key];v=views[key[-3:]];section=np.ix_(r['rows'],r['cols']);take=r['support']
        for field,value in [('owner',np.full(take.shape,i)),('support',take),('geometry',r['geometry'])]:
            block=v[field][section];block[take]=value[take];v[field][section]=block
    return views,source_numbers


def attach_results(selection, records, source_numbers, mode):
    """Attach each mode to fixed owners; a failed owner never falls back."""
    views=deepcopy(selection)
    for v in views.values():
        shape=v['support'].shape
        v.update({key:np.zeros(shape,bool) for key in ('valid','finite','search_boundary','native_completed')})
        v.update(total=np.full((*shape,2),np.nan),peak=np.full(shape,np.nan),snr=np.full(shape,np.nan))
    for number,key in source_numbers.items():
        if key not in records:continue
        r=records[key];result=r['results'].get(mode)
        if result is None:continue
        v=views[key[-3:]];section=np.ix_(r['rows'],r['cols'])
        take=v['owner'][section]==number
        for field,name in [('valid','valid'),('finite','finite'),('search_boundary','search_boundary'),
                           ('total','total'),('peak','correlation_peak'),('snr','snr')]:
            block=v[field][section];block[take]=result[name][take];v[field][section]=block
        block=v['native_completed'][section];block[take]=True;v['native_completed'][section]=block
    return views


def dual_valid(views):
    a,b=views['iw1'],views['iw2']
    return a['support']&b['support']&a['valid']&b['valid']


def cohort_summary(views_by_mode, cohort):
    """Report attrition and changes without selecting for offset agreement."""
    common=cohort&dual_valid(views_by_mode[0])&dual_valid(views_by_mode[1])
    report={'input_supported_n':int(cohort.sum()),'common_native_valid_n':int(common.sum()),'modes':{}}
    for mode,views in views_by_mode.items():
        accepted=cohort&dual_valid(views)
        delta=views['iw2']['total']-views['iw1']['total']
        entry={'dual_native_valid_n':int(accepted.sum()),'swaths':{},
            'difference_on_mode_valid_px':{axis:statistics(delta[...,k][accepted]) for k,axis in enumerate(('azimuth','range'))},
            'difference_on_common_valid_px':{axis:statistics(delta[...,k][common]) for k,axis in enumerate(('azimuth','range'))}}
        for swath,v in views.items():
            entry['swaths'][swath]={'native_completed_n':int((cohort&v['native_completed']).sum()),
                'native_valid_n':int((cohort&v['valid']).sum()),
                'nonfinite_n':int((cohort&v['native_completed']&~v['finite']).sum()),
                'search_boundary_n':int((cohort&v['native_completed']&v['search_boundary']).sum()),
                'rejected_n':int((cohort&~v['valid']).sum()),
                'peak':statistics(v['peak'][cohort]),'snr':statistics(v['snr'][cohort])}
        report['modes'][str(mode)]=entry
    report['validity_transitions']={
        'valid_both':int(common.sum()),
        'only_deramp0':int((cohort&dual_valid(views_by_mode[0])&~dual_valid(views_by_mode[1])).sum()),
        'only_deramp1':int((cohort&~dual_valid(views_by_mode[0])&dual_valid(views_by_mode[1])).sum()),
        'neither':int((cohort&~dual_valid(views_by_mode[0])&~dual_valid(views_by_mode[1])).sum())}
    report['deramp1_minus_deramp0_on_common_valid_px']={swath:{axis:statistics(
        (views_by_mode[1][swath]['total']-views_by_mode[0][swath]['total'])[...,k][common])
        for k,axis in enumerate(('azimuth','range'))} for swath in ('iw1','iw2')}
    return report


def compare_deramp_modes(out, pair, params_by_mode, views_by_mode, saved, interior, errors, pair_label=None):
    a,b=views_by_mode[0],views_by_mode[1]
    for swath in ('iw1','iw2'):
        for field in ('support','owner','geometry'):
            np.testing.assert_array_equal(a[swath][field],b[swath][field])
    p0,p1=deepcopy(params_by_mode[0]),deepcopy(params_by_mode[1])
    p0.pop('deramp');p1.pop('deramp')
    if p0!=p1:raise ValueError('Paired treatments must differ only in deramp')
    support=a['iw1']['support']&a['iw2']['support']
    common=support&dual_valid(a)&dual_valid(b)
    cohorts={'all_input_supported':support,'input_supported_burst_interiors':support&interior,
             'common_native_valid':common,'common_native_valid_burst_interiors':common&interior}
    for threshold in (.3,.5):
        baseline=support&dual_valid(a)&(a['iw1']['peak']>=threshold)&(a['iw2']['peak']>=threshold)
        strong=common&baseline&(b['iw1']['peak']>=threshold)&(b['iw2']['peak']>=threshold)
        cohorts[f'deramp0_both_peak_ge_{threshold}_burst_interiors']=baseline&interior
        cohorts[f'all_four_peak_ge_{threshold}_burst_interiors']=strong&interior
    report={'pair':pair_label or pair.name,'source_pair':str(pair),'status':'incomplete' if errors else 'complete',
        'matching_by_deramp':{str(k):v for k,v in params_by_mode.items()},
        'same_support_owners_centers_geometry':True,'only_matching_parameter_changed':'deramp',
        'selection':'Fixed latest fully supported burst per swath, before either mode; no agreement gate or quality fallback.',
        'difference_sign':'IW2 minus IW1; changes are deramp1 minus deramp0; [azimuth, range] reference SLC pixels',
        'groups':{name:cohort_summary(views_by_mode,mask) for name,mask in cohorts.items()},
        'job_errors':errors,'limitations':'Swath agreement is not accuracy. Common-quality groups can lose observations; inspect fixed-support and baseline-cohort attrition. Windows overlap spatially. No independent displacement truth.'}
    np.savez_compressed(out/'comparison_masks.npz',center_line=saved['center_line'],center_sample=saved['center_sample'],
                        **cohorts)
    write_json(out/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained',sharex=True)
    line=saved['center_line'];strong=cohorts['all_four_peak_ge_0.3_burst_interiors']
    for col,(axis,k) in enumerate([('range',1),('azimuth',0)]):
        for mode,views in views_by_mode.items():
            delta=(views['iw2']['total']-views['iw1']['total'])[...,k]
            available=support&np.isfinite(delta);bad=available&~dual_valid(views)
            ax=axes[mode,col]
            ax.scatter(line[available],delta[available],s=12,color='.75',label='Finite raw / fixed support')
            ax.scatter(line[bad],delta[bad],s=20,marker='x',color='red',label='Rejected in this mode')
            ax.scatter(line[strong],delta[strong],s=18,label='Same four-peak >=0.3 interior cohort')
            ax.set(title=f'deramp={mode}: {axis}; valid {int((support&dual_valid(views)).sum())}/{int(support.sum())}',
                   ylabel='IW2 minus IW1 [px]')
            ax.axhline(0,color='black',lw=.8);ax.legend(fontsize=7)
        ax=axes[2,col]
        for swath in ('iw1','iw2'):
            delta=(b[swath]['total']-a[swath]['total'])[...,k]
            ax.scatter(line[common&interior],delta[common&interior],s=14,label=swath.upper())
        ax.axhline(0,color='black',lw=.8)
        ax.set(title=f'Change in each swath: {axis}',xlabel='Global reference azimuth line',ylabel='deramp1 minus deramp0 [px]')
        ax.legend(fontsize=8)
    fig.suptitle(f'{pair_label or pair.name}: unspliced, fixed supported patches; disagreement does not establish accuracy')
    fig.savefig(out/'deramp_comparison.png',dpi=160);plt.close(fig)
    return report


def write_mode_outputs(out, pair, params, saved, views, overlap, source_numbers, interior, distance, pair_label=None):
    yy,xx=saved['center_line'],saved['center_sample']
    a,b=views['iw1'],views['iw2'];dual=a['support']&b['support'];accepted=dual&a['valid']&b['valid']
    delta=b['total']-a['total']
    report={'pair':pair_label or pair.name,'source_pair':str(pair),'status':'matched','matching':params,
        'eligible_geometry_centers':int(overlap.sum()),'dual_input_supported':int(dual.sum()),
        'dual_native_valid':int(accepted.sum()),'global_center_samples':np.unique(xx[dual]).tolist(),
        'dual_supported_away_from_along_track_joins':int(interior.sum()),
        'geometry_center_separation_m':statistics(distance[...,0][dual]),
        'geometry_max_corner_separation_m':statistics(distance[...,1:].max(axis=-1)[dual]),
        'selection':'latest fully supported native burst per swath, chosen before matching quality; no agreement selection',
        'difference_sign':'IW2 minus IW1; azimuth/range reference SLC pixels',
        'groups':{},'by_burst_pair':[],
        'limitations':'One acquisition pair; narrow shared coverage; spatially correlated samples; no independent displacement truth or bias correction.'}
    for label,threshold in [('all_native_valid',0),('both_peak_ge_0.3',.3),('both_peak_ge_0.5',.5)]:
        mask=accepted&(a['peak']>=threshold)&(b['peak']>=threshold)
        entry={'n':int(mask.sum()),'difference_px':{axis:statistics(delta[...,k][mask]) for k,axis in enumerate(('azimuth','range'))}}
        merged_good=mask&saved['valid'];entry['merged_valid_n']=int(merged_good.sum())
        merged=np.stack([saved[f'total/{axis}_px'] for axis in ('azimuth','range')],axis=-1)
        entry['merged_minus_swath_px']={swath:{axis:statistics((merged-v['total'])[...,k][merged_good])
            for k,axis in enumerate(('azimuth','range'))} for swath,v in views.items()}
        report['groups'][label]=entry
        interior_good=mask&interior
        report['groups'][label+'_burst_interiors']={'n':int(interior_good.sum()),
            'difference_px':{axis:statistics(delta[...,k][interior_good])
                             for k,axis in enumerate(('azimuth','range'))}}
    for i,j in sorted(set(zip(a['owner'][accepted],b['owner'][accepted]))):
        mask=accepted&(a['owner']==i)&(b['owner']==j)
        strong=mask&(a['peak']>=.3)&(b['peak']>=.3)
        report['by_burst_pair'].append({'bursts':[source_numbers[int(i)],source_numbers[int(j)]],
            'native_valid':int(mask.sum()),'strong_n':int(strong.sum()),
            'difference_px':{axis:statistics(delta[...,k][strong]) for k,axis in enumerate(('azimuth','range'))}})
    details={f'{swath}_{field}':v[field] for swath,v in views.items()
             for field in ('valid','finite','search_boundary','native_completed','snr')}
    np.savez_compressed(out/'observations.npz',center_line=yy,center_sample=xx,dual_support=dual,burst_interior=interior,
        dual_valid=accepted,iw1_total=a['total'],iw2_total=b['total'],iw1_peak=a['peak'],iw2_peak=b['peak'],
        iw1_owner=a['owner'],iw2_owner=b['owner'],geometry_distance_m=distance,merged_total=merged,**details)
    write_json(out/'report.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    strong=accepted&(a['peak']>=.3)&(b['peak']>=.3)
    for col,(axis,k) in enumerate([('range',1),('azimuth',0)]):
        ax=axes[0,col]
        for view,label,color in [(a,'IW1','tab:blue'),(b,'IW2','tab:orange')]:
            ax.scatter(yy[accepted],view['total'][...,k][accepted],s=12,alpha=.65,label=label,color=color)
        ax.scatter(yy[accepted&saved['valid']],merged[...,k][accepted&saved['valid']],s=8,color='.25',alpha=.5,label='Merged result')
        ax.set(title=f'Independent observations: {axis}',xlabel='Global reference azimuth line',ylabel='Offset [SLC pixels]');ax.legend()
        ax=axes[1,col]
        ax.scatter(yy[accepted&~strong],delta[...,k][accepted&~strong],s=12,color='.75',label='Other native-valid')
        points=ax.scatter(yy[strong],delta[...,k][strong],s=18,c=xx[strong],cmap='viridis',label='Both peaks >= 0.3')
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'IW2 minus IW1: {axis}',xlabel='Global reference azimuth line',ylabel='Difference [SLC pixels]')
        ax.legend(fontsize=8)
        if strong.any():fig.colorbar(points,ax=ax,label='Global reference range sample',shrink=.75)
    fig.suptitle(f'{pair_label or pair.name}: same radar centers, full chip/search support, unspliced aligned bursts')
    fig.savefig(out/'swath_overlap.png',dpi=160);plt.close(fig)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--compare-deramp', action='store_true',
                        help='Run paired deramp=0 and deramp=1 on fixed unspliced supported patches')
    parser.add_argument('--acquisition-dates',nargs=2,metavar=('EARLIER','LATER'),
                        help='Use these aligned dates in the same stack on the saved grid; omit merged-result comparisons')
    args = parser.parse_args(argv)
    pair, out = args.pair.resolve(), args.output.resolve();stack = pair.parent.parent
    if out.exists() or out == stack or stack in out.parents or out in stack.parents:
        raise ValueError('Choose a new output directory outside the source stack')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s');gdal.UseExceptions()
    manifest = read_manifest(pair);params = deepcopy(manifest['matching']['matching'])
    params['engine'] = 'isce3'
    gross = manifest['matching']['gross_offsets']
    if gross['mode'] != 'zero':raise ValueError('This comparison requires zero gross offsets')
    with h5py.File(pair/'radar/offsets.h5') as h:
        saved = {k: h[k][:] for k in ('center_line','center_sample','valid','input_valid',
            'source_burst_index','total/azimuth_px','total/range_px','correlation_peak')}
    params,saved,days=comparison_reference(params,saved,manifest['stack_pair']['acquisition_dates'],
                                          args.acquisition_dates,args.compare_deramp)
    modes=(0,1) if args.compare_deramp else (0,)
    params_by_mode={mode:dict(deepcopy(params),deramp=mode) for mode in modes}
    mode_dirs={mode:out/f'deramp-{mode}' if args.compare_deramp else out for mode in modes}
    dates=[read_manifest(stack/'acquisitions'/day) for day in days]
    pair_label='_'.join(days)
    provenance={'acquisition_dates':days,'grid_template_pair':str(pair),
                'merged_reference_available':args.acquisition_dates is None,
                'acquisition_burst_times':{day:date.get('burst_times',{}) for day,date in zip(days,dates)}}
    if args.acquisition_dates and any(date.get('status')!='complete' for date in dates):
        raise ValueError('Requested aligned acquisitions must be complete')
    if args.acquisition_dates:
        frame=read_manifest(stack/'geometry');expected=frame.get('geometry_identity')
        if not expected or manifest['stack_pair'].get('geometry_identity')!=expected or any(
                date.get('specification',{}).get('geometry')!=expected for date in dates):
            raise ValueError('Requested acquisitions and grid template must share the current stack geometry')
    yy, xx = saved['center_line'], saved['center_sample']
    starts_y = (yy[:,0]-(params['window'][0]-1)/2).astype('int32')
    starts_x = (xx[0]-(params['window'][1]-1)/2).astype('int32')
    shape = yy.shape;sources = [e for e in manifest['assembly']['sources'] if e['burst_id'].endswith(('_iw1','_iw2'))]
    masks = {}
    for e in sources:
        dy,dx=e['offset'];a,b,c,d=e['valid_window'];halo=params['search'];window=params['window']
        masks[e['burst_id']] = ((starts_y[:,None]-dy-halo[0]>=a)&
            (starts_y[:,None]-dy+window[0]+halo[0]<=b)&
            (starts_x[None,:]-dx-halo[1]>=c)&(starts_x[None,:]-dx+window[1]+halo[1]<=d))
    unions = {swath:np.logical_or.reduce([masks[e['burst_id']] for e in sources if e['burst_id'].endswith(swath)]) for swath in ('iw1','iw2')}
    overlap=unions['iw1']&unions['iw2']
    if not overlap.any():raise ValueError('No complete shared chip/search footprint on the saved matching grid')
    watched={pair/'manifest.json',pair/'radar/offsets.h5',stack/'geometry/manifest.json',stack/'stack_manifest.json'}
    watched.update(stack/'acquisitions'/day/'manifest.json' for day in days)
    watched.update(Path(p) for date in dates for p in date['alignment_products'])
    watched.update(Path(p) for e in sources for p in e['coreg']['geometry'].values())
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
    out.mkdir(parents=True)
    for directory in mode_dirs.values():directory.mkdir(exist_ok=True)
    write_json(out/'report.json',{'status':'preparing','source_pair':str(pair),'compare_deramp':args.compare_deramp})
    checksums=ChecksumCache();jobs=[];records={}
    geod=Geod(ellps='WGS84')
    for e in sources:
        key=e['burst_id'];selected=masks[key]&overlap
        rows=np.flatnonzero(selected.any(axis=1));cols=np.flatnonzero(selected.any(axis=0))
        if not rows.size or not cols.size:continue
        # Keep consecutive indices so native strides represent precisely the saved grid.
        rows=np.arange(rows[0],rows[-1]+1);cols=np.arange(cols[0],cols[-1]+1)
        dy,dx=e['offset'];grid=MatchingGrid(starts_y[rows]-dy,starts_x[cols]-dx,
            tuple(params['window']),tuple(params['search']),tuple(params['stride']))
        line,sample=grid.centers
        np.testing.assert_array_equal(line+dy,yy[np.ix_(rows,cols)])
        np.testing.assert_array_equal(sample+dx,xx[np.ix_(rows,cols)])
        paths=[date['slcs'][key] for date in dates]
        for date,path in zip(dates,paths):
            if checksums(path)!=date['alignment_products'][str(Path(path).resolve())]:
                raise ValueError(f'Aligned input checksum mismatch: {path}')
        support=valid_windows(*(gdal.Open(p) for p in paths),grid,(0,0))&selected[np.ix_(rows,cols)]
        transformer=Transformer.from_crs(e['coreg']['geometry_crs'],4326,always_xy=True)
        geometries=[]
        # Center plus all four reference-chip corners verify coincident patches.
        half=[(v-1)/2 for v in grid.window]
        for a,b in [(0,0),(-half[0],-half[1]),(-half[0],half[1]),(half[0],-half[1]),(half[0],half[1])]:
            gx=sample_geometry(e['coreg']['geometry']['x'],line+a,sample+b)
            gy=sample_geometry(e['coreg']['geometry']['y'],line+a,sample+b)
            geometries.append(np.stack(transformer.transform(gx,gy),axis=-1))
        records[key]={'entry':e,'rows':rows,'cols':cols,'support':support,'geometry':np.stack(geometries,axis=-2),'results':{}}
        for mode in modes:
            native_args=(*paths,mode_dirs[mode]/'native'/key,params_by_mode[mode],{'gross_offsets':gross},'cpu')
            jobs.append(((mode,key),native_args,{'_grid':grid,'_valid_input':support}))
        logging.info('%s: %d fully supported overlap windows',key,int(support.sum()))
    selection,source_numbers=select_views(sources,records,shape)
    dual=selection['iw1']['support']&selection['iw2']['support']
    if not dual.any():raise ValueError('No dual input-supported patches after actual raster masks')
    interior=burst_interior_mask(sources,yy,xx,{s:v['owner'] for s,v in selection.items()},dual,
                                 params['window'],params['search'])
    a,b=selection['iw1'],selection['iw2']
    _,_,distance=geod.inv(a['geometry'][...,0],a['geometry'][...,1],b['geometry'][...,0],b['geometry'][...,1])
    if not np.isfinite(distance[dual]).all():raise ValueError('Nonfinite geometry at a shared patch')
    np.savez_compressed(out/'selection.npz',center_line=yy,center_sample=xx,dual_support=dual,
        burst_interior=interior,iw1_owner=a['owner'],iw2_owner=b['owner'],geometry_distance_m=distance)
    errors=[];completed=[]
    write_json(out/'report.json',{'status':'matching','source_pair':str(pair),'compare_deramp':args.compare_deramp,
                                'input_supported_n':int(dual.sum()),'job_count':len(jobs)})
    with _native_executor(min(args.workers,len(jobs)),single_threaded=True) as pool:
        for (mode,key),result,error in pool.imap_unordered(run_job,jobs):
            record=records[key]
            if error:
                errors.append({'deramp':mode,'burst':key,'error':error})
                logging.error('deramp=%d %s: %s',mode,key,error)
            else:
                np.testing.assert_array_equal(result['input_valid'],record['support'])
                local_line,local_sample=result['grid'].centers;dy,dx=record['entry']['offset']
                np.testing.assert_array_equal(local_line+dy,yy[np.ix_(record['rows'],record['cols'])])
                np.testing.assert_array_equal(local_sample+dx,xx[np.ix_(record['rows'],record['cols'])])
                record['results'][mode]=result
                arrays={field:result[field] for field in ('total','gross','residual','valid','input_valid','finite','search_boundary','covariance')}
                np.savez_compressed(mode_dirs[mode]/f'{key}.npz',rows=record['rows'],cols=record['cols'],
                    center_line=local_line+dy,center_sample=local_sample+dx,**arrays,
                    peak=result['correlation_peak'],snr=result['snr'],geometry_lonlat=record['geometry'],
                    effective_json=json.dumps(result['effective_parameters']))
                logging.info('deramp=%d %s: %d/%d supported windows native-valid',mode,key,
                             int(result['valid'].sum()),int(record['support'].sum()))
            completed.append({'deramp':mode,'burst':key,'status':'failed' if error else 'complete'})
            write_json(out/'progress.json',{'completed_jobs':completed,'job_count':len(jobs),'job_errors':errors})
    views_by_mode={mode:attach_results(selection,records,source_numbers,mode) for mode in modes}
    mode_reports={}
    for mode,views in views_by_mode.items():
        report=write_mode_outputs(mode_dirs[mode],pair,params_by_mode[mode],saved,views,overlap,source_numbers,interior,distance,pair_label)
        report.update(provenance)
        report['job_errors']=[e for e in errors if e['deramp']==mode]
        report['status']='incomplete' if report['job_errors'] else 'complete'
        report['source_burst_ids']=source_numbers
        mode_reports[mode]=report
        write_json(mode_dirs[mode]/'report.json',report)
    if args.compare_deramp:
        report=compare_deramp_modes(out,pair,params_by_mode,views_by_mode,saved,interior,errors,pair_label)
    report.update(provenance)
    unchanged=all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in before.items())
    for mode,mode_report in mode_reports.items():
        mode_report['production_sizes_mtimes_unchanged']=unchanged
        if not unchanged:mode_report['status']='incomplete'
        write_json(mode_dirs[mode]/'report.json',mode_report)
    report['production_sizes_mtimes_unchanged']=unchanged
    report['source_burst_ids']=source_numbers
    report['verified_aligned_inputs']={day:{key:{'path':date['slcs'][key],
        'sha256':date['alignment_products'][str(Path(date['slcs'][key]).resolve())]}
        for key in records} for day,date in zip(days,dates)}
    if not unchanged:report['status']='incomplete'
    write_json(out/'report.json',report)
    if not unchanged:raise ValueError('Source files changed during comparison')
    logging.info('Completed swath comparison: %s',out)
    return 1 if errors else 0


if __name__=='__main__':raise SystemExit(main())
