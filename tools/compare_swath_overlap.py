#!/usr/bin/env python3
"""Compare independent IW1/IW2 burst matches at identical saved radar centers.

Reads retained aligned bursts, never assembled SLCs. Support and the primary
burst choice are established before matching, without selecting for offset
agreement. Native observations, geometry checks and plots are diagnostic only.
"""
import argparse
from copy import deepcopy
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


def run_job(job):
    name, args, kwargs = job
    return name, _native_worker((args, kwargs))


def statistics(values):
    a = np.asarray(values);a = a[np.isfinite(a)]
    if not a.size:
        return {'n': 0}
    return {'n': int(a.size), 'median': float(np.median(a)),
            'p10': float(np.quantile(a,.1)), 'p90': float(np.quantile(a,.9)),
            'median_absolute': float(np.median(abs(a))),
            'p90_absolute': float(np.quantile(abs(a),.9)), 'max_absolute': float(abs(a).max())}


def burst_interior_mask(sources, line, sample, owners, dual, window, search):
    """Exclude chip/search footprints crossing an along-track ownership change."""
    start_y=(line-(window[0]-1)/2).astype('int32')-search[0]
    start_x=(sample-(window[1]-1)/2).astype('int32')-search[1]
    height,width=[window[k]+2*search[k] for k in (0,1)]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    pair, out = args.pair.resolve(), args.output.resolve();stack = pair.parent.parent
    if out.exists() or out == stack or stack in out.parents or out in stack.parents:
        raise ValueError('Choose a new output directory outside the source stack')
    if args.workers < 1:
        raise ValueError('workers must be positive')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s');gdal.UseExceptions()
    manifest = read_manifest(pair);params = deepcopy(manifest['matching']['matching'])
    params['engine'] = 'isce3'
    gross = manifest['matching']['gross_offsets']
    if gross['mode'] != 'zero' or params['deramp'] != 0:
        raise ValueError('This comparison requires saved magnitude-first matching and zero gross offsets')
    days = manifest['stack_pair']['acquisition_dates']
    dates = [read_manifest(stack/'acquisitions'/day) for day in days]
    with h5py.File(pair/'radar/offsets.h5') as h:
        saved = {k: h[k][:] for k in ('center_line','center_sample','valid','input_valid',
            'source_burst_index','total/azimuth_px','total/range_px','correlation_peak')}
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
    out.mkdir(parents=True)
    watched={pair/'manifest.json',pair/'radar/offsets.h5',stack/'geometry/manifest.json',stack/'stack_manifest.json'}
    watched.update(stack/'acquisitions'/day/'manifest.json' for day in days)
    watched.update(Path(p) for date in dates for p in date['alignment_products'])
    watched.update(Path(p) for e in sources for p in e['coreg']['geometry'].values())
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in watched}
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
        records[key]={'entry':e,'rows':rows,'cols':cols,'support':support,'geometry':np.stack(geometries,axis=-2)}
        native_args=(*paths,out/'native'/key,params,{'gross_offsets':gross},'cpu')
        jobs.append((key,native_args,{'_grid':grid,'_valid_input':support}))
        logging.info('%s: %d fully supported overlap windows',key,int(support.sum()))
    with _native_executor(min(args.workers,len(jobs)),single_threaded=True) as pool:
        for key,result in pool.imap_unordered(run_job,jobs):
            record=records[key];record['result']=result
            np.savez_compressed(out/f'{key}.npz',rows=record['rows'],cols=record['cols'],
                total=result['total'],valid=result['valid'],input_valid=result['input_valid'],
                peak=result['correlation_peak'],snr=result['snr'],geometry_lonlat=record['geometry'],
                effective_json=json.dumps(result['effective_parameters']))
            logging.info('Matched %s: %d native-valid windows',key,int(result['valid'].sum()))
    # Choose each swath's latest fully supported burst BEFORE considering native
    # quality or offset values. Retain every burst observation in the NPZ files.
    views={s:{'owner':np.zeros(shape,'int16'),'support':np.zeros(shape,bool),
        'valid':np.zeros(shape,bool),'total':np.full((*shape,2),np.nan),
        'peak':np.full(shape,np.nan),'geometry':np.full((*shape,5,2),np.nan)} for s in ('iw1','iw2')}
    source_numbers={}
    for i,e in enumerate(sorted(sources,key=lambda e:(e['offset'][0],e['burst_id'])),1):
        key=e['burst_id'];source_numbers[i]=key
        if key not in records:continue
        r=records[key];v=views[key[-3:]];section=np.ix_(r['rows'],r['cols']);take=r['support']
        for field,value in [('owner',np.full(take.shape,i)),('support',take),('valid',r['result']['valid']),
                            ('total',r['result']['total']),('peak',r['result']['correlation_peak']),('geometry',r['geometry'])]:
            block=v[field][section];block[take]=value[take];v[field][section]=block
    a,b=views['iw1'],views['iw2'];dual=a['support']&b['support'];accepted=dual&a['valid']&b['valid']
    interior=burst_interior_mask(sources,yy,xx,{s:v['owner'] for s,v in views.items()},dual,
                                 params['window'],params['search'])
    _,_,distance=geod.inv(a['geometry'][...,0],a['geometry'][...,1],b['geometry'][...,0],b['geometry'][...,1])
    finite_geometry=np.isfinite(distance).all(axis=-1)
    if not finite_geometry[dual].all():raise ValueError('Nonfinite geometry at a shared patch')
    delta=b['total']-a['total']
    report={'pair':pair.name,'source_pair':str(pair),'status':'matched','matching':params,
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
    np.savez_compressed(out/'observations.npz',center_line=yy,center_sample=xx,dual_support=dual,burst_interior=interior,
        dual_valid=accepted,iw1_total=a['total'],iw2_total=b['total'],iw1_peak=a['peak'],iw2_peak=b['peak'],
        iw1_owner=a['owner'],iw2_owner=b['owner'],geometry_distance_m=distance,merged_total=merged)
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
    fig.suptitle(f'{pair.name}: same radar centers, full chip/search support, unspliced aligned bursts')
    fig.savefig(out/'swath_overlap.png',dpi=160);plt.close(fig)
    unchanged=all((p.stat().st_size,p.stat().st_mtime_ns)==stamp for p,stamp in before.items())
    if not unchanged:raise ValueError('Source files changed during comparison')
    report['production_sizes_mtimes_unchanged']=unchanged
    report['status']='complete'
    write_json(out/'report.json',report)
    logging.info('Completed swath comparison: %s',out)


if __name__=='__main__':main()
