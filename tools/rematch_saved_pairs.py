#!/usr/bin/env python3
"""Rematch selected saved stack pairs in a fresh directory, reusing aligned bursts.
Run in the processing (ISCE3) environment. No SAFE discovery or COMPASS calls.
"""
import argparse
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path,help='Stack YAML, optionally inheriting the production YAML')
    p.add_argument('--pair',action='append',required=True,help='EARLIER_LATER saved pair; repeat for multiple dates')
    p.add_argument('--output',type=Path,required=True,help='Fresh experiment directory outside the production stack')
    p.add_argument('--window',nargs=2,type=int,metavar=('AZIMUTH','RANGE'))
    p.add_argument('--search',nargs=2,type=int,metavar=('AZIMUTH','RANGE'))
    p.add_argument('--stride',nargs=2,type=int,metavar=('AZIMUTH','RANGE'))
    p.add_argument('--first-center',nargs=2,type=float,metavar=('LINE','SAMPLE'))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dry-run',action='store_true',help='Print settings and missing inputs without writes or native matching')
    return p


def outside(output,stack):
    if output==stack or stack in output.parents or output in stack.parents:
        raise ValueError('Output must be outside the production stack and cannot contain it')


def select_settings(cfg,args):
    result=deepcopy(cfg)
    for key in ('window','search','stride','first_center'):
        value=getattr(args,key)
        if value is not None:result['matching'][key]=value
    result['matching']['backend']='cpu'
    if result['matching'].get('engine','isce3')!='isce3':raise ValueError('This runner supports ISCE3 CPU only')
    if result['gross_offsets']['mode']=='velocity':
        raise ValueError('Velocity-prior rematching is not implemented; use the saved zero/constant gross offsets')
    for key in ('window','search','stride'):
        value=result['matching'][key]
        if len(value)!=2 or any(type(v)is not int or v<=0 for v in value):raise ValueError(f'Invalid matching.{key}')
    m=result['matching'];raw=m['raw_oversampling']
    if m['zoom']<2*raw or m['zoom']>2*raw*min(m['search']):raise ValueError('zoom does not fit the requested search')
    if m.get('range_oversampling',1)==2 and m['deramp']!=0:raise ValueError('range_oversampling=2 requires deramp=0')
    return result


def comparison(source,selected):
    """Compare shared physical centers; IDs and seam support may change with window size."""
    names=['center_line','center_sample','input_valid','valid','correlation_peak','total/azimuth_px','total/range_px']
    with h5py.File(source) as h:a={k:h[k][:].ravel() for k in names}
    with h5py.File(selected) as h:b={k:h[k][:].ravel() for k in names}
    def coords(d):return np.rec.fromarrays([d['center_line'],d['center_sample']],names='line,sample')
    _,ia,ib=np.intersect1d(coords(a),coords(b),return_indices=True)
    common=a['input_valid'][ia].astype(bool)&b['input_valid'][ib].astype(bool)
    q={'shared_centers':len(ia),'common_input_valid':int(common.sum()),'baseline_input_valid':int(a['input_valid'].sum()),'trial_input_valid':int(b['input_valid'].sum()),'peak_checks':{}}
    for threshold in [0,.06,.1,.2]:
        q['peak_checks'][str(threshold)]={'baseline_on_common_support':int(np.sum(common&(a['correlation_peak'][ia]>=threshold))),'trial_on_common_support':int(np.sum(common&(b['correlation_peak'][ib]>=threshold)))}
    q['offset_difference_on_common_input_support_px']={}
    for axis in ['azimuth','range']:
        d=b['total/'+axis+'_px'][ib]-a['total/'+axis+'_px'][ia];v=d[common&np.isfinite(d)]
        q['offset_difference_on_common_input_support_px'][axis]={'n':len(v),'median':float(np.median(v)) if len(v) else None,'p90_absolute':float(np.quantile(abs(v),.9)) if len(v) else None}
    q['interpretation']='Higher peaks, smaller differences or smoother maps do not establish accuracy. Window size changes peak statistics, footprint, support and spatial dependence.'
    return q


def run(args):
    from glacier_offsets.stack_config import load_status_config,MERGE_FIRST
    from glacier_offsets.matching_grid import build_grid
    cfg=select_settings(load_status_config(args.config),args)
    stack=Path(cfg['output']['directory']).resolve();out=args.output.resolve();outside(out,stack)
    if args.workers<1:raise ValueError('workers must be positive')
    if out.exists():raise ValueError('Choose a fresh output directory; existing experiments are never overwritten')
    if cfg['stack']['processing'] not in MERGE_FIRST:raise ValueError('This runner requires a saved merge-first stack')
    records=[];dependencies=set();missing=[]
    for name in dict.fromkeys(args.pair):
        # Pair arguments must name directories directly beneath stack/pairs.
        if '/' in name or '\\' in name or name in ('.','..'):raise ValueError('Invalid pair ID')
        pair=stack/'pairs'/name;mp=pair/'manifest.json'
        if not mp.is_file():raise ValueError(f'Missing saved pair: {mp}')
        original=json.loads(mp.read_text());days=original['stack_pair']['acquisition_dates']
        if name!='_'.join(days):raise ValueError('Pair ID disagrees with saved dates')
        if cfg['gross_offsets']!=original['matching']['gross_offsets']:raise ValueError('Keep the saved gross-offset settings')
        shape=tuple(original['coreg']['reference_grid'][k] for k in ['length','width']);m=cfg['matching']
        grid=build_grid(shape,shape,m['window'],m['search'],m['stride'],first_center=m.get('first_center'))
        dependencies.update([mp,pair/'radar/offsets.h5',stack/'geometry/manifest.json'])
        for day in days:
            ap=stack/'acquisitions'/day/'manifest.json';dependencies.add(ap)
            if ap.is_file():
                date=json.loads(ap.read_text());dependencies.update(Path(p) for p in date['slcs'].values())
        records.append((name,pair,original,days,grid.shape))
    missing=[str(p) for p in sorted(dependencies) if not p.is_file()]
    plan={'production_stack':str(stack),'output':str(out),'matching':cfg['matching'],'workers':args.workers,'pairs':[{'pair_id':r[0],'grid_shape':list(r[4])} for r in records],'missing_inputs':missing,'input_policy':'Retained original aligned per-burst SLCs and reference geometry; production is read-only. Merged SLCs are reconstructed only in experiment/_inputs.'}
    if args.dry_run:
        print(json.dumps(plan,indent=2));return 2 if missing else 0
    if missing:raise ValueError('Missing required retained products: '+', '.join(missing[:5]))
    from osgeo import gdal
    from glacier_offsets.ampcor import match
    from glacier_offsets.geocode import geocode
    from glacier_offsets.diagnostics import diagnostics
    from glacier_offsets.products import ChecksumCache,identity,write_json
    from glacier_offsets.stack_storage import assemble_date,read_manifest,release_date
    from glacier_offsets.stack_lock import stack_lock
    gdal.UseExceptions();os.environ['OMP_NUM_THREADS']=str(args.workers)
    # Lock production against concurrent writers, without creating or changing scientific files.
    with stack_lock(stack):
        frame=read_manifest(stack/'geometry');cache=ChecksumCache();dates={}
        for name,pair,original,days,shape in records:
            if frame['geometry_identity']!=original['stack_pair']['geometry_identity']:raise ValueError('Pair and current geometry identities differ')
            for key in ('burst_id','offset','valid_window','ownership_windows'):
                if [s.get(key) for s in frame['sources']] != [s.get(key) for s in original['assembly']['sources']]:
                    raise ValueError('Saved pair differs from current reference layout: '+key)
            for day in days:dates[day]=read_manifest(stack/'acquisitions'/day)
        recorded={}
        recorded.update(frame['products'])
        for date in dates.values():
            products=date.get('alignment_products',date['products'])
            for p in date['slcs'].values():
                digest=products[str(Path(p).resolve())];recorded[p]=digest
        dependencies.update(Path(p) for p in recorded)
        before={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in dependencies}
        for p,digest in recorded.items():
            if cache(p)!=digest:raise ValueError('Retained input checksum mismatch: '+str(p))
        out.mkdir(parents=True);write_json(out/'experiment.json',plan|{'status':'running','retained_input_checksums':recorded})
        workcfg=deepcopy(cfg);workcfg['output']['directory']=str(out/'_inputs');workcfg['inputs']['burst_id']='stack';workcfg['storage']['keep_merged_slcs']=False
        remaining={day:sum(day in r[3] for r in records) for day in dates};assemblies={};results=[]
        for name,pair,original,days,shape in records:
            for day in days:
                if day not in assemblies:
                    logging.info('Reassembling retained aligned bursts for %s',day)
                    assemblies[day]=assemble_date(workcfg,SimpleNamespace(date=day),frame,dates[day],force=True,checksums=cache)
            destination=out/'pairs'/name;destination.mkdir(parents=True)
            manifest=deepcopy(original)
            for key in ['match_identity','geocode_identity','diagnostics_identity','matching','matching_summary','geocoding','diagnostics']:
                manifest.pop(key,None)
            manifest['coreg_identity']=identity({'saved_coreg':original['coreg_identity'],'experiment_assemblies':[assemblies[d]['assembly_identity'] for d in days]})
            manifest['coreg']['slcs']=dict(zip(['reference','secondary'],[assemblies[d]['slc'] for d in days]))
            manifest['stages']={'coregister':'retained alignment; no COMPASS'}
            manifest['slc_previews']={}
            manifest['experiment']={'source_pair':str(pair),'settings':cfg['matching'],'changed_grid_requires_new_center_geometry':True}
            # Keep the baseline geographic map grid; center geometry is recomputed for trial windows.
            write_json(destination/'manifest.json',manifest)
            logging.info('Matching %s with window %s',name,cfg['matching']['window'])
            match(workcfg,destination,manifest);geocode(workcfg,destination,manifest);diagnostics(workcfg,destination,manifest)
            result=comparison(pair/'radar/offsets.h5',destination/'radar/offsets.h5');write_json(destination/'comparison.json',result);results.append({'pair_id':name,'comparison':result})
            for day in days:
                remaining[day]-=1
                if remaining[day]==0:
                    if not Path(assemblies[day]['slc']).resolve().is_relative_to(out):raise ValueError('Refusing cleanup outside experiment')
                    release_date(workcfg,SimpleNamespace(date=day),assemblies[day])
            manifest['slc_previews']=dict(zip(['reference','secondary'],[assemblies[d].get('slc_preview') for d in days]));write_json(destination/'manifest.json',manifest)
        for name,pair,original,days,shape in records:
            destination=out/'pairs'/name
            saved=read_manifest(destination)
            saved['slc_previews']={side:assemblies[day]['slc_preview'] for side,day in zip(('reference','secondary'),days)}
            write_json(destination/'manifest.json',saved)
        if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in before.items()):raise ValueError('Production inputs changed during experiment')
        write_json(out/'experiment.json',plan|{'status':'complete','retained_input_checksums':recorded,'production_sizes_mtimes_unchanged':True,'results':results})
    return 0


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:raise SystemExit(run(parser().parse_args()))
    except (ValueError,OSError,KeyError,ImportError) as error:raise SystemExit(str(error))
