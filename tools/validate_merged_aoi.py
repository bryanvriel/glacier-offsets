#!/usr/bin/env python3
"""Validate assembled tracking from a completed AOI snapshot, without raw SAFE I/O.

The saved selection is explicit input. This is not fresh AOI/orbit/DEM validation.
Production CLI commands perform their normal source-input preflight separately.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time

import h5py
import numpy as np

from glacier_offsets.assembly import assemble
from glacier_offsets.ampcor import match, native_match, _native_executor
from glacier_offsets.config import load_config, gross_values
from glacier_offsets.diagnostics import diagnostics
from glacier_offsets.geocode import geocode
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import checksum, write_json


def snapshot(paths):
    return {str(p): {'sha256':checksum(p),'mtime_ns':p.stat().st_mtime_ns} for p in paths}


def boundary_controls(cfg, root, manifest, directory):
    with h5py.File(root/'radar/offsets.h5') as h:
        line,sample=h['center_line'][:],h['center_sample'][:]
        starts_y=(line[:,0]-(h.attrs['window'][0]-1)/2).astype('int32')
        starts_x=(sample[0]-(h.attrs['window'][1]-1)/2).astype('int32')
        total=np.stack([h[f'total/{axis}_px'][:] for axis in ('azimuth','range')],axis=-1)
        owner=h['source_burst_index'][:];crossing=h['seam_crossing'][:];valid=h['valid'][:]
        peak=h['correlation_peak'][:];snr=h['snr'][:]
        window,search,stride=[tuple(h.attrs[key]) for key in ('window','search','stride')]
    gross=gross_values(cfg)
    reports=[]
    sources=manifest['assembly']['sources']
    with _native_executor() as worker:
        for boundary in range(1,len(sources)):
            cut=sources[boundary]['offset'][0]+sources[boundary]['valid_window'][0]
            for index in (boundary-1,boundary):
                entry=sources[index];dy,dx=entry['offset'];shape=entry['coreg']['reference_grid']
                yy=starts_y-dy;xx=starts_x-dx
                ir=np.flatnonzero((np.abs(line[:,0]-cut)<=128)&(yy>=max(0,search[0]-gross[0]))&
                                  (yy<=min(shape['length']-window[0],shape['length']-window[0]-search[0]-gross[0])))
                ic=np.flatnonzero((xx>=max(0,search[1]-gross[1]))&
                                  (xx<=min(shape['width']-window[1],shape['width']-window[1]-search[1]-gross[1])))
                if not len(ir) or not len(ic):continue
                grid=MatchingGrid(yy[ir],xx[ic],window,search,stride)
                name=f'boundary-{boundary}-{entry["burst_id"]}'
                r=native_match(entry['coreg']['slcs']['reference'],entry['coreg']['slcs']['secondary'],
                               root/'validation-native'/name,cfg['matching'],cfg,cfg['matching']['backend'],
                               _grid=grid,_worker=worker)
                section=np.ix_(ir,ic)
                assembled=total[section];same_source=(owner[section]==index+1)&~crossing[section]
                common=valid[section]&r['valid']
                control=common&same_source
                exact=bool(np.array_equal(assembled[control],r['total'][control],equal_nan=True))
                if not control.any() or not exact:
                    raise AssertionError(f'{name}: unchanged-source native control failed ({control.sum()} windows)')
                strong=common&crossing[section]&(snr[section]>=5)&(peak[section]>=.2)&(r['snr']>=5)&(r['correlation_peak']>=.2)
                difference=assembled-r['total']
                stats={}
                for k,axis in enumerate(('azimuth','range')):
                    values=difference[...,k][strong]
                    stats[axis]={'count':int(values.size),'median_px':float(np.median(values)) if values.size else None,
                                  'p95_abs_px':float(np.quantile(np.abs(values),.95)) if values.size else None,
                                  'max_abs_px':float(np.max(np.abs(values))) if values.size else None}
                np.savez_compressed(directory/f'{name}.npz',center_line=line[section],center_sample=sample[section],
                                    original_total=r['total'],assembled_total=assembled,original_valid=r['valid'],
                                    assembled_valid=valid[section],pure_source_control=control,
                                    quality_crossing_comparable=strong,source_burst_index=owner[section])
                reports.append({'boundary_global_line':cut,'source_burst_id':entry['burst_id'],
                                'same_source_controls':int(control.sum()),'same_source_offsets_exact':exact,
                                'quality_crossing_comparisons':stats,
                                'quality_crossing_changed_over_quarter_pixel':int((strong&(np.max(np.abs(difference),axis=-1)>.25)).sum())})
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--from-completed-bursts',action='store_true',required=True,
                        help='Explicitly use the saved selection and existing COMPASS products')
    parser.add_argument('--report-directory',type=Path,required=True)
    parser.add_argument('--output-directory',type=Path,help='Optional separate output parent for merged/')
    parser.add_argument('--skip-boundary-controls',action='store_true')
    args=parser.parse_args()
    cfg=load_config(args.config)
    if cfg['gross_offsets']['mode']=='velocity':
        raise ValueError('This source-snapshot validator supports zero/constant guidance; fresh velocity guidance requires original orbit metadata')
    source_root=Path(cfg['output']['directory'])
    selection=json.loads((source_root/'selection.json').read_text())
    ids=selection['selected_ids']
    if len(ids)<2:raise ValueError('At least two completed selected bursts required')
    sources=[(source_root/key,json.loads((source_root/key/'manifest.json').read_text())) for key in ids]
    directory=args.report_directory.resolve();directory.mkdir(parents=True,exist_ok=True)
    original=[p for source,_ in sources for p in [source/'manifest.json',source/'radar/offsets.h5',
              *(source/'geo').glob('*.tif'),source/'geo/center_geometry.h5'] if p.is_file()]
    before=snapshot(original)
    if args.output_directory:cfg['output']['directory']=str(args.output_directory.resolve())
    effective=deepcopy(cfg);effective.pop('aoi',None);effective['inputs']['burst_id']='merged'
    report={'scope':'saved selection and completed COMPASS products; no raw SAFE/orbit/DEM/AOI reread',
            'source_selection_identity':selection['selection_identity'],'selected_ids':ids,
            'status':'running','timings_seconds':{}}
    write_json(directory/'validation.json',report)
    start=time.perf_counter()
    root,manifest=assemble(cfg,sources)
    report['timings_seconds']['assembly']=time.perf_counter()-start
    for name,fn in [('match',match),('geocode',geocode),('diagnostics',diagnostics)]:
        start=time.perf_counter();fn(effective,root,manifest)
        report['timings_seconds'][name]=time.perf_counter()-start
        print(name,report['timings_seconds'][name],flush=True)
    report.update(merged_directory=str(root),assembly_identity=manifest['coreg_identity'],
                  matching=manifest['matching_summary'],geocoding=manifest['geocoding'])
    report['native_worker_peak_rss_mib']=manifest['matching']['native_parameters'].get('worker_peak_rss_bytes',0)/1024**2
    write_json(directory/'validation.json',report)
    if not args.skip_boundary_controls:
        report['boundary_controls']=boundary_controls(effective,root,manifest,directory)
    products=[root/'radar/offsets.h5',root/'radar/matching_grid.h5',*(root/'geo').iterdir()]
    science=snapshot(products)
    start=time.perf_counter()
    _,restart=assemble(cfg,sources);match(effective,root,restart);geocode(effective,root,restart)
    report['timings_seconds']['restart']=time.perf_counter()-start
    report['restart_checksums_and_mtimes_unchanged']=science==snapshot(products)
    report['original_burst_products_unchanged']=before==snapshot(original)
    if not report['restart_checksums_and_mtimes_unchanged'] or not report['original_burst_products_unchanged']:
        raise AssertionError('Restart or original-product preservation failed')
    report['parent_peak_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024)
    report['status']='passed; independent absolute accuracy and fresh raw-input workflow not assessed'
    write_json(directory/'validation.json',report)
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
