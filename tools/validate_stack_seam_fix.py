#!/usr/bin/env python3
"""Full-AOI interior-ownership validation using existing aligned stack bursts.

Writes an isolated validation directory, never migrates a production stack and
never runs COMPASS. Both matching and geocoding use the production code paths.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from osgeo import gdal

from glacier_offsets.ampcor import match
from glacier_offsets.config import load_defaults
from glacier_offsets.geocode import geocode
from glacier_offsets.products import identity, write_json
from glacier_offsets.stack_ownership import POLICY
from glacier_offsets.stack_pairs import common_map_grid, prepared_pair
from glacier_offsets.stack_storage import METHOD, assemble_date, product_hashes, write_reference_layout


def plot_comparison(source, destination, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cmap=plt.get_cmap('RdBu_r').copy(); cmap.set_bad('.75')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    with h5py.File(source/'radar/offsets.h5') as old,h5py.File(destination/'radar/offsets.h5') as new:
        y,x=new['center_line'][:],new['center_sample'][:]
        for row,(axis,limit) in enumerate([('azimuth',6),('range',4)]):
            a,b=(h[f'total/{axis}_px'][:] for h in (old,new))
            valid_a,valid_b=(h['valid'][:] for h in (old,new))
            fields=[np.where(valid_a,a,np.nan),np.where(valid_b,b,np.nan),
                    np.where(valid_a&valid_b,b-a,np.nan)]
            for col,(field,title) in enumerate(zip(fields,['Previous splice','Interior splice','Interior − previous'])):
                bound=1 if col==2 else limit
                im=axes[row,col].pcolormesh(x,y,field,cmap=cmap,vmin=-bound,vmax=bound,shading='nearest')
                axes[row,col].invert_yaxis()
                axes[row,col].set(title=f'{axis.capitalize()}: {title}',xlabel='Range sample',ylabel='Azimuth line')
                fig.colorbar(im,ax=axes[row,col],label='SLC pixels',extend='both',shrink=.8)
    fig.suptitle(f'{source.name}: full radar AOI, configured zero quality cutoffs\n'
                 'Grey: invalid. Color scales clip extreme offsets. No smoothing or gap filling.')
    fig.savefig(output,dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stack',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    source,out=args.stack.resolve(),args.output.resolve()
    if out.exists():
        raise ValueError('Validation output must be a new directory')
    if out == source or source in out.parents:
        raise ValueError('Validation output must be outside the original stack')
    gdal.UseExceptions()
    old=json.loads((source/'geometry/manifest.json').read_text())
    old_pairs=[(p,json.loads((p/'manifest.json').read_text())) for p in sorted((source/'pairs').iterdir())
               if (p/'radar/offsets.h5').is_file()]
    if not old_pairs:
        raise ValueError('No completed source pairs')
    originals=[p for p in source.rglob('*') if p.is_file()]
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in originals}
    layout=write_reference_layout(out/'geometry',old['sources'])
    specification=old['specification'] | {'method':METHOD,'ownership_policy':POLICY}
    frame=dict(layout,specification=specification,geometry_identity=identity(specification),
               software=old['software'],status='complete',
               validation_source_geometry=old['geometry_identity'])
    frame['products']=product_hashes([frame['coreg']['source_index'],*frame['coreg']['geometry'].values()])
    write_json(out/'geometry/manifest.json',frame)
    cfg=load_defaults(); cfg['output']={'directory':str(out)}
    cfg['inputs'].update(burst_id='stack',polarization=old_pairs[0][1]['coreg_specification']['polarization'])
    dates={}; acquisitions={}
    for path in sorted((source/'acquisitions').glob('*/manifest.json')):
        aligned=json.loads(path.read_text()); day=aligned['date']
        acquisition=SimpleNamespace(date=day,scene=SimpleNamespace(start=datetime.fromisoformat(day)),
            bursts={key:SimpleNamespace(start=datetime.fromisoformat(t)) for key,t in aligned['burst_times'].items()})
        acquisitions[day]=acquisition
        aligned=deepcopy(aligned)
        aligned['specification'].update(method=METHOD,geometry=frame['geometry_identity'])
        aligned['alignment_identity']=identity(aligned['specification'])
        aligned['validation_reused_alignment']=str(path)
        aligned.pop('assembly_identity',None)
        print(f'Assembling {day}',flush=True)
        dates[day]=assemble_date(cfg,acquisition,frame,aligned)
    plan=SimpleNamespace(reference=SimpleNamespace(date=old_pairs[0][1]['stack_pair']['geometry_reference_date']))
    report={'status':'running','source':str(source),'output':str(out),'method':METHOD,
            'ownership':frame['ownership'],'geometry_identity':frame['geometry_identity'],
            'scope':'Full AOI, existing aligned bursts; no fresh SAFE/COMPASS validation or displacement truth.',
            'pairs':[]}
    for path,previous in old_pairs:
        cfg['matching']=deepcopy(previous['matching']['matching'])
        cfg['matching']['backend']='cpu'
        cfg['gross_offsets']=deepcopy(previous['matching']['gross_offsets'])
        if cfg['gross_offsets']['mode'] != 'zero':
            raise ValueError('This validation currently requires zero gross offsets')
        geo=previous['geocoding']
        cfg['geocoding']={'epsg':geo['epsg'],'spacing_m':geo['spacing_m'],
                          'resampling':geo['resampling'],'mask':geo['source_mask']}
        target=common_map_grid(cfg,frame)
        a,b=(acquisitions[d] for d in previous['stack_pair']['acquisition_dates'])
        a.scene.start=datetime.fromisoformat(previous['acquisition_times'][0])
        b.scene.start=datetime.fromisoformat(previous['acquisition_times'][1])
        directory,m=prepared_pair(cfg,plan,frame,a,b,dates,target)
        print(f'Matching {path.name}',flush=True)
        match(cfg,directory,m)
        print(f'Geocoding {path.name}',flush=True)
        geocode(cfg,directory,m)
        entry={'pair':path.name,'old_match_identity':previous['match_identity'],
               'match_identity':m['match_identity'],'matching_summary':m['matching_summary'],
               'old_matching_summary':previous['matching_summary'],'geocoding':m['geocoding'],
               'map_grid_unchanged':target==previous['map_grid'],'joins':[]}
        with h5py.File(path/'radar/offsets.h5') as orig,h5py.File(directory/'radar/offsets.h5') as new:
            unchanged=0; max_difference=0.
            y=new['center_line'][:]
            old_seam=orig['seam_crossing'][:]; new_seam=new['seam_crossing'][:]
            same=~old_seam & ~new_seam & (orig['source_burst_index'][:]==new['source_burst_index'][:])
            same &= orig['valid'][:] & new['valid'][:]
            for axis in ('azimuth','range'):
                delta=new[f'total/{axis}_px'][:]-orig[f'total/{axis}_px'][:]
                max_difference=max(max_difference,float(np.max(np.abs(delta[same]))) if same.any() else 0.)
            unchanged=int(same.sum())
            entry['unchanged_source_control_count']=unchanged
            entry['unchanged_source_max_difference_px']=max_difference
            if not unchanged or max_difference>1/32:
                raise AssertionError('Full-AOI unchanged-source controls failed')
            for cut in frame['ownership']['cuts']:
                lower=next(s for s in old['sources'] if s['burst_id']==cut['lower_burst_id'])
                old_cut=lower['offset'][0]+lower['valid_window'][0]
                # Read-only descriptive count: same-column neighbors agree on
                # nonzero motion, but an intervening estimate collapses to zero.
                summary={'old_cut':old_cut,'new_cut':cut['cut_global_line'],'fields':{}}
                for axis in ('azimuth','range'):
                    per_version={}
                    for label,h in [('old',orig),('new',new)]:
                        v=h[f'total/{axis}_px'][:]; valid=h['valid'][:]
                        counts={}
                        for location in [old_cut,cut['cut_global_line']]:
                            j=int(np.argmin(abs(y[:,0]-location)))
                            lo,hi=max(0,j-3),min(len(y)-1,j+3)
                            baseline=(v[lo]+v[hi])/2
                            supported=valid[lo]&valid[hi]&(abs(v[lo]-v[hi])<=.5)&(abs(baseline)>1)
                            affected=(abs(y-location)<40)&valid&supported[None,:]
                            collapse=affected&(abs(v)<1)&(abs(v-baseline)>1)
                            counts[str(location)]={'neighbor_supported_centers':int(affected.sum()),
                                                   'near_zero_outliers':int(collapse.sum())}
                        per_version[label]=counts
                    summary['fields'][axis]=per_version
                entry['joins'].append(summary)
        if not entry['map_grid_unchanged']:
            raise AssertionError('Common map grid changed')
        report['pairs'].append(entry)
        write_json(args.report,report)
        plot_comparison(path,directory,args.report.parent/f'interior-{path.name}.png')
    report['source_sizes_mtimes_unchanged']=all((p.stat().st_size,p.stat().st_mtime_ns)==before[p] for p in originals)
    if not report['source_sizes_mtimes_unchanged']:
        raise AssertionError('Original products changed')
    report['status']='completed'
    write_json(args.report,report)


if __name__=='__main__':
    main()
