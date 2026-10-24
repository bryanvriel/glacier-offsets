"""Analyze completed full-pair preprocessing comparisons at two scalar zooms."""
import argparse
import json
from pathlib import Path

import numpy as np

from glacier_offsets.products import write_json
from glacier_offsets.stack_storage import read_manifest
from tools.compare_merged_preprocessing import read_offsets, refinement_settings
from tools.compare_swath_overlap import statistics


def change_statistics(a,b,mask):
    result={}
    for axis in ('azimuth','range'):
        delta=b[f'total/{axis}_px']-a[f'total/{axis}_px']
        result[axis]=statistics(delta[mask])
        result[axis]['abs_gt_1_px_n']=int((mask&(abs(delta)>1)).sum())
        result[axis]['changed_n']=int((mask&(delta!=0)).sum())
    return result


def analyze(previous,current):
    previous,current=Path(previous).resolve(),Path(current).resolve()
    summaries=[json.loads((p/'comparison.json').read_text()) for p in (previous,current)]
    if summaries[0]['source_pair']!=summaries[1]['source_pair']:
        raise ValueError('Zoom comparisons must reference the same production pair')
    data={};manifests={};names={}
    for epoch,root in [('previous',previous),('current',current)]:
        for treatment in ('deramp1','range2x'):
            manifest=read_manifest(root/treatment)
            zoom=manifest['matching']['matching']['zoom'];name=f'{treatment}_zoom{zoom}'
            if name in data:raise ValueError('Expected two distinct zoom values')
            arrays=read_offsets(root/treatment)
            segments=manifest['matching']['native_parameters']['segments']
            if any(s['corrSurfaceZoomInWindow']!=zoom for s in segments):
                raise ValueError('Native effective zoom differs from requested zoom')
            data[name]=arrays;manifests[name]=manifest;names[(epoch,treatment)]=name
    first=next(iter(data.values()));common=np.array(first['valid'],copy=True)
    for name,arrays in data.items():
        for key in ('center_line','center_sample','source_burst_index','input_valid','seam_crossing'):
            np.testing.assert_array_equal(first[key],arrays[key],err_msg=f'{name}: changed {key}')
        common&=arrays['valid']
    strong=common.copy()
    for arrays in data.values():strong&=arrays['correlation_peak']>=.3
    previous_quality=common.copy()
    for treatment in ('deramp1','range2x'):
        previous_quality&=data[names[('previous',treatment)]]['correlation_peak']>=.3
    cohorts={'common_all_four':common,'common_seam_crossing':common&first['seam_crossing'],
             'common_interior':common&~first['seam_crossing'],'all_four_peaks_ge_0.3':strong,
             'previous_both_peaks_ge_0.3':previous_quality}
    report={'source_pair':summaries[0]['source_pair'],'previous':str(previous),'current':str(current),
        'same_centers_ownership_input_support_and_seam_flags':True,
        'windows':int(common.size),'input_supported_n':int(first['input_valid'].sum()),
        'common_valid_all_four_n':int(common.sum()),'treatments':{},'changes':{},
        'limitations':'One pair; disagreement is not accuracy. Scalar zoom changes both axes. Range2x zoom32 matches the range refinement extent of deramp1 zoom16 but doubles its azimuth extent.'}
    for name,arrays in data.items():
        manifest=manifests[name];matching=manifest['matching'];native=matching['native_parameters']
        _,extents=refinement_settings(matching['matching'])
        report['treatments'][name]={'valid_n':int(arrays['valid'].sum()),
            'refinement_half_width_original_px':extents['range2x_half_width_original_px' if name.startswith('range2x') else 'deramp1_half_width_original_px'],
            'native_pool_wall_seconds':native['wall_seconds'],'summed_native_seconds':matching['elapsed_seconds'],
            'max_recorded_worker_rss_GiB':max(s['worker_peak_rss_bytes'] for s in native['segments'])/1024**3,
            'workers':native['workers'],'memory_note':'Maximum per-worker high-water mark, excluding parent; not concurrent process-tree RSS.'}
    comparisons={}
    for treatment in ('deramp1','range2x'):
        comparisons[f'{treatment}_zoom_change']=(names[('previous',treatment)],names[('current',treatment)])
    for epoch in ('previous','current'):
        comparisons[f'range2x_minus_deramp1_{epoch}']=(names[(epoch,'deramp1')],names[(epoch,'range2x')])
    comparisons['restored_range_extent_comparison']=(names[('previous','deramp1')],names[('current','range2x')])
    for label,(a,b) in comparisons.items():
        report['changes'][label]={'difference_sign':f'{b} minus {a}',
            'groups':{group:change_statistics(data[a],data[b],mask) for group,mask in cohorts.items()},
            'validity_transitions':{'valid_both':int((data[a]['valid']&data[b]['valid']).sum()),
                'only_first':int((data[a]['valid']&~data[b]['valid']).sum()),
                'only_second':int((~data[a]['valid']&data[b]['valid']).sum())}}
    np.savez_compressed(current/'zoom_sensitivity_masks.npz',**cohorts)
    write_json(current/'zoom_sensitivity.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,treatment in enumerate(('deramp1','range2x')):
        a=data[names[('previous',treatment)]];b=data[names[('current',treatment)]]
        for row,axis in enumerate(('range','azimuth')):
            delta=b[f'total/{axis}_px']-a[f'total/{axis}_px']
            finite=abs(delta[common]);limit=max(float(np.quantile(finite,.99)),1/64) if finite.size else 1/64
            artist=axes[row,col].imshow(np.where(common,delta,np.nan),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row,col].set(title=f'{treatment}: zoom change in {axis} [original px]',
                              xlabel='Matching range column',ylabel='Matching azimuth row')
            fig.colorbar(artist,ax=axes[row,col],shrink=.7)
    fig.suptitle('Scalar zoom sensitivity on the same four-treatment accepted observations')
    fig.savefig(current/'zoom_changes.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(19,9),layout='constrained')
    order=[names[(epoch,t)] for t in ('deramp1','range2x') for epoch in ('previous','current')]
    for row,axis in enumerate(('range','azimuth')):
        finite=np.concatenate([abs(data[name][f'total/{axis}_px'][common]) for name in order])
        limit=max(float(np.quantile(finite,.95)),.1) if finite.size else .1
        for col,name in enumerate(order):
            artist=axes[row,col].imshow(np.where(common,data[name][f'total/{axis}_px'],np.nan),aspect='auto',
                                        cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row,col].set_title(f'{name}: {axis} [original px]')
            fig.colorbar(artist,ax=axes[row,col],shrink=.65)
    fig.suptitle('Complete merged-pair fields: same observations and component display scales')
    fig.savefig(current/'zoom_field_comparison.png',dpi=160);plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous',type=Path);parser.add_argument('current',type=Path)
    args=parser.parse_args();analyze(args.previous,args.current)
