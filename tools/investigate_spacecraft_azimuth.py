"""Spacecraft-order controls on independent, same-ground IW1/IW2 observations.

No offset-agreement selection or empirical correction is applied. New pair
experiments are run separately; this analysis reads their retained observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from glacier_offsets.products import write_json
from glacier_offsets.geocode import sample_geometry
from tools.summarize_range2x_pairs import grouped_mean


def load_observations(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def eligible(observations,threshold=.3):
    return observations['dual_valid']&observations['burst_interior']&(
        observations['iw1_peak']>=threshold)&(observations['iw2_peak']>=threshold)


def counterbalanced_effect(ab,ba):
    """Same-point odd/even components; neither selects for offset agreement."""
    return (ab-ba)/2,(ab+ba)/2


def temporal_closure(ab,bc,ac):
    """Same-ground additive temporal closure, retaining each offset component."""
    return ab+bc-ac


def summarize(values,groups,rng):
    if not values.size:return {'n':0,'group_n':0,'mean_px':None,'median_px':None,'ci95_px':None,
        'group_ids':[],'group_counts':[],'group_means_px':[],'group_medians_px':[],
        'p10_px':None,'p90_px':None,'max_absolute_px':None,'absolute_gt_0_25_n':0,'absolute_gt_1_n':0}
    result=grouped_mean(values,groups,rng)
    keys=np.unique(groups)
    result.update(group_n=int(len(keys)),group_ids=keys.tolist(),
                  group_counts=[int((groups==key).sum()) for key in keys],
                  group_means_px=[float(values[groups==key].mean()) for key in keys],
                  p10_px=float(np.percentile(values,10)),p90_px=float(np.percentile(values,90)),
                  max_absolute_px=float(abs(values).max()),
                  absolute_gt_0_25_n=int((abs(values)>.25).sum()),
                  absolute_gt_1_n=int((abs(values)>1).sum()))
    return result


def differential_bistatic_seconds(meta1,meta2,rho1,rho2):
    """IW2-IW1 of installed s1-reader's additive tagged-time correction."""
    light_speed=299792458.
    def correction(meta,rho):
        return (meta['iw2_mid_range_m']+rho)/light_speed-meta['rank']/meta['prf_raw_Hz']
    return correction(meta2,rho2)-correction(meta1,rho1)


def sampled_bistatic_audit(spec,metadata,data):
    """Evaluate the analytic timing term at the same ground patches via saved offsets."""
    stack=Path(spec['source_stack']);frame=json.loads((stack/'geometry/manifest.json').read_text())
    sources={e['burst_id']:e for e in frame['sources']}
    selected=spec['timing_selection_pair'];selection=data[selected][0]
    numbers=json.loads((Path(spec['pairs'][selected]['overlap'])/'report.json').read_text())['source_burst_ids']
    mask=selection['dual_support']&selection['burst_interior']
    deltas={};ranges={};counts={}
    for day,meta in metadata.items():
        acquisition=json.loads((stack/'acquisitions'/day/'manifest.json').read_text())
        corrected={};rho={}
        for swath in ('iw1','iw2'):
            corrected[swath]=np.full(mask.shape,np.nan);rho[swath]=np.full(mask.shape,np.nan)
            for number,key in numbers.items():
                if not key.endswith(swath):continue
                take=mask&(selection[f'{swath}_owner']==int(number))
                if not take.any():continue
                dy,dx=sources[key]['offset'];line=selection['center_line'][take]-dy;sample=selection['center_sample'][take]-dx
                validity=acquisition['valid_windows'].get(key,{})
                offsets=validity.get('offset_paths')
                if not offsets and day!=frame['specification']['anchor']['reference_date']:
                    raise ValueError('Missing registered range offsets on a non-reference acquisition')
                offset=sample_geometry(offsets[1],line,sample) if offsets else np.zeros(line.shape)
                if np.any(~np.isfinite(offset)) or np.any(abs(offset)>100000):
                    raise ValueError('Nonfinite or sentinel geometric range offsets on selected support')
                m=meta[key];slant=m['starting_range_m']+(sample+offset)*m['range_spacing_m']
                rho[swath][take]=slant
                corrected[swath][take]=(m['iw2_mid_range_m']+slant)/299792458.-m['rank']/m['prf_raw_Hz']
        deltas[day]=corrected['iw2']-corrected['iw1'];ranges[day]=rho['iw2']-rho['iw1']
        counts[day]={'n':int(np.isfinite(deltas[day][mask]).sum()),
            'max_same_ground_slant_range_discrepancy_m':float(np.nanmax(abs(ranges[day][mask]))),
            'mean_differential_seconds':float(np.nanmean(deltas[day][mask]))}
    prf=frame['coreg']['reference_grid']['prf'];predictions={}
    for name,entry in spec['pairs'].items():
        early,late=name.split('_')
        if early not in deltas or late not in deltas or entry['geometry_group']!='pilot':continue
        predicted=-(deltas[late]-deltas[early])*prf;good=mask&np.isfinite(predicted)
        predictions[name]={'n':int(good.sum()),'mean_px':float(predicted[good].mean()),
                            'max_absolute_px':float(abs(predicted[good]).max())}
    return {'model':'Installed s1-reader correction C=(IW2_mid_range+slant_range)/c-rank/raw_PRF; observed uncorrected pair effect modeled as -PRF_ref * delta_date(C_IW2-C_IW1)',
        'range_mapping':'Saved native range.off evaluated at selected reference chip centers, then original acquisition range grid; no imagery resampling',
        'selected_ground_centers_n':int(mask.sum()),'dates':counts,'predicted_pair_effects':predictions,
        'limitation':'Tests this explicit analytic bistatic timing term only; does not test FM-rate mismatch, instrument timing calibration, ionosphere or general coregistration errors.'}


def analyze(specification,output):
    rng=np.random.default_rng(0);spec=json.loads(Path(specification).read_text())
    detail_rng=np.random.default_rng(1)
    metadata=json.loads(Path(spec['timing_metadata']).read_text()) if spec.get('timing_metadata') else {}
    watched={Path(specification)}
    if spec.get('timing_metadata'):watched.add(Path(spec['timing_metadata']))
    for entry in spec['pairs'].values():
        root=Path(entry['overlap']);watched.add(root/'report.json')
        watched.update(root/f'deramp-{mode}/observations.npz' for mode in (0,1))
    if spec.get('source_stack'):
        stack=Path(spec['source_stack']);watched.add(stack/'geometry/manifest.json')
        for day in metadata:
            path=stack/'acquisitions'/day/'manifest.json';watched.add(path)
            acquisition=json.loads(path.read_text())
            for record in acquisition['valid_windows'].values():
                for key in ('offset_paths','offset_products'):
                    for path in record.get(key,[]):watched.add(Path(path))
    stamps={path:(path.stat().st_size,path.stat().st_mtime_ns) for path in watched}
    pairs={};data={};pair_report={}
    for name,entry in spec['pairs'].items():
        root=Path(entry['overlap']);views={mode:load_observations(root/f'deramp-{mode}/observations.npz') for mode in (0,1)}
        pairs[name]=entry;data[name]=views
        report={'missions':entry['missions'],'geometry_group':entry['geometry_group'],'modes':{}}
        saved=json.loads((root/'report.json').read_text())
        report['input_supported_n']=int(views[0]['dual_support'].sum())
        report['production_sizes_mtimes_unchanged']=saved['production_sizes_mtimes_unchanged']
        for mode,v in views.items():
            delta=(v['iw2_total']-v['iw1_total'])[...,0]
            report['modes'][str(mode)]={}
            for threshold in (0,.3,.5):
                mask=eligible(v,threshold);groups=(v['iw1_owner'][mask].astype(int)*1000+v['iw2_owner'][mask])
                report['modes'][str(mode)][f'both_peaks_ge_{threshold}']=summarize(delta[mask],groups,rng)
        pair_report[name]=report
    contrasts={}
    for label,first,second in spec['counterbalanced_contrasts']:
        if pairs[first]['missions']!=['S1A','S1B'] or pairs[second]['missions']!=['S1B','S1A']:
            raise ValueError('Contrast requires A->B followed by B->A labels')
        if pairs[first]['geometry_group']!=pairs[second]['geometry_group']:
            raise ValueError('Use the same preparation policy in counterbalanced contrasts')
        contrasts[label]={'pairs':[first,second],'modes':{}}
        for mode in (0,1):
            a,b=data[first][mode],data[second][mode]
            for key in ('center_line','center_sample'):
                np.testing.assert_array_equal(a[key],b[key])
            mask=eligible(a)&eligible(b)
            for key in ('iw1_owner','iw2_owner'):np.testing.assert_array_equal(a[key][mask],b[key][mask])
            groups=a['iw1_owner'][mask].astype(int)*1000+a['iw2_owner'][mask]
            da=(a['iw2_total']-a['iw1_total'])[...,0][mask]
            db=(b['iw2_total']-b['iw1_total'])[...,0][mask]
            odd,even=counterbalanced_effect(da,db)
            contrasts[label]['modes'][str(mode)]={
                'spacecraft_order_odd_component':summarize(odd,groups,rng),
                'order_even_component':summarize(even,groups,rng),
                'A_to_B_same_points':summarize(da,groups,rng),'B_to_A_same_points':summarize(db,groups,rng)}
    closures={}
    for label,a,b,direct in spec.get('closure_triangles',[]):
        closures[label]={'pairs':[a,b,direct],'modes':{}}
        for mode in (0,1):
            views=[data[name][mode] for name in (a,b,direct)]
            for v in views[1:]:
                for key in ('center_line','center_sample'):np.testing.assert_array_equal(views[0][key],v[key])
            mask=np.logical_and.reduce([eligible(v) for v in views])
            for v in views[1:]:
                for key in ('iw1_owner','iw2_owner'):np.testing.assert_array_equal(views[0][key][mask],v[key][mask])
            groups=views[0]['iw1_owner'][mask].astype(int)*1000+views[0]['iw2_owner'][mask]
            delta=[(v['iw2_total']-v['iw1_total'])[...,0][mask] for v in views]
            result=summarize(temporal_closure(*delta),groups,rng)
            result['per_swath_components']={swath:{axis:summarize(temporal_closure(
                *[v[f'{swath}_total'][...,k][mask] for v in views]),groups,detail_rng)
                for k,axis in enumerate(('azimuth','range'))} for swath in ('iw1','iw2')}
            closures[label]['modes'][str(mode)]=result
        # Mode sensitivity on the identical three-pair, two-mode quality intersection.
        all_views=[data[name][mode] for mode in (0,1) for name in (a,b,direct)]
        common=np.logical_and.reduce([eligible(v) for v in all_views])
        groups=all_views[0]['iw1_owner'][common].astype(int)*1000+all_views[0]['iw2_owner'][common]
        closures[label]['both_modes_common_quality']={}
        for mode in (0,1):
            delta=[(data[name][mode]['iw2_total']-data[name][mode]['iw1_total'])[...,0][common]
                   for name in (a,b,direct)]
            closures[label]['both_modes_common_quality'][str(mode)]=summarize(temporal_closure(*delta),groups,detail_rng)
        closures[label]['threshold_sensitivity']={}
        for threshold in (0,.5):
            by_mode={}
            for mode in (0,1):
                views=[data[name][mode] for name in (a,b,direct)]
                mask=np.logical_and.reduce([eligible(v,threshold) for v in views])
                same_owners=np.logical_and.reduce([views[0][key]==v[key]
                    for v in views[1:] for key in ('iw1_owner','iw2_owner')])
                excluded=int((mask&~same_owners).sum());mask&=same_owners
                groups=views[0]['iw1_owner'][mask].astype(int)*1000+views[0]['iw2_owner'][mask]
                delta=[(v['iw2_total']-v['iw1_total'])[...,0][mask] for v in views]
                result=summarize(temporal_closure(*delta),groups,detail_rng)
                result['excluded_different_source_owners_n']=excluded;by_mode[str(mode)]=result
            closures[label]['threshold_sensitivity'][f'both_peaks_ge_{threshold}']=by_mode
    timing={}
    for day,bursts in metadata.items():
        values=[]
        for key,m1 in bursts.items():
            if not key.endswith('iw1'):continue
            m2=bursts[key[:-3]+'iw2']
            # Equal slant range is the same-target zero-Doppler limit. A separate
            # sampled-offset audit can bound deviations from this approximation.
            values.append(differential_bistatic_seconds(m1,m2,850000.,850000.))
        timing[day]={'platform':next(iter(bursts.values()))['platform'],
                    'equal_range_IW2_minus_IW1_seconds':values,
                    'median_seconds':float(np.median(values))}
    report={'selection':'Input support, native validity, burst interiors, then each swath peak threshold; no cross-method or swath-agreement gate',
        'pairs':pair_report,'same_ground_counterbalanced_contrasts':contrasts,'closure_triangles':closures,
        'raw_timing_term_audit':timing,
        'uncertainty':'4000 burst-pair block bootstrap draws, RNG seed 0 (seed 1 for added per-swath/common-mode closure details); groups can share bursts and spatially correlated errors. CIs are conditional descriptive intervals, not independent acquisition replication.',
        'causal_limit':'Mission labels are tied to acquisition dates. Order reversal on identical patches supports a mission-associated component but does not identify sensor hardware versus acquisition-specific processing/geometry errors. No empirical correction is applied.'}
    if spec.get('source_stack') and metadata:report['sampled_bistatic_audit']=sampled_bistatic_audit(spec,metadata,data)
    if spec.get('near_zero_reference_band_px') is not None:
        report['near_zero_reference_band_px']=spec['near_zero_reference_band_px']
    unchanged=all((path.stat().st_size,path.stat().st_mtime_ns)==stamp for path,stamp in stamps.items())
    report.update(status='complete' if unchanged else 'incomplete',input_sizes_mtimes_unchanged=unchanged)
    if not unchanged:raise ValueError('An investigation input changed during analysis')
    write_json(output,report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained',sharex=True)
    names=list(pairs)
    for mode,ax in enumerate(axes):
        for i,name in enumerate(names):
            r=pair_report[name]['modes'][str(mode)]['both_peaks_ge_0.3']
            if r['n']:
                mean=r['mean_px'];lo,hi=r['ci95_px']
                ax.plot([i,i],[lo,hi],color='black');ax.scatter(i,mean,s=45)
                ax.text(i,hi+.006,f'n={r["n"]}, groups={r["group_n"]}',ha='center',fontsize=7)
        ax.axhline(0,color='black',lw=.8);ax.set(ylabel='IW2 minus IW1 azimuth [px]',title=f'Unspliced deramp={mode}; peaks >=0.3; no agreement trimming')
    axes[-1].set_xticks(range(len(names)),[name+'\n'+'→'.join(pairs[name]['missions']) for name in names],rotation=35,ha='right')
    fig.savefig(Path(output).with_suffix('.png'),dpi=160);plt.close(fig)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();analyze(args.spec,args.output)
