"""Validate a fixed published spacecraft coefficient on unseen 2017 links.

The original calibration and netCDF stay unchanged. Published training dates are
purged from temporal validation, and source-burst groups are additionally purged
from training for the spatial generalization check. This validates only the
requested calendar year; it supplies no evidence for an extension into 2018.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from glacier_offsets.swath_models import design,link_weights,multiplier_draws,train,bootstrap_parameters
from glacier_offsets.velocity_calibration import compare_calibrations
from glacier_offsets.velocity_io import Collection,digest,read_json_variable,write_json


def link_mae_change(rows,predictions,multipliers=None):
    """Average burst-group contrasts within links, then balance complete links."""
    predictions=np.asarray(predictions)
    if predictions.ndim==1:
        predictions=predictions[None,:]
    multiplier=np.ones_like(predictions) if multipliers is None else np.broadcast_to(multipliers,predictions.shape)
    before=[];after=[];weight=[]
    for pair in sorted({r['pair'] for r in rows}):
        index=np.array([i for i,r in enumerate(rows) if r['pair']==pair])
        w=multiplier[:,index]
        observed=np.array([rows[i]['mean'] for i in index])
        before.append(abs(np.sum(w*observed,axis=1)/np.sum(w,axis=1)))
        after.append(abs(np.sum(w*(observed-predictions[:,index]),axis=1)/np.sum(w,axis=1)))
        weight.append(np.mean(w,axis=1))
    w=np.asarray(weight).T
    return np.sum((np.asarray(after)-before).T*w,axis=1)/np.sum(w,axis=1)


def validate_fixed(rows,dates,missions,calibration,year='2017',draws=1000):
    training_dates=set(calibration['provenance']['training_dates'])
    within=np.array([r['earlier'][:4]==year and r['later'][:4]==year for r in rows])
    training=within & np.array([r['earlier'] in training_dates and r['later'] in training_dates for r in rows])
    testing=within & np.array([r['earlier'] not in training_dates and r['later'] not in training_dates for r in rows])
    training_rows=[r for r,t in zip(rows,training) if t]
    testing_rows=[r for r,t in zip(rows,testing) if t]
    if len({r['pair'] for r in testing_rows})<10:
        raise ValueError('At least ten unseen links are required for fixed-coefficient validation')
    factors=multiplier_draws(rows,draws,12345)
    x=design(testing_rows,dates,missions,'spacecraft')[:,0]
    fixed=x*calibration['beta_px']
    fixed_change=float(link_mae_change(testing_rows,fixed)[0])
    fixed_interval=np.quantile(link_mae_change(testing_rows,np.broadcast_to(fixed,(draws,len(x))),factors[:,testing]),[.025,.975])
    h1=train(training_rows,dates,missions,'spacecraft')
    h2=train(testing_rows,dates,missions,'spacecraft')
    h1_draws=bootstrap_parameters(h1,factors[:,training])[:,0]
    h2_draws=bootstrap_parameters(h2,factors[:,testing])[:,0]
    h1_interval=np.quantile(h1_draws,[.025,.975])
    split_interval=np.quantile(h2_draws-h1_draws,[.025,.975])
    folds=[]
    prediction=np.full(len(testing_rows),np.nan)
    prediction_draws=np.full((draws,len(testing_rows)),np.nan)
    for group in sorted({(r['burst1'],r['burst2']) for r in testing_rows}):
        take=np.array([(r['burst1'],r['burst2'])==group for r in testing_rows])
        shared=set(group)
        purged=training & np.array([not shared.intersection((r['burst1'],r['burst2'])) for r in rows])
        fitting=[r for r,t in zip(rows,purged) if t]
        model=train(fitting,dates,missions,'spacecraft')
        if model['rank']!=1:
            raise ValueError('A source-burst-purged training fold is not identifiable')
        samples=bootstrap_parameters(model,factors[:,purged])[:,0]
        prediction[take]=x[take]*model['parameters'][0]
        prediction_draws[:,take]=samples[:,None]*x[take]
        folds.append({'test_group':group,'train_rows':np.where(purged)[0],'test_rows':np.where(testing)[0][take],
                      'training_bursts':sorted({r[k] for r in fitting for k in ('burst1','burst2')}),
                      'beta_px':float(model['parameters'][0])})
    group_change=float(link_mae_change(testing_rows,prediction)[0])
    group_interval=np.quantile(link_mae_change(testing_rows,prediction_draws,factors[:,testing]),[.025,.975])
    requirements={'fixed_temporal_improves':bool(fixed_change<0 and fixed_interval[1]<0),
                  'purged_groups_improve':bool(group_change<0 and group_interval[1]<0),
                  'published_beta_compatible_with_training':bool(h1_interval[0]<=calibration['beta_px']<=h1_interval[1]),
                  'split_period_coefficients_consistent':bool(split_interval[0]<=0<=split_interval[1])}
    return {'schema':'glacier-offsets-fixed-calibration-validation-v1','passed':all(requirements.values()),
            'requirements':requirements,'year':year,'published_beta_px':calibration['beta_px'],
            'training_rows':np.where(training)[0],'temporal_test_rows':np.where(testing)[0],
            'excluded_shared_acquisition_rows':np.where(within & ~training & ~testing)[0],
            'held_out_acquisitions':{'training_dates':sorted(training_dates),'test_links':len(set(r['pair'] for r in testing_rows)),
                'mae_change_px':fixed_change,'mae_change_95_px':fixed_interval},
            'held_out_burst_groups':{'folds':folds,'mae_change_px':group_change,'mae_change_95_px':group_interval},
            'split_period_estimates':{'training_beta_px':float(h1['parameters'][0]),'training_beta_95_px':h1_interval,
                'withheld_beta_px':float(h2['parameters'][0]),'withheld_minus_training_95_px':split_interval},
            'draws':draws,'seed':12345,'uncertainty':'Shared acquisition/source-burst multiplier dependence sensitivity',
            'scope':'Empirical relative correction in this calendar year only; excludes absolute motion/model error'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--catalog',required=True,type=Path)
    parser.add_argument('--calibration',required=True,type=Path)
    parser.add_argument('--inputs',required=True,nargs='+')
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    root=args.experiment.resolve()
    fit=json.loads((root/'fit.json').read_text());plan=json.loads((root/'plan.json').read_text())
    analysis=root/'analysis/primary.json'
    if fit.get('status')!='complete' or fit['specification']['experiment']!=plan['identity'] or digest(analysis)!=fit['products'].get(str(analysis)):
        raise ValueError('Independent same-ground overlap evidence is incomplete or corrupted')
    for pair,sha in fit['specification']['measurements'].items():
        path=root/'measurements'/pair/'report.json'
        if digest(path)!=sha:raise ValueError('Independent measurement report checksum mismatch')
        for name,expected in json.loads(path.read_text())['products'].items():
            if digest(name)!=expected:raise ValueError('Independent measured observation checksum mismatch')
    catalog=json.loads(args.catalog.read_text());calibration=json.loads(args.calibration.read_text())
    missions={r['date']:r['mission'] for r in catalog['inventory'] if r.get('eligible')}
    rows=json.loads(analysis.read_text())['rows'];dates=sorted({r[k] for r in rows for k in ('earlier','later')})
    result=validate_fixed(rows,dates,missions,calibration)
    comparison=compare_calibrations(rows,dates,missions)
    blockers=[key for key,value in result['requirements'].items() if not value]
    with Collection(args.inputs,'2017-01-01','2018-01-01') as collection:
        for h in collection.handles:
            if read_json_variable(h,'calibration_json')!=calibration:blockers.append('Applied calibration does not match the fixed candidate')
            if any(h.attrs[k]!=plan['specification'][k] for k in ('geometry_identity','layout_identity')):blockers.append('Reference geometry/layout differs from overlap evidence')
            for key,value in calibration['scope']['matching'].items():
                if plan['specification']['config']['matching'].get(key)!=value:blockers.append('Matching treatment differs: '+key)
            e,l=(h[k][:] for k in ('earlier_burst_time','later_burst_time'));good=np.isfinite(e+l)
            start=np.datetime64('2017-01-01').astype('datetime64[s]').astype('int64');end=np.datetime64('2018-01-01').astype('datetime64[s]').astype('int64')
            if (e[good]<start).any() or (l[good]>=end).any():blockers.append('This evidence covers 2017 only')
        result.update(schema='glacier-offsets-calibration-validation-v1',passed=not blockers,blockers=blockers,
                      input_sha256=[s['sha256'] for s in collection.sources],cross_boundary_policy='excluded',
                      split_year_estimates=comparison['split_year_estimates'])
    result['evidence']=[{'path':str(p.resolve()),'sha256':digest(p)} for p in (analysis,root/'fit.json',root/'plan.json',args.catalog,args.calibration,Path(__file__))]
    write_json(args.output,result)
    print(json.dumps({'passed':result['passed'],'blockers':blockers,'report':str(args.output)},indent=2))


if __name__=='__main__':main()
