"""Independent overlap validation of shared versus year-specific spacecraft bias.

Consumes already measured same-ground burst-pair contrasts. It never changes a
calibration's scope, fits offsets, or applies a correction to production data.
"""
import numpy as np
from .swath_models import (train, predict, validation_folds, multiplier_draws,
                           bootstrap_parameters)


def _link_errors(rows,prediction):
    result={}
    for pair in sorted({r['pair'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['pair']==pair]
        if np.isfinite(prediction[indices]).all():
            result[pair]=abs(np.mean([rows[i]['mean']-prediction[i] for i in indices]))
    return result


def compare_calibrations(rows,dates,missions,draws=1000,seed=12345):
    """Validate using acquisition and shared-burst-purged held-out groups.

    A shared coefficient is supported only when its split-year difference is
    consistent with zero and held-out prediction is within one SE of the
    year-specific alternative. Row counts are burst-pair means, not windows.
    """
    if draws<1000:
        raise ValueError('Calibration comparison requires at least 1000 shared dependence draws')
    if not rows or any(not np.isfinite(r['mean']) for r in rows):
        raise ValueError('Finite, independent same-ground contrast rows are required')
    years=sorted({d[:4] for d in dates})
    multipliers=multiplier_draws(rows,draws,seed)
    joint=train(rows,dates,missions,'spacecraft')
    joint_draws=bootstrap_parameters(joint,multipliers)[:,0]
    estimates={}
    year_draws={}
    for year in years:
        mask=np.array([r['earlier'][:4]==year and r['later'][:4]==year for r in rows])
        subset=[r for r,keep in zip(rows,mask) if keep]
        fitted=train(subset,dates,missions,'spacecraft')
        if fitted['rank']!=1:
            estimates[year]={'identifiable':False,'rows':len(subset)}
            continue
        samples=bootstrap_parameters(fitted,multipliers[:,mask])[:,0]
        year_draws[year]=samples
        estimates[year]={'identifiable':True,'beta_px':float(fitted['parameters'][0]),
                         'beta_std_px':float(np.std(samples,ddof=1)), 'interval_95_px':np.quantile(samples,[.025,.975]),
                         'rows':len(subset),'links':len({r['pair'] for r in subset})}
    ledgers=[]
    for kind,label,train_mask,test_mask in validation_folds(rows,dates):
        if kind not in ('leave_acquisition_out','leave_burst_group_out') or not test_mask.any():
            continue
        training=[r for r,take in zip(rows,train_mask) if take]
        testing=[r for r,take in zip(rows,test_mask) if take]
        common=train(training,dates,missions,'spacecraft')
        cp=predict(common,testing,dates,missions)
        yp=np.full(len(testing),np.nan)
        for year in years:
            subset=[r for r in training if r['earlier'][:4]==year and r['later'][:4]==year]
            fitted=train(subset,dates,missions,'spacecraft')
            indices=[i for i,r in enumerate(testing) if r['earlier'][:4]==year and r['later'][:4]==year]
            if fitted['rank']==1 and indices:
                yp[indices]=predict(fitted,[testing[i] for i in indices],dates,missions)
        common_errors,year_errors=_link_errors(testing,cp),_link_errors(testing,yp)
        baseline_errors=_link_errors(testing,np.zeros(len(testing)))
        links=sorted(common_errors.keys() & year_errors.keys())
        ledgers.append({'kind':kind,'label':label,'train_rows':np.where(train_mask)[0],
                        'test_rows':np.where(test_mask)[0],'links':links,
                        'common_error_px':[common_errors[p] for p in links],
                        'year_specific_error_px':[year_errors[p] for p in links],
                        'baseline_error_px':[baseline_errors[p] for p in links],
                        'unavailable_test_rows':int(np.sum(~np.isfinite(cp)|~np.isfinite(yp)))})
    protocols={}
    for kind in ('leave_acquisition_out','leave_burst_group_out'):
        folds=[f for f in ledgers if f['kind']==kind and f['links']]
        delta=np.array([np.mean(np.array(f['common_error_px'])-f['year_specific_error_px']) for f in folds])
        se=float(np.std(delta,ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else np.nan
        protocols[kind]={'folds':len(folds),'mean_common_minus_year_specific_mae_px':float(np.mean(delta)) if len(delta) else None,
                         'fold_standard_error_px':se,'common_within_one_se':bool(len(delta)>1 and np.mean(delta)<=se)}
    differences=[]
    for i,year in enumerate(years):
        for other in years[i+1:]:
            if year in year_draws and other in year_draws:
                interval=np.quantile(year_draws[other]-year_draws[year],[.025,.975])
                differences.append({'years':[year,other],'difference_95_px':interval,'consistent_with_zero':bool(interval[0]<=0<=interval[1])})
    for year,estimate in estimates.items():
        validation={}
        for kind in ('leave_acquisition_out','leave_burst_group_out'):
            changes=[]
            for fold in ledgers:
                if fold['kind']!=kind:
                    continue
                index=[i for i,pair in enumerate(fold['links']) if pair[:4]==year and pair[11:15]==year]
                if index:
                    changes.append(np.mean(np.array(fold['year_specific_error_px'])[index]-np.array(fold['baseline_error_px'])[index]))
            se=float(np.std(changes,ddof=1)/np.sqrt(len(changes))) if len(changes)>1 else np.nan
            validation[kind]={'folds':len(changes),'mae_change_vs_baseline_px':float(np.mean(changes)) if changes else None,
                              'standard_error_px':se,'supported':bool(len(changes)>1 and np.mean(changes)<=se)}
        estimate['heldout_validation']=validation
        estimate['independently_supported']=bool(estimate['identifiable'] and all(v['supported'] for v in validation.values()))
    supported=(len(year_draws)>=2 and len(year_draws)==len(years) and
               all(d['consistent_with_zero'] for d in differences) and
               all(p['common_within_one_se'] for p in protocols.values()))
    return {'schema':'glacier-offsets-overlap-calibration-comparison-v1','shared_calibration_supported':supported,
            'recommendation':'shared' if supported else 'independently validate year-specific corrections; exclude unvalidated cross-year pairs',
            'joint':{'beta_px':float(joint['parameters'][0]),'beta_std_px':float(np.std(joint_draws,ddof=1)),
                     'rank':joint['rank']}, 'split_year_estimates':estimates,'year_differences':differences,
            'held_out_acquisitions':protocols['leave_acquisition_out'],'held_out_burst_groups':protocols['leave_burst_group_out'],
            'folds':ledgers,'draws':draws,'seed':seed,
            'uncertainty':'Shared acquisition/source-burst multiplier sensitivity; no independent-window claim',
            'scope_change_authorized':False}
