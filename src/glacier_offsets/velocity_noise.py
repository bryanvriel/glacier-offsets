"""Cross-fitted control fields and permutation-based spatial noise transfer.

These generators are experimental observation models, not uncertainty estimators.
The original with-replacement generator remains unchanged in velocity_quality.
"""
import numpy as np
from .velocity_basis import harmonic, utc_days
from .velocity_quality import quiet_median
from .velocity_systematics import (OrderRegionalDesign, order_screen,
                                  robust_background, calendar_times)


def acquisition_days(early, late, mask):
    values=np.r_[early[mask],late[mask]]
    return np.unique(np.floor(values[np.isfinite(values)]))


def crossfit_control_bank(field, covariance, split=None):
    """Predict each calendar half from the opposite acquisition-purged half.

    Remove secular/order predictions, retain control harmonics and failed-match
    residuals, and center each source within its held-out half. Source support
    requires a defined source secular prediction (>=1 clean training epoch). The
    shared order/harmonic training fit still requires >=8 epochs per source.
    Source centering removes the held-out constant prediction offset. The cross-boundary pair is excluded from
    the donor bank. Validity, peak and SNR remain paired with residuals.
    """
    split=float(utc_days('2017-07-01')) if split is None else float(split)
    early,late=calendar_times(field)
    halves=[late<split,early>=split]
    residual=np.full_like(field['data'],np.nan)
    valid=np.zeros_like(field['valid']);records=[];design=OrderRegionalDesign(field,covariance)
    for half,test in enumerate(halves):
        test_days=acquisition_days(early,late,test)
        train=halves[1-half].copy()
        train &= ~np.isin(np.floor(early),test_days)&~np.isin(np.floor(late),test_days)
        good=order_screen(field,allowed=train)&train[None]
        training_counts=good.sum(1);supported=training_counts>=1
        fit=design.fit(good,details=True)
        prediction,_=robust_background(field,train)
        prediction[...,1]=fit['prediction'][...,0]-np.einsum(
            'stj,j->st',np.nan_to_num(harmonic(field['early'],field['late'])),fit['cs'])
        r=field['data']-prediction
        usable=field['valid']&test[None]&supported[:,None]&np.isfinite(r).all(-1)
        center=quiet_median(np.where(usable[...,None],r,np.nan),axis=1)
        residual[:,test]=r[:,test]-center[:,None]
        valid[:,test]=usable[:,test]&np.isfinite(residual[:,test]).all(-1)
        records.append({'heldout_half':half+1,'train_indices':np.flatnonzero(train),
            'test_indices':np.flatnonzero(test),'train_acquisition_days':acquisition_days(early,late,train),
            'test_acquisition_days':test_days,'supported_training_sources':int(supported.sum()),
            'active_science_training_sources':int(np.sum(training_counts>=8)),
            'source_train_epoch_counts':training_counts,
            'bank_valid_pairs':int(valid[:,test].sum()),'order_m':fit['order_m'],
            'centering_ra_m':center})
    return {**field,'residual':residual,'valid':valid},records


def permutation_cycles(rng, candidates, count):
    """Exhaust each random permutation before reusing a finite donor pool."""
    candidates=np.asarray(candidates)
    if not len(candidates):raise ValueError('Empty donor pool')
    return np.concatenate([rng.permutation(candidates) for _ in range(int(np.ceil(count/len(candidates))))])[:count]


def spatial_mapping(rng, donor_groups, target_groups):
    """Persistent source mapping; block/source recycling only if unavoidable."""
    dg=np.unique(donor_groups);tg=np.unique(target_groups)
    chosen=permutation_cycles(rng,dg,len(tg));mapping=np.empty(len(target_groups),int)
    for group,donor_group in zip(tg,chosen):
        members=np.flatnonzero(target_groups==group)
        candidates=np.flatnonzero(donor_groups==donor_group)
        mapping[members]=permutation_cycles(rng,candidates,len(members))
    return mapping,chosen


def seasonal_time_indices(rng, early, late, block_length=5, split=None):
    """Shared moving blocks within matching calendar halves.

    The destination pair crossing July 1 receives a held-out H2 donor. Circular
    wrapping is confined to a half; no block mixes H1 and H2. Pair order need not
    be preserved after the deterministic order component has been removed.
    """
    split=float(utc_days('2017-07-01')) if split is None else float(split)
    early,late=np.asarray(early),np.asarray(late);mid=(early+late)/2
    result=np.full(len(early),-1,int)
    for destination,donor in [(mid<split,np.flatnonzero(late<split)),
                              (mid>=split,np.flatnonzero(early>=split))]:
        where=np.flatnonzero(destination&np.isfinite(mid))
        if not len(donor):raise ValueError('A calendar half has no donor pairs')
        starts=rng.integers(len(donor),size=int(np.ceil(len(where)/block_length)))
        local=((starts[:,None]+np.arange(block_length))%len(donor)).ravel()[:len(where)]
        result[where]=donor[local]
    if np.any(result<0):raise ValueError('Complete calendar labels required')
    return result


def transfer_control_field(rng, donor, target, time_index, replacement=False):
    """Transfer a whole field with residual/validity/quality paired exactly."""
    if replacement:
        mapping=np.empty(len(target['group']),int);dg=np.unique(donor['group']);chosen=[]
        for group in np.unique(target['group']):
            members=np.flatnonzero(target['group']==group);g=rng.choice(dg);chosen.append(g)
            candidates=np.flatnonzero(donor['group']==g)
            mapping[members]=rng.choice(candidates,len(members),replace=True)
        chosen=np.asarray(chosen)
    else:mapping,chosen=spatial_mapping(rng,donor['group'],target['group'])
    ix=np.ix_(mapping,np.asarray(time_index,int));residual=donor['residual'][ix].copy()
    valid=target['valid']&donor['valid'][ix]&np.isfinite(residual).all(-1)
    result={**target,'data':residual,'valid':valid,'snr':donor['snr'][ix].copy(),'peak':donor['peak'][ix].copy()}
    diagnostic={'donor_source_indices':mapping,'donor_blocks':chosen,
        'unique_donor_sources':len(np.unique(mapping)),'source_assignments':len(mapping),
        'unique_donor_blocks':len(np.unique(chosen)),'block_assignments':len(chosen)}
    return result,diagnostic
