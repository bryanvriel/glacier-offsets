"""Blocked predictive selection, paired resampling and injection validation."""
import logging
import numpy as np
from scipy.stats import chi2
from .velocity_basis import definition, design, harmonic
from .velocity_fit import (WEIGHT_MODELS, calibrate_noise, covariance_ra, fit, diagnostics,
                           flow_direction, blocked_folds, quality_rule)
from .velocity_inspect import M2_PERIOD, O1_PERIOD, phase_coverage


def point_arrays(point, noise, weighting, take=None):
    good = point['valid'].copy()
    if take is not None:
        good &= take
    if weighting=='quality_rule':
        good &= quality_rule(point['snr'], point['peak'])
    ra = covariance_ra(noise, point['snr'][good], point['peak'][good], weighting=='quality_robust')
    covariance = point['inverse'][None] @ ra @ point['inverse'].T[None]
    return point['early'][good], point['late'][good], point['data'][good], covariance, good


def fit_point(point, noise, candidate, parameterization, take=None, direction=None, data=None):
    early, late, values, covariance, good = point_arrays(point, noise, candidate['weighting'], take)
    if data is not None:
        values = data[good]
    if point.get('role')=='control' and direction is None:
        # On bedrock there is no physical flow axis. EN is authoritative; the
        # east axis is an explicit projection convention for control summaries.
        direction = np.array([1., 0.])
    if direction is None:
        # Both parameterizations use the same secular axis, independent of their
        # selected temporal or quality model. Reassess it on every resample.
        se,sl,sd,sc,sg = point_arrays(point,noise,'constant',take)
        direction = flow_direction(se,sl,sd if data is None else data[sg],sc)
    result = fit(early, late, values, covariance, candidate['basis'], parameterization,
                 direction=direction, penalty=candidate['penalty'], robust='robust' in candidate['weighting'])
    result['phase_coverage'] = phase_coverage(early, late)
    return result


def control_residuals(points, *, excluded_region=None, train=None):
    """Cross-fit robust secular motion on controls; retain pair and spatial groups."""
    residuals, snrs, peaks, groups, records = [], [], [], [], []
    for point in points:
        if point['role']!='control' or point['region']==excluded_region:
            continue
        valid = point['valid'].copy()
        if train is not None:
            valid &= train
        early, late, data = point['early'], point['late'], point['data']
        ra = np.linalg.inv(point['inverse'])
        predicted = np.full_like(data, np.nan)
        for training, test in blocked_folds(early[valid], late[valid]):
            index = np.where(valid)[0]
            if training.sum() < 5:
                continue
            subset = index[training]
            dt = late[subset]-early[subset]
            median = np.median(data[subset]/dt[:, None], axis=0)
            for _ in range(10):
                r = data[subset]-dt[:, None]*median
                scale = np.maximum(1.4826*np.median(abs(r), axis=0), 1e-3)
                w = np.minimum(1., 2.45/np.maximum(np.linalg.norm(r/scale, axis=1), 1e-12))
                median = np.sum(data[subset]*(w*dt)[:, None], axis=0)/np.sum(w*dt*dt)
            index_test = index[test]
            predicted[index_test] = (late[index_test]-early[index_test])[:, None]*median
        residual = (data-predicted) @ ra.T
        take = valid & np.isfinite(residual).all(-1)
        if take.sum():
            residuals.extend(residual[take])
            snrs.extend(point['snr'][take])
            peaks.extend(point['peak'][take])
            groups.extend([point['region']+':'+point['spatial_block']]*take.sum())
            records.append({'point': point, 'residual_ra': residual, 'valid': take})
    if not residuals:
        raise ValueError('No cross-fitted residuals from reviewed controls')
    return np.array(residuals), np.array(snrs), np.array(peaks), np.array(groups), records


def learn_noise(points, excluded_region=None, train=None):
    arrays = control_residuals(points, excluded_region=excluded_region, train=train)
    return calibrate_noise(*arrays[:4]), arrays[4]


def candidate_bank(early, late, backgrounds=None, weights=None, penalties=None):
    backgrounds = backgrounds or ['constant', 'linear', 'spline_l2', 'spline_l1']
    weights = weights or list(WEIGHT_MODELS)
    penalties = np.logspace(-4, 4, 9) if penalties is None else penalties
    result = []
    for background in backgrounds:
        basis = definition(early, late, background)
        for weighting in weights:
            for penalty in penalties if background.startswith('spline') else [0.]:
                result.append({'background': background, 'basis': basis, 'weighting': weighting, 'penalty': float(penalty)})
    return result


def select_models(points, candidates):
    """Paired 60-day CV, acquisition purging and leave-control-region-out noise.

    Every candidate is scored on the same held-out observations, including those
    the quality-rule candidate rejects during training. No quality-based score
    attrition can make a model appear better.
    """
    reference = points[0]
    # Scene acquisitions have equal dates but slightly different burst seconds.
    # Use UTC dates to purge all source bursts from the same acquisition.
    early, late = np.floor(reference['early']), np.floor(reference['late'])
    # Missing endpoints in one burst do not delete other bursts from CV.
    for point in points[1:]:
        early = np.where(np.isfinite(early), early, np.floor(point['early']))
        late = np.where(np.isfinite(late), late, np.floor(point['late']))
    finite = np.isfinite(early+late)
    if finite.sum()<10:
        raise ValueError('Insufficient observed burst endpoints for predictive folds')
    folds = []
    for train_subset,test_subset in blocked_folds(early[finite],late[finite]):
        train,test = np.zeros(len(early),bool),np.zeros(len(early),bool)
        train[finite],test[finite] = train_subset,test_subset
        folds.append((train,test))
    scores = {model: [[] for _ in candidates] for model in ('free_en', 'along_flow')}
    failures={model:[] for model in scores}
    ledger = []
    for fold, (train, test) in enumerate(folds):
        logging.info('Pilot predictive fold %s/%s (%s candidates)', fold+1, len(folds), len(candidates))
        if train.sum() < 10:
            continue
        noises = {}
        for region in sorted({p['region'] if p['role']=='control' else None for p in points},key=lambda r:r or ''):
            noises[region] = learn_noise(points, excluded_region=region, train=train)[0]
        fold_scores = {model: [[] for _ in candidates] for model in scores}
        for point in points:
            noise = noises[point['region'] if point['role']=='control' else None]
            test_good = test & point['valid']
            if not test_good.any():
                continue
            # Fixed constant calibration for scoring; only training weights vary.
            _, _, heldout, cov, _ = point_arrays(point, noise, 'constant', test_good)
            factor = np.linalg.cholesky(cov)
            for ci, candidate in enumerate(candidates):
                for model in scores:
                    try:
                        fitted = fit_point(point, noise, candidate, model, train)
                        matrix = design(point['early'][test_good], point['late'][test_good], candidate['basis'], model, fitted['direction'])[0]
                        residual = heldout-np.einsum('ijk,k->ij', matrix, fitted['coefficients'])
                        loss = np.mean(np.sum(np.linalg.solve(factor, residual[..., None])[..., 0]**2, axis=1))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        loss = np.inf
                        failures[model].append({'fold':fold,'candidate_index':ci,'region':point['region'],
                                                'source_id':point['source_id'],'reason':str(exc)})
                    fold_scores[model][ci].append((point['region'], point['spatial_block'], float(loss)))
        for model in scores:
            for ci, values in enumerate(fold_scores[model]):
                # Equal control/ice ROI contribution, with balanced spatial blocks.
                region_losses = [np.mean([v[2] for v in values if v[0]==region]) for region in sorted({v[0] for v in values})]
                scores[model][ci].append(float(np.mean(region_losses)) if region_losses else np.inf)
        ledger.append({'fold': fold, 'train_pair_indices': np.where(train)[0], 'test_pair_indices': np.where(test)[0],
                       'purged_pair_indices': np.where(~train & ~test)[0], 'excluded_control_regions': [k for k in noises if k]})
    reports = {}
    for model, losses in scores.items():
        mean = np.array([np.mean(v) if len(v) >= 3 else np.inf for v in losses])
        if not np.isfinite(mean).any():
            raise ValueError('Fewer than three usable predictive folds or all candidate fits failed')
        best = int(np.argmin(mean))
        se = float(np.std(losses[best], ddof=1)/np.sqrt(len(losses[best])))
        eligible = np.where(np.isfinite(mean) & (mean <= mean[best]+se))[0]
        def complexity(i):
            candidate = candidates[i]
            return (WEIGHT_MODELS.index(candidate['weighting']),
                    ['constant','linear','spline_l2','spline_l1'].index(candidate['background']), -candidate['penalty'])
        selected = int(min(eligible, key=complexity))
        # Retain all candidates within one SE for recovery-constrained selection.
        reports[model] = {'selected_index': selected, 'selected': candidates[selected], 'one_se_indices': eligible,
                          'failed_fits':failures[model],
                          'best_index': best, 'best_standard_error': se,
                          'scores': [{'candidate': candidate, 'mean_predictive_loss': mean[i], 'fold_losses': losses[i]}
                                     for i, candidate in enumerate(candidates)]}
    return reports, ledger


def paired_bootstrap(points, noise, selected, draws=1000, seed=12345):
    """Shared ~30-day temporal and 3 km spatial block draws for both models."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    valid_dates = np.concatenate([p['early'][p['valid']] for p in points])
    origin = valid_dates.min()
    regions = sorted({p['region'] for p in points if p['role']=='target'})
    labels = {r:sorted({p['region']+':'+p['spatial_block'] for p in points if p['region']==r}) for r in regions}
    records = {model: [] for model in selected}
    for draw in range(draws):
        spatial = [label for region_labels in labels.values()
                   for label in rng.choice(region_labels,len(region_labels),replace=True)]
        # Shared block multipliers preserve paired model comparisons and
        # correlations between neighboring pairs and shared acquisitions.
        max_block = max(int(np.nanmax((p['late']-origin)/30)) for p in points)
        temporal = rng.multinomial(max_block+1, np.full(max_block+1, 1/(max_block+1)))
        for model, candidate in selected.items():
            regional = {}
            for label in spatial:
                for point in points:
                    if point['region']+':'+point['spatial_block'] != label or point['role']!='target':
                        continue
                    valid = point['valid']
                    midpoint = (point['early']+point['late'])/2
                    block = np.clip(np.nan_to_num((midpoint-origin)/30, nan=0).astype(int), 0, max_block)
                    index = np.repeat(np.where(valid)[0], temporal[block[valid]])
                    if len(index) < 10:
                        continue
                    sub = {k: (v[index] if isinstance(v, np.ndarray) and v.shape[:1]==valid.shape else v) for k,v in point.items()}
                    try:
                        fitted = fit_point(sub, noise, candidate, model)
                        diag = diagnostics(fitted)
                        # Vector coefficients permit circular aggregation without phase wraps.
                        regional.setdefault(point['region'], []).append(diag['along_coefficients'])
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        pass
            records[model].append({r: np.mean(cs, axis=0) for r,cs in regional.items()})
    return {'draws': draws, 'seed_sequence': [seed, 2], 'block_days': 30, 'spatial_blocks_m': 3000,
            'scope': 'paired temporal/spatial block sampling; flow direction re-estimated in each ice refit', 'coefficients': records}


def bootstrap_intervals(bootstrap):
    """Regional circular intervals and paired parameterization differences."""
    summaries = {}
    models = bootstrap['coefficients']
    regions = {r for draws in models.values() for draw in draws for r in draw}
    for region in regions:
        summaries[region] = {}
        for model,draws in models.items():
            cs = np.array([draw[region] for draw in draws if region in draw])
            if not len(cs):
                continue
            amplitude = np.linalg.norm(cs,axis=1)
            phase = np.arctan2(cs[:,1],cs[:,0])
            center = np.angle(np.mean(np.exp(1j*phase)))
            angular = np.angle(np.exp(1j*(phase-center)))
            summaries[region][model] = {'successful_draws':len(cs),'coefficient_covariance':np.cov(cs.T),
                 'amplitude_95':np.quantile(amplitude,[.025,.975]),'phase_center_radians':center,
                 'phase_offset_95_radians':np.quantile(angular,[.025,.975]),
                 'phase_interpretation':'Circular interval conditional on supported nonzero amplitude'}
        difference = []
        if set(models)=={'free_en','along_flow'}:
            for free,along in zip(models['free_en'],models['along_flow']):
                if region in free and region in along:
                    difference.append(np.asarray(free[region])-np.asarray(along[region]))
        if difference:
            summaries[region]['paired_free_minus_along_coefficients_95'] = np.quantile(difference,[.025,.975],axis=0)
    return summaries


def tune_shortlist(points,noise,comparisons,residuals,draws,seed,checkpoint=None):
    """Choose within one predictive SE, subject to independent tuning recovery."""
    selected = {model:record['selected'] for model,record in comparisons.items()}
    ledger = [] if checkpoint is None else list(checkpoint.get('attempts',[]))
    all_passed = True
    for model,comparison in comparisons.items():
        indices = list(comparison['one_se_indices'])
        def preference(i):
            c = comparison['scores'][i]['candidate']
            return (WEIGHT_MODELS.index(c['weighting']),['constant','linear','spline_l2','spline_l1'].index(c['background']),-c['penalty'])
        passed = False
        for index in sorted(indices,key=preference):
            candidate = comparison['scores'][index]['candidate']
            logging.info('Tuning %s candidate %s: %s / %s / penalty=%g',model,index,
                         candidate['background'],candidate['weighting'],candidate['penalty'])
            previous=next((entry for entry in ledger if entry['model']==model and entry['candidate_index']==int(index)),None)
            if previous is not None:
                passed=previous['passed']
                if passed:
                    selected[model]=candidate
                    break
                continue
            report = recovery(points,noise,{model:candidate},residuals,draws=draws,seed=seed,
                              tuning=True,sensitivities=False,period_names=('joint',),screening=True)
            rows = [r for r in report['rows'] if r.get('model')==model and r.get('period')=='joint']
            failures = [r for r in rows if r.get('draws',0)<.95*draws or
                not .90<=r.get('coverage_95',0)<=.99 or
                (r.get('amplitude')==.1 and (r.get('median_amplitude_error',np.inf)>.2 or r.get('median_phase_error_degrees',np.inf)>20)) or
                (r.get('amplitude')==0 and r.get('null_false_positive_rate',1)>.05)]
            passed = bool(rows) and not failures
            ledger.append({'model':model,'candidate_index':int(index),'candidate':candidate,'passed':passed,'rows':rows})
            if checkpoint is not None:
                checkpoint['attempts']=ledger
                checkpoint['save'](ledger)
            if passed:
                selected[model] = candidate
                break
        all_passed &= passed
    return selected,{'seed_sequence':[seed,0],'passed':all_passed,'attempts':ledger}


def _noise_draw(rng, point, noise, residual_records,cache=None):
    """Resample whole empirical standardized control blocks, preserving RA pairs."""
    chosen=int(rng.integers(len(residual_records)))
    control = residual_records[chosen]
    cached=None if cache is None else cache.get(('control',chosen))
    # Remove each control observation's own quality-dependent scale before
    # recoloring at the target. A constant baseline here counts heteroskedastic
    # quality effects twice and can greatly inflate synthetic noise.
    if cached is None:
        residual=control['residual_ra'];available=control['valid']
        control_covariance=covariance_ra(noise,control['point']['snr'][available],
                                         control['point']['peak'][available],quality=True)
        centered=residual[available]-np.asarray(noise.get('control_bias_m',[0.,0.]))
        standard=np.linalg.solve(np.linalg.cholesky(control_covariance),centered[...,None])[...,0]
        control_times=(control['point']['early'][available]+control['point']['late'][available])/2
        labels=np.floor((control_times-control_times.min())/30).astype(int)
        blocks=[standard[labels==label] for label in np.unique(labels)]
        if cache is not None:cache[('control',chosen)]=blocks
    else:
        blocks=cached
    sampled = []
    while len(sampled) < len(point['early']):
        sampled.extend(blocks[rng.integers(len(blocks))])
    sampled = np.asarray(sampled[:len(point['early'])])
    factor=None if cache is None else cache.get(('target',id(point)))
    if factor is None:
        factor=np.linalg.cholesky(covariance_ra(noise,point['snr'],point['peak'],quality=True))
        if cache is not None:cache[('target',id(point))]=factor
    ra = np.einsum('nij,nj->ni', factor, sampled)
    return ra @ point['inverse'].T


def _vertical(early, late, period, amplitude, phase):
    omega = 2*np.pi/period
    return amplitude*(np.cos(omega*late-phase)-np.cos(omega*early-phase))


def horizontal_azimuth(degrees):
    """Local EN unit vector for degrees clockwise from geographic north."""
    angle=np.deg2rad(degrees)
    return np.array([np.sin(angle),np.cos(angle)])


def recovery(points, noise, selected, residual_records, draws=200, seed=12345, tuning=False,
             sensitivities=True,period_names=None,screening=False):
    """Known-signal/null recovery with disjoint, reproducible tuning/validation RNGs."""
    stream = 0 if tuning else 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    cases = [('baseline', {})]
    if sensitivities:
        cases += [('direction_'+str(a), {'angle': a}) for a in (90,105)]
        cases += [('mismatch_'+str(a), {'mismatch': a}) for a in (-20,-10,-5,5,10,20)]
        cases += [('horizontal_M2', {'horizontal': (M2_PERIOD, .03)}),
                  ('horizontal_O1', {'horizontal': (O1_PERIOD, .01)})]
        cases += [('vertical_'+str(f), {'vertical_fraction': f}) for f in (.05,.1,.2)]
        cases += [('range_tide_scale_'+str(s), {'tide_scale': s}) for s in (.8,1.,1.2)]
        cases += [('calibration_uncertainty', {'calibration': True}), ('linear_trend', {'trend': True}),
                  ('abrupt_change', {'step': True}), ('azimuth_vertical_tides', {'azimuth_tides': True})]
    rows = []
    noise_cache={}
    def report(stopped=False):
        return {'seed_sequence':[seed,stream], 'draws_per_case':draws, 'rows':rows,
                'screening_stopped_on_failure':stopped,
                'controlled_direction_convention':'degrees clockwise from geographic north; EN=[sin(azimuth),cos(azimuth)]',
                'noise':'paired empirical control residuals resampled in ~30-day blocks and recolored by calibrated covariance',
                'horizontal_O1_injection_m_per_day':.01,
                'interpretation':'Msf-frequency motion conditional on correction and nuisance-signal assumptions'}
    for point in points:
        if point['role']!='target':
            continue
        good = point['valid']
        if good.sum() < 10:
            continue
        early, late = point['early'], point['late']
        # Constant control covariance only establishes the mean flow axis.
        _, _, observed, cov, _ = point_arrays(point, noise, 'constant')
        direction = flow_direction(early[good], late[good], observed, cov)
        mean_flow = np.median(observed/(late[good]-early[good])[:, None], axis=0)
        completed_periods={}
        for period_name, start, end in [('2017', '2017-01-01', '2018-01-01'), ('2018','2018-01-01','2019-01-01'), ('joint',None,None)]:
            if period_names is not None and period_name not in period_names:
                continue
            from .velocity_basis import utc_days
            period_mask = good.copy()
            if start:
                period_mask &= (early >= utc_days(start)) & (late < utc_days(end))
            if period_mask.sum() < 10:
                continue
            key=period_mask.tobytes()
            if key in completed_periods:
                previous,saved_rows=completed_periods[key]
                rows.extend(dict(row,period=period_name,equivalent_to_period=previous) for row in saved_rows)
                continue
            period_begin=len(rows)
            logging.info('%s recovery: %s source %s, period %s',
                         'Tuning' if tuning else 'Validation',point['region'],point['source_id'],period_name)
            for case, spec in cases:
                if 'tide_scale' in spec and 'tidal_range_bias' not in point:
                    rows.append({'region':point['region'],'source_id':point['source_id'],'period':period_name,
                                 'case':case,'status':'unavailable: stored range tide bias absent'})
                    continue
                if spec.get('calibration') and 'calibration_std_en' not in point:
                    rows.append({'region':point['region'],'source_id':point['source_id'],'period':period_name,
                                 'case':case,'status':'unavailable: calibration sensitivity absent'})
                    continue
                amplitudes = [0., .025, .05, .1, .2] if case=='baseline' else [.1]
                for amplitude in amplitudes:
                    stats = {m: [] for m in selected}
                    for draw in range(draws):
                        phase = 2*np.pi*(draw+rng.random())/draws
                        assumed = horizontal_azimuth(spec['angle']) if 'angle' in spec else direction.copy()
                        angle=np.deg2rad(spec.get('mismatch',0))
                        truth_dir=np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]]) @ assumed
                        cs = amplitude*np.array([np.cos(phase),np.sin(phase)])
                        signal = harmonic(early, late) @ cs
                        truth = (late-early)[:,None]*mean_flow+signal[:,None]*truth_dir
                        nuisance_phase = rng.uniform(-np.pi,np.pi)
                        if 'horizontal' in spec:
                            p,a = spec['horizontal']
                            truth += (harmonic(early,late,p) @ (a*np.array([np.cos(nuisance_phase),np.sin(nuisance_phase)])))[:,None]*truth_dir
                        if 'vertical_fraction' in spec or spec.get('azimuth_tides'):
                            fraction = spec.get('vertical_fraction',1.)
                            vertical = _vertical(early,late,M2_PERIOD,.6*fraction,nuisance_phase)+_vertical(early,late,O1_PERIOD,.2*fraction,-nuisance_phase)
                            projection = point['geometry'][:,2].copy()
                            if spec.get('azimuth_tides'):
                                projection[0] = 0.
                            truth += vertical[:,None]*(point['inverse'] @ projection)
                        if 'tide_scale' in spec:
                            truth += (1-spec['tide_scale'])*point['tidal_range_bias'][:,None]*point['inverse'][:,0]
                        if spec.get('calibration'):
                            # One shared calibration draw, not independent pixel noise.
                            truth += rng.normal()*point['calibration_std_en']
                        if spec.get('trend'):
                            ref = np.nanmedian(early)
                            truth += (.001*((late-ref)**2-(early-ref)**2)/2)[:,None]*truth_dir
                        if spec.get('step'):
                            boundary = np.nanmedian(early)
                            truth += (.3*(np.maximum(late-boundary,0)-np.maximum(early-boundary,0)))[:,None]*truth_dir
                        noisy = truth+_noise_draw(rng,point,noise,residual_records,cache=noise_cache)
                        for model,candidate in selected.items():
                            try:
                                # Controlled 2x2 uses its explicit direction; measured
                                # directions are re-estimated from each noisy secular fit.
                                fixed = assumed if 'angle' in spec or 'mismatch' in spec else None
                                fitted = fit_point(point,noise,candidate,model,period_mask,direction=fixed,data=noisy)
                                d = diagnostics(fitted)
                                true_cs = cs*np.dot(truth_dir,fitted['direction'])
                                true_amplitude = np.linalg.norm(true_cs)
                                true_phase = np.arctan2(true_cs[1],true_cs[0])
                                difference = d['along_coefficients']-true_cs
                                covered = difference @ np.linalg.pinv(d['along_covariance']) @ difference <= chi2.ppf(.95,2)
                                phase_error = abs(np.angle(np.exp(1j*(d['phase']-true_phase))))*180/np.pi if amplitude else np.nan
                                stats[model].append((abs(d['amplitude']-true_amplitude)/max(true_amplitude,1e-15) if amplitude else d['amplitude'],
                                                     phase_error,d['p_value']<.05,covered,d['amplitude']))
                            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                                pass
                    for model,values in stats.items():
                        if not values:
                            rows.append({'region':point['region'],'source_id':point['source_id'],'period':period_name,'case':case,
                                         'amplitude':amplitude,'model':model,'status':'all fits failed'})
                            if screening:
                                return report(True)
                            continue
                        values = np.asarray(values)
                        rows.append({'region':point['region'],'source_id':point['source_id'],'period':period_name,'case':case,
                                     'amplitude':amplitude,'model':model,'draws':len(values),'attempted_draws':draws,
                                     'median_amplitude_error':float(np.median(values[:,0])),
                                     'median_phase_error_degrees':float(np.median(values[:,1])) if amplitude else None,
                                     'null_false_positive_rate':float(np.mean(values[:,2])) if amplitude==0 else None,
                                     'coverage_95':float(np.mean(values[:,3])), 'median_recovered_amplitude':float(np.median(values[:,4]))})
                        row=rows[-1]
                        if screening and (len(values)<.95*draws or not .90<=row['coverage_95']<=.99 or
                            (amplitude==.1 and (row['median_amplitude_error']>.2 or row['median_phase_error_degrees']>20)) or
                            (amplitude==0 and row['null_false_positive_rate']>.05)):
                            return report(True)
            completed_periods[key]=(period_name,rows[period_begin:])
    return report()


def acceptance(report, minimum_draws=200):
    """Require every reviewed target and model to pass the clean benchmark/null."""
    failures = []
    baseline = [r for r in report['rows'] if r.get('case')=='baseline' and r.get('period')=='joint']
    groups = {(r['region'],r.get('source_id'),r.get('model')) for r in baseline}
    for region,source,model in groups:
        amplitudes = {r.get('amplitude') for r in baseline if (r['region'],r.get('source_id'),r.get('model'))==(region,source,model)}
        if amplitudes != {0.,.025,.05,.1,.2}:
            failures.append(f'{region}/{source}/{model}: incomplete injection amplitude bank')
    if {r.get('model') for r in baseline} != {'free_en','along_flow'}:
        failures.append('Both parameterizations must have validation results')
    for row in baseline:
        label = f'{row["region"]}/{row.get("source_id")}/{row.get("model")}/{row.get("amplitude")}'
        if row.get('draws',0) < minimum_draws or row.get('draws',0) < .95*row.get('attempted_draws',1):
            failures.append(label+': insufficient successful draws')
            continue
        if row['amplitude']==.1 and (row['median_amplitude_error']>.2 or row['median_phase_error_degrees']>20):
            failures.append(label+': amplitude/phase benchmark failed')
        if row['amplitude']==0 and row['null_false_positive_rate']>.05:
            failures.append(label+': null false positives exceed 5%')
        if not .90 <= row['coverage_95'] <= .99:
            failures.append(label+': nominal interval coverage outside 90–99%')
    if not baseline:
        failures.append('No clean joint-record validation results')
    return {'passed':not failures,'failures':failures,'requirements':{'amplitude_error':.2,'phase_error_degrees':20,'null_fpr':.05,
            'coverage_95_range':[.90,.99], 'minimum_validation_draws':minimum_draws}}
