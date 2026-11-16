"""Track/year empirical noise transfer for the temporal fitting CLI."""
import numpy as np


def learn_noise_nisar(points, excluded_region, train, options):
    from .velocity_pilot import control_residuals
    from .velocity_fit import calibrate_noise
    from .velocity_pair_noise import calibrate_pairs
    labels = sorted({str(g) for p in points for g in p['noise_group']})
    by_group = {}; pair_group = {}; records = []
    for label in labels:
        selected = []
        for point in points:
            p = dict(point)
            p['valid'] = point['valid'] & (point['noise_group'] == label)
            selected.append(p)
            pair_group.update({str(pair): str(group) for pair, group in zip(point['pair_id'], point['noise_group'])})
        arrays = control_residuals(selected, excluded_region=excluded_region, train=train)
        try: model = calibrate_noise(*arrays[:4])
        except ValueError as exc: raise ValueError(f'NISAR noise group {label}: {exc}') from exc
        if options and options['mode'] == 'pair':
            calibration = calibrate_pairs(arrays[4], model, min_blocks=options['min_pair_blocks'], shrinkage_blocks=options['shrinkage_blocks'])
            calibration['pairs'] = {k: v for k, v in calibration['pairs'].items() if pair_group[k] == label}
            model['pair_calibration'] = calibration
        by_group[label] = model; records.extend(arrays[4])
    first = next(iter(by_group.values()))
    noise = dict(first, by_group=by_group, pair_noise_group=pair_group, group_by='track_year')
    # The top-level matrix is a reporting summary; transfer always selects a group.
    noise['covariance_ra_m2'] = np.mean([v['covariance_ra_m2'] for v in by_group.values()], axis=0).tolist()
    if options and options['mode'] == 'pair':
        noise['pair_calibration'] = {'schema': 'glacier-offsets-pair-noise-v1', 'bias_applied': False,
                                    'pairs': {k: v for model in by_group.values() for k, v in model['pair_calibration']['pairs'].items()}}
    return noise, records


def grouped_covariance(noise, snr, peak, quality, pair_ids, diagnostics):
    from .velocity_fit import covariance_ra
    if pair_ids is None: raise ValueError('NISAR covariance transfer requires pair identities')
    pair_ids = np.asarray(pair_ids).astype(str); matrices = np.empty((len(pair_ids), 2, 2)); reports = []
    for i, pair in enumerate(pair_ids):
        if pair not in noise['pair_noise_group']: raise ValueError(f'Unknown NISAR noise group for pair {pair}')
        label = noise['pair_noise_group'][pair]
        local = noise['by_group'][label]
        value = covariance_ra(local, np.asarray(snr)[i:i+1], np.asarray(peak)[i:i+1], quality, [pair], return_diagnostics=diagnostics)
        if diagnostics:
            matrices[i] = value[0][0]; reports.append(value[1])
        else: matrices[i] = value[0]
    if not diagnostics: return matrices
    # Preserve the ordinary transfer diagnostic field shapes.
    return matrices, {k: np.concatenate([np.atleast_1d(r[k]) for r in reports], axis=0) if isinstance(reports[0][k], np.ndarray) else reports[0][k] for k in reports[0]}
