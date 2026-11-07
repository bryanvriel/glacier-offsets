"""Pair-specific radar noise estimated from independent control blocks."""
import numpy as np


SCHEMA = 'glacier-offsets-pair-noise-v1'


def calibrate_pairs(records, pooled, *, min_blocks=8, shrinkage_blocks=20.):
    """Estimate robust pair scatter, shrinking sparse estimates toward pooled noise.

    One median residual per physical spatial block prevents copied or overlapping
    observations from inflating support. Bias is recorded, never subtracted from
    glacier measurements. Unsupported pairs retain the pooled covariance.
    """
    samples = {}
    for record in records:
        point = record['point']
        group = point['spatial_block']
        for i, pair in enumerate(point['pair_id']):
            blocks = samples.setdefault(str(pair), {})
            if record['valid'][i]:
                blocks.setdefault(group, []).append(record['residual_ra'][i])
    baseline = np.asarray(pooled['covariance_ra_m2'])
    result = {}
    for pair, blocks in samples.items():
        values = np.array([np.median(v, axis=0) for v in blocks.values()]).reshape(-1, 2)
        n = len(values)
        bias = np.median(values, axis=0) if n else np.zeros(2)
        weight = n/(n+shrinkage_blocks) if n >= min_blocks else 0.
        covariance = baseline.copy()
        if weight:
            centered = values-bias
            scale = np.maximum(1.4826*np.median(abs(centered), axis=0), 1e-4)
            clipped = np.clip(centered, -4*scale, 4*scale)
            scatter = clipped.T @ clipped/max(n-1, 1)
            covariance = (1-weight)*baseline+weight*scatter
        result[pair] = {'covariance_ra_m2': covariance.tolist(), 'bias_ra_m': bias.tolist() if n else None,
                        'independent_blocks': n, 'pair_weight': weight, 'pooled_fallback': weight == 0.}
    return {'schema': SCHEMA, 'min_blocks': min_blocks, 'shrinkage_blocks': shrinkage_blocks,
            'bias_applied': False, 'pairs': result}


def pair_covariances(noise, pair_ids):
    """Look up pair matrices by identity; unseen CV pairs use pooled noise."""
    baseline = noise['covariance_ra_m2']
    pairs = noise['pair_calibration']['pairs']
    return np.array([pairs.get(str(pair), {}).get('covariance_ra_m2', baseline) for pair in pair_ids])


def pooled_noise(noise):
    return {k: v for k, v in noise.items() if k != 'pair_calibration'}
