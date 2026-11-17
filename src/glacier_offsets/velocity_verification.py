"""Strict physical replay and precision-aware checks of stored diagnostics."""
import numpy as np


def check_stored_rms(computed, stored, *, tolerance=1e-6, source_ids=None):
    """Check values against the storage dtype's rounding interval plus roundoff.

    A float32 RMS near 100 m cannot represent micrometre differences. This
    accounts only for encoding precision; physical-model agreement is checked
    separately in float64 by ``check_physical_rms``.
    """
    computed, stored = np.asarray(computed, dtype=float), np.asarray(stored)
    if computed.shape != stored.shape or stored.dtype.kind != 'f':
        raise ValueError('Invalid stored radar RMS shape or dtype')
    values = stored.astype(float)
    lower = np.nextafter(stored, np.array(-np.inf, dtype=stored.dtype)).astype(float)
    upper = np.nextafter(stored, np.array(np.inf, dtype=stored.dtype)).astype(float)
    lower = values-(values-lower)/2
    upper = values+(upper-values)/2
    invalid = ~np.isfinite(computed) | ~np.isfinite(values) | (computed < 0) | (values < 0)
    error = np.maximum(lower-computed, computed-upper)
    bad = invalid | (error > tolerance)
    if bad.any():
        location = tuple(np.argwhere(bad)[0])
        source = source_ids[location[0]] if source_ids is not None else location[0]
        raise ValueError(f'Stored radar RMS does not reproduce: source {source}, channel {location[-1]}, '
                         f'computed={computed[location]:.12g} m, stored={values[location]:.12g} m, '
                         f'dtype={stored.dtype}; outside storage rounding interval by {error[location]:.3g} m '
                         f'(tolerance {tolerance:g} m)')
    return float(np.max(abs(computed-values), initial=0.))


def check_physical_rms(computed, replayed, *, tolerance=1e-6, source_ids=None):
    """Compare independent predictions to the solver in full precision."""
    computed, replayed = np.asarray(computed, dtype=float), np.asarray(replayed, dtype=float)
    if computed.shape != replayed.shape:
        raise ValueError('Invalid physical radar RMS shape')
    difference = abs(computed-replayed)
    bad = ~np.isfinite(difference) | (difference > tolerance)
    if bad.any():
        location = tuple(np.argwhere(bad)[0])
        source = source_ids[location[0]] if source_ids is not None else location[0]
        raise ValueError(f'Standalone radar prediction/RMS does not reproduce: source {source}, '
                         f'channel {location[-1]}, independent={computed[location]:.12g} m, '
                         f'solver={replayed[location]:.12g} m, difference={difference[location]:.3g} m '
                         f'(tolerance {tolerance:g} m)')
    return float(np.max(difference, initial=0.))
