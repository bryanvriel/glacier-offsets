"""Block-level aggregation of independent-source Msf C/S coefficients.

Each unique source is assigned to exactly one square block by its location,
so no source is counted twice. That also avoids the native-cell overcount
(about five 200 m cells per Ampcor source). A block estimate is the
equal-weight mean of its sources' C/S. Its uncertainty is a calendar-cluster
sandwich in which the sources' per-cluster score influences are averaged
*before* the outer product, so errors shared by all sources through common
acquisitions are counted once rather than averaged away. Amplitude and phase
are derived only afterwards, from the block C/S and its covariance, with the
noise bias E|m|^2 = |mu|^2 + tr(Sigma) removed in a separate debiased amplitude.

The block-mean influences are kept, so any linear combination of blocks
(neighbours, transects, regional means) has the same shared-cluster uncertainty.
"""
import numpy as np
from scipy.stats import f as f_dist


def block_keys(x, y, size):
    """Integer (ix, iy) block keys for coordinates in metres."""
    return np.floor(np.asarray(x)/size).astype(np.int64), np.floor(np.asarray(y)/size).astype(np.int64)


class BlockAccumulator:
    """Streaming sums over sources for a fixed set of blocks.

    ``labels`` passed to :meth:`add` index ``range(nblocks)``; ``influence`` is
    the per-source C/S cluster influence (sources, clusters, 2) from
    :func:`velocity_pixel_general.solve`, ``cr1`` its small-sample factor, and
    ``pointwise`` the per-source sandwich covariance (used only for the design
    effect diagnostic).
    """

    def __init__(self, nblocks, nclusters):
        self.count = np.zeros(nblocks, np.int64)
        self.cs = np.zeros((nblocks, 2))
        self.influence = np.zeros((nblocks, nclusters, 2))
        self.cr1 = np.zeros(nblocks)
        self.pointwise = np.zeros((nblocks, 2, 2))
        self.nclusters = nclusters

    def add(self, labels, cs, influence, cr1, pointwise):
        labels = np.asarray(labels)
        if influence.shape[1] != self.nclusters:
            raise ValueError('Cluster count differs from the accumulator')
        np.add.at(self.count, labels, 1)
        np.add.at(self.cs, labels, cs)
        np.add.at(self.influence, labels, np.nan_to_num(influence))
        np.add.at(self.cr1, labels, cr1)
        np.add.at(self.pointwise, labels, pointwise)

    def merge(self, other):
        """Add another accumulator over the same blocks and clusters (e.g. from a worker); returns self."""
        if other.count.shape != self.count.shape or other.nclusters != self.nclusters:
            raise ValueError('Accumulators cover different blocks or clusters')
        for name in ('count', 'cs', 'influence', 'cr1', 'pointwise'):
            setattr(self, name, getattr(self, name)+getattr(other, name))
        return self

    def finish(self):
        n = np.maximum(self.count, 1)
        influence = self.influence/n[:, None, None]
        factor = self.cr1/n
        cov = factor[:, None, None]*np.einsum('bgi,bgj->bij', influence, influence)
        active = (np.abs(influence).sum(2) > 0).sum(1)
        independent = self.pointwise/n[:, None, None]**2  # covariance if sources were independent
        design = np.divide(np.trace(cov, axis1=1, axis2=2), np.trace(independent, axis1=1, axis2=2),
                           out=np.full(len(n), np.nan), where=np.trace(independent, axis1=1, axis2=2) > 0)
        return {'sources': self.count, 'mean': self.cs/n[:, None], 'covariance': cov, 'influence': influence,
                'cr1': factor, 'clusters': active, 'design_effect': design,
                'effective_sources': np.divide(self.count, design, out=np.full(len(n), np.nan), where=design > 0)}


def combine(influence, cr1, weights):
    """Covariance of sum_b w_b * mean_b for blocks sharing calendar clusters."""
    w = np.asarray(weights, float)
    total = np.einsum('b,bgi->gi', w, influence)
    factor = float(np.sum(np.abs(w)*cr1)/max(np.sum(np.abs(w)), 1e-30))
    return factor*total.T@total


def statistics(mean, covariance, clusters, inflation=1.):
    """Wald test, debiased amplitude and phase for block C/S.

    ``inflation`` multiplies standard errors (covariance by its square). The
    test uses the small-sample sandwich reference T*(G-2)/(2(G-1)) ~ F(2, G-2)
    for G active clusters.
    """
    cov = covariance*inflation**2
    inv = np.linalg.pinv(cov, hermitian=True)
    t = np.einsum('bi,bij,bj->b', mean, inv, mean)
    g = np.asarray(clusters, float)
    valid = g > 2
    scaled = np.where(valid, t*(g-2)/np.maximum(2*(g-1), 1e-30), np.nan)
    p = np.where(valid, f_dist.sf(scaled, 2, np.maximum(g-2, 1)), np.nan)
    amplitude = np.hypot(mean[:, 0], mean[:, 1])
    phase = np.degrees(np.arctan2(mean[:, 1], mean[:, 0]))
    trace = np.trace(cov, axis1=1, axis2=2)
    debiased = np.sqrt(np.maximum(amplitude**2-trace, 0.))
    unit = np.divide(mean, amplitude[:, None], out=np.zeros_like(mean), where=amplitude[:, None] > 1e-12)
    normal = np.divide(np.stack([-mean[:, 1], mean[:, 0]], 1), amplitude[:, None]**2, out=np.zeros_like(mean),
                       where=amplitude[:, None] > 1e-12)
    amplitude_sigma = np.sqrt(np.maximum(np.einsum('bi,bij,bj->b', unit, cov, unit), 0))
    phase_sigma = np.degrees(np.sqrt(np.maximum(np.einsum('bi,bij,bj->b', normal, cov, normal), 0)))
    return {'wald': t, 'p_value': p, 'amplitude': amplitude, 'amplitude_debiased': debiased,
            'amplitude_sigma': amplitude_sigma, 'phase_deg': phase, 'phase_sigma_deg': phase_sigma,
            'component_sigma': np.sqrt(trace/2)}


def benjamini_hochberg(p, q=.05):
    """Boolean discoveries at false-discovery rate q (NaN p-values never discovered)."""
    p = np.asarray(p, float)
    finite = np.flatnonzero(np.isfinite(p))
    out = np.zeros(p.shape, bool)
    if not len(finite):
        return out
    order = finite[np.argsort(p[finite])]
    m = len(order)
    passed = p[order] <= q*np.arange(1, m+1)/m
    if passed.any():
        out[order[:np.flatnonzero(passed).max()+1]] = True
    return out
