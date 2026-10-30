"""Named-parameter pixel inversion for one or more calendar-year regimes.

Each observed calendar year (by pair calendar midpoint) has its own EN secular
velocity and, where both spacecraft orders occur, an azimuth order offset.
Residual vertical M2/O1 and along-flow Msf coefficients are shared. Secular
displacement integrates each year's velocity over the exact burst-endpoint
overlap with that year; harmonics are integrated exactly. Annual noise and
order groups follow the recorded midpoint year. For two within-year regimes
this reproduces the version-1 two-year design column for column.
"""
import numpy as np

from .velocity_basis import harmonic, PERIOD
from .velocity_collection import year_start
from .velocity_quality import quiet_median
from .velocity_systematics import order_screen, weighted_median
from .velocity_vertical import vertical_columns


VERTICAL_NAMES = ('vertical_M2_C', 'vertical_M2_S', 'vertical_O1_C', 'vertical_O1_S')
TARGET_NAMES = ('Msf_C', 'Msf_S')
MODELS = ('residual_vertical', 'azimuth_direct')


class Layout:
    """Calendar-year regimes and their named parameters.

    Regime k spans [Jan 1 of year k, Jan 1 of the next observed year); the
    first and last regimes are open-ended, so a gap year belongs to the
    preceding regime instead of introducing an unconstrained velocity.
    """

    def __init__(self, years, order_years=None):
        self.years = tuple(int(y) for y in years)
        if not self.years or list(self.years) != sorted(set(self.years)):
            raise ValueError('Regime years must be unique and increasing')
        order_years = self.years if order_years is None else set(order_years)
        self.order_years = tuple(y for y in self.years if y in order_years)
        self.bounds = np.array([-np.inf, *[year_start(y) for y in self.years[1:]], np.inf])

    @classmethod
    def from_observations(cls, h):
        years = h['years'][:].tolist()
        index = h['year_index'][:]
        order = h['order'][:]
        return cls(years, [y for k, y in enumerate(years) if np.any(order[index == k] != 0)])

    @property
    def joint_names(self):
        return (tuple(f'{axis}_velocity_{y}' for y in self.years for axis in ('east', 'north'))
                + tuple(f'order_{y}' for y in self.order_years) + VERTICAL_NAMES + TARGET_NAMES)

    @property
    def azimuth_names(self):
        return (tuple(f'azimuth_velocity_{y}' for y in self.years)
                + tuple(f'order_{y}' for y in self.order_years) + TARGET_NAMES)

    def names(self, model):
        return self.joint_names if model == 'residual_vertical' else self.azimuth_names

    def overlaps(self, early, late):
        """Exact days of each pair interval inside each regime (NaN for missing endpoints)."""
        early = np.asarray(early, float)[..., None]
        late = np.asarray(late, float)[..., None]
        overlap = np.minimum(late, self.bounds[1:])-np.maximum(early, self.bounds[:-1])
        return np.where(np.isnan(overlap), np.nan, np.maximum(overlap, 0.))

    def record(self):
        return {'years': list(self.years), 'order_years': list(self.order_years),
                'regime_starts_utc': ['open' if not np.isfinite(b) else f'{y}-01-01T00:00:00Z'
                                      for b, y in zip(self.bounds[:-1], self.years)],
                'joint_parameters': list(self.joint_names), 'azimuth_parameters': list(self.azimuth_names)}


def parameter_units(names):
    return ['m' if name.startswith(('order_', 'vertical_')) else 'm day-1' for name in names]


def load_field(h, body, index, remove_mixed_windows=False):
    """Observations for selected sources with solid-Earth corrections applied once."""
    field = {k: h[k][index] for k in ('early', 'late', 'data', 'peak')}
    valid = h['valid_including_mixed'][index]
    if remove_mixed_windows:
        valid[h['known_mixed_source'][index]] = False
    if body is not None:
        field['data'] = field['data']-body['set_ra_m'][index]
    field['valid'] = valid & np.isfinite(field['data']).all(-1)
    for k in ('order', 'year_index', 'calendar_early', 'calendar_late', 'within_year', 'file_index'):
        field[k] = h[k][:]
    field['order'] = field['order'].astype(float)
    field['geometry_by_file'] = h['geometry_by_file'][:, index]
    return field


def pair_geometry(field):
    """(source, pair, radar, ENU) geometry from each pair's own package."""
    if 'geometry_by_file' in field:
        return np.moveaxis(field['geometry_by_file'][field['file_index']], 0, 1)
    return np.moveaxis(field['geometry_by_year'][field['year_index']], 0, 1)


def regime_geometry(field, k):
    """Geometry of the first pair labeled with regime k (version-1 convention)."""
    if 'geometry_by_file' not in field:
        return field['geometry_by_year'][k]
    first = np.flatnonzero(field['year_index'] == k)[0]
    return field['geometry_by_file'][field['file_index'][first]]


def background_rates(field, train, threshold=.08):
    """Train-only robust radar secular rates and order bias (version-1 estimator)."""
    dt = field['late']-field['early']
    data = field['data']
    order = field['order']
    w = (field['valid'] & train[None] & np.isfinite(field['peak']) & (field['peak'] >= threshold)).astype(float)
    rate = data/dt[..., None]
    vr = weighted_median(rate[..., 0], w)
    plus = w*(order > 0)
    minus = w*(order < 0)
    yp = weighted_median(rate[..., 1], plus)
    ym = weighted_median(rate[..., 1], minus)
    xp = weighted_median(order/dt, plus)
    xm = weighted_median(order/dt, minus)
    with np.errstate(invalid='ignore', divide='ignore'):
        bias = (yp-ym)/(xp-xm)
        va = yp-bias*xp
    va = np.where(np.isfinite(va), va, weighted_median(rate[..., 1], w))
    return np.stack([vr, va], axis=-1), np.nan_to_num(bias)


def _local(field, take):
    local = {k: field[k][:, take] for k in ('early', 'late', 'data', 'valid', 'peak')}
    local.update(order=np.broadcast_to(field['order'][take], local['valid'].shape),
                 calendar_early=field['calendar_early'][take], calendar_late=field['calendar_late'][take])
    return local


def _angles(velocities):
    """Maximum pairwise annual direction change; zero for one regime."""
    ns = velocities.shape[1]
    if len(velocities) == 1:
        return np.zeros(ns)
    norms = np.linalg.norm(velocities, axis=2)
    angle = np.full(ns, np.nan)
    for i in range(len(velocities)):
        for j in range(i+1, len(velocities)):
            dot = np.einsum('si,si->s', velocities[i], velocities[j])
            cosine = np.divide(dot, norms[i]*norms[j], out=np.full(ns, np.nan), where=norms[i]*norms[j] > 1e-8)
            angle = np.fmax(angle, np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return angle


def screen(field, layout, screening):
    """Version-1 screen within each year; purged annual predictions across years."""
    good = np.zeros_like(field['valid'])
    within = field['within_year'] if 'within_year' in field else np.ones(field['valid'].shape[1], bool)
    velocities, trainings = [], []
    ns = field['valid'].shape[0]
    for k in range(len(layout.years)):
        take = (field['year_index'] == k) & within
        if not take.any():
            velocities.append(np.zeros((ns, 2)))
            trainings.append(None)
            continue
        local = _local(field, take)
        selected = order_screen(local, threshold=screening['min_peak'],
                                max_range_residual_m=screening['max_range_residual_m'],
                                max_azimuth_residual_m=screening['max_azimuth_residual_m'])
        good[:, take] = selected
        rates, bias = background_rates({**local, 'valid': selected}, np.ones(take.sum(), bool), screening['min_peak'])
        # Version 1 recovered rates from order-removed interval predictions;
        # keep its exact arithmetic so two-year results replay bit for bit.
        dt = local['late']-local['early']
        prediction = np.nan_to_num(dt)[..., None]*rates[:, None]
        prediction[..., 1] += local['order']*bias[:, None]
        prediction[..., 1] -= local['order']*bias[:, None]
        rates = quiet_median(prediction/dt[..., None], axis=1)
        g = regime_geometry(field, k)
        finite = np.isfinite(g[..., :2]).all(axis=(1, 2))
        safe = np.where(finite[:, None, None], g[..., :2], np.eye(2))
        velocities.append(np.linalg.solve(safe, np.nan_to_num(rates)[..., None])[..., 0])
        trainings.append(local)
    for t in np.flatnonzero(~within):
        good[:, t] = cross_year_screen(field, layout, t, trainings, screening)
    velocities = np.array(velocities)
    mean = np.mean(velocities, axis=0)
    speed = np.linalg.norm(mean, axis=1)
    axis = np.divide(mean, speed[:, None], out=np.tile([1., 0.], (len(mean), 1)), where=speed[:, None] > 1e-8)
    return good, axis, velocities, speed, _angles(velocities)


def cross_year_screen(field, layout, t, trainings, screening):
    """Predict a cross-year pair from integrated annual training backgrounds.

    Each annual robust background is trained on that year's within-year pairs,
    excluding every pair sharing an acquisition day with the tested pair.
    """
    shared = np.floor([field['calendar_early'][t], field['calendar_late'][t]])
    early, late = field['early'][:, t], field['late'][:, t]
    overlap = np.nan_to_num(layout.overlaps(early, late))
    label = field['year_index'][t]
    prediction = np.zeros((len(early), 2))
    supported = np.isfinite(early+late)
    for k, local in enumerate(trainings):
        need = overlap[:, k] > 0
        if not need.any() and k != label:
            continue
        if local is None:
            supported &= ~need if k != label else False
            continue
        train = ~(np.isin(np.floor(local['calendar_early']), shared) | np.isin(np.floor(local['calendar_late']), shared))
        rates, bias = background_rates(local, train, screening['min_peak'])
        prediction += overlap[:, k, None]*np.nan_to_num(rates)
        supported &= ~need | np.isfinite(rates).all(-1)
        if k == label:
            prediction[:, 1] += field['order'][t]*bias
    residual = field['data'][:, t]-prediction
    peak = field['peak'][:, t]
    return (field['valid'][:, t] & supported & np.isfinite(peak) & (peak >= screening['min_peak'])
            & np.isfinite(residual).all(-1) & (abs(residual[:, 0]) <= screening['max_range_residual_m'])
            & (abs(residual[:, 1]) <= screening['max_azimuth_residual_m']))


def design(field, layout, axis, covariances, model='residual_vertical'):
    early, late = field['early'], field['late']
    year = field['year_index']
    order = field['order']
    overlap = np.nan_to_num(layout.overlaps(early, late))
    ns, nt = early.shape
    hb = np.nan_to_num(harmonic(early, late))
    names = layout.names(model)
    joint = model == 'residual_vertical'
    if model not in MODELS:
        raise ValueError('Unknown pixel model')
    a = np.zeros((ns, nt, 2 if joint else 1, len(names)))
    geom = pair_geometry(field)
    ny = len(layout.years)
    order_columns = {y: names.index(f'order_{y}') for y in layout.order_years}
    if joint:
        for k in range(ny):
            a[:, :, :, 2*k:2*k+2] = overlap[:, :, k, None, None]*geom[..., :2]
        for k, y in enumerate(layout.years):
            if y in order_columns:
                a[:, :, 1, order_columns[y]] = order[None]*(year == k)[None]
        start = names.index(VERTICAL_NAMES[0])
        a[:, :, :, start:start+4] = vertical_columns(early, late)[:, :, None, :]*geom[..., 2, None]
        a[:, :, :, -2:] = hb[:, :, None, :]*np.einsum('stci,si->stc', geom[..., :2], axis)[..., None]
        inv = np.linalg.inv(np.linalg.cholesky(covariances)[year])
        y = field['data']
    else:
        for k, yr in enumerate(layout.years):
            a[:, :, 0, k] = overlap[:, :, k]
            if yr in order_columns:
                a[:, :, 0, order_columns[yr]] = order[None]*(year == k)[None]
        a[:, :, 0, -2:] = hb
        inv = (1/np.sqrt(covariances[year, 1, 1]))[:, None, None]
        y = field['data'][:, :, [1]]
    a = np.nan_to_num(np.einsum('tij,stjk->stik', inv, a))
    y = np.nan_to_num(np.einsum('tij,stj->sti', inv, y))
    return a, y, geom, names


def solve(field, layout, good, axis, covariances, model='residual_vertical', *, cluster_days=30.):
    """Independent-source GLS with calendar-cluster diagnostic covariance."""
    a, y, geom, names = design(field, layout, axis, covariances, model)
    ns, nt, nc, np_ = a.shape
    w = good.astype(float)
    normal = np.einsum('st,stci,stcj->sij', w, a, a)
    rhs = np.einsum('st,stci,stc->si', w, a, y)
    scale = np.sqrt(np.maximum(np.diagonal(normal, axis1=1, axis2=2), 1e-30))
    unit = normal/scale[:, :, None]/scale[:, None, :]
    eig = np.linalg.eigvalsh(unit)
    rank = np.sum(eig > np.maximum(eig[:, -1, None]*1e-10, 1e-12), axis=1)
    inverse = np.linalg.pinv(unit, rcond=1e-10, hermitian=True)/scale[:, :, None]/scale[:, None, :]
    coef = np.einsum('sij,sj->si', inverse, rhs)
    prediction = np.einsum('stci,si->stc', a, coef)
    residual = y-prediction
    # Target information after marginalizing all nuisance coefficients.
    nn = normal[:, :-2, :-2]
    nh = normal[:, :-2, -2:]
    sc = np.sqrt(np.maximum(np.diagonal(nn, axis1=1, axis2=2), 1e-30))
    ninv = np.linalg.pinv(nn/sc[:, :, None]/sc[:, None, :], rcond=1e-10, hermitian=True)/sc[:, :, None]/sc[:, None, :]
    info = normal[:, -2:, -2:]-np.einsum('sji,sjk,skl->sil', nh, ninv, nh)
    info = (info+info.swapaxes(1, 2))/2
    teig = np.linalg.eigvalsh(info)
    condition = np.divide(teig[:, 1], teig[:, 0], out=np.full(ns, np.inf), where=teig[:, 0] > 0)
    estimable = (rank == np_) & (teig[:, 0] > 1e-8) & (condition < 1e8)
    mid = (field['calendar_early']+field['calendar_late'])/2
    labels = np.floor((mid-mid.min())/cluster_days).astype(int)
    blocks = np.unique(labels)
    indicator = (blocks[:, None] == labels[None]).astype(float)
    scores = np.einsum('st,stci,stc->sti', w, a, residual)
    clusters = np.einsum('bt,sti->sbi', indicator, scores)
    adjusted = np.einsum('sip,sbp->sbi', inverse[:, -2:, :], clusters)
    active = np.einsum('bt,st->sb', indicator, w) > 0
    nb = active.sum(1)
    nobs = nc*good.sum(1)
    cr1 = (np.divide(nb, nb-1, out=np.zeros(ns, float), where=nb > 1)
           * np.divide(nobs-1, nobs-rank, out=np.zeros(ns, float), where=nobs > rank))
    cov = np.einsum('sbi,sbj,s->sij', adjusted, adjusted, cr1)
    cs = coef[:, -2:]
    amplitude = np.linalg.norm(cs, axis=1)
    phase = np.degrees(np.arctan2(cs[:, 1], cs[:, 0]))
    gradient = np.divide(cs, amplitude[:, None], out=np.zeros_like(cs), where=amplitude[:, None] > 1e-12)
    amplitude_sigma = np.sqrt(np.maximum(0, np.einsum('si,sij,sj->s', gradient, cov, gradient)))
    phase_gradient = np.divide(np.stack([-cs[:, 1], cs[:, 0]], axis=1), amplitude[:, None]**2,
                               out=np.zeros_like(cs), where=amplitude[:, None] > 1e-12)
    phase_sigma = np.degrees(np.sqrt(np.maximum(0, np.einsum('si,sij,sj->s', phase_gradient, cov, phase_gradient))))
    if nc == 2:
        physical = np.einsum('tij,stj->sti', np.linalg.cholesky(covariances)[field['year_index']], residual)
    else:
        physical = residual*np.sqrt(covariances[field['year_index'], 1, 1])[None, :, None]
    count = good.sum(1)
    rms = np.sqrt(np.einsum('st,stc->sc', w, physical**2)/np.maximum(count[:, None], 1))
    sse = np.einsum('st,stc->s', w, residual**2)
    improvement = np.einsum('si,sij,sj->s', cs, info, cs)
    partial_r2 = improvement/np.maximum(sse+improvement, 1e-30)
    return {'coefficients': coef, 'names': names, 'cs': cs, 'amplitude': amplitude, 'phase_deg': phase,
            'coefficient_covariance': cov, 'amplitude_sigma': amplitude_sigma, 'phase_sigma_deg': phase_sigma,
            'target_condition': condition, 'rank': rank, 'estimable': estimable, 'rms_m': rms,
            'partial_r2': partial_r2, 'calendar_clusters': nb}


def coverage(field, layout, good, options):
    counts = np.stack([good[:, field['year_index'] == k].sum(1) for k in range(len(layout.years))], axis=1)
    lo = np.min(np.where(good, field['early'], np.inf), axis=1)
    hi = np.max(np.where(good, field['late'], -np.inf), axis=1)
    span = np.where(good.any(1), hi-lo, 0.)
    mid = (field['early']+field['late'])/2
    nbin = options['phase_bins']
    labels = np.floor(np.nan_to_num(np.mod(mid, PERIOD)/PERIOD)*nbin).astype(int) % nbin
    bins = np.stack([np.any(good & (labels == b), axis=1) for b in range(nbin)], axis=1).sum(1)
    fit = ((counts.min(1) >= options['min_pairs_per_year']) & (counts.sum(1) >= options['min_pairs_total'])
           & (span >= options['min_span_days']) & (bins >= options['min_phase_bins']))
    standard = fit & (counts.sum(1) >= options['min_pairs_standard'])
    return {'counts': counts, 'span_days': span, 'phase_bins': bins, 'fit_support': fit, 'standard_coverage': standard}


def projections(field, layout, axis):
    """Azimuth projection of the flow axis for each regime's geometry."""
    return np.stack([np.einsum('si,si->s', regime_geometry(field, k)[:, 1, :2], axis)
                     for k in range(len(layout.years))], axis=1)


def predict(field, layout, coefficients, names, axis):
    """Physical radar prediction from named joint coefficients alone."""
    coef = np.asarray(coefficients)
    index = {name: j for j, name in enumerate(names)}
    early, late = field['early'], field['late']
    geom = pair_geometry(field)
    overlap = np.nan_to_num(layout.overlaps(early, late))
    prediction = np.zeros(early.shape+(2,))
    for k, y in enumerate(layout.years):
        velocity = coef[:, [index[f'east_velocity_{y}'], index[f'north_velocity_{y}']]]
        prediction += overlap[:, :, k, None]*np.einsum('stci,si->stc', geom[..., :2], velocity)
        if f'order_{y}' in index:
            prediction[..., 1] += field['order'][None]*(field['year_index'] == k)[None]*coef[:, index[f'order_{y}'], None]
    vertical = coef[:, [index[name] for name in VERTICAL_NAMES]]
    prediction += np.einsum('stj,sj->st', vertical_columns(early, late), vertical)[..., None]*geom[..., 2]
    target = coef[:, [index[name] for name in TARGET_NAMES]]
    prediction += (np.einsum('stj,sj->st', np.nan_to_num(harmonic(early, late)), target)[..., None]
                   * np.einsum('stci,si->stc', geom[..., :2], axis))
    return prediction


def fit_sources(h, body, layout, covariances, parameters, index):
    """Screen, coverage and both model fits for one bounded source chunk."""
    s, c = parameters['screening'], parameters['coverage']
    f = load_field(h, body, index, s['remove_mixed_windows'])
    good, axis, velocities, speed, angle = screen(f, layout, s)
    support = coverage(f, layout, good, c)
    values = {'clean_pairs_by_year': support['counts'], 'span_days': support['span_days'],
              'phase_bins': support['phase_bins'], 'fit_support': support['fit_support'],
              'standard_coverage': support['standard_coverage'], 'flow_axis_EN': axis,
              'robust_velocity_EN_by_year': np.moveaxis(velocities, 0, 1), 'mean_flow_speed': speed,
              'year_direction_change_deg': angle, 'azimuth_flow_projection_by_year': projections(f, layout, axis)}
    fits = {}
    for model in MODELS:
        fitted = solve(f, layout, good, axis, covariances, model, cluster_days=parameters['uncertainty']['cluster_days'])
        estimable = fitted['estimable'] & support['fit_support']
        fitted['estimable'] = estimable
        for name, value in fitted.items():
            if name != 'names' and np.issubdtype(np.asarray(value).dtype, np.floating):
                fitted[name] = np.where(estimable.reshape((-1,)+(1,)*(np.ndim(value)-1)), value, np.nan)
        fits[model] = fitted
    return f, good, values, fits
