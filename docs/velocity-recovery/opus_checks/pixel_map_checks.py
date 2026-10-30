"""Checks on the 2017+2018 pixelwise Msf maps (PIXEL_MAPS_2017_2018_2026-10-29.md).

Parts (default: all):
  collinearity  range-channel angle between Msf and vertical M2/O1 columns; M2 height -> Msf gain
  cats          residual vertical M2/O1 vs the CATS correction each input actually applied; hinge-zone
                bands vs response factor; range-only bedrock fit with and without a range order term
  calibration   bedrock Wald tail rates of the stored cluster covariance (chi2 and small-cluster F)
  noise         control azimuth/range noise retrained on each half-year with the run's own procedure
  halves        regional mean along-flow Msf vectors per half-year and per year
  gps           King, Makinson & Gudmundsson (2011, GRL) LAR1-3 GPS Msf and M2 alias vs the map

Run in the `ice` environment from any directory:
  python docs/velocity-recovery/opus_checks/pixel_map_checks.py [part ...]
Reads .runtime/velocity-recovery-larsen/maps-2017-2018/ and the two input NetCDFs; writes nothing.
"""
import os, sys
from pathlib import Path
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools')]
os.chdir(REPO)   # run_pixel_velocity_inversion and the noise record use repo-relative paths
import numpy as np
import h5py
import h5netcdf
from pyproj import Transformer
from scipy.stats import f as fdist
from glacier_offsets.velocity_basis import harmonic, utc_days, PERIOD
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_pixel import screen_years, design
from run_pixel_velocity_inversion import get_field

OUT = REPO/'.runtime/velocity-recovery-larsen/maps-2017-2018'
INPUTS = [REPO/'.runtime/velocity-recovery-larsen/frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc',
          REPO/'work/larsen/stack/packages2018/horizontal_2018.nc']
parts = sys.argv[1:] or ['collinearity', 'cats', 'calibration', 'noise', 'halves', 'gps']

h = h5py.File(OUT/'observations.h5'); body = h5py.File(OUT/'solid_earth_tides.h5'); c = h5py.File(OUT/'source_coefficients.h5')
cov = c['covariance_by_year'][:]
speed = c['mean_flow_speed'][:]; standard = c['standard_coverage'][:]; est = c['residual_vertical/estimable'][:]
P = c['azimuth_flow_projection_by_year'][:]; proj = np.abs(P).min(1)
coef = c['residual_vertical/coefficients'][:]
x = h['x'][:][h['col'][:]]/1e3; y = h['y'][:][h['row'][:]]/1e3
display = est & standard & (speed >= .1) & (proj >= .3)
bedrock = est & standard & (speed < .03)
shelf = est & standard & (speed > .3) & (x < -2150) & (y > 1015)
print(f'display sources {display.sum()}, bedrock-like (speed<0.03) {bedrock.sum()}, west shelf interior {shelf.sum()}')


def mean_vec(z):
    m = np.mean(z)
    return f'{abs(m):.3f} @ {np.degrees(np.angle(m)):6.1f}'


def m2_to_msf_gain():
    """Least-squares map T with vertical-M2 columns ~= Msf columns @ T (same sampled pattern)."""
    e, l = h['early'][0], h['late'][0]; m = np.isfinite(e) & np.isfinite(l)
    a = np.nan_to_num(harmonic(e[None], l[None]))[0][m]; b = vertical_columns(e[None], l[None])[0][m][:, :2]
    return np.linalg.lstsq(a, b, rcond=None)[0]


if 'collinearity' in parts:
    print('\n[collinearity] principal angles between range-channel column pairs, 300 display sources')
    idx = np.sort(np.random.default_rng(0).choice(np.flatnonzero(display), 300, replace=False))
    e, l, v = h['early'][idx], h['late'][idx], h['valid'][idx]
    hb = np.nan_to_num(harmonic(e, l)); vc = vertical_columns(e, l); angles = []
    for i in range(len(idx)):
        m = v[i] & np.isfinite(e[i]) & np.isfinite(l[i]); qa = np.linalg.qr(hb[i, m])[0]
        row = []
        for sl in (slice(0, 2), slice(2, 4)):
            sv = np.linalg.svd(qa.T @ np.linalg.qr(vc[i, m][:, sl])[0], compute_uv=False)
            row.extend(np.degrees(np.arccos(np.clip(sv, 0, 1))))
        angles.append(row)
    angles = np.array(angles)
    print(f'  Msf vs vertical M2: median {np.median(angles[:, :2], 0).round(3)} deg, max {angles[:, :2].max(0).round(3)}')
    print(f'  Msf vs vertical O1: median {np.median(angles[:, 2:], 0).round(1)} deg')
    T = m2_to_msf_gain()
    print(f'  1 m of M2 height = {np.linalg.svd(T, compute_uv=False).round(4)} m/day of Msf velocity')
    geom = h['geometry_by_file'][0]; axis = c['flow_axis_EN'][:]
    vert = np.linalg.norm((coef[:, 6:8] @ T.T)*geom[:, 0, 2, None], axis=1)
    along = np.linalg.norm(coef[:, 10:12]*np.einsum('si,si->s', geom[:, 0, :2], axis)[:, None], axis=1)
    for name, s in (('display', display), ('bedrock', bedrock)):
        print(f'  {name}: median |M2 height| {np.median(np.linalg.norm(coef[s, 6:8], axis=1)):.3f} m; range Msf-equivalent of '
              f'vertical term {np.median(vert[s]):.4f} vs range part of along-flow Msf {np.median(along[s]):.4f} m/day')
    joint = c['residual_vertical/cs'][:]; direct = c['azimuth_direct/cs'][:]
    sign = np.sign(P[:, 0]); ok = display & (proj >= .5)
    print(f'  joint vs direct-azimuth/projection: median |difference| '
          f'{np.median(np.linalg.norm(joint[ok] - direct[ok]/P[ok, 0, None], axis=1)):.4f} m/day '
          '(year-1 projection; the report quotes 0.0007 with its own conversion)')

if 'cats' in parts:
    print('\n[cats] residual vertical terms vs applied CATS2008_v2023 correction')
    row, col = h['row'][:], h['col'][:]; tv = []
    for p in INPUTS:
        with h5netcdf.File(p, 'r') as f:
            tv.append(f.variables['tidal_vertical_displacement_m'][:][:, row, col].T)
            resp = f.variables['tidal_response_factor'][:][row, col]
    tv = np.concatenate(tv, 1).astype(float); e, l = h['early'][:], h['late'][:]
    rng = np.random.default_rng(1)
    for name, s in (('west shelf interior', shelf), ('bedrock speed<0.03', bedrock)):
        idx = np.sort(rng.choice(np.flatnonzero(s), min(600, s.sum()), replace=False)); fits = []
        for i in idx:
            m = np.isfinite(e[i]) & np.isfinite(l[i]) & np.isfinite(tv[i])
            fits.append(np.linalg.lstsq(vertical_columns(e[i][m], l[i][m]), tv[i][m], rcond=None)[0])
        fits = np.array(fits)
        for k, sl in (('M2', slice(0, 2)), ('O1', slice(2, 4))):
            zc = fits[:, sl][:, 0] + 1j*fits[:, sl][:, 1]; zr = coef[idx, 6 + sl.start] + 1j*coef[idx, 7 + sl.start]
            ratio = np.mean(zr/zc) if np.median(abs(zc)) > 1e-3 else np.nan
            print(f'  {name:20s} {k}: CATS applied {np.median(abs(zc)):.3f} m @ {np.degrees(np.angle(np.mean(zc))):6.1f}; '
                  f'residual mean {mean_vec(zr)}; mean(residual/CATS) {ratio:.3f}; response median {np.nanmedian(resp[idx]):.2f}')
    z = coef[:, 6] + 1j*coef[:, 7]; ph = np.degrees(np.angle(z)); ok = est & standard
    for name, s in (('in-phase bands (|M2|>0.2, phase 60..200)', ok & (abs(z) > .2) & ((ph > 60) | (ph < -160))),
                    ('shelf-type (phase -100..-30, |M2|>0.08)', ok & (abs(z) > .08) & (ph < -30) & (ph > -100))):
        r = resp[s]
        print(f'  {name}: n={s.sum()}, response 10/50/90% {np.nanpercentile(r, [10, 50, 90]).round(2)}, '
              f'=0 {np.mean(r == 0):.2f}, 0<r<1 {np.mean((r > 0) & (r < 1)):.2f}')
    # Range-only bedrock fits: secular per year + vertical M2/O1, optionally a per-year range order term.
    fits = {'base': [], 'order': []}
    for k in range(0, bedrock.sum(), 1000):
        idx = np.flatnonzero(bedrock)[k:k + 1000]; f = get_field(h, body, idx); good = screen_years(f)[0]
        dt = np.nan_to_num(f['late'] - f['early']); yr = f['year_index']; vc = vertical_columns(f['early'], f['late'])
        gU = f['geometry_by_year'][yr][..., 0, 2].T
        cols = [dt*(yr == 0), dt*(yr == 1)] + [vc[..., j]*gU for j in range(4)]
        extra = [np.broadcast_to(f['order']*(yr == j), dt.shape) for j in (0, 1)]
        for name, cc in (('base', cols), ('order', cols + extra)):
            a = np.stack(cc, -1); w = good.astype(float); r = np.nan_to_num(f['data'][..., 0])
            n = np.einsum('st,sti,stj->sij', w, a, a); rhs = np.einsum('st,sti,st->si', w, a, r); ok = good.sum(1) >= 40
            out = np.full((len(idx), a.shape[-1]), np.nan); out[ok] = np.linalg.solve(n[ok], rhs[ok][..., None])[..., 0]
            fits[name].append(out)
    for name, rows in fits.items():
        F = np.concatenate(rows); z = F[:, 2] + 1j*F[:, 3]; z = z[np.isfinite(z)]
        extra = f'; median range order 2017/2018 {np.nanmedian(F[:, 6]):.3f}/{np.nanmedian(F[:, 7]):.3f} m' if name == 'order' else ''
        print(f'  bedrock range-only [{name}]: mean M2 {mean_vec(z)}, median |M2| {np.median(abs(z)):.3f} m{extra}')

if 'calibration' in parts:
    print('\n[calibration] bedrock Wald statistics of stored cluster covariance (regional median C/S removed)')
    for model in ('azimuth_direct', 'residual_vertical'):
        s = bedrock & c[model + '/estimable'][:]
        cs = c[model + '/cs'][:][s]; C = c[model + '/coefficient_covariance'][:][s]
        G = c[model + '/calendar_clusters'][:][s].astype(float); d = cs - np.median(cs, 0)
        w = np.einsum('si,si->s', d, np.linalg.solve(C, d[..., None])[..., 0])
        line = f'  {model}: clusters {np.median(G):.0f}; empirical/claimed trace {np.trace(np.cov(cs.T))/np.mean(np.trace(C, axis1=1, axis2=2)):.2f}'
        for a in (.05, .01):
            line += (f'; alpha {a}: chi2 {np.mean(w > -2*np.log(a)):.3f}, '
                     f'F {np.mean(w > 2*(G - 1)/(G - 2)*fdist.ppf(1 - a, 2, G - 2)):.3f}')
        print(line + f'; median |cs| {np.median(np.linalg.norm(cs, axis=1)):.4f} m/day')

if 'noise' in parts:
    from glacier_offsets.velocity_fit import calibrate_noise
    from glacier_offsets.velocity_systematics import order_residual, order_screen
    print('\n[noise] control noise retrained on each half-year with the run\'s noise() procedure')
    ids = h['source_id'][:]; tr = Transformer.from_crs(4326, 3031, always_xy=True)
    support = np.load(REPO/'.runtime/velocity-recovery-larsen/2017-order-set/fitted-support.npz')
    for year, split, half in ((0, '2017-07-01', 'H1'), (0, '2017-07-01', 'H2'), (1, '2018-07-01', 'H1'), (1, '2018-07-01', 'H2')):
        R, S, Q, G = [], [], [], []
        for rid in ('control_1', 'control_2'):
            index = np.unique(np.searchsorted(ids, support[rid + '__source_id'])); f = get_field(h, body, index)
            take = f['year_index'] == year
            local = {k: f[k][:, take] for k in ('early', 'late', 'data', 'valid', 'peak')}
            local.update(order=np.broadcast_to(f['order'][take], local['valid'].shape),
                         calendar_early=f['calendar_early'][take], calendar_late=f['calendar_late'][take])
            sel = (local['calendar_late'] < float(utc_days(split))) if half == 'H1' else (local['calendar_early'] >= float(utc_days(split)))
            r = order_residual(local, allowed=sel); good = order_screen(local, allowed=sel)
            R.extend(r[good]); Q.extend(local['peak'][good]); S.extend(h['snr'][index][:, take][good])
            xx, yy = tr.transform(h['longitude'][index], h['latitude'][index])
            lab = np.array([f'{rid}:{int(np.floor(a/3000))}:{int(np.floor(b/3000))}' for a, b in zip(xx, yy)])
            G.extend(np.broadcast_to(lab[:, None], good.shape)[good])
        m = calibrate_noise(np.array(R), np.array(S), np.array(Q), np.array(G))
        print(f'  {2017 + year} {half}: sd range/azimuth {np.sqrt(np.diag(m["covariance_ra_m2"])).round(3)} m, n={m["samples"]}')
    print(f'  used by the run: {np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).round(3)} (2017 = frozen H1, 2018 = 2018 H1)')

if 'halves' in parts:
    print('\n[halves] mean along-flow Msf vector (m/day @ deg) and median |A|; direct-azimuth fits per subset')
    base = est & standard & (speed >= .3) & (proj >= .5) & (c['year_direction_change_deg'][:] <= 20)
    regions = {'west (x<-2150, y>1010)': base & (x < -2150) & (y > 1010), 'central (-2150..-2120)': base & (x >= -2150) & (x < -2120),
               'east (x>-2100)': base & (x > -2100), 'bedrock speed<0.03': bedrock}
    ce, cl, yr = h['calendar_early'][:], h['calendar_late'][:], h['year_index'][:]
    subsets = {'2017H1': (yr == 0) & (cl < utc_days('2017-07-01')), '2017H2': (yr == 0) & (ce >= utc_days('2017-07-01')),
               '2018H1': (yr == 1) & (cl < utc_days('2018-07-01')), '2018H2': (yr == 1) & (ce >= utc_days('2018-07-01')),
               '2017': yr == 0, '2018': yr == 1}
    rng = np.random.default_rng(3)
    for name, s in regions.items():
        idx = np.sort(rng.choice(np.flatnonzero(s), min(3000, s.sum()), replace=False))
        f = get_field(h, body, idx); good = screen_years(f)[0]
        hb = np.nan_to_num(harmonic(f['early'], f['late'])); dt = np.nan_to_num(f['late'] - f['early'])
        az = np.nan_to_num(f['data'][..., 1]); cells = []
        for sub, t in subsets.items():
            w = (good & t[None]).astype(float)
            a = np.concatenate([dt[..., None], np.broadcast_to(f['order'][None, :, None]*1., dt.shape + (1,)), hb], -1)
            n = np.einsum('st,sti,stj->sij', w, a, a); r = np.einsum('st,sti,st->si', w, a, az)
            ok = (w.sum(1) >= 12) & (np.linalg.det(n) > 1e-6); co = np.linalg.solve(n[ok], r[ok][..., None])[..., 0]
            sign = 1. if name.startswith('bedrock') else P[idx[ok], 0 if sub.startswith('2017') else 1]
            z = (co[:, 2] + 1j*co[:, 3])/sign
            cells.append(f'{sub} {mean_vec(z)} med {np.median(abs(z)):.3f}')
        print(f'  {name} (n={len(idx)})\n    ' + '\n    '.join(cells))

if 'gps' in parts:
    print('\n[gps] King et al. (2011, GRL) Table 1 along-flow velocity; amplitudes read as m/day (table labels mm/hr,'
          ' but only mm/day matches their Figs. 2-3); Greenwich lags, UTC, no nodal corrections')
    jd0 = 2451544.5   # tau = 0 at 2000-01-01 00:00 UTC

    def astro(tau):
        t = (jd0 + tau - 2451545.0)/36525
        return (np.radians(218.3164477 + 481267.88123421*t), np.radians(280.46646 + 36000.76983*t),
                np.radians(180 + 360*tau))   # s, h, mean solar hour angle at Greenwich

    sites = {'LAR1': (-67.0133, 298.4876, (0.097, 6), (0.350, 185)), 'LAR2': (-68.0004, 295.7058, (0.089, 14), (0.276, 167)),
             'LAR3': (-68.5008, 297.9982, (0.087, 14), (0.330, 176))}   # lat, lon (JGR Table 1), Msf, M2 (GRL Table 1)
    tau = np.linspace(6210, 6940, 200000); s, hh, _ = astro(tau)
    basis = np.stack([np.cos(2*np.pi*tau/PERIOD), np.sin(2*np.pi*tau/PERIOD)], -1)
    src = np.flatnonzero(est & standard & (speed > .5))[0]
    e, l = h['early'][src], h['late'][src]; m = np.isfinite(e) & np.isfinite(l); e, l = e[m], l[m]
    order, yr = h['order'][:][m], h['year_index'][:][m]
    om = 2*np.pi*(2 - 2*(13.17639648 - 0.98564736)/360)   # M2 rad/day
    tr = Transformer.from_crs(4326, 3031, always_xy=True); cs = c['residual_vertical/cs'][:]
    X, Y = h['x'][:][h['col'][:]], h['y'][:][h['row'][:]]
    for site, (lat, lon, (am, gm), (a2, g2)) in sites.items():
        cm = np.linalg.lstsq(basis, am*np.cos(2*s - 2*hh - np.radians(gm)), rcond=None)[0]; zm = cm[0] + 1j*cm[1]

        def disp(t):
            s_, h_, T_ = astro(t)
            return a2/om*np.sin(2*T_ - 2*s_ + 2*h_ - np.radians(g2))

        a = np.column_stack([l - e, *(order*(yr == j) for j in (0, 1)), np.nan_to_num(harmonic(e[None], l[None]))[0]])
        ca = np.linalg.lstsq(a, disp(l) - disp(e), rcond=None)[0]; za = ca[-2] + 1j*ca[-1]
        px, py = tr.transform(lon - 360, lat); dist = np.hypot(X - px, Y - py)
        near = est & standard & (dist < 10e3) & (speed > .3) & (proj >= .5)
        ph = lambda z: np.degrees(np.angle(z)) % 360
        line = (f'  {site} ({px/1e3:.1f}, {py/1e3:.1f}) km: GPS Msf {abs(zm):.3f} @ {ph(zm):.1f}; M2 displacement {100*a2/om:.1f} cm '
                f'-> alias {abs(za):.4f} @ {ph(za):.1f}; sum {abs(zm + za):.3f} @ {ph(zm + za):.1f}')
        if near.any():
            zz = cs[near, 0] + 1j*cs[near, 1]
            line += (f'\n    map within 10 km: n={near.sum()}, mean {abs(zz.mean()):.3f} @ {ph(zz.mean()):.1f}, '
                     f'speed {np.median(speed[near]):.2f} m/day, nearest {dist[near].min()/1e3:.1f} km')
        else:
            line += f'\n    no estimable moving sources within 10 km (nearest source of any kind {dist.min()/1e3:.1f} km)'
        print(line)
