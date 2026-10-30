"""Checks on the 2017 order-set experiment, using its own estimator, screen and generator.

Parts (default: all):
  boot       observed-data bootstrap: paired time x space vs temporal-only vs spatial-only
  halves     separate Jan-Jun / Jul-Dec fits, order coefficients and a difference test
  generator  null noise generator as implemented, and variants (no replacement, parity, Jan-Jun bank)

Run in the `ice` environment from any directory:
  python docs/velocity-recovery/opus_checks/order_set_checks.py [part ...]
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools')]
import numpy as np
import run_order_set_pilot as R
from glacier_offsets.velocity_basis import harmonic, utc_days
from glacier_offsets.velocity_quality import temporal_counts, cluster_weights, moving_indices, quiet_median
from glacier_offsets.velocity_systematics import (OrderRegionalDesign, OrderScreenCache, order_screen,
                                                  robust_background, calendar_times)

root = REPO/'.runtime/velocity-recovery-larsen'; out = root/'2017-order-set'
parts = sys.argv[1:] or ['boot', 'halves', 'generator']
fields = R.load(root, out); cov = R.calibrate(fields, out)
obs = {r['region']: r for r in json.load(open(out/'observed.json'))['rows'] if r['variant'] == 'primary_azimuth'}
designs = {f['id']: OrderRegionalDesign(f, cov) for f in fields}
fits = {f['id']: designs[f['id']].fit(f['clean'], details=True) for f in fields}
for f in fields:
    assert np.allclose(fits[f['id']]['cs'], obs[f['id']]['cs_m_per_day'], atol=1e-10), f['id']
print('Reproduced primary_azimuth coefficients for all regions.')


def sig_amp(cs, C):
    g = cs/np.linalg.norm(cs)
    return float(np.sqrt(g @ C @ g))


if 'boot' in parts:
    # Paired (time x space) vs temporal-only vs spatial-only resampling, rescreened as in observed().
    ndraw = 400
    print(f'\n[boot] {ndraw} draws per scheme; trace ratios relative to temporal-only')
    caches = {f['id']: OrderScreenCache(f) for f in fields}
    ones = np.ones(len(calendar_times(fields[0])[0]))
    good1 = {f['id']: caches[f['id']].mask_for(ones) for f in fields}
    res = {}
    for i, scheme in enumerate(('both', 'time', 'space')):
        rng = np.random.default_rng([2026, i])
        vals = {f['id']: [] for f in fields}
        for b in range(ndraw):
            tc = temporal_counts(rng, *calendar_times(fields[0])) if scheme != 'space' else ones
            for f in fields:
                sw = cluster_weights(f['group'], rng) if scheme != 'time' else None
                good = caches[f['id']].mask_for(tc) if scheme != 'space' else good1[f['id']]
                try: vals[f['id']].append(designs[f['id']].fit(good, temporal=tc, spatial=sw)['cs'])
                except (ValueError, np.linalg.LinAlgError): pass
        res[scheme] = {k: np.cov((np.array(v) - fits[k]['cs']).T) for k, v in vals.items()}
    print(f'{"region":10s} {"sigA both":>9s} {"time":>7s} {"space":>7s} | {"tr both/time":>12s} {"space/time":>10s} {"(time+space)/both":>17s}')
    for f in fields:
        k = f['id']; cs = fits[k]['cs']; C = {s: res[s][k] for s in res}
        tb, tt, ts = (np.trace(C[s]) for s in ('both', 'time', 'space'))
        print(f'{k:10s} {sig_amp(cs, C["both"]):9.4f} {sig_amp(cs, C["time"]):7.4f} {sig_amp(cs, C["space"]):7.4f} | '
              f'{tb/tt:12.2f} {ts/tt:10.2f} {(tt+ts)/tb:17.2f}   (report sigma {obs[k]["sigma_amplitude"]:.4f})')

if 'halves' in parts:
    # Fit H1 (Jan-Jun) and H2 (Jul-Dec) separately; temporal-only bootstrap within each half.
    print('\n[halves] separate half-year fits (primary azimuth); sigma from 300 within-half temporal block draws')
    e, l = calendar_times(fields[0]); split = utc_days('2017-07-01')
    halves = {'H1': l < split, 'H2': e >= split}
    print(f'{"region":10s} {"half":4s} {"A":>7s} {"sigA":>7s} {"phase":>7s} {"order_m":>18s}')
    for f in fields:
        k = f['id']; d = designs[k]; rows = {}
        for h, m in halves.items():
            fit = d.fit(f['clean'] & m[None], details=True)
            rng = np.random.default_rng([7, len(k), int(h[1])]); vals = []
            for b in range(300):
                tc = np.zeros(len(e)); tc[m] = temporal_counts(rng, e[m], l[m])
                try: vals.append(d.fit(f['clean'] & m[None], temporal=tc)['cs'])
                except (ValueError, np.linalg.LinAlgError): pass
            C = np.cov((np.array(vals) - fit['cs']).T); cs = fit['cs']
            rows[h] = (cs, C)
            print(f'{k:10s} {h:4s} {np.linalg.norm(cs):7.4f} {sig_amp(cs, C):7.4f} {np.degrees(np.arctan2(cs[1], cs[0])):7.1f} '
                  f'{np.array2string(np.round(fit["order_m"], 3)):>18s}')
        dcs = rows['H1'][0] - rows['H2'][0]; Cd = rows['H1'][1] + rows['H2'][1]
        print(f'{"":10s} H1-H2 difference Wald (2 dof) = {dcs @ np.linalg.solve(Cd, dcs):.2f}')

if 'generator' in parts:
    # Null generator as in recovery(), and variants: (a) as implemented; (b) donor blocks/sources without
    # replacement; (c) target-parity-preserving time blocks; (d) both; (e) H1 bank (H2-trained), as implemented.
    ndraw = 200
    split = utc_days('2017-07-01')

    def bank_for(f, bank_is_h2=True):
        e, l = calendar_times(f)
        first = l < split; last = e >= split
        train, test = (first, last) if bank_is_h2 else (last, first)
        acq = np.unique(np.floor(np.r_[e[test], l[test]]))
        train = train & ~np.isin(np.floor(e), acq) & ~np.isin(np.floor(l), acq)
        pred, _ = robust_background(f, train); r = f['data'] - pred
        good = order_screen(f, allowed=train) & train[None]
        fitted = designs[f['id']].fit(good, details=True)
        so = fitted['prediction'][..., 0] - np.einsum('stj,j->st', harmonic(f['early'], f['late']), fitted['cs'])
        r = r.copy(); r[..., 1] = f['data'][..., 1] - so
        half = np.flatnonzero(test)
        center = quiet_median(np.where(f['valid'][:, half, None], r[:, half], np.nan), axis=1)
        return {**f, 'residual': r - center[:, None]}, half

    def draw(rng, donor, target, ti, norep):
        # Same as paired_field_draw(..., 'paired_quality'), optionally without replacement.
        n, t = target['valid'].shape
        dg = np.unique(donor['group']); tg = np.unique(target['group'])
        if norep:
            reps = int(np.ceil(len(tg)/len(dg)))
            chosen = np.concatenate([rng.permutation(dg) for _ in range(reps)])[:len(tg)]
        else:
            chosen = [rng.choice(dg) for _ in tg]
        mapping = np.empty(n, int)
        for g, c in zip(tg, chosen):
            members = np.flatnonzero(target['group'] == g); cand = np.flatnonzero(donor['group'] == c)
            if norep:
                reps = int(np.ceil(len(members)/len(cand)))
                mapping[members] = np.concatenate([rng.permutation(cand) for _ in range(reps)])[:len(members)]
            else:
                mapping[members] = rng.choice(cand, size=len(members), replace=True)
        ix = np.ix_(mapping, ti); residual = donor['residual'][ix].copy()
        valid = target['valid'] & donor['valid'][ix] & np.isfinite(residual).all(-1)
        return {**target, 'valid': valid, 'data': residual, 'snr': donor['snr'][ix].copy(), 'peak': donor['peak'][ix].copy()}

    def pair_sign(f):
        o = np.where(f['order'] == 0, np.nan, np.sign(f['order']))
        return np.nanmedian(o, axis=0)

    nt = fields[0]['valid'].shape[1]
    qt = pair_sign(fields[0])

    def time_index(rng, half, qd, parity):
        if not parity:
            return np.resize(np.r_[half[moving_indices(rng, len(half), 5)], half[moving_indices(rng, len(half), 5)]], nt)
        ti = []
        for s0 in range(0, nt, 5):
            ok = [p for p in range(len(half) - 4) if qd[half[p]] == qt[s0]]
            p = rng.choice(ok); ti.extend(half[p:p + 5])
        return np.array(ti[:nt])

    truth = {}
    for f in fields:
        tp, oldbias = robust_background(f, np.ones(nt, bool))
        tp[..., 1] -= f['order']*oldbias[:, None]
        bursts = designs[f['id']].bursts; oc = fits[f['id']]['order_m']
        tp[..., 1] += f['order']*np.array([oc[np.flatnonzero(bursts == x)[0]] for x in f['burst']])[:, None]
        truth[f['id']] = tp
    controls = [f for f in fields if f['role'] == 'control']
    variants = {'a_as_run': (True, False, False), 'b_norep': (True, True, False), 'c_parity': (True, False, True),
                'd_norep_parity': (True, True, True), 'e_H1bank': (False, False, False)}
    print(f'\n[generator] null, {ndraw} draws per variant; null second-moment trace / observed trace (report: '
          '1.04 0.88 1.41 2.70 1.75 1.05)')
    table = {}
    for vname, (h2, norep, parity) in variants.items():
        t0 = time.time()
        banks = [bank_for(f, h2) for f in controls]
        qds = [pair_sign(b[0]) for b in banks]
        rng = np.random.default_rng([12345, 5, len(vname)])
        vals = {f['id']: [] for f in fields}; ret = {f['id']: [] for f in fields}
        for b in range(ndraw):
            j = int(rng.integers(len(banks))); bank, half = banks[j]
            ti = time_index(rng, half, qds[j], parity)
            for f in fields:
                base = draw(rng, bank, f, ti, norep)
                syn = {**base, 'data': base['data'] + truth[f['id']]}
                good = order_screen(syn)
                ret[f['id']].append(good.sum()/max(f['valid'].sum(), 1))
                try: vals[f['id']].append(designs[f['id']].fit(good, data=syn['data'])['cs'])
                except (ValueError, np.linalg.LinAlgError): pass
        # Second moment about zero: the true null coefficient is zero.
        table[vname] = {k: (np.trace(np.array(v).T @ np.array(v)/len(v)), np.median(ret[k])) for k, v in vals.items()}
        print(f'  {vname} done in {time.time()-t0:.0f}s')
    print(f'{"region":10s} ' + ' '.join(f'{v:>15s}' for v in variants) + '   (retention a / observed clean frac)')
    for f in fields:
        k = f['id']; to = np.trace(np.array(obs[k]['covariance']))
        print(f'{k:10s} ' + ' '.join(f'{table[v][k][0]/to:15.2f}' for v in variants)
              + f'   {table["a_as_run"][k][1]:.2f} / {f["clean"].sum()/f["valid"].sum():.2f} (H1bank {table["e_H1bank"][k][1]:.2f})')
