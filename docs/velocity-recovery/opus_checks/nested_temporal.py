"""Nested check: temporal-only vs paired (time x space) rescreened bootstrap on null synthetic datasets.

Generator = recovery() null case as implemented (H2 control bank, donor blocks/sources with replacement).
Compares median bootstrap covariance trace with the empirical null covariance in null-recovery.npz.
Run in the `ice` environment from any directory:
  python docs/velocity-recovery/opus_checks/nested_temporal.py
"""
import json, multiprocessing as mp, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO/'src'), str(REPO/'tools')]
import numpy as np
import run_order_set_pilot as R
from glacier_offsets.velocity_basis import harmonic, utc_days
from glacier_offsets.velocity_quality import (temporal_counts, cluster_weights, moving_indices, quiet_median,
                                              paired_field_draw)
from glacier_offsets.velocity_systematics import (OrderRegionalDesign, OrderScreenCache, order_screen,
                                                  robust_background, calendar_times)

NDATA, NBOOT = 20, 150
root = REPO/'.runtime/velocity-recovery-larsen'; out = root/'2017-order-set'
fields = R.load(root, out); cov = R.calibrate(fields, out)
designs = {f['id']: OrderRegionalDesign(f, cov) for f in fields}
fits = {f['id']: designs[f['id']].fit(f['clean'], details=True) for f in fields}
nt = fields[0]['valid'].shape[1]; split = utc_days('2017-07-01')

banks = []
for f in fields:
    if f['role'] != 'control': continue
    half = f['bank_half']; first = calendar_times(f)[1] < split
    good = order_screen(f, allowed=first) & first[None]
    fitted = designs[f['id']].fit(good, details=True)
    so = fitted['prediction'][..., 0] - np.einsum('stj,j->st', harmonic(f['early'], f['late']), fitted['cs'])
    r = f['bank_residual'].copy(); r[..., 1] = f['data'][..., 1] - so
    center = quiet_median(np.where(f['valid'][:, half, None], r[:, half], np.nan), axis=1)
    banks.append(({**f, 'residual': r - center[:, None]}, half))
truth = {}
for f in fields:
    tp, oldbias = robust_background(f, np.ones(nt, bool)); tp[..., 1] -= f['order']*oldbias[:, None]
    bursts = designs[f['id']].bursts; oc = fits[f['id']]['order_m']
    tp[..., 1] += f['order']*np.array([oc[np.flatnonzero(bursts == x)[0]] for x in f['burst']])[:, None]
    truth[f['id']] = tp

# Pre-generate datasets in the parent so workers only bootstrap.
rng = np.random.default_rng([4242, 1]); datasets = []
for b in range(NDATA):
    bank, half = banks[int(rng.integers(len(banks)))]
    ti = np.resize(np.r_[half[moving_indices(rng, len(half), 5)], half[moving_indices(rng, len(half), 5)]], nt)
    for fi, f in enumerate(fields):
        base = paired_field_draw(rng, bank, f, 'paired_quality', time_index=ti)
        datasets.append((b, fi, {**base, 'data': base['data'] + truth[f['id']]}))


def job(arg):
    b, fi, syn = arg
    f = fields[fi]; d = OrderRegionalDesign(syn, cov)
    good0 = order_screen(syn); cs = d.fit(good0)['cs']; cache = OrderScreenCache(syn)
    r = np.random.default_rng([99, b, fi]); out_ = {}
    for scheme in ('time', 'both'):
        vals = []
        for _ in range(NBOOT):
            tc = temporal_counts(r, *calendar_times(fields[0]))
            sw = cluster_weights(syn['group'], r) if scheme == 'both' else None
            try: vals.append(d.fit(cache.mask_for(tc), temporal=tc, spatial=sw)['cs'])
            except (ValueError, np.linalg.LinAlgError): pass
        C = np.cov((np.array(vals) - cs).T)
        out_[scheme] = (np.trace(C), float(cs @ np.linalg.solve(C, cs)), float(np.quantile(
            np.einsum('ni,ij,nj->n', np.array(vals) - cs, np.linalg.inv(C), np.array(vals) - cs), .95)))
    return f['id'], out_


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('fork')) as pool:
        res = list(pool.map(job, datasets, chunksize=2))
    with np.load(out/'null-recovery.npz') as h:
        emp = {f['id']: h[f['id'] + '__primary_azimuth'][:, :2] for f in fields}
    print(f'{NDATA} null datasets x {NBOOT} rescreened resamples; ratios = median bootstrap trace / empirical trace (200 fields)')
    print(f'{"region":10s} {"time/emp":>9s} {"both/emp":>9s} {"rej time":>9s} {"rej both":>9s}')
    for f in fields:
        k = f['id']; e = emp[k]; te = np.trace(e.T @ e/len(e))
        rr = [o for rid, o in res if rid == k]
        rt = np.median([o['time'][0] for o in rr])/te; rb = np.median([o['both'][0] for o in rr])/te
        jt = np.mean([o['time'][1] > o['time'][2] for o in rr]); jb = np.mean([o['both'][1] > o['both'][2] for o in rr])
        print(f'{k:10s} {rt:9.2f} {rb:9.2f} {jt:9.2f} {jb:9.2f}')
