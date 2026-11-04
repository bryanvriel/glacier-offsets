"""Azimuth residual diagnostics on regional medians: S1A/S1B alternation, tide correlation, common mode."""
import io, contextlib, runpy, numpy as np, h5py
import os
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean_msf.py')
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(OLD, run_name='x')
stash, harmonic, early, late, dt, f, regions = g['stash'], g['harmonic'], g['early'], g['late'], g['dt'], g['f'], g['regions']
alt = np.array([1. if (x.decode() if isinstance(x, bytes) else x) == 'S1A' else -1. for x in f['earlier_spacecraft'][:]])
x, y = f['x'][:], f['y'][:]
def region_mean(var, reg):
    w, s, e, n = reg['bounds']
    ci = np.where((x >= w) & (x <= e))[0]; ri = np.where((y >= s) & (y <= n))[0]
    return np.nanmedian(f[var][:, ri.min():ri.max()+1, ci.min():ci.max()+1].reshape(60, -1), 1)
def lsq(cols, d):
    good = np.isfinite(d) & np.all(np.isfinite(cols), 1)
    A, b = cols[good], d[good]
    m, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ m
    s2 = r @ r/(len(b)-A.shape[1])
    return m, np.sqrt(np.diag(np.linalg.inv(A.T @ A))*s2), r, good
H = harmonic(early, late)
resid = {}
print('Azimuth channel, regional median series. Columns: secular(dt), Msf C/S (velocity-integral), alternation sign')
for reg in regions:
    rid = reg['id']; s = stash[rid]
    az = s['azm']
    dz = region_mean('tidal_vertical_displacement_m', reg)   # CATS later-minus-earlier vertical (m)
    base = np.stack([dt, H[:, 0], H[:, 1]], -1)
    m0, e0, r0, g0 = lsq(base, az)
    m1, e1, r1, g1 = lsq(np.column_stack([base, alt]), az)
    if reg['role'] == 'target':
        m2, e2, r2, g2 = lsq(np.column_stack([base, alt, np.nan_to_num(dz)]), az)
    else:
        m2, e2 = np.r_[m1, np.nan], np.r_[e1, np.nan]
    a0 = np.hypot(*m0[1:3]); a1 = np.hypot(*m1[1:3])
    print(f'{rid:10s} alt={m1[3]:+.3f}±{e1[3]:.3f} m  rms {np.std(r0):.3f}->{np.std(r1):.3f}  |MsfDispl coef| {a0:.3f}->{a1:.3f} (m/d along az)'
          f'  tide-dz coef={m2[4]:+.3f}±{e2[4]:.3f}')
    full = np.full(60, np.nan); full[g1] = r1; resid[rid] = full
print('\nCorrelation of azimuth residuals (after secular+Msf+alt) between regions:')
ids = list(resid)
for i in ids:
    print(f'{i:10s}', ' '.join(f'{np.corrcoef(*[v[np.isfinite(resid[i])&np.isfinite(resid[j])] for v in (resid[i],resid[j])])[0,1]:+.2f}' for j in ids))

print('\nSame alternation fit on the RAW package (before spacecraft/swath correction), regional medians of clean-mask observations:')
raw = h5py.File('/Users/briel/data/insar/glacier_offsets/.runtime/velocity-recovery-larsen/frozen-2017/displacements_raw.nc', 'r')
print('raw vars:', [k for k in raw.keys() if 'azimuth' in k][:6])
