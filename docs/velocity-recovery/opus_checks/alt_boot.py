"""Azimuth-only along-flow Msf with/without spacecraft-order term; white-noise and 30-day block-bootstrap sigma."""
import io, contextlib, runpy, numpy as np
import os
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean_msf.py')
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(OLD, run_name='x')
stash, harmonic, early, late, dt, f = g['stash'], g['harmonic'], g['early'], g['late'], g['dt'], g['f']
alt = np.array([1. if (x.decode() if isinstance(x, bytes) else x) == 'S1A' else -1. for x in f['earlier_spacecraft'][:]])
H = harmonic(early, late)
rng = np.random.default_rng(0)
blocks = np.floor((early-early.min())/30).astype(int)
U1 = None
def design(sa, u, with_alt):
    gu = sa @ u
    cols = [sa[0]*dt, sa[1]*dt, gu*H[:, 0], gu*H[:, 1]] + ([alt] if with_alt else [])
    return np.stack(cols, -1)
def solve(A, d, rows):
    m, *_ = np.linalg.lstsq(A[rows], d[rows], rcond=None)
    return m
for rid, s in stash.items():
    rg, az, sr, sa = s['rgm'], s['azm'], s['srm'], s['sam']
    G = np.stack((sr, sa))
    v = np.nanmedian(np.linalg.solve(G, np.stack((rg, az)))/dt, 1)
    if rid == 'target_1':
        U1 = v/np.linalg.norm(v)
    u = U1 if s['role'] == 'control' else v/np.linalg.norm(v)
    good = np.where(np.isfinite(az))[0]
    out = []
    for with_alt in (False, True):
        A = design(sa, u, with_alt)
        m = solve(A, az, good)
        r = az[good]-A[good]@m
        cov = np.linalg.inv(A[good].T@A[good])*(r@r/(len(good)-A.shape[1]))
        amp = np.hypot(m[2], m[3]); J = m[2:4]/amp
        sw = np.sqrt(J@cov[2:4, 2:4]@J)
        ub = np.unique(blocks[good]); draws = []
        for _ in range(2000):
            pick = rng.choice(ub, len(ub))
            rows = np.concatenate([good[blocks[good] == b] for b in pick])
            draws.append(solve(A, az, rows)[2:4])
        draws = np.array(draws); cb = np.cov(draws.T)
        sb = np.sqrt(J@cb@J)
        wald = m[2:4]@np.linalg.solve(cb, m[2:4])
        out.append(f'{"alt " if with_alt else "base"}: A={amp:.3f} ph={np.degrees(np.arctan2(m[3], m[2])):6.1f} sig_white={sw:.3f} sig_block={sb:.3f} Wald_block={wald:5.1f}')
    print(f'{rid:10s} ' + ' | '.join(out))
