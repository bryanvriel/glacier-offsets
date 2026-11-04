import io, contextlib, runpy, numpy as np
import os
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean_msf.py')
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(OLD, run_name='x')
stash, harmonic, early, late, dt, f = g['stash'], g['harmonic'], g['early'], g['late'], g['dt'], g['f']
alt = np.array([1. if (x.decode() if isinstance(x, bytes) else x) == 'S1A' else -1. for x in f['earlier_spacecraft'][:]])
periods = [12.8, 13.2, 13.66, 14.0, 14.19, 14.765294, 15.3, 15.95, 16.7, 18.0, 20.0, 23.0, 27.0, 33.0, 45.0]
print('azimuth-only, secular + harmonic(P) + spacecraft-order; white-noise Wald (2 dof) by period')
print(f'{"":10s}' + ''.join(f'{p:7.2f}' for p in periods))
U1 = None
for rid, s in stash.items():
    rg, az, sr, sa = s['rgm'], s['azm'], s['srm'], s['sam']
    G = np.stack((sr, sa)); v = np.nanmedian(np.linalg.solve(G, np.stack((rg, az)))/dt, 1)
    if rid == 'target_1': U1 = v/np.linalg.norm(v)
    u = U1 if s['role'] == 'control' else v/np.linalg.norm(v)
    good = np.isfinite(az); row = []
    for P in periods:
        H = harmonic(early, late, P); gu = sa @ u
        A = np.stack([sa[0]*dt, sa[1]*dt, gu*H[:, 0], gu*H[:, 1], alt], -1)[good]
        m, *_ = np.linalg.lstsq(A, az[good], rcond=None); r = az[good]-A@m
        cov = np.linalg.inv(A.T@A)*(r@r/(good.sum()-5))
        row.append(m[2:4]@np.linalg.solve(cov[2:4, 2:4], m[2:4]))
    print(f'{rid:10s}' + ''.join(f'{w:7.1f}' for w in row))
