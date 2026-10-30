import os, numpy as np, runpy, io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'clean_msf.py'), run_name='x')
stash, harmonic, early, late, dt = g['stash'], g['harmonic'], g['early'], g['late'], g['dt']
U1 = g['U1']
def chan_fit(d, gvec, u, sig, tide=None):
    """Secular EN fixed from both channels; Msf from one channel only."""
    good = np.isfinite(d)
    H = harmonic(early, late)
    gu = gvec @ u
    cols = [gvec[0]*dt, gvec[1]*dt, gu*H[:, 0], gu*H[:, 1]]
    A = np.stack(cols, -1)[good]
    m, *_ = np.linalg.lstsq(A, d[good], rcond=None)
    r = d[good]-A@m
    cov = np.linalg.pinv(A.T@A)*(r@r/(good.sum()-4))
    C, S = m[2], m[3]; a = np.hypot(C, S); J = np.array([C, S])/a
    return a, np.sqrt(J@cov[2:4, 2:4]@J), np.degrees(np.arctan2(S, C))
for rid, s in stash.items():
    rg, az, sr, sa = s['rgm'], s['azm'], s['srm'], s['sam']
    G = np.stack((sr, sa))
    v = np.nanmedian(np.linalg.solve(G, np.stack((rg, az)))/dt, 1)
    u = U1 if s['role'] == 'control' else v/np.linalg.norm(v)
    ar = chan_fit(rg, sr, u, None); aa = chan_fit(az, sa, u, None)
    print(f'{rid:10s} |s_r.u|={abs(sr@u):.2f} |s_a.u|={abs(sa@u):.2f}  range-only: {ar[0]:.3f}±{ar[1]:.3f} ph={ar[2]:7.1f}   azimuth-only: {aa[0]:.3f}±{aa[1]:.3f} ph={aa[2]:7.1f}')
