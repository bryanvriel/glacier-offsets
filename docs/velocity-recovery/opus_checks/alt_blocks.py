import os, io, contextlib, runpy, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alt_boot.py'), run_name='x')
stash, harmonic, early, late, dt, alt, design, solve = (g[k] for k in ('stash','harmonic','early','late','dt','alt','design','solve'))
rng = np.random.default_rng(1)
print('azimuth-only along-flow Msf: block-bootstrap sigma (m/d) by block length; base | with spacecraft-order term')
U1 = None
for rid, s in stash.items():
    rg, az, sr, sa = s['rgm'], s['azm'], s['srm'], s['sam']
    G = np.stack((sr, sa)); v = np.nanmedian(np.linalg.solve(G, np.stack((rg, az)))/dt, 1)
    if rid == 'target_1': U1 = v/np.linalg.norm(v)
    u = U1 if s['role'] == 'control' else v/np.linalg.norm(v)
    good = np.where(np.isfinite(az))[0]; txt = []
    for with_alt in (False, True):
        A = design(sa, u, with_alt); m = solve(A, az, good); J = m[2:4]/np.hypot(*m[2:4])
        for L in (15, 30, 60):
            blocks = np.floor((early-early.min())/L).astype(int); ub = np.unique(blocks[good])
            d = [solve(A, az, np.concatenate([good[blocks[good]==b] for b in rng.choice(ub, len(ub))]))[2:4] for _ in range(1500)]
            txt.append(f'{np.sqrt(J@np.cov(np.array(d).T)@J):.3f}')
    print(f'{rid:10s} base 15/30/60d: {" ".join(txt[:3])}   | alt 15/30/60d: {" ".join(txt[3:])}')
