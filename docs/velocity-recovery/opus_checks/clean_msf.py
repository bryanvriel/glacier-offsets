"""Independent check: blunder rate, clean noise, and Msf precision with/without spatial averaging."""
import json, sys
import h5py, numpy as np

ROOT = '/Users/briel/data/insar/glacier_offsets/.runtime/velocity-recovery-larsen'
NC = ROOT + '/frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc'
P = 14.765294
f = h5py.File(NC, 'r')
x = f['x'][:]; y = f['y'][:]
early = f['earlier_time'][:]/86400. - 10957.0  # days since 2000-01-01
late = f['later_time'][:]/86400. - 10957.0
dt = late - early
regions = json.load(open(ROOT + '/regions.reviewed.json'))['regions']


def harmonic(e, l, period=P):
    w = 2*np.pi/period; mid = (e+l)/2; d = l-e
    fac = d*np.sinc(d/period)
    return np.stack((fac*np.cos(w*mid), fac*np.sin(w*mid)), -1)


def region_stack(reg):
    w, s, e, n = reg['bounds']
    ci = np.where((x >= w) & (x <= e))[0]; ri = np.where((y >= s) & (y <= n))[0]
    sl = (slice(None), slice(ri.min(), ri.max()+1), slice(ci.min(), ci.max()+1))
    sl2 = sl[1:]
    out = {k: f[k][sl].reshape(60, -1) for k in ('range_displacement_m', 'azimuth_displacement_m', 'snr',
                                               'correlation_peak', 'horizontal_valid_mask', 'tidal_range_bias_m')}
    rs = f['range_sensitivity_enu'][(slice(None),)+sl2].reshape(3, -1)
    az = f['azimuth_sensitivity_enu'][(slice(None),)+sl2].reshape(3, -1)
    src = f['source_observation_index'][sl2].ravel()
    ok = src >= 0
    _, first = np.unique(np.where(ok, src, -1), return_index=True)
    first = np.array([i for i in first if ok[i]])
    return {k: v[:, first] for k, v in out.items()}, rs[:2, first], az[:2, first]


def robust_secular(rg, az, sr, sa):
    """Per source: median-based secular EN, then radar-space residuals."""
    G = np.stack((sr, sa))  # 2x2 radar<-EN
    en = np.linalg.solve(G, np.stack((rg, az)))  # 2 x npairs
    v = np.nanmedian(en/dt, axis=1)
    pred = G @ (v[:, None]*dt[None])
    return v, rg-pred[0], az-pred[1]


def fit(rg, az, sr, sa, sig, along=True, tide=None, u=None, period=P):
    """Weighted LS in radar space: secular EN + Msf (+ optional tide-scale)."""
    good = np.isfinite(rg) & np.isfinite(az)
    G = np.stack((sr, sa))
    if u is None:
        v, _, _ = robust_secular(rg, az, sr, sa)
        u = v/np.linalg.norm(v)
    H = harmonic(early, late, period)
    cols = []
    for ch in range(2):
        g = G[ch]
        c = [g[0]*dt, g[1]*dt]
        if along:
            gu = g @ u
            c += [gu*H[:, 0], gu*H[:, 1]]
        else:
            c += [g[0]*H[:, 0], g[0]*H[:, 1], g[1]*H[:, 0], g[1]*H[:, 1]]
        if tide is not None:
            c += [tide if ch == 0 else 0*tide]
        cols.append(np.stack(c, -1))
    A = np.concatenate([cols[0][good], cols[1][good]])
    d = np.concatenate([rg[good], az[good]])
    wts = np.concatenate([np.full(good.sum(), 1/sig[0]), np.full(good.sum(), 1/sig[1])])
    Aw, dw = A*wts[:, None], d*wts
    m, *_ = np.linalg.lstsq(Aw, dw, rcond=None)
    r = dw - Aw @ m
    chi2 = r @ r/(len(dw)-len(m))
    cov = np.linalg.inv(Aw.T @ Aw)*chi2   # a posteriori scaling
    res = d - A @ m
    n = good.sum()
    return m, cov, chi2, res[:n], res[n:], n, u


def amp_summary(m, cov):
    C, S = m[2], m[3]
    a = np.hypot(C, S)
    J = np.array([C, S])/max(a, 1e-12)
    sa = np.sqrt(J @ cov[2:4, 2:4] @ J)
    return a, sa, np.degrees(np.arctan2(S, C))


print(f'{"region":10s} {"nsrc":>5s} {"blunder%":>8s} {"pk<.06 of bl":>12s} {"clean rg sd":>11s} {"clean az sd":>11s}')
stash = {}
for reg in regions:
    d, sr, sa = region_stack(reg)
    rg, az = d['range_displacement_m'], d['azimuth_displacement_m']
    valid = (d['horizontal_valid_mask'] == 1) & np.isfinite(rg) & np.isfinite(az)
    rg = np.where(valid, rg, np.nan); az = np.where(valid, az, np.nan)
    nsrc = rg.shape[1]
    resr = np.full_like(rg, np.nan); resa = np.full_like(az, np.nan)
    for j in range(nsrc):
        if np.isfinite(rg[:, j]).sum() < 20:
            continue
        _, resr[:, j], resa[:, j] = robust_secular(rg[:, j], az[:, j], sr[:, j], sa[:, j])
    blunder = valid & ((abs(resr) > 2.0) | (abs(resa) > 5.0))
    lowpk = d['correlation_peak'] < 0.06
    clean = valid & ~blunder
    sdr = 1.4826*np.nanmedian(abs(resr[clean])); sda = 1.4826*np.nanmedian(abs(resa[clean]))
    print(f'{reg["id"]:10s} {nsrc:5d} {100*blunder.sum()/valid.sum():8.1f} {100*(blunder&lowpk).sum()/max(blunder.sum(),1):11.1f}% {sdr:11.3f} {sda:11.3f}'
          f'   (low-peak & not blunder: {100*(clean&lowpk).sum()/clean.sum():.1f}% of clean)')
    stash[reg['id']] = dict(rg=np.where(clean, rg, np.nan), az=np.where(clean, az, np.nan), sr=sr, sa=sa,
                            tide=d['tidal_range_bias_m'], sd=(sdr, sda), role=reg['role'], peak=d['correlation_peak'], snr=d['snr'])

print('\nPer-source Msf (along-flow) on blunder-cleaned data, a-posteriori sigma; sampled pilot sources only')
samp = json.load(open(ROOT + '/2017/sampling.json'))
for s in samp:
    if s['role'] != 'target':
        continue
    rg = f['range_displacement_m'][:, s['row'], s['col']]; az = f['azimuth_displacement_m'][:, s['row'], s['col']]
    hv = f['horizontal_valid_mask'][:, s['row'], s['col']] == 1
    sr = f['range_sensitivity_enu'][:2, s['row'], s['col']]; sa = f['azimuth_sensitivity_enu'][:2, s['row'], s['col']]
    rg = np.where(hv, rg, np.nan); az = np.where(hv, az, np.nan)
    _, r1, a1 = robust_secular(rg, az, sr, sa)
    bad = (abs(r1) > 2.0) | (abs(a1) > 5.0)
    rg = np.where(bad, np.nan, rg); az = np.where(bad, np.nan, az)
    sd = stash[s['region']]['sd']
    m, cov, chi2, rr, ra, n, u = fit(rg, az, sr, sa, sd)
    a, sa_, ph = amp_summary(m, cov)
    print(f'  {s["region"]} {s["source_id"]}: n={n} removed={int(np.nansum(bad))} amp={a:.3f}±{sa_:.3f} m/d  '
          f'resid sd rg={np.std(rr):.2f} az={np.std(ra):.2f} m')

print('\nRegional per-pair median over all clean unique sources, then along-flow / free-EN fit')
for rid, s in stash.items():
    rgm = np.nanmedian(s['rg'], 1); azm = np.nanmedian(s['az'], 1)
    srm = np.median(s['sr'], 1); sam = np.median(s['sa'], 1)
    nvalid = np.isfinite(s['rg']).sum(1)
    keep = nvalid >= 0.3*s['rg'].shape[1]
    rgm[~keep] = np.nan; azm[~keep] = np.nan
    # Use a-posteriori sigma from a first pass with unit weights per channel.
    uu = U1 if s['role'] == 'control' else None
    m0, c0, chi0, rr0, ra0, n, u = fit(rgm, azm, srm, sam, (1., 1.), u=uu)
    sig = (np.std(rr0), np.std(ra0))
    m, cov, chi2, rr, ra, n, u = fit(rgm, azm, srm, sam, sig, u=uu)
    if rid == 'target_1':
        U1 = u
    a, sa_, ph = amp_summary(m, cov)
    tm = np.nanmedian(s['tide'], 1); tm[~keep] = np.nan
    if s['role'] == 'target':
        mt, ct, *_ = fit(rgm, azm, srm, sam, sig, tide=tm, u=uu)
    else:
        mt, ct = np.r_[m, 0.], np.pad(cov, ((0, 1), (0, 1)))
    at, sat, pht = amp_summary(mt, ct)
    scan = []
    for per in (12.5, 13.0, 13.66, 14.0, 14.2, P, 15.5, 16.5, 18, 20, 24, 30, 40):
        ms, cs, *_ = fit(rgm, azm, srm, sam, sig, u=uu, period=per)
        aa, ss, _ = amp_summary(ms, cs)
        scan.append(f'{per:.1f}:{aa/ss:.1f}')
    print('     amp/sigma by period ->', ' '.join(scan))
    flow_az = np.degrees(np.arctan2(u[0], u[1])) % 360
    print(f'  {rid:10s} ({s["role"]}) n={n} flow={flow_az:5.1f}deg  resid sd rg={sig[0]:.3f} az={sig[1]:.3f} m | '
          f'Msf amp={a:.3f}±{sa_:.3f} m/d phase={ph:6.1f} | with tide-scale col: amp={at:.3f}±{sat:.3f} '
          f'tide coef={mt[-1]:+.3f}±{np.sqrt(ct[-1,-1]):.3f}')
    stash[rid].update(rgm=rgm, azm=azm, srm=srm, sam=sam, sig=sig)

if __name__ == '__main__' and len(sys.argv) > 1:
    np.savez(sys.argv[1], **{k+'_rgm': v['rgm'] for k, v in stash.items()},
             **{k+'_azm': v['azm'] for k, v in stash.items()}, early=early, late=late)
