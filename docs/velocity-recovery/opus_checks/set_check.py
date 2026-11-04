import datetime as dt, numpy as np, h5py, pysolid
NC='/Users/briel/data/insar/glacier_offsets/.runtime/velocity-recovery-larsen/frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc'
f=h5py.File(NC,'r')
P=14.765294
et=f['earlier_time'][:]; lt=f['later_time'][:]
def at(t0, lat, lon):
    t=dt.datetime(1970,1,1)+dt.timedelta(seconds=float(t0))
    t=t.replace(second=0, microsecond=0)
    out=pysolid.calc_solid_earth_tides_point(lat, lon, t, t+dt.timedelta(minutes=1), step_sec=60, display=False, verbose=False)
    return np.array([out[1][0], out[2][0], out[3][0]])  # E,N,U metres
early=et/86400.-10957; late=lt/86400.-10957; d=late-early
def harmonic(e,l,period=P):
    w=2*np.pi/period; mid=(e+l)/2; dd=l-e; fac=dd*np.sinc(dd/period)
    return np.stack((fac*np.cos(w*mid), fac*np.sin(w*mid)),-1)
for (r,c,lab) in [(280,540,'control_1'),(600,440,'target_1')]:
    lat=f['observation_latitude'][r,c]; lon=f['observation_longitude'][r,c]
    sr=f['range_sensitivity_enu'][:,r,c]; sa=f['azimuth_sensitivity_enu'][:,r,c]
    enu=np.array([[at(a,lat,lon), at(b,lat,lon)] for a,b in zip(et,lt)])
    dd=enu[:,1]-enu[:,0]
    rg=dd@sr; az=dd@sa
    print(f'{lab} lat={lat:.2f} lon={lon:.2f}: SET up std={enu[:,0,2].std():.3f} m; range-diff std={rg.std():.3f} m, az-diff std={az.std():.4f} m')
    # Same estimator as range-only regional fit: secular + along-flow Msf in range channel, flow axis as target_1 (~71.6 deg)
    u=np.array([np.sin(np.radians(71.6)), np.cos(np.radians(71.6))])
    H=harmonic(early,late); gu=sr[:2]@u
    A=np.stack([sr[0]*d, sr[1]*d, gu*H[:,0], gu*H[:,1]],-1)
    m,*_=np.linalg.lstsq(A,rg,rcond=None)
    print(f'   apparent along-flow Msf from SET in range: amp={np.hypot(m[2],m[3]):.3f} m/d phase={np.degrees(np.arctan2(m[3],m[2])):.1f} deg')
