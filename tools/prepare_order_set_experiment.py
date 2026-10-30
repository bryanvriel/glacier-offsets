"""Cache exact-source ENU solid-Earth tides; run in the isce3 environment.

Uses PySolid's grid kernel on a single actual observation coordinate, evaluated
at the bracketing UTC seconds and linearly interpolated to the stored burst time.
No raster interpolation, minute rounding, flexure multiplier, or source edits.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import hashlib
import importlib.metadata
import h5py
import numpy as np
from pysolid.solid import solid_grid, solid_point


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tide(lat,lon,seconds):
    epoch=datetime(2000,1,1,tzinfo=timezone.utc)
    base=int(np.floor(seconds));fraction=seconds-base
    values=[]
    for offset in (0,1):
        d=epoch+timedelta(seconds=base+offset)
        values.append(np.array(solid_grid(d.year,d.month,d.day,d.hour,d.minute,d.second,
                                         float(lat),.01,1,float(lon),.01,1))[:,0,0])
    return values[0]*(1-fraction)+values[1]*fraction


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    a=p.parse_args();out=a.root/'2017-order-set';out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((a.root/'2017-quality-regional/observations.json').read_text())
    cache=a.root/'2017-quality-regional/observations.npz'
    if sha(cache)!=meta['cache_sha256']:raise ValueError('Changed observation cache')
    source=Path(meta['input']['path'])
    if sha(source)!=meta['input']['sha256']:raise ValueError('Changed frozen input')
    if (out/'corrections.json').exists() and (out/'corrected-inputs.npz').exists():
        existing=json.loads((out/'corrections.json').read_text())
        if (existing['input']['sha256']!=sha(source) or
            existing['observation_cache_sha256']!=sha(cache) or
            existing['output_sha256']!=sha(out/'corrected-inputs.npz')):
            raise ValueError('Stale or changed correction cache: use a separate experiment directory')
        print('Verified existing immutable solid-Earth correction cache; keeping recorded kernel result',flush=True)
        return
    result={};validation=[];stats=[]
    with h5py.File(source) as h,np.load(cache) as saved:
        lat=h['observation_latitude'][:];lon=h['observation_longitude'][:];owner=h['source_burst_index'][:]
        earlier=h['earlier_spacecraft'].asstr()[:];later=h['later_spacecraft'].asstr()[:]
        if not set(earlier).union(later)<= {'S1A','S1B'}:raise ValueError('Unknown spacecraft')
        order=(later=='S1B').astype(float)-(earlier=='S1B').astype(float)
        for region in json.loads((a.root/'regions.reviewed.json').read_text())['regions']:
            rid=region['id'];f={k:saved[rid+'__'+k] for k in meta['keys']};row,col=f['row'],f['col']
            la,lo=lat[row,col],lon[row,col]
            enu=[]
            for key in ('early','late'):
                values=np.full((*f[key].shape,3),np.nan)
                for s in range(len(row)):
                    for t in np.flatnonzero(np.isfinite(f[key][s])):
                        values[s,t]=tide(la[s],lo[s],f[key][s,t]*86400.)
                enu.append(values)
            delta=enu[1]-enu[0];radar=np.einsum('sci,sti->stc',f['geometry'],delta)
            f.update(latitude=la,longitude=lo,burst=owner[row,col],order=np.broadcast_to(order,f['valid'].shape).copy(),
                     set_enu_delta_m=delta,set_ra_m=radar)
            result.update({rid+'__'+k:v for k,v in f.items()})
            # Independent public point kernel: one-minute interpolation at exact UTC.
            for s,t in [(0,0),(len(row)//2,29),(-1,59)]:
                seconds=f['early'][s,t]*86400.
                if not np.isfinite(seconds):continue
                d=datetime(2000,1,1)+timedelta(seconds=float(seconds));sec=d.hour*3600+d.minute*60+d.second+d.microsecond/1e6
                secs,*xyz=solid_point(float(la[s]),float(lo[s]),d.year,d.month,d.day,60)
                point=np.array([np.interp(sec,secs,x) for x in xyz])
                validation.append(float(np.max(np.abs(point-enu[0][s,t]))))
            stats.append({'region':rid,'sources':len(row),'burst_indices':np.unique(f['burst']).tolist(),
                          'set_delta_ra_rms_m':np.sqrt(np.nanmean(radar**2,axis=(0,1))).tolist()})
            print(rid,stats[-1],flush=True)
    np.savez_compressed(out/'corrected-inputs.npz',**result)
    record={'input':{'path':str(source.resolve()),'sha256':sha(source)},'observation_cache_sha256':sha(cache),
            'keys':list(f),'output_sha256':sha(out/'corrected-inputs.npz'),
            'pysolid_version':importlib.metadata.version('pysolid'),'kernel':'solid_grid at exact source lat/lon, bracketing seconds',
            'time':'exact stored owning-burst endpoints, UTC seconds from 2000-01-01; linear subsecond interpolation',
            'formula':'data_corrected_RA = cached_RA - geometry_RA_ENU @ (SET_later_ENU - SET_earlier_ENU)',
            'set_response':'unit body displacement; no ocean flexure scaling',
            'point_kernel_max_difference_m':max(validation),'regions':stats,
            'ocean_height':'existing CATS2008_v2023 correction retained',
            'ocean_loading':'not applied: no local load coefficients found in data/glaciers, data/geodesy, data/insar, data/mintpy; CATS ocean height is not a load model',
            'convention':'body tide displacement relative to tide-free reference, following PySolid; upstream geocoding convention not independently validated',
            'software_sha256':sha(Path(__file__))}
    (out/'corrections.json').write_text(json.dumps(record,indent=2)+'\n')
    print('Independent point-kernel interpolation max difference m',max(validation),flush=True)

if __name__=='__main__':main()
