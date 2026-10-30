"""Validated full-field ENU body tides, evaluated with PySolid in isce3.

Bilinear interpolation of a 0.1-degree spatial grid and linear interpolation
between bracketing UTC seconds. Exact one-point kernel and the previous exact
2017 pilot correction provide independent numerical error checks.
"""
import argparse
from collections import OrderedDict
from datetime import datetime,timedelta,timezone
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import importlib.metadata
import h5py
import numpy as np
from pysolid.solid import solid_grid


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def run(out,spacing):
    source=out/'observations.h5';target=out/'solid_earth_tides.h5';meta=out/'solid_earth_tides.json';identity=sha(source)
    if target.exists() and meta.exists():
        record=json.loads(meta.read_text())
        if record['observation_cache_sha256']!=identity or record['output_sha256']!=sha(target):raise ValueError('Changed body-tide cache')
        logging.info('Reusing verified solid-Earth correction');return
    epoch=datetime(2000,1,1,tzinfo=timezone.utc)
    with h5py.File(source) as h:
        lat=h['latitude'][:];lon=h['longitude'][:];finite=np.isfinite(lat+lon);early=h['early'][:];late=h['late'][:]
        owner=h['owner'][:];files=h['file_index'][:];geometry=h['geometry_by_file'][:]
        la0=np.floor(np.min(lat[finite])/spacing)*spacing-spacing;lo0=np.floor(np.min(lon[finite])/spacing)*spacing-spacing
        nlat=int(np.ceil((np.max(lat[finite])-la0)/spacing))+2;nlon=int(np.ceil((np.max(lon[finite])-lo0)/spacing))+2
        @lru_cache(maxsize=80)
        def grid(seconds):
            d=epoch+timedelta(seconds=int(seconds))
            return np.stack(solid_grid(d.year,d.month,d.day,d.hour,d.minute,d.second,la0,spacing,nlat,lo0,spacing,nlon),axis=-1)
        def value(seconds,la,lo):
            lower=int(np.floor(seconds));fraction=seconds-lower
            a=grid(lower)*(1-fraction)+grid(lower+1)*fraction
            y=(la-la0)/spacing;x=(lo-lo0)/spacing;iy=np.floor(y).astype(int);ix=np.floor(x).astype(int);fy=(y-iy)[:,None];fx=(x-ix)[:,None]
            return ((1-fy)*(1-fx)*a[iy,ix]+fy*(1-fx)*a[iy+1,ix]+(1-fy)*fx*a[iy,ix+1]+fy*fx*a[iy+1,ix+1])
        def exact(seconds,la,lo):
            lower=int(np.floor(seconds));fraction=seconds-lower;v=[]
            for second in (lower,lower+1):
                d=epoch+timedelta(seconds=second)
                v.append(np.array(solid_grid(d.year,d.month,d.day,d.hour,d.minute,d.second,float(la),.01,1,float(lo),.01,1))[:,0,0])
            return v[0]*(1-fraction)+v[1]*fraction
        result=np.full((*early.shape,2),np.nan);validation=[];rng=np.random.default_rng(12345)
        for t in range(early.shape[1]):
            for burst in np.unique(owner):
                rows=np.flatnonzero((owner==burst)&finite&np.isfinite(early[:,t]+late[:,t]))
                if not len(rows):continue
                e,l=early[rows[0],t]*86400.,late[rows[0],t]*86400.
                if not np.all(early[rows,t]==early[rows[0],t]) or not np.all(late[rows,t]==late[rows[0],t]):raise ValueError('Source endpoints differ inside owning burst')
                delta=value(l,lat[rows],lon[rows])-value(e,lat[rows],lon[rows])
                result[rows,t]=np.einsum('sci,si->sc',geometry[files[t],rows],delta)
                if t%10==0:
                    s=int(rng.integers(len(rows)));r=rows[s]
                    point=exact(l,lat[r],lon[r])-exact(e,lat[r],lon[r])
                    validation.append(float(np.max(abs(point-delta[s]))))
            if (t+1)%10==0:logging.info('Body tide %d/%d pairs; grid %dx%d',t+1,early.shape[1],nlat,nlon)
        if max(validation)>1e-5:raise ValueError('Body-tide interpolation exceeds 10 micrometres; refine spacing')
        prior=Path('.runtime/velocity-recovery-larsen/2017-order-set/corrected-inputs.npz');differences=[]
        if prior.exists():
            ids=h['source_id'][:];year=h['year_index'][:];ti=np.flatnonzero(year==0)
            with np.load(prior) as p:
                for rid in ('target_1','target_2','target_3','target_4','control_1','control_2'):
                    indices=np.searchsorted(ids,p[rid+'__source_id']);old=p[rid+'__set_ra_m']
                    if old.shape[1]!=len(ti):raise ValueError('2017 pair ordering differs')
                    differences.append(float(np.nanmax(abs(result[indices][:,ti]-old))))
        if differences and max(differences)>1e-5:raise ValueError('Body correction disagrees with exact pilot')
        temporary=out/'solid_earth_tides.partial.h5'
        with h5py.File(temporary,'w') as dst:
            dst.create_dataset('set_ra_m',data=result,chunks=(512,result.shape[1],2),compression='gzip',compression_opts=2)
            dst.attrs.update(status='complete',formula='geometry @ (SET_later_ENU - SET_earlier_ENU)',observation_cache_sha256=identity)
        temporary.replace(target)
        record={'observation_cache_sha256':identity,'output_sha256':sha(target),'pysolid_version':importlib.metadata.version('pysolid'),
            'spacing_degrees':spacing,'grid_shape':[nlat,nlon],'exact_validation_samples':len(validation),
            'max_ENU_endpoint_difference_error_m':max(validation),'max_exact_2017_pilot_RA_difference_m':max(differences) if differences else None,
            'method':'full ENU; source longitude/latitude; bilinear spatial grid and bracketing exact UTC seconds; own geometry per package; no flexure scaling',
            'ocean_loading':'not applied; no local model coefficients','software_sha256':sha(Path(__file__))}
        meta.write_text(json.dumps(record,indent=2)+'\n');logging.info('Validated correction: max point delta error %.3g m, pilot difference %.3g m',max(validation),max(differences))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--spacing',type=float,default=.1)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');run(a.output,a.spacing)
