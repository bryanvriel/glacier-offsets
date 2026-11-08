"""Prepare an isolated Evans source cache and full ENU solid-Earth correction.

Run in ice with PYTHONPATH=src; supply an interpreter with PySolid separately.
Reads the existing GPS-audit source ledger and the original horizontal package.
"""
import argparse
import json
import logging
from pathlib import Path
import subprocess
import time

import h5py
import numpy as np
import pandas as pd

from glacier_offsets.velocity_basis import EPOCH_SECONDS
from glacier_offsets.velocity_io import digest, write_json
from glacier_offsets.velocity_solid_earth import worker_file, check_worker


def prepare(root, out, python, spacing=.1):
    out.mkdir(parents=True, exist_ok=True)
    package=root/'work/stack/packages/horizontal_tide_corrected.nc'
    audit=root/'work/stack/packages/gps_informed_audit'
    pilot=root/'work/stack/packages/velocity_fit_robust'
    inputs=[package,audit/'arrays.npz',audit/'observations.csv.gz',pilot/'sampling.json',pilot/'noise.json']
    identity={str(p.resolve()):digest(p) for p in inputs}
    identity['preparer_sha256']=digest(Path(__file__))
    record=out/'observations.json';path=out/'observations.h5'
    if record.exists():
        saved=json.loads(record.read_text())
        if saved['identity']!=identity or digest(path)!=saved['sha256']:
            raise ValueError('Changed preparation inputs/code: use a separate experiment directory')
    else:
        meta=pd.read_csv(audit/'observations.csv.gz',usecols=['source_id','regions','spatial_block']).drop_duplicates('source_id').set_index('source_id')
        sampling=json.loads((pilot/'sampling.json').read_text())
        pilot_ids=[v['source_id'] for v in sampling if v['role']=='target']
        with np.load(audit/'arrays.npz') as z,h5py.File(package) as h:
            meta=meta.loc[z['source_id']]
            take=meta.regions.str.contains('target_|control_').to_numpy()
            ids=z['source_id'][take];row=z['row'][take];col=z['col'][take]
            owner=h['source_burst_index'][:][row,col].astype(int)-1
            enu=h['enu'][:].astype(str).tolist();order_enu=[enu.index(k) for k in ['east','north','up']]
            geometry=np.stack([np.moveaxis(h[k][:],0,-1)[row,col][:,order_enu] for k in ['range_sensitivity_enu','azimuth_sensitivity_enu']],axis=1)
            early=((h['earlier_burst_time'][:]-EPOCH_SECONDS)/86400.)[:,owner].T
            late=((h['later_burst_time'][:]-EPOCH_SECONDS)/86400.)[:,owner].T
            assert np.allclose(early,z['early'][take],equal_nan=True)
            assert np.allclose(late,z['late'][take],equal_nan=True)
            earlier=h['earlier_spacecraft'][:].astype(str);later=h['later_spacecraft'][:].astype(str)
            if not set(earlier).union(later)<={'S1A','S1B'}:raise ValueError('Unknown spacecraft')
            order=(later=='S1B').astype(float)-(earlier=='S1B').astype(float)
            peak=np.empty_like(early);snr=np.empty_like(early)
            data=np.empty((*early.shape,2))
            for t in range(early.shape[1]):
                peak[:,t]=h['correlation_peak'][t][row,col]
                snr[:,t]=h['snr'][t][row,col]
                for ch,k in enumerate(['range_displacement_m','azimuth_displacement_m']):
                    data[:,t,ch]=h[k][t][row,col]
            valid=z['valid'][take]&np.isfinite(data).all(-1)
            # EN and RA are stored separately in float32; their round trip is
            # consistent to micrometres, not float64 machine precision.
            roundtrip_error=float(np.max(abs(data[valid]-z['data_ra'][take][valid])))
            assert np.allclose(data[valid],z['data_ra'][take][valid],atol=1e-5,rtol=1e-12)
            strings=h5py.string_dtype()
            with h5py.File(path,'w') as target:
                arrays={'source_id':ids,'row':row,'col':col,'owner':owner,
                    'geometry_by_file':geometry[None],'early':early,'late':late,'data':data,
                    'valid':valid,'peak':peak,'snr':snr,'order':order,
                    'longitude':h['observation_longitude'][:][row,col],
                    'latitude':h['observation_latitude'][:][row,col],
                    'x':h['x'][:][col],'y':h['y'][:][row],
                    'pilot_target':np.isin(ids,pilot_ids),
                    'calendar_early':(h['earlier_time'][:]-EPOCH_SECONDS)/86400.,
                    'calendar_late':(h['later_time'][:]-EPOCH_SECONDS)/86400.,
                    'file_index':np.zeros(early.shape[1],np.int32)}
                for k,v in arrays.items():target.create_dataset(k,data=v,compression='gzip',compression_opts=2)
                for k,v in {'regions':meta.regions[take].to_numpy(),
                    'group':meta.spatial_block[take].to_numpy(),
                    'pair_id':h['pair_id'][:].astype(str),
                    'earlier_spacecraft':earlier,'later_spacecraft':later}.items():
                    target.create_dataset(k,data=np.asarray(v,dtype=object),dtype=strings)
                target.attrs.update(status='complete',order_convention='B_later - B_earlier',
                    data='original package RA metres; existing CATS range correction retained',
                    scope='unique sources in the four original targets and two control regions')
            saved={'identity':identity,'sha256':digest(path),'sources':len(ids),
                'pairs':early.shape[1],'pilot_target_sources':int(np.isin(ids,pilot_ids).sum()),
                'data_check_max_RA_EN_roundtrip_error_m':roundtrip_error}
            write_json(record,saved)
        logging.info('Prepared %d unique sources / %d pairs',saved['sources'],saved['pairs'])
    worker=Path(worker_file());versions=check_worker(str(python))
    correction_identity={'observations_sha256':digest(path),'worker_sha256':digest(worker),
                         'spacing_degrees':spacing,'dependencies':versions}
    corrections=out/'solid_earth.json';result=out/'solid_earth.h5'
    if corrections.exists():
        correction=json.loads(corrections.read_text())
        if correction['identity']!=correction_identity or correction['sha256']!=digest(result):
            raise ValueError('Changed solid-Earth inputs/code: use a separate experiment directory')
    else:
        # The worker checkpoints completed pairs; incomplete files are resumable.
        command=[str(python),'-I',str(worker),'--observations',str(path),'--output',str(result),
                 '--spacing',str(spacing),'--chunk','8']
        start=time.monotonic()
        with (out/'solid_earth.log').open('a') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if process.returncode:raise ValueError('Solid-Earth worker failed; see solid_earth.log')
        with h5py.File(result) as h:
            assert h.attrs['status']=='complete'
            summary={k:v.item() if isinstance(v,np.generic) else v for k,v in h.attrs.items()}
        write_json(corrections,{'identity':correction_identity,'sha256':digest(result),
            'summary':summary,'wall_seconds':time.monotonic()-start,
            'formula':'RA_corrected = original_RA - geometry_RA_ENU @ (SET_later_ENU - SET_earlier_ENU)',
            'ocean_tide':'existing CATS correction retained','ocean_loading':'not applied'})
        logging.info('Prepared solid-Earth correction; sampled max error %.3g m',
                     summary['max_ENU_endpoint_difference_error_m'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--solid-earth-python',type=Path,required=True)
    p.add_argument('--spacing',type=float,default=.1)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    prepare(a.root.resolve(),a.output.resolve(),a.solid_earth_python,a.spacing)
