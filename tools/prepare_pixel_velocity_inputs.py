"""Verify compatible packages and cache unique radar-source observations.

Run in ice. Source associations must agree exactly. Each package's own ENU
geometry is retained, allowing numerical differences between export platforms.
"""
import argparse
from pathlib import Path
import json
import logging
import h5py
import h5netcdf
import numpy as np
from glacier_offsets.velocity_io import digest,revision,verify_hashes,write_json
from glacier_offsets.velocity_basis import EPOCH_SECONDS
from glacier_offsets.velocity_pixel_mixed import pair_validity


def prepare(inputs,out):
    out.mkdir(parents=True,exist_ok=True);cache=out/'observations.h5';meta=out/'inputs.json'
    identities=[{'path':str(p.resolve()),'sha256':digest(p),'revision':list(revision(p))} for p in inputs]
    if cache.exists() and meta.exists():
        old=json.loads(meta.read_text())
        if old['inputs']!=identities or old['cache_sha256']!=digest(cache):raise ValueError('Changed inputs/cache; use a new output directory')
        with h5py.File(cache) as h:
            if 'valid_including_mixed' not in h:raise ValueError('Legacy cache lacks mixed-window validity; use a new directory or recover through the inversion YAML')
        logging.info('Reusing checksum-verified source cache');return
    for path in inputs:
        with h5netcdf.File(path,'r') as h:verify_hashes(h)
        logging.info('Verified internal content hashes %s',path)
    handles=[h5py.File(path,'r') for path in inputs]
    try:
        base=handles[0];geometry_differences=[]
        for h in handles:
            for attr in ('geometry_identity','geometry_reference_date','layout_identity','geocoding_resampling'):
                if h.attrs[attr]!=base.attrs[attr]:raise ValueError('Incompatible '+attr)
            if h.attrs['geocoding_resampling']!='nearest':raise ValueError('Nearest-source geocoding required')
            for key in ('x','y','source_burst_index','source_observation_index','reference_center_line','reference_center_sample','observation_latitude','observation_longitude'):
                if not np.array_equal(h[key][:],base[key][:],equal_nan=True):raise ValueError('Incompatible '+key)
            enu=h['enu'].asstr()[:].tolist()
            if set(enu)!= {'east','north','up'}:raise ValueError('Unknown ENU convention')
            dif={k:float(np.nanmax(abs(h[k][:]-base[k][:]))) for k in ('range_sensitivity_enu','azimuth_sensitivity_enu')}
            if max(dif.values())>1e-5:raise ValueError('Geometry change exceeds numerical-export tolerance')
            geometry_differences.append(dif)
        association=base['source_observation_index'][:];supported=association>=0
        source_ids,first=np.unique(association[supported],return_index=True)
        positions=np.flatnonzero(supported)[first];rr,cc=np.unravel_index(positions,association.shape)
        owner=base['source_burst_index'][:][rr,cc].astype(int)-1
        if np.any(owner<0):raise ValueError('Missing owning burst')
        back=np.full(association.shape,-1,np.int32);back[supported]=np.searchsorted(source_ids,association[supported])
        pairs=[];seen=set();histories=[]
        for fi,h in enumerate(handles):
            history={key:json.loads(h[key][()].decode()) for key in ('calibration_json','tidal_correction_provenance_json','horizontal_conversion_provenance_json') if key in h}
            if 'tidal_correction_provenance_json' not in history:raise ValueError('Ocean-height correction required')
            histories.append(history)
            for ti,pair in enumerate(h['pair_id'].asstr()[:]):
                if pair in seen:raise ValueError('Duplicate pair '+pair)
                seen.add(pair)
                e=(h['earlier_burst_time'][ti]-EPOCH_SECONDS)/86400.;l=(h['later_burst_time'][ti]-EPOCH_SECONDS)/86400.
                scene=(h['earlier_time'][ti]-EPOCH_SECONDS)/86400.
                year=int(pair[:4])
                if pair[11:15]!=str(year):raise ValueError('Unvalidated cross-year pair')
                pairs.append((scene,fi,ti,pair,year,e,l))
        pairs.sort(key=lambda p:p[0]);years=sorted(set(p[4] for p in pairs));ns,nt=len(source_ids),len(pairs)
        if years!=[2017,2018]:raise ValueError('Expected 2017 and 2018')
        tmp=out/'observations.partial.h5'
        with h5py.File(tmp,'w') as result:
            result.attrs['status']='writing'
            result.create_dataset('x',data=base['x'][:]);result.create_dataset('y',data=base['y'][:])
            for k,v in [('source_id',source_ids),('row',rr),('col',cc),('owner',owner),('map_source_index',back),('latitude',base['observation_latitude'][:][rr,cc]),('longitude',base['observation_longitude'][:][rr,cc])]:result.create_dataset(k,data=v)
            geometry=[]
            for h in handles:
                indices=[h['enu'].asstr()[:].tolist().index(k) for k in ('east','north','up')]
                geometry.append(np.stack([h[k][:][indices][:,rr,cc].T for k in ('range_sensitivity_enu','azimuth_sensitivity_enu')],axis=1))
            result.create_dataset('geometry_by_file',data=np.array(geometry))
            options={'compression':'gzip','compression_opts':2,'chunks':(min(512,ns),nt)}
            for name,dtype in [('early','f8'),('late','f8'),('valid','?'),('snr','f4'),('peak','f4')]:result.create_dataset(name,(ns,nt),dtype=dtype,**options)
            result.create_dataset('data',(ns,nt,2),dtype='f8',compression='gzip',compression_opts=2,chunks=(min(512,ns),nt,2))
            # Fill one pair at a time without materializing native full-year cubes.
            data=np.full((ns,nt,2),np.nan);valid=np.zeros((ns,nt),bool);snr=np.full((ns,nt),np.nan,np.float32);peak=snr.copy()
            early=np.full((ns,nt),np.nan);late=early.copy();mixed=np.zeros(ns,bool)
            scene_early=[];scene_late=[];orders=[];files=[];year_index=[]
            for j,(_,fi,ti,pair,year,e,l) in enumerate(pairs):
                h=handles[fi]
                early[:,j]=e[owner];late[:,j]=l[owner]
                data[:,j]=np.stack([h[k][ti][rr,cc] for k in ('range_displacement_m','azimuth_displacement_m')],axis=-1)
                good,pair_mixed=pair_validity(h,ti,rr,cc,data[:,j],early[:,j],late[:,j])
                mixed |= pair_mixed
                valid[:,j]=good;snr[:,j]=h['snr'][ti][rr,cc];peak[:,j]=h['correlation_peak'][ti][rr,cc]
                earlier=h['earlier_spacecraft'].asstr()[ti];later=h['later_spacecraft'].asstr()[ti]
                if earlier not in ('S1A','S1B') or later not in ('S1A','S1B'):raise ValueError('Unknown spacecraft')
                orders.append(int(later=='S1B')-int(earlier=='S1B'));files.append(fi);year_index.append(years.index(year))
                scene_early.append((h['earlier_time'][ti]-EPOCH_SECONDS)/86400.);scene_late.append((h['later_time'][ti]-EPOCH_SECONDS)/86400.)
                if (j+1)%20==0:logging.info('Read %d/%d pairs',j+1,nt)
            # Keep eligibility before optional source-wide mixed-window removal.
            # The inversion applies its own recorded screening.remove_mixed_windows.
            result.create_dataset('valid_including_mixed',data=valid,compression='gzip',compression_opts=2)
            for k,v in [('data',data),('valid',valid),('snr',snr),('peak',peak),('early',early),('late',late),('calendar_early',scene_early),('calendar_late',scene_late),('order',orders),('file_index',files),('year_index',year_index),('years',years),('known_mixed_source',mixed)]:
                if k in result:result[k][:]=v
                else:result.create_dataset(k,data=v)
            result.create_dataset('pair_id',data=np.array([p[3] for p in pairs],dtype=h5py.string_dtype()))
            result.attrs.update(status='complete',EPSG=int(np.asarray(base.attrs['EPSG']).ravel()[0]),geometry_policy='each input retains its own ENU projection',inputs_json=json.dumps(identities))
        if [list(revision(p)) for p in inputs]!=[r['revision'] for r in identities]:raise ValueError('Input changed during extraction')
        tmp.replace(cache)
        write_json(meta,{'inputs':identities,'cache_sha256':digest(cache),'unique_sources':ns,'map_cells':int(supported.sum()),'pairs':nt,'years':years,
            'geometry_max_abs_differences':geometry_differences,'correction_history':histories,
            'mixed_source_exclusion':0,'known_mixed_sources':int(mixed.sum()),
            'mixed_policy':'mixed-window validity retained; optional source-wide removal controlled by inversion YAML; raw radar values retained where swath correction is unavailable for flagged mixed windows',
            'software_sha256':digest(Path(__file__))})
        logging.info('Cached %d unique sources, %d pairs; %d known mixed sources retained for optional screening',ns,nt,mixed.sum())
    finally:
        for h in handles:h.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s');prepare(a.inputs,a.output)
