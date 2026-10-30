"""Repeat regional frequency scans with an explicit target-conditioning guard.

The raw scan includes 12 days, whose two harmonic columns are almost collinear
at a six-day cadence. Keep that raw result for audit; write a separate guarded
scan and its provenance. No coefficient products or detection gates are changed.
"""
import argparse
import json
import logging
from pathlib import Path
import numpy as np
from glacier_offsets.velocity_basis import PERIOD
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_quality import (screen,crossfit_residual,flow_axis,RegionalDesign,
    reference_times,cluster_weights)
from tools.revalidate_velocity_quality import extract


def guarded_scan(fields,covariance,draws,condition_limit=1e6):
    rng=np.random.default_rng(np.random.SeedSequence([12345,3]));rows=[]
    periods=np.unique(np.r_[np.arange(12,20.001,.05),PERIOD]);axis=None
    for f in fields:
        f['clean']=screen(f,.08,crossfit_residual(f))
        u=flow_axis(f,f['clean']) if f['role']=='target' else axis
        if f['id']=='target_1':axis=u
        e,l=reference_times(f);mid=(e+l)/2
        labels=np.where(np.isfinite(mid),np.floor((mid-np.nanmin(mid))/30),-1).astype(int)
        labs=np.unique(labels[labels>=0]);groups=f['group'];unique=np.unique(groups)
        signs=rng.choice([-1.,1.],size=(draws,len(labs)))
        spatial=np.array([np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique)) for _ in range(draws)])
        multiplier=(signs[:,:,None]*spatial[:,None,:]).reshape(draws,-1)
        for channel in ('joint','range','azimuth'):
            base=RegionalDesign(f,f['clean'],covariance,channel)
            residual=np.nan_to_num(base.yd-base.fit(u)['prediction_scaled'])
            blocks=[];coeff=[];valid_period=[];excluded=[];conditions=[]
            for p in periods:
                d=RegionalDesign(f,f['clean'],covariance,channel,p)
                fit=d.fit(u);condition=float(np.linalg.cond(fit['information']))
                if not np.isfinite(condition) or condition>condition_limit:
                    excluded.append({'period_days':p,'target_information_condition':condition})
                    continue
                op=d.operator(u);cs=np.einsum('istc,stc->i',op,np.nan_to_num(d.yd))
                coeff.append(cs);valid_period.append(p);conditions.append(condition)
                contribution=np.einsum('istc,stc->ist',op,residual)
                grouped=np.zeros((2,len(labs),len(unique)))
                for j,g in enumerate(unique):
                    x=contribution[:,groups==g].sum(1)
                    for k,label in enumerate(labs):grouped[:,k,j]=x[:,labels==label].sum(1)
                blocks.append(grouped.reshape(2,-1))
            null=np.einsum('db,pib->dpi',multiplier,np.asarray(blocks))
            cov=np.array([np.cov(null[:,j].T) for j in range(len(valid_period))]);inv=np.linalg.pinv(cov,hermitian=True)
            stats=np.einsum('pi,pij,pj->p',coeff,inv,coeff)
            nullstats=np.einsum('dpi,pij,dpj->dp',null,inv,null);maxima=nullstats.max(1)
            ix=valid_period.index(PERIOD);peak=int(np.argmax(stats))
            rows.append({'region':f['id'],'channel':channel,'periods_days':valid_period,
                'amplitudes_m_per_day':np.linalg.norm(coeff,axis=1),'statistics':stats,
                'target_information_conditions':conditions,'excluded_ill_conditioned_periods':excluded,
                'peak_period_days':valid_period[peak],
                'Msf_scan_adjusted_p':float((1+np.sum(maxima>=stats[ix]))/(draws+1)),
                'peak_scan_adjusted_p':float((1+np.sum(maxima>=stats[peak]))/(draws+1))})
        logging.info('Guarded frequency scan: %s',f['id'])
    return {'draws':draws,'target_information_condition_limit':condition_limit,
        'reason':'Msf phase inference needs two estimable harmonic columns; 12-day columns nearly collapse at six-day cadence',
        'method':'same whole-field 30-day wild signs and 3-km resampling as raw scan, excluding ill-conditioned periods before computing maxima',
        'scope':'maximum over accepted periods within each ROI/channel; no additional across-scan correction','rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    p.add_argument('--draws',type=int,default=1000);a=p.parse_args();root=a.root.resolve();out=root/'2017-quality-regional'
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    fields,_=extract(root/'frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc',
        json.loads((root/'regions.reviewed.json').read_text())['regions'],out)
    cov=np.array(json.loads((out/'clean_noise.json').read_text())['pooled']['covariance_ra_m2'])
    result=guarded_scan(fields,cov,a.draws)
    result['source_sha256']=digest(__file__)
    write_json(out/'period_scans_conditioned.json',result)

if __name__=='__main__':main()
