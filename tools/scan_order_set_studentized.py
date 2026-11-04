"""Conditional studentized wild-cluster frequency scan after order and SET.

The null residual contains no fitted harmonic subtraction. Each wild replicate
refits the source secular and burst order, and recomputes its cluster covariance
under the alternative at every frequency. This avoids treating a null-model
score covariance as the alternative's coefficient uncertainty. Screens remain
fixed: this is an exploratory conditional scan, separate from rescreened tests.
"""
from pathlib import Path
import logging
import numpy as np
from run_order_set_pilot import load,calibrate
from glacier_offsets.velocity_systematics import OrderRegionalDesign,calendar_times
from glacier_offsets.velocity_quality import cluster_weights
from glacier_offsets.velocity_basis import PERIOD
from glacier_offsets.velocity_io import write_json


def run(root):
    out=root/'2017-order-set';fields=load(root,out);cov=calibrate(fields,out)
    periods=np.unique(np.r_[1/np.linspace(1/45,1/12.05,400),PERIOD,14.19,15.95])
    rows=[];arrays={};rng=np.random.default_rng(12349)
    for f in fields:
        e,l=calendar_times(f);mid=(e+l)/2;labels=np.floor((mid-np.nanmin(mid))/30).astype(int);blocks=np.unique(labels)
        good=f['clean'];sw=cluster_weights(f['group']);d0=OrderRegionalDesign(f,cov)
        b=d0.b[...,0,0];oc=d0.q[...,0,d0.nh:];y=np.nan_to_num(f['data'][...,1])/d0.factor[0,0]
        nn=np.sum(good*b**2,axis=1);ny=np.sum(good*b*y,axis=1)
        no=np.einsum('st,st,stj->sj',good,b,oc);oo=np.einsum('s,st,sti,stj->ij',sw,good,oc,oc)
        oy=np.einsum('s,st,sti,st->i',sw,good,oc,y)
        order=np.linalg.solve(oo-np.einsum('s,si,sj,s->ij',sw,no,no,1/nn),oy-np.einsum('s,si,s,s->i',sw,no,ny,1/nn))
        secular=(ny-no@order)/nn
        nullpred=(b*secular[:,None]+np.einsum('sti,i->st',oc,order))*d0.factor[0,0]
        residual=np.nan_to_num(f['data'][...,1]-nullpred)
        webb=np.array([-np.sqrt(1.5),-1.,-np.sqrt(.5),np.sqrt(.5),1.,np.sqrt(1.5)])
        weights=rng.choice(webb,(1999,len(blocks)));cr1=len(blocks)/(len(blocks)-1)
        coefficients=[];statistics=[];nullstatistics=[]
        for pi,period in enumerate(periods):
            d=OrderRegionalDesign(f,cov,period=period)
            fitted=d.fit(good,details=True,operator=True);op=fitted['operator']
            def score(value):
                return np.stack([np.einsum('istc,stc->i',op[:,:,labels==block],value[:,labels==block,None]) for block in blocks])
            r=np.nan_to_num(f['data'][...,1]-fitted['prediction'][...,0]);scores=score(r)
            covariance=scores.T@scores*cr1;inv=np.linalg.pinv(covariance,hermitian=True)
            statistic=fitted['cs']@inv@fitted['cs']
            # Linear residual projection for each input block provides the exact
            # refitted residual-cluster scores for every wild weight vector.
            source=[];maps=[]
            for block in blocks:
                component=np.where(labels[None]==block,residual,0.)
                ra=np.zeros_like(f['data']);ra[...,1]=component
                result=d.fit(good,data=ra,details=True)
                source.append(result['cs']);maps.append(score(component-result['prediction'][...,0]))
            source=np.array(source);maps=np.array(maps) # input-block, output-block, coefficient
            csstar=weights@source
            scoresstar=np.einsum('nb,bci->nci',weights,maps)
            covstar=np.einsum('nbi,nbj->nij',scoresstar,scoresstar)*cr1
            invstar=np.linalg.pinv(covstar,hermitian=True)
            nullstat=np.einsum('ni,nij,nj->n',csstar,invstar,csstar)
            coefficients.append(fitted['cs']);statistics.append(statistic);nullstatistics.append(nullstat)
        stat=np.array(statistics);nullstat=np.array(nullstatistics).T;maxnull=np.max(nullstat,axis=1)
        mi=int(np.argmin(abs(periods-PERIOD)));best=int(np.argmax(stat));away=np.where(abs(periods-PERIOD)>.6,stat,-1);ai=int(np.argmax(away))
        rows.append({'region':f['id'],'Msf_statistic':float(stat[mi]),
            'Msf_pointwise_p':float((1+np.sum(nullstat[:,mi]>=stat[mi]))/2000),
            'Msf_max_scan_p':float((1+np.sum(maxnull>=stat[mi]))/2000),
            'largest_period_days':float(periods[best]),'largest_statistic':float(stat[best]),
            'largest_away_period_days':float(periods[ai]),'largest_away_statistic':float(stat[ai])})
        arrays.update({f['id']+'__cs':coefficients,f['id']+'__statistic':stat})
        logging.info('Studentized frequency scan %s %s',f['id'],rows[-1])
    np.savez_compressed(out/'studentized-scans.npz',period_days=periods,**arrays)
    write_json(out/'studentized-scans.json',{'rows':rows,'method':__doc__,'bootstrap_draws':1999,'periods':len(periods),'blocks_days':30,
        'multiplicity':'within-region maximum over periods; no additional across-region scan-family correction; primary four-target test reported separately',
        'construction':'1999 Webb multipliers shared across frequencies, source secular and burst order refitted, alternative cluster covariance recomputed per replicate, CR1 B/(B-1)'})

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    run(Path('.runtime/velocity-recovery-larsen'))
