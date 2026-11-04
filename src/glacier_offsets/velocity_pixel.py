"""Independent-source Msf inversion in radar displacement space.

Shared 2017/2018 harmonic, year-specific secular and spacecraft-order terms,
optional residual vertical M2/O1, exact interval integration. No spatial average
or smoothing. Calendar-cluster covariance is a diagnostic, not the calibrated
regional paired bootstrap from the pilot.
"""
import numpy as np
from .velocity_basis import harmonic,PERIOD
from .velocity_vertical import vertical_columns
from .velocity_systematics import order_screen,robust_background
from .velocity_pixel_config import DEFAULT_PARAMETERS

JOINT_NAMES=('east_velocity_2017','north_velocity_2017','east_velocity_2018','north_velocity_2018',
             'order_2017','order_2018','vertical_M2_C','vertical_M2_S','vertical_O1_C','vertical_O1_S','Msf_C','Msf_S')
AZIMUTH_NAMES=('azimuth_velocity_2017','azimuth_velocity_2018','order_2017','order_2018','Msf_C','Msf_S')


def screen_years(field,screening=None):
    screening=DEFAULT_PARAMETERS['screening'] if screening is None else screening
    good=np.zeros_like(field['valid']);velocities=[]
    for year in (0,1):
        take=field['year_index']==year
        local={k:field[k][:,take] for k in ('data','valid','early','late','peak')}
        local.update(order=np.broadcast_to(field['order'][take],local['valid'].shape),
                     calendar_early=field['calendar_early'][take],calendar_late=field['calendar_late'][take])
        selected=order_screen(local,threshold=screening['min_peak'],
            max_range_residual_m=screening['max_range_residual_m'],max_azimuth_residual_m=screening['max_azimuth_residual_m']);good[:,take]=selected
        pred,bias=robust_background({**local,'valid':selected},np.ones(take.sum(),bool),threshold=screening['min_peak'])
        dt=local['late']-local['early'];pred[...,1]-=local['order']*bias[:,None]
        # Predictions are exactly secular after removal of the order component.
        rates=pred[:,0]/dt[:,0,None]
        # Some sources lack the first burst endpoint; use any finite interval.
        from .velocity_quality import quiet_median
        rates=quiet_median(pred/dt[...,None],axis=1)
        g=field['geometry_by_year'][year]
        finite=np.isfinite(g[...,:2]).all(axis=(1,2));safe=np.where(finite[:,None,None],g[...,:2],np.eye(2))
        velocities.append(np.linalg.solve(safe,np.nan_to_num(rates)[...,None])[...,0])
    velocities=np.array(velocities);mean=np.mean(velocities,axis=0);speed=np.linalg.norm(mean,axis=1)
    axis=np.divide(mean,speed[:,None],out=np.tile([1.,0.],(len(mean),1)),where=speed[:,None]>1e-8)
    norms=np.linalg.norm(velocities,axis=2);dot=np.einsum('si,si->s',velocities[0],velocities[1])
    cosine=np.divide(dot,norms[0]*norms[1],out=np.full(len(mean),np.nan),where=norms[0]*norms[1]>1e-8)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    return good,axis,velocities,speed,angle


def design(field,axis,covariances,model='residual_vertical'):
    early,late=field['early'],field['late'];dt=np.nan_to_num(late-early);year=field['year_index'];order=field['order']
    ns,nt=dt.shape;hb=np.nan_to_num(harmonic(early,late));joint=model=='residual_vertical'
    if not joint and model!='azimuth_direct':raise ValueError('Unknown pixel model')
    names=JOINT_NAMES if joint else AZIMUTH_NAMES;channels=2 if joint else 1
    a=np.zeros((ns,nt,channels,len(names)))
    geom=np.moveaxis(field['geometry_by_year'][year],0,1) # source,time,radar,ENU
    if joint:
        for y in (0,1):
            mask=(year==y)[None,:,None]
            a[:,:,:,(2*y):(2*y+2)]=dt[:,:,None,None]*geom[...,:2]*mask[...,None]
            a[:,:,1,4+y]=order[None]*(year==y)[None]
        a[:,:,:,6:10]=vertical_columns(early,late)[:,:,None,:]*geom[...,2,None]
        a[:,:,:,-2:]=hb[:,:,None,:]*np.einsum('stci,si->stc',geom[...,:2],axis)[...,None]
        factor=np.linalg.cholesky(covariances)[year];inv=np.linalg.inv(factor)
        y=field['data']
    else:
        for yr in (0,1):
            a[:,:,0,yr]=dt*(year==yr)[None];a[:,:,0,2+yr]=order[None]*(year==yr)[None]
        a[:,:,0,-2:]=hb
        inv=(1/np.sqrt(covariances[year,1,1]))[:,None,None];y=field['data'][:,:,[1]]
    a=np.nan_to_num(np.einsum('tij,stjk->stik',inv,a));y=np.nan_to_num(np.einsum('tij,stj->sti',inv,y))
    return a,y,geom,names


def solve(field,good,axis,covariances,model='residual_vertical',*,cluster_days=30.):
    a,y,geom,names=design(field,axis,covariances,model);ns,nt,nc,np_=a.shape
    w=good.astype(float);normal=np.einsum('st,stci,stcj->sij',w,a,a);rhs=np.einsum('st,stci,stc->si',w,a,y)
    scale=np.sqrt(np.maximum(np.diagonal(normal,axis1=1,axis2=2),1e-30));unit=normal/scale[:,:,None]/scale[:,None,:]
    eig=np.linalg.eigvalsh(unit);rank=np.sum(eig>np.maximum(eig[:,-1,None]*1e-10,1e-12),axis=1)
    inverse=np.linalg.pinv(unit,rcond=1e-10,hermitian=True)/scale[:,:,None]/scale[:,None,:]
    coef=np.einsum('sij,sj->si',inverse,rhs)
    prediction=np.einsum('stci,si->stc',a,coef);residual=y-prediction
    # Target information after marginalizing all nuisance coefficients.
    nn=normal[:,:-2,:-2];nh=normal[:,:-2,-2:]
    sc=np.sqrt(np.maximum(np.diagonal(nn,axis1=1,axis2=2),1e-30));ninv=np.linalg.pinv(nn/sc[:,:,None]/sc[:,None,:],rcond=1e-10,hermitian=True)/sc[:,:,None]/sc[:,None,:]
    info=normal[:,-2:,-2:]-np.einsum('sji,sjk,skl->sil',nh,ninv,nh);info=(info+info.swapaxes(1,2))/2
    teig=np.linalg.eigvalsh(info);condition=np.divide(teig[:,1],teig[:,0],out=np.full(ns,np.inf),where=teig[:,0]>0)
    estimable=(rank==np_)&(teig[:,0]>1e-8)&(condition<1e8)
    # Whole-calendar cluster scores retain range/azimuth and shared endpoints.
    mid=(field['calendar_early']+field['calendar_late'])/2;labels=np.floor((mid-mid.min())/cluster_days).astype(int)
    blocks=np.unique(labels);indicator=(blocks[:,None]==labels[None]).astype(float)
    scores=np.einsum('st,stci,stc->sti',w,a,residual)
    clusters=np.einsum('bt,sti->sbi',indicator,scores)
    adjusted=np.einsum('sip,sbp->sbi',inverse[:,-2:,:],clusters)
    active=np.einsum('bt,st->sb',indicator,w)>0;nb=active.sum(1);nobs=nc*good.sum(1)
    cr1=np.divide(nb,nb-1,out=np.zeros(ns,float),where=nb>1)*np.divide(nobs-1,nobs-rank,out=np.zeros(ns,float),where=nobs>rank)
    cov=np.einsum('sbi,sbj,s->sij',adjusted,adjusted,cr1)
    cs=coef[:,-2:];amplitude=np.linalg.norm(cs,axis=1);phase=np.degrees(np.arctan2(cs[:,1],cs[:,0]))
    gradient=np.divide(cs,amplitude[:,None],out=np.zeros_like(cs),where=amplitude[:,None]>1e-12)
    amplitude_sigma=np.sqrt(np.maximum(0,np.einsum('si,sij,sj->s',gradient,cov,gradient)))
    phase_gradient=np.divide(np.stack([-cs[:,1],cs[:,0]],axis=1),amplitude[:,None]**2,out=np.zeros_like(cs),where=amplitude[:,None]>1e-12)
    phase_sigma=np.degrees(np.sqrt(np.maximum(0,np.einsum('si,sij,sj->s',phase_gradient,cov,phase_gradient))))
    # Convert residuals back to radar metres for separate-channel RMS.
    if nc==2:physical=np.einsum('tij,stj->sti',np.linalg.cholesky(covariances)[field['year_index']],residual)
    else:physical=residual*np.sqrt(covariances[field['year_index'],1,1])[None,:,None]
    count=good.sum(1);rms=np.sqrt(np.einsum('st,stc->sc',w,physical**2)/np.maximum(count[:,None],1))
    sse=np.einsum('st,stc->s',w,residual**2);improvement=np.einsum('si,sij,sj->s',cs,info,cs)
    partial_r2=improvement/np.maximum(sse+improvement,1e-30)
    return {'coefficients':coef,'names':names,'cs':cs,'amplitude':amplitude,'phase_deg':phase,
        'coefficient_covariance':cov,'amplitude_sigma':amplitude_sigma,'phase_sigma_deg':phase_sigma,
        'target_condition':condition,'rank':rank,'estimable':estimable,'rms_m':rms,'partial_r2':partial_r2,'calendar_clusters':nb}


def coverage(field,good,options=None):
    options=DEFAULT_PARAMETERS['coverage'] if options is None else options
    counts=np.stack([good[:,field['year_index']==year].sum(1) for year in (0,1)],axis=1)
    lo=np.min(np.where(good,field['early'],np.inf),axis=1);hi=np.max(np.where(good,field['late'],-np.inf),axis=1)
    span=np.where(good.any(1),hi-lo,0.)
    mid=(field['early']+field['late'])/2
    nbin=options['phase_bins'];labels=np.floor(np.nan_to_num(np.mod(mid,PERIOD)/PERIOD)*nbin).astype(int)%nbin
    bins=np.stack([np.any(good&(labels==b),axis=1) for b in range(nbin)],axis=1).sum(1)
    fit=((counts.min(1)>=options['min_pairs_per_year'])&(counts.sum(1)>=options['min_pairs_total'])
         &(span>=options['min_span_days'])&(bins>=options['min_phase_bins']))
    standard=fit&(counts.sum(1)>=options['min_pairs_standard'])
    return {'counts':counts,'span_days':span,'phase_bins':bins,'fit_support':fit,'standard_coverage':standard}
