"""Exploratory regional Msf inference after body tide and spacecraft nuisance.

Independent of production mapping. The order sign is B_later - B_earlier;
its coefficient is a displacement (m), not velocity. The primary target is
satellite-azimuth velocity; along-flow estimates require a supplied EN axis.
"""
import numpy as np
from .velocity_basis import harmonic, PERIOD
from .velocity_quality import cluster_weights, quiet_median, reference_times, purged_folds
from .velocity_vertical import vertical_columns


def calendar_times(field):
    """Complete scene calendar for resampling labels; never fit endpoints."""
    if 'calendar_early' in field:return field['calendar_early'],field['calendar_late']
    return reference_times(field)


def weighted_median(values, weights, index=None):
    """Row medians, including integer bootstrap counts and missing support."""
    values=np.asarray(values);weights=np.broadcast_to(weights,values.shape)
    weights=np.where(np.isfinite(values),weights,0.)
    if index is None:index=np.argsort(values,axis=1)
    sorted_values=np.take_along_axis(values,index,axis=1)
    cumulative=np.cumsum(np.take_along_axis(weights,index,axis=1),axis=1)
    total=cumulative[:,-1]
    selected=np.argmax(cumulative>=total[:,None]/2,axis=1)
    return np.where(total>0,sorted_values[np.arange(len(values)),selected],np.nan)


def robust_background(field,train,counts=None,threshold=.08):
    """Train-only robust secular RA plus source-specific alternating azimuth.

    Positive/negative-order medians avoid leverage from failed matches. The
    screen permits source-dependent order amplitudes; the science fit uses a
    shared coefficient per owning burst. No harmonic enters this screen.
    """
    dt=field['late']-field['early'];data=field['data'];order=field['order']
    counts=np.ones(dt.shape[1]) if counts is None else counts
    mask=field['valid'] & train[None] & np.isfinite(field['peak']) & (field['peak']>=threshold)
    w=mask*counts[None];rate=data/dt[...,None]
    vr=weighted_median(rate[...,0],w)
    plus=w*(order>0);minus=w*(order<0)
    yp=weighted_median(rate[...,1],plus);ym=weighted_median(rate[...,1],minus)
    xp=weighted_median(order/dt,plus);xm=weighted_median(order/dt,minus)
    with np.errstate(invalid='ignore',divide='ignore'):
        bias=(yp-ym)/(xp-xm);va=yp-bias*xp
    fallback=weighted_median(rate[...,1],w)
    va=np.where(np.isfinite(va),va,fallback);bias=np.nan_to_num(bias)
    velocity=np.stack([vr,va],axis=-1)
    prediction=np.nan_to_num(dt)[...,None]*velocity[:,None]
    prediction[...,1]+=order*bias[:,None]
    return prediction,bias


def order_residual(field,threshold=.08,counts=None,allowed=None):
    residual=np.full_like(field['data'],np.nan)
    for train,test in purged_folds(*calendar_times(field),allowed=allowed):
        prediction,_=robust_background(field,train,counts,threshold)
        residual[:,test]=field['data'][:,test]-prediction[:,test]
    return residual


def order_screen(field,threshold=.08,counts=None,allowed=None,*,max_range_residual_m=2.,max_azimuth_residual_m=5.):
    residual=order_residual(field,threshold,counts,allowed)
    return (field['valid'] & np.isfinite(field['peak']) & (field['peak']>=threshold)
            & np.isfinite(residual).all(-1) & (abs(residual[...,0])<=max_range_residual_m)
            & (abs(residual[...,1])<=max_azimuth_residual_m))


def inverse_normal(matrix):
    scale=np.sqrt(np.maximum(np.diagonal(matrix,axis1=-2,axis2=-1),1e-30))
    unit=matrix/scale[...,None]/scale[...,None,:]
    return np.linalg.pinv(unit,rcond=1e-10,hermitian=True)/scale[...,None]/scale[...,None,:]


class OrderRegionalDesign:
    """Schur-eliminate source backgrounds, then shared per-burst order terms.

    `azimuth_direct` uses one source secular velocity and shared radar C/S.
    `azimuth_along`, `fixed_vertical`, `residual_vertical` project shared EN
    Msf along direction. The latter also has four source-specific U columns.
    """
    def __init__(self,field,covariance,model='azimuth_direct',order=True,period=PERIOD):
        if model not in ('azimuth_direct','azimuth_along','fixed_vertical','residual_vertical'):
            raise ValueError('Unknown model')
        self.field=field;self.model=model;self.direct=model=='azimuth_direct'
        self.channels=[1] if model.startswith('azimuth') else [0,1]
        self.bursts=np.unique(field['burst']) if order else np.array([],int)
        self.factor=np.linalg.cholesky(np.asarray(covariance)[np.ix_(self.channels,self.channels)])
        geometry=field['geometry'][:,self.channels]
        dt=np.nan_to_num(field['late']-field['early'])
        hb=np.nan_to_num(harmonic(field['early'],field['late'],period))
        if self.direct:target=hb[:,:,None,:]
        else:target=np.einsum('sci,stj->stcij',geometry[...,:2],hb).reshape(*dt.shape,len(self.channels),4)
        self.nh=target.shape[-1]
        oc=np.zeros((*dt.shape,len(self.channels),len(self.bursts)))
        for j,burst in enumerate(self.bursts):
            oc[:,:,-1,j]=(field['burst']==burst)[:,None]*field['order']
        q=np.concatenate([target,oc],axis=-1)
        secular=geometry[:,None,:,:2] if len(self.channels)==2 else np.ones((len(dt),1,1,1))
        b=dt[:,:,None,None]*secular;self.n_secular=b.shape[-1]
        if model=='residual_vertical':
            b=np.concatenate([b,vertical_columns(field['early'],field['late'])[:,:,None,:]*geometry[:,None,:,2,None]],axis=-1)
        self.b=np.linalg.solve(self.factor[None,None],b)
        self.q=np.linalg.solve(self.factor[None,None],q)
        self.bb=np.einsum('stci,stcj->stij',self.b,self.b)
        self.bq=np.einsum('stci,stcj->stij',self.b,self.q)
        self.qq=np.einsum('stci,stcj->stij',self.q,self.q)

    def fit(self,good,direction=None,data=None,temporal=None,spatial=None,details=False,operator=False):
        s,t=good.shape;tc=np.ones(t) if temporal is None else temporal
        sw=cluster_weights(self.field['group']) if spatial is None else spatial
        sw=np.where((good&(tc[None]>0)).sum(1)>=8,sw,0.)
        w=good*tc[None]
        nn=np.einsum('st,stij->sij',w,self.bb)
        nq=np.einsum('st,stij->sij',w,self.bq)
        inv=inverse_normal(nn)
        information=np.einsum('s,sij->ij',sw,np.einsum('st,stij->sij',w,self.qq)-np.einsum('sji,sjk,skl->sil',nq,inv,nq))
        values=self.field['data'] if data is None else data
        y=np.nan_to_num(np.linalg.solve(self.factor[None,None],values[:,:,self.channels,None])[...,0])
        ny=np.einsum('st,stci,stc->si',w,self.b,y)
        qy=np.einsum('st,stci,stc->si',w,self.q,y)
        rhs=np.einsum('s,si->i',sw,qy-np.einsum('sji,sjk,sk->si',nq,inv,ny))
        project=np.zeros((self.nh+len(self.bursts),2+len(self.bursts)))
        project[:self.nh,:2]=(np.eye(2) if self.direct else np.array([[direction[0],0],[0,direction[0]],[direction[1],0],[0,direction[1]]]))
        project[self.nh:,2:]=np.eye(len(self.bursts))
        system=project.T@information@project;system=(system+system.T)/2
        # Remove unrepresented bursts in resamples without dropping the target.
        active=np.r_[True,True,np.diag(system)[2:]>1e-12]
        small=system[np.ix_(active,active)]
        if np.linalg.matrix_rank(small)<len(small):raise ValueError('Unidentifiable harmonic/order design')
        solved=np.zeros(len(system));solved[active]=np.linalg.solve(small,(project.T@rhs)[active])
        full=project@solved
        marginal=system[:2,:2]-system[:2,2:]@np.linalg.pinv(system[2:,2:],hermitian=True)@system[2:,:2]
        if np.linalg.eigvalsh(marginal).min()<=0:raise ValueError('Nonpositive target information')
        result={'cs':solved[:2],'order_m':solved[2:],'information':marginal,'condition':float(np.linalg.cond(marginal)),
                'active_sources':int(np.sum(sw>0))}
        if details:
            nuisance=np.einsum('sij,sj->si',inv,ny-np.einsum('sij,j->si',nq,full))
            pred=np.einsum('stci,si->stc',self.b,nuisance)+np.einsum('stci,i->stc',self.q,full)
            result.update(prediction=np.einsum('ij,stj->sti',self.factor,pred),nuisance=nuisance)
        if operator:
            adjusted=self.q-np.einsum('stci,sij,sjk->stck',self.b,inv,nq)
            ps=project[:,active]
            op=np.einsum('ij,kj,stck,st,s->istc',np.linalg.inv(small),ps,adjusted,w,sw)[:2]
            # Operator accepts unwhitened selected-channel observations.
            result['operator']=np.einsum('istc,cj->istj',op,np.linalg.inv(self.factor))
        return result


def holm(pvalues):
    p=np.array(pvalues);index=np.argsort(p);adjusted=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[index])
    result=np.empty_like(p);result[index]=np.minimum(adjusted,1.)
    return result


class OrderScreenCache:
    """Reuse only sorting and fold geometry when resampling unchanged data.

    Medians, training support, predictions, and screen decisions are recomputed
    for every draw. A new cache is required for each synthetic dataset.
    """
    def __init__(self,field,threshold=.08):
        self.field=field;self.threshold=threshold
        self.dt=field['late']-field['early'];self.rate=field['data']/self.dt[...,None]
        self.x=field['order']/self.dt
        self.ir=np.argsort(self.rate[...,0],axis=1);self.ia=np.argsort(self.rate[...,1],axis=1);self.ix=np.argsort(self.x,axis=1)
        self.mask=field['valid']&np.isfinite(field['peak'])&(field['peak']>=threshold)
        self.folds=list(purged_folds(*calendar_times(field)))

    def mask_for(self,counts):
        f=self.field;residual=np.full_like(f['data'],np.nan)
        for train,test in self.folds:
            w=self.mask*train[None]*counts[None]
            vr=weighted_median(self.rate[...,0],w,self.ir)
            plus=w*(f['order']>0);minus=w*(f['order']<0)
            yp=weighted_median(self.rate[...,1],plus,self.ia);ym=weighted_median(self.rate[...,1],minus,self.ia)
            xp=weighted_median(self.x,plus,self.ix);xm=weighted_median(self.x,minus,self.ix)
            with np.errstate(invalid='ignore',divide='ignore'):
                bias=(yp-ym)/(xp-xm);va=yp-bias*xp
            fallback=weighted_median(self.rate[...,1],w,self.ia)
            va=np.where(np.isfinite(va),va,fallback);bias=np.nan_to_num(bias)
            prediction=np.nan_to_num(self.dt[:,test])[...,None]*np.stack([vr,va],axis=-1)[:,None]
            prediction[...,1]+=f['order'][:,test]*bias[:,None]
            residual[:,test]=f['data'][:,test]-prediction
        return self.mask&np.isfinite(residual).all(-1)&(abs(residual[...,0])<=2)&(abs(residual[...,1])<=5)
