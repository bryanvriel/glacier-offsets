"""Regional along-flow Msf with source-specific residual vertical M2/O1.

Radar observations remain tide-corrected. Vertical coefficients describe residual
height harmonics in metres; horizontal coefficients describe instantaneous
velocity in m/day. No free secular U or horizontal M2/O1 is introduced.
"""
import numpy as np
from .velocity_basis import harmonic
from .velocity_inspect import M2_PERIOD,O1_PERIOD
from .velocity_quality import cluster_weights


def vertical_columns(early,late):
    columns=[]
    for period in (M2_PERIOD,O1_PERIOD):
        omega=2*np.pi/period
        columns.extend([np.cos(omega*late)-np.cos(omega*early),
                        np.sin(omega*late)-np.sin(omega*early)])
    return np.nan_to_num(np.stack(columns,axis=-1))


class VerticalRegionalDesign:
    """Shared horizontal harmonic, independent secular/tidal terms per source.

    One geometry/basis instance can be reused for new masks, observations and
    bootstrap weights. All comparisons exclude sources with <8 distinct selected
    epochs. Nuisance normal matrices are normalized before pseudoinversion.
    """
    def __init__(self,field,covariance,model='residual_vertical'):
        if model not in ('residual_vertical','fixed_vertical','azimuth_only'):
            raise ValueError('Unknown regional model')
        self.field=field;self.model=model
        self.channels=[1] if model=='azimuth_only' else [0,1]
        covariance=np.asarray(covariance)
        self.factor=np.linalg.cholesky(covariance[np.ix_(self.channels,self.channels)])
        geometry=field['geometry'][:,self.channels]
        dt=field['late']-field['early']
        temporal=np.nan_to_num(harmonic(field['early'],field['late']))
        target=np.einsum('sci,stj->stcij',geometry[...,:2],temporal).reshape(*dt.shape,len(self.channels),4)
        background=np.nan_to_num(dt)[:,:,None,None]*(geometry[:,None,:,:2] if len(self.channels)==2 else np.ones((len(dt),1,1,1)))
        self.n_secular=background.shape[-1]
        if model=='residual_vertical':
            vertical=vertical_columns(field['early'],field['late'])[:,:,None,:]*geometry[:,None,:,2,None]
            background=np.concatenate([background,vertical],axis=-1)
        self.b=np.linalg.solve(self.factor[None,None],background)
        self.h=np.linalg.solve(self.factor[None,None],target)
        self.bb=np.einsum('stci,stcj->stij',self.b,self.b)
        self.bh=np.einsum('stci,stcj->stij',self.b,self.h)
        self.hh=np.einsum('stci,stcj->stij',self.h,self.h)

    def fit(self,good,direction,data=None,temporal=None,spatial=None,details=False):
        s,t=good.shape
        temporal=np.ones(t) if temporal is None else temporal
        spatial=cluster_weights(self.field['group']) if spatial is None else spatial
        support=(good&(temporal[None]>0)).sum(1)>=8
        spatial=np.where(support,spatial,0.)
        weight=good*temporal[None]
        nn=np.einsum('st,stij->sij',weight,self.bb)
        nh=np.einsum('st,stij->sij',weight,self.bh)
        hh=np.einsum('st,stij->sij',weight,self.hh)
        scales=np.sqrt(np.maximum(np.diagonal(nn,axis1=1,axis2=2),1e-30))
        normalized=nn/scales[:,:,None]/scales[:,None,:]
        inv=np.linalg.pinv(normalized,rcond=1e-10,hermitian=True)/scales[:,:,None]/scales[:,None,:]
        values=self.field['data'] if data is None else data
        y=np.linalg.solve(self.factor[None,None],values[:,:,self.channels,None])[...,0]
        y=np.nan_to_num(y)
        ny=np.einsum('st,stci,stc->si',weight,self.b,y)
        hy=np.einsum('st,stci,stc->si',weight,self.h,y)
        information=np.einsum('s,sij->ij',spatial,hh-np.einsum('sji,sjk,skl->sil',nh,inv,nh))
        rhs=np.einsum('s,si->i',spatial,hy-np.einsum('sji,sjk,sk->si',nh,inv,ny))
        project=np.array([[direction[0],0],[0,direction[0]],[direction[1],0],[0,direction[1]]])
        system=project.T@information@project
        system=(system+system.T)/2
        if np.linalg.matrix_rank(system)<2 or np.linalg.eigvalsh(system).min()<=0:
            raise ValueError('Horizontal target is not identifiable')
        cs=np.linalg.solve(system,project.T@rhs)
        result={'cs':cs,'information':system,'condition':float(np.linalg.cond(system)),
                'active_sources':int(np.sum(spatial>0))}
        if details:
            full=project@cs
            nuisance=np.einsum('sij,sj->si',inv,ny-np.einsum('sij,j->si',nh,full))
            predicted=np.einsum('stci,si->stc',self.b,nuisance)+np.einsum('stci,i->stc',self.h,full)
            result['prediction']=np.einsum('ij,stj->sti',self.factor,predicted)
            result['nuisance']=nuisance
            if self.model=='residual_vertical':
                result['vertical_cs_mean']=np.average(nuisance[:,self.n_secular:],axis=0,weights=spatial)
        return result
