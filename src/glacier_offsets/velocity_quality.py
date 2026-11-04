"""Exploratory quality-screen validation and regional, dependent Msf inference.

This experiment is separate from production map eligibility. Screens never use
Msf amplitude. Regional coefficients share a harmonic but retain source-specific
secular motion and exact burst intervals; copied sources are not replicates.
"""
import warnings
import numpy as np
from .velocity_basis import harmonic


def quiet_median(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def reference_times(field):
    """Calendar labels from all available bursts; never substitute fit endpoints."""
    return quiet_median(field['early'],axis=0),quiet_median(field['late'],axis=0)


def purged_folds(early, late, width=60., allowed=None):
    """Calendar blocks with all shared acquisition dates purged from training."""
    e, l = np.asarray(early), np.asarray(late)
    usable = np.isfinite(e+l) & (l>e)
    if allowed is not None:
        usable &= allowed
    origin = np.nanmin(e[usable])
    labels = np.floor(((e+l)/2-origin)/width)
    for label in np.unique(labels[usable]):
        test = usable & (labels==label)
        acquisitions = np.unique(np.floor(np.r_[e[test], l[test]]))
        train = usable & ~test & ~np.isin(np.floor(e),acquisitions) & ~np.isin(np.floor(l),acquisitions)
        yield train, test


def secular_residual(data, dt, valid, train):
    """Robust source/channel velocities estimated only on the training subset."""
    mask = valid & train[None,:]
    velocity = quiet_median(np.where(mask[...,None],data/dt[...,None],np.nan),axis=1)
    return data-dt[...,None]*velocity[:,None,:]


def crossfit_residual(field, allowed=None):
    e, l = field['early'], field['late']
    residual = np.full_like(field['data'],np.nan)
    for train,test in purged_folds(*reference_times(field),allowed=allowed):
        r=secular_residual(field['data'],l-e,field['valid'],train)
        residual[:,test]=r[:,test]
    return residual


def screen(field, threshold, residual=None, supplement=True):
    """Reject absent/low peak, optionally gross cross-fitted RA residuals.

The 2/5 m residual limits are fixed diagnostic choices, not calibrated truth.
Synthetic injections must pass through this same observation-dependent screen.
"""
    good=field['valid'] & np.isfinite(field['peak']) & (field['peak']>=threshold)
    if supplement:
        if residual is None:
            residual=crossfit_residual(field)
        good &= np.isfinite(residual).all(-1) & (abs(residual[...,0])<=2.) & (abs(residual[...,1])<=5.)
    return good


def temporal_counts(rng, early, late, width=30.):
    """Resample whole calendar blocks; counts apply to every source/channel."""
    mid=(early+late)/2
    labels=np.floor((mid-np.nanmin(mid))/width)
    blocks=[np.flatnonzero(labels==label) for label in np.unique(labels[np.isfinite(labels)])]
    chosen=rng.integers(len(blocks),size=len(blocks))
    return np.bincount(np.concatenate([blocks[i] for i in chosen]),minlength=len(mid)).astype(float)


def moving_indices(rng, length, block_length=5):
    starts=rng.integers(length,size=int(np.ceil(length/block_length)))
    return ((starts[:,None]+np.arange(block_length))%length).ravel()[:length]


def cluster_weights(groups, rng=None):
    """Equal total contribution per physical 3 km block, optionally resampled."""
    _, index, count=np.unique(groups,return_inverse=True,return_counts=True)
    multiplicity=np.ones(len(count)) if rng is None else np.bincount(rng.integers(len(count),size=len(count)),minlength=len(count))
    return multiplicity[index]/count[index]


def flow_axis(field, good, time_counts=None, spatial_weights=None):
    data=field['data'];dt=field['late']-field['early']
    if time_counts is None:
        take=np.arange(data.shape[1])
    else:
        take=np.repeat(np.arange(data.shape[1]),np.asarray(time_counts,int))
    rates=np.where(good[...,None],data/dt[...,None],np.nan)[:,take]
    radar=quiet_median(rates,axis=1)
    en=np.einsum('sij,sj->si',np.linalg.inv(field['geometry'][...,:2]),radar)
    groups=field['group'];block=[]
    for g in np.unique(groups):
        include=groups==g
        if spatial_weights is not None and not np.any(spatial_weights[include]>0):
            continue
        value=quiet_median(en[include],axis=0)
        mult=1 if spatial_weights is None else int(round(np.sum(spatial_weights[include])))
        block.extend([value]*mult)
    v=quiet_median(np.asarray(block),axis=0)
    if not np.isfinite(v).all() or np.linalg.norm(v)<1e-8:
        return np.array([1.,0.])
    return v/np.linalg.norm(v)


class RegionalDesign:
    """Eliminate each source's secular terms before fitting shared harmonics.

    The sufficient statistics retain separate source backgrounds even when
    temporal/spatial weights change. Single-channel fits have one secular term,
    avoiding two collinear east/north secular columns.
    """
    def __init__(self,field,good,sigma,channel='joint',period=14.765294):
        self.field=field;self.good=good;self.channel=channel
        self.sigma=sigma
        select={'joint':[0,1],'range':[0],'azimuth':[1]}[channel]
        self.select=select
        geom=field['geometry'][:,select,:2]
        dt=field['late']-field['early']
        h=harmonic(field['early'].ravel(),field['late'].ravel(),period).reshape(*dt.shape,2)
        h=np.nan_to_num(h)
        target=np.einsum('sci,stj->stcij',geom,h).reshape(*dt.shape,len(select),4)
        background=dt[:,:,None,None]*(geom[:,None] if channel=='joint' else np.ones((len(geom),1,1,1)))
        covariance=np.asarray(sigma)
        if covariance.ndim==1:
            covariance=np.diag(covariance**2)
        self.factor=np.linalg.cholesky(covariance[np.ix_(select,select)])
        self.h=np.linalg.solve(self.factor[None,None],target)
        self.b=np.linalg.solve(self.factor[None,None],np.nan_to_num(background))
        self.yd=np.linalg.solve(self.factor[None,None],field['data'][:,:,select,None])[...,0]
        self.hh=np.einsum('stci,stcj->stij',self.h,self.h)
        self.bb=np.einsum('stci,stcj->stij',self.b,self.b)
        self.bh=np.einsum('stci,stcj->stij',self.b,self.h)

    def fit(self,direction,temporal=None,spatial=None,data=None,free=False):
        s,t=self.good.shape
        temporal=np.ones(t) if temporal is None else temporal
        spatial=cluster_weights(self.field['group']) if spatial is None else spatial
        weight=self.good*temporal[None]
        nn=np.einsum('st,stij->sij',weight,self.bb)
        nh=np.einsum('st,stij->sij',weight,self.bh)
        hh=np.einsum('st,stij->sij',weight,self.hh)
        y=self.yd if data is None else np.linalg.solve(self.factor[None,None],data[:,:,self.select,None])[...,0]
        y=np.nan_to_num(y)
        ny=np.einsum('st,stci,stc->si',weight,self.b,y)
        hy=np.einsum('st,stci,stc->si',weight,self.h,y)
        inv=np.linalg.pinv(nn,hermitian=True)
        ni=np.einsum('sij,sjk->sik',inv,nh)
        info=np.einsum('s,sij->ij',spatial,hh-np.einsum('sji,sjk->sik',nh,ni))
        rhs=np.einsum('s,si->i',spatial,hy-np.einsum('sji,sjk,sk->si',nh,inv,ny))
        proj=np.eye(4) if free else np.array([[direction[0],0],[0,direction[0]],[direction[1],0],[0,direction[1]]])
        system=proj.T@info@proj
        if np.linalg.matrix_rank(system)<len(system):
            raise ValueError('Regional target rank deficient')
        coeff=np.linalg.solve(system,proj.T@rhs)
        full=proj@coeff
        nuisance=np.einsum('sij,sj->si',inv,ny-np.einsum('sij,j->si',nh,full))
        pred=np.einsum('stci,si->stc',self.b,nuisance)+np.einsum('stci,i->stc',self.h,full)
        return {'cs':coeff,'full_cs':full,'prediction_scaled':pred,'direction':direction,
                'information':system,'n_sources':int(np.sum(self.good.sum(1)>=5)),
                'n_epochs':int(np.sum(self.good.sum(0)>0)), 'rank':int(np.linalg.matrix_rank(system))}

    def operator(self,direction):
        """Linear coefficient operator at fixed mask/direction, on scaled RA data."""
        weight=self.good
        spatial=cluster_weights(self.field['group'])
        nn=np.einsum('st,stij->sij',weight,self.bb)
        nh=np.einsum('st,stij->sij',weight,self.bh)
        inv=np.linalg.pinv(nn,hermitian=True)
        adjusted=self.h-np.einsum('stci,sij,sjk->stck',self.b,inv,nh)
        proj=np.array([[direction[0],0],[0,direction[0]],[direction[1],0],[0,direction[1]]])
        a=np.einsum('stci,ij->stcj',adjusted,proj)
        information=np.einsum('s,st,stci,stcj->ij',spatial,weight,a,a)
        return np.einsum('ij,stcj,st,s->istc',np.linalg.inv(information),a,weight,spatial)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self,value):
        self._sigma=np.asarray(value)


def paired_field_draw(rng, donor, target, mode, legacy_noise=None, time_index=None):
    """Resample dependent control fields and preserve quality/failure association.

    paired_quality transfers residual, validity, SNR and peak together. clean
    transfers the clean mask too. legacy transfers standardized residuals but
    keeps target qualities, reproducing the suspect exchangeability assumption.
    Each target spatial block draws a donor spatial block; all share time indices.
    """
    from .velocity_fit import covariance_ra
    n,t=target['valid'].shape
    ti=moving_indices(rng,t) if time_index is None else time_index
    donor_groups=np.unique(donor['group'])
    mapping=np.empty(n,int)
    for group in np.unique(target['group']):
        members=np.flatnonzero(target['group']==group)
        candidates=np.flatnonzero(donor['group']==rng.choice(donor_groups))
        mapping[members]=rng.choice(candidates,size=len(members),replace=True)
    ix=np.ix_(mapping,ti)
    residual=donor['residual'][ix].copy()
    valid=target['valid'] & donor['valid'][ix] & np.isfinite(residual).all(-1)
    result={**target,'valid':valid,'data':residual}
    if mode=='legacy':
        cov=covariance_ra(legacy_noise,donor['snr'][ix].ravel(),donor['peak'][ix].ravel(),True)
        standard=np.linalg.solve(np.linalg.cholesky(cov),np.nan_to_num(residual).reshape(-1,2)[...,None])[...,0]
        scale=covariance_ra(legacy_noise,target['snr'].ravel(),target['peak'].ravel(),True)
        result['data']=np.einsum('nij,nj->ni',np.linalg.cholesky(scale),standard).reshape(n,t,2)
    elif mode in ('paired_quality','clean'):
        result['snr']=donor['snr'][ix].copy()
        result['peak']=donor['peak'][ix].copy()
        if mode=='clean':
            result['valid'] &= donor['clean'][ix]
    else:
        raise ValueError('Unknown residual generator')
    return result
