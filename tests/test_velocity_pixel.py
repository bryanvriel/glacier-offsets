import unittest
import numpy as np
from numpy.testing import assert_allclose
from glacier_offsets.velocity_basis import harmonic,utc_days
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_pixel import solve,screen_years,coverage,design


def fixture():
    rng=np.random.default_rng(81);ns=5
    calendar=np.r_[float(utc_days('2017-01-01'))+np.arange(60)*6+.335,
                   float(utc_days('2018-01-02'))+np.arange(59)*6+.335]
    e=np.broadcast_to(calendar,(ns,119)).copy();e+=np.arange(ns)[:,None]/86400.;l=e+6
    year=np.r_[np.zeros(60,int),np.ones(59,int)];order=(-1.)**np.arange(119)
    g=np.broadcast_to([[-.5,.1,-.86],[.1,.99,.002]],(2,ns,2,3)).copy();g[1,:,:,0]+=1e-8
    axis=np.tile([.8,.6],(ns,1));cs=np.tile([.08,.06],(ns,1))+rng.normal(0,.005,(ns,2))
    v=rng.normal(0,.04,(2,ns,2))+[.7,.4];bias=np.array([-.4,-.25]);z=rng.normal(0,.08,(ns,4))
    geom=np.moveaxis(g[year],0,1);data=(l-e)[...,None]*np.einsum('stci,sti->stc',geom[...,:2],np.moveaxis(v[year],0,1))
    data[...,1]+=order[None]*bias[year][None]
    data+=np.einsum('stj,sj->st',vertical_columns(e,l),z)[...,None]*geom[...,2]
    data+=np.einsum('stj,sj->st',harmonic(e,l),cs)[...,None]*np.einsum('stci,si->stc',geom[...,:2],axis)
    f={'early':e,'late':l,'geometry_by_year':g,'year_index':year,'order':order,'calendar_early':calendar,'calendar_late':calendar+6,
       'data':data,'valid':np.ones((ns,119),bool),'peak':np.ones((ns,119))*.2}
    return f,axis,cs,v,bias,z


class PixelTests(unittest.TestCase):
    def test_known_two_year_signal_secular_order_and_vertical_recovery(self):
        f,u,cs,v,bias,z=fixture();cov=np.array([[[.09,.03],[.03,.64]],[[.16,.04],[.04,.49]]])
        result=solve(f,f['valid'],u,cov)
        self.assertTrue(result['estimable'].all());assert_allclose(result['cs'],cs,atol=2e-11)
        assert_allclose(result['coefficients'][:,:4],np.concatenate([v[0],v[1]],axis=1),atol=2e-11)
        assert_allclose(result['coefficients'][:,4:6],np.broadcast_to(bias,(5,2)),atol=2e-11)
        assert_allclose(result['coefficients'][:,6:10],z,atol=2e-11)

    def test_dense_gls_and_cluster_covariance_with_missingness(self):
        f,u,cs,_,_,_=fixture();rng=np.random.default_rng(12);f['data']+=rng.normal(size=f['data'].shape)*[.3,.8]
        good=f['valid'].copy();good[0,10:20]=False;cov=np.array([np.diag([.09,.64]),np.diag([.16,.49])])
        result=solve(f,good,u,cov)
        # Build an independent physical design directly from the model equation.
        for s in range(5):
            g=f['geometry_by_year'][f['year_index'],s];dt=f['late'][s]-f['early'][s];cols=[]
            for year in (0,1):
                for en in (0,1):cols.append(dt[:,None]*g[:,:,en]*(f['year_index']==year)[:,None])
            for year in (0,1):cols.append(np.stack([np.zeros(119),f['order']*(f['year_index']==year)],axis=-1))
            for j in range(4):cols.append(vertical_columns(f['early'][s],f['late'][s])[:,j,None]*g[:,:,2])
            for j in range(2):cols.append(harmonic(f['early'][s],f['late'][s])[:,j,None]*(g[:,:,:2]@u[s]))
            a=np.stack(cols,axis=-1);L=np.linalg.cholesky(cov)[f['year_index']]
            aw=np.linalg.solve(L,a);yw=np.linalg.solve(L,f['data'][s,:, :,None])[...,0]
            flat=aw[good[s]].reshape(-1,12);y=yw[good[s]].ravel();coef=np.linalg.lstsq(flat,y,rcond=None)[0]
            assert_allclose(result['coefficients'][s],coef,atol=1e-10)
            residual=yw-np.einsum('tci,i->tc',aw,coef);normal=flat.T@flat;inv=np.linalg.inv(normal)
            mid=(f['calendar_early']+f['calendar_late'])/2;labels=np.floor((mid-mid.min())/30).astype(int)
            scores=[]
            for b in np.unique(labels):
                take=good[s]&(labels==b)
                if take.any():scores.append(np.einsum('tci,tc->i',aw[take],residual[take]))
            scores=np.array(scores);n=len(y);B=len(scores)
            C=inv@scores.T@scores@inv*B/(B-1)*(n-1)/(n-12)
            assert_allclose(result['coefficient_covariance'][s],C[-2:,-2:],rtol=1e-8,atol=1e-11)

    def test_known_year_order_restoration_does_not_change_msf(self):
        f,u,cs,_,_,_=fixture();cov=np.array([np.eye(2),np.eye(2)]);before=solve(f,f['valid'],u,cov)
        f['data'][...,1]+=f['order'][None]*np.where(f['year_index']==0,.18,-.09)[None]
        after=solve(f,f['valid'],u,cov);assert_allclose(after['cs'],before['cs'],atol=1e-11)

    def test_yearwise_screen_uses_different_secular_motion_and_order(self):
        f,u,cs,_,_,_=fixture();f['data'][:,:,0]=np.where(f['year_index']==0,1.2,4.8)[None]
        f['data'][:,:,1]=np.where(f['year_index']==0,3.,9.)[None]+f['order'][None]*np.where(f['year_index']==0,8.,-7.)[None]
        good,_,_,_,_=screen_years(f);self.assertTrue(good.all())
        f['data'][0,65,1]+=30;good,_,_,_,_=screen_years(f);self.assertFalse(good[0,65]);self.assertTrue(good[1:].all())

    def test_one_year_only_does_not_pass_two_year_coverage_or_rank(self):
        f,u,_,_,_,_=fixture();good=f['valid'].copy();good[:,60:]=False
        cov=np.array([np.eye(2),np.eye(2)]);self.assertFalse(solve(f,good,u,cov)['estimable'].any())
        self.assertFalse(coverage(f,good)['fit_support'].any())

if __name__=='__main__':unittest.main()
