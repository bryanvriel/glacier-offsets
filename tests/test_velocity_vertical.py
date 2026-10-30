import unittest
import numpy as np
from numpy.testing import assert_allclose
from glacier_offsets.velocity_vertical import VerticalRegionalDesign,vertical_columns
from glacier_offsets.velocity_basis import harmonic
from glacier_offsets.velocity_inspect import M2_PERIOD,O1_PERIOD


def fixture():
    rng=np.random.default_rng(87);s,t=8,60
    e=np.broadcast_to(np.arange(t)*6.+6209.3,(s,t)).copy()+np.arange(s)[:,None]/86400
    l=e+6;g=np.broadcast_to([[-.5,.1,-.86],[.1,.99,.002]],(s,2,3)).copy()
    f={'early':e,'late':l,'geometry':g,'valid':np.ones((s,t),bool),'group':np.array([str(i//2) for i in range(s)])}
    u=np.array([.8,.6]);cs=np.array([.08,.06]);v=rng.normal(0,.05,(s,2))+[.7,.4]
    z=rng.normal(0,.08,(s,4))
    data=np.einsum('sij,sj->si',g[...,:2],v)[:,None]*(l-e)[...,None]
    data+=np.einsum('stj,j->st',harmonic(e,l),cs)[...,None]*np.einsum('sci,i->sc',g[...,:2],u)[:,None]
    data+=np.einsum('stj,sj->st',vertical_columns(e,l),z)[...,None]*g[:,None,:,2]
    f['data']=data
    return f,u,cs,v,z


class VerticalTests(unittest.TestCase):
    def test_known_source_specific_vertical_and_shared_horizontal_recovery(self):
        f,u,cs,v,z=fixture();d=VerticalRegionalDesign(f,np.array([[.09,.04],[.04,.64]]))
        fit=d.fit(f['valid'],u,details=True)
        assert_allclose(fit['cs'],cs,atol=2e-13)
        assert_allclose(fit['nuisance'][:,:2],v,atol=2e-13)
        assert_allclose(fit['nuisance'][:,2:],z,atol=2e-13)
        assert_allclose(fit['prediction'],f['data'],atol=1e-12)

    def test_agrees_with_independent_dense_global_solution_for_noisy_data(self):
        f,u,cs,v,z=fixture();rng=np.random.default_rng(3)
        f['data']+=rng.normal(size=f['data'].shape)*[.3,.8]
        good=f['valid'];good[0,4:7]=False
        d=VerticalRegionalDesign(f,np.diag([.09,.64]));result=d.fit(good,u)
        ns,nt=good.shape;full=np.zeros((ns*nt*2,ns*6+2));y=[]
        for s in range(ns):
            e,l=f['early'][s],f['late'][s];g=f['geometry'][s]
            b=np.concatenate([(l-e)[:,None,None]*g[None,:,:2],vertical_columns(e,l)[:,None]*g[None,:,2,None]],axis=-1)
            h=harmonic(e,l)[:,None]*(g[:,:2]@u)[None,:,None]
            scale=np.array([.3,.8])[None,:,None]
            full[s*nt*2:(s+1)*nt*2,s*6:(s+1)*6]=(b/scale).reshape(-1,6)
            full[s*nt*2:(s+1)*nt*2,-2:]=(h/scale).reshape(-1,2)
            y.extend((f['data'][s]/np.array([.3,.8])).ravel())
        mask=np.repeat(good.ravel(),2)
        actual=np.linalg.lstsq(full[mask],np.array(y)[mask],rcond=None)[0][-2:]
        assert_allclose(result['cs'],actual,atol=1e-12)

    def test_height_basis_is_endpoint_difference_not_velocity_integral(self):
        e=np.array([6209.3]);l=e+.1
        z=vertical_columns(e,l)[0]
        assert_allclose(z[:2],[np.cos(2*np.pi*l[0]/M2_PERIOD)-np.cos(2*np.pi*e[0]/M2_PERIOD),np.sin(2*np.pi*l[0]/M2_PERIOD)-np.sin(2*np.pi*e[0]/M2_PERIOD)])

    def test_restoring_known_tide_changes_only_vertical_coefficients(self):
        f,u,cs,v,z=fixture();d=VerticalRegionalDesign(f,np.diag([.09,.64]));before=d.fit(f['valid'],u,details=True)
        correction=np.array([.5,.1,.15,-.04])
        restored=f['data']+np.einsum('stj,j->st',vertical_columns(f['early'],f['late']),correction)[...,None]*f['geometry'][:,None,:,2]
        after=d.fit(f['valid'],u,data=restored,details=True)
        assert_allclose(after['cs'],before['cs'],atol=2e-13)
        assert_allclose(after['nuisance'][:,2:]-before['nuisance'][:,2:],np.broadcast_to(correction,z.shape),atol=1e-12)

    def test_matched_fixed_and_azimuth_models_reproduce_existing_estimator(self):
        from glacier_offsets.velocity_quality import RegionalDesign
        f,u,_,_,_=fixture();cov=np.array([[.09,.04],[.04,.64]])
        good=f['valid'].copy();good[1,5:12]=False
        temporal=np.repeat([0.,2.,1.],20);spatial=np.array([.5,.5,1.,1.,.5,.5,.5,.5])
        for model,channel in [('fixed_vertical','joint'),('azimuth_only','azimuth')]:
            old=RegionalDesign(f,good,cov,channel).fit(u,temporal,spatial)
            new=VerticalRegionalDesign(f,cov,model).fit(good,u,temporal=temporal,spatial=spatial)
            assert_allclose(new['cs'],old['cs'],atol=1e-12)
            assert_allclose(new['information'],old['information'],rtol=1e-12)

if __name__=='__main__':unittest.main()
