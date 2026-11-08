import unittest
import numpy as np
from numpy.testing import assert_allclose
from glacier_offsets.velocity_systematics import (OrderRegionalDesign,order_screen,holm,
    OrderScreenCache,calendar_times,robust_background)
from glacier_offsets.velocity_basis import harmonic
from test_velocity_vertical import fixture

class OrderTests(unittest.TestCase):
    def make(self):
        f,u,cs,v,z=fixture()
        f.update(burst=np.arange(8)//4,order=np.broadcast_to((-1.)**np.arange(60),(8,60)).copy(),peak=np.full((8,60),.2))
        f['data'][...,1]+=f['order']*np.where(f['burst']==0,-.4,-.2)[:,None]
        return f,u,cs

    def test_order_vertical_recovery_and_operator(self):
        f,u,cs=self.make();d=OrderRegionalDesign(f,np.array([[.09,.04],[.04,.64]]),'residual_vertical')
        fit=d.fit(f['valid'],u,details=True,operator=True)
        assert_allclose(fit['cs'],cs,atol=1e-12)
        assert_allclose(fit['order_m'],[-.4,-.2],atol=1e-12)
        assert_allclose(fit['prediction'],f['data'],atol=1e-12)
        assert_allclose(np.einsum('istc,stc->i',fit['operator'],f['data']),cs,atol=1e-12)

    def test_direct_azimuth_matches_dense_gls_with_missing_and_weights(self):
        f,u,_=self.make();rng=np.random.default_rng(3);f['data']+=rng.normal(size=f['data'].shape)*.1
        good=f['valid'].copy();good[0,3:7]=False
        tc=np.tile([1.,0.,2.],20);sw=np.arange(1.,9)/10
        d=OrderRegionalDesign(f,np.diag([.09,.64]));fit=d.fit(good,temporal=tc,spatial=sw,operator=True)
        h=harmonic(f['early'],f['late']);full=np.zeros((8*60,12))
        for s in range(8):
            sl=slice(s*60,(s+1)*60);full[sl,s]=f['late'][s]-f['early'][s]
            full[sl,8:10]=h[s];full[sl,10+f['burst'][s]]=f['order'][s]
        weights=(good*tc[None]*sw[:,None]).ravel();y=f['data'][...,1].ravel()
        solution=np.linalg.lstsq(full*np.sqrt(weights[:,None]),y*np.sqrt(weights),rcond=None)[0]
        assert_allclose(fit['cs'],solution[8:10],atol=1e-12)
        assert_allclose(fit['order_m'],solution[10:],atol=1e-12)
        assert_allclose(np.einsum('istc,stc->i',fit['operator'],f['data'][:,:,[1]]),fit['cs'],atol=1e-12)

    def test_restored_burst_constant_order_correction_leaves_msf_unchanged(self):
        f,u,_=self.make();d=OrderRegionalDesign(f,np.eye(2));before=d.fit(f['valid'])
        y=f['data'].copy();y[...,1]+=f['order']*np.where(f['burst']==0,.181,-.181)[:,None]
        after=d.fit(f['valid'],data=y)
        assert_allclose(after['cs'],before['cs'],atol=1e-12)
        assert_allclose(after['order_m']-before['order_m'],[.181,-.181],atol=1e-12)

    def test_screen_uses_crossfit_order_and_rejects_failed_match(self):
        f,u,_=self.make();f['data'][...,0]=1.2;f['data'][...,1]=3+f['order']*8
        good=order_screen(f);self.assertTrue(good.all())
        f['data'][0,0,1]+=30;f['peak'][1,5]=.01
        good=order_screen(f);self.assertFalse(good[0,0]);self.assertFalse(good[1,5]);self.assertTrue(good[2:].all())

    def test_cached_rescreen_and_complete_calendar_with_burst_gaps(self):
        f,u,_=self.make();f['calendar_early']=f['early'][0].copy();f['calendar_late']=f['late'][0].copy()
        f['early'][:,9:11]=np.nan;f['late'][:,9:11]=np.nan;f['valid'][:,9:11]=False
        cache=OrderScreenCache(f);rng=np.random.default_rng(13)
        for _ in range(5):
            counts=rng.integers(0,4,size=60)
            self.assertTrue(np.array_equal(cache.mask_for(counts),order_screen(f,counts=counts)))
        self.assertTrue(np.isfinite(sum(calendar_times(f))).all())

    def test_wild_cluster_residual_projection_matches_explicit_refit(self):
        f,u,_=self.make();rng=np.random.default_rng(9);d=OrderRegionalDesign(f,np.eye(2))
        fitted=d.fit(f['valid'],operator=True);op=fitted['operator'];labels=np.arange(60)//5;blocks=np.unique(labels)
        residual=rng.normal(size=(8,60))
        def scores(value):
            return np.stack([np.einsum('istc,stc->i',op[:,:,labels==b],value[:,labels==b,None]) for b in blocks])
        source=[];maps=[]
        for b in blocks:
            part=np.where(labels[None]==b,residual,0.);y=np.zeros_like(f['data']);y[...,1]=part
            fit=d.fit(f['valid'],data=y,details=True);source.append(fit['cs']);maps.append(scores(part-fit['prediction'][...,0]))
        weights=rng.choice([-1.,1.],len(blocks));y=np.zeros_like(f['data']);y[...,1]=residual*weights[labels]
        explicit=d.fit(f['valid'],data=y,details=True)
        assert_allclose(weights@np.array(source),explicit['cs'],atol=1e-13)
        assert_allclose(np.einsum('b,bci->ci',weights,np.array(maps)),scores(y[...,1]-explicit['prediction'][...,0]),atol=1e-13)

    def test_holm_stepdown_monotone(self):
        assert_allclose(holm([.04,.001,.02,.5]),[.08,.004,.06,.5])

    def test_both_component_order_recovery_and_operator(self):
        f,u,cs=self.make()
        f['data'][...,0]+=f['order']*np.where(f['burst']==0,.35,-.15)[:,None]
        d=OrderRegionalDesign(f,np.array([[.09,.04],[.04,.64]]),
                              'residual_vertical',order_range=True)
        good=f['valid'].copy();good[0,3:7]=False
        fit=d.fit(good,u,temporal=np.tile([1.,0.,2.],20),
                  spatial=np.arange(1.,9)/10,details=True,operator=True)
        self.assertEqual(d.order_terms,[(0,0),(1,0),(0,1),(1,1)])
        assert_allclose(fit['cs'],cs,atol=1e-11)
        assert_allclose(fit['order_m'],[.35,-.15,-.4,-.2],atol=1e-11)
        assert_allclose(fit['prediction'],f['data'],atol=1e-11)
        assert_allclose(np.einsum('istc,stc->i',fit['operator'],f['data']),cs,atol=1e-11)

    def test_two_component_screen_and_cached_resampling(self):
        f,_,_=self.make()
        f['data'][...,0]=1.2+f['order']*8
        f['data'][...,1]=3+f['order']*10
        self.assertFalse(order_screen(f).all())
        self.assertTrue(order_screen(f,order_range=True).all())
        f['data'][0,5,0]+=30;f['peak'][1,7]=.01
        cache=OrderScreenCache(f,order_range=True)
        rng=np.random.default_rng(14)
        for _ in range(5):
            counts=rng.integers(0,4,size=60)
            actual=cache.mask_for(counts)
            expected=order_screen(f,counts=counts,order_range=True)
            self.assertTrue(np.array_equal(actual,expected))
            self.assertFalse(actual[0,5]);self.assertFalse(actual[1,7])

    def test_train_only_order_background_ignores_heldout_values(self):
        f,_,_=self.make()
        f['data'][...,0]=1.2+f['order']*.3
        f['data'][...,1]=3+f['order']*.5
        train=np.arange(60)<30
        prediction,bias=robust_background(f,train,order_range=True)
        changed={**f,'data':f['data'].copy()}
        changed['data'][:,~train]+=1e6
        repeated,other=robust_background(changed,train,order_range=True)
        assert_allclose(repeated,prediction,atol=0)
        assert_allclose(other,bias,atol=0)
        assert_allclose(bias,np.tile([.3,.5],(8,1)),atol=1e-8)

if __name__=='__main__':unittest.main()
