"""Independent numerical checks for quality-conditioned, regional inference."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from glacier_offsets.velocity_basis import harmonic
from glacier_offsets.velocity_quality import (purged_folds, screen, crossfit_residual,
    paired_field_draw, RegionalDesign, cluster_weights, flow_axis)


def fixture():
    rng=np.random.default_rng(9)
    s,t=12,60
    early=np.broadcast_to(np.arange(t)*6.+6209.3,(s,t)).copy()+np.arange(s)[:,None]*.7/86400
    late=early+6
    geom=np.broadcast_to(np.array([[-.5,.1,-.86],[.1,.99,.003]]),(s,2,3)).copy()
    field={'early':early,'late':late,'geometry':geom,'valid':np.ones((s,t),bool),
           'peak':np.full((s,t),.4),'snr':np.full((s,t),20.),'group':np.array([str(i//3) for i in range(s)])}
    dt=late-early
    velocity=rng.normal(size=(s,2))*.1+[1.,.2]
    signal=np.einsum('stj,j->st',harmonic(early,late),[.08,.06])
    u=np.array([1.,0.])
    en=velocity[:,None,:]*dt[...,None]+signal[...,None]*u
    field['data']=np.einsum('sij,stj->sti',geom[...,:2],en)
    return field


class QualityValidationTests(unittest.TestCase):
    def test_acquisition_purge_and_training_subset(self):
        f=fixture();e,l=f['early'][0],f['late'][0]
        allowed=np.arange(60)<30
        for train,test in purged_folds(e,l,allowed=allowed):
            self.assertFalse(np.any(train&~allowed));self.assertFalse(np.any(test&~allowed))
            a=set(np.floor(np.r_[e[train],l[train]]));b=set(np.floor(np.r_[e[test],l[test]]))
            self.assertFalse(a&b)

    def test_region_fit_matches_explicit_source_nuisance_lstsq(self):
        f=fixture();u=np.array([1.,0.]);sigma=[.3,.8]
        for channel in ('joint','range','azimuth'):
            d=RegionalDesign(f,f['valid'],sigma,channel)
            cs=d.fit(u)['cs']
            assert_allclose(cs,[.08,.06],atol=2e-14)
            assert_allclose(np.einsum('istc,stc->i',d.operator(u),d.yd),cs,atol=1e-13)
            # Independently build one dense design with separate source seculars.
            sel=d.select;ns,nt=f['valid'].shape;bg=2 if channel=='joint' else 1
            a=np.zeros((ns*nt*len(sel),ns*bg+2));y=[]
            for s in range(ns):
                h=harmonic(f['early'][s],f['late'][s]);dt=f['late'][s]-f['early'][s]
                g=f['geometry'][s,sel,:2]
                b=dt[:,None,None]*(g[None] if bg==2 else np.ones((1,1,1)))
                a[s*nt*len(sel):(s+1)*nt*len(sel),s*bg:(s+1)*bg]=(b/np.array(sigma)[sel][None,:,None]).reshape(-1,bg)
                a[s*nt*len(sel):(s+1)*nt*len(sel),-2:]=(h[:,None,:]*(g@u)[None,:,None]/np.array(sigma)[sel][None,:,None]).reshape(-1,2)
                y.extend((f['data'][s,:,sel].T/np.array(sigma)[sel]).ravel())
            actual=np.linalg.lstsq(a,y,rcond=None)[0][-2:]
            assert_allclose(cs,actual,atol=1e-12)

    def test_paired_generator_preserves_failure_quality_but_legacy_does_not(self):
        f=fixture();f['residual']=np.zeros_like(f['data'])
        f['residual'][:,10]=100.;f['peak'][:,10]=.03;f['clean']=f['peak']>=.06
        target={**f,'peak':np.full_like(f['peak'],.4)}
        noise={'covariance_ra_m2':[[1,0],[0,1]],'floors_m':[1,1],'correlation':0.,'quality_models':{}}
        order=np.roll(np.arange(60),7)
        paired=paired_field_draw(np.random.default_rng(4),f,target,'paired_quality',noise,order)
        legacy=paired_field_draw(np.random.default_rng(4),f,target,'legacy',noise,order)
        large=np.linalg.norm(paired['data'],axis=-1)>50
        self.assertTrue(np.all(paired['peak'][large]<.06))
        self.assertTrue(np.all(legacy['peak'][large]>.06))
        self.assertFalse(np.any(screen(paired,.06,supplement=False)&large))
        clean=paired_field_draw(np.random.default_rng(4),f,target,'clean',noise,order)
        self.assertFalse(np.any(clean['valid']&large))

    def test_screen_does_not_remove_benchmark_signal_and_direction_is_physical_en(self):
        f=fixture();residual=crossfit_residual(f)
        good=screen(f,.06,residual)
        self.assertTrue(good.all())
        direction=flow_axis(f,good)
        self.assertGreater(direction[0],.9)
        # No peak fallback through high SNR for this failed-match screen.
        f['peak'][0,4]=.02;f['snr'][0,4]=100
        self.assertFalse(screen(f,.06,residual)[0,4])

    def test_duplicate_sources_cannot_inflate_equal_block_weight(self):
        assert_allclose(cluster_weights(np.array(['a','a','b'])),[.5,.5,1.])

    def test_repeating_sources_within_blocks_adds_no_regional_information(self):
        f=fixture();u=np.array([1.,0.])
        covariance=np.array([[.09,.035],[.035,.64]])
        original=RegionalDesign(f,f['valid'],covariance).fit(u)
        copied={k:np.repeat(v,4,axis=0) for k,v in f.items()}
        repeated=RegionalDesign(copied,copied['valid'],covariance).fit(u)
        assert_allclose(original['cs'],repeated['cs'],atol=1e-13)
        assert_allclose(original['information'],repeated['information'],rtol=1e-13)

    def test_control_screen_selection_is_unchanged_by_second_half_data(self):
        from tools.revalidate_velocity_quality import select_screen
        f=fixture();f.update(id='control',role='control')
        threshold,rows=select_screen([f])
        # Fixture starts in 2017; July split is determined by UTC times, not index.
        from glacier_offsets.velocity_basis import utc_days
        last=f['early']>=float(utc_days('2017-07-01'))
        f['data'][last]=10000;f['peak'][last]=.001
        other,other_rows=select_screen([f])
        self.assertEqual(threshold,other);self.assertEqual(rows,other_rows)

    def test_nyquist_period_is_ill_conditioned_despite_full_numerical_rank(self):
        f=fixture();u=np.array([1.,0.])
        nyquist=RegionalDesign(f,f['valid'],[.3,.8],'azimuth',12.).fit(u)
        msf=RegionalDesign(f,f['valid'],[.3,.8],'azimuth').fit(u)
        self.assertGreater(np.linalg.cond(nyquist['information']),1e6)
        self.assertLess(np.linalg.cond(msf['information']),2.)

if __name__=='__main__':unittest.main()
