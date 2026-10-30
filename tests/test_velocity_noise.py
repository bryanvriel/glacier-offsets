import unittest
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal
from glacier_offsets.velocity_noise import (permutation_cycles,spatial_mapping,
    transfer_control_field,seasonal_time_indices,crossfit_control_bank)
from glacier_offsets.velocity_basis import utc_days,harmonic
from glacier_offsets.velocity_systematics import calendar_times
from test_velocity_vertical import fixture

class NoiseTests(unittest.TestCase):
    def test_each_pool_exhausted_before_recycling(self):
        rng=np.random.default_rng(9);a=permutation_cycles(rng,np.arange(7),19)
        self.assertEqual(len(np.unique(a[:7])),7);self.assertEqual(len(np.unique(a[7:14])),7)
        counts=np.bincount(a,minlength=7);self.assertLessEqual(counts.max()-counts.min(),1)

    def test_balanced_mapping_is_bijection_and_preserves_field_mean(self):
        rng=np.random.default_rng(3);groups=np.repeat(np.arange(8),4)
        mapping,blocks=spatial_mapping(rng,groups,groups)
        self.assertEqual(len(np.unique(mapping)),len(groups));self.assertEqual(len(np.unique(blocks)),8)
        # Independent Gaussian population has known regional mean variance 1/n.
        # Permuting the full field preserves every realization's mean, including
        # any common-mode component, rather than adding spatial sampling noise.
        values=rng.normal(size=(2000,32))+rng.normal(size=(2000,1))
        assert_allclose(values[:,mapping].mean(1),values.mean(1),atol=1e-14)

    def test_quality_and_missingness_travel_with_persistent_sources(self):
        donor={'group':np.repeat(np.arange(3),2),'residual':np.arange(6*5*2).reshape(6,5,2).astype(float),
            'valid':np.ones((6,5),bool),'snr':np.arange(30).reshape(6,5),'peak':np.arange(30).reshape(6,5)/100}
        donor['valid'][1,2]=False
        target={'group':donor['group'],'valid':np.ones((6,5),bool)};ti=np.array([2,1,0,4,3])
        out,diag=transfer_control_field(np.random.default_rng(4),donor,target,ti)
        ix=np.ix_(diag['donor_source_indices'],ti)
        for key,source in [('data','residual'),('snr','snr'),('peak','peak'),('valid','valid')]:assert_array_equal(out[key],donor[source][ix])

    def test_common_temporal_field_is_retained(self):
        groups=np.repeat(np.arange(2),3);data=np.broadcast_to(np.arange(7)[None,:,None],(6,7,2)).copy()
        donor={'group':groups,'residual':data,'valid':np.ones((6,7),bool),'peak':np.ones((6,7)),'snr':np.ones((6,7))}
        target={'group':groups,'valid':donor['valid']};ti=np.array([6,5,4,3,2,1,0])
        result,_=transfer_control_field(np.random.default_rng(5),donor,target,ti)
        assert_array_equal(result['data'],data[:,ti])

    def field(self):
        f,u,cs,v,z=fixture();n,t=f['valid'].shape
        f['early']=np.broadcast_to(float(utc_days('2017-01-01'))+np.arange(t)*6.+.3,(n,t)).copy();f['late']=f['early']+6
        f.update(order=np.broadcast_to((-1.)**np.arange(t),(n,t)),burst=np.arange(n)//4,
                 peak=np.ones((n,t))*.2,snr=np.ones((n,t))*10)
        return f

    def test_seasonal_indices_never_draw_from_opposite_half_or_crossing_pair(self):
        f=self.field();e,l=calendar_times(f);split=float(utc_days('2017-07-01'));dest=(e+l)/2<split
        for seed in range(10):
            ti=seasonal_time_indices(np.random.default_rng(seed),e,l)
            self.assertTrue(np.all(l[ti[dest]]<split));self.assertTrue(np.all(e[ti[~dest]]>=split))

    def test_crossfit_is_acquisition_purged_and_preserves_harmonic_not_failures_screened(self):
        f=self.field();e,l=f['early'],f['late'];h=harmonic(e,l);signal=np.einsum('stj,j->st',h,[.04,.025])
        f['data']=np.stack([np.ones_like(e)*.3,1.8+f['order']*.4+signal],axis=-1)
        # A held-out low-quality failure stays in the bank, paired with quality.
        f['data'][0,2,1]+=30;f['peak'][0,2]=.01
        bank,folds=crossfit_control_bank(f,np.eye(2))
        for record in folds:
            self.assertEqual(len(np.intersect1d(record['train_acquisition_days'],record['test_acquisition_days'])),0)
            test=record['test_indices'];s=1
            # Centering changes a constant only, not the within-half waveform.
            diff=bank['residual'][s,test,1]-signal[s,test]
            assert_allclose(diff,diff[0],atol=1e-12)
        self.assertTrue(bank['valid'][0,2]);self.assertGreater(bank['residual'][0,2,1],25)
        assert_array_equal(bank['peak'],f['peak'])

    def test_undefined_training_baseline_does_not_become_a_valid_bank_sample(self):
        f=self.field();e,l=calendar_times(f);split=float(utc_days('2017-07-01'));h1=l<split;h2=e>=split
        f['data']=np.stack([np.ones_like(f['early'])*.3,1.8+f['order']*.4],axis=-1)
        f['peak'][0,h2]=.01
        bank,folds=crossfit_control_bank(f,np.eye(2))
        self.assertFalse(bank['valid'][0,h1].any())
        self.assertTrue(bank['valid'][0,h2].all())
        self.assertEqual(folds[0]['source_train_epoch_counts'][0],0)

if __name__=='__main__':unittest.main()
