"""Explicit selection attrition and identifiable conditional correction fits."""
from copy import deepcopy
import unittest
import numpy as np

from tools.summarize_range2x_pairs import phase_fit_cohorts,sine_fit


class PhaseReportTests(unittest.TestCase):
    def sample(self):
        phase=np.tile(np.linspace(-.4375,.4375,8),2)[None,:]
        swath=np.array([[1]*8+[2]*8]);amplitude=np.where(swath==1,.08,.05)
        baseline=phase-amplitude*np.sin(2*np.pi*phase)
        data={}
        for name in ('deramp0','deramp1','range2x'):
            data[name]={'valid':np.ones(phase.shape,bool),'correlation_peak':np.ones(phase.shape),
                        'snr':np.full(phase.shape,25.),'total/range_px':baseline.copy() if name=='deramp0' else phase.copy(),
                        'total/azimuth_px':np.where(swath==1,0.,17.)}
        data['deramp0']['valid'][0,0]=False
        data['deramp1']['correlation_peak'][0,1]=.29
        data['range2x']['total/azimuth_px'][0,2]=.3
        data['range2x']['snr'][0,3]=19.99
        return data,swath

    def test_each_stage_reports_count_loss_and_both_swath_amplitudes(self):
        data,swath=self.sample();report=phase_fit_cohorts(data,swath)
        stages=list(report['cohorts'].values())
        self.assertEqual([stage['n'] for stage in stages],[15,14,13,12])
        self.assertEqual([stage['removed_from_previous_stage_n'] for stage in stages[1:]],[1,1,1])
        self.assertEqual(stages[-1]['n_by_swath'],{'iw1':4,'iw2':8})
        for treatment in ('deramp1','range2x'):
            for swath,amplitude in (('iw1',.08),('iw2',.05)):
                fit=stages[-1]['fits'][treatment][swath]
                self.assertAlmostEqual(fit['amplitude_px'],amplitude,places=10)
                self.assertEqual(sum(fit['phase_bin_counts']),fit['n'])

    def test_uncut_quality_cohort_retains_peak_jumps_and_high_snr_is_minimum(self):
        data,swath=self.sample();a=phase_fit_cohorts(data,swath)['cohorts']
        data['deramp0']['snr'][0,4]=19.
        b=phase_fit_cohorts(data,swath)['cohorts']
        self.assertEqual(list(a.values())[1]['n'],14)
        self.assertEqual(list(b.values())[1]['n'],14)
        self.assertEqual(list(b.values())[-1]['n'],11)

    def test_nonidentifiable_and_empty_fits_do_not_fabricate_amplitudes(self):
        for y,x in [(np.array([]),np.array([])),(np.arange(4.),np.zeros(4))]:
            fit=sine_fit(y,x);self.assertIsNone(fit['amplitude_px']);self.assertIsNone(fit['pearson_r'])


if __name__=='__main__':unittest.main()
