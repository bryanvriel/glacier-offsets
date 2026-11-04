"""Actual calibration restoration, half-year secular sensitivity and audit."""
import json
from pathlib import Path
import h5py
import numpy as np
from run_order_set_pilot import load,calibrate,build_state,bootstrap,uncertainty
from glacier_offsets.velocity_systematics import OrderRegionalDesign
from glacier_offsets.velocity_quality import cluster_weights
from glacier_offsets.velocity_basis import utc_days
from glacier_offsets.velocity_io import write_json,digest


def run(root):
    out=root/'2017-order-set';fields=load(root,out);cov=calibrate(fields,out);state=build_state(fields,cov)
    rows=[];support={};split=float(utc_days('2017-07-01'))
    with h5py.File(root/'frozen-2017/displacements_raw.nc') as raw,h5py.File(root/'frozen-2017/displacements_swath_corrected.nc') as corrected:
        for f in fields:
            rr,cc=f['row'],f['col'];r0,r1=int(min(rr)),int(max(rr))+1;c0,c1=int(min(cc)),int(max(cc))+1
            delta=(raw['azimuth_displacement_m'][:,r0:r1,c0:c1]-corrected['azimuth_displacement_m'][:,r0:r1,c0:c1])[:,rr-r0,cc-c0].T
            if not np.isfinite(delta[f['clean']]).all():raise ValueError('Missing restoration on matched support')
            y=f['data'].copy();y[...,1]+=delta
            d=state[f['id']]['designs']['primary_azimuth'];before=d.fit(f['clean']);after=d.fit(f['clean'],data=y)
            # Per-source constant velocity in each half, integrated across split pairs.
            step=OrderRegionalDesign(f,cov);dt=f['late']-f['early']
            first=np.nan_to_num(np.clip(np.minimum(f['late'],split)-f['early'],0,dt));second=np.nan_to_num(dt)-first
            b=np.stack([first,second],axis=-1)[:,:,None,:]/step.factor[0,0]
            step.b=b;step.bb=np.einsum('stci,stcj->stij',b,b);step.bq=np.einsum('stci,stcj->stij',b,step.q)
            stepped=step.fit(f['clean'])
            values,fail=bootstrap(f,step,500,np.random.default_rng(12348))
            early,late=f['calendar_early'],f['calendar_late'];half=[late<split,early>=split]
            fractions=[float(np.sum(f['clean'][:,h])/np.sum(f['valid'][:,h])) for h in half]
            rows.append({'region':f['id'],'actual_restored_swath_cs_change_m_day':after['cs']-before['cs'],
                'actual_restored_swath_order_change_m':after['order_m']-before['order_m'],
                'baseline_amplitude_m_day':float(np.linalg.norm(before['cs'])),
                'half_year_secular_amplitude_m_day':float(np.linalg.norm(stepped['cs'])),
                'half_year_secular_cs_change_m_day':stepped['cs']-before['cs'],
                'half_year_secular_uncertainty':uncertainty(stepped['cs'],values),'bootstrap_failures':fail,
                'H1_H2_retained_fraction_of_valid':fractions,
                'H1_H2_valid_pair_counts':[int(f['valid'][:,h].sum()) for h in half]})
            support.update({f['id']+'__'+k:f[k] for k in ('source_id','row','col','early','late','group','geometry','order','burst','clean')})
            support[f['id']+'__axis_EN']=state[f['id']]['axis']
            print(f['id'],rows[-1]['actual_restored_swath_cs_change_m_day'],rows[-1]['half_year_secular_amplitude_m_day'],flush=True)
    np.savez_compressed(out/'fitted-support.npz',**support)
    write_json(out/'correction-secular-sensitivity.json',{'rows':rows,
        'restoration':'restore actual raw-minus-swath-corrected azimuth at each sampled source; matched corrected-data mask, fixed covariance, fixed axis; no assumption of perfectly constant pixel spacing',
        'secular':'two source-specific azimuth secular rates split 2017-07-01 with exact interval integration; common Msf and burst-order; 500 rescreened temporal/spatial bootstraps',
        'calving':'calendar sensitivity only; no new imagery/rift intersection validation, therefore not a causal A68 test'})
    # Independently reconstruct every saved prediction from coefficients and basis.
    errors=[]
    if (out/'fitted-coefficients.npz').exists():
        with np.load(out/'fitted-coefficients.npz') as saved:
            for f in fields:
                for name,d in state[f['id']]['designs'].items():
                    key=f['id']+'__'+name+'__';cs=saved[key+'cs'];order=saved[key+'order_m'];nu=saved[key+'nuisance']
                    if d.direct:hc=cs
                    else:
                        u=state[f['id']]['axis'];hc=np.r_[u[0]*cs,u[1]*cs]
                    coeff=np.r_[hc,order]
                    prediction=np.einsum('stci,si->stc',d.b,nu)+np.einsum('stci,i->stc',d.q,coeff)
                    prediction=np.einsum('ij,stj->sti',d.factor,prediction)
                    errors.append(float(np.max(abs(prediction-saved[key+'prediction']))))
        write_json(out/'reconstruction-check.json',{'fits':len(errors),'maximum_prediction_difference_m':max(errors)})

if __name__=='__main__':run(Path('.runtime/velocity-recovery-larsen'))
