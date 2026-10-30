"""Audit complete order/SET artifacts and reconstruct raw radar predictions."""
import json
from pathlib import Path
import h5py
import numpy as np
from glacier_offsets.velocity_basis import harmonic
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_io import digest,write_json


def run(root):
    out=root/'2017-order-set';read=lambda name:json.loads((out/name).read_text())
    corrections=read('corrections.json');recipe=read('recipe.json')
    assert digest(out/'corrections.json')==recipe['corrections_sha256']
    assert digest(out/'corrected-inputs.npz')==corrections['output_sha256']
    frozen=json.loads((root/'frozen-2017/frozen.json').read_text())['inputs'];preserved=[]
    for source in frozen:
        # The freeze manifest records the original byte-level file digest.
        expected=source.get('sha256',source.get('frozen_sha256'))
        if expected is None:raise ValueError('Freeze manifest missing file hash')
        actual=digest(source['frozen']);assert actual==expected,source['frozen']
        preserved.append({'path':source['frozen'],'sha256':actual})
    observed=read('observed.json')['rows'];variants=list(recipe['variants']);errors=[]
    with np.load(out/'fitted-support.npz') as support,np.load(out/'fitted-coefficients.npz') as coefficients,np.load(out/'bootstrap.npz') as boot:
        for rid in sorted(set(r['region'] for r in observed)):
            field={k:support[rid+'__'+k] for k in ('early','late','geometry','order','burst','axis_EN','row','col','clean')}
            e,l=field['early'],field['late'];dt=np.nan_to_num(l-e);h=np.nan_to_num(harmonic(e,l));g=field['geometry'];axis=field['axis_EN']
            for name in variants:
                key=rid+'__'+name+'__';cs=coefficients[key+'cs'];nu=coefficients[key+'nuisance'];oc=coefficients[key+'order_m']
                rec=next(r for r in observed if r['region']==rid and r['variant']==name);model=recipe['variants'][name][0]
                if model=='azimuth_direct':prediction=(dt*nu[:,0,None]+np.einsum('stj,j->st',h,cs))[...,None]
                elif model=='azimuth_along':prediction=(dt*nu[:,0,None]+np.einsum('stj,j->st',h,cs)*(g[:,1,:2]@axis)[:,None])[...,None]
                else:
                    prediction=dt[...,None]*np.einsum('sci,si->sc',g[...,:2],nu[:,:2])[:,None]
                    prediction+=np.einsum('stj,j->st',h,cs)[...,None]*(g[...,:2]@axis)[:,None]
                    if model=='residual_vertical':prediction+=np.einsum('stj,sj->st',vertical_columns(e,l),nu[:,2:])[...,None]*g[:,None,:,2]
                if len(oc):
                    mapping={burst:b for burst,b in zip(rec['burst_indices'],oc)}
                    bias=np.array([mapping[int(b)] for b in field['burst']])
                    prediction[...,-1]+=field['order']*bias[:,None]
                errors.append(float(np.max(abs(prediction-coefficients[key+'prediction']))))
                assert np.isfinite(boot[rid][:,variants.index(name)]).all()
    assert max(errors)<1e-10
    rows=read('recovery.json')['rows'];assert len(rows)==168 and all(r['draws']==200 and r['failures']==0 for r in rows)
    nested=read('nested-calibration.json')['rows'];assert len(nested)==240
    assert sum(r['successful_draws'] for r in nested)==47760
    assert sum(r['bootstrap_failures'] for r in nested)==0
    for rid in sorted(set(r['region'] for r in observed)):
        for case in ('null','benchmark'):
            indices=sorted(r['draw'] for r in nested if r['region']==rid and r['case']==case)
            assert indices==list(range(20)) if case=='null' else indices==list(range(0,200,10))
    # Confirm actual raw/corrected calibration support and grid compatibility.
    with h5py.File(root/'frozen-2017/displacements_raw.nc') as raw,h5py.File(root/'frozen-2017/displacements_swath_corrected.nc') as corrected,np.load(out/'fitted-support.npz') as support:
        for key in ('x','y','source_burst_index','source_observation_index'):assert np.array_equal(raw[key][:],corrected[key][:])
        for rid in sorted(set(r['region'] for r in observed)):
            rr=support[rid+'__row'];cc=support[rid+'__col'];good=support[rid+'__clean'];r0,r1=int(min(rr)),int(max(rr))+1;c0,c1=int(min(cc)),int(max(cc))+1
            delta=(raw['azimuth_displacement_m'][:,r0:r1,c0:c1]-corrected['azimuth_displacement_m'][:,r0:r1,c0:c1])[:,rr-r0,cc-c0].T
            assert np.isfinite(delta[good]).all()
    checkpoint={'recipe_sha256':digest(out/'recipe.json'),'observed_sha256':digest(out/'observed.json'),'correction_cache_sha256':digest(out/'corrected-inputs.npz')}
    if (out/'recovery-inputs.json').exists():assert read('recovery-inputs.json')==checkpoint
    else:write_json(out/'recovery-inputs.json',checkpoint)
    files=[Path(p) for p in ['src/glacier_offsets/velocity_systematics.py','tests/test_velocity_systematics.py',
        'tools/prepare_order_set_experiment.py','tools/run_order_set_pilot.py','tools/scan_order_set_studentized.py',
        'tools/check_order_set_sensitivities.py','tools/summarize_order_set_pilot.py','tools/verify_order_set_pilot.py']]
    artifacts=[p for p in sorted(out.iterdir()) if p.is_file() and p.name!='verification.json' and p.suffix!='.log']
    result={'status':'complete exploratory experiment; scientific acceptance gate fails uncertainty calibration',
        'preserved_frozen_inputs':preserved,'observed_fits':36,'recovery_rows':168,'synthetic_estimator_fits':33600,
        'nested_regional_datasets':240,'nested_resamples':47760,'fit_failures':0,
        'independent_prediction_max_difference_m':max(errors),'current_code_sha256':{str(p):digest(p) for p in files},
        'artifacts_sha256':{p.name:digest(p) for p in artifacts},'tests':'52 temporal inference tests passed in ice; seven unrelated native matcher tests require isce3',
        'note':'Generator cache-reuse and checkpoint guards were added after numerical runs; model equations and stored numerical arrays unchanged.'}
    write_json(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('current_code_sha256','artifacts_sha256','preserved_frozen_inputs')},indent=2))

if __name__=='__main__':run(Path('.runtime/velocity-recovery-larsen'))
