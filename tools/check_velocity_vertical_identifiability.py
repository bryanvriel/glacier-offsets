"""Check vertical M2/O1 plus horizontal Msf at actual 2017 source geometries.

Design/information audit only: no data coefficients, tide corrections, recovery
claims or production gates are changed. Vertical parameters are height harmonics
in metres; horizontal parameters are instantaneous velocity harmonics in m/day.
"""
import json
from pathlib import Path
import numpy as np
from glacier_offsets.velocity_basis import harmonic
from glacier_offsets.velocity_inspect import M2_PERIOD,O1_PERIOD
from glacier_offsets.velocity_quality import screen,crossfit_residual
from glacier_offsets.velocity_io import digest,write_json


def height_difference(e,l,period):
    w=2*np.pi/period
    return np.c_[np.cos(w*l)-np.cos(w*e),np.sin(w*l)-np.sin(w*e)]


def matrix(e,l,g,u,mode,ideal_alias=False):
    h=harmonic(e,l);dt=l-e;n=len(e)
    background=dt[:,None,None]*g[None,:,:2]
    if mode=='along_flow':
        horizontal=h[:,None,:]*(g[:,:2]@u)[None,:,None]
    else:
        horizontal=np.einsum('ci,tj->tcij',g[:,:2],h).reshape(n,2,4)
    m2=h if ideal_alias else height_difference(e,l,M2_PERIOD)
    vertical=np.concatenate([m2[:,None,:]*g[None,:,2,None],
        height_difference(e,l,O1_PERIOD)[:,None,:]*g[None,:,2,None]],axis=-1)
    return background,horizontal,vertical


def marginalize(a,b):
    u,s,_=np.linalg.svd(b,full_matrices=False)
    keep=s>s[0]*1e-12
    return a-u[:,keep]@(u[:,keep].T@a)


def audit(root):
    prior=root/'2017-quality-regional';out=root/'vertical-identifiability';out.mkdir(exist_ok=True)
    metadata=json.loads((prior/'observations.json').read_text())
    regional=json.loads((prior/'regional_fits.json').read_text())['rows']
    noise=json.loads((prior/'clean_noise.json').read_text())['pooled']['covariance_ra_m2']
    factor=np.linalg.cholesky(noise);rows=[]
    with np.load(prior/'observations.npz') as cache:
        for rid in dict.fromkeys(r['region'] for r in regional):
            field={key:cache[rid+'__'+key] for key in metadata['keys']}
            good=screen(field,.08,crossfit_residual(field))
            u=np.array(next(r['direction_en'] for r in regional if r['region']==rid and r['channel']=='joint'))
            for i,g in enumerate(field['geometry']):
                take=good[i]
                if take.sum()<20:continue
                e,l=field['early'][i,take],field['late'][i,take]
                for mode in ('along_flow','free_en'):
                    bg,h,v=matrix(e,l,g,u,mode)
                    def white(a):return np.linalg.solve(factor[None],a).reshape(-1,a.shape[-1])
                    b,hh,vv=map(white,(bg,h,v))
                    a=np.c_[b,hh,vv];norm=a/np.linalg.norm(a,axis=0)
                    singular=np.linalg.svd(norm,compute_uv=False)
                    marginal=marginalize(hh,np.c_[b,vv]);base=marginalize(hh,b)
                    normalized=marginal/np.linalg.norm(hh,axis=0)
                    marginal_singular=np.linalg.svd(normalized,compute_uv=False)
                    ideal=np.concatenate(matrix(e,l,g,u,mode,ideal_alias=True),axis=-1)
                    ideal=white(ideal);ideal/=np.linalg.norm(ideal,axis=0)
                    ideal_s=np.linalg.svd(ideal,compute_uv=False)
                    sigma=np.linalg.inv(marginal.T@marginal)
                    sigma0=np.linalg.inv(base.T@base)
                    projection=np.eye(2) if mode=='along_flow' else np.array([[u[0],0,u[1],0],[0,u[0],0,u[1]]])
                    scale=np.sqrt(np.trace(projection@sigma@projection.T)/np.trace(projection@sigma0@projection.T))
                    gu=np.c_[g[:,:2]@u,g[:,2]]
                    gn=np.linalg.solve(factor,gu);gn/=np.linalg.norm(gn,axis=0)
                    rows.append({'region':rid,'source_id':int(field['source_id'][i]),'mode':mode,'pairs':int(take.sum()),
                        'column_count':a.shape[-1],'numerical_rank_rtol_1e10':int(np.sum(singular>singular[0]*1e-10)),
                        'normalized_global_condition':float(singular[0]/singular[-1]),
                        'ideal_M2_Msf_alias_rank_rtol_1e10':int(np.sum(ideal_s>ideal_s[0]*1e-10)),
                        'marginal_horizontal_normalized_singular_values':marginal_singular,
                        'rms_along_CS_formal_sigma_inflation_vs_fixed_vertical':float(scale),
                        'along_vertical_geometry_condition':float(np.linalg.cond(gn)),
                        'range_along_sensitivity':float(g[0,:2]@u),'azimuth_along_sensitivity':float(g[1,:2]@u),
                        'range_up_sensitivity':float(g[0,2]),'azimuth_up_sensitivity':float(g[1,2])})
    summary=[]
    for rid in dict.fromkeys(r['region'] for r in rows):
        for mode in ('along_flow','free_en'):
            subset=[r for r in rows if r['region']==rid and r['mode']==mode]
            summary.append({'region':rid,'mode':mode,'sources':len(subset),
                'normalized_condition_quantiles':np.quantile([r['normalized_global_condition'] for r in subset],[0,.5,1]),
                'formal_horizontal_sigma_inflation_quantiles':np.quantile([r['rms_along_CS_formal_sigma_inflation_vs_fixed_vertical'] for r in subset],[0,.5,1]),
                'numerical_rank_values':sorted({r['numerical_rank_rtol_1e10'] for r in subset}),
                'ideal_alias_rank_values':sorted({r['ideal_M2_Msf_alias_rank_rtol_1e10'] for r in subset}),
                'geometry_sensitivities_median':{key:float(np.median([r[key] for r in subset])) for key in (
                    'range_along_sensitivity','azimuth_along_sensitivity','range_up_sensitivity','azimuth_up_sensitivity')}})
    # Controlled ideal-alias check verifies the geometric null-space statement.
    assert all(r['ideal_alias_rank_values']==([8] if r['mode']=='along_flow' else [8]) for r in summary)
    assert all(r['normalized_condition_quantiles'][1]<10 for r in summary if r['mode']=='along_flow')
    write_json(out/'design_audit.json',{'scope':'design/information audit only; no measured joint-vertical fit or recovery validation',
        'model':'free EN secular velocity; along-flow or free EN Msf velocity; U M2/O1 displacement heights; no secular U velocity',
        'noise_scope':'working first-half clean-control RA covariance, independent observations; not dependent-noise confidence intervals',
        'source_sha256':digest(__file__),'inputs':{str(p):digest(p) for p in (prior/'observations.npz',prior/'regional_fits.json',prior/'clean_noise.json')},
        'summary':summary,'rows':rows})
    for r in summary:print(r)

if __name__=='__main__':audit(Path('.runtime/velocity-recovery-larsen').resolve())
