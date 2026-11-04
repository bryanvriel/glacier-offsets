"""Verify independent-source fits, native mapping, and yearwise ROI diagnostics."""
import argparse
import json
from pathlib import Path
import h5py
import h5netcdf
import numpy as np
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_basis import harmonic
from glacier_offsets.velocity_vertical import vertical_columns
from glacier_offsets.velocity_pixel import screen_years,solve
from glacier_offsets.velocity_pixel_config import parameters_from_recipe
from glacier_offsets.velocity_systematics import OrderRegionalDesign,robust_background
from glacier_offsets.velocity_quality import flow_axis
from .velocity_pixel_inversion import get_field
from .velocity_pixel_runtime import auxiliary_path,verify_code
from .velocity_pixel_mixed import check_mixed_support


def regional_axis(field,good):
    # Preserve the regional pilot's order-removed axis calculation without
    # depending on an unpackaged tool script.
    _,bias=robust_background(field,np.ones(good.shape[1],bool))
    data=field['data'].copy();data[...,1]-=field['order']*bias[:,None]
    return flow_axis({**field,'data':data},good)


def verify(out,regional_support=None):
    support_file=auxiliary_path(out,'regional_support',regional_support)
    metadata=json.loads((out/'inputs.json').read_text());recipe=json.loads((out/'recipe.json').read_text())
    parameters=parameters_from_recipe(recipe)
    check_mixed_support(out,recipe)
    completion=json.loads((out/'fit-completion.json').read_text())
    if digest(out/'recipe.json')!=completion['recipe_sha256'] or digest(out/'source_coefficients.h5')!=completion['output_sha256']:
        raise ValueError('Stored fit/recipe changed')
    for item in metadata['inputs']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Input package changed')
        with h5py.File(item['path']) as source:
            for key in ('range_displacement_m','azimuth_displacement_m'):
                if source[key].attrs['units']!='m':raise ValueError('Wrong displacement units')
            for key in ('earlier_burst_time','later_burst_time'):
                if source[key].attrs['units']!='seconds since 1970-01-01 00:00:00':raise ValueError('Wrong endpoint time units')
                if source[key].attrs['calendar'] not in ('proleptic_gregorian','standard','gregorian'):raise ValueError('Wrong endpoint calendar')
    verify_code(recipe['code_sha256'])
    rng=np.random.default_rng(12345);errors=[];rms_errors=[];region_rows=[]
    with h5py.File(out/'observations.h5') as h,h5py.File(out/'solid_earth_tides.h5') as body,h5py.File(out/'source_coefficients.h5') as fits:
        eligible=np.flatnonzero(fits['residual_vertical/estimable'][:]);selected=np.sort(rng.choice(eligible,min(512,len(eligible)),replace=False))
        if not len(selected):raise ValueError('No estimable sources to verify')
        cov=fits['covariance_by_year'][:];f=get_field(h,body,selected,parameters['screening']['remove_mixed_windows']);good,u,_,_,_=screen_years(f,parameters['screening'])
        original=fits['residual_vertical/coefficients'][selected];axis=fits['flow_axis_EN'][selected]
        repeated=solve(f,good,u,cov,cluster_days=parameters['uncertainty']['cluster_days'])
        errors.append(float(np.max(abs(repeated['coefficients']-original))))
        if errors[-1]>1e-9:raise ValueError('Source fit does not reproduce')
        dt=np.nan_to_num(f['late']-f['early']);year=f['year_index'];g=np.moveaxis(f['geometry_by_year'][year],0,1)
        v=np.stack([original[:,:2],original[:,2:4]],axis=0);velocity=np.moveaxis(v[year],0,1)
        predicted=dt[...,None]*np.einsum('stci,sti->stc',g[...,:2],velocity)
        predicted[...,1]+=f['order'][None]*original[:,4:6][:,year]
        predicted+=np.einsum('stj,sj->st',vertical_columns(f['early'],f['late']),original[:,6:10])[...,None]*g[...,2]
        predicted+=np.einsum('stj,sj->st',np.nan_to_num(harmonic(f['early'],f['late'])),original[:,-2:])[...,None]*np.einsum('stci,si->stc',g[...,:2],axis)
        residual=np.where(good[...,None],np.nan_to_num(f['data']-predicted),0.)
        rms=np.sqrt(np.sum(residual**2,axis=1)/good.sum(1)[:,None]);difference=float(np.max(abs(rms-fits['residual_vertical/rms_m'][selected])))
        if difference>1e-6:raise ValueError('Standalone radar prediction/RMS does not reproduce')
        rms_errors.append(difference)
        # Compare maps with source tables at representative points and copied cells.
        with h5netcdf.File(out/'coefficient_maps.nc','r') as maps:
            rr,cc=h['row'][selected],h['col'][selected]
            values=maps['residual_vertical_coefficients'][:][:,rr,cc].T
            if not np.array_equal(values,original,equal_nan=True):raise ValueError('Mapped source coefficients changed')
            back=h['map_source_index'][:];grid=maps['msf_along_amplitude'][:];expected=fits['residual_vertical/amplitude'][:][np.maximum(back,0)]
            if not np.array_equal(grid[back>=0],expected[back>=0],equal_nan=True):raise ValueError('Copied map-cell values differ from source fits')
        # The same frozen 2017 ROI sources provide descriptive, separate-year
        # regional fits. This is an agreement diagnostic, without new p-values.
        if support_file is not None:
            ids=h['source_id'][:]
            target_axes={}
            with np.load(support_file) as previous:
                for rid in ('target_1','target_2','target_3','target_4','control_1','control_2'):
                    source_ids=previous[rid+'__source_id'];index=np.searchsorted(ids,source_ids)
                    if np.any(index>=len(ids)) or not np.array_equal(ids[index],source_ids):raise ValueError('ROI source association changed')
                    order=np.argsort(index);index=index[order];field=get_field(h,body,index,parameters['screening']['remove_mixed_windows']);clean,_,_,_,_=screen_years(field,parameters['screening'])
                    for yr in (0,1):
                        take=field['year_index']==yr
                        local={k:field[k][:,take] for k in ('early','late','data','valid','peak')}
                        local.update(order=np.broadcast_to(field['order'][take],local['valid'].shape),geometry=field['geometry_by_year'][yr],
                            group=previous[rid+'__group'][order],burst=h['owner'][index],calendar_early=field['calendar_early'][take],calendar_late=field['calendar_late'][take])
                        mask=clean[:,take];direction=regional_axis(local,mask) if rid.startswith('target') else target_axes[yr]
                        if rid=='target_1':target_axes[yr]=direction
                        for model in ('azimuth_direct','residual_vertical'):
                            fitted=OrderRegionalDesign(local,cov[yr],model).fit(mask,direction)
                            cs=fitted['cs'];region_rows.append({'region':rid,'year':2017+yr,'model':model,
                                'amplitude_m_day':float(np.linalg.norm(cs)),'phase_deg':float(np.degrees(np.arctan2(cs[1],cs[0]))),
                                'order_m':fitted['order_m'],'active_sources':fitted['active_sources'],
                                'note':'separate-year regional descriptive fit on frozen source cohort; no bootstrap/p-value in this map audit'})
            write_json(out/'regional-year-check.json',{'rows':region_rows})
        else:
            write_json(out/'regional-year-check.json',{'rows':[],'status':'not run; no regional_support supplied'})
        print('Reconstruction max coefficient difference',max(errors),'radar RMS difference',max(rms_errors))
        for rid in ('target_1','target_2','target_3','target_4'):
            print(rid,[(r['year'],round(r['amplitude_m_day'],4),round(r['phase_deg'],1)) for r in region_rows if r['region']==rid and r['model']=='residual_vertical'])
    files=[p for p in out.iterdir() if p.is_file() and p.name not in ('verification.json',) and p.suffix!='.log']
    write_json(out/'verification.json',{'status':'complete exploratory pixelwise inversion',
        'inputs_unchanged':True,'sampled_sources':len(selected),'coefficient_replay_max_error':max(errors),
        'independent_prediction_rms_max_error_m':max(rms_errors),'native_copy_mapping_exact':True,
        'parameters':parameters,'regional_support_sha256':None if support_file is None else digest(support_file),'artifacts_sha256':{p.name:digest(p) for p in files},
        'verification_code_sha256':digest(Path(__file__))})

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--regional-support',type=Path)
    a=p.parse_args(argv);verify(a.output.resolve(),a.regional_support)


if __name__=='__main__':main()
