"""Full independent-source 2017+2018 Msf and residual-vertical inversion.

Run in ice after input extraction and PySolid correction preparation. This is
an exploratory coefficient map, not a production significance/detection gate.
"""
import argparse
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import shutil
import time
import yaml
import h5py
import numpy as np
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_pixel import screen_years,solve,coverage,JOINT_NAMES,AZIMUTH_NAMES
from glacier_offsets.velocity_pixel_config import load_config,validate_parameters
from .velocity_pixel_mixed import SUPPORT_FILE,prepare_mixed_support,selected_validity
from .velocity_pixel_runtime import NUMERICAL_MODULES,AUXILIARY_FILES,module_fingerprints,auxiliary_fingerprints


def stage_prepared_inputs(source,out,regions=None,regional_support=None):
    """Copy reusable caches and frozen weights, never fit products, into a new run."""
    names=('inputs.json','observations.h5','solid_earth_tides.json','solid_earth_tides.h5','noise.json')
    for name in names:
        if not (source/name).is_file():raise ValueError(f'Missing prepared input: {source/name}')
    metadata=json.loads((source/'inputs.json').read_text());tides=json.loads((source/'solid_earth_tides.json').read_text())
    if digest(source/'observations.h5')!=metadata['cache_sha256'] or digest(source/'solid_earth_tides.h5')!=tides['output_sha256']:
        raise ValueError('Prepared input/correction cache changed')
    copies={name:source/name for name in names}
    if (source/SUPPORT_FILE).is_file():copies[SUPPORT_FILE]=source/SUPPORT_FILE
    for key,override in {'regions':regions,'regional_support':regional_support}.items():
        candidate=Path(override) if override is not None else source/AUXILIARY_FILES[key]
        if override is not None and not candidate.is_file():raise ValueError(f'Missing {key}: {candidate}')
        if candidate.is_file():copies[AUXILIARY_FILES[key]]=candidate
    out.mkdir(parents=True,exist_ok=True)
    for name,origin in copies.items():
        destination=out/name
        if destination.exists():
            if digest(destination)!=digest(origin):raise ValueError(f'Existing {name} differs; use a new output directory')
        else:
            temporary=out/(name+'.copying')
            shutil.copyfile(origin,temporary);temporary.replace(destination)


def get_field(h,body,index,remove_mixed_windows=False):
    field={k:h[k][index] for k in ('early','late','data','valid','peak')}
    field['valid']=selected_validity(h,index,remove_mixed_windows)
    field['data']=field['data']-body['set_ra_m'][index]
    field['valid'] &= np.isfinite(field['data']).all(-1)
    for k in ('order','year_index','calendar_early','calendar_late'):field[k]=h[k][:]
    geometry=h['geometry_by_file'][:][:,index]
    files=h['file_index'][:]
    by_year=[geometry[files[np.flatnonzero(field['year_index']==year)[0]]] for year in (0,1)]
    field['geometry_by_year']=np.array(by_year)
    return field


def noise(h,body,out):
    """Load prepared yearly covariances; inversion never retrains calibration."""
    record=out/'noise.json'
    if not record.is_file():
        raise ValueError('Missing prepared noise.json; supply frozen yearly control covariances in input_directory')
    covariance=np.array(json.loads(record.read_text())['covariance_by_year'])
    if covariance.shape!=(2,2,2) or not np.isfinite(covariance).all():
        raise ValueError('noise.json must contain two finite 2x2 yearly covariances')
    if not np.allclose(covariance,covariance.swapaxes(1,2)):
        raise ValueError('Noise covariance must be symmetric')
    try:np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:raise ValueError('Noise covariance must be positive definite') from error
    return covariance


def run(out,batch=512,parameters=None,config_record=None):
    parameters=validate_parameters(parameters)
    if isinstance(batch,bool) or not isinstance(batch,int) or batch<1:raise ValueError('batch size must be a positive integer')
    if (out/'source_coefficients.partial.h5').exists():raise ValueError('Existing partial checkpoint; inspect it or use a new output directory')
    source=out/'observations.h5';earth=out/'solid_earth_tides.h5'
    inputs=json.loads((out/'inputs.json').read_text());tides=json.loads((out/'solid_earth_tides.json').read_text())
    if digest(source)!=inputs['cache_sha256'] or digest(earth)!=tides['output_sha256']:raise ValueError('Source/correction cache changed')
    recipe_file=out/'recipe.json';fingerprints=module_fingerprints(NUMERICAL_MODULES)
    if recipe_file.exists():
        old=json.loads(recipe_file.read_text())
        if old.get('parameters')!=parameters or old.get('code_sha256')!=fingerprints:
            raise ValueError('Inversion recipe changed; use new output directory')
    mixed_policy=prepare_mixed_support(out,parameters['screening']['remove_mixed_windows'])
    with h5py.File(source) as h,h5py.File(earth) as body:
        cov=noise(h,body,out)
    s=parameters['screening'];c=parameters['coverage'];cluster_days=parameters['uncertainty']['cluster_days']
    recipe={'inputs_sha256':digest(out/'inputs.json'),'observations_sha256':inputs['cache_sha256'],'solid_earth_sha256':tides['output_sha256'],
        'parameters':parameters,'noise_sha256':digest(out/'noise.json'),'mixed_window_policy':mixed_policy,
        'model':'independent local along-flow Msf C/S + shared-across-years source vertical M2/O1 height C/S; year-specific EN secular and azimuth order offsets',
        'comparator':'direct satellite-azimuth Msf with year-specific secular/order',
        'screen':f"peak>={s['min_peak']:g} and acquisition-purged order-aware RA residual<={s['max_range_residual_m']:g}/{s['max_azimuth_residual_m']:g} m, trained separately within each year",
        'min_fit':f"{c['min_pairs_per_year']} clean pairs each year, >={c['min_pairs_total']} total, >={c['min_span_days']:g} days, >={c['min_phase_bins']}/{c['phase_bins']} Msf phase bins; full model rank",
        'standard_coverage':f"same and >={c['min_pairs_standard']} total clean pairs",
        'flow_axis':'robust order-removed EN velocity per year, equally averaged; fixed during each point fit',
        'covariance':'year-specific H1 control RA covariance; constant across pairs inside year',
        'diagnostic_uncertainty':f'pointwise {cluster_days:g}-day calendar-cluster sandwich, CR1; no spatial resampling; unvalidated as calibrated pixelwise inference',
        'phase_origin':'2000-01-01 00:00 UTC; v=C*cos(omega*tau)+S*sin(omega*tau); phi=atan2(S,C)',
        'interpretation':'exploratory coefficients, no detection/FDR or physical-Msf certification',
        'code_sha256':fingerprints,'auxiliary_sha256':auxiliary_fingerprints(out)}
    if recipe_file.exists() and json.loads(recipe_file.read_text())!=recipe:raise ValueError('Inversion recipe changed; use new output directory')
    if not recipe_file.exists():write_json(recipe_file,recipe)
    if config_record is not None and not (out/'inversion-config.resolved.yaml').exists():
        (out/'inversion-config.resolved.yaml').write_text(yaml.safe_dump(config_record,sort_keys=False))
    with h5py.File(source) as h,h5py.File(earth) as body:
        ns=len(h['source_id']);nt=len(h['order']);destination=out/'source_coefficients.h5'
        if destination.exists():
            with h5py.File(destination) as complete:
                if complete.attrs.get('status')=='complete' and complete.attrs.get('recipe_sha256')==digest(recipe_file):
                    logging.info('Reusing completed source fits');return
            raise ValueError('Existing incomplete output; inspect partial checkpoint')
        tmp=out/'source_coefficients.partial.h5';start=time.time()
        with h5py.File(tmp,'w') as result:
            result.attrs.update(status='writing',recipe_sha256=digest(recipe_file),uncertainty=recipe['diagnostic_uncertainty'])
            result.create_dataset('source_id',data=h['source_id'][:]);result.create_dataset('covariance_by_year',data=cov)
            specs={'clean_pairs_by_year':((ns,2),'i2'),'span_days':((ns,),'f4'),'phase_bins':((ns,),'i1'),
                'fit_support':((ns,),'?'),'standard_coverage':((ns,),'?'),'flow_axis_EN':((ns,2),'f8'),
                'robust_velocity_EN_by_year':((ns,2,2),'f8'),'mean_flow_speed':((ns,),'f4'),'year_direction_change_deg':((ns,),'f4'),
                'azimuth_flow_projection_by_year':((ns,2),'f4')}
            for name,(shape,dtype) in specs.items():result.create_dataset(name,shape,dtype=dtype,compression='gzip',compression_opts=2)
            for model,p in [('residual_vertical',12),('azimuth_direct',6)]:
                group=result.create_group(model);group.attrs['parameter_names']=json.dumps(list(JOINT_NAMES if p==12 else AZIMUTH_NAMES))
                for name,shape,dtype in [('coefficients',(ns,p),'f8'),('cs',(ns,2),'f8'),('amplitude',(ns,),'f4'),('phase_deg',(ns,),'f4'),
                    ('amplitude_sigma',(ns,),'f4'),('phase_sigma_deg',(ns,),'f4'),('coefficient_covariance',(ns,2,2),'f8'),
                    ('target_condition',(ns,),'f4'),('rank',(ns,),'i1'),('estimable',(ns,),'?'),('rms_m',(ns,2 if p==12 else 1),'f4'),
                    ('partial_r2',(ns,),'f4'),('calendar_clusters',(ns,),'i4')]:
                    group.create_dataset(name,shape,dtype=dtype,compression='gzip',compression_opts=2)
            for begin in range(0,ns,batch):
                sl=slice(begin,min(begin+batch,ns));f=get_field(h,body,sl,s['remove_mixed_windows'])
                good,axis,velocities,speed,angle=screen_years(f,s);support=coverage(f,good,c)
                values={'clean_pairs_by_year':support['counts'],'span_days':support['span_days'],'phase_bins':support['phase_bins'],
                    'fit_support':support['fit_support'],'standard_coverage':support['standard_coverage'],'flow_axis_EN':axis,
                    'robust_velocity_EN_by_year':np.moveaxis(velocities,0,1),'mean_flow_speed':speed,'year_direction_change_deg':angle,
                    'azimuth_flow_projection_by_year':np.stack([np.einsum('si,si->s',g[:,1,:2],axis) for g in f['geometry_by_year']],axis=1)}
                for name,value in values.items():result[name][sl]=value
                for model in ('residual_vertical','azimuth_direct'):
                    fitted=solve(f,good,axis,cov,model,cluster_days=cluster_days);estimable=fitted['estimable']&support['fit_support']
                    fitted['estimable']=estimable
                    for name in result[model]:
                        value=fitted[name]
                        if np.issubdtype(value.dtype,np.floating):value=np.where(estimable.reshape((-1,)+(1,)*(value.ndim-1)),value,np.nan)
                        result[model][name][sl]=value
                if begin//batch%10==0:logging.info('Fitted %d/%d unique sources (%.1fs)',sl.stop,ns,time.time()-start)
            result.attrs.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat())
        tmp.replace(destination)
        write_json(out/'fit-completion.json',{'sources':ns,'pairs':nt,'elapsed_seconds':time.time()-start,'output_sha256':digest(destination),'recipe_sha256':digest(recipe_file),'noise_sha256':digest(out/'noise.json')})
        logging.info('Completed %d independent source fits for each model in %.1f seconds',ns,time.time()-start)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path,nargs='?',help='Existing prepared directory (legacy defaults); omit when using --config')
    p.add_argument('--config',type=Path,help='Separate pixel-inversion YAML; paths are relative to the YAML')
    p.add_argument('--batch-size',type=int,help='Override execution.batch_size; no effect on scientific parameters')
    p.add_argument('--remove-mixed-windows',action=argparse.BooleanOptionalAction,default=None,
                   help='Override mixed-source exclusion; default is false unless enabled in YAML')
    a=p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    try:
        if a.config:
            if a.output is not None:p.error('Specify either --config or a positional output directory')
            config=load_config(a.config)
            if a.remove_mixed_windows is not None:config['screening']['remove_mixed_windows']=a.remove_mixed_windows
            if a.batch_size is not None:
                if a.batch_size<1:raise ValueError('batch size must be positive')
                config['execution']['batch_size']=a.batch_size
            out=Path(config['output']);stage_prepared_inputs(Path(config['input_directory']),out,config['regions'],config['regional_support'])
            parameters={k:config[k] for k in ('screening','coverage','uncertainty','display')}
            run(out,config['execution']['batch_size'],parameters,config)
        else:
            if a.output is None:p.error('Specify --config CONFIG.yaml or a prepared output directory')
            parameters=None if a.remove_mixed_windows is None else {'screening':{'remove_mixed_windows':a.remove_mixed_windows}}
            run(a.output,512 if a.batch_size is None else a.batch_size,parameters)
    except ValueError as error:
        p.error(str(error))


if __name__=='__main__':main()
