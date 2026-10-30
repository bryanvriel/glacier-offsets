"""Configuration-driven Msf velocity experiment, independent of ISCE3."""
import argparse
from copy import copy
import json
import logging
from pathlib import Path
import sys

import h5netcdf
import numpy as np
import yaml
from .velocity_basis import utc_days, evaluate
from .velocity_fit import diagnostics, false_discovery
from .velocity_io import Collection, digest, fingerprint, serializable, write_json, writer_lock, save_fit
from .velocity_inspect import propose, sample_points
from .velocity_pilot import (candidate_bank, learn_noise, select_models, paired_bootstrap, recovery,
                             acceptance, fit_point, tune_shortlist, bootstrap_intervals)

DEFAULTS = {'regions': None, 'start': None, 'end': None, 'calibration_validation': None,
            'sampling': {'points_per_region': 3},
            'pilot': {'bootstrap_draws': 1000, 'validation_draws': 200, 'tuning_draws': 100, 'seed': 12345},
            'models': {'backgrounds': ['constant','linear','spline_l2','spline_l1'],
                       'weights': ['constant','constant_robust','quality_robust','quality_rule'],
                       'penalties': np.logspace(-4,4,9).tolist()},
            'map': {'tile_size': 16, 'min_pairs_one_year': 40, 'min_days_one_year': 270,
                    'min_pairs_two_year': 60, 'min_days_two_year': 540,
                    'phase_coverage': .75, 'fdr': .05, 'max_phase_std_degrees': 20}}


def load_settings(path):
    path = Path(path).resolve()
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict) or set(raw)-set(DEFAULTS)-{'inputs','output'}:
        raise ValueError('Fit YAML contains unknown fields or is not a mapping')
    cfg = {}
    for key,value in DEFAULTS.items():
        supplied = raw.get(key, value)
        if isinstance(value,dict):
            if not isinstance(supplied,dict) or set(supplied)-set(value):
                raise ValueError(f'Unknown or invalid {key} options')
            supplied = value | supplied
        cfg[key] = supplied
    if not isinstance(raw.get('inputs'),list) or not raw['inputs'] or not all(isinstance(p,str) for p in raw['inputs']):
        raise ValueError('inputs must be a nonempty list of horizontal netCDF paths')
    def resolve(value):
        p = Path(value).expanduser()
        return str((path.parent/p).resolve())
    cfg['inputs'] = [resolve(p) for p in raw['inputs']]
    if not isinstance(raw.get('output'),str):
        raise ValueError('output must be a directory path')
    cfg['output'] = resolve(raw['output'])
    for key in ('regions','calibration_validation'):
        if cfg[key] is not None:
            cfg[key] = resolve(cfg[key])
    if cfg['start'] is not None and cfg['end'] is not None and utc_days(str(cfg['end'])) <= utc_days(str(cfg['start'])):
        raise ValueError('end must follow start; end is exclusive UTC')
    for key in ('start','end'):
        if cfg[key] is not None:
            cfg[key] = str(cfg[key])
            utc_days(cfg[key])
    for key in ('bootstrap_draws','validation_draws','tuning_draws','seed'):
        if type(cfg['pilot'][key]) is not int or cfg['pilot'][key] < 1:
            raise ValueError(f'pilot.{key} must be a positive integer')
    for key,value in cfg['sampling'].items():
        if type(value) is not int or value < 1:
            raise ValueError(f'sampling.{key} must be positive')
    if type(cfg['map']['tile_size']) is not int or not 1 <= cfg['map']['tile_size'] <= 128:
        raise ValueError('map.tile_size must be between 1 and 128')
    for key,value in cfg['map'].items():
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'map.{key} must be finite and positive')
    if cfg['map']['phase_coverage'] > 1 or cfg['map']['fdr'] > .05:
        raise ValueError('phase coverage must be <=1 and FDR <=0.05')
    backgrounds,weights,penalties = (cfg['models'][k] for k in ('backgrounds','weights','penalties'))
    if not backgrounds or set(backgrounds)-set(DEFAULTS['models']['backgrounds']):
        raise ValueError('Invalid model backgrounds')
    if not weights or set(weights)-set(DEFAULTS['models']['weights']):
        raise ValueError('Invalid weighting bank')
    if not penalties or not np.isfinite(penalties).all() or min(penalties) < 0:
        raise ValueError('Penalties must be finite and nonnegative')
    return cfg


def implementation():
    names = ('basis','fit','io','inspect','pilot','recovery','calibration')
    return {f'velocity_{name}.py':digest(Path(__file__).with_name(f'velocity_{name}.py')) for name in names}


def reviewed_regions(cfg, collection):
    if cfg['regions'] is None:
        raise ValueError('Pilot requires a reviewed regions JSON; run --stage inspect first')
    with open(cfg['regions']) as stream:
        record = json.load(stream)
    if record.get('EPSG') != int(collection.handles[0].attrs['EPSG']):
        raise ValueError('Region EPSG does not match the netCDF grid')
    regions = record.get('regions',[])
    if len({r['id'] for r in regions}) != len(regions):
        raise ValueError('Region IDs must be unique')
    for region in regions:
        if region.get('role') not in ('target','control') or region.get('reviewed') is not True or not region.get('review_notes','').strip():
            raise ValueError('Each target/control needs reviewed=true and documented review_notes; proposals are not control truth')
        bounds = region.get('bounds',[])
        if len(bounds)!=4 or not np.isfinite(bounds).all() or bounds[0]>=bounds[2] or bounds[1]>=bounds[3]:
            raise ValueError('Region bounds must be west,south,east,north in projected metres')
    if sum(r['role']=='target' for r in regions)<4 or sum(r['role']=='control' for r in regions)<2:
        raise ValueError('At least four reviewed ice targets and two reviewed bedrock controls are required')
    return regions


def run_recipe(cfg, collection):
    return {'settings':cfg, 'source_fingerprints':[{k:v for k,v in s.items() if k!='correction_history'} for s in collection.sources],
            'correction_history': [s['correction_history'] for s in collection.sources],
            'implementation':implementation(), 'regions_sha256':digest(cfg['regions']) if cfg['regions'] else None,
            'calibration_validation_sha256':digest(cfg['calibration_validation']) if cfg['calibration_validation'] else None}


def calibration_gate(cfg, collection):
    """Require scientific validation evidence scoped to exact input corrections."""
    if cfg['calibration_validation'] is None:
        return ['Spacecraft/swath calibration needs split-year and held-out validation evidence']
    with open(cfg['calibration_validation']) as stream:
        report = json.load(stream)
    reasons = []
    if report.get('schema') != 'glacier-offsets-calibration-validation-v1' or report.get('passed') is not True:
        reasons.append('Calibration validation report is unsupported or failed')
    if report.get('input_sha256') != [s['sha256'] for s in collection.sources]:
        reasons.append('Calibration validation is not bound to these exact netCDF inputs')
    for protocol in ('held_out_acquisitions','held_out_burst_groups','split_year_estimates'):
        if not report.get(protocol):
            reasons.append('Calibration evidence missing '+protocol)
    for evidence in report.get('evidence',[]):
        path = Path(evidence['path'])
        if not path.is_absolute():
            path = Path(cfg['calibration_validation']).parent/path
        if not path.is_file() or digest(path)!=evidence.get('sha256'):
            reasons.append('Calibration evidence checksum mismatch: '+str(path))
    if not report.get('evidence'):
        reasons.append('Independent same-ground overlap evidence is required')
    if report.get('cross_boundary_policy') not in ('validated','excluded'):
        reasons.append('Cross-boundary correction policy is missing')
    if report.get('cross_boundary_policy')=='excluded':
        for pair in collection.pairs:
            finite = pair[5]
            if any(str(np.datetime64('2000-01-01')+np.timedelta64(int(t),'D'))[:4] !=
                   str(np.datetime64('2000-01-01')+np.timedelta64(int(u),'D'))[:4] for t,u in zip(pair[3][finite],pair[4][finite])):
                reasons.append('An unvalidated cross-year pair remains in the inputs')
                break
    return reasons


def coverage_valid(result, cfg, two_year):
    options = cfg['map']
    suffix = 'two_year' if two_year else 'one_year'
    return (diagnostics(result)['identifiable'] and result['n_pairs'] >= options['min_pairs_'+suffix] and
            result['span_days'] >= options['min_days_'+suffix] and result['phase_coverage'] >= options['phase_coverage'])


def cv_numerical_recipe(cfg,recipe):
    """Predictive-fitting dependencies, independent of injection scenarios."""
    import hashlib
    import inspect
    from . import velocity_pilot
    names=('select_models','point_arrays','fit_point','control_residuals','learn_noise','candidate_bank')
    return {'sources':recipe['source_fingerprints'],'regions_sha256':recipe['regions_sha256'],
            'settings':{k:cfg[k] for k in ('start','end','sampling','models')},
            'files':{k:recipe['implementation'][k] for k in ('velocity_basis.py','velocity_fit.py','velocity_io.py','velocity_inspect.py')},
            'functions':{k:hashlib.sha256(inspect.getsource(getattr(velocity_pilot,k)).encode()).hexdigest() for k in names}}


class PilotProgress:
    """Reuse only checksummed stages from the exact scientific pilot recipe."""
    def __init__(self,directory,recipe,cv_identity=None):
        self.directory=Path(directory)
        self.path=self.directory/'pilot_progress.json'
        key=fingerprint(recipe)
        saved=json.loads(self.path.read_text()) if self.path.exists() else {}
        self.previous_recipe=saved.get('recipe_identity')
        self.cv_identity=cv_identity
        self.state={'recipe_identity':key,'stages':saved.get('stages',{})}

    def read(self,name):
        record=self.state['stages'].get(name)
        path=self.directory/name
        matching=bool(record and record.get('recipe_identity',self.previous_recipe)==self.state['recipe_identity'])
        if name=='cross_validation.json' and self.cv_identity is not None and record:
            matching=matching or record.get('numerical_identity')==self.cv_identity
        if matching and path.is_file() and digest(path)==record['sha256']:
            logging.info('Reusing verified pilot stage %s',name)
            return json.loads(path.read_text())
        return None

    def save(self,name,value):
        path=self.directory/name
        write_json(path,value)
        self.state['stages'][name]={'sha256':digest(path),'recipe_identity':self.state['recipe_identity']}
        if name=='cross_validation.json' and self.cv_identity is not None:
            self.state['stages'][name]['numerical_identity']=self.cv_identity
        write_json(self.path,self.state)


def pilot(cfg, collection):
    regions = reviewed_regions(cfg, collection)
    points = sample_points(collection, regions, cfg['sampling']['points_per_region'])
    if set(p['region'] for p in points) != set(r['id'] for r in regions):
        raise ValueError('At least one unique, covered 3 km source block is required in every region')
    directory = Path(cfg['output'])
    recipe=run_recipe(cfg,collection)
    progress=PilotProgress(directory,recipe,cv_identity=fingerprint(cv_numerical_recipe(cfg,recipe)))
    write_json(directory/'sampling.json',[{k:p[k] for k in ('region','role','source_id','row','col','spatial_block')} for p in points])
    noise,residuals = learn_noise(points)
    write_json(directory/'noise.json',noise)
    early = collection.selected_early
    late = collection.selected_late
    bank = candidate_bank(early,late,**cfg['models'])
    cached=progress.read('cross_validation.json')
    if cached is None:
        comparisons,ledger = select_models(points,bank)
        progress.save('cross_validation.json',{'models':comparisons,'folds':ledger,'pair_ids':[p[2] for p in collection.pairs]})
    else:
        comparisons,ledger=cached['models'],cached['folds']
    logging.info('Predictive selection complete; starting independent tuning recovery')
    tuning=progress.read('tuning_recovery.json')
    if tuning is None:
        attempts=progress.read('tuning_progress.json') or {'attempts':[]}
        attempts['save']=lambda rows:progress.save('tuning_progress.json',{'attempts':rows})
        selected,tuning = tune_shortlist(points,noise,comparisons,residuals,cfg['pilot']['tuning_draws'],cfg['pilot']['seed'],checkpoint=attempts)
        tuning['selected']=selected
        progress.save('tuning_recovery.json',tuning)
    else:
        selected=tuning['selected']
    validation=progress.read('validation_recovery.json')
    if validation is None:
        validation = recovery(points,noise,selected,residuals,draws=cfg['pilot']['validation_draws'],seed=cfg['pilot']['seed'])
        progress.save('validation_recovery.json',validation)
    accepted = acceptance(validation)
    bootstrap=progress.read('paired_bootstrap.json')
    if bootstrap is None:
        bootstrap = paired_bootstrap(points,noise,selected,draws=cfg['pilot']['bootstrap_draws'],seed=cfg['pilot']['seed'])
        progress.save('paired_bootstrap.json',bootstrap)
    write_json(directory/'paired_intervals.json',bootstrap_intervals(bootstrap))
    two_year = collection.selected_span_days >= 540
    results = {m:{} for m in selected}
    annual = []
    for point in points:
        for model,candidate in selected.items():
            common_direction = None
            for period,start,end in [('joint',None,None),('2017','2017-01-01','2018-01-01'),('2018','2018-01-01','2019-01-01')]:
                take = point['valid'].copy()
                if start:
                    take &= (point['early']>=utc_days(start)) & (point['late']<utc_days(end))
                try:
                    result = fit_point(point,noise,candidate,model,take,direction=common_direction)
                except (ValueError,RuntimeError,np.linalg.LinAlgError):
                    continue
                summary = diagnostics(result)
                if period=='joint':
                    common_direction = result['direction']
                annual.append({'region':point['region'],'source_id':point['source_id'],'model':model,'period':period,
                               'along_coefficients':summary['along_coefficients'],'along_covariance':summary['along_covariance'],
                               'direction_en':result['direction'],
                               'target_coefficients':result['coefficients'][result['target_indices']],
                               'n_pairs':result['n_pairs'],'span_days':result['span_days']})
                if period=='joint':
                    result['fit_valid'] = coverage_valid(result,cfg,two_year)
                    results[model][(point['row'],point['col'])] = result
    for model,values in results.items():
        save_fit(directory/f'pilot_{model}.nc',collection,values,model,recipe)
    write_json(directory/'year_comparison.json',annual)
    _daily_plots(directory,results,points)
    reasons = calibration_gate(cfg,collection)+accepted['failures']
    if not tuning['passed']:
        reasons.append('No candidate passed tuning recovery within one predictive standard error')
    if cfg['pilot']['bootstrap_draws']<1000:
        reasons.append('Fewer than 1000 paired block-resampling draws')
    if cfg['models'] != DEFAULTS['models']:
        reasons.append('The full prescribed background/weight/penalty bank was not evaluated')
    report = {'recipe_identity':fingerprint(recipe),'recipe':recipe,'selected':selected,'noise':noise,
              'acceptance':accepted,'map_allowed':not reasons,'map_blockers':reasons,'sampling':[{
                  k:p[k] for k in ('region','role','source_id','row','col','spatial_block')} for p in points],
              'scientific_interpretation':'Msf-frequency motion conditional on correction and nuisance-signal assumptions',
              'uncertainty_scope':'model-conditional covariance in netCDF; paired block draws in paired_bootstrap.json; correction/model sensitivity separate'}
    write_json(directory/'pilot.json',report)
    lines = ['# Msf pilot', '', f'Map gate: {"passed" if report["map_allowed"] else "blocked"}.', '',
             'Results describe Msf-frequency motion conditional on correction and nuisance-signal assumptions.', '',
             '## Map blockers', '']+[f'- {reason}' for reason in reasons]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    collection.check_revisions()
    return report


def _daily_plots(directory,results,points):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for point in points:
        if point['role']!='target':
            continue
        fig,axes = plt.subplots(2,1,figsize=(11,6),sharex=True,constrained_layout=True)
        good = point['valid']
        early,late = point['early'][good],point['late'][good]
        days = np.arange(np.floor(early.min()),np.ceil(late.max())+1)
        curves = {'UTC_days_since_2000':days}
        for model,values in results.items():
            result = values.get((point['row'],point['col']))
            if result is None:
                continue
            velocity = evaluate(result['coefficients'],result['basis'],model,result['direction'],days)
            curves[model] = velocity
            for axis,ax in enumerate(axes):
                ax.plot(days,velocity[:,axis],label=model)
        for axis,ax in enumerate(axes):
            ax.scatter((early+late)/2,point['data'][good,axis]/(late-early),s=8,color='.4',label='pair average')
            ax.set_ylabel(('East','North')[axis]+' (m/day)')
            ax.legend()
        axes[0].set_title(point['region']+f' / source {point["source_id"]}')
        axes[1].set_xlabel('UTC elapsed days since 2000-01-01')
        fig.savefig(directory/f'daily_{point["region"]}_{point["source_id"]}.png',dpi=130)
        plt.close(fig)
        np.savez_compressed(directory/f'daily_{point["region"]}_{point["source_id"]}.npz',**curves)


def map_stage(cfg,collection):
    """Resume checksummed tiles only after the exact pilot recipe passes."""
    reviewed_regions(cfg,collection)
    directory = Path(cfg['output'])
    pilot_path = directory/'pilot.json'
    if not pilot_path.is_file():
        raise ValueError('Run a reviewed, validated pilot before mapping')
    report = json.loads(pilot_path.read_text())
    recipe = run_recipe(cfg,collection)
    if report.get('recipe_identity')!=fingerprint(recipe) or not report.get('map_allowed'):
        raise ValueError('Mapping blocked: stale/failed pilot or unvalidated calibration; see pilot.json')
    reasons = calibration_gate(cfg,collection)
    if reasons:
        raise ValueError('; '.join(reasons))
    selected,noise = report['selected'],report['noise']
    two_year = collection.selected_span_days>=540
    checkpoint_path = directory/'tiles.json'
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {'recipe_identity':fingerprint(recipe),'tiles':{}}
    if checkpoint['recipe_identity']!=fingerprint(recipe):
        raise ValueError('Map checkpoint belongs to a different experiment; choose a new output directory')
    for tile_id,section in enumerate(collection.sections(cfg['map']['tile_size'])):
        key = str(tile_id)
        saved = checkpoint['tiles'].get(key,{})
        if saved and all(Path(p).is_file() and digest(p)==sha for p,sha in saved['files'].items()):
            continue
        logging.info('Fitting velocity tile %s',tile_id+1)
        tile = collection.tile(section)
        local = copy(collection)
        local.x,local.y = collection.x[section[1]],collection.y[section[0]]
        local.association = collection.association[section]
        results = {m:{} for m in selected}
        cache = {}
        for row,col in np.ndindex(local.association.shape):
            source = int(local.association[row,col])
            if source < 0:
                continue
            if source not in cache:
                point = {k:v[:,row,col] for k,v in tile.items() if k not in ('inverse','geometry')}
                point['inverse'] = tile['inverse'][row,col]
                cache[source] = {}
                for model,candidate in selected.items():
                    try:
                        result = fit_point(point,noise,candidate,model)
                        result['fit_valid'] = coverage_valid(result,cfg,two_year)
                        cache[source][model] = result
                    except (ValueError,RuntimeError,np.linalg.LinAlgError):
                        pass
            for model,result in cache[source].items():
                results[model][(row,col)] = result
        files = {}
        for model,values in results.items():
            if values:
                path = directory/'tiles'/f'{tile_id:06d}_{model}.nc'
                save_fit(path,local,values,model,recipe)
                files[str(path)] = digest(path)
        checkpoint['tiles'][key] = {'section':[[s.start,s.stop] for s in section],'files':files}
        write_json(checkpoint_path,checkpoint)
    _combine_maps(directory,collection,checkpoint,selected,cfg)
    collection.check_revisions()
    write_json(directory/'map_manifest.json',{'status':'complete','recipe_identity':fingerprint(recipe),
        'products':{str(directory/f'map_{m}.nc'):digest(directory/f'map_{m}.nc') for m in selected}})
    return {'status':'complete','models':list(selected),'tiles':len(checkpoint['tiles'])}


def _combine_maps(directory,collection,checkpoint,selected,cfg):
    """Assemble tiles, then control FDR over unique sources across both models."""
    import os
    outputs = {}
    source = collection.association
    unique,first = np.unique(source.ravel(),return_index=True)
    first = first[unique>=0]
    tests = []
    for model in selected:
        files = [(record,next((p for p in record['files'] if p.endswith('_'+model+'.nc')),None)) for record in checkpoint['tiles'].values()]
        template = next((p for _,p in files if p),None)
        if template is None:
            raise ValueError('No estimable map tiles for '+model)
        temp = directory/f'.map_{model}.nc'
        with h5netcdf.File(template) as src,h5netcdf.File(temp,'w') as out:
            out.dimensions = {k:(collection.shape[0] if k=='y' else collection.shape[1] if k=='x' else len(v)) for k,v in src.dimensions.items()}
            out.attrs.update(dict(src.attrs))
            out.attrs['map_multiple_testing'] = 'Benjamini-Yekutieli; unique sources jointly across both parameterizations'
            for name,v in src.variables.items():
                options = {'fillvalue':np.nan,'compression':'gzip'} if v.dtype.kind=='f' else {}
                target = out.create_variable(name,v.dimensions,dtype=v.dtype,**options)
                target.attrs.update({k:val for k,val in v.attrs.items() if k!='_FillValue'})
                if name=='x':
                    target[:] = collection.x
                elif name=='y':
                    target[:] = collection.y
                elif name=='source_observation_index':
                    target[:] = source
                elif v.dimensions[-2:]!=('y','x'):
                    target[...] = v[...]
                elif name in ('fit_valid','detection_valid','phase_valid'):
                    target[:] = 0
            for record,path in files:
                if not path:
                    continue
                section = tuple(slice(*s) for s in record['section'])
                with h5netcdf.File(path) as tile:
                    for name,v in tile.variables.items():
                        if v.dimensions[-2:]==('y','x') and name!='source_observation_index':
                            out[name][(...,*section)] = v[...]
            p = out['p_value'][:].ravel()
            valid = out['fit_valid'][:].ravel().astype(bool)
            tests.append(np.where(valid[first],p[first],np.nan))
        outputs[model] = temp
    passing = false_discovery(np.concatenate(tests),cfg['map']['fdr'])
    for i,(model,path) in enumerate(outputs.items()):
        source_ids = source.ravel()[first]
        detected = source_ids[passing[i*len(first):(i+1)*len(first)]]
        with h5netcdf.File(path,'a') as h:
            detection = np.isin(source,detected) & (h['fit_valid'][:]==1)
            h['detection_valid'][:] = detection.astype('uint8')
            h['phase_valid'][:] = (detection & (h['phase_std'][:] <= np.deg2rad(cfg['map']['max_phase_std_degrees']))).astype('uint8')
            phase = h['phase'][:]
            phase[h['phase_valid'][:]!=1] = np.nan
            h['phase'][:] = phase
        collection.check_revisions()
        os.replace(path,directory/f'map_{model}.nc')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets fit-velocity')
    parser.add_argument('config')
    parser.add_argument('--stage',choices=('inspect','pilot','map'),required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    try:
        # Console/module entry points must also fail clearly if caller injected
        # the processing-only runtime before Python imported NumPy.
        runtime = Path(__file__).resolve().parents[2]/'.runtime'/'deps'
        if any(Path(p or '.').resolve()==runtime for p in sys.path):
            raise ValueError('Remove .runtime/deps from PYTHONPATH and use the ice environment for fitting')
        cfg = load_settings(args.config)
        if args.stage!='inspect' and 'spline_l1' in cfg['models']['backgrounds']:
            import importlib.util
            if importlib.util.find_spec('cvxopt') is None:
                raise ValueError('The full L1 bank requires cvxopt from the existing ice environment; select that interpreter')
        logging.info('Temporal runtime: %s; NumPy %s from %s',sys.executable,np.__version__,np.__file__)
        with writer_lock(cfg['output']),Collection(cfg['inputs'],cfg['start'],cfg['end']) as collection:
            if args.stage=='inspect':
                report = propose(collection,cfg['output'])
                write_json(Path(cfg['output'])/'inspection_recipe.json',run_recipe(cfg,collection))
                result = {'stage':'inspect','report':str(Path(cfg['output'])/'inspection.json'),
                          'regions_proposed':len(report['regions']),'missing_regions':report['missing_regions']}
            elif args.stage=='pilot':
                report = pilot(cfg,collection)
                result = {'stage':'pilot','map_allowed':report['map_allowed'],'map_blockers':report['map_blockers']}
            else:
                result = map_stage(cfg,collection)
        print(json.dumps(serializable(result),indent=2,allow_nan=False))
        return 0
    except KeyboardInterrupt:
        logging.error('Interrupted; completed velocity tiles and previous outputs are preserved')
        return 130
    except (ValueError,RuntimeError,OSError,KeyError,ImportError,np.linalg.LinAlgError) as exc:
        logging.error('%s',exc)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
