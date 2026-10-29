"""Freeze a scoped spacecraft calibration from verified overlap fit results."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile

import yaml

from .products import identity
from .swath_correction_config import METHOD, MATCHING_KEYS, load_calibration
from .swath_experiment_io import METHOD as EXPERIMENT_METHOD, Inputs, reference_metadata


def _read(inputs, path, expected=None):
    inputs.file(path, expected)
    return json.loads(Path(path).read_text())


def _record(record, experiment):
    if (record.get('status') != 'complete' or
            record.get('identity') != identity(record['specification']) or
            record['specification'].get('experiment') != experiment):
        raise ValueError('Calibration fit or dependency is incomplete or inconsistent')


def _validation(analysis):
    scores = [analysis['validation_score_intervals'][kind]['spacecraft'] for kind in
              ('leave_acquisition_out', 'leave_burst_group_out')]
    if all(s.get('descriptive_interval95_px') and s['descriptive_interval95_px'][1] < 0 for s in scores):
        return 'supported improvement within the declared descriptive validation'
    if all(s['mae_change_vs_baseline_px'] >= 0 for s in scores):
        return 'no improvement'
    return 'inconclusive'


def _publish(path, content, inputs, *, force=False):
    path = Path(path).resolve()
    if str(path) in inputs.revisions:
        raise ValueError('Calibration output would replace a consumed source file')
    if path.exists():
        if path.read_bytes() == content:
            inputs.unchanged()
            return
        if not force:
            raise ValueError(f'Calibration output exists with different contents: {path}; choose another path or use --force')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.'+path.name, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        inputs.unchanged()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_calibration(output, destination=None, *, cfg=None, force=False):
    """Export current or legacy results, without SAFE discovery or native processing.

    Recorded implementation hashes remain provenance, not a requirement to have
    the old source checkout installed. Only experiment-owned paths are relocated.
    The caller holds the results lock and, when using stack metadata, its lock.
    """
    root = Path(output).resolve(strict=True)
    target = Path(destination).resolve() if destination else root/'calibration.json'
    protected = ('analysis', 'measurements', 'physics', 'previews', '.cache')
    if any(target.is_relative_to(root/name) for name in protected):
        raise ValueError('Calibration output must be outside retained result products')
    if target.name in ('plan.json', 'fit.json', 'summary.json', 'audit.json', 'physics_status.json',
                       'measure_stage.json', 'measurement_status.json', 'preview_status.json', 'pilot.json'):
        raise ValueError('Calibration output would replace a reserved result file')
    if cfg is not None and target.is_relative_to(Path(cfg['output']['directory']).resolve()):
        raise ValueError('Calibration output must be outside the production stack')
    inputs = Inputs()
    plan = _read(inputs, root/'plan.json')
    spec = plan['specification']
    if spec.get('method') != EXPERIMENT_METHOD:
        raise ValueError('Unsupported swath estimation result format')
    experiment = identity({k: v for k, v in spec.items() if k != 'source_hashes'})
    if plan.get('status') != 'complete' or plan['identity'] != experiment:
        raise ValueError('Calibration plan identity is inconsistent')
    fit = _read(inputs, root/'fit.json')
    _record(fit, experiment)
    physics = _read(inputs, root/'physics_status.json', fit['specification']['physics_sha256'])
    if physics.get('status') != 'complete':
        raise ValueError('Complete the physics stage before exporting calibration')
    measurements = _read(inputs, root/'measure_stage.json')
    if measurements.get('status') != 'complete' or measurements.get('pilot_only'):
        raise ValueError('Complete all planned measurements before exporting calibration')

    # Old reports contain server-absolute paths. Establish their recorded result
    # root from the registered primary analysis; never relocate external inputs.
    primary_paths = [Path(p) for p in fit['products'] if Path(p).parts[-2:] == ('analysis', 'primary.json')]
    if len(primary_paths) != 1:
        raise ValueError('Fit must register exactly one primary analysis')
    recorded_root = primary_paths[0].parent.parent
    for product, digest in fit['products'].items():
        try:
            relative = Path(product).relative_to(recorded_root)
        except ValueError as exc:
            raise ValueError('Fit product is outside the recorded results directory') from exc
        if '..' in relative.parts:
            raise ValueError('Invalid recorded fit product path')
        inputs.file(root/relative, digest)
    for kind, directory in (('measurements', 'measurements'), ('physics_reports', 'physics')):
        for pair, digest in fit['specification'].get(kind, {}).items():
            if Path(pair).name != pair:
                raise ValueError('Invalid recorded pair identifier')
            _record(_read(inputs, root/directory/pair/'report.json', digest), experiment)
    analysis = _read(inputs, root/'analysis/primary.json')
    model = analysis['models']['spacecraft']
    covariance = model.get('covariance')
    if (not analysis.get('rows') or model.get('rank') != 1 or
            model.get('parameter_names') != ['beta_B_minus_A_px'] or
            covariance is None or analysis['uncertainty']['insufficient_groups']):
        raise ValueError('Primary spacecraft coefficient or its uncertainty is not identifiable')
    beta, variance = model['parameters'][0], covariance[0][0]
    if not all(math.isfinite(v) for v in (beta, variance)) or variance < 0:
        raise ValueError('Primary coefficient or covariance is invalid')

    reference = plan.get('reference_metadata')
    if cfg is not None:
        stack = Path(cfg['output']['directory'])
        current = reference_metadata(_read(inputs, stack/'geometry/manifest.json'),
                                     _read(inputs, stack/'stack_manifest.json'))
        if reference is not None and current != reference:
            raise ValueError('Configured stack reference differs from calibration results')
        reference = current
    if reference is None:
        raise ValueError('Legacy results need CONFIG.yaml to read retained reference and track metadata')
    if (reference['geometry_identity'] != spec['geometry_identity'] or
            reference['layout_identity'] != spec['layout_identity']):
        raise ValueError('Calibration results and reference geometry/layout differ')
    source_cfg = spec['config']
    production = Path((cfg or source_cfg)['output']['directory']).resolve()
    if (root == production or root.is_relative_to(production) or production.is_relative_to(root) or
            target.is_relative_to(production)):
        raise ValueError('Calibration results and output must be separate from the production stack')
    matching = {k: source_cfg['matching'].get(k) for k in MATCHING_KEYS}
    dates = spec['dates']
    if dates != sorted(fit['metadata']) or not dates:
        raise ValueError('Fit acquisition dates differ from calibration plan')
    scope = {k: reference[k] for k in ('geometry_identity', 'relative_orbit', 'direction', 'polarization')}
    scope.update(processing=source_cfg['stack']['processing'], swath_merge=source_cfg['stack']['swath_merge'],
                 matching=matching, start_date=min(dates), end_date=max(dates))
    cal = {'method': METHOD, 'units': 'original_reference_azimuth_pixels', 'allocation': 'symmetric',
           'swaths': [1, 2], 'beta_px': beta, 'beta_std_px': math.sqrt(variance),
           'reference_prf_hz': reference['reference_prf_hz'], 'scope': scope,
           'provenance': {
               'label': f'Spacecraft-order relative IW1/IW2 azimuth calibration, {min(dates)} to {max(dates)}',
               'estimate': 'all-data descriptive primary spacecraft coefficient',
               'population': plan['primary_selection'],
               'weighting': 'equal total weight per link; equal burst-pair-group weights within each link',
               'minimum_link_support': None,
               'validation_status': _validation(analysis),
               'validation_mae_change': analysis['validation_score_intervals'],
               'physical_cause': 'unidentified; empirical calibration, not a validated timing correction',
               'accuracy': 'Absolute accuracy and constant-swath extension beyond measured overlap remain unvalidated.',
               'uncertainty': 'Shared acquisition/source-burst multiplier descriptive standard deviation; excludes model error and absolute/common motion bias.',
               'training_dates': dates,
               'training_ipf_versions': sorted({v['ipf'] for v in fit['metadata'].values()}),
               'training_links': analysis['graph']['links'],
               'training_windows': sum(len(row['values']) for row in analysis['rows']),
               'geometry_reference_date': reference['reference_date'],
               'source_experiment_identity': experiment,
               'source_plan_sha256': inputs.file(root/'plan.json'),
               'source_fit_sha256': inputs.file(root/'fit.json'),
               'source_analysis_sha256': inputs.file(root/'analysis/primary.json')}}
    bundle = source_cfg.get('_reference_bundle', {}).get('data', {})
    if bundle:
        cal['provenance']['reference_bundle_fingerprint'] = bundle['fingerprint']
    content = (json.dumps(cal, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    # Validate the exact serialized candidate against the real correction loader
    # before publication, including applicability to a supplied current config.
    validation_cfg = deepcopy(cfg or source_cfg)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        import hashlib
        digest = hashlib.sha256(content).hexdigest()
        validation_cfg['stack']['swath_correction'] = {
            'enabled': True, 'calibration': str(temporary), 'calibration_sha256': digest, 'pairs_directory': None}
        load_calibration(validation_cfg)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    inputs.unchanged()
    _publish(target, content, inputs, force=force)
    _publish(Path(str(target)+'.sha256'), (digest+'  '+target.name+'\n').encode(), inputs, force=force)
    settings = {'stack': {'swath_correction': {
        'enabled': True, 'calibration': str(target), 'calibration_sha256': digest}}}
    settings_path = root/'calibration-settings.yaml'
    _publish(settings_path, yaml.safe_dump(settings, sort_keys=False).encode(), inputs, force=force)
    return {'status': 'complete', 'calibration': str(target), 'calibration_sha256': digest,
            'validation_status': cal['provenance']['validation_status'],
            'settings_file': str(settings_path), 'settings': settings}
