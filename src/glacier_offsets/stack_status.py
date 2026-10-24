"""Read-only completion inspection; no raw input discovery or checksum reads."""
from copy import deepcopy
import json
from pathlib import Path

from .products import identity
from .stack_stages import STAGES, STAGE_DESCRIPTIONS

SYMBOLS = {'complete': '✓', 'pending': '○', 'stale': '~', 'incomplete': '!',
           'blocked': '−', 'unknown': '?', 'not_applicable': '−'}


def state(status, reason):
    return {'status': status, 'reason': reason}


def read_json(path):
    if not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected a JSON object')
        return value, None
    except (OSError, ValueError) as exc:
        return {}, str(exc)


def recorded(manifest, key, paths, expected=None, released=None):
    if not manifest.get(key):
        return state('pending', 'No completion identity recorded')
    if expected is not None and manifest[key] != expected:
        return state('stale', 'Saved identity differs from current settings or dependencies')
    if not paths:
        return state('incomplete', 'No output products recorded')
    released_paths = {str(Path(p).resolve()) for p in (released or {})}
    missing = next((str(p) for p in paths if not Path(p).is_file() and
                    str(Path(p).resolve()) not in released_paths), None)
    if missing:
        return state('incomplete', f'Missing output: {missing}')
    return state('complete', 'Completion identity and key output files present' if not released else
                 'Assembly complete; merged SLC cache intentionally released')


def dependent(upstream, check):
    if upstream['status'] != 'complete':
        return state('blocked', f"Prerequisite is {upstream['status']}: {upstream['reason']}")
    return check()


def reference_status(cfg, frame, error):
    from .stack_storage import METHOD
    from .stack_mosaic import policy
    if error:
        return state('unknown', error)
    result = recorded(frame, 'geometry_identity', frame.get('products', {}))
    if result['status'] != 'complete':
        return result
    spec = frame.get('specification', {})
    if identity(spec) != frame['geometry_identity']:
        return state('incomplete', 'Reference specification does not match its completion identity')
    if spec.get('method') != METHOD or spec.get('ownership_policy') != policy(cfg):
        return state('stale', 'Reference ownership policy changed; use a new output root or stack rebuild-layout for the supported policy update')
    anchor = spec.get('anchor', {})
    requested = cfg['stack'].get('reference_date')
    if requested is not None and str(requested) != anchor.get('reference_date'):
        return state('stale', 'Configured reference date differs from the pinned reference')
    selection = anchor.get('selection', {})
    for key in ('polarization', 'burst_id', 'subswaths'):
        if key in selection and cfg['inputs'].get(key) != selection[key]:
            return state('stale', f'Reference selection changed: inputs.{key}')
    dem = spec.get('inputs', {}).get('dem', {}).get('path')
    if dem and Path(cfg['inputs'].get('dem', '')).resolve() != Path(dem).resolve():
        return state('stale', 'Configured DEM path differs from the pinned reference')
    old_aoi = (selection.get('aoi') or {}).get('specification')
    if old_aoi != cfg.get('aoi'):
        return state('stale', 'Configured AOI differs from the pinned reference')
    return result


def matching_status(cfg, manifest, root):
    from .ampcor import matching_specification
    current = deepcopy(manifest)
    if cfg['gross_offsets'].get('mode') == 'velocity':
        saved = manifest.get('matching', {}).get('velocity_provenance')
        if not saved:
            return state('unknown', 'Saved velocity-source fingerprints unavailable')
        settings = cfg['gross_offsets']['velocity']
        current['velocity_provenance'] = {k: settings[k] | {'sha256': saved[k]['sha256']} for k in ('vx', 'vy')}
    native_engine = None
    if cfg['matching'].get('engine') == 'isce2':
        native_engine = manifest.get('matching', {}).get('native_engine')
        if native_engine is None:
            return state('unknown', 'Saved ISCE2 native provenance unavailable; status never starts native processes')
    expected = identity(matching_specification(cfg, current, native_engine=native_engine))
    return recorded(manifest, 'match_identity', [root/'radar/offsets.h5'], expected)


def pair_status(cfg, manifest, root, upstream):
    from .geocode import geocoding_specification
    result = {}
    checks = {
        'match': lambda: matching_status(cfg, manifest, root),
        'geocode': lambda: recorded(manifest, 'geocode_identity', [root/'geo'/p for p in
            ('geocoding.json', 'range_displacement_m.tif', 'azimuth_displacement_m.tif',
             'range_velocity_m_per_day.tif', 'azimuth_velocity_m_per_day.tif',
             'valid_mask.tif', 'quality_valid_mask.tif', 'center_geometry.h5')],
            identity(geocoding_specification(cfg, manifest))),
        'diagnostics': lambda: recorded(manifest, 'diagnostics_identity', [root/'diagnostics/report.json'],
            identity({'match': manifest['match_identity'], 'geocode': manifest['geocode_identity'],
                      'method': 'stack-diagnostics-v1'})),
    }
    for stage, check in checks.items():
        try:
            upstream = dependent(upstream, check)
        except (KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            upstream = state('unknown', f'Cannot compare saved metadata: {exc}')
        result[stage] = upstream
    return result


def inspect_stack(cfg):
    from .stack_storage import METHOD, assembly_specification
    from .storage import released_files
    root = Path(cfg['output']['directory'])
    catalog, catalog_error = read_json(root/'stack_manifest.json')
    frame, error = read_json(root/'geometry/manifest.json')
    reference = reference_status(cfg, frame, error)
    report = {'output_directory': str(root), 'inspection':
              'Saved active plan, completion identities and key output existence; no input discovery or checksums.',
              'reference': reference, 'acquisitions': {}, 'pairs': {}, 'stages': {}, 'notes': []}
    if catalog_error:
        report['notes'].append(f'Cannot read stack catalog: {catalog_error}; showing saved directories')
    if not catalog:
        report['notes'].append('No saved stack catalog; showing existing acquisition/pair directories')
    acquisitions = (catalog.get('acquisitions', {}) if catalog else
                    {p.parent.name: {} for p in sorted((root/'acquisitions').glob('*/manifest.json'))})
    pairs = (catalog.get('pairs', {}) if catalog else
             {p.parent.name: {} for p in sorted((root/'pairs').glob('*/manifest.json'))})
    dates = {}
    for day, entry in sorted(acquisitions.items()):
        if not entry.get('active', True):
            continue
        path = root/'acquisitions'/day
        m, err = read_json(path/'manifest.json')
        dates[day] = m
        def alignment():
            if err:
                return state('unknown', err)
            result = recorded(m, 'alignment_identity', m.get('alignment_products', m.get('products', {})))
            spec = m.get('specification', {})
            if result['status'] == 'complete' and (spec.get('method') != METHOD or
                    spec.get('geometry') != frame.get('geometry_identity') or identity(spec) != m['alignment_identity']):
                return state('stale', 'Alignment dependencies differ from the saved reference')
            return result
        aligned = dependent(reference, alignment)
        released = {p: digest for p, digest in released_files(m, 'assembly_identity', m.get('assembly_identity')).items()
                    if m.get('products', {}).get(p) == digest}
        assembled = dependent(aligned, lambda: recorded(m, 'assembly_identity',
            [*m.get('products', {}), path/'slc.tif', path/'valid.tif', path/'slc.tif.msk'],
            identity(assembly_specification(frame, m)), released=released))
        if released:
            report['notes'].append(f'{day}: merged SLC cache intentionally released; aligned coreg products retained.')
        report['acquisitions'][day] = {'coregister': aligned, 'assemble': assembled}
    for key, entry in sorted(pairs.items()):
        if not entry.get('active', True):
            continue
        path = root/'pairs'/key
        m, err = read_json(path/'manifest.json')
        upstream = reference
        pair_dates = m.get('stack_pair', {}).get('acquisition_dates', key.split('_'))
        for day in pair_dates:
            dependency = report['acquisitions'].get(day, {}).get('assemble', state('pending', f'Acquisition {day} missing'))
            if dependency['status'] != 'complete':
                upstream = dependency
                break
        if upstream['status'] == 'complete' and m:
            spec = m.get('coreg_specification', {}).get('stack', {})
            expected_dates = [dates.get(d, {}).get('assembly_identity') for d in pair_dates]
            if spec.get('geometry') != frame.get('geometry_identity') or spec.get('dates') != expected_dates:
                upstream = state('stale', 'Pair dependencies differ from the prepared acquisitions')
        if err:
            upstream = state('unknown', err)
        report['pairs'][key] = pair_status(cfg, m, path, upstream)
    for stage in STAGES:
        group = report['acquisitions'] if stage in ('coregister', 'assemble') else report['pairs']
        units = [e[stage] for e in group.values()]
        if stage == 'coregister':
            units = [reference, *units]
        count = sum(s['status'] == 'complete' for s in units)
        status = ('complete' if count == len(units) else next(
            (s for s in ('unknown', 'incomplete', 'stale', 'blocked') if any(u['status'] == s for u in units)), 'pending'))
        if not units:
            status = 'not_applicable' if catalog else 'pending'
        report['stages'][stage] = {'status': status, 'completed': count, 'total': len(units),
                                  'description': STAGE_DESCRIPTIONS[stage]}
    return report


def format_status(report):
    lines = [f"Stack: {report['output_directory']}", report['inspection'], '']
    for stage, entry in report['stages'].items():
        lines.append(f"{SYMBOLS[entry['status']]} {stage:12} {entry['completed']}/{entry['total']} complete  {entry['description']}")
    lines.append(f"\nReference geometry: {SYMBOLS[report['reference']['status']]}")
    for title, group in [('Acquisitions', 'acquisitions'), ('Pairs', 'pairs')]:
        if report[group]:
            lines.append(f'\n{title}:')
        for name, stages in report[group].items():
            labels = '  '.join(f'{SYMBOLS[s["status"]]} {stage}' for stage, s in stages.items())
            lines.append(f'  {name}  {labels}')
    problems = [('reference', 'coregister', report['reference'])]
    problems += [(name, stage, s) for group in ('acquisitions', 'pairs')
                 for name, stages in report[group].items() for stage, s in stages.items()]
    for name, stage, s in problems:
        if s['status'] not in ('complete', 'pending', 'blocked'):
            lines.append(f"  {SYMBOLS[s['status']]} {name} / {stage}: {s['reason']}")
    lines.extend(report['notes'])
    lines.append('\n✓ complete  ○ pending  ~ stale  ! incomplete  − blocked/no work  ? unknown')
    return '\n'.join(lines)
