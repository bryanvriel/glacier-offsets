"""Sequential, resumable stack stages with independent acquisition/pair failures."""
from copy import deepcopy
from collections import Counter
import gc
import json
import logging
from pathlib import Path
import resource
import sys
import time

from .ampcor import match, matching_specification
from .diagnostics import diagnostics
from .environment import inspect_environment
from .geocode import geocode, geocoding_specification
from .inputs import select_pair
from .products import ChecksumCache, identity, write_json
from .stack import plan_stack, read_catalog, validate_stack
from .stack_pairs import common_map_grid, matching_template, prepared_pair
from .stack_stages import STAGES
from .stack_lock import stack_lock
from .stack_storage import (SourceHashes, alignment_verified, assemble_date, assembly_specification, date_specification, prepare_date,
                            prepare_geometry, read_manifest, reference_specification, release_date, verified)
from .storage import keep_merged_slcs, released_files

def invalidate_pair(root, stage='coregister'):
    manifest = read_manifest(root)
    if not manifest:
        return
    names = {'coregister': 'coreg_identity', 'match': 'match_identity', 'geocode': 'geocode_identity',
             'diagnostics': 'diagnostics_identity'}
    start = STAGES.index(stage)
    for downstream in STAGES[start:]:
        if downstream in names:
            manifest.pop(names[downstream], None)
        manifest.get('stages', {}).pop(downstream, None)
    write_json(Path(root)/'manifest.json', manifest)


def invalidate_date(root, alignment=False):
    manifest = read_manifest(root)
    if not manifest:
        return
    manifest.pop('assembly_identity', None)
    if alignment:
        manifest.pop('alignment_identity', None)
    manifest['status'] = 'stale'
    write_json(Path(root)/'manifest.json', manifest)


def execute_stack(cfg, command, force=False):
    with stack_lock(cfg['output']['directory'], create=True):
        return _execute_stack(cfg, command, force=force)


def _execute_stack(cfg, command, force=False):
    from .stack_isce2 import enabled, prepare_common_layout
    isce2 = enabled(cfg)
    invocation_started = time.perf_counter()
    if command not in (*STAGES, 'run'):
        raise ValueError(f'Unknown stack stage {command}')
    root = Path(cfg['output']['directory'])
    retain_slcs = keep_merged_slcs(cfg)
    old = read_catalog(root)
    plan = plan_stack(cfg, previous=old)
    # Native reference placement/DEM checks happen before any scientific mutation.
    validate_stack(cfg, plan, check_dates=False)
    if cfg['matching'].get('engine') == 'isce2':
        from .isce2_matching import provenance
        provenance(cfg)  # Fail before scientific mutation if the native runtime is unavailable.
    hashes = SourceHashes()
    checksums = ChecksumCache()
    software = inspect_environment()
    specification = reference_specification(cfg, plan, hashes, software)
    fingerprint = identity(specification)
    if old.get('reference_specification_identity') not in (None, fingerprint):
        raise ValueError('Pinned reference geometry inputs/software/support policy changed; use a new output root '
                         '(for the ownership-only update, use stack rebuild-layout)')
    report = deepcopy(plan.report)
    report['acquisitions'] = {a.date: dict(old.get('acquisitions', {}).get(a.date, {})) | {'active': True}
                              for a in plan.acquisitions}
    for day, entry in old.get('acquisitions', {}).items():
        if day not in report['acquisitions']:
            report['acquisitions'][day] = entry | {'active': False}
    for key, entry in report['pairs'].items():
        previous = old.get('pairs', {}).get(key, {})
        report['pairs'][key] = previous | entry | {'status': previous.get('status', 'planned')}
    for key, value in old.get('pairs', {}).items():
        if key not in report['pairs']:
            report['pairs'][key] = value | {'active': False}
    report.update(command=command, force=force, outcome='running')
    if 'anchor' not in old:
        report.pop('anchor', None)  # Pin only after complete reference preparation.
    for key in ('anchor', 'reference_specification_identity', 'geometry_identity', 'layout_identity', 'map_grid'):
        if key in old:
            report[key] = old[key]

    def save():
        write_json(root/'stack_manifest.json', report)

    def finish():
        scale = 1 if sys.platform == 'darwin' else 1024
        report['metrics'] = {'elapsed_seconds': time.perf_counter()-invocation_started,
            'product_checksum_files_hashed': checksums.files_hashed,
            'product_checksum_bytes_hashed': checksums.bytes_hashed,
            'product_checksum_cache_hits': checksums.hits,
            'parent_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale),
            'children_peak_rss_bytes': int(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*scale),
            'rss_scope': 'process lifetime high-water marks; child processes reported separately',
            'output_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())}
        save()

    def invalidate_pairs(date=None, stage='coregister'):
        for key, entry in report['pairs'].items():
            if date is None or date in (entry['earlier'], entry['later']):
                invalidate_pair(root/'pairs'/key, stage)
                entry.update(status='stale', stages=read_manifest(root/'pairs'/key).get('stages', {}))
                entry.pop('error', None)

    def invalidate_dates(alignment):
        for day, entry in report['acquisitions'].items():
            invalidate_date(root/'acquisitions'/day, alignment=alignment)
            entry.update(status='stale')

    if force:
        if command in ('coregister', 'run'):
            invalidate_pairs()
            invalidate_dates(alignment=True)
        elif command == 'assemble':
            invalidate_pairs()
            invalidate_dates(alignment=False)
        else:
            invalidate_pairs(stage=command)
    save()
    failed = False
    try:
        logging.info('Verifying stack reference products')
        reference_previous = read_manifest(root/'geometry')
        if not verified(reference_previous, 'geometry_identity', fingerprint, checksums=checksums):
            invalidate_pairs()
            save()
        frame = prepare_geometry(cfg, plan, specification, software,
                                 force=force and command in ('coregister', 'run'),
                                 process=command in ('coregister', 'run'), expected_layout=None if isce2 else old.get('layout_identity'),
                                 checksums=checksums)
        report.update(anchor=plan.anchor, reference_specification_identity=fingerprint,
                      geometry_identity=frame['geometry_identity'], reference_status='complete')
        report['layout_identity'] = frame['layout_identity']
        save()
    except KeyboardInterrupt:
        report.update(outcome='interrupted', reference_status='interrupted')
        finish()
        raise
    except Exception as exc:
        invalidate_pairs()
        for entry in report['acquisitions'].values():
            entry.update(status='blocked', error='Reference preparation failed')
        for entry in report['pairs'].values():
            if entry['active']:
                entry.update(status='blocked', error='Reference preparation failed')
        preflight = command not in ('coregister', 'run')
        report.update(outcome='preflight_failed' if preflight else 'failed',
                      reference_status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        logging.error('Stack reference: %s', exc)
        finish()
        return 2 if preflight else 1

    dates = {}
    reference_bursts = {b.id: b for b in plan.bursts}
    for acquisition_number, acquisition in enumerate(plan.acquisitions, 1):
        started = time.perf_counter()
        entry = report['acquisitions'][acquisition.date]
        entry.update(status='running', directory=str(root/'acquisitions'/acquisition.date))
        entry.pop('error', None)
        save()
        try:
            logging.info('Checking stack acquisition %s (%d/%d)', acquisition.date,
                         acquisition_number, len(plan.acquisitions))
            if acquisition.error:
                raise ValueError(acquisition.error)
            for key in reference_bursts.keys() & acquisition.bursts.keys():
                effective = deepcopy(cfg)
                effective['inputs']['burst_id'] = key
                select_pair(effective, [(reference_bursts[key], acquisition.bursts[key])])
            specification = date_specification(acquisition, frame, hashes)
            previous = read_manifest(root/'acquisitions'/acquisition.date)
            if not alignment_verified(previous, identity(specification), checksums=checksums):
                invalidate_pairs(acquisition.date)
                save()
            elif previous.get('assembly_identity') and not verified(previous, 'assembly_identity', previous['assembly_identity'],
                                                                   allow_released=True, checksums=checksums):
                invalidate_pairs(acquisition.date)
                invalidate_date(root/'acquisitions'/acquisition.date)
                save()
            aligned = prepare_date(cfg, plan, acquisition, frame, specification,
                                   force=force and command in ('coregister', 'run'),
                                   process=command in ('coregister', 'run'), checksums=checksums)
            if isce2:
                dates[acquisition.date] = aligned
            if command != 'coregister' and not isce2:
                assembly_identity = identity(assembly_specification(frame, aligned))
                logical = verified(aligned, 'assembly_identity', assembly_identity, allow_released=True, checksums=checksums)
                if not logical:
                    invalidate_pairs(acquisition.date)
                    save()
                need_slc = command == 'assemble' or (retain_slcs and command in ('match', 'run'))
                if logical and not (force and command in ('assemble', 'run')) and not need_slc:
                    dates[acquisition.date] = aligned
                elif command == 'run' and not retain_slcs:
                    # Only materialize a new/rebuilt merged date when a pair
                    # actually needs matching. This bounds the working set by
                    # the active pair graph instead of all acquisitions.
                    dates[acquisition.date] = deepcopy(aligned) | {
                        'assembly_identity': assembly_identity, 'assembly_pending': True, 'products': {},
                        'slc': str((root/'acquisitions'/acquisition.date/'slc.tif').resolve())}
                else:
                    dates[acquisition.date] = assemble_date(cfg, acquisition, frame, aligned,
                        force=force and command in ('assemble', 'run'),
                        process=command in ('assemble', 'run') or (logical and need_slc), checksums=checksums)
                entry.update(assembly_identity=dates[acquisition.date]['assembly_identity'],
                             valid_samples=dates[acquisition.date].get('valid_samples'),
                             coverage_fraction=dates[acquisition.date].get('coverage_fraction'))
            entry.update(status='complete', alignment_identity=aligned['alignment_identity'],
                         completed_stage='coregister' if command == 'coregister' or isce2 else 'assemble')
        except KeyboardInterrupt:
            invalidate_pairs(acquisition.date)
            entry['status'] = 'interrupted'
            report['outcome'] = 'interrupted'
            finish()
            raise
        except Exception as exc:
            failed = True
            invalidate_pairs(acquisition.date)
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            logging.error('Stack date %s: %s', acquisition.date, exc)
        finally:
            entry['elapsed_seconds'] = time.perf_counter()-started
            save()
            gc.collect()
    if isce2:
        # All aligned dates must be inspected before any date is merged. Never
        # silently shrink the common-validity inventory to successful dates.
        if failed:
            for entry in report['pairs'].values():
                if entry['active']:
                    entry.update(status='blocked', error='Common validity requires all active acquisitions to be aligned')
            report['outcome'] = 'failed'
            finish()
            return 1
        try:
            previous_layout = frame['layout_identity']
            logging.info('Preparing ISCE2 stack-common valid rectangles and overwrite layout')
            frame = prepare_common_layout(cfg, plan, frame, dates, checksums)
            if frame['layout_identity'] != previous_layout:
                invalidate_pairs()
            report.update(layout_identity=frame['layout_identity'],
                          common_validity=frame['common_validity'],
                          common_validity_identity=frame['common_validity_identity'])
            save()
            for acquisition in plan.acquisitions:
                day = acquisition.date
                aligned = dates[day]
                assembly_id = identity(assembly_specification(frame, aligned))
                logical = verified(aligned, 'assembly_identity', assembly_id, allow_released=True, checksums=checksums)
                if not logical:
                    invalidate_pairs(day)
                    if command == 'coregister':
                        invalidate_date(root/'acquisitions'/day)
                if command == 'coregister':
                    continue
                need_slc = command == 'assemble' or (retain_slcs and command in ('match', 'run'))
                if logical and not (force and command in ('assemble', 'run')) and not need_slc:
                    dates[day] = aligned
                elif command == 'run' and not retain_slcs:
                    dates[day] = deepcopy(aligned) | {'assembly_identity': assembly_id,
                        'assembly_pending': True, 'products': {},
                        'slc': str((root/'acquisitions'/day/'slc.tif').resolve())}
                else:
                    dates[day] = assemble_date(cfg, acquisition, frame, aligned,
                        force=force and command in ('assemble', 'run'),
                        process=command in ('assemble', 'run') or (logical and need_slc), checksums=checksums)
                report['acquisitions'][day].update(assembly_identity=assembly_id, completed_stage='assemble',
                    valid_samples=dates[day].get('valid_samples'), coverage_fraction=dates[day].get('coverage_fraction'))
            save()
        except KeyboardInterrupt:
            report['outcome'] = 'interrupted'
            finish()
            raise
        except Exception as exc:
            invalidate_pairs()
            report.update(outcome='failed', error=f'ISCE2 common validity/assembly: {type(exc).__name__}: {exc}')
            logging.error('%s', report['error'])
            finish()
            return 1
    if command in ('coregister', 'assemble'):
        for key, entry in report['pairs'].items():
            manifest = read_manifest(root/'pairs'/key)
            if manifest.get('coreg_identity') and manifest.get('match_identity'):
                # Reconcile downstream settings even when only preparation was requested.
                if cfg['gross_offsets']['mode'] == 'velocity':
                    velocity = cfg['gross_offsets']['velocity']
                    manifest['velocity_provenance'] = {k: velocity[k] | {'sha256': hashes(velocity[k]['path'])['sha256']}
                                                       for k in ('vx', 'vy')}
                if manifest['match_identity'] != identity(matching_specification(cfg, manifest)):
                    invalidate_pair(root/'pairs'/key, 'match')
                    entry['status'] = 'stale'
                elif manifest.get('geocode_identity') and manifest['geocode_identity'] != identity(geocoding_specification(cfg, manifest)):
                    invalidate_pair(root/'pairs'/key, 'geocode')
                    entry['status'] = 'stale'
        for key, entry in report['pairs'].items():
            if entry['active'] and any(report['acquisitions'][d]['status'] != 'complete'
                                       for d in (entry['earlier'], entry['later'])):
                entry['status'] = 'blocked'
        report['outcome'] = 'failed' if failed else 'complete'
        finish()
        return 1 if failed else 0

    template_failure_stage = 'match'
    try:
        template = matching_template(cfg, frame)
        specification = {'geometry_identity': frame['geometry_identity'], 'shape': template.shape,
                         'starts_line': template.starts_line.tolist(), 'starts_sample': template.starts_sample.tolist(),
                         'window': template.window, 'search': template.search, 'stride': template.stride}
        report['matching_template_identity'] = identity(specification)
        report['matching_template'] = specification
        template_failure_stage = 'geocode'
        target_grid = (frame.get('map_grid') if isce2 else None) or common_map_grid(cfg, frame)
        if isce2 and (target_grid['epsg'] != cfg['geocoding']['epsg'] or
                      target_grid['transform'][1] != cfg['geocoding']['spacing_m']):
            target_grid = common_map_grid(cfg, frame)
        report['map_grid'] = target_grid
        report['map_grid_identity'] = identity(target_grid)
        grid_path = root/'geometry/map_grid.json'
        if not grid_path.is_file() or json.loads(grid_path.read_text()) != target_grid:
            write_json(grid_path, target_grid)
        velocity = None
        if cfg['gross_offsets']['mode'] == 'velocity':
            template_failure_stage = 'match'
            settings = cfg['gross_offsets']['velocity']
            velocity = {key: settings[key] | {'sha256': hashes(settings[key]['path'])['sha256']}
                        for key in ('vx', 'vy')}
    except KeyboardInterrupt:
        report['outcome'] = 'interrupted'
        finish()
        raise
    except Exception as exc:
        invalidate_pairs(stage=template_failure_stage)
        report.update(outcome='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        logging.error('Stack matching/map template: %s', exc)
        finish()
        return 1

    by_date = {a.date: a for a in plan.acquisitions}
    remaining = Counter(day for pair in plan.pairs for day in pair)
    for earlier, later in plan.pairs:
        started = time.perf_counter()
        key = f'{earlier}_{later}'
        entry = report['pairs'][key]
        if earlier not in dates or later not in dates:
            entry.update(status='blocked', error='Acquisition preparation failed or prerequisites are incomplete')
            failed = True
            save()
            continue
        entry.update(status='running', directory=str(root/'pairs'/key))
        entry.pop('error', None)
        save()
        manifest = None
        directory = root/'pairs'/key
        active_stage = 'coregister'
        try:
            directory, manifest = prepared_pair(cfg, plan, frame, by_date[earlier], by_date[later], dates,
                                                 target_grid, velocity)
            effective = deepcopy(cfg)
            effective.pop('aoi', None)
            effective['inputs']['burst_id'] = 'stack'
            active_stage = 'match'
            if command in ('match', 'run'):
                current = (not force and manifest.get('match_identity') == identity(matching_specification(effective, manifest))
                           and (directory/'radar/offsets.h5').is_file())
                if not current:
                    for day in (earlier, later):
                        product = dates[day]
                        if (product.get('assembly_pending') or
                                released_files(product, 'assembly_identity', product['assembly_identity'])):
                            dates[day] = assemble_date(cfg, by_date[day], frame, product, checksums=checksums)
                            dates[day].pop('assembly_pending', None)
                            report['acquisitions'][day].update(completed_stage='assemble',
                                valid_samples=dates[day]['valid_samples'], coverage_fraction=dates[day]['coverage_fraction'])
                    logging.info('Matching stack pair %s with %s Ampcor on merged SLCs',
                                 key, effective['matching'].get('engine', 'isce3').upper())
                match(effective, directory, manifest, force=force, native_pair=reference_bursts)
            elif manifest.get('match_identity') != identity(matching_specification(effective, manifest)):
                invalidate_pair(directory, 'match')
                raise ValueError('Pair matching missing or changed; run stack match first')
            if (manifest.get('match_identity') == identity(matching_specification(effective, manifest)) and
                    (directory/'radar/offsets.h5').is_file()):
                # Cleanup is downstream of committed offsets. A disk error or
                # interruption here must not invalidate successful matching.
                active_stage = 'geocode'
                for day in (earlier, later):
                    remaining[day] -= 1
                    if remaining[day] == 0:
                        release_date(cfg, by_date[day], dates[day])
                manifest['slc_previews'] = {side: dates[day]['slc_preview']
                                           for side, day in zip(('reference', 'secondary'), (earlier, later))
                                           if 'slc_preview' in dates[day]}
                write_json(directory/'manifest.json', manifest)
            if manifest.get('geocode_identity') and manifest['geocode_identity'] != identity(geocoding_specification(effective, manifest)):
                manifest.pop('geocode_identity', None)
                manifest.pop('diagnostics_identity', None)
                for stage in ('geocode', 'diagnostics'):
                    manifest['stages'].pop(stage, None)
                write_json(directory/'manifest.json', manifest)
            if command in ('geocode', 'run'):
                active_stage = 'geocode'
                geocode(effective, directory, manifest, force=force)
            if command == 'diagnostics':
                active_stage = 'geocode'
                if manifest.get('geocode_identity') != identity(geocoding_specification(effective, manifest)):
                    invalidate_pair(directory, 'geocode')
                    raise ValueError('Pair geocoding missing or changed; run stack geocode first')
            if command in ('diagnostics', 'run'):
                active_stage = 'diagnostics'
                diagnostic_identity = identity({'match': manifest['match_identity'], 'geocode': manifest['geocode_identity'],
                                                'method': 'stack-diagnostics-v1'})
                if force or manifest.get('diagnostics_identity') != diagnostic_identity or not (directory/'diagnostics/report.json').is_file():
                    manifest['stages'].pop('diagnostics', None)
                    manifest.pop('diagnostics_identity', None)
                    write_json(directory/'manifest.json', manifest)
                    diagnostics(effective, directory, manifest)
                    manifest['diagnostics_identity'] = diagnostic_identity
                    write_json(directory/'manifest.json', manifest)
            entry.update(status='complete', stages=manifest['stages'],
                         matching_coverage=manifest.get('matching_summary', {}),
                         map_coverage=manifest.get('geocoding', {}),
                         scientific_coverage='empty' if manifest.get('matching_summary', {}).get('quality_valid') == 0 else 'observations')
        except KeyboardInterrupt:
            invalidate_pair(directory, active_stage)
            entry['status'] = 'interrupted'
            report['outcome'] = 'interrupted'
            finish()
            raise
        except Exception as exc:
            failed = True
            invalidate_pair(directory, active_stage)
            entry.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            logging.error('Stack pair %s: %s', key, exc)
        finally:
            entry['elapsed_seconds'] = time.perf_counter()-started
            save()
            gc.collect()
    report['outcome'] = 'failed' if failed else 'complete'
    finish()
    return 1 if failed else 0
