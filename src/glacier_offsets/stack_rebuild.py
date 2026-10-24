"""Journalled ownership-only migration that retains verified COMPASS products."""
from copy import deepcopy
import json
import logging
from pathlib import Path
import re
import shutil

from .environment import inspect_environment
from .products import ChecksumCache, identity, write_json
from .stack import plan_stack, read_catalog, validate_stack
from .stack_lock import stack_lock
from .stack_ownership import POLICY, interior_ownership
from .stack_mosaic import POLICY as COMMON_POLICY, SWATH_MIDPOINT_POLICY, implementation, ownership, policy
from .stack_storage import (METHOD, SourceHashes, product_hashes, read_manifest,
                            reference_specification, verified, write_reference_layout)

PREVIOUS_POLICY = {'method': 'reference-overlap-midpoint-v1', 'edge_margin_lines': 8,
                   'narrow_overlap': 'error', 'missing_support': 'mask; no source substitution'}
LAYOUT_FILES = ('source.tif', 'x.vrt', 'y.vrt', 'z.vrt', 'layover_shadow_mask.vrt')
ASSEMBLY_FILES = ('slc.tif', 'slc.tif.msk', 'valid.tif', 'amplitude_preview.npz')


def _owned(root, relative):
    path = root/relative
    if not path.is_relative_to(root) or '..' in path.parts:
        raise ValueError(f'Invalid rebuild path: {path}')
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f'Rebuild cannot modify a symlink: {ancestor}')
    return path


def _verify_native(manifest, fingerprint, checksums, label):
    if manifest.get('status') != 'complete' or not verified(manifest, 'identity', fingerprint, checksums=checksums):
        raise ValueError(f'{label}: retained COMPASS products are incomplete or damaged; rebuild aborted')


def _protected_paths(products):
    protected = set()
    for name in products:
        path = Path(name).resolve()
        protected.update((path, *path.parents))
    return protected


def _preflight(cfg, old, frame, specification, checksums):
    root = Path(cfg['output']['directory']).resolve()
    old_id, new_id = frame['geometry_identity'], identity(specification)
    if (identity(frame['specification']) != old_id or
            old.get('reference_specification_identity') != old_id or
            old.get('layout_identity') != frame['layout_identity']):
        raise ValueError('Stack reference pins are inconsistent; rebuild aborted')
    previous = frame['specification']
    selected_policy = policy(cfg)
    common_migration = (previous.get('ownership_policy') in (COMMON_POLICY, SWATH_MIDPOINT_POLICY)
                        and selected_policy in (COMMON_POLICY, SWATH_MIDPOINT_POLICY))
    legacy_migration = previous.get('ownership_policy') == PREVIOUS_POLICY and selected_policy == POLICY
    if (previous.get('method') != METHOD or not (common_migration or legacy_migration) or
            previous | {'ownership_policy': selected_policy} != specification):
        raise ValueError('Rebuild requires supported ownership policies and unchanged reference inputs, selection and software')
    if common_migration:
        if not frame.get('common_validity_identity') or identity(frame['common_validity']) != frame['common_validity_identity']:
            raise ValueError('Common validity is missing or inconsistent; rebuild aborted')
        _, owners = ownership(frame['sources'], swath_midpoint=selected_policy == SWATH_MIDPOINT_POLICY)
    else:
        _, owners = interior_ownership(frame['sources'])
    updates, originals, native = {}, {}, {}

    def retain_metadata(path, before, after):
        relative = str(path.relative_to(root))
        originals[relative] = before
        updates[relative] = after

    keys = [e['burst_id'] for e in frame['sources']]
    if keys != previous['anchor']['selection']['selected_ids']:
        raise ValueError('Reference sources differ from the pinned burst selection')
    for entry in frame['sources']:
        key = entry['burst_id']
        path = _owned(root, f'geometry/bursts/{key}/manifest.json')
        if Path(entry['manifest']).resolve() != path:
            raise ValueError(f'{key}: reference manifest is outside its expected location')
        data = json.loads(path.read_text())
        _verify_native(data, identity({'reference': old_id, 'burst_id': key}), checksums, key)
        if data['coreg'] != entry['coreg']:
            raise ValueError(f'{key}: reference source metadata differs from its burst manifest')
        required = [entry['coreg']['slcs']['reference'], *entry['coreg']['geometry'].values()]
        if {str(Path(p).resolve()) for p in required} - {str(Path(p).resolve()) for p in data['products']}:
            raise ValueError(f'{key}: native reference files are not registered')
        native.update(data['products'])
        retain_metadata(path, data, data | {'identity': identity({'reference': new_id, 'burst_id': key})})
    complete_dates = 0
    delete = []
    for path in sorted((root/'acquisitions').glob('*/manifest.json')):
        path = _owned(root, str(path.relative_to(root)))
        data = json.loads(path.read_text())
        spec = data.get('specification', {})
        if spec.get('method') != METHOD or spec.get('geometry') != old_id:
            raise ValueError(f'{path.parent.name}: alignment does not belong to the pinned reference')
        old_alignment = identity(spec)
        new_spec = spec | {'geometry': new_id}
        new_alignment = identity(new_spec)
        child_slcs = {}
        for child in sorted((path.parent/'bursts').glob('*/manifest.json')):
            child = _owned(root, str(child.relative_to(root)))
            key = child.parent.name
            burst = json.loads(child.read_text())
            if burst.get('status') != 'complete':
                continue  # Missing/incomplete bursts are prepared by the next run.
            if key not in keys or key not in spec.get('burst_times', {}):
                raise ValueError(f'{child}: burst is absent from the saved alignment specification')
            _verify_native(burst, identity({'date': old_alignment, 'burst_id': key}), checksums, str(child))
            if str(Path(burst['slc']).resolve()) not in {str(Path(p).resolve()) for p in burst['products']}:
                raise ValueError(f'{child}: aligned SLC is not registered')
            native.update(burst['products'])
            child_slcs[key] = burst['slc']
            retain_metadata(child, burst, burst | {'identity': identity({'date': new_alignment, 'burst_id': key})})
        expected = {k for k, t in spec.get('burst_times', {}).items() if t is not None}
        aligned = data.get('alignment_identity') == old_alignment
        if data.get('alignment_identity') not in (None, old_alignment):
            raise ValueError(f'{path.parent.name}: alignment identity does not match its specification')
        products = {}
        if aligned:
            slcs = data['slcs']
            reference_date = previous['anchor']['reference_date']
            available = ({e['burst_id']: e['coreg']['slcs']['reference'] for e in frame['sources']}
                         if path.parent.name == reference_date else child_slcs)
            if set(slcs) != expected or any(available.get(k) != v for k, v in slcs.items()):
                raise ValueError(f'{path.parent.name}: aligned source inventory is incomplete or inconsistent')
            products = {str(Path(p).resolve()): digest for p, digest in
                        data.get('alignment_products', data.get('products', {})).items()}
            # Only native SLCs belong to an alignment. Empty dates used the old
            # source-index raster as a completion marker; replace it after staging.
            names = [str(Path(name).resolve()) for name in slcs.values()]
            for name in names:
                if name not in products or checksums(name) != products[name]:
                    raise ValueError(f'{path.parent.name}: retained aligned SLC is missing or damaged')
            if common_migration:
                # Resampling offsets and validity metadata are needed again
                # when the active inventory changes. Keep every alignment
                # product, including the stable reference marker of empty dates.
                for name, digest in products.items():
                    if checksums(name) != digest:
                        raise ValueError(f'{path.parent.name}: retained alignment product is missing or damaged')
                if set(data.get('valid_windows', {})) != set(slcs):
                    raise ValueError(f'{path.parent.name}: retained valid rectangles are incomplete')
            else:
                products = {name: products[name] for name in names}
            native.update(products)
            complete_dates += 1
        after = {'date': path.parent.name, 'specification': new_spec, 'burst_times': spec.get('burst_times', {}),
                 'status': 'aligned' if aligned else 'aligning', 'slcs': data.get('slcs', {}) if aligned else {}}
        if aligned:
            after.update(alignment_identity=new_alignment, products=products, alignment_products=products)
            if common_migration:
                after['valid_windows'] = deepcopy(data['valid_windows'])
        retain_metadata(path, data, after)
        delete.extend(str((path.parent/name).relative_to(root)) for name in ASSEMBLY_FILES)
    if common_migration:
        missing = [day for day in frame['common_validity']['dates']
                   if f'acquisitions/{day}/manifest.json' not in updates]
        if missing:
            raise ValueError(f'Common-validity acquisition metadata is missing: {missing}; rebuild aborted')
    for directory in sorted((root/'pairs').glob('*')):
        # Finder writes AppleDouble sidecars on external volumes. These files
        # are not pair outputs and must not enter the rebuild deletion journal.
        if directory.is_file() and (directory.name.startswith('._') or directory.name == '.DS_Store'):
            continue
        _owned(root, str(directory.relative_to(root)))
        if not directory.is_dir() or directory.name not in old.get('pairs', {}) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}', directory.name):
            raise ValueError(f'Unregistered pair output in {directory}; rebuild aborted')
        delete.append(str(directory.relative_to(root)))
    delete.append('geometry/map_grid.json')
    protected = _protected_paths(native)
    for relative in delete:
        path = _owned(root, relative)
        if path in protected:
            raise ValueError(f'Rebuild would delete a retained native product: {path}')
    for name, digest in native.items():
        if checksums(name) != digest:
            raise ValueError(f'Retained native product changed: {name}')
    return {'updates': updates, 'before': originals | {'geometry/manifest.json': frame, 'stack_manifest.json': old},
            'native_products': native, 'delete': delete,
            'summary': {'aligned_dates': complete_dates, 'native_products': len(native),
                        'pair_directories': sum(p.startswith('pairs/') for p in delete),
                        'range_cuts': sum('cut_global_sample' in c for c in owners.get('cuts', [])),
                        'azimuth_cuts': sum('cut_global_line' in c for c in owners.get('cuts', [])),
                        'ownership_policy': selected_policy['method']}}


def _ignore_missing_output(operation, path, error):
    # On macOS, unlinking a file can also remove its AppleDouble sidecar.
    # Python 3.12 rmtree still visits its cached sidecar entry afterwards.
    # Continue traversal rather than abandoning the rest of the pair tree.
    if not isinstance(error, FileNotFoundError):
        raise error


def _apply(root, journal, checksums):
    """Idempotent publication; normal stages refuse an unfinished journal."""
    protected = _protected_paths(journal['native_products'])
    for relative in journal['delete']:
        parts = Path(relative).parts
        assembly = (len(parts) == 3 and parts[0] == 'acquisitions' and
                    re.fullmatch(r'\d{4}-\d{2}-\d{2}', parts[1]) and parts[2] in ASSEMBLY_FILES)
        pair = len(parts) == 2 and parts[0] == 'pairs' and parts[1] in journal['before']['stack_manifest.json'].get('pairs', {})
        if not (assembly or pair or relative == 'geometry/map_grid.json'):
            raise ValueError(f'Unrecognized derived output in rebuild journal: {relative}')
        path = _owned(root, relative)
        if path in protected:
            raise ValueError(f'Rebuild would delete a retained native product: {path}')
        if path.is_dir():
            shutil.rmtree(path, onexc=_ignore_missing_output)
        else:
            path.unlink(missing_ok=True)
    for relative, digest in journal['layout_products'].items():
        source = _owned(root, f'.layout-rebuild/staged/{Path(relative).name}')
        destination = _owned(root, relative)
        if source.is_file():
            if checksums(source) != digest:
                raise ValueError(f'Staged layout product changed: {source}')
            source.replace(destination)
        elif not destination.is_file() or checksums(destination) != digest:
            raise ValueError(f'Layout product missing or damaged; rebuild cannot resume: {destination}')
    for relative, data in journal['updates'].items():
        write_json(_owned(root, relative), data)
    # Commit the catalog last, then unblock normal processing.
    write_json(root/'geometry/manifest.json', journal['frame'])
    write_json(root/'stack_manifest.json', journal['catalog'])
    journal['status'] = 'complete'
    write_json(root/'.layout-rebuild/journal.json', journal)


def rebuild_layout(cfg, *, dry_run=False):
    root = Path(cfg['output']['directory']).resolve()
    with stack_lock(root, rebuilding=True):
        journal_path = _owned(root, '.layout-rebuild/journal.json')
        journal = json.loads(journal_path.read_text()) if journal_path.is_file() else {}
        pending = journal and journal.get('status') != 'complete'
        old = journal['before']['stack_manifest.json'] if pending else read_catalog(root)
        frame = journal['before']['geometry/manifest.json'] if pending else read_manifest(root/'geometry')
        if not frame.get('geometry_identity') or frame.get('status') != 'complete':
            raise ValueError('A prepared stack reference is required for an ownership rebuild')
        plan = plan_stack(cfg, previous=old)
        validate_stack(cfg, plan, check_dates=False)
        specification = reference_specification(cfg, plan, SourceHashes(), inspect_environment())
        checksums = ChecksumCache()
        if pending:
            if journal['frame']['specification'] != specification:
                raise ValueError('Reference configuration changed during an unfinished rebuild')
            for name, digest in journal['native_products'].items():
                if checksums(name) != digest:
                    raise ValueError(f'Retained native product changed during rebuild: {name}')
            if not dry_run:
                _apply(root, journal, checksums)
            return journal['summary'] | {'status': 'resume_ready' if dry_run else 'complete'}
        if frame.get('specification') == specification:
            return {'status': 'already_current', 'ownership_policy': policy(cfg)['method']}
        logging.info('Verifying retained COMPASS products before ownership rebuild')
        journal = _preflight(cfg, old, frame, specification, checksums)
        if dry_run:
            return journal['summary'] | {'status': 'ready', 'dry_run': True}
        staging = _owned(root, '.layout-rebuild/staged')
        staging.mkdir(parents=True, exist_ok=True)
        layout = write_reference_layout(staging, frame['sources'], checksums=checksums,
                                        ownership_policy=specification['ownership_policy'],
                                        implementation=implementation(cfg))
        if layout['coreg']['reference_grid'] != frame['coreg']['reference_grid']:
            raise ValueError('Ownership rebuild changed the native radar grid')
        layout_products = product_hashes([staging/name for name in LAYOUT_FILES], checksums=checksums)
        layout_products = {f'geometry/{Path(name).name}': digest for name, digest in layout_products.items()}
        coreg = layout['coreg']
        coreg['source_index'] = str(root/'geometry/source.tif')
        coreg['geometry'] = {field: str(root/f'geometry/{field}.vrt') for field in coreg['geometry']}
        new_frame = dict(layout, geometry_identity=identity(specification), specification=specification,
                         software=frame['software'], status='complete')
        if frame.get('common_validity_identity'):
            common = deepcopy(frame['common_validity'])
            common['method'] = specification['ownership_policy']['method']
            common['dates'] = {day: identity(journal['updates'][f'acquisitions/{day}/manifest.json']['specification'])
                               for day in common['dates']}
            new_frame.update(common_validity=common, common_validity_identity=identity(common))
            if 'map_grid' in frame:
                new_frame['map_grid'] = frame['map_grid']
        # Geometry verification should cover reference products, not every date.
        new_frame['products'] = {name: digest for entry in frame['sources']
                                 for name, digest in read_manifest(Path(entry['manifest']).parent)['products'].items()}
        new_frame['products'].update({str(root/name): digest for name, digest in layout_products.items()})
        for relative, data in journal['updates'].items():
            if (not frame.get('common_validity_identity') and relative.startswith('acquisitions/')
                    and data.get('alignment_identity') and not data['slcs']):
                products = {str(root/'geometry/source.tif'): layout_products['geometry/source.tif']}
                data.update(products=products, alignment_products=products)
        catalog = deepcopy(old)
        catalog.update(reference_specification_identity=new_frame['geometry_identity'],
                       geometry_identity=new_frame['geometry_identity'], layout_identity=new_frame['layout_identity'],
                       reference_status='complete', command='rebuild-layout', outcome='complete')
        for key in ('common_validity', 'common_validity_identity'):
            if key in new_frame:
                catalog[key] = new_frame[key]
        for key in ('map_grid', 'map_grid_identity', 'matching_template', 'matching_template_identity', 'metrics', 'error'):
            catalog.pop(key, None)
        for day, entry in catalog.get('acquisitions', {}).items():
            data = journal['updates'].get(f'acquisitions/{day}/manifest.json', {})
            active = entry.get('active', True)
            entry.clear()
            entry.update(active=active, status='aligned' if data.get('alignment_identity') else 'planned',
                         directory=str(root/'acquisitions'/day))
            if data.get('alignment_identity'):
                entry.update(alignment_identity=data['alignment_identity'], completed_stage='coregister')
        for entry in catalog.get('pairs', {}).values():
            keep = {k: entry[k] for k in ('earlier', 'later', 'active', 'separation_seconds') if k in entry}
            entry.clear()
            entry.update(keep, status='planned')
        journal.update(frame=new_frame, catalog=catalog, layout_products=layout_products, status='applying')
        write_json(root/'.layout-rebuild/before.json', journal['before'])
        write_json(journal_path, journal)
        _apply(root, journal, checksums)
        return journal['summary'] | {'status': 'complete'}
