"""AOI orchestration: independent COMPASS coregistration, then assembled tracking."""
from copy import deepcopy
import gc
import json
import logging
from pathlib import Path

from .assembly import assemble, invalidate, release_merged
from .products import identity, write_json
from .storage import keep_merged_slcs


def execute_assembled_batch(cfg, command, force=False):
    from .batch import (InvocationContext, dem_extent, effective_config, inspect_environment,
                        input_provenance, product_summary, select_aoi, validate_burst)
    from .pipeline import execute
    from .ampcor import match, matching_specification
    from .geocode import geocode, geocoding_specification
    from .diagnostics import diagnostics
    pairs, selection = select_aoi(cfg)
    root = Path(cfg['output']['directory'])
    write_json(root/'selection.json', selection)
    entries = {p[0].id: {'status': 'pending', 'stage': 'coregister',
                         'products': {'manifest.json': str(root/p[0].id/'manifest.json'),
                                      'coreg': str(root/p[0].id/'coreg')}} for p in pairs}
    report = {'command': command, 'force': force, 'processing': 'SLC assembly before tracking',
              'selection_identity': selection['selection_identity'], 'selected_ids': selection['selected_ids'],
              'bursts': entries, 'merged': {'status': 'pending', 'directory': str(root/'merged')}, 'outcome': 'running'}
    def save():write_json(root/'batch_manifest.json', report)
    save()
    if not pairs:
        invalidate(root/'merged', 'empty_selection')
        report.update(outcome='empty_selection');report['merged']['status']='blocked_empty_selection';save()
        return 2
    try:
        context = InvocationContext(tuple(pairs), json.dumps(input_provenance(cfg)),
                                    json.dumps(inspect_environment()), json.dumps(selection))
        extent = dem_extent(cfg)
        from .assembly import validate_native_layout
        validate_native_layout(pairs,cfg)
    except KeyboardInterrupt:
        report['outcome']='interrupted';save();raise
    except Exception as exc:
        invalidate(root/'merged', 'preflight_failed')
        report.update(outcome='preflight_failed',error=f'{type(exc).__name__}: {exc}');save();raise
    sources = []
    for pair in pairs:
        key=pair[0].id;entry=entries[key];entry['status']='running';save()
        try:
            effective=effective_config(cfg,key)
            validate_burst(effective,pair,extent)
            if command=='run':
                execute(effective,'coregister',force=force,context=context)
            path=root/key/'manifest.json'
            if not path.is_file():raise ValueError('Coregistration manifest missing; run coregister first')
            manifest=json.loads(path.read_text())
            if not manifest.get('coreg_identity'):raise ValueError('Coregistration incomplete; run coregister first')
            spec=manifest['coreg_specification']
            if spec['inputs']!=context.provenance or spec['polarization']!=cfg['inputs']['polarization']:
                raise ValueError('Inputs or polarization changed; run coregister first')
            sources.append((root/key,manifest))
            entry.update(status='complete',coreg_identity=manifest['coreg_identity'])
        except KeyboardInterrupt:
            entry['status']='interrupted';report['outcome']='interrupted';save();raise
        except Exception as exc:
            entry.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            logging.error('%s: %s',key,exc)
        finally:
            save();gc.collect()
    if len(sources)!=len(pairs):
        invalidate(root/'merged','incomplete_source_bursts')
        report['merged']['status']='blocked_incomplete_bursts';report['outcome']='failed';save()
        return 1
    try:
        report['merged']['status']='assembling';save()
        merged,manifest=assemble(cfg,sources,native_pairs={p[0].id:p for p in pairs},
                                 force=force and command in {'assemble','run'}, allow_released=command!='assemble')
        effective=deepcopy(cfg);effective.pop('aoi',None);effective['inputs']['burst_id']='merged'
        if command in {'match','run'}:
            report['merged']['status']='matching';save()
            current = (not force and manifest.get('match_identity') == identity(matching_specification(effective,manifest))
                       and (merged/'radar/offsets.h5').is_file())
            if (not current or keep_merged_slcs(cfg)) and any(not Path(p).is_file() for p in manifest['coreg']['slcs'].values()):
                merged, manifest = assemble(cfg, sources, native_pairs={p[0].id:p for p in pairs})
            match(effective,merged,manifest,force=force,native_pair={p[0].id:p for p in pairs})
        elif command in {'geocode','diagnostics'}:
            if manifest.get('match_identity')!=identity(matching_specification(effective,manifest)):
                raise ValueError('Merged matching missing or changed; run match with the AOI config first')
        if (command != 'assemble' and manifest.get('match_identity') == identity(matching_specification(effective,manifest))
                and (merged/'radar/offsets.h5').is_file()):
            release_merged(cfg, merged, manifest)
        if command in {'geocode','run'}:
            report['merged']['status']='geocoding';save()
            geocode(effective,merged,manifest,force=force)
        if command=='diagnostics' and manifest.get('geocode_identity')!=identity(geocoding_specification(effective,manifest)):
            raise ValueError('Merged geocoding missing or changed; run geocode with the AOI config first')
        if command in {'diagnostics','run'}:
            diagnostics(effective,merged,manifest)
        if command not in {'assemble','match','geocode','diagnostics','run'}:
            raise ValueError(f'Unknown assembled stage {command}')
        report['merged'].update(product_summary(merged))
        report['merged'].update(status='complete',assembly_identity=manifest['coreg_identity'],
            source_burst_ids=manifest['assembly']['source_burst_ids'],
            products={key:str(merged/key) for key in ('manifest.json','assembly.json','coreg','radar','geo','diagnostics')})
        report['outcome']='complete';save()
        return 0
    except KeyboardInterrupt:
        report['merged']['status']='interrupted';report['outcome']='interrupted';save();raise
    except Exception as exc:
        if report['merged']['status']=='assembling':invalidate(root/'merged','required_assembly_failed')
        report['merged'].update(status='failed',error=f'{type(exc).__name__}: {exc}')
        report['outcome']='failed';save()
        logging.error('Merged AOI processing: %s',exc)
        return 1
