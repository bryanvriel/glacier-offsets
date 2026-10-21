"""Restartable single-pair stages."""
import json
from pathlib import Path
from .compass_coreg import coregister
from .ampcor import match, matching_specification
from .products import identity, input_provenance
from .geocode import geocode, geocoding_specification
from .diagnostics import diagnostics


def execute(cfg, command, force=False, context=None):
    if command in {'coregister', 'run'}:
        root, manifest = coregister(cfg, force=force, **({'context': context} if context else {}))
        if command == 'coregister': return
    else:
        root=Path(cfg['output']['directory'])/cfg['inputs']['burst_id']
        p=root/'manifest.json'
        if not p.is_file(): raise ValueError('Coregistration manifest missing; run coregister first')
        manifest=json.loads(p.read_text())
        if not manifest.get('coreg_identity'):raise ValueError('Coregistration is incomplete; run coregister first')
        provenance = context.provenance if context else input_provenance(cfg)
        if provenance != manifest['coreg_specification']['inputs'] or cfg['inputs']['polarization'] != manifest['coreg_specification']['polarization']:
            raise ValueError('Inputs or polarization changed; run coregister first')
        if command in {'geocode','diagnostics'} and manifest.get('match_identity') != identity(matching_specification(cfg,manifest)):
            raise ValueError('Matching configuration changed or matching is incomplete; run match first')
        if command == 'diagnostics' and manifest.get('geocode_identity') != identity(geocoding_specification(cfg,manifest)):
            raise ValueError('Geocoding configuration changed or geocoding is incomplete; run geocode first')
    if command in {'match','run'}:
        match(cfg,root,manifest,force=force, **({'native_pair': context.pair(cfg['inputs']['burst_id'])} if context else {}))
        if command=='match':return
    if command in {'geocode','run'}:
        geocode(cfg,root,manifest,force=force)
        if command=='geocode':return
    if command in {'diagnostics','run'}:
        diagnostics(cfg,root,manifest)
        return
    raise ValueError(f'Unknown processing stage: {command}')
