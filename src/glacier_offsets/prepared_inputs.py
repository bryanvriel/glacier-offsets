"""Resolve a successful preparation manifest into stack initializer inputs."""
import json
from pathlib import Path


def prepared_paths(path, safe_directory=None, orbit_directory=None, dem=None):
    supplied_dem = dem
    path = Path(path).expanduser().resolve(strict=True)
    with path.open() as stream:
        m = json.load(stream)
    if not isinstance(m,dict) or m.get('method') != 's1-input-preparation-v1':
        raise ValueError('Expected a glacier-offsets prepare-inputs manifest')
    scenes = m.get('scenes')
    if (m.get('status') != 'complete' or m.get('failed_steps') != 0 or not isinstance(scenes,list) or not scenes or
            any(not isinstance(s,dict) or s.get('status')!='ok' for s in scenes)):
        raise ValueError('Prepared inputs have failures or are incomplete; rerun prepare-inputs before stack init')
    download = m.get('burst_download')
    if download:
        from .burst_downloads import METHOD, collection_identity
        collection_manifest = Path(download['manifest'])
        current = json.loads(collection_manifest.read_text()) if collection_manifest.is_file() else {}
        if (current.get('method') != METHOD or current.get('status') != 'complete' or
                any(g.get('status') != 'complete' for g in current.get('acquisitions', [])) or
                collection_identity(current) != download['identity']):
            raise ValueError('Burst download collection changed; rerun prepare-inputs before stack init')
    def local(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent/p).resolve())
    safe = safe_directory or m.get('safe_directory')
    orbits = orbit_directory or m.get('orbit_directory')
    dem = dem or m.get('dem')
    if not safe:
        raise ValueError('Single-SAFE preparation requires an explicit --safe-dir for stack initialization')
    if not orbits or not dem:
        raise ValueError('A shared DEM and orbit directory are required; provide --dem/--orbit-dir '
                         'when preparation skipped them or used per-safe DEMs')
    # Explicit CLI paths retain their usual current-directory interpretation.
    return (str(Path(safe_directory).expanduser().resolve()) if safe_directory else local(safe),
            str(Path(orbit_directory).expanduser().resolve()) if orbit_directory else local(orbits),
            str(Path(supplied_dem).expanduser().resolve()) if supplied_dem else local(dem))
