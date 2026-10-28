"""Portable, immutable stack reference assets with separate runtime path resolution."""
from copy import deepcopy
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import xml.etree.ElementTree as ET

from .products import ChecksumCache, checksum, identity, write_json

SCHEMA = 'glacier-offsets-reference-v1'
MANIFEST = 'reference.json'
TOKEN = '@reference/'


def _relative(name):
    if not isinstance(name, str):
        raise ValueError(f'Invalid reference bundle path: {name!r}')
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or str(p) != name:
        raise ValueError(f'Invalid reference bundle path: {name!r}')
    return p


def _path(root, name):
    p = root / str(_relative(name))
    if any(a.is_symlink() for a in (p, *p.parents) if a != root and root in a.parents):
        raise ValueError(f'Reference bundle dependency is linked: {name}')
    if not p.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Reference bundle dependency escapes its directory: {name}')
    return p


def _walk(value, convert):
    if isinstance(value, dict):
        return {convert(k): _walk(v, convert) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, convert) for v in value]
    return convert(value) if isinstance(value, str) else value


def read_bundle(root, *, verify=False, checksums=None):
    root = Path(root).resolve()
    try:
        document = (root / MANIFEST).read_text()
        data = json.loads(document)
        if data['schema'] != SCHEMA:
            raise ValueError('Unsupported reference bundle schema')
        expected = data['fingerprint']
        if identity({k: v for k, v in data.items() if k != 'fingerprint'}) != expected:
            raise ValueError('Reference bundle manifest fingerprint mismatch')
        frame = data['frame']
        if frame['status'] != 'complete' or identity(frame['specification']) != frame['geometry_identity']:
            raise ValueError('Reference bundle geometry identity is inconsistent')
        if not frame.get('map_grid') or not frame.get('sources'):
            raise ValueError('Reference bundle needs a completed layout and map grid')
        files = data['files']
        if not isinstance(files, dict) or not files:
            raise ValueError('Reference bundle file inventory is empty')
        for name, digest in files.items():
            path = _path(root, name)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f'Reference bundle dependency missing or linked: {name}')
            if verify and (checksums or checksum)(path) != digest:
                raise ValueError(f'Reference bundle checksum mismatch: {name}')
        for coreg in [frame['coreg'], *[s['coreg'] for s in frame['sources']]]:
            paths = [*coreg.get('geometry', {}).values(), *coreg.get('slcs', {}).values()]
            paths += [coreg[k] for k in ('reference_directory', 'source_index') if k in coreg]
            if not paths or any(not p.startswith(TOKEN) for p in paths):
                raise ValueError('Reference bundle contains unresolved geometry dependencies')
        # Resolve only declared assets. Original provenance intentionally retains its paths.
        def token(s):
            if s.startswith(TOKEN):
                name = s[len(TOKEN):]
                path = _path(root, name)
                if name not in files and not path.is_dir():
                    raise ValueError(f'Unregistered reference bundle dependency: {name}')
            return s
        _walk({k: v for k, v in data.items() if k not in ('files', 'origin')}, token)
        for key in ('safe', 'orbit', 'dem'):
            if not data['inputs'][key].startswith(TOKEN):
                raise ValueError(f'Invalid bundled reference input: {key}')
        if verify and (root / MANIFEST).read_text() != document:
            raise ValueError('Reference bundle manifest changed during verification')
        return data
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Malformed reference bundle: {root}') from exc


def resolved(data, root):
    return _walk(data, lambda s: str(_path(Path(root), s[len(TOKEN):])) if s.startswith(TOKEN) else s)


def configure(cfg):
    """Select managed assets first, allowing the original transfer disk to be offline."""
    requested = cfg.get('stack', {}).get('reference_directory')
    if not requested:
        return cfg
    managed = Path(cfg['output']['directory']) / '.reference'
    root = managed if (managed / MANIFEST).is_file() else Path(requested)
    data = read_bundle(root)
    if root == managed and Path(requested).resolve() != managed.resolve():
        if (Path(requested) / MANIFEST).exists():
            if read_bundle(requested)['fingerprint'] != data['fingerprint']:
                raise ValueError('Reference replacement requires a new output directory')
        else:
            pin = Path(cfg['output']['directory']) / '.reference-source.json'
            if pin.exists() and json.loads(pin.read_text())['directory'] != str(Path(requested).resolve()):
                raise ValueError('Cannot verify replacement reference directory; use the pinned bundle or a new output directory')
    local = resolved(data, root)
    origin = data['frame']['specification']
    date = origin['anchor']['reference_date']
    requested_date = cfg['stack'].get('reference_date')
    if requested_date is not None and str(requested_date) != date:
        raise ValueError('Configured reference_date conflicts with reference bundle')
    cfg['stack']['reference_date'] = date
    contract = data['contract']
    from .stack_mosaic import policy
    if policy(cfg) != origin['ownership_policy']:
        raise ValueError('Processing/ownership policy conflicts with reference bundle')
    for key in ('window', 'search', 'stride', 'first_center'):
        if cfg['matching'].get(key) != contract['matching'].get(key):
            raise ValueError(f'matching.{key} conflicts with reference bundle matching grid')
    for key in ('epsg', 'spacing_m'):
        if cfg['geocoding'].get(key) != contract['geocoding'][key]:
            raise ValueError(f'geocoding.{key} conflicts with reference bundle map grid')
    selection = origin['anchor']['selection']
    if isinstance(cfg['inputs'].get('polarization'), str):
        cfg['inputs']['polarization'] = cfg['inputs']['polarization'].upper()
    for key in ('polarization', 'subswaths', 'burst_id'):
        if cfg['inputs'].get(key) != selection.get(key):
            raise ValueError(f'inputs.{key} conflicts with reference bundle')
    dem = cfg['inputs'].get('dem')
    old_dem = origin['inputs']['dem']
    if dem and dem not in (old_dem['path'], local['inputs']['dem']):
        if not Path(dem).is_file() or checksum(dem) != old_dem['sha256']:
            raise ValueError('Configured DEM conflicts with reference bundle')
    cfg['inputs']['dem'] = local['inputs']['dem']
    old_aoi = (selection.get('aoi') or {}).get('specification')
    aoi = cfg.get('aoi')
    if aoi not in (None, old_aoi, local.get('aoi')):
        from .aoi import load_aoi
        if load_aoi(aoi).report['geometry_identity'] != selection['aoi']['geometry_identity']:
            raise ValueError('Configured AOI conflicts with reference bundle')
    cfg['aoi'] = local.get('aoi')
    cfg['_reference_bundle'] = {'root': str(root.resolve()), 'data': data, 'resolved': local}
    return cfg


def runtime_frame(cfg):
    selected = cfg['_reference_bundle']
    data, local = selected['data'], selected['resolved']
    frame = deepcopy(local['frame'])
    # Historical identities describe the original scientific reference, not its disk location.
    frame['specification'] = deepcopy(data['frame']['specification'])
    frame['reference_bundle'] = {'schema': SCHEMA, 'fingerprint': data['fingerprint'],
                               'origin': deepcopy(data['origin'])}
    frame['products'] = {str(Path(selected['root']) / p): d for p, d in data['files'].items()}
    return frame


def validate_native_geometry(cfg):
    """Fail preflight on unsupported native products before any date is processed."""
    from .compass_coreg import inspect_pair
    from pyproj import CRS
    for source in runtime_frame(cfg)['sources']:
        coreg = source['coreg']
        slc = coreg['slcs']['reference']
        inspected = inspect_pair(slc, slc, coreg['reference_directory'])
        if inspected['reference_grid'] != coreg['reference_grid']:
            raise ValueError(f"{source['burst_id']}: bundled native radar grid differs from its manifest")
        if not CRS.from_user_input(inspected['geometry_crs']).equals(CRS.from_user_input(coreg['geometry_crs'])):
            raise ValueError(f"{source['burst_id']}: bundled native geometry CRS differs from its manifest")


def _copy_bundle(source, stage, data):
    """Copy authenticated assets only; ignore incidental filesystem metadata."""
    stage.mkdir()
    names = [*data['files'], MANIFEST]
    for index, name in enumerate(names):
        if index % 20 == 0:
            logging.info('Copying authenticated reference assets (%d/%d)', index, len(names))
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_path(Path(source), name), destination)


def install(cfg, *, checksums=None):
    """Called with destination lock held; publish a complete managed snapshot once."""
    if not cfg.get('_reference_bundle'):
        return
    output = Path(cfg['output']['directory'])
    target = output / '.reference'
    selected = cfg['_reference_bundle']
    cache = checksums or ChecksumCache()
    logging.info('Verifying portable reference assets before import')
    data = read_bundle(selected['root'], verify=True, checksums=cache)
    existing_path = output / 'geometry/manifest.json'
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text())
        if existing.get('reference_bundle', {}).get('fingerprint') != data['fingerprint']:
            raise ValueError('Cannot replace an initialized stack reference; use a new output directory')
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix='.reference-import-', dir=output) as folder:
            stage = Path(folder) / 'bundle'
            _copy_bundle(selected['root'], stage, data)
            staged = read_bundle(stage, verify=True, checksums=cache)
            current = read_bundle(selected['root'], verify=True, checksums=cache)
            if staged['fingerprint'] != data['fingerprint'] or current['fingerprint'] != data['fingerprint']:
                raise ValueError('Reference bundle changed during import')
            stage.rename(target)
    elif read_bundle(target, verify=True, checksums=cache)['fingerprint'] != data['fingerprint']:
        raise ValueError('Reference replacement requires a new output directory')
    write_json(output / '.reference-source.json', {'directory': str(Path(cfg['stack']['reference_directory']).resolve()),
                                                   'fingerprint': data['fingerprint']})
    configure(cfg)
    # Keep packaging's supplemental orbit cache local to the working geometry.
    geometry = output / 'geometry'
    geometry.mkdir(exist_ok=True)
    for name in ('reference_orbit.h5', 'reference_orbit.json'):
        source = target / 'geometry' / name
        if source.exists() and not (geometry / name).exists():
            shutil.copyfile(source, geometry / name)


class _Collector:
    def __init__(self, geometry, checksums=None):
        self.geometry = geometry.resolve()
        self.mapping = {}
        self.digests = {}
        self.inputs = checksums or ChecksumCache()

    def add(self, path, relative=None):
        path = Path(path).absolute()
        if not path.is_file():
            raise ValueError(f'Reference dependency is missing: {path}')
        key = str(path)
        if key in self.mapping:
            return
        if relative is None:
            owned = path if path.is_relative_to(self.geometry) else path.resolve()
            relative = ('geometry/' + str(owned.relative_to(self.geometry)) if owned.is_relative_to(self.geometry)
                        else 'assets/' + self.inputs(path)[:16] + '/' + path.name)
        self.mapping[key] = relative
        self.digests[key] = self.inputs(path)
        if path.suffix.lower() == '.vrt' and not path.name.startswith('._'):
            tree = ET.parse(path)
            for node in tree.iter('SourceFilename'):
                source = node.text
                if not source:
                    raise ValueError(f'Empty VRT source: {path}')
                dependency = Path(source)
                if node.get('relativeToVRT', '0') == '1':
                    dependency = path.parent / dependency
                self.add(dependency.resolve())
        if path.suffix.lower() in ('.tif', '.tiff', '.vrt') and not path.name.startswith('._'):
            from osgeo import gdal
            ds = gdal.Open(str(path))
            if ds is None:
                raise ValueError(f'Cannot open reference raster: {path}')
            for dependency in ds.GetFileList() or []:
                candidate = Path(dependency).absolute()
                if str(candidate) != key:
                    self.add(candidate)
        # Sidecars must retain their basename association with the raster.
        for suffix in ('.msk', '.aux.xml', '.ovr', '.hdr', '.xml'):
            ancillary = Path(str(path) + suffix)
            if ancillary.is_file():
                self.add(ancillary, relative + suffix)

    def tree(self, source, relative):
        source = Path(source).absolute()
        if source.is_dir():
            self.mapping[str(source)] = relative
            for p in sorted(source.rglob('*')):
                if p.is_file():
                    self.add(p, relative + '/' + str(p.relative_to(source)))
        else:
            self.add(source, relative)

    def convert(self, s):
        if s in self.mapping:
            return TOKEN + self.mapping[s]
        # Reference directories appear in coreg metadata alongside individual file paths.
        if s.startswith('/'):
            p = Path(s).resolve()
            if str(p) in self.mapping:
                return TOKEN + self.mapping[str(p)]
            if p.is_relative_to(self.geometry):
                return TOKEN + 'geometry/' + str(p.relative_to(self.geometry))
        return s

    def copy(self, stage):
        copied = set()
        for index, (name, relative) in enumerate(list(self.mapping.items())):
            if index % 20 == 0:
                logging.info('Copying reference assets (%d/%d)', index, len(self.mapping))
            source = Path(name)
            if relative in copied or source.is_dir() or relative == 'geometry/manifest.json':
                # reference.json carries the authoritative frame. The source
                # manifest is tracked for concurrent-change detection only.
                continue
            copied.add(relative)
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.suffix.lower() == '.vrt' and not source.name.startswith('._'):
                tree = ET.parse(source)
                for node in tree.iter('SourceFilename'):
                    p = Path(node.text)
                    if node.get('relativeToVRT', '0') == '1':
                        p = source.parent / p
                    node.text = os.path.relpath(stage / self.mapping[str(p.resolve())], destination.parent)
                    node.set('relativeToVRT', '1')
                tree.write(destination, encoding='utf-8')
            elif source.suffix.lower() == '.json' and not source.name.startswith('._') and source.resolve().is_relative_to(self.geometry):
                write_json(destination, _walk(json.loads(source.read_text()), self.convert))
            else:
                shutil.copyfile(source, destination)


def export_reference(cfg, output):
    from .stack_lock import stack_lock
    from .stack_storage import read_manifest, verified
    root = Path(cfg['output']['directory']).resolve()
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise ValueError('Reference export destination already exists')
    if destination.is_relative_to(root):
        raise ValueError('Reference export destination must be outside the source stack')
    with stack_lock(root):
        if cfg.get('_reference_bundle'):
            source = cfg['_reference_bundle']['root']
            data = read_bundle(source, verify=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.reference-export-', dir=destination.parent) as folder:
                stage = Path(folder) / 'bundle'
                _copy_bundle(source, stage, data)
                staged = read_bundle(stage, verify=True)
                current = read_bundle(source, verify=True)
                if staged['fingerprint'] != data['fingerprint'] or current['fingerprint'] != data['fingerprint']:
                    raise ValueError('Reference bundle changed during export')
                if destination.exists():
                    raise ValueError('Reference export destination already exists')
                stage.rename(destination)
            return {'output': str(destination), 'fingerprint': data['fingerprint']}
        logging.info('Verifying source reference products before export')
        frame = read_manifest(root / 'geometry')
        cache = ChecksumCache()
        if (frame.get('status') != 'complete' or not frame.get('geometry_identity') or
                identity(frame.get('specification')) != frame['geometry_identity'] or
                not verified(frame, 'geometry_identity', frame['geometry_identity'], checksums=cache)):
            raise ValueError('Export requires a complete, checksum-verified reference')
        catalog = json.loads((root / 'stack_manifest.json').read_text())
        frame = deepcopy(frame)
        frame['map_grid'] = frame.get('map_grid') or catalog.get('map_grid')
        if not frame['map_grid']:
            raise ValueError('Export requires a pinned common map grid; run stack geocode first')
        from .stack_mosaic import policy
        from .stack_pairs import matching_template
        if policy(cfg) != frame['specification']['ownership_policy']:
            raise ValueError('Configured ownership policy differs from the completed reference')
        recorded_template = catalog.get('matching_template')
        if recorded_template:
            template = matching_template(cfg, frame)
            for key, value in [('window', template.window), ('search', template.search), ('stride', template.stride),
                               ('starts_line', template.starts_line.tolist()), ('starts_sample', template.starts_sample.tolist())]:
                if list(value) != recorded_template[key]:
                    raise ValueError('Configured matching grid differs from the completed stack')
        for key, value in [('epsg', frame['map_grid']['epsg']), ('spacing_m', frame['map_grid']['transform'][1])]:
            if cfg['geocoding'][key] != value:
                raise ValueError('Configured geocoding differs from the completed stack')
        collector = _Collector(root / 'geometry', cache)
        for name in frame['products']:
            collector.add(name)
        collector.add(root / 'geometry/manifest.json')
        for name in ('reference_orbit.h5', 'reference_orbit.json'):
            if (root / 'geometry' / name).is_file():
                collector.add(root / 'geometry' / name)
        for source in frame['sources']:
            collector.add(source['manifest'])
            directory = Path(source['coreg']['reference_directory'])
            for path in directory.iterdir():
                if path.is_file() and not path.name.startswith('.'):
                    collector.add(path)
        inputs = frame['specification']['inputs']
        for key, record in inputs.items():
            if checksum(record['path']) != record['sha256']:
                raise ValueError(f'Pinned reference {key} checksum mismatch')
            collector.tree(record['path'], 'inputs/' + key + '/' + Path(record['path']).name)
        selection = frame['specification']['anchor']['selection']
        aoi = (selection.get('aoi') or {}).get('specification')
        for path, digest in (selection.get('aoi') or {}).get('source_checksums', {}).items():
            if checksum(path) != digest:
                raise ValueError('Pinned AOI checksum mismatch')
            collector.add(path, 'inputs/aoi/' + Path(path).name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.reference-export-', dir=destination.parent) as folder:
            stage = Path(folder) / 'bundle'
            stage.mkdir()
            collector.copy(stage)
            from .package_geometry import retain_reference_orbit
            # Legacy references can be enriched in the export staging area only.
            retain_reference_orbit(stage / 'geometry', inputs['orbit'])
            original_specification = deepcopy(frame['specification'])
            portable = _walk(frame, collector.convert)
            portable['specification'] = original_specification
            files = {str(p.relative_to(stage)): checksum(p) for p in sorted(stage.rglob('*')) if p.is_file()}
            portable['products'] = {TOKEN + p: digest for p, digest in files.items()}
            data = {'schema': SCHEMA, 'frame': portable, 'files': files,
                    'inputs': {key: collector.convert(record['path']) for key, record in inputs.items()},
                    'aoi': _walk(aoi, collector.convert),
                    'contract': {'matching': {key: cfg['matching'].get(key) for key in ('window', 'search', 'stride', 'first_center')},
                                 'geocoding': {key: cfg['geocoding'][key] for key in ('epsg', 'spacing_m')}},
                    'origin': {'geometry_identity': frame['geometry_identity'], 'layout_identity': frame['layout_identity'],
                               'reference_date': original_specification['anchor']['reference_date'],
                               'software': frame.get('software', {})}}
            data['fingerprint'] = identity(data)
            write_json(stage / MANIFEST, data)
            read_bundle(stage)
            # The source manifest and directory inputs must also remain unchanged.
            checked = set()
            for name in collector.mapping:
                real = Path(name).resolve()
                if real in checked:
                    continue
                checked.add(real)
                if Path(name).is_file() and collector.digests[name] != checksum(name):
                    raise ValueError(f'Reference input changed while exporting: {name}')
            for record in inputs.values():
                if checksum(record['path']) != record['sha256']:
                    raise ValueError('Reference inputs changed during export')
            if destination.exists():
                raise ValueError('Reference export destination already exists')
            stage.rename(destination)
    logging.info('Exported portable reference to %s', destination)
    return {'output': str(destination), 'fingerprint': data['fingerprint'], 'files': len(files),
            'reference_date': original_specification['anchor']['reference_date']}
