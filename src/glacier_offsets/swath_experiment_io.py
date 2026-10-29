"""Experiment-local identities and read-only snapshots of completed stack inputs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
from pathlib import Path
import os
import tempfile

import numpy as np

from glacier_offsets.products import ChecksumCache, identity, write_json

METHOD = 'larsen-unspliced-range2x-experiment-v1'
COHORTS = ('primary', 'native', 'peak03', 'peak05')


def read_json(path):
    return json.loads(Path(path).read_text())


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def file_revision(path):
    return ChecksumCache._revision(Path(path).stat())


class Inputs:
    """Verify registered files and reject changes during a calibration invocation."""
    def __init__(self):
        self.hashes = ChecksumCache()
        self.revisions = {}
        self.products = {}

    def file(self, path, expected=None):
        path = str(Path(path).resolve(strict=True))
        digest = self.hashes(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input checksum mismatch: {path}')
        revision = file_revision(path)
        if path in self.revisions and self.revisions[path] != revision:
            raise ValueError(f'Input changed during experiment: {path}')
        self.revisions[path] = revision
        self.products[path] = digest
        return digest

    def unchanged(self):
        for path, revision in self.revisions.items():
            if file_revision(path) != revision:
                raise ValueError(f'Input changed during experiment: {path}')


def source_identity():
    root = Path(__file__).resolve().parent
    files = [root/'swath_calibration.py', root/'swath_calibration_export.py', *root.glob('swath_experiment_*.py')]
    files += [root/name for name in ('swath_observations.py', 'swath_overlap_support.py',
                                    'swath_physics.py', 'swath_models.py', 'range_oversampling.py',
                                    'ampcor.py', 'fftw_cache.py', 'matching_grid.py', 'geocode.py')]
    hashes = ChecksumCache()
    return {str(p): hashes(p) for p in sorted(set(files)) if p.exists()}


def cohorts(data):
    base = data['dual_valid'] & data['burst_interior'] & data['geometry_eligible']
    flags = []
    for swath in ('iw1', 'iw2'):
        flags.append(((data[f'{swath}_peak'] >= .3) & (data[f'{swath}_snr'] >= 8)) |
                     (data[f'{swath}_snr'] >= 20))
    return {'native': base, 'primary': base & flags[0] & flags[1],
            'peak03': base & (data['iw1_peak'] >= .3) & (data['iw2_peak'] >= .3),
            'peak05': base & (data['iw1_peak'] >= .5) & (data['iw2_peak'] >= .5)}


class Experiment:
    def __init__(self, config, output, workers=4):
        from glacier_offsets.stack_config import load_stack_config
        self.config = Path(config).resolve()
        self.cfg = load_stack_config(self.config)
        self.stack = Path(self.cfg['output']['directory']).resolve()
        self.output = Path(output).resolve()
        if self.output == self.stack or self.stack in self.output.parents or self.output in self.stack.parents:
            raise ValueError('Experiment output must be outside the production stack and its ancestors')
        self.workers = workers
        if not 1 <= workers <= 4:
            raise ValueError('Choose between one and four workers')
        p = self.cfg['matching']
        if (p.get('engine', 'isce3'), p['backend'], p['deramp'], p.get('range_oversampling')) != ('isce3', 'cpu', 0, 2):
            raise ValueError('Experiment requires the production ISCE3 CPU deramp-0 range-2x treatment')
        if self.cfg['gross_offsets']['mode'] != 'zero':
            raise ValueError('This experiment requires zero gross offsets')
        if self.cfg['geocoding']['resampling'] != 'nearest':
            raise ValueError('Preview comparison requires production nearest-source mapping')
        self.inputs = Inputs()
        self.inputs.file(self.stack/'geometry/manifest.json')
        self.frame = read_json(self.stack/'geometry/manifest.json')
        if self.frame.get('status') != 'complete':
            raise ValueError('Pinned reference geometry is incomplete')
        self.inputs.file(self.stack/'stack_manifest.json')
        self.catalog = read_json(self.stack/'stack_manifest.json')
        inventory = sorted((x for x in self.catalog['inventory'] if x.get('eligible')), key=lambda x: x['date'])
        self.dates = [x['date'] for x in inventory]
        self.inventory = {x['date']: x for x in inventory}
        self.links = [{'pair': f'{a}_{self.dates[i+step]}', 'earlier': a,
                       'later': self.dates[i+step], 'step': step}
                      for i, a in enumerate(self.dates) for step in (1, 2) if i+step < len(self.dates)]
        self.sources = source_identity()
        self.spec = {'method': METHOD, 'config': self.cfg,
                     'geometry_identity': self.frame['geometry_identity'],
                     'layout_identity': self.frame['layout_identity'], 'dates': self.dates,
                     'links': self.links, 'source_hashes': self.sources}
        # Geometry/configuration define the experiment; stage-local source hashes
        # invalidate only affected work when implementation changes.
        self.key = identity({k: v for k, v in self.spec.items() if k != 'source_hashes'})
        self.output.mkdir(parents=True, exist_ok=True)

    def implementation(self, *names):
        return {p: digest for p, digest in self.sources.items() if Path(p).name in names}

    def checked_report(self, path, complete=True):
        self.inputs.file(path)
        record = read_json(path)
        spec = record.get('specification', {})
        if spec.get('experiment') != self.key or (complete and record.get('status') != 'complete'):
            raise ValueError(f'Stale or incomplete experiment artifact: {path}')
        if record.get('identity') != identity(spec):
            raise ValueError(f'Inconsistent experiment artifact identity: {path}')
        for source, expected in spec.get('implementation', {}).items():
            if self.inputs.hashes(source) != expected:
                raise ValueError(f'Artifact implementation changed; regenerate {path}')
        for product, digest in record.get('products', {}).items():
            self.inputs.file(product, digest)
        if 'alignments' in spec and 'link' in spec:
            alignments = [self.verify_acquisition(spec['link'][key])['alignment_identity']
                          for key in ('earlier', 'later')]
            if alignments != spec['alignments']:
                raise ValueError(f'Underlying alignments changed; regenerate {path}')
        if 'measurement' in spec:
            measured = self.checked_report(self.output/'measurements'/Path(path).parent.name/'report.json')
            if measured['identity'] != spec['measurement']:
                raise ValueError(f'Underlying measurement changed; regenerate {path}')
        if 'audit' in spec:
            audited = self.checked_report(self.output/'audit.json')
            if audited['identity'] != spec['audit']:
                raise ValueError(f'Underlying audit changed; regenerate {path}')
        for name, directory in (('physics_reports', 'physics'), ('measurements', 'measurements')):
            for pair, digest in spec.get(name, {}).items():
                dependency = self.output/directory/pair/'report.json'
                self.inputs.file(dependency, digest)
                self.checked_report(dependency, complete=False)
        if 'physics_sha256' in spec:
            self.inputs.file(self.output/'physics_status.json', spec['physics_sha256'])
        return record

    @contextmanager
    def lock(self):
        from .stack_lock import stack_lock
        # Protect retained alignment inputs against concurrent stack cleanup or
        # rebuilding throughout measurement and subsequent verification.
        with stack_lock(self.stack), result_lock(self.output):
            self.inputs.unchanged()
            yield

    def acquisition(self, day):
        record = read_json(self.stack/'acquisitions'/day/'manifest.json')
        if record.get('status') not in ('aligned', 'complete') or not record.get('alignment_identity'):
            raise ValueError(f'{day}: alignment is not complete')
        if identity(record['specification']) != record['alignment_identity']:
            raise ValueError(f'{day}: inconsistent alignment identity')
        if record['specification']['geometry'] != self.frame['geometry_identity']:
            raise ValueError(f'{day}: different pinned geometry')
        return record

    def verify_geometry(self):
        for path, digest in self.frame['products'].items():
            self.inputs.file(path, digest)

    def verify_acquisition(self, day):
        record = self.acquisition(day)
        for path, digest in record['alignment_products'].items():
            self.inputs.file(path, digest)
        return record

    def current(self, path, specification):
        path = Path(path)
        if not path.exists():
            return False
        record = read_json(path)
        if record.get('identity') != identity(specification) or record.get('status') != 'complete':
            return False
        for product, digest in record.get('products', {}).items():
            if not Path(product).is_file() or self.inputs.hashes(product) != digest:
                return False
        return True

    def finish(self, path, specification, report, products=()):
        self.inputs.unchanged()
        record = dict(report, identity=identity(specification), specification=specification,
                      completed_utc=datetime.now(timezone.utc).isoformat(),
                      products={str(Path(p).resolve()): self.inputs.hashes(p) for p in products})
        write_json(path, record)
        return record

    def plan(self):
        previous = self.output/'plan.json'
        if previous.exists() and read_json(previous).get('identity') != self.key:
            raise ValueError('Experiment specification changed; select a new output directory')
        versions = {}
        for name in ('numpy', 'scipy', 'h5py', 'compass', 's1reader', 'isce3'):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = self.frame['software']['packages'].get(name, {}).get('version')
        record = {'identity': self.key, 'specification': self.spec, 'status': 'complete',
                  'versions': versions, 'workers': self.workers, 'links': self.links,
                  'planned_cycle_count': max(0, len(self.links)-len(self.dates)+1),
                  'validation': ['leave_acquisition_out', 'leave_link_out', 'leave_burst_group_out'],
                  'bootstrap': {'draws': 1000, 'seed': 0, 'shared_factors': 'acquisition and source burst'},
                  'primary_selection': 'native-valid, geometry eligible, burst interior; each swath (peak>=.3 & SNR>=8) | SNR>=20',
                  'preview_allocation': 'IW1=-delta/2, IW2=+delta/2; subtract in original pixels',
                  'preview_spatial_assumption': 'constant relative swath bias outside measured overlap'}
        record['reference_metadata'] = reference_metadata(self.frame, self.catalog)
        write_json(previous, record)
        return record


@contextmanager
def result_lock(output):
    with (Path(output)/'.writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another swath calibration writer is running') from exc
        yield


def reference_metadata(frame, catalog):
    """Small portable export context, independent of raw SAFE/orbit availability."""
    if (frame.get('status') != 'complete' or
            identity(frame['specification']) != frame['geometry_identity']):
        raise ValueError('Pinned reference geometry is incomplete or inconsistent')
    polarization = frame['specification']['anchor']['selection']['polarization']
    inventory = [entry for entry in catalog['inventory'] if entry.get('eligible')]
    if any(polarization not in entry.get('polarizations', [entry.get('polarization')]) for entry in inventory):
        raise ValueError('Reference polarization is unavailable in the calibration acquisitions')
    groups = {(entry['relative_orbit'], entry['direction']) for entry in inventory}
    if len(groups) != 1:
        raise ValueError('Calibration requires one compatible track, direction and polarization')
    track, direction = groups.pop()
    return {'geometry_identity': frame['geometry_identity'],
            'layout_identity': frame['layout_identity'],
            'reference_date': frame['specification']['anchor']['reference_date'],
            'reference_prf_hz': frame['coreg']['reference_grid']['prf'],
            'relative_orbit': track, 'direction': direction, 'polarization': polarization}
