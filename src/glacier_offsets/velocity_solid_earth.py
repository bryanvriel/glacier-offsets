"""Run the self-contained PySolid worker and publish its correction artifact.

The worker runs as a file under the selected interpreter (default: the current
one; ``execution.solid_earth_python`` overrides it), so PySolid's native
dependencies never enter this process. Corrections are a separate immutable
artifact bound to the observation cache, worker code, grid spacing and the
worker's dependency versions. They are subtracted exactly once at fit time.
"""
from importlib.util import find_spec
import json
import logging
from pathlib import Path
import subprocess
import sys

import h5py

from .velocity_cache import Artifact
from .velocity_io import digest


SOLID_EARTH = 'solid_earth.h5'


class Interrupted(RuntimeError):
    """A resumable stage stopped before completion."""


def worker_file():
    spec = find_spec('glacier_offsets.solid_earth_worker')
    if spec is None or spec.origin is None:
        raise ValueError('The installed package lacks solid_earth_worker.py')
    return Path(spec.origin)


def interpreter(execution):
    return execution.get('solid_earth_python') or sys.executable


def check_worker(python):
    """Confirm the worker interpreter can import pysolid, numpy and h5py."""
    if not Path(python).is_file():
        raise ValueError(f'PySolid interpreter not found: {python}; set execution.solid_earth_python')
    result = subprocess.run([python, '-I', str(worker_file()), '--versions'], capture_output=True, text=True)
    if result.returncode:
        detail = (result.stderr.strip().splitlines() or ['unknown error'])[-1]
        raise ValueError(f'PySolid worker cannot run under {python}: {detail}. '
                         'Install pysolid there or set execution.solid_earth_python to an interpreter that has it')
    return json.loads(result.stdout.strip().splitlines()[-1])


def solid_earth_artifact(cache_directory, observations, python, spacing):
    identity = {'stage': 'solid_earth', 'observations': observations.id, 'worker_sha256': digest(worker_file()),
                'spacing_degrees': float(spacing), 'dependencies': check_worker(python),
                'method': 'full ENU PySolid grid; bilinear space, bracketing UTC seconds; own package geometry'}
    return Artifact(cache_directory, 'solid_earth', identity)


def prepare_solid_earth(cache_directory, observations, execution, stop_after=None):
    python = interpreter(execution)
    spacing = execution['solid_earth_spacing_deg']
    artifact = solid_earth_artifact(cache_directory, observations, python, spacing)
    record = artifact.complete()
    if record is not None:
        logging.info('Reusing verified solid-Earth correction %s', artifact.id)
        return artifact, record
    with artifact.lock():
        record = artifact.complete()
        if record is not None:
            return artifact, record
        partial, resumed = artifact.begin()
        if resumed:
            logging.info('Resuming solid-Earth correction %s', artifact.id)
        command = [python, '-I', str(worker_file()), '--observations', str(observations.path('observations.h5')),
                   '--output', str(partial/SOLID_EARTH), '--spacing', repr(float(spacing)),
                   '--chunk', str(execution['preparation_chunk_pairs'])]
        if stop_after is not None:
            command += ['--stop-after', str(stop_after)]
        logging.info('Running PySolid worker with %s', python)
        process = subprocess.run(command, capture_output=True, text=True)
        for line in process.stdout.splitlines()[-1:]:
            logging.info('Solid-Earth worker: %s', line)
        if process.returncode == 130:
            raise Interrupted('Solid-Earth correction interrupted; rerun to resume')
        if process.returncode:
            raise ValueError(process.stderr.strip().splitlines()[-1] if process.stderr.strip() else 'Solid-Earth worker failed')
        with h5py.File(partial/SOLID_EARTH, 'r') as h:
            if h.attrs.get('status') != 'complete':
                raise ValueError('Solid-Earth worker did not complete its output')
            summary = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in h.attrs.items() if k != 'status'}
        record = artifact.publish([SOLID_EARTH], {'summary': summary, 'observations_sha256': observations.record()['files']['observations.h5'],
                                                  'ocean_loading': 'not applied; no local load-tide coefficients',
                                                  'formula': 'radar_corrected = radar - geometry_RA_ENU @ (SET_ENU(later) - SET_ENU(earlier))'})
        logging.info('Published solid-Earth correction %s (max sampled error %.3g m)', artifact.id,
                     summary.get('max_ENU_endpoint_difference_error_m', float('nan')))
        return artifact, record
