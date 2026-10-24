"""FFTW measured-plan (wisdom) files for native CPU Ampcor workers.

Only the native library's documented wisdom file API is used. Two caches exist:

- A temporary per-invocation cache that recycled workers share to save
  warm-up time. Its last-writer-wins publication can mix plan sets within a
  run, so it gives no reproducibility.
- Pinned stack wisdom. CPU Ampcor compiles in `FFTW_MEASURE`, so concurrently
  started workers time candidate algorithms independently and can choose
  plans that round differently, moving near-tied correlation peaks by one
  oversampled sample. Stack matching instead imports one wisdom file measured
  once by a quiet process and never republishes it, so runs on the same host
  and FFTW/ISCE3 build are bit-identical. The file is keyed by that host/build
  and the FFT-shaping matching parameters; it is never distributed between
  machines, treated as a scientific product or included in cache identities.
"""
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np

# Matching settings that determine native FFT plan shapes.
PLAN_PARAMETERS = ('window', 'search', 'batch', 'raw_oversampling', 'oversampling', 'zoom', 'deramp')
WISDOM_METHOD = 'fftw-measure-pinned-wisdom-v1'
_pinned_directory = None


def library():
    for name in ('libfftw3f.dylib', 'libfftw3f.so', 'libfftw3f.so.3'):
        path = Path(sys.prefix) / 'lib' / name
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            for function in ('fftwf_import_wisdom_from_filename', 'fftwf_export_wisdom_to_filename'):
                entry = getattr(lib, function)
                entry.argtypes = [ctypes.c_char_p]
                entry.restype = ctypes.c_int
            lib.fftwf_export_wisdom_to_string.restype = ctypes.c_void_p
            lib.fftwf_free.argtypes = [ctypes.c_void_p]
            lib.path = path
            return lib
    return None


def restore(path):
    lib = library()
    if lib is not None and Path(path).exists():
        if not lib.fftwf_import_wisdom_from_filename(str(path).encode()):
            raise RuntimeError('Cannot restore native FFTW plan cache')
    return lib


def save(lib, path):
    if lib is not None:
        # Workers have separate FFTW state; last writer wins only for warm-up
        # efficiency. Unique temporary files and atomic publication ensure that
        # every reader sees a complete cache, including during worker recycling.
        with tempfile.NamedTemporaryFile(prefix=Path(path).name + '.', suffix='.tmp',
                                         dir=Path(path).parent, delete=False) as stream:
            temporary = stream.name
        try:
            if not lib.fftwf_export_wisdom_to_filename(temporary.encode()):
                raise RuntimeError('Cannot save native FFTW plan cache')
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def export(lib):
    """The process's current wisdom as text."""
    pointer = lib.fftwf_export_wisdom_to_string()
    try:
        return ctypes.string_at(pointer).decode()
    finally:
        lib.fftwf_free(pointer)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _cpu_model():
    if sys.platform == 'darwin':
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def wisdom_key(params):
    """Host, build and plan-shaping settings that a wisdom file is valid for."""
    import isce3
    lib = library()
    if lib is None:
        raise RuntimeError('No shared single-precision FFTW library in this native environment')
    version = ctypes.c_char_p(ctypes.addressof(ctypes.c_char.in_dll(lib, 'fftwf_version'))).value.decode()
    return {'method': WISDOM_METHOD, 'fftw_version': version, 'fftw_library_sha256': _sha256(lib.path),
            'isce3_version': isce3.__version__, 'machine': platform.machine(), 'cpu': _cpu_model(),
            'matching': {name: params[name] for name in PLAN_PARAMETERS}}


@contextmanager
def pinned(directory):
    """Pin CPU stack matching to persistent wisdom stored under `directory`."""
    global _pinned_directory
    previous, _pinned_directory = _pinned_directory, Path(directory)
    try:
        yield
    finally:
        _pinned_directory = previous


def pinned_wisdom(params):
    """Return the pinned wisdom record for these settings, measuring it once.

    Returns None outside `pinned`. A missing file is measured by one quiet
    spawned process, under an exclusive lock, and published atomically; an
    existing file is reused unchanged.
    """
    if _pinned_directory is None:
        return None
    from .products import identity
    key = wisdom_key(params)
    digest = identity(key)
    directory = _pinned_directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{digest}.wisdom'
    with open(directory / f'{digest}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        created = not path.is_file()
        if created:
            text = measure(params)
            temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
            temporary.write_text(text)
            os.replace(temporary, path)
            record = path.with_suffix('.json')
            record.with_name(f'.{record.name}.tmp').write_text(json.dumps(key, indent=2, sort_keys=True))
            os.replace(record.with_name(f'.{record.name}.tmp'), record)
    return {'method': WISDOM_METHOD, 'path': str(path), 'key_identity': digest,
            'sha256': _sha256(path), 'measured_now': created}


def measure(params):
    """Measure every native chunk shape these settings can use, in one fresh process."""
    from .ampcor import _native_executor
    with _native_executor(1, single_threaded=True) as pool:
        return pool.apply(_measure_worker, (params,))


def _measure_worker(params):
    """Plan every chunk shape `min(batch, grid)` on synthetic rasters.

    FFTW_MEASURE chooses plans by timing; image content does not matter.
    """
    from osgeo import gdal
    from .ampcor import _native_match
    from .matching_grid import MatchingGrid
    lib = library()
    window, search, stride = params['window'], params['search'], params['stride']
    rows, columns = params['batch']
    margin = 8
    shape = [2*(search[k]+margin) + window[k] + ((rows, columns)[k]-1)*stride[k] for k in (0, 1)]
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory(prefix='glacier-offsets-wisdom-') as scratch:
        paths = []
        for name in ('reference', 'secondary'):
            path = Path(scratch) / f'{name}.tif'
            ds = gdal.GetDriverByName('GTiff').Create(str(path), shape[1], shape[0], 1, gdal.GDT_CFloat32,
                                                      options=['TILED=NO'])
            ds.GetRasterBand(1).WriteArray((rng.standard_normal(shape) + 1j*rng.standard_normal(shape)).astype('complex64'))
            ds = None
            paths.append(path)
        for height in range(1, rows+1):
            for width in range(1, columns+1):
                grid = MatchingGrid(np.arange(height, dtype=np.int32)*stride[0] + search[0] + margin,
                                    np.arange(width, dtype=np.int32)*stride[1] + search[1] + margin,
                                    tuple(window), tuple(search), tuple(stride))
                _native_match(*paths, Path(scratch) / f'{height}x{width}', params, {'gross_offsets': {'mode': 'zero'}},
                              'cpu', _grid=grid, _valid_input=np.ones(grid.shape, bool))
    return export(lib)
