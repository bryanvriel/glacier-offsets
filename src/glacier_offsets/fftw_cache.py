"""Carry FFTW's measured plans across recycled workers in one invocation.

Only the native library's documented wisdom file API is used. The cache lives
in a temporary directory, on this host with this Python/native environment; it
is never distributed between machines or treated as a scientific product.
"""
import ctypes
import os
from pathlib import Path
import sys
import tempfile


def library():
    for name in ('libfftw3f.dylib', 'libfftw3f.so', 'libfftw3f.so.3'):
        path = Path(sys.prefix) / 'lib' / name
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            for function in ('fftwf_import_wisdom_from_filename', 'fftwf_export_wisdom_to_filename'):
                entry = getattr(lib, function)
                entry.argtypes = [ctypes.c_char_p]
                entry.restype = ctypes.c_int
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
