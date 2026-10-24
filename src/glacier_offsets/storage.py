"""Disposable merged SLCs with durable cleanup records and small QC previews."""
from pathlib import Path
import tempfile

import numpy as np
from osgeo import gdal

from .products import checksum, write_json

METHOD = 'merged-slc-release-v1'


def keep_merged_slcs(cfg):
    options = cfg.get('storage', {})
    if (not isinstance(options, dict) or set(options) - {'keep_merged_slcs'} or
            type(options.get('keep_merged_slcs', False)) is not bool):
        raise ValueError('storage.keep_merged_slcs must be true or false')
    return options.get('keep_merged_slcs', False)


def released_files(manifest, key, fingerprint):
    record = manifest.get('slc_cleanup', {})
    if (record.get('method') == METHOD and record.get('identity_key') == key and
            record.get('identity') == fingerprint and manifest.get(key) == fingerprint):
        return record.get('files', {})
    return {}


def products_valid(products, manifest, key, fingerprint, *, allow_released=False, hashes=True, checksums=None):
    if manifest.get(key) != fingerprint or not products:
        return False
    released = released_files(manifest, key, fingerprint) if allow_released else {}
    for name, digest in products.items():
        path = Path(name)
        if path.is_file():
            if hashes and (checksum if checksums is None else checksums)(path) != digest:
                return False
        elif path.exists() or released.get(name) != digest:
            return False
    return True


def cache_preview(root, slc, fingerprint):
    """Store the same bounded nearest-sample preview diagnostics used before."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root/'amplitude_preview.npz'
    ds = gdal.Open(str(slc))
    preview = ds.ReadAsArray(buf_xsize=1200, buf_ysize=300)
    ds = None
    with tempfile.NamedTemporaryFile(dir=root, prefix='.preview-', suffix='.npz', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, preview=preview)
            stream.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return {'path': str(target.resolve()), 'sha256': checksum(target),
            'source': str(Path(slc).resolve()), 'identity': fingerprint}


def preview_array(manifest, side):
    source = Path(manifest['coreg']['slcs'][side])
    if source.is_file():
        ds = gdal.Open(str(source))
        return ds.ReadAsArray(buf_xsize=1200, buf_ysize=300)
    saved = manifest.get('slc_previews', {}).get(side, {})
    path = Path(saved.get('path', ''))
    if (saved.get('source') != str(source.resolve()) or not path.is_file() or
            checksum(path) != saved.get('sha256')):
        raise ValueError('Merged SLC preview missing or changed; rerun assembly to restore diagnostic inputs')
    with np.load(path, allow_pickle=False) as data:
        return data['preview']


def release_slcs(root, manifest, key, slcs, products):
    """Journal exact owned files before unlinking; interrupted cleanup is resumable.

    Callers establish that all active matching consumers have committed offsets.
    This never traverses a directory or deletes any unregistered/coreg source.
    """
    root = Path(root).resolve()
    fingerprint = manifest[key]
    files = dict(released_files(manifest, key, fingerprint))
    for source in slcs:
        for path in (Path(source), Path(str(source) + '.msk')):
            name = str(path.resolve())
            if name not in products:
                if path.exists():
                    raise ValueError(f'Unregistered merged SLC cannot be cleaned: {path}')
                continue
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError(f'Merged SLC cleanup path is not an owned file: {path}')
            files[name] = products[name]
    if not files:
        return
    allowed = {str(Path(path).resolve()) for source in slcs
               for path in (source, str(source) + '.msk')}
    if set(files) - allowed or any(not Path(p).is_relative_to(root) or Path(p).is_symlink() for p in files):
        raise ValueError('Invalid merged SLC cleanup inventory')
    record = {'method': METHOD, 'identity_key': key, 'identity': fingerprint,
              'files': files, 'status': 'releasing'}
    manifest['slc_cleanup'] = record
    write_json(root/'manifest.json', manifest)
    for name in files:
        Path(name).unlink(missing_ok=True)
    record['status'] = 'released'
    write_json(root/'manifest.json', manifest)
