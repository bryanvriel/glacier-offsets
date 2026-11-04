"""Explicit recovery of a lost completion marker from published offset provenance."""
from datetime import datetime, timezone
import json
import logging

import h5py
import numpy as np

from .ampcor import matching_specification
from .assembly import source_indices
from .products import checksum, identity, write_json
from .stack_pairs import matching_template


def recover_matching(cfg, directory, manifest):
    """Called under the stack lock, after current geometry/acquisition checks.

    Recovery is explicit: ordinary restarts must not resurrect old results after
    an interrupted forced recomputation. This never executes native matching.
    """
    expected = matching_specification(cfg, manifest)
    fingerprint = identity(expected)
    path = directory/'radar/offsets.h5'
    if manifest.get('match_identity') == fingerprint and path.is_file():
        return
    if manifest.get('match_identity') is not None:
        raise ValueError('Existing matching completion differs; run stack match instead of recovery')
    digest = checksum(path)
    with h5py.File(path) as h:
        metadata = json.loads(h.attrs['metadata_json'])
        differences = [key for key, value in expected.items() if metadata.get(key) != value]
        if differences:
            raise ValueError(f'Saved offsets differ in {differences}; run stack match instead of recovery')
        for key, value in {
            'input_provenance': manifest['coreg_specification']['inputs'],
            'stack_pair': manifest['stack_pair'],
            'reference_grid': manifest['coreg']['reference_grid'],
            'source_burst_ids': manifest['assembly']['source_burst_ids'],
            'polarization': cfg['inputs']['polarization'],
        }.items():
            if metadata.get(key) != value:
                raise ValueError(f'Saved offsets {key} differs; matching cannot be recovered')
        if h.attrs.get('units') != 'full-resolution reference SLC pixels':
            raise ValueError('Saved offsets have unsupported units')
        grid = matching_template(cfg, manifest)
        for name, centers in zip(('center_line', 'center_sample'), grid.centers):
            np.testing.assert_array_equal(h[name][:], centers)
        fields = [f'{kind}/{axis}_px' for kind in ('gross', 'residual', 'total') for axis in ('azimuth', 'range')]
        fields += ['snr', 'correlation_peak', 'input_valid', 'finite', 'search_boundary', 'valid',
                   'rejection_flags', 'source_burst_index', 'interval_seconds', 'seam_crossing', 'seam_search_known']
        arrays = {}
        for name in (*fields, 'covariance'):
            array = h[name][:]  # Read every required dataset, including compressed chunks.
            shape = (*grid.shape, 3) if name == 'covariance' else grid.shape
            if array.shape != shape:
                raise ValueError(f'Saved offsets dataset {name} has an incompatible shape')
            arrays[name] = array
        np.testing.assert_array_equal(arrays['source_burst_index'], source_indices(manifest['coreg'], *grid.centers))
        intervals = np.full(grid.shape, np.nan)
        for index, source in enumerate(manifest['assembly']['sources'], 1):
            value = source.get('interval_seconds')
            intervals[arrays['source_burst_index'] == index] = np.nan if value is None else value
        np.testing.assert_array_equal(arrays['interval_seconds'], intervals)
        summary = {'windows': int(np.prod(grid.shape)),
                   'input_valid': int(arrays['input_valid'].sum()),
                   'quality_valid': int(arrays['valid'].sum()),
                   'seam_crossing': int(arrays['seam_crossing'].sum()),
                   'seam_crossing_valid': int((arrays['seam_crossing'] & arrays['valid']).sum())}
    if checksum(path) != digest:
        raise ValueError('Saved offsets changed during matching recovery')
    for stage in ('geocode', 'diagnostics'):
        manifest.pop(f'{stage}_identity', None)
        manifest['stages'].pop(stage, None)
    manifest.update(match_identity=fingerprint, matching=metadata, matching_summary=summary,
                    matching_recovery={'method': 'explicit-published-offset-provenance-v1',
                        'offsets_sha256': digest, 'recovered_utc': datetime.now(timezone.utc).isoformat()})
    manifest['stages']['match'] = 'complete'
    write_json(directory/'manifest.json', manifest)
    logging.info('Recovered matching completion for %s from unchanged offsets', directory.name)
