"""Retain or exclude mixed-swath sources without discarding other validity flags."""
from pathlib import Path
import json

import h5py
import numpy as np

from .velocity_io import digest,revision


SUPPORT_FILE='mixed_window_support.h5'


def pair_validity(package,slot,rows,cols,data,early,late):
    """Retain raw mixed-window radar values when only swath correction is unavailable."""
    mixed=(package['cross_swath_mixed'][slot][rows,cols]==1 if 'cross_swath_mixed' in package
           else np.zeros(len(rows),bool))
    good=((package['horizontal_valid_mask'][slot][rows,cols]==1)&np.isfinite(data).all(-1)
          &np.isfinite(early+late)&(late>early))
    if 'correction_available' in package:
        good&=(package['correction_available'][slot][rows,cols]==1)|mixed
    if 'tidal_correction_available' in package:
        good&=package['tidal_correction_available'][slot][rows,cols]==1
    return good,mixed


def prepare_mixed_support(out,remove_mixed_windows=False):
    """Recover only the masks lost from legacy caches; leave cached observations untouched."""
    out=Path(out)
    with h5py.File(out/'observations.h5') as h:
        mixed=h['known_mixed_source'][:] if 'known_mixed_source' in h else np.zeros(len(h['source_id']),bool)
        count=int(mixed.sum())
        policy={'remove_mixed_windows':remove_mixed_windows,'known_mixed_sources':count,
                'validity':'cached','support_sha256':None}
        if remove_mixed_windows or not count or 'valid_including_mixed' in h:return policy
        policy['validity']='recovered original masks for legacy mixed sources'
        cache_sha=digest(out/'observations.h5');metadata_sha=digest(out/'inputs.json');support=out/SUPPORT_FILE
        if not support.exists():
            metadata=json.loads((out/'inputs.json').read_text())
            packages=[Path(item['path']) for item in metadata.get('inputs',[])]
            if not packages:raise ValueError('Legacy mixed-window recovery needs original package paths in inputs.json')
            for path in packages:
                if not path.is_file():
                    raise ValueError(f'Cannot retain mixed windows from this legacy cache: original package is unavailable: {path}. '
                                     'Restore that package or use a new cache containing valid_including_mixed.')
            index=np.flatnonzero(mixed);rows=h['row'][:][index];cols=h['col'][:][index]
            ids=h['source_id'][:][index];files=h['file_index'][:];pair_ids=h['pair_id'].asstr()[:]
            early=h['early'][index];late=h['late'][index];data=h['data'][index]
            good=np.zeros(early.shape,bool)
            # Verify exact source packages before using their original validity masks.
            for fi,(path,item) in enumerate(zip(packages,metadata['inputs'])):
                before=revision(path)
                if digest(path)!=item['sha256']:raise ValueError(f'Original package changed: {path}')
                with h5py.File(path) as package:
                    if not np.array_equal(package['source_observation_index'][:][rows,cols],ids):
                        raise ValueError('Original mixed-window source associations changed')
                    lookup={pair:j for j,pair in enumerate(package['pair_id'].asstr()[:])}
                    for ti in np.flatnonzero(files==fi):
                        if pair_ids[ti] not in lookup:raise ValueError('Original mixed-window pair associations changed')
                        good[:,ti],_=pair_validity(package,lookup[pair_ids[ti]],rows,cols,data[:,ti],early[:,ti],late[:,ti])
                if revision(path)!=before:raise ValueError(f'Original package changed during mask recovery: {path}')
            temporary=out/'mixed_window_support.partial.h5'
            with h5py.File(temporary,'w') as restored:
                restored.attrs.update(status='complete',observations_sha256=cache_sha,inputs_sha256=metadata_sha)
                restored['source_id']=ids;restored['pair_id']=np.array(pair_ids,dtype=h5py.string_dtype())
                restored.create_dataset('valid_including_mixed',data=good,compression='gzip')
            temporary.replace(support)
        with h5py.File(support) as restored:
            if (restored.attrs.get('status')!='complete' or restored.attrs.get('observations_sha256')!=cache_sha
                    or restored.attrs.get('inputs_sha256')!=metadata_sha
                    or not np.array_equal(restored['source_id'][:],h['source_id'][:][mixed])
                    or not np.array_equal(restored['pair_id'].asstr()[:],h['pair_id'].asstr()[:])):
                raise ValueError('Mixed-window support does not match this observation cache')
        policy['support_sha256']=digest(support)
        return policy


def check_mixed_support(out,recipe):
    expected=recipe.get('mixed_window_policy',{}).get('support_sha256')
    if expected is not None and digest(Path(out)/SUPPORT_FILE)!=expected:
        raise ValueError('Mixed-window support changed')


def selected_validity(h,index,remove_mixed_windows=False):
    """Read the requested policy, including legacy support when necessary."""
    name='valid_including_mixed' if 'valid_including_mixed' in h else 'valid'
    valid=h[name][index].copy()
    mixed=h['known_mixed_source'][index] if 'known_mixed_source' in h else np.zeros(valid.shape[0],bool)
    if remove_mixed_windows:
        valid[mixed]=False
    elif mixed.any() and name=='valid':
        support=Path(h.filename).parent/SUPPORT_FILE
        if not support.is_file():
            raise ValueError('Legacy cache lacks retained mixed-window masks; run the inversion in a new output directory to recover them')
        with h5py.File(support) as restored:
            ids=restored['source_id'][:];wanted=h['source_id'][index][mixed];take=np.searchsorted(ids,wanted)
            if np.any(take>=len(ids)) or not np.array_equal(ids[take],wanted):raise ValueError('Mixed-window source IDs changed')
            valid[mixed]=restored['valid_including_mixed'][:][take]
    return valid
