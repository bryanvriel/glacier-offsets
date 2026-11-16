"""GOFF observation caches, geometry-aware screening and grouped noise."""
import json
import logging
from time import perf_counter
from pathlib import Path

import h5py
import numpy as np
from pyproj import CRS, Transformer

from .nisar_collection import NisarCollection
from .velocity_cache import Artifact
from .velocity_io import digest, write_json


def prepare_nisar_observations(cache, manifest, chunk, hook):
    artifact = Artifact(cache, 'observations', {'schema': 'nisar-observations-v4-compact', 'collection': manifest,
                        'implementation': {k: digest(Path(__file__).with_name(k)) for k in ('nisar_velocity.py', 'nisar_collection.py', 'nisar_validation.py', 'displacement.py')}})
    record = artifact.complete()
    if record:
        logging.info('Reusing verified NISAR observations %s', artifact.id)
        return artifact, record
    with artifact.lock():
        record = artifact.complete()
        if record: return artifact, record
        partial, resumed = artifact.begin(); path = partial/'observations.h5'
        with NisarCollection([i['path'] for i in manifest['inputs']], cache_directory=cache,
                             verified_inputs=manifest['inputs']) as c:
            nt = len(c.members)
            if not path.exists():
                with h5py.File(path, 'w') as out:
                    out.attrs.update(status='writing', completed_pairs=0, schema='glacier-offsets-velocity-observations-v4',
                                     mission='NISAR', EPSG=3031, crs_wkt=CRS.from_epsg(3031).to_wkt(),
                                     order_spacecraft_json='{}', spatial_support='correlated geocoded map locations')
                    sections = list(c.sections(256))
                    mapping = np.full(c.shape, -1, dtype='i8')
                    flat_ids = []
                    ns = 0
                    started = last = perf_counter()
                    logging.info('Finding supported NISAR locations in %d tiles', len(sections))
                    for number, sec in enumerate(sections):
                        present = np.zeros(tuple(a.stop-a.start for a in sec), bool)
                        for r in c.members:
                            if 'horizontal_valid_mask' in c.handles[r['file']]:
                                good = ((c.read(r, 'valid_mask', sec) == 1)
                                        & (c.read(r, 'horizontal_valid_mask', sec) == 1)
                                        & (c.read(r, 'tidal_correction_available', sec) == 1))
                            else:
                                good = c.tile(sec, [c.members.index(r)], radar_only=True)['valid'][0]
                            present |= good
                        rr, cc = np.nonzero(present)
                        ids = (rr+sec[0].start)*c.shape[1]+cc+sec[1].start
                        flat_ids.extend(ids)
                        mapping[sec][present] = np.arange(ns, ns+len(ids))
                        ns += len(ids)
                        if number == 0 or number+1 == len(sections) or perf_counter()-last >= 5:
                            logging.info('NISAR support tiles %d/%d; %d locations (%.1fs)', number+1, len(sections), ns, perf_counter()-started)
                            last = perf_counter()
                    if not ns:
                        raise ValueError('No usable NISAR observation locations')
                    ids = np.array(flat_ids, dtype='i8')
                    rows, cols = np.divmod(ids, c.shape[1])
                    lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(c.x[cols], c.y[rows])
                    out.attrs['map_cells'] = int(np.prod(c.shape))
                    out.attrs['supported_locations'] = ns
                    out.attrs['location_order'] = '256-cell spatial tiles; original row-major map IDs retained'
                    for k, a in [('source_id', ids), ('row', rows), ('col', cols),
                                 ('owner', np.zeros(ns, int)), ('x', c.x), ('y', c.y), ('latitude', lat), ('longitude', lon),
                                 ('map_source_index', mapping), ('known_mixed_source', np.zeros(ns, bool))]: out[k] = a
                    out['crs'] = np.int32(0); out['crs'].attrs.update(dict(c.handles[0]['crs'].attrs))
                    out['burst_id'] = np.array([], dtype=h5py.string_dtype())
                    out['pair_id'] = np.array([r['pair_id'] for r in c.members], dtype=h5py.string_dtype())
                    out['noise_group'] = np.array([r['noise_group'] for r in c.members], dtype=h5py.string_dtype())
                    for k in ('reference_acquisition', 'secondary_acquisition'):
                        out[k] = np.array([r[k] for r in c.members], dtype=h5py.string_dtype())
                    out['track'] = [r['track'] for r in c.members]
                    out['file_index'] = [r['file'] for r in c.members]; out['slot'] = [r['slot'] for r in c.members]
                    out['order'] = np.zeros(nt, 'i1'); out['year'] = [r['year'] for r in c.members]; out['years'] = manifest['years']
                    out['year_index'] = [manifest['years'].index(r['year']) for r in c.members]
                    for k in ('calendar_early', 'calendar_late', 'within_year'):
                        src = k+'_day' if k.startswith('calendar_') else k
                        out[k] = [r[src] for r in manifest['pairs']]
                    out['input_sha256'] = np.array([i['sha256'] for i in manifest['inputs']], dtype=h5py.string_dtype())
                    for k in ('height_m', 'tidal_response_factor', 'azimuth_ground_spacing_m_per_slc_line'):
                        # Location summaries are ancillary; scientific geometry remains pair-specific.
                        arr = c.read(c.members[0], k, (slice(0, c.shape[0]), slice(0, c.shape[1])))
                        out[k] = arr[rows, cols]
                    options = dict(compression='gzip', compression_opts=2)
                    for k in ('early', 'late', 'peak', 'snr'):
                        out.create_dataset(k, (ns, nt), dtype='f8', chunks=(min(2048, ns), min(chunk, nt)), fillvalue=np.nan, **options)
                    for k in ('valid_including_mixed', 'mixed_window', 'overlap_suppressed'):
                        out.create_dataset(k, (ns, nt), dtype='?', chunks=(min(2048, ns), min(chunk, nt)), **options)
                    out.create_dataset('data', (ns, nt, 2), dtype='f8', chunks=(min(2048, ns), min(chunk, nt), 2), fillvalue=np.nan, **options)
                    out.create_dataset('geometry_by_pair', (ns, nt, 2, 3), dtype='f8', chunks=(min(2048, ns), min(chunk, nt), 2, 3), fillvalue=np.nan, **options)
            with h5py.File(path, 'r+') as out:
                ns = len(out['source_id'])
                mapping = out['map_source_index'][:]
                start = int(out.attrs['completed_pairs'])
                logging.info('Preparing NISAR observations: %d supported / %d map locations, %d products; resume at %d',
                             ns, np.prod(c.shape), nt, start)
                sections = [sec for sec in c.sections(256) if (mapping[sec] >= 0).any()]
                # Read bounded spatial tiles; publication checkpoints only fully populated pair chunks.
                for begin in range(start, nt, chunk):
                    stop = min(begin+chunk, nt)
                    started = last = perf_counter()
                    for number, sec in enumerate(sections):
                        a = c.tile(sec, range(begin, stop), radar_only=True)
                        keep = mapping[sec] >= 0
                        loc = mapping[sec][keep]
                        # Compact locations are contiguous within a spatial tile.
                        target = slice(int(loc[0]), int(loc[-1])+1)
                        if target.stop-target.start != len(loc):
                            raise ValueError('Invalid compact observation mapping')
                        for k, source in [('early', 'early'), ('late', 'late'), ('peak', 'peak'), ('snr', 'snr'), ('valid_including_mixed', 'valid'), ('overlap_suppressed', 'overlap_suppressed')]:
                            out[k][target, begin:stop] = a[source][:, keep].T
                        out['data'][target, begin:stop] = np.moveaxis(a['data_ra'][:, keep], 0, 1)
                        out['geometry_by_pair'][target, begin:stop] = np.moveaxis(a['geometry'][:, keep], 0, 1)
                        if number == 0 or number+1 == len(sections) or perf_counter()-last >= 5:
                            logging.info('NISAR pairs %d-%d/%d, tile %d/%d (%.1fs)', begin+1, stop, nt, number+1, len(sections), perf_counter()-started)
                            last = perf_counter()
                    out.attrs['completed_pairs'] = stop; out.flush()
                    if hook: hook('observations', stop)
                out.attrs['status'] = 'complete'
            c.check_revisions()
        return artifact, artifact.publish(['observations.h5'], {'manifest': manifest, 'summary': {'map_cells': int(np.prod(c.shape)), 'supported_locations': int(ns), 'pairs': nt}})


def _purged_predictions_reference(field, allowed=None):
    """Held-out annual EN backgrounds projected into each tested radar frame."""
    from .velocity_pixel_general import pair_geometry, Layout
    geom = pair_geometry(field); early, late = field['early'], field['late']
    years = field['year_index']; ny = int(years.max())+1
    # Exact annual overlaps are supplied by the caller's actual year layout.
    overlap = field.get('overlap')
    if overlap is None:
        overlap = np.stack([(late-early)*(years == k)[None] for k in range(ny)], -1)
    ns, nt = early.shape; predictions = np.full((ns, nt, 2), np.nan)
    eligible = field['valid'] & np.isfinite(field['peak']) & (field['peak'] >= field.get('min_peak', .08))
    if allowed is not None: eligible &= np.asarray(allowed)[None]
    # Work source by source, avoiding a source x pair x pair design tensor.
    for s in range(ns):
        design = (geom[s, :, :, :2, None]*overlap[s, :, None, None, :]).transpose(0, 1, 3, 2).reshape(nt, 2, 2*ny)
        for t in range(nt):
            shared = np.floor([early[s, t], late[s, t]])
            training = eligible[s] & ~np.isin(np.floor(early[s]), shared) & ~np.isin(np.floor(late[s]), shared)
            if 'reference_acquisition' in field:
                acq = [field['reference_acquisition'][t], field['secondary_acquisition'][t]]
                training &= ~np.isin(field['reference_acquisition'], acq) & ~np.isin(field['secondary_acquisition'], acq)
            if training.sum() < 3: continue
            x, y = design[training].reshape(-1, 2*ny), field['data'][s, training].reshape(-1)
            if not np.isfinite(x).all() or not np.isfinite(y).all(): continue
            active = np.any(abs(x) > 1e-12, axis=0)
            if not active.any() or np.linalg.matrix_rank(x[:, active]) < active.sum(): continue
            weights = np.ones(len(x)); coef = np.zeros(2*ny)
            for _ in range(5):
                coef[active] = np.linalg.lstsq(x[:, active]*weights[:, None], y*weights, rcond=None)[0]
                residual = y-x@coef; scale = max(1e-4, 1.4826*np.median(abs(residual-np.median(residual))))
                weights = np.sqrt(np.minimum(1., 2.45*scale/np.maximum(abs(residual), 1e-12)))
            predictions[s, t] = design[t]@coef
    return predictions


def purged_predictions(field, allowed=None, *, valid_only=False):
    """Batched held-out IRLS; retain the scalar SVD solve for ill-conditioned cases.

    Every held-out observation has exactly the original date/acquisition purge,
    training support, rank test, five IRLS updates and MAD floor. Well-conditioned
    systems use batched normal equations; ambiguous systems replay the reference.
    """
    from .velocity_pixel_general import pair_geometry
    geom = pair_geometry(field)
    early, late = field['early'], field['late']
    ns, nt = early.shape
    ny = int(field['year_index'].max())+1
    overlap = field.get('overlap')
    if overlap is None:
        overlap = np.stack([(late-early)*(field['year_index'] == k)[None] for k in range(ny)], -1)
    design = (geom[..., :2, None]*overlap[:, :, None, None, :]).transpose(0, 1, 2, 4, 3).reshape(ns, nt, 2, 2*ny)
    eligible = field['valid'] & np.isfinite(field['peak']) & (field['peak'] >= field.get('min_peak', .08))
    if allowed is not None: eligible &= np.asarray(allowed)[None]
    # Empty locations need no held-out fits at all.
    sources = np.flatnonzero(eligible.sum(axis=1) >= 3)
    predictions = np.full((ns, nt, 2), np.nan)
    if not len(sources): return predictions
    d = design[sources]; data = field['data'][sources]
    e, l = np.floor(early[sources]), np.floor(late[sources])
    fallback = np.zeros(ns, bool)
    for t in range(nt):
        training = eligible[sources] & (e != e[:, t, None]) & (e != l[:, t, None]) & (l != e[:, t, None]) & (l != l[:, t, None])
        if 'reference_acquisition' in field:
            ref, sec = field['reference_acquisition'], field['secondary_acquisition']
            training &= ~np.isin(ref, [ref[t], sec[t]]) & ~np.isin(sec, [ref[t], sec[t]])
        finite = np.isfinite(d).all(axis=(-1, -2)) & np.isfinite(data).all(axis=-1)
        usable = (training.sum(axis=1) >= 3) & np.all(~training | finite, axis=1)
        if valid_only:
            usable &= field['valid'][sources, t] & np.isfinite(field['peak'][sources, t]) & (field['peak'][sources, t] >= field.get('min_peak', .08))
        chosen = np.flatnonzero(usable)
        if not len(chosen): continue
        mask = np.repeat(training[chosen], 2, axis=1)
        x = np.where(mask[..., None], d[chosen].reshape(len(chosen), -1, 2*ny), 0.)
        y = np.where(mask, data[chosen].reshape(len(chosen), -1), 0.)
        active = np.any(abs(x) > 1e-12, axis=1)
        x = np.where(active[:, None], x, 0.)
        singular = np.linalg.svd(x, compute_uv=False)
        na = active.sum(axis=1)
        smallest = singular[np.arange(len(chosen)), np.maximum(na-1, 0)]
        cutoff = singular[:, 0]*np.maximum(mask.sum(axis=1), na)*np.finfo(float).eps
        full_rank = (na > 0) & (smallest > cutoff)
        stable = smallest > singular[:, 0]*1e-4
        fallback[sources[chosen[full_rank & ~stable]]] = True
        keep = full_rank & stable
        if not keep.any(): continue
        chosen, x, y, mask, active = chosen[keep], x[keep], y[keep], mask[keep], active[keep]
        weights = np.ones(y.shape)
        coef = np.zeros((len(chosen), 2*ny))
        for _ in range(5):
            wx = x*weights[..., None]
            gram = wx.swapaxes(1, 2)@wx
            rhs = np.einsum('smp,sm->sp', wx, y*weights)
            gram += np.eye(2*ny)[None]*(~active)[:, None]
            eig = np.linalg.eigvalsh(gram)
            unsafe = eig[:, 0] <= eig[:, -1]*1e-8
            fallback[sources[chosen[unsafe]]] = True
            safe_gram = np.where(unsafe[:, None, None], np.eye(2*ny), gram)
            coef = np.linalg.solve(safe_gram, rhs[..., None])[..., 0]
            residual = np.where(mask, y-np.einsum('smp,sp->sm', x, coef), np.nan)
            median = np.nanmedian(residual, axis=1)
            scale = np.maximum(1e-4, 1.4826*np.nanmedian(abs(residual-median[:, None]), axis=1))
            weights = np.where(mask, np.sqrt(np.minimum(1., 2.45*scale[:, None]/np.maximum(abs(residual), 1e-12))), 0.)
        predictions[sources[chosen], t] = np.einsum('scp,sp->sc', d[chosen, t], coef)
    if fallback.any():
        # Reference keeps LAPACK's least-squares arithmetic for difficult geometry.
        subset = dict(field)
        for key in ('early', 'late', 'data', 'valid', 'peak', 'geometry_by_pair', 'overlap'):
            if key in subset: subset[key] = subset[key][fallback]
        subset['geometry_by_pair'] = geom[fallback]
        predictions[fallback] = _purged_predictions_reference(subset, allowed)
    if valid_only:
        take = field['valid'] & np.isfinite(field['peak']) & (field['peak'] >= field.get('min_peak', .08))
        predictions[~take] = np.nan
    return predictions


def screen_nisar(field, layout, screening):
    from .velocity_pixel_general import pair_geometry, _angles
    geom = pair_geometry(field)
    enriched = field | {'overlap': layout.overlaps(field['early'], field['late']), 'min_peak': screening['min_peak']}
    prediction = purged_predictions(enriched, valid_only=True)
    residual = field['data']-prediction
    good = (field['valid'] & np.isfinite(field['peak']) & (field['peak'] >= screening['min_peak'])
            & np.isfinite(residual).all(-1) & (abs(residual[..., 0]) <= screening['max_range_residual_m'])
            & (abs(residual[..., 1]) <= screening['max_azimuth_residual_m']))
    velocities = []
    for k in range(len(layout.years)):
        take = field['year_index'] == k
        rates = field['data'][:, take]/(field['late'][:, take]-field['early'][:, take])[..., None]
        matrices = geom[:, take, :, :2]
        safe = np.where(np.isfinite(matrices).all(axis=(-1, -2))[..., None, None], matrices, np.eye(2))
        en = np.einsum('stij,stj->sti', np.linalg.pinv(safe), rates)
        from .velocity_quality import quiet_median
        velocities.append(quiet_median(np.where(good[:, take, None], en, np.nan), axis=1))
    velocities = np.array(velocities); mean = np.mean(velocities, axis=0); speed = np.linalg.norm(mean, axis=-1)
    axis = np.divide(mean, speed[:, None], out=np.tile([1., 0.], (len(mean), 1)), where=np.isfinite(speed[:, None]) & (speed[:, None] > 1e-8))
    return good, axis, velocities, speed, _angles(velocities)


def grouped_noise(config, observations, solid_earth, years, partial):
    from .velocity_annual_noise import check_covariance, select_controls, control_regions
    from .velocity_pixel_general import load_field, Layout
    from .velocity_fit import calibrate_noise
    with h5py.File(observations.path('observations.h5')) as h:
        group = h['noise_group'][:].astype(str); pair_ids = h['pair_id'][:].astype(str).tolist()
        labels = sorted(set(group)); by_group = {}; sources = []
        if config['noise']['mode'] == 'supplied':
            for entry in config['noise']['files']:
                record = json.loads(Path(entry['path']).read_text()); sources.append({'path': entry['path'], 'sha256': digest(entry['path'])})
                for label, cov in record.get('covariance_ra_m2_by_group', {}).items():
                    if label in by_group: raise ValueError(f'Duplicate supplied covariance group {label}')
                    by_group[label] = check_covariance(cov, label)
            missing = set(labels)-by_group.keys()
            if missing: raise ValueError(f'Missing track/year covariance group(s): {sorted(missing)}')
        else:
            _, regions = select_controls(observations.path('observations.h5'), config['noise']['regions'], config['noise']['block_size_m'])
            bank = {label: [[], [], [], []] for label in labels}
            for region in regions:
                if not len(region['index']): continue
                body = h5py.File(solid_earth.path('solid_earth.h5')) if solid_earth else None
                try: field = load_field(h, body, region['index'])
                finally:
                    if body: body.close()
                field['overlap'] = Layout.from_observations(h).overlaps(field['early'], field['late'])
                field['min_peak'] = config['screening']['min_peak']
                residual = field['data']-purged_predictions(field, valid_only=True)
                for label in labels:
                    take = group == label
                    good = field['valid'][:, take] & np.isfinite(residual[:, take]).all(-1) & (field['peak'][:, take] >= field['min_peak'])
                    good &= (abs(residual[:, take, 0]) <= config['screening']['max_range_residual_m']) & (abs(residual[:, take, 1]) <= config['screening']['max_azimuth_residual_m'])
                    arrays = [residual[:, take][good], h['snr'][region['index']][:, take][good], field['peak'][:, take][good], np.broadcast_to(region['groups'][:, None], good.shape)[good]]
                    for dest, value in zip(bank[label], arrays): dest.extend(value)
            for label, arrays in bank.items():
                try: model = calibrate_noise(np.asarray(arrays[0]).reshape(-1, 2), *[np.array(a) for a in arrays[1:]])
                except ValueError as exc: raise ValueError(f'Track/year {label}: insufficient held-out control support ({exc})') from exc
                by_group[label] = check_covariance(model['covariance_ra_m2'], label)
        matrices = np.array([by_group[label] for label in group])
        # Annual values are summaries only; all NISAR solves use by-pair matrices.
        by_year = {str(y): np.mean(matrices[h['year'][:] == y], axis=0).tolist() for y in years}
    result = {'schema': 'glacier-offsets-track-year-noise-v2', 'mode': 'pair', 'group_by': 'track_year',
              'pair_ids': pair_ids, 'noise_group': group.tolist(), 'covariance_ra_m2_by_pair': matrices.tolist(),
              'covariance_ra_m2_by_year': by_year, 'covariance_ra_m2_by_group': {k: v.tolist() for k, v in by_group.items()},
              'sources': sources, 'bias_applied': False}
    write_json(partial/'noise.json', result)
