"""Evans comparisons using common pixels, exact endpoints, and held-out acquisitions."""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from pyproj import Transformer

from .velocity_collection import days
from .velocity_io import digest, write_json, writer_lock
from .velocity_multitrack import common_grid, joint_field, screen_tracks, solver_options
from .velocity_multitrack_workflow import inputs
from .velocity_pixel_general import design, predict, solve
from .velocity_prepare import pair_validity
from .velocity_workflow import run_workflow
from .velocity_workflow_config import MULTITRACK, load_workflow_config


def acquisition_folds(rows, width=60):
    """Hold out the same calendar blocks on all tracks, purging exact shared acquisitions."""
    early = np.array([p['calendar_early_day'] for p in rows])
    late = np.array([p['calendar_late_day'] for p in rows])
    block = np.floor(((early+late)/2-early.min())/width).astype(int)
    acquisitions = np.array([[p['early_acquisition'], p['late_acquisition']] for p in rows])
    for label in np.unique(block):
        test = block == label
        shared = np.unique(acquisitions[test])
        yield ~test & ~np.isin(acquisitions, shared).any(1), test


def replay_sources(track, native, package, validity, remove_mixed):
    """Check cache rows and the once-corrected radar observations against the package."""
    native = np.unique(native)
    field = track.field(native, remove_mixed)
    r, c = track.obs['row'][native], track.obs['col'][native]
    owner = track.owner[native]
    with h5py.File(package, 'r') as source:
        np.testing.assert_array_equal(source['source_observation_index'][:][r, c], track.obs['source_id'][native])
        np.testing.assert_array_equal(source['source_burst_index'][:][r, c]-1, owner)
        enu_order = source['enu'][:].astype(str).tolist()
        indices = [enu_order.index(k) for k in ('east','north','up')]
        g = np.stack([source[key][:][indices][:, r, c].T for key in ('range_sensitivity_enu', 'azimuth_sensitivity_enu')], axis=1)
        np.testing.assert_array_equal(g, track.geometry[native])
        for j, p in enumerate(track.rows):
            slot = p['slot']
            e, l = [days(source[k][slot])[owner] for k in ('earlier_burst_time', 'later_burst_time')]
            ra = np.stack([source[k][slot][r,c] for k in ('range_displacement_m','azimuth_displacement_m')], axis=-1).astype(float)
            good, _ = pair_validity(source, slot, r, c, ra, e, l, validity)
            if track.body is not None:
                ra -= track.body['set_ra_m'][native, j]
            good &= np.isfinite(ra).all(1)
            if remove_mixed:
                good &= ~track.mixed[native]
            np.testing.assert_array_equal(field['early'][:, j], e)
            np.testing.assert_array_equal(field['late'][:, j], l)
            np.testing.assert_array_equal(field['data'][:, j], ra)
            np.testing.assert_array_equal(field['valid'][:, j], good)
    return len(native)


def _fit(field, layout, mask, axis, cov, options):
    return solve(field, layout, mask, axis, {'by_pair': cov}, annual_uncertainty=True, **options)


def _fit_record(fit):
    return {k: fit[k][0] for k in ('estimable', 'rank', 'amplitude', 'phase_deg', 'amplitude_sigma',
                                    'rms_m', 'target_min_information', 'target_condition')}


def held_out(fields, tracks, layout, config):
    """Compare the two models on identical held-out RA pairs on each track."""
    splits = [len(t.rows) for t in tracks]
    indices = np.split(np.arange(sum(splits)), np.cumsum(splits)[:-1])
    covariance = np.concatenate([t.covariance for t in tracks])
    joined = joint_field(fields)
    screen_options = config['screening']
    options = solver_options(config)
    losses = [[] for _ in tracks]
    for train, test in acquisition_folds([p for t in tracks for p in t.rows]):
        training_fields = [{**f, 'valid': f['valid'] & train[ix]} for f, ix in zip(fields, indices)]
        masks, axis, _, _ = screen_tracks(training_fields, tracks, screen_options)
        joint = _fit(joined, layout, np.concatenate(masks, axis=1), axis, covariance, options)
        if not joint['estimable'][0]:
            continue
        joint_pred = predict(joined, layout, joint['coefficients'], joint['names'], axis)[0]
        for i, (f, t, train_mask, ix) in enumerate(zip(fields, tracks, masks, indices)):
            accepted = test[ix] & f['valid'][0] & (f['peak'][0] >= screen_options['min_peak'])
            if not accepted.any():
                continue
            baseline = _fit(f, t.layout, train_mask, axis, t.covariance, options)
            if not baseline['estimable'][0]:
                continue
            native_pred = predict(f, t.layout, baseline['coefficients'], baseline['names'], axis)[0]
            reference = f['data'][0, accepted]
            r = np.stack([reference-native_pred[accepted], reference-joint_pred[ix][accepted]], axis=1)
            white = np.linalg.solve(np.linalg.cholesky(t.covariance[accepted]), r.transpose(0,2,1)).transpose(0,2,1)
            losses[i].append((r, white))
    result = {}
    for t, saved in zip(tracks, losses):
        if not saved:
            result[str(t.number)] = {'test_pairs': 0, 'folds': 0}
            continue
        r, white = [np.concatenate([v[i] for v in saved]) for i in range(2)]
        result[str(t.number)] = {'test_pairs': len(r), 'folds': len(saved),
                                'native_rms_ra_m': np.sqrt(np.mean(r[:, 0]**2, axis=0)),
                                'joint_rms_ra_m': np.sqrt(np.mean(r[:, 1]**2, axis=0)),
                                'native_whitened_rms': np.sqrt(np.mean(white[:, 0]**2)),
                                'joint_whitened_rms': np.sqrt(np.mean(white[:, 1]**2))}
    return result


def injections(field, layout, mask, axis, covariance, options):
    """Recover known Msf/vertical terms and measure omitted-M2 leakage at exact times."""
    names = layout.joint_names
    coefficients = np.zeros((1,len(names)))
    coefficients[0, names.index('vertical_M2_C')] = .20
    coefficients[0, names.index('vertical_M2_S')] = -.10
    tide = predict(field, layout, coefficients, names, axis)
    quiet = _fit({**field, 'data': tide}, layout, mask, axis, covariance, options)
    coefficients[0,:2*len(layout.years)] = np.tile(axis[0]*1.3, len(layout.years))
    coefficients[0,-2:] = [.08, -.06]
    mixed = predict(field, layout, coefficients, names, axis)
    recovered = _fit({**field, 'data': mixed}, layout, mask, axis, covariance, options)
    if not (recovered['estimable'][0] and quiet['estimable'][0]):
        return {'estimable': False}
    a, y, _, _ = design({**field,'data': tide}, layout, axis, {'by_pair': covariance})
    omit = np.array([not k.startswith('vertical_') for k in names])
    no_vertical = np.linalg.lstsq(a[0,mask[0]][:,:,omit].reshape(-1,int(omit.sum())),
                                 y[0,mask[0]].ravel(), rcond=1e-10)[0]
    return {'estimable': True, 'msf_injected_cs_m_day': coefficients[0,-2:],
            'msf_recovered_cs_m_day': recovered['cs'][0],
            'max_coefficient_error': float(np.max(abs(recovered['coefficients']-coefficients))),
            'vertical_m2_only_msf_m_day': quiet['amplitude'][0],
            'vertical_m2_omitted_msf_m_day': np.linalg.norm(no_vertical[-2:])}


def control_tuples(grid, regions, limit=12):
    """Choose separated, independent-chip representatives of reviewed grounded boxes."""
    from .velocity_annual_noise import control_regions
    epsg, boxes = control_regions(regions)
    x, y = Transformer.from_crs(3031, epsg or 3031, always_xy=True).transform(grid.x[grid.col], grid.y[grid.row])
    blocks = np.floor(np.stack([grid.x[grid.col], grid.y[grid.row]], axis=1)/3000).astype(int)
    found, used, seen_blocks = [], [set() for _ in range(grid.native_id.shape[1])], set()
    for box in boxes:
        west, south, east, north = box['bounds']
        candidates = np.flatnonzero((x>=west) & (x<east) & (y>=south) & (y<north))
        for i in candidates[np.argsort(grid.chip_separation_m[candidates], kind='stable')]:
            block = tuple(blocks[i])
            if block in seen_blocks or any(s in bag for s, bag in zip(grid.native_id[i], used)):
                continue
            found.append((box['id'], i))
            seen_blocks.add(block)
            for s, bag in zip(grid.native_id[i], used):
                bag.add(s)
            break
    if not found:
        raise ValueError('No reviewed control boxes share multitrack support')
    return [found[i] for i in np.linspace(0, len(found)-1, min(limit, len(found)), dtype=int)]


def run_pilot(config_path, points_path):
    config = load_workflow_config(config_path)
    if config.get('collection', {}).get('mode') != MULTITRACK:
        raise ValueError('Pilot requires collection.mode: sentinel1_multitrack')
    points = []
    with Path(points_path).open(newline='') as f:
        for row in csv.DictReader(f):
            points.append((row['name'], float(row['longitude']), float(row['latitude'])))
    if not points or len({name for name, _, _ in points}) != len(points) or not np.isfinite([(x,y) for _,x,y in points]).all():
        raise ValueError('Point CSV requires unique names and finite longitude/latitude')
    manifest = run_workflow(config_path, stages='prepare')
    config = manifest['config']
    options = solver_options(config)
    out = Path(config['output'])
    project = Transformer.from_crs(4326,3031,always_xy=True)
    with writer_lock(out), inputs(pilot_collection(out, manifest),
                                  manifest['native_artifacts'], verify=True) as (tracks, layout):
        grid = common_grid([t.obs for t in tracks])
        selections, status = [], []
        for label, lon, lat in points:
            x,y = project.transform(lon,lat)
            available = []
            for t in tracks:
                xx, yy = t.obs['x'][:], t.obs['y'][:]
                c,r = np.argmin(abs(xx-x)),np.argmin(abs(yy-y))
                if (abs(xx[c]-x) <= abs(xx[1]-xx[0])/2 and abs(yy[r]-y) <= abs(yy[1]-yy[0])/2
                        and t.obs['map_source_index'][r,c] >= 0):
                    available.append(t.number)
            r,c = np.argmin(abs(grid.y-y)), np.argmin(abs(grid.x-x))
            source = grid.back[r,c] if len(available) == len(tracks) else -1
            if source >= 0:
                selections.append((label,'GPS',source))
            else:
                status.append({'name': label, 'role': 'GPS', 'available_tracks': available, 'status': 'outside_common_support'})
        selections.extend((label,'grounded_control',i) for label,i in control_tuples(grid, config['noise']['regions']))
        replay = {}
        for i, (track, entry) in enumerate(zip(tracks, manifest['native_artifacts'])):
            replay[str(track.number)] = replay_sources(track, grid.native_index[[i for _,_,i in selections],i],
                                                       entry['input']['path'], entry['input']['validity_mask'],
                                                       config['screening']['remove_mixed_windows'])
        for label, role, i in selections:
            native = grid.native_index[i:i+1]
            fields = [t.field(native[:, j], config['screening']['remove_mixed_windows']) for j,t in enumerate(tracks)]
            masks, axis, _, _ = screen_tracks(fields, tracks, config['screening'])
            joined = joint_field(fields)
            cov = np.concatenate([t.covariance for t in tracks])
            mask = np.concatenate(masks, axis=1)
            fit = _fit(joined, layout, mask, axis, cov, options)
            baseline = [_fit(f,t.layout,g,axis,t.covariance,options) for f,t,g in zip(fields,tracks,masks)]
            accepted = {str(t.number): {str(y): int(g[0,t.year_index==j].sum()) for j,y in enumerate(layout.years)}
                        for t,g in zip(tracks,masks)}
            status.append({'name': label, 'role': role, 'status': 'fitted' if fit['estimable'][0] else 'rank_deficient',
                           'cell_xy_m': [grid.x[grid.col[i]],grid.y[grid.row[i]]],
                           'native_source_id': grid.native_id[i], 'native_chip_separation_m': grid.chip_separation_m[i],
                           'flow_axis_EN': axis[0], 'accepted_by_track_year': accepted, 'joint': _fit_record(fit),
                           'single_track': {str(t.number): _fit_record(v) for t,v in zip(tracks,baseline)},
                           'held_out': held_out(fields,tracks,layout,config),
                           'injection': {'joint': injections(joined,layout,mask,axis,cov,options),
                                         **{str(t.number): injections(f,t.layout,g,axis,t.covariance,options)
                                            for f,t,g in zip(fields,tracks,masks)}}})
        tested = [p['injection']['joint'] for p in status if p['role']=='GPS' and p['status']=='fitted'
                  and p['injection']['joint']['estimable']]
        passed = bool(tested) and all(r['max_coefficient_error']<1e-7 and r['vertical_m2_only_msf_m_day']<1e-7 for r in tested)
        control_sigma = [p['joint']['amplitude_sigma'] for p in status if p['role']=='grounded_control'
                         and p['joint']['estimable']]
        control_summary = {'distinct_3km_block_representatives': len(control_sigma),
                           'median_conditional_msf_amplitude_sigma_m_day': np.median(control_sigma) if control_sigma else None}
        record = {'source_replay_native_counts': replay, 'points_csv_sha256': digest(points_path),
                  'collection_sha256': manifest['collection_sha256'], 'fit_injection_passed': passed,
                  'controls_uncertainty': control_summary, 'points': status}
        write_json(out/'evans-pilot.json', record)
        write_report(out, status, replay, passed, control_summary)
        if not passed:
            raise ValueError(f'Pilot synthetic recovery failed; see {out / "evans-pilot.json"}')
    return out/'evans-pilot.md'


def pilot_collection(out, manifest):
    path = out/'collection-manifest.json'
    if digest(path) != manifest['collection_sha256']:
        raise ValueError('Multitrack collection changed during pilot preparation')
    return json.loads(path.read_text())


def write_report(out, points, replay, passed, controls):
    lines = ['# Evans multitrack pilot', '',
             f'Native source replay matched both original packages ({replay}).',
             f'Known-signal injection passed: {str(passed).lower()}.', '',
             'Sixty-day holdouts purge shared acquisitions. Each fold uses the same joint training-only flow axis '
             'for both comparisons. Scores use the same peak-screened held-out pairs on each track.',
             '', '| Site | Track | Pairs | Track-only whitened RMS | Joint whitened RMS |',
             '| --- | --- | --- | --- | --- |']
    for p in points:
        if p['role'] != 'GPS':
            continue
        if 'held_out' not in p:
            for track in p['available_tracks']:
                lines.append(f"| {p['name']} | {int(track):03d} only | — | — | — |")
            continue
        for track,result in p['held_out'].items():
            n = result['test_pairs']
            values = f"{result['native_whitened_rms']:.3f} | {result['joint_whitened_rms']:.3f}" if n else '— | —'
            lines.append(f"| {p['name']} | {int(track):03d} | {n} | {values} |")
    lines += ['', '| Site | Joint Msf amplitude (m/day) | Joint phase (°) | Joint / path 65 minimum Msf information |',
              '| --- | --- | --- | --- |']
    for p in points:
        if p['role'] == 'GPS' and p['status'] == 'fitted':
            joint = p['joint']
            base = p['single_track'].get('65', {}).get('target_min_information', 0.)
            ratio = f"{joint['target_min_information']/base:.2f}" if base > 1e-8 else '—'
            lines.append(f"| {p['name']} | {joint['amplitude']:.3f} | {joint['phase_deg']:.1f} | {ratio} |")
    if controls['distinct_3km_block_representatives']:
        lines += ['', f"The median conditional Msf amplitude SE across {controls['distinct_3km_block_representatives']} "
                      f"grounded 3 km block representatives was {controls['median_conditional_msf_amplitude_sigma_m_day']:.4f} m/day. "
                      'Each representative uses distinct native chips.']
    lines += ['', 'The JSON also contains the grounded-control fits, per-track rank and Msf information, '
              'and M2-only injections with and without a residual vertical model.', '']
    (out/'evans-pilot.md').write_text('\n'.join(lines))
