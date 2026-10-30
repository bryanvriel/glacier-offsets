"""Coverage-first ROI proposals and actual-timestamp alias diagnostics."""
import logging
from pathlib import Path
import numpy as np
from .velocity_basis import PERIOD, harmonic
from .velocity_io import write_json

M2_PERIOD = 12.4206012/24
O1_PERIOD = 25.819342/24


def alias_diagnostics(early, late):
    good = np.isfinite(early+late) & (late > early)
    early, late = np.asarray(early)[good], np.asarray(late)[good]
    if len(early) < 4:
        return {'n_pairs': len(early), 'identifiable': False}
    msf = harmonic(early, late)
    m2 = harmonic(early, late, M2_PERIOD)
    q1, _ = np.linalg.qr(msf)
    q2, _ = np.linalg.qr(m2)
    values = np.linalg.svd(q1.T @ q2, compute_uv=False)
    normalized = np.c_[msf/np.linalg.norm(msf, axis=0), m2/np.linalg.norm(m2, axis=0)]
    singular = np.linalg.svd(normalized, compute_uv=False)
    phase = np.mod((early+late)/2, PERIOD)/PERIOD
    occupied = np.unique(np.floor(phase*12).astype(int)).size
    return {'n_pairs': len(early), 'span_days': float(late.max()-early.min()), 'Msf_M2_canonical_correlations': values,
            'Msf_M2_condition': float(singular[0]/singular[-1]), 'phase_bins_of_12': occupied,
            'fractional_utc_day_std': float(np.std(np.mod(np.r_[early, late], 1))),
            'median_pair_days': float(np.median(late-early)), 'viewing_geometries_at_this_point': 1,
            'interpretation': 'One point/source at a time; different bursts are not independent views of this point'}


def phase_coverage(early, late):
    phase = np.mod((early+late)/2, PERIOD)/PERIOD
    return len(np.unique(np.floor(phase*12).astype(int)))/12


def propose(collection, directory, size_m=8000.):
    """Select candidate regions from coverage, flow, geometry and tidal response only."""
    directory = Path(directory)
    n_pairs = len(collection.pairs)
    counts = np.zeros(collection.shape, int)
    speed = np.full(collection.shape, np.nan)
    for i, section in enumerate(collection.sections(128)):
        if i%10==0:
            logging.info('Inspecting coverage tile %s', i+1)
        tile = collection.tile(section)
        counts[section] = tile['valid'].sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            v = tile['data']/(tile['late']-tile['early'])[..., None]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            speed[section] = np.linalg.norm(np.nanmedian(v, axis=0), axis=-1)
    h = collection.handles[0]
    response = h['tidal_response_factor'][:] if 'tidal_response_factor' in h.variables else np.full(collection.shape, np.nan)
    height = h['height_m'][:] if 'height_m' in h.variables else np.full(collection.shape, np.nan)
    geom = h['horizontal_geometry_valid_mask'][:] if 'horizontal_geometry_valid_mask' in h.variables else np.isfinite(speed)
    dx, dy = np.median(abs(np.diff(collection.x))), np.median(abs(np.diff(collection.y)))
    rx, ry = max(1, int(size_m/dx/2)), max(1, int(size_m/dy/2))
    proposals = []
    for role, desired in [('target', 4), ('control', 2)]:
        candidates = []
        for row in range(ry, collection.shape[0]-ry, max(1, ry)):
            for col in range(rx, collection.shape[1]-rx, max(1, rx)):
                section = (slice(row-ry, row+ry), slice(col-rx, col+rx))
                sources = collection.association[section]
                unique, first = np.unique(sources.ravel(), return_index=True)
                first = first[unique >= 0]
                if len(first) < 10:
                    continue
                take = (geom[section].ravel()[first]==1) & (counts[section].ravel()[first] >= min(40, .65*n_pairs))
                coverage = take.mean()
                if coverage < .6:
                    continue
                flow = np.nanmedian(speed[section].ravel()[first][take])
                tide = np.nanmedian(response[section].ravel()[first][take])
                elevation = np.nanmedian(height[section].ravel()[first][take])
                if role=='target' and not (np.isfinite(tide) and tide >= .8 and flow >= .05):
                    continue
                if role=='control' and not (np.isfinite(tide) and tide <= .02 and flow <= .05 and elevation > 10):
                    continue
                score = coverage-(min(flow, 1) if role=='control' else 0)
                candidates.append((score, row, col, flow, tide, elevation, len(first), coverage))
        selected = []
        for item in sorted(candidates, reverse=True):
            _, row, col, flow, tide, elevation, unique, coverage = item
            x, y = collection.x[col], collection.y[row]
            if any(np.hypot(x-p['center_xy'][0], y-p['center_xy'][1]) < 1.5*size_m for p in selected):
                continue
            roi = {'id': f'{role}_{len(selected)+1}', 'role': role, 'center_xy': [float(x), float(y)],
                   'bounds': [float(x-size_m/2), float(y-size_m/2), float(x+size_m/2), float(y+size_m/2)],
                   'reviewed': False, 'review_notes': '', 'unique_sources': unique, 'coverage_fraction': coverage,
                   'median_flow_m_per_day': flow, 'median_tidal_response': tide, 'median_height_m': elevation,
                   'selection': 'coverage, geometry, robust mean flow and tide response; no Msf fits',
                   'caution': 'Candidate only; stable grounded ice is not independently identified bedrock' if role=='control' else 'Floating-ice candidate; inspect imagery/grounding context'}
            selected.append(roi)
            if len(selected)==desired:
                break
        proposals.extend(selected)
    aliases = []
    for burst in np.unique(collection.owner[collection.owner > 0]):
        take = collection.owner==burst
        representative = np.argmax(np.where(take, counts, -1))
        row, col = np.unravel_index(representative, collection.shape)
        point = collection.point(row, col)
        good = point['valid']
        aliases.append({'burst_index': int(burst), 'row': int(row), 'col': int(col),
                        'sampling': alias_diagnostics(point['early'][good], point['late'][good])})
    report = {'sources': collection.sources, 'pair_ids': [p[2] for p in collection.pairs], 'shape': collection.shape,
              'source_count': int(len(np.unique(collection.association[collection.association >= 0]))),
              'regions': proposals, 'aliases': aliases,
              'review_required': True, 'control_truth_established': False,
              'observation_mask':'horizontal valid support, exact available burst endpoints, and available spacecraft correction for swath-corrected inputs',
              'missing_regions': {'targets': 4-sum(p['role']=='target' for p in proposals),
                                  'controls': 2-sum(p['role']=='control' for p in proposals)},
              'next_step': 'Review/edit candidate bounds and identify independent bedrock controls before pilot calibration'}
    write_json(directory/'inspection.json', report)
    write_json(directory/'regions.proposed.json', {'EPSG': int(h.attrs['EPSG']), 'regions': proposals})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    extent = [collection.x.min(), collection.x.max(), collection.y.min(), collection.y.max()]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    for ax, image, label in zip(axes, (counts, speed, response), ('Valid pairs', 'Median flow (m/day)', 'Tidal response factor')):
        view = ax.imshow(image, extent=extent, origin='upper' if collection.y[0] > collection.y[-1] else 'lower',
                         vmin=0, vmax=(n_pairs if label=='Valid pairs' else 2 if 'flow' in label else 1))
        fig.colorbar(view, ax=ax, shrink=.65)
        ax.set_title(label)
        ax.set_xlabel('Projected easting (m)')
        for roi in proposals:
            west, south, east, north = roi['bounds']
            color = 'white' if roi['role']=='target' else 'magenta'
            ax.add_patch(Rectangle((west, south), east-west, north-south, fill=False, ec=color))
            ax.text(west, north, roi['id'], color=color, fontsize=7)
    axes[0].set_ylabel('Projected northing (m)')
    fig.savefig(directory/'regions.proposed.png', dpi=150)
    plt.close(fig)
    collection.check_revisions()
    return report


def sample_points(collection, regions, per_region=9):
    """Choose one unique source per ~3 km block, before seeing harmonic estimates."""
    points = []
    seen = set()
    for region in regions:
        logging.info('Sampling reviewed region %s',region['id'])
        west, south, east, north = region['bounds']
        rows, cols = np.where((collection.x[None, :] >= west) & (collection.x[None, :] <= east) &
                              (collection.y[:, None] >= south) & (collection.y[:, None] <= north) & (collection.association >= 0))
        # Center-first deterministic coverage sampling; no amplitude selection.
        order = np.argsort((collection.x[cols]-(west+east)/2)**2+(collection.y[rows]-(south+north)/2)**2)
        blocks = set()
        positions = []
        attempted = set()
        count = 0
        for i in order:
            row, col = rows[i], cols[i]
            source = int(collection.association[row, col])
            if source in seen or source in attempted:
                continue
            block=collection.spatial_block(row,col)
            if block in blocks:
                continue
            position=collection.observation_position(row,col)
            if any(np.hypot(position[0]-x,position[1]-y)<3000 for x,y in positions):
                continue
            attempted.add(source)
            point = collection.point(row, col)
            if point['spatial_block'] in blocks or point['valid'].sum() < 10:
                continue
            point.update(region=region['id'], role=region['role'])
            blocks.add(point['spatial_block'])
            positions.append(position)
            seen.add(source)
            points.append(point)
            count += 1
            if count >= per_region:
                break
    return points
