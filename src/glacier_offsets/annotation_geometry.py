"""Interpret overlapping geolocation records in burst2safe annotations in memory."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import logging
import math
from pathlib import Path

from pyproj import Geod


def _row_runs(grid, line):
    """Keep source row order; a repeated/decreasing pixel begins another row."""
    runs = []
    for point in grid:
        if int(point.findtext('line')) != line:
            continue
        if not runs or int(point.findtext('pixel')) <= int(runs[-1][-1].findtext('pixel')):
            runs.append([])
        runs[-1].append(point)
    return runs


def _orbit_row(row, orbit, spacing, bounds):
    """Check zero Doppler and infer the row's nominal reference height.

    Annotation coordinates may use one nominal height per source slice rather
    than each point's height field. Solve the range equation along the WGS84
    normal and require one common height across the row. No coordinates change.
    """
    import numpy as np
    import isce3

    ellipsoid = isce3.core.Ellipsoid()
    heights, along = [], []
    for point in row:
        lon, lat = np.deg2rad([float(point.findtext('longitude')), float(point.findtext('latitude'))])
        if not np.isfinite([lon, lat]).all() or abs(lon) > math.pi or abs(lat) > math.pi/2:
            raise ValueError('invalid coordinates in source row')
        time = (isce3.core.DateTime(point.findtext('azimuthTime')) - orbit.reference_epoch).total_seconds()
        position, velocity = orbit.interpolate(time)
        xyz0 = ellipsoid.lon_lat_to_xyz([lon, lat, 0.])
        normal = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
        delta = xyz0 - position
        distance = float(point.findtext('slantRangeTime')) * isce3.core.speed_of_light/2
        dot = np.dot(delta, normal)
        discriminant = dot*dot - (np.dot(delta, delta) - distance*distance)
        if not np.isfinite(distance) or distance <= 0 or discriminant <= 0:
            raise ValueError('invalid range intersection in source row')
        # Stable small root of h**2 + 2*dot*h + |delta|**2-range**2 = 0.
        height = (np.dot(delta, delta) - distance*distance) / (-dot + np.sqrt(discriminant))
        target = delta + height*normal
        speed = np.linalg.norm(velocity)
        if not np.isfinite(height) or not np.isfinite(speed) or speed <= 0:
            raise ValueError('invalid orbit intersection in source row')
        heights.append(float(height))
        along.append(float(abs(np.dot(target, velocity)/speed)))
    if (not heights or max(heights)-min(heights) > 1.0 or max(along) > 2*spacing
            or min(heights) < bounds[0] or max(heights) > bounds[1]):
        raise ValueError('source row is inconsistent with a common radar reference surface')
    return {'height_m': float(np.median(heights)), 'height_span_m': max(heights)-min(heights),
            'max_zero_doppler_distance_m': max(along)}


def source_boundary_row(tree, line, orbit):
    """Select a complete row after proving a source-reference-surface join.

    Both adjacent unambiguous rows must have a coherent radar reference height.
    Every competing fragment must match one of those surfaces. Prefer a complete
    following-source row; otherwise retain the complete preceding-source row.
    This only normalizes approximate discovery footprints, never SLC geometry.
    """
    grid = tree.find('geolocationGrid/geolocationGridPointList')
    lines = int(tree.findtext('swathTiming/linesPerBurst'))
    samples = int(tree.findtext('swathTiming/samplesPerBurst'))
    spacing = float(tree.findtext('imageAnnotation/imageInformation/azimuthPixelSpacing'))
    reported = [float(p.findtext('height')) for p in grid]
    if not all(math.isfinite(h) for h in reported):
        raise ValueError('invalid annotated heights')
    bounds = min(0., min(reported))-2*spacing, max(0., max(reported))+2*spacing
    previous, following = _row_runs(grid, line-lines), _row_runs(grid, line+lines)
    if len(previous) != 1 or len(following) != 1:
        raise ValueError('source join needs unambiguous adjacent rows')
    pixels = [int(p.findtext('pixel')) for p in previous[0]]
    if (len(pixels) < 3 or pixels != sorted(set(pixels)) or pixels[0] != 0 or pixels[-1] != samples-1
            or [int(p.findtext('pixel')) for p in following[0]] != pixels):
        raise ValueError('source join needs matching complete adjacent rows')
    before = _orbit_row(previous[0], orbit, spacing, bounds)
    after = _orbit_row(following[0], orbit, spacing, bounds)
    if abs(before['height_m']-after['height_m']) <= 2.:
        raise ValueError('coordinate conflict is not a change of source reference surface')
    candidates = []
    for row in _row_runs(grid, line):
        actual = [int(p.findtext('pixel')) for p in row]
        if any(pixel not in pixels for pixel in actual):
            raise ValueError('source row has unexpected range samples')
        checked = _orbit_row(row, orbit, spacing, bounds)
        distances = [abs(checked['height_m']-r['height_m']) for r in (before, after)]
        if min(distances) > 1.:
            raise ValueError('source fragment does not match either neighbouring reference surface')
        if actual == pixels:
            candidates.append((distances[1] > 1., row, checked))
    if not candidates:
        raise ValueError('no complete source boundary row remains')
    best = min(candidate[0] for candidate in candidates)
    selected = [candidate for candidate in candidates if candidate[0] == best]
    if len(selected) != 1:
        raise ValueError('ambiguous complete source boundary rows')
    _, row, checked = selected[0]
    return row, {'line': line, 'previous': before, 'following': after, 'selected': checked,
                 'source': 'previous' if best else 'following', 'point_count': len(row)}


def boundary_grid(tree, orbit=None):
    """Return a normalized copy and affected burst indices, or (None, set()).

    Assembled annotations can retain both sides of a source-burst join at the
    same (line, pixel). Only internal burst boundaries qualify. Prefer the
    record nearest the following burst's annotated start; never average points
    or repair arbitrary polygons. Conflicts exceeding one azimuth time sample
    (10% rounding allowance) are rejected. Larger coordinate differences require
    independent orbit validation of complete source rows on different nominal
    reference surfaces; without an orbit the two-azimuth-pixel limit remains.
    """
    grid = tree.find('geolocationGrid/geolocationGridPointList')
    if grid is None:
        return None, set()
    groups = defaultdict(list)
    for point in grid:
        groups[int(point.findtext('line')), int(point.findtext('pixel'))].append(point)
    duplicates = {key: points for key, points in groups.items() if len(points) > 1}
    if not duplicates:
        return None, set()

    def fail(reason):
        raise ValueError(f'Conflicting annotation geolocation boundary records: {reason}')

    if tree.findtext('imageAnnotation/imageInformation/productComposition') != 'Assembled':
        fail('duplicate line/pixel outside an assembled product')
    bursts = tree.findall('swathTiming/burstList/burst')
    lines = int(tree.findtext('swathTiming/linesPerBurst'))
    samples = int(tree.findtext('swathTiming/samplesPerBurst'))
    interval = float(tree.findtext('imageAnnotation/imageInformation/azimuthTimeInterval'))
    spacing = float(tree.findtext('imageAnnotation/imageInformation/azimuthPixelSpacing'))
    if lines <= 0 or samples <= 1 or any(not math.isfinite(v) or v <= 0 for v in (interval, spacing)):
        fail('invalid sampling')
    expected_lines = set(range(0, (len(bursts) + 1) * lines, lines))
    if {key[0] for key in groups} != expected_lines or int(grid.get('count')) != len(grid):
        fail('unexpected grid rows or point count')
    geod = Geod(ellps='WGS84')
    chosen, affected, source_rows = {}, set(), {}
    for (line, pixel), points in duplicates.items():
        if line <= 0 or line >= len(bursts) * lines or line % lines or not 0 <= pixel < samples:
            fail(f'duplicate outside an internal burst boundary: line {line}, pixel {pixel}')
        index = line // lines
        target = datetime.fromisoformat(bursts[index].findtext('azimuthTime'))
        times = [datetime.fromisoformat(p.findtext('azimuthTime')) for p in points]
        coords = [(float(p.findtext('longitude')), float(p.findtext('latitude'))) for p in points]
        if any(not math.isfinite(x) or not math.isfinite(y) or
               not -180 <= x <= 180 or not -90 <= y <= 90 for x, y in coords):
            fail('invalid coordinates')
        if ((max(times) - min(times)).total_seconds() > 1.1 * interval or
                any(abs((t - target).total_seconds()) > 1.1 * interval for t in times)):
            fail(f'timestamps disagree at line {line}, pixel {pixel}')
        if any(geod.inv(*a, *b)[2] > 2 * spacing for a in coords for b in coords):
            if orbit is None:
                fail(f'coordinates disagree at line {line}, pixel {pixel}')
            if line not in source_rows:
                try:
                    row, report = source_boundary_row(tree, line, orbit)
                except (ValueError, RuntimeError) as exc:
                    fail(f'coordinates disagree at line {line}, pixel {pixel}: {exc}')
                source_rows[line] = row
                logging.info('Source-consistent geolocation row %d: %s source; nominal heights %.3f/%.3f m; '
                             'height spread %.4f m; zero-Doppler residual %.4f m', line, report['source'],
                             report['previous']['height_m'], report['following']['height_m'],
                             report['selected']['height_span_m'], report['selected']['max_zero_doppler_distance_m'])
        order = sorted(range(len(points)), key=lambda i: abs((times[i] - target).total_seconds()))
        best = order[0]
        for other in order[1:]:
            if (line not in source_rows and
                    abs((times[other] - target).total_seconds()) == abs((times[best] - target).total_seconds())
                    and coords[other] != coords[best]):
                fail(f'ambiguous timestamps at line {line}, pixel {pixel}')
        chosen[line, pixel] = points[best]
        affected.update((index - 1, index))

    # Replace the entire affected row, including nonduplicated points. Never
    # mix points from different source reference surfaces or synthesize points.
    for line, row in source_rows.items():
        chosen.update({(line, int(point.findtext('pixel'))): point for point in row})
    normalized = deepcopy(tree)
    output = normalized.find('geolocationGrid/geolocationGridPointList')
    output[:] = [deepcopy(chosen.get(key, points[0])) for key, points in sorted(groups.items())]
    output.set('count', str(len(output)))
    return normalized, affected


def normalize_burst_geometry(safe, swath, polarization, items, tree=None):
    """Replace only affected native centers/borders; preserve all radar metadata."""
    from .input_preparation import annotation_paths, safe_xml
    if tree is None:
        paths = [p for p in annotation_paths(Path(safe))
                 if f'iw{swath}-slc-{polarization.lower()}' in p.lower()]
        if len(paths) != 1:
            raise ValueError(f'Expected one IW{swath} {polarization} annotation in {safe}')
        tree = safe_xml(Path(safe), paths[0])
    orbit = getattr(items[0], 'orbit', None) if items else None
    normalized, affected = boundary_grid(tree, orbit=orbit)
    if normalized is None:
        return items
    from s1reader.s1_reader import get_burst_centers_and_boundaries
    from .aoi import polygon, wgs84_parts
    count = len(tree.findall('swathTiming/burstList/burst'))
    centers, borders = get_burst_centers_and_boundaries(normalized, num_bursts=count)
    if len(borders) != count or len(centers) != count:
        raise ValueError('Normalized geolocation grid does not match the annotation burst count')
    for index in affected:
        # Retain strict validation, including antimeridian wrapping. No buffer(0),
        # convex hull or make_valid can conceal an unrelated malformed outline.
        if not borders[index] or centers[index].is_empty:
            raise ValueError('Normalized annotation has no footprint/center')
        for part in borders[index]:
            wgs84_parts(polygon(part, 'Normalized annotation footprint'))
    result = [replace(b, center=centers[b.i_burst], border=borders[b.i_burst])
              if b.i_burst in affected else b for b in items]
    logging.info('Normalized overlapping IW%d geolocation boundary records in %s', swath, Path(safe).name)
    return result
