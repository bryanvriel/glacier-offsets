"""Interpret overlapping geolocation records in burst2safe annotations in memory."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import logging
import math
from pathlib import Path

from pyproj import Geod


def boundary_grid(tree):
    """Return a normalized copy and affected burst indices, or (None, set()).

    Assembled annotations can retain both sides of a source-burst join at the
    same (line, pixel). Only internal burst boundaries qualify. Prefer the
    record nearest the following burst's annotated start; never average points
    or repair arbitrary polygons. Conflicts exceeding one azimuth time sample
    (10% rounding allowance) or two ground azimuth pixels are rejected.
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
    chosen, affected = {}, set()
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
            fail(f'coordinates disagree at line {line}, pixel {pixel}')
        order = sorted(range(len(points)), key=lambda i: abs((times[i] - target).total_seconds()))
        best = order[0]
        for other in order[1:]:
            if (abs((times[other] - target).total_seconds()) == abs((times[best] - target).total_seconds())
                    and coords[other] != coords[best]):
                fail(f'ambiguous timestamps at line {line}, pixel {pixel}')
        chosen[line, pixel] = points[best]
        affected.update((index - 1, index))

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
    normalized, affected = boundary_grid(tree)
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
