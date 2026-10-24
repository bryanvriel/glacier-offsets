"""Reference-only interior ownership of overlapping stack bursts."""
from copy import deepcopy
import re


POLICY = {'method': 'reference-swath-interior-midpoint-v2', 'edge_margin_lines': 8,
          'edge_margin_samples': 8,
          'narrow_overlap': 'error', 'missing_support': 'mask; no source substitution'}


def subtract_window(window, excluded):
    """Return disjoint rectangles after removing an intersecting rectangle."""
    y0, y1, x0, x1 = window
    a, b = max(y0, excluded[0]), min(y1, excluded[1])
    c, d = max(x0, excluded[2]), min(x1, excluded[3])
    if a >= b or c >= d:
        return [window]
    return [w for w in ([y0, a, x0, x1], [b, y1, x0, x1],
                         [a, b, x0, c], [a, b, d, x1]) if w[0] < w[1] and w[2] < w[3]]


def _swath(entry):
    # Native Sentinel-1 IDs end in _iw1/_iw2/_iw3. Unlabelled synthetic or
    # legacy sources form one along-track group, retaining their time cuts.
    match = re.search(r'(?:^|_)(iw[123])$', entry['burst_id'].lower())
    return match.group(1) if match else 'unlabelled'


def _union(intervals):
    result = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(stop, result[-1][1])
        else:
            result.append([start, stop])
    return result


def _range_ownership(bounds, regions, swaths, *, edge_margin_samples=POLICY['edge_margin_samples']):
    """Cut swath support unions, after along-track ownership is resolved.

    A sweep over rectangle edges avoids pairwise range/time cuts leaving holes
    at staggered four-burst intersections. No full-resolution arrays are needed.
    """
    edges = sorted({y for windows in regions for w in windows for y in w[:2]})
    selected = [[] for _ in regions]
    previous = {}
    cuts = []
    previous_cuts = {}
    for y0, y1 in zip(edges, edges[1:]):
        active = [(i, w[2], w[3]) for i, windows in enumerate(regions)
                  for w in windows if w[0] <= y0 and y1 <= w[1]]
        support = {swath: _union((a, b) for i, a, b in active if swaths[i] == swath)
                   for swath in sorted(set(swaths))}
        components = sorted((a, b, swath) for swath, intervals in support.items() for a, b in intervals)
        excluded = {swath: [] for swath in support}
        for n, (a, b, near) in enumerate(components):
            for m, (c, d, far) in enumerate(components[n+1:], n+1):
                if near == far or c >= b:
                    continue
                if c <= a or d <= b:
                    raise ValueError(f'{near} / {far}: Interior ownership requires '
                                     'distinct range starts and non-nested swath support')
                # An intervening swath owns any triple overlap. Only adjacent
                # support components need a switch or an edge-margin check.
                if m != n+1:
                    continue
                if b-c < 2*edge_margin_samples:
                    raise ValueError(f'{near} / {far}: overlap has {b-c} samples; interior ownership '
                                     f"requires at least {2*edge_margin_samples} for the edge margin")
                cut = (c+b)//2
                excluded[near].append([cut, b])
                excluded[far].append([c, cut])
                key = (near, far, c, b, cut)
                old = previous_cuts.get(key)
                if old is not None and old['overlap_global'][1] == y0:
                    old['overlap_global'][1] = y1
                else:
                    old = {'near_swath': near, 'far_swath': far,
                           'overlap_global': [y0, y1, c, b], 'cut_global_sample': cut}
                    cuts.append(old)
                    previous_cuts[key] = old
        strip = []
        for i, a, b in active:
            windows = [[y0, y1, a, b]]
            for c, d in excluded[swaths[i]]:
                windows = [part for w in windows for part in subtract_window(w, [y0, y1, c, d])]
            strip.extend((i, w[2], w[3]) for w in windows)
        # Check the partition on compressed intervals, including unshared edges.
        ordered = sorted((a, b) for _, a, b in strip)
        union = _union((x0, x1) for a, b, x0, x1 in bounds if a <= y0 and y1 <= b)
        if any(b > c for (_, b), (c, _) in zip(ordered, ordered[1:])) or _union(ordered) != union:
            raise ValueError('Interior ownership cannot partition the reference coverage without holes or overlaps')
        for i, a, b in strip:
            key = (i, a, b)
            old = previous.get(key)
            if old is not None and old[1] == y0:
                old[1] = y1
            else:
                old = [y0, y1, a, b]
                selected[i].append(old)
                previous[key] = old
    return selected, cuts


def interior_ownership(sources):
    """Partition shared reference rectangles without trimming unshared coverage.

    Resolve along-track cuts within each swath, then range cuts between swaths.
    Windows are half-open. Holes inside an assigned region remain unsupported;
    another acquisition or burst never decides the ownership of that region.
    """
    sources = deepcopy(sources)
    bounds = []
    for entry in sources:
        dy, dx = entry['offset']
        y0, y1, x0, x1 = entry['valid_window']
        bounds.append([y0+dy, y1+dy, x0+dx, x1+dx])
    regions = [[w] for w in bounds]
    swaths = [_swath(entry) for entry in sources]
    cuts = []
    for first in range(len(bounds)):
        for second in range(first+1, len(bounds)):
            if swaths[first] != swaths[second]:
                continue
            # Orient by radar time without reordering the sources: their indices
            # also identify SLC and geometry ownership.
            i, j = sorted((first, second), key=lambda index: sources[index]['offset'][0])
            upper, lower = bounds[i], bounds[j]
            y0, y1 = max(upper[0], lower[0]), min(upper[1], lower[1])
            x0, x1 = max(upper[2], lower[2]), min(upper[3], lower[3])
            if y0 >= y1 or x0 >= x1:
                continue
            if sources[j]['offset'][0] <= sources[i]['offset'][0] or lower[1] <= upper[1]:
                raise ValueError(f"{sources[i]['burst_id']} / {sources[j]['burst_id']}: "
                                 'Interior ownership requires distinct radar start times and '
                                 'non-nested reference bursts')
            if y1-y0 < 2*POLICY['edge_margin_lines']:
                raise ValueError(f"{sources[i]['burst_id']} / {sources[j]['burst_id']}: "
                                 f'overlap has {y1-y0} lines; interior ownership requires at least '
                                 f"{2*POLICY['edge_margin_lines']} for the edge margin")
            cut = (y0+y1)//2
            regions[i] = [part for w in regions[i] for part in subtract_window(w, [cut, y1, x0, x1])]
            regions[j] = [part for w in regions[j] for part in subtract_window(w, [y0, cut, x0, x1])]
            cuts.append({'upper_burst_id': sources[i]['burst_id'], 'lower_burst_id': sources[j]['burst_id'],
                         'overlap_global': [y0, y1, x0, x1], 'cut_global_line': cut})
    if len(set(swaths)) > 1:
        regions, range_cuts = _range_ownership(bounds, regions, swaths)
        cuts.extend(range_cuts)
    for entry, windows in zip(sources, regions):
        dy, dx = entry['offset']
        entry['ownership_windows'] = [[a-dy, b-dy, c-dx, d-dx] for a, b, c, d in windows]
    return sources, dict(POLICY, cuts=cuts)
