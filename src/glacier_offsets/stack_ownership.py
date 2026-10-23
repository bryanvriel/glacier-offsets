"""Reference-only interior ownership of overlapping stack bursts."""
from copy import deepcopy


POLICY = {'method': 'reference-overlap-midpoint-v1', 'edge_margin_lines': 8,
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


def interior_ownership(sources):
    """Partition shared reference rectangles without trimming unshared coverage.

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
    cuts = []
    for first in range(len(bounds)):
        for second in range(first+1, len(bounds)):
            # Inventory groups sources by swath; IW2 starts can precede later
            # IW1 bursts. Orient each overlap by radar time without reordering
            # sources, whose indices also identify SLC and geometry ownership.
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
    for entry, windows in zip(sources, regions):
        dy, dx = entry['offset']
        entry['ownership_windows'] = [[a-dy, b-dy, c-dx, d-dx] for a, b, c, d in windows]
    return sources, dict(POLICY, cuts=cuts)
