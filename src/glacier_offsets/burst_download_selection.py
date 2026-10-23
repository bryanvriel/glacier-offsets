"""Deterministic contiguous burst subsets for explicitly partial acquisitions."""
from collections import Counter

PARTIAL_POLICY = 'iw1-contiguous-v1'


def select_contiguous_subsets(available, requested):
    """Keep every observed swath, optimizing IW1 AOI count then total coverage.

    Each candidate is a contiguous interval containing an AOI-requested burst.
    Adjacent included swaths must have start/end IDs within one, matching
    burst2safe's validity rule. Dynamic programming checks nine predecessor
    bounds per interval instead of a Cartesian product of all swath intervals.
    Ties favor the lowest start/end bounds in IW1/IW2/IW3 order.
    """
    requested = set(requested)
    states = {}
    swaths = sorted(available)
    for swath in swaths:
        rows = sorted(available[swath], key=lambda row: row['burst_id'])
        ids = [row['burst_id'] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f'{swath}: ambiguous duplicate burst products')
        prefix = [0]
        for row in rows:
            prefix.append(prefix[-1] + (row['granule'] in requested))
        following = {}
        for i, lo in enumerate(ids):
            for j in range(i, len(ids)):
                if j > i and ids[j] != ids[j-1] + 1:
                    break
                aoi_count = prefix[j+1] - prefix[i]
                if not aoi_count:
                    continue
                hi = ids[j]
                score = (aoi_count if swath == 'IW1' else 0, aoi_count, j-i+1)
                if swath == swaths[0]:
                    following[lo, hi] = (score, ((lo, hi),))
                    continue
                predecessors = [states[a, b] for a in range(lo-1, lo+2)
                                for b in range(hi-1, hi+2) if (a, b) in states]
                if predecessors:
                    best = min(predecessors, key=_rank)
                    following[lo, hi] = (tuple(a+b for a, b in zip(best[0], score)),
                                         best[1] + ((lo, hi),))
        if not following:
            raise ValueError('No compatible contiguous subsets retain AOI coverage in every observed swath')
        states = following
    if not states:
        raise ValueError('No available bursts for partial acquisition')
    best = min(states.values(), key=_rank)
    ranges = dict(zip(swaths, best[1]))
    selected = {row['granule'] for swath, rows in available.items() for row in rows
                if ranges[swath][0] <= row['burst_id'] <= ranges[swath][1]}
    return selected, {swath: list(bounds) for swath, bounds in ranges.items()}


def _rank(state):
    return tuple(-value for value in state[0]) + (state[1],)


def coverage_report(available, selected, requested, missing, ranges):
    """Retain identifiers and footprints for available-but-discarded products."""
    rows = [row for swath in sorted(available) for row in available[swath]]
    discarded = [row | {'aoi_requested': row['granule'] in requested,
                        'reason': 'contiguity_or_cross_swath_compatibility'}
                 for row in sorted(rows, key=lambda row: row['granule']) if row['granule'] not in selected]
    return {'policy': PARTIAL_POLICY, 'partial': bool(missing or discarded),
            'available_burst_count': len(rows), 'retained_burst_count': len(selected),
            'missing_bursts': missing, 'discarded_bursts': discarded, 'retained_ranges': ranges}


def validate_coverage(group, request):
    """Accepted omissions require explicit policy provenance, not failure flags."""
    coverage = group.get('coverage')
    if coverage is None and not request.get('allow_partial', False):
        return  # Legacy and strict collections keep their existing contract.
    try:
        if not request.get('allow_partial', False) or coverage['policy'] != PARTIAL_POLICY:
            raise ValueError('Partial coverage requires the explicit --allow-partial policy')
        retained = {row['granule']: row for row in group['bursts']}
        discarded = {row['granule']: row for row in coverage['discarded_bursts']}
        missing = {(row['swath'], row['burst_id']) for row in coverage['missing_bursts']}
        available = list(retained.values()) + list(discarded.values())
        slots = [(row['swath'], row['burst_id']) for row in available]
        counts = Counter(row['swath'] for row in retained.values())
        ranges = {swath: [min(row['burst_id'] for row in retained.values() if row['swath'] == swath),
                          max(row['burst_id'] for row in retained.values() if row['swath'] == swath)]
                  for swath in counts}
        if (coverage['partial'] != bool(missing or discarded) or
                coverage['retained_burst_count'] != len(retained) or
                coverage['available_burst_count'] != len(available) or
                len(retained) != len(group['bursts']) or
                len(discarded) != len(coverage['discarded_bursts']) or
                len(missing) != len(coverage['missing_bursts']) or
                retained.keys() & discarded.keys() or missing & set(slots) or len(set(slots)) != len(slots) or
                not set(group['requested']) <= retained.keys() | discarded.keys() or
                set(group['context']) != retained.keys() - set(group['requested']) or
                set(counts) != {row['swath'] for row in available} or ranges != coverage['retained_ranges']):
            raise ValueError('Inconsistent retained, unavailable or discarded burst records')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f'Invalid partial coverage for {group.get("id", "acquisition")}: {exc}') from exc
