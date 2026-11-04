"""Ordered adjacent swaths and the sign of their relative azimuth contrast."""
import re


def validate_pair(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError('Choose ordered adjacent swaths: 1 2 or 2 3')
    pair = tuple(value)
    if any(type(s) is not int for s in pair) or pair not in ((1, 2), (2, 3)):
        raise ValueError('Choose ordered adjacent swaths: 1 2 or 2 3')
    return pair


def select_pair(sources, requested=None):
    available = set()
    for source in sources:
        match = re.search(r'_iw([123])$', source['burst_id'])
        if not match:
            raise ValueError(f"Invalid source burst ID: {source['burst_id']}")
        available.add(int(match[1]))
    if requested is None:
        if available == {1, 2, 3}:
            raise ValueError('Reference contains three swaths; select --swaths 1 2 or --swaths 2 3')
        requested = sorted(available)
        if len(requested) != 2:
            raise ValueError(f'Calibration requires two adjacent swaths; reference swaths are {requested}')
    pair = validate_pair(requested)
    if not set(pair) <= available:
        raise ValueError(f'Requested swaths {list(pair)} are absent from reference swaths {sorted(available)}')
    return pair


def recorded_pair(record):
    """Old artifacts without an explicit pair describe IW1/IW2 only."""
    pair = record.get('swaths', (1, 2))
    return validate_pair(pair.tolist() if hasattr(pair, 'tolist') else pair)


def names(pair):
    return tuple(f'iw{s}' for s in validate_pair(pair))


def allocation(pair):
    low, high = validate_pair(pair)
    return f'IW{low}=-delta/2, IW{high}=+delta/2; subtract in original pixels'
