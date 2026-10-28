"""Metadata-only geographic burst discovery using COMPASS-compatible s1-reader."""
from dataclasses import dataclass
from datetime import datetime
import logging
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Burst:
    id: str
    swath: int
    start: datetime
    native: object = None


def load_safe_bursts(path, orbit_path, swath_num, pol='vv', burst_ids=None, flag_apply_eap=True):
    """Use native parsing with explicit non-metadata paths on macOS SAFE copies.

    s1-reader's directory loader takes the first os.listdir match, including
    AppleDouble files. Its public XML parser retains identical science while
    avoiding any dependency on directory insertion order.
    """
    from s1reader.s1_reader import load_bursts, burst_from_xml
    safe = Path(path)
    annotations = safe / 'annotation'
    if not safe.is_dir() or not any(p.name.startswith('._') for p in annotations.glob('*')):
        return load_bursts(path, orbit_path, swath_num, pol, burst_ids=burst_ids, flag_apply_eap=flag_apply_eap)
    from .input_preparation import annotation_paths
    pol = pol.lower()
    if swath_num not in (1, 2, 3) or pol not in ('hh', 'hv', 'vv', 'vh'):
        raise ValueError('Invalid Sentinel-1 swath or polarization')
    paths = annotation_paths(safe)
    def annotation(swath):
        matches = [p for p in paths if f'iw{swath}-slc-{pol}' in p.lower()]
        if len(matches) != 1:
            raise ValueError(f'Expected one IW{swath} {pol} annotation in {safe}')
        return str(safe / matches[0])
    rasters = sorted(p for p in (safe / 'measurement').glob(f'*iw{swath_num}-slc-{pol}*tiff')
                     if not p.name.startswith('.'))
    if len(rasters) > 1:
        raise ValueError(f'Ambiguous IW{swath_num} {pol} measurement in {safe}')
    items = burst_from_xml(annotation(swath_num), str(orbit_path), str(rasters[0]) if rasters else '',
                           annotation(2), flag_apply_eap=flag_apply_eap)
    if burst_ids:
        ids = burst_ids if isinstance(burst_ids, (list, tuple, set)) else [burst_ids]
        wanted = set(map(str, ids))
        items = [b for b in items if str(b.burst_id) in wanted]
    return items


def inventory(safe, orbit, polarization, swaths, loader=None):
    native_loader = loader is None or loader.__module__.startswith('s1reader')
    if loader is None:
        loader = load_safe_bursts
    result = []
    for swath in swaths:
        annotation = None
        if native_loader:
            from .input_preparation import annotation_paths, safe_xml
            annotations = [p for p in annotation_paths(Path(safe))
                           if f'iw{swath}-slc-{polarization.lower()}' in p.lower()]
            if len(annotations) == 1:
                annotation = safe_xml(Path(safe), annotations[0])
                bursts = annotation.find('swathTiming/burstList')
                if bursts is not None and bursts.get('count') == '0' and len(bursts) == 0:
                    # Keep the annotation on disk for neighboring-swath
                    # corrections, but do not ask s1-reader to parse dummy data.
                    continue
        try:
            items = loader(str(safe), str(orbit), swath, polarization.lower(), flag_apply_eap=False)
            if annotation is not None:
                from .annotation_geometry import normalize_burst_geometry
                items = normalize_burst_geometry(safe, swath, polarization, items, annotation)
        except Exception as exc:
            raise ValueError(f"Cannot parse {Path(safe).name} IW{swath} {polarization}: {exc}") from exc
        result.extend(Burst(str(b.burst_id), swath, b.sensing_start, b) for b in items)
    if result and native_loader:
        validate_orbit(orbit, result)
    return result


def validate_orbit(path, bursts):
    tree = ET.parse(path)
    header = tree.find('Earth_Explorer_Header/Fixed_Header')
    if header is None or header.findtext('File_Type') != 'AUX_POEORB':
        raise ValueError('Orbit must be a precise AUX_POEORB file')
    mission = header.findtext('Mission', '').replace('Sentinel-', 'S')
    for b in bursts:
        if b.native.platform_id != mission:
            raise ValueError(f'Orbit mission {mission} differs from SAFE {b.native.platform_id}')
    def parse(text):
        return datetime.fromisoformat(text.removeprefix('UTC='))
    start = parse(header.findtext('Validity_Period/Validity_Start'))
    stop = parse(header.findtext('Validity_Period/Validity_Stop'))
    if any(not start <= b.start <= stop for b in bursts):
        raise ValueError('Precise orbit validity does not cover burst sensing times')
    times = [parse(x.text) for x in tree.findall('Data_Block/List_of_OSVs/OSV/UTC')]
    if len(times) < 4 or any(y <= x for x,y in zip(times,times[1:])):
        raise ValueError('Orbit state vectors must have increasing times')
    if any(not times[0] <= b.start <= times[-1] for b in bursts):
        raise ValueError('Orbit state vectors do not bracket burst sensing times')


def intersect(reference, secondary):
    def index(items):
        out = {}
        for b in items:
            if b.id in out:
                raise ValueError(f"Ambiguous duplicate geographic burst ID {b.id}")
            out[b.id] = b
        return out
    ref, sec = index(reference), index(secondary)
    common = []
    for key in sorted(ref.keys() & sec.keys()):
        a, b = ref[key], sec[key]
        if a.swath != b.swath:
            raise ValueError(f"Swath mismatch for burst {key}")
        if b.start <= a.start:
            raise ValueError("Secondary acquisition must follow reference acquisition")
        common.append((a, b))
    missing = len(ref.keys() ^ sec.keys())
    if missing:
        logging.warning("%d burst IDs do not occur in both acquisitions", missing)
    return common


def discover(cfg, loader=None):
    inp = cfg["inputs"]
    rows = [inventory(inp[f"{side}_safe"], inp[f"{side}_orbit"], inp["polarization"], inp["subswaths"], loader)
            for side in ("reference", "secondary")]
    return intersect(*rows)


def discover_validated(cfg):
    """Validate shared scene/orbit metadata, then inventory each scene once."""
    from .initialization import read_scene, read_precise_orbit, order_pair, ORBIT_START_MARGIN_SECONDS, ORBIT_STOP_MARGIN_SECONDS
    from datetime import timedelta
    inp = cfg['inputs']
    scenes = [read_scene(inp[f'{s}_safe']) for s in ('reference', 'secondary')]
    if order_pair(scenes) != tuple(scenes):
        raise ValueError('Secondary acquisition must follow reference acquisition')
    for side, scene in zip(('reference', 'secondary'), scenes):
        if inp['polarization'] not in scene.polarizations:
            raise ValueError(f'{side} scene does not contain {inp["polarization"]}')
        try:
            orbit = read_precise_orbit(inp[f'{side}_orbit'])
        except (ET.ParseError, OSError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f'{side} precise orbit metadata is invalid: {exc}') from exc
        start = scene.start - timedelta(seconds=ORBIT_START_MARGIN_SECONDS)
        stop = scene.stop + timedelta(seconds=ORBIT_STOP_MARGIN_SECONDS)
        if (orbit is None or orbit.mission != scene.mission or
                not orbit.start <= start < stop <= orbit.stop or
                not orbit.vector_start <= start < stop <= orbit.vector_stop):
            raise ValueError(f'{side} precise orbit incompatible with scene mission/times (including interpolation margins)')
    return discover(cfg)


def select_pair(cfg, common=None):
    common = discover(cfg) if common is None else common
    wanted = cfg["inputs"].get("burst_id")
    for pair in common:
        if pair[0].id == wanted:
            a, b = (p.native for p in pair)
            if a is not None and b is not None:
                for key in ("polarization", "orbit_direction"):
                    if getattr(a, key) != getattr(b, key):
                        raise ValueError(f"Incompatible burst {key}")
                ga, gb = a.as_isce3_radargrid(), b.as_isce3_radargrid()
                if any(not math.isfinite(v) or v <= 0 for g in (ga, gb)
                       for v in (g.range_pixel_spacing, g.prf)):
                    raise ValueError('Incompatible burst sampling: spacing/PRF must be finite and positive')
                if abs(ga.range_pixel_spacing / gb.range_pixel_spacing - 1) > 1e-6 or abs(ga.prf / gb.prf - 1) > 1e-4:
                    raise ValueError("Incompatible burst sampling")
            return pair
    raise ValueError(f"Burst {wanted!r} is not common; use glacier-offsets list-bursts to choose an ID")
