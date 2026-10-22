"""Local scene metadata and precise-orbit resolution for pair configuration.

Scene readers and the orbit catalog are independent of pair generation so a
future stack planner can index each acquisition once and reuse this logic.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

from .config import load_defaults, merge_config

# Matches installed s1-reader's ANX-selection margin and interpolation padding.
ORBIT_START_MARGIN_SECONDS = (12 * 86400 / 175) + 60
ORBIT_STOP_MARGIN_SECONDS = 60


def utc_time(text):
    value = datetime.fromisoformat(text.strip().removeprefix('UTC=').replace('Z', '+00:00'))
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def local_xml(content):
    root = ET.fromstring(content)
    for element in root.iter():
        element.tag = element.tag.rsplit('}', 1)[-1]
    return root


@dataclass(frozen=True)
class SceneMetadata:
    path: Path
    mission: str
    start: datetime
    stop: datetime
    polarizations: tuple[str, ...]
    relative_orbit: int
    direction: str


def read_scene(path):
    """Read only manifest.safe, from one unpacked SAFE or one local ZIP."""
    path = Path(path).expanduser().resolve(strict=True)
    try:
        if path.is_dir():
            root = local_xml((path / 'manifest.safe').read_bytes())
        elif path.suffix.lower() == '.zip':
            with ZipFile(path) as archive:
                names = [name for name in archive.namelist() if Path(name).name == 'manifest.safe']
                if len(names) != 1:
                    raise ValueError('ZIP must contain exactly one manifest.safe')
                root = local_xml(archive.read(names[0]))
        else:
            raise ValueError('Expected an unpacked SAFE directory or ZIP file')
        acquisition = root.find('.//acquisitionPeriod')
        platform = root.find('.//platform')
        start, stop = (utc_time(acquisition.findtext(key)) for key in ['startTime', 'stopTime'])
        if platform.findtext('familyName', '').upper() != 'SENTINEL-1':
            raise ValueError('Only Sentinel-1 acquisitions are supported')
        mission = 'S1' + platform.findtext('number', '').strip().upper()
        if mission not in {'S1A', 'S1B', 'S1C', 'S1D'}:
            raise ValueError(f'Unsupported mission {mission}')
        if root.findtext('.//mode') != 'IW' or root.findtext('.//productType') != 'SLC':
            raise ValueError('Only Sentinel-1 IW SLC acquisitions are supported')
        polarizations = tuple(sorted({e.text.strip().upper() for e in root.iter('transmitterReceiverPolarisation')}))
        tracks = {int(e.text) for e in root.iter('relativeOrbitNumber')}
        direction = root.findtext('.//pass', '').upper()
        if len(tracks) != 1 or direction not in {'ASCENDING', 'DESCENDING'}:
            raise ValueError('Manifest must identify one relative orbit and an orbit direction')
        if not polarizations or not set(polarizations) <= {'HH', 'HV', 'VV', 'VH'} or stop <= start:
            raise ValueError('Invalid acquisition interval or polarization metadata')
        return SceneMetadata(path, mission, start, stop, polarizations, tracks.pop(), direction)
    except (ET.ParseError, BadZipFile, OSError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f'Cannot read scene metadata from {path}: {exc}') from exc


@dataclass(frozen=True)
class PreciseOrbit:
    path: Path
    mission: str
    start: datetime
    stop: datetime
    vector_start: datetime
    vector_stop: datetime
    creation: datetime
    version: int


def read_precise_orbit(path):
    root = local_xml(Path(path).read_bytes())
    header = root.find('Earth_Explorer_Header/Fixed_Header')
    if header is None or header.findtext('File_Type') != 'AUX_POEORB':
        return None
    mission = header.findtext('Mission', '').upper().replace('SENTINEL-', 'S')
    start = utc_time(header.findtext('Validity_Period/Validity_Start'))
    stop = utc_time(header.findtext('Validity_Period/Validity_Stop'))
    times = [utc_time(e.text) for e in root.findall('Data_Block/List_of_OSVs/OSV/UTC')]
    if stop <= start or len(times) < 4 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Orbit validity and state-vector times must increase')
    created = header.findtext('Source/Creation_Date')
    return PreciseOrbit(Path(path).resolve(), mission, start, stop, times[0], times[-1],
                        utc_time(created) if created else datetime.min,
                        int(header.findtext('File_Version', '0')))


class OrbitCatalog:
    """Index a local orbit directory once; resolve each scene independently."""
    def __init__(self, directory):
        directory = Path(directory).expanduser().resolve(strict=True)
        if not directory.is_dir():
            raise ValueError('Orbit directory must be a directory')
        self.orbits = []
        for root, directories, files in os.walk(directory):
            directories[:] = sorted(name for name in directories if not name.startswith('.'))
            for name in sorted(files):
                path = Path(root) / name
                if name.startswith('.') or not path.is_file() or path.suffix.lower() not in {'.eof', '.xml'}:
                    continue
                try:
                    orbit = read_precise_orbit(path)
                    if orbit is not None:
                        self.orbits.append(orbit)
                except (ET.ParseError, OSError, AttributeError, TypeError, ValueError) as exc:
                    logging.warning('Ignoring invalid orbit candidate %s: %s', path, exc)
        if not self.orbits:
            raise ValueError(f'No valid precise AUX_POEORB files in {directory}')

    def resolve(self, scene):
        start = scene.start - timedelta(seconds=ORBIT_START_MARGIN_SECONDS)
        stop = scene.stop + timedelta(seconds=ORBIT_STOP_MARGIN_SECONDS)
        candidates = [o for o in self.orbits if o.mission == scene.mission and
                      o.start <= start and o.stop >= stop and
                      o.vector_start <= start and o.vector_stop >= stop]
        if not candidates:
            raise ValueError(f'No precise orbit for {scene.mission} {scene.path.name} covering '
                             f'{start.isoformat()} through {stop.isoformat()} UTC '
                             '(including ascending-node/interpolation margins)')
        # Prefer the most recently generated precise solution, then version.
        # Path breaks ties deterministically, independently of directory order.
        candidates.sort(key=lambda o: (o.creation, o.version, str(o.path)))
        chosen = candidates[-1]
        if len(candidates) > 1:
            logging.info('%d precise orbits cover %s; selecting latest creation/version: %s',
                         len(candidates), scene.path.name, chosen.path)
        return chosen


def order_pair(scenes, max_separation_days=None):
    if len(scenes) != 2:
        raise ValueError('Pair initialization requires exactly two acquisitions')
    reference, secondary = sorted(scenes, key=lambda s: s.start)
    if reference.path == secondary.path or secondary.start <= reference.start:
        raise ValueError('Pair must contain two distinct acquisitions with increasing sensing times')
    if (reference.relative_orbit, reference.direction) != (secondary.relative_orbit, secondary.direction):
        raise ValueError('Acquisitions must share a relative orbit and orbit direction')
    if max_separation_days is not None:
        if not math.isfinite(max_separation_days) or max_separation_days <= 0:
            raise ValueError('Maximum separation must be finite and positive')
        days = (secondary.start - reference.start).total_seconds() / 86400
        if days > max_separation_days:
            raise ValueError(f'Pair separation {days:.9f} days exceeds maximum {max_separation_days:g} days')
    return reference, secondary


def make_pair_config(safes, dem, orbit_directory, output_directory=None, polarization=None,
                     burst_id=None, epsg=None, spacing_m=None, max_separation_days=None,
                     aoi=None, bbox=None, aoi_layer=None):
    if sum(v is not None for v in (burst_id, aoi, bbox)) > 1:
        raise ValueError('Choose --burst-id, --aoi or --bbox exclusively')
    if aoi_layer is not None and aoi is None:
        raise ValueError('--aoi-layer requires --aoi')
    reference, secondary = order_pair([read_scene(p) for p in safes], max_separation_days)
    common = set(reference.polarizations) & set(secondary.polarizations)
    if polarization is None:
        if len(common) != 1:
            raise ValueError(f'Specify --polarization; common polarizations: {sorted(common)}')
        polarization = common.pop()
    polarization = polarization.upper()
    if polarization not in reference.polarizations or polarization not in secondary.polarizations:
        raise ValueError(f'Polarization {polarization} is not present in both acquisitions')
    catalog = OrbitCatalog(orbit_directory)
    orbits = [catalog.resolve(scene) for scene in [reference, secondary]]
    dem = Path(dem).expanduser().resolve(strict=True)
    from osgeo import gdal
    from pyproj import CRS
    from .geocode import validate_geocoding
    gdal.UseExceptions()
    raster = gdal.Open(str(dem))
    if raster is None or not raster.GetProjection():
        raise ValueError('DEM must be a GDAL-readable raster with a declared CRS')
    if epsg is None:
        epsg = CRS.from_wkt(raster.GetProjection()).to_epsg()
        if epsg is None:
            raise ValueError('DEM CRS has no EPSG identifier; specify projected metre --epsg')
    raster = None
    defaults = load_defaults()
    geo_overrides = {'epsg': epsg}
    if spacing_m is not None:
        geo_overrides['spacing_m'] = spacing_m
    geocoding = merge_config(defaults['geocoding'], geo_overrides)
    validate_geocoding(geocoding)
    if output_directory is None:
        names = [s.path.name.removesuffix('.zip').removesuffix('.ZIP').removesuffix('.SAFE')
                 for s in [reference, secondary]]
        output_directory = Path('work') / '__'.join(names)
    cfg = merge_config(defaults, {
        'inputs': {'reference_safe': str(reference.path), 'secondary_safe': str(secondary.path),
                   'reference_orbit': str(orbits[0].path), 'secondary_orbit': str(orbits[1].path),
                   'dem': str(dem), 'polarization': polarization},
        'output': {'directory': str(Path(output_directory).expanduser().resolve())},
        'geocoding': geocoding,
    })
    if burst_id:
        cfg['inputs']['burst_id'] = burst_id
        from .inputs import select_pair
        select_pair(cfg)  # Metadata-only validation, never coregistration/matching.
    if aoi is not None or bbox is not None:
        cfg['aoi'] = {'path': str(Path(aoi).expanduser().resolve())} if aoi is not None else {'bbox': bbox}
        if aoi_layer is not None:
            cfg['aoi']['layer'] = aoi_layer
        from .aoi import select_aoi
        pairs, _ = select_aoi(cfg)
        if not pairs:
            raise ValueError('AOI selects no common bursts')
    return cfg
