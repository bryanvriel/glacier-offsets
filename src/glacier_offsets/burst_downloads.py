"""AOI-driven ASF burst planning and atomic, restartable SAFE collections.

Optional ASF/burst2safe imports stay at the service boundary. The manifest is
also the preparation contract: incomplete collections must never lose dates
silently when consumed by the recursive SAFE scanner.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta
from importlib.metadata import version
import json
import logging
from pathlib import Path
import re
import tempfile
import time as clock
import xml.etree.ElementTree as ET

from shapely.geometry import shape, mapping

from .aoi import load_aoi, polygon, project_footprint, wgs84_parts
from .initialization import read_scene, utc_time
from .products import checksum, identity, write_json

METHOD = 's1-burst-download-v1'
MANIFEST = 'download_manifest.json'


def filesystem_metadata(path):
    """AppleDouble/resource-fork sidecars are not collection or SAFE content."""
    return any(part == '.DS_Store' or part.startswith('._') for part in Path(path).parts)


def dependencies():
    try:
        import asf_search
        from burst2safe.burst2safe import burst2safe
        from burst2safe.safe import Safe
        from burst2safe.utils import create_burst_info
    except ImportError as exc:
        raise ImportError('Burst downloads require: python -m pip install -e ".[burst-download]"') from exc
    return asf_search, burst2safe, Safe, create_burst_info


def request_from_args(args):
    if any(not re.fullmatch(r'\d{4}-\d{2}-\d{2}', v) for v in (args.start_date, args.end_date)):
        raise ValueError('Dates must use YYYY-MM-DD')
    start, end = (date.fromisoformat(v) for v in (args.start_date, args.end_date))
    if start > end:
        raise ValueError('start-date must be on or before end-date')
    if not 1 <= args.relative_orbit <= 175:
        raise ValueError('relative-orbit must be between 1 and 175')
    spec = {'path': str(args.aoi.expanduser().resolve()), 'layer': args.aoi_layer} if args.aoi else {'bbox': args.bbox}
    if args.aoi_layer and not args.aoi:
        raise ValueError('--aoi-layer requires --aoi')
    aoi = load_aoi(spec)
    request = {'aoi': aoi.report, 'relative_orbit': args.relative_orbit,
               'polarization': args.polarization, 'subswaths': sorted(set(args.subswaths or (1, 2, 3))),
               'start_date': start.isoformat(), 'end_date': end.isoformat()}
    return request, aoi


def selection_identity(request):
    return identity({k: request[k] for k in ('relative_orbit', 'polarization', 'subswaths')} |
                    {'aoi': request['aoi']['geometry_identity']})


def check_extension(old, request):
    previous = old['request']
    if (old.get('method') != METHOD or selection_identity(previous) != selection_identity(request) or
            request['start_date'] > previous['start_date'] or request['end_date'] < previous['end_date']):
        raise ValueError('Changed selection or narrowed dates require a new output collection')


def record(product, request):
    """Validate local ASF metadata, including fields used by burst2safe 2.0.x."""
    p = product.properties
    try:
        mission = {'SENTINEL-1A': 'S1A', 'SENTINEL-1B': 'S1B',
                   'SENTINEL-1C': 'S1C', 'SENTINEL-1D': 'S1D'}[p['platform'].upper()]
        swath = p['burst']['subswath'].upper()
        pol = p['polarization'].upper()
        track, orbit = int(p['pathNumber']), int(p['orbit'])
        start, stop = utc_time(p['startTime']), utc_time(p['stopTime'])
        if (p['beamModeType'] != 'IW' or pol != request['polarization'] or
                track != request['relative_orbit'] or swath not in [f'IW{s}' for s in request['subswaths']] or
                not request['start_date'] <= start.date().isoformat() <= request['end_date']):
            return None
        granule = p['fileID']
        burst_id, index = int(p['burst']['relativeBurstID']), int(p['burst']['burstIndex'])
        parent = product.umm['InputGranules'][0].split('-')[0]
        if (not isinstance(granule, str) or not granule.endswith('-BURST') or
                not parent.startswith(mission + '_IW_SLC') or start > stop or
                orbit <= 0 or burst_id < 0 or index < 0 or
                p['flightDirection'].upper() not in {'ASCENDING', 'DESCENDING'}):
            raise ValueError('invalid granule, parent, interval, orbit, index or direction')
        identifier, parent_fields = granule.split('_'), parent.split('_')
        if (identifier[0] != 'S1' or int(identifier[1]) != burst_id or identifier[2] != swath or
                identifier[4] != pol or
                abs((datetime.strptime(identifier[3], '%Y%m%dT%H%M%S') - start).total_seconds()) > 2 or
                int(parent_fields[7]) != orbit or
                not datetime.strptime(parent_fields[5], '%Y%m%dT%H%M%S') <= start <=
                    datetime.strptime(parent_fields[6], '%Y%m%dT%H%M%S')):
            raise ValueError('granule/parent identifiers disagree with burst metadata')
        geom = wgs84_parts(polygon(shape(product.geometry), f'ASF {granule}'))
        return {'granule': granule, 'parent': parent, 'mission': mission, 'absolute_orbit': orbit,
                'relative_orbit': track, 'swath': swath, 'polarization': pol, 'burst_id': burst_id,
                'burst_index': index, 'start': start.isoformat(), 'stop': stop.isoformat(),
                'direction': p['flightDirection'].upper(), 'geometry': json.loads(json.dumps(mapping(geom)))}
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(f'Invalid ASF burst metadata: {exc}') from exc


def plan_downloads(request, aoi, asf, safe_class, create_info):
    """Search polygon exteriors, then intersect in the AOI's native CRS.

    Context searches use explicit geographic burst IDs without an AOI filter,
    allowing disconnected polygons (and holes) to require intervening imagery.
    """
    start = datetime.combine(date.fromisoformat(request['start_date']), time.min)
    end = datetime.combine(date.fromisoformat(request['end_date']) + timedelta(days=1), time.min)
    options = dict(dataset=asf.constants.DATASET.SLC_BURST, beamMode='IW',
                   relativeOrbit=request['relative_orbit'], polarization=request['polarization'],
                   start=start.isoformat() + 'Z', end=end.isoformat() + 'Z')
    geographic = wgs84_parts(aoi.geometry, aoi.crs)
    parts = geographic.geoms if geographic.geom_type == 'MultiPolygon' else (geographic,)
    products, rows, groups = {}, {}, defaultdict(list)

    def ingest(results):
        found = []
        for product in results:
            row = record(product, request)
            if row is None:
                continue
            name = row['granule']
            if name in rows and rows[name] != row:
                raise ValueError(f'Conflicting ASF records for {name}')
            rows[name], products[name] = row, product
            found.append(name)
        return found

    candidates = set()
    for part in parts:
        # ASF does not accept holes; retain them for the local intersection.
        from shapely.geometry import Polygon
        candidates.update(ingest(asf.search(intersectsWith=Polygon(part.exterior).wkt, **options)))
    for name in sorted(candidates):
        row = rows[name]
        geom = project_footprint(shape(row['geometry']), aoi.crs)
        if aoi.geometry.intersection(geom).area > 0:
            groups[(row['mission'], row['absolute_orbit'])].append(name)
    acquisitions = []
    for (mission, orbit), requested in sorted(groups.items()):
        group = {'id': f'{mission}_{orbit:06d}', 'mission': mission, 'absolute_orbit': orbit,
                 'requested': sorted(requested), 'context': [], 'bursts': [], 'errors': []}
        acquisitions.append(group)
        chosen = set(requested)
        try:
            for swath in sorted({rows[n]['swath'] for n in requested}):
                subset = [rows[n] for n in requested if rows[n]['swath'] == swath]
                lo, hi = min(r['burst_id'] for r in subset), max(r['burst_id'] for r in subset)
                ids = [f'{request["relative_orbit"]:03d}_{i:06d}_{swath}' for i in range(lo, hi + 1)]
                known = {r['burst_id'] for r in rows.values() if r['mission'] == mission and
                         r['absolute_orbit'] == orbit and r['swath'] == swath}
                if not set(range(lo, hi + 1)) <= known:
                    ingest(asf.search(absoluteOrbit=orbit, fullBurstID=ids, **options))
                available = [n for n, r in rows.items() if r['mission'] == mission and
                             r['absolute_orbit'] == orbit and r['swath'] == swath and lo <= r['burst_id'] <= hi]
                counts = Counter(rows[n]['burst_id'] for n in available)
                if any(counts[i] != 1 for i in range(lo, hi + 1)):
                    raise ValueError(f'{swath}: missing interior or ambiguous duplicate burst products: {dict(counts)}')
                chosen.update(available)
            group['context'] = sorted(chosen - set(requested))
            group['bursts'] = [rows[n] for n in sorted(chosen)]
            group['date'] = min(r['start'] for r in group['bursts'])[:10]
            if len({r['direction'] for r in group['bursts']}) != 1:
                raise ValueError('Incompatible orbit directions')
            # create_burst_info does not create directories or transfer imagery.
            safe_class.check_group_validity([create_info(products[n], Path('.')) for n in sorted(chosen)])
        except Exception as exc:
            group['errors'].append(str(exc))
            group['bursts'] = [rows[n] for n in sorted(chosen)]
            group['date'] = min(r['start'] for r in group['bursts'])[:10]
    dates = Counter(g['date'] for g in acquisitions)
    for group in acquisitions:
        if dates[group['date']] > 1:
            group['errors'].append('Multiple acquisition groups on one UTC date violate stack constraints')
    return acquisitions


def _xml(path):
    root = ET.parse(path).getroot()
    for node in root.iter():
        node.tag = node.tag.rsplit('}', 1)[-1]
    return root


def validate_safe(path, group, request):
    """Check every complex sample is readable using bounded TIFF windows.

    Values, including isolated complex zeros, are never changed or classified.
    Empty helper annotations are validated but excluded from imagery coverage.
    """
    from osgeo import gdal
    import numpy as np
    gdal.UseExceptions()
    path = Path(path)
    scene = read_scene(path)
    if (scene.mission != group['mission'] or scene.relative_orbit != request['relative_orbit'] or
            request['polarization'] not in scene.polarizations or scene.start.date().isoformat() != group['date']):
        raise ValueError('Published SAFE scene metadata differs from the download plan')
    manifest = _xml(path / 'manifest.safe')
    for node in manifest.iter('fileLocation'):
        href = node.get('href', '')
        target = (path / href).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file():
            raise ValueError(f'Missing or unsafe manifest reference: {href}')
    expected = Counter(r['swath'] for r in group['bursts'])
    seen, helpers, measurements, annotations = Counter(), [], set(), set()
    for annotation in sorted((path / 'annotation').glob('*.xml')):
        if filesystem_metadata(annotation.relative_to(path)):
            continue
        root = _xml(annotation)
        swath = root.findtext('adsHeader/swath')
        pol = root.findtext('adsHeader/polarisation')
        if swath not in {'IW1', 'IW2', 'IW3'} or swath in annotations:
            raise ValueError(f'Invalid/duplicate annotation swath: {swath}')
        annotations.add(swath)
        if pol != request['polarization']:
            raise ValueError(f'Unexpected annotation polarization: {annotation.name}')
        for field in ('generalAnnotation/productInformation/radarFrequency',
                      'generalAnnotation/productInformation/rangeSamplingRate',
                      'imageAnnotation/imageInformation/azimuthTimeInterval',
                      'imageAnnotation/imageInformation/slantRangeTime'):
            value = float(root.findtext(field, 'nan'))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Missing/invalid reader metadata {field}: {annotation.name}')
        utc_time(root.findtext('imageAnnotation/imageInformation/ascendingNodeTime', ''))
        bursts = root.find('swathTiming/burstList')
        if bursts is None or int(bursts.get('count', '-1')) != len(bursts):
            raise ValueError(f'Invalid annotation burst list: {annotation.name}')
        if not len(bursts):
            if swath in expected:
                raise ValueError(f'Empty annotation for requested swath {swath}')
            helpers.append(swath)
            continue
        seen[swath] += len(bursts)
        width = int(root.findtext('imageAnnotation/imageInformation/numberOfSamples'))
        height = int(root.findtext('imageAnnotation/imageInformation/numberOfLines'))
        lines = int(root.findtext('swathTiming/linesPerBurst'))
        samples = int(root.findtext('swathTiming/samplesPerBurst'))
        if min(width, height, lines, samples) <= 0 or height != lines * len(bursts) or width != samples:
            raise ValueError(f'Inconsistent annotation dimensions: {annotation.name}')
        times = [utc_time(b.findtext('azimuthTime')) for b in bursts]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Nonincreasing annotation burst times')
        planned_times = sorted(utc_time(r['start']) for r in group['bursts'] if r['swath'] == swath)
        # ASF timestamps have only second precision, while SAFE annotations
        # preserve fractional sensing times.
        if len(times) != len(planned_times) or any(abs((a-b).total_seconds()) > 2
                                                   for a, b in zip(times, planned_times)):
            raise ValueError('Annotation sensing times differ from planned bursts')
        if not root.findall('.//geolocationGridPoint'):
            raise ValueError('Populated annotation has no geolocation points')
        matches = list((path / 'measurement').glob(annotation.stem + '.tif*'))
        if len(matches) != 1:
            raise ValueError(f'Missing/ambiguous measurement for {annotation.name}')
        measurement = matches[0]
        measurements.add(measurement.resolve())
        ds = gdal.Open(str(measurement), gdal.GA_ReadOnly)
        if (ds is None or ds.RasterCount != 1 or (ds.RasterXSize, ds.RasterYSize) != (width, height) or
                not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType)):
            raise ValueError(f'Invalid complex measurement: {measurement}')
        for y in range(0, height, 128):
            for x in range(0, width, 2048):
                data = ds.ReadAsArray(x, y, min(2048, width-x), min(128, height-y))
                if data is None or not np.isfinite(data).all():
                    raise ValueError(f'Unreadable/nonfinite measurement: {measurement}')
        ds = None
    if seen != expected or set(seen) | set(helpers) != {'IW1', 'IW2', 'IW3'}:
        raise ValueError(f'Incomplete imagery or helper annotations: {dict(seen)}, helpers={helpers}')
    actual = {p.resolve() for p in (path / 'measurement').iterdir()
              if p.is_file() and not filesystem_metadata(p.relative_to(path))}
    if actual != measurements:
        raise ValueError('Unplanned measurement files in reconstructed SAFE')
    for xml in path.rglob('*.xml'):
        if not filesystem_metadata(xml.relative_to(path)):
            _xml(xml)
    files = {p.relative_to(path).as_posix(): checksum(p) for p in sorted(path.rglob('*'))
             if p.is_file() and not filesystem_metadata(p.relative_to(path))}
    return {'files': files, 'bytes': sum((path / name).stat().st_size for name in files),
            'imagery_bursts': dict(seen), 'helper_swaths': sorted(helpers)}


def verify_files(path, validation):
    files = validation.get('files', {})
    actual = {p.relative_to(path).as_posix() for p in path.rglob('*')
              if p.is_file() and not filesystem_metadata(p.relative_to(path))}
    if not files or actual != set(files):
        raise ValueError(f'Incomplete or changed SAFE file inventory: {path}')
    for name, digest in files.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or checksum(target) != digest:
            raise ValueError(f'SAFE checksum mismatch: {path / name}')


def collection_identity(manifest):
    """Stable handoff identity; verification timings/reuse flags do not matter."""
    try:
        return identity({'request': manifest['request'], 'acquisitions': [
            {key: group[key] for key in ('id', 'bursts', 'safe', 'validation')}
            for group in manifest['acquisitions']]})
    except (KeyError, TypeError) as exc:
        raise ValueError('Inconsistent burst download manifest') from exc


def managed_collection(path):
    """Return verified collection provenance for either root or safe/ input."""
    from .stack import scan_safes
    path = Path(path).resolve()
    root = path.parent if path.name == 'safe' else path
    manifest_path = root / MANIFEST
    if not manifest_path.exists():
        if path.name == 'safe' and (root / '.staging').exists():
            raise ValueError('Managed burst collection has no download manifest')
        return None
    m = json.loads(manifest_path.read_text())
    groups = m.get('acquisitions')
    if (m.get('method') != METHOD or m.get('status') != 'complete' or not groups or
            any(g.get('status') != 'complete' or g.get('errors') for g in groups)):
        raise ValueError('Burst collection is running, failed or incomplete; rerun download-bursts')
    safes = []
    for group in groups:
        safe = (root / group['safe']).resolve()
        if not safe.is_relative_to((root / 'safe').resolve()) or not safe.is_dir():
            raise ValueError('Invalid managed SAFE path')
        verify_files(safe, group['validation'])
        safes.append(safe)
    if len(set(safes)) != len(safes) or set(scan_safes(root / 'safe')) != set(safes):
        raise ValueError('Burst collection inventory is inconsistent with its manifest')
    return {'manifest': str(manifest_path), 'sha256': checksum(manifest_path),
            'identity': collection_identity(m), 'download': m,
            'safe_directory': str(root / 'safe')}


@contextmanager
def collection_lock(root):
    """OS lock releases on interruption; the hidden lock file is reusable."""
    import fcntl
    with (root / '.download.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another download is using this collection') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def download(request, groups, root, converter):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with collection_lock(root):
        return _download(request, groups, root, converter)


def _download(request, groups, root, converter):
    manifest_path = root / MANIFEST
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if old:
        check_extension(old, request)
    elif any(p.name != '.download.lock' and not filesystem_metadata(p.name) for p in root.iterdir()):
        raise ValueError('Output collection must be empty or contain a download manifest')
    previous = {g['id']: g for g in old.get('acquisitions', [])}
    # A temporarily empty/incomplete provider response must not drop known dates.
    missing = set(previous) - {g['id'] for g in groups}
    if missing:
        raise ValueError(f'ASF search omitted existing acquisition groups: {sorted(missing)}')
    versions = {p: version(p) for p in ('burst2safe', 'asf-search')}
    manifest = {'method': METHOD, 'status': 'running', 'request': request,
                'selection_identity': selection_identity(request), 'versions': versions,
                'acquisitions': groups}
    for group in groups:
        group['status'] = 'pending'
    write_json(manifest_path, manifest)
    for group in groups:
        if group['errors']:
            group['status'] = 'failed'
            logging.error('%s: %s', group['id'], '; '.join(group['errors']))
            write_json(manifest_path, manifest)
            continue
        began = clock.monotonic()
        try:
            prior = previous.get(group['id'])
            if prior and prior.get('status') == 'complete' and prior.get('bursts') == group['bursts']:
                safe = root / prior['safe']
                try:
                    verify_files(safe, prior['validation'])
                except (ValueError, OSError):
                    pass
                else:
                    group.update(safe=prior['safe'], validation=prior['validation'], status='complete', reused=True,
                                 versions=prior.get('versions', old.get('versions')),
                                 downloaded_bytes=prior.get('downloaded_bytes'),
                                 download_elapsed_seconds=prior.get('download_elapsed_seconds'))
                    logging.info('Reusing verified %s', group['id'])
                    continue
            group['status'] = 'running'
            write_json(manifest_path, manifest)
            # Check without append before upstream can try writing a .netrc file.
            from burst2safe.auth import check_earthdata_credentials
            check_earthdata_credentials(append=False)
            staging_root = root / '.staging'
            staging_root.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=group['id'] + '-', dir=staging_root) as scratch:
                scratch = Path(scratch)
                safe = Path(converter(granules=[r['granule'] for r in group['bursts']],
                                      all_anns=True, keep_files=True, work_dir=scratch)).resolve()
                if not safe.is_relative_to(scratch) or not safe.is_dir() or safe.suffix != '.SAFE':
                    raise ValueError('burst2safe returned an invalid staged SAFE path')
                validation = validate_safe(safe, group, request)
                downloaded = [p for p in scratch.rglob('*') if p.is_file() and not p.is_relative_to(safe)
                              and not filesystem_metadata(p.relative_to(scratch))]
                group['downloaded_bytes'] = sum(p.stat().st_size for p in downloaded)
                # Each visible directory contains exactly one complete acquisition.
                publication = scratch / 'publication'
                publication.mkdir()
                name = safe.name
                safe.rename(publication / name)
                destination = root / 'safe' / group['id']
                destination.parent.mkdir(exist_ok=True)
                backup = scratch / 'previous'
                if destination.exists():
                    destination.rename(backup)
                try:
                    publication.rename(destination)
                except BaseException:
                    if backup.exists():
                        backup.rename(destination)
                    raise
                group.update(safe=(destination / name).relative_to(root).as_posix(),
                             validation=validation, status='complete', reused=False, versions=versions,
                             download_elapsed_seconds=clock.monotonic() - began)
                logging.info('Published %s (%d bytes)', group['id'], validation['bytes'])
        except KeyboardInterrupt:
            group.update(status='interrupted', error='Interrupted')
            manifest['status'] = 'interrupted'
            write_json(manifest_path, manifest)
            raise
        except Exception as exc:
            # Keep secrets/URLs out of persisted exception messages from providers.
            detail = re.sub(r'https?://\S+', '[provider URL]', str(exc))
            import os
            for key in ('EARTHDATA_TOKEN', 'EARTHDATA_USERNAME', 'EARTHDATA_PASSWORD'):
                secret = os.environ.get(key)
                if secret:
                    detail = detail.replace(secret, '[redacted]')
            group.update(status='failed', error=f'{type(exc).__name__}: {detail}')
            logging.error('%s: %s', group['id'], group['error'])
        finally:
            group['elapsed_seconds'] = clock.monotonic() - began
            write_json(manifest_path, manifest)
    manifest['status'] = 'complete' if groups and all(g['status'] == 'complete' for g in groups) else 'failed'
    write_json(manifest_path, manifest)
    return 0 if manifest['status'] == 'complete' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='glacier-offsets download-bursts', description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--aoi', type=Path)
    selector.add_argument('--bbox', nargs=4, type=float, metavar=('W', 'S', 'E', 'N'))
    parser.add_argument('--aoi-layer')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--relative-orbit', required=True, type=int)
    parser.add_argument('--polarization', required=True, type=str.upper, choices=('HH', 'HV', 'VH', 'VV'))
    parser.add_argument('--subswaths', nargs='+', type=int, choices=(1, 2, 3))
    parser.add_argument('-o', '--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    try:
        request, aoi = request_from_args(args)
        manifest_path = args.output.expanduser().resolve() / MANIFEST
        if manifest_path.exists():
            check_extension(json.loads(manifest_path.read_text()), request)
        asf, converter, safe_class, create_info = dependencies()
        groups = plan_downloads(request, aoi, asf, safe_class, create_info)
        print(json.dumps({'request': request, 'acquisitions': groups}, indent=2, sort_keys=True))
        if args.dry_run:
            return 0 if groups and all(not g['errors'] for g in groups) else 2
        result = download(request, groups, args.output, converter)
        print(f'Download manifest: {args.output.expanduser().resolve() / MANIFEST}')
        return result
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.error('Burst download preflight failed (%s): %s', type(exc).__name__, exc)
        return 2
