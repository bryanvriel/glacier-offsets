"""KML 2.2 export of annotation-derived Sentinel-1 burst footprints."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from shapely.geometry.polygon import orient
from .initialization import OrbitCatalog, read_scene
from .inputs import Burst, validate_orbit

KML_NAMESPACE = 'http://www.opengis.net/kml/2.2'
ET.register_namespace('', KML_NAMESPACE)
SWATH_COLORS = {1: 'ffff8800', 2: 'ff00aa00', 3: 'ff4444ee'}  # aabbggrr


def element(parent, tag, text=None, **attributes):
    child = ET.SubElement(parent, f'{{{KML_NAMESPACE}}}{tag}', attributes)
    if text is not None:
        child.text = str(text)
    return child


def coordinates(points):
    pairs = [(float(p[0]), float(p[1])) for p in points]
    if not pairs or any(not math.isfinite(lon) or not math.isfinite(lat) or
                        not -180 <= lon <= 180 or not -90 <= lat <= 90 for lon, lat in pairs):
        raise ValueError('Footprint coordinates must be finite longitude/latitude degrees')
    return ' '.join(f'{lon:.10f},{lat:.10f},0' for lon, lat in pairs)


def append_polygon(parent, polygon):
    if polygon.geom_type != 'Polygon' or polygon.is_empty or not polygon.is_valid:
        raise ValueError('Burst has an empty or invalid annotation footprint')
    polygon = orient(polygon, sign=1.0)  # KML exterior counterclockwise; holes clockwise.
    geometry = element(parent, 'Polygon')
    element(geometry, 'tessellate', 1)
    element(geometry, 'altitudeMode', 'clampToGround')
    for name, ring in [('outerBoundaryIs', polygon.exterior)] + [('innerBoundaryIs', r) for r in polygon.interiors]:
        boundary = element(geometry, name)
        element(element(boundary, 'LinearRing'), 'coordinates', coordinates(ring.coords))


def burst_document(bursts, scene, polarization, orbit_path=None):
    if not bursts:
        raise ValueError('No bursts were found for the requested polarization/subswaths')
    root = ET.Element(f'{{{KML_NAMESPACE}}}kml')
    document = element(root, 'Document')
    element(document, 'name', scene.path.name)
    element(document, 'description',
            'Burst footprints from SAFE annotation geolocation grids; these are not valid-data masks.\n'
            f'SAFE: {scene.path}\nPolarization: {polarization}\n'
            f'Orbit: {orbit_path or "none; burst IDs use the annotation ascending-node time"}')
    for swath, color in SWATH_COLORS.items():
        style = element(document, 'Style', id=f'iw{swath}')
        # A center Point makes the name visible in Google Earth without an icon.
        element(element(style, 'IconStyle'), 'scale', 0)
        label = element(style, 'LabelStyle'); element(label, 'color', color); element(label, 'scale', .8)
        line = element(style, 'LineStyle'); element(line, 'color', color); element(line, 'width', 2)
        fill = element(style, 'PolyStyle'); element(fill, 'color', '40' + color[2:])
        element(fill, 'fill', 1); element(fill, 'outline', 1)
    seen = set()
    for swath in sorted({b.swath for b in bursts}):
        folder = element(document, 'Folder'); element(folder, 'name', f'IW{swath}')
        for burst in sorted((b for b in bursts if b.swath == swath), key=lambda b: b.id):
            if burst.id in seen:
                raise ValueError(f'Duplicate geographic burst ID {burst.id}')
            seen.add(burst.id)
            placemark = element(folder, 'Placemark')
            element(placemark, 'name', burst.id)
            element(placemark, 'description', f'{burst.id}\nIW{swath} {polarization}\nSensing start: {burst.start.isoformat()}Z')
            element(placemark, 'styleUrl', f'#iw{swath}')
            data = element(placemark, 'ExtendedData')
            for key, value in {'burst_id': burst.id, 'subswath': f'IW{swath}', 'polarization': polarization,
                               'sensing_start_UTC': burst.start.isoformat()+'Z', 'mission': scene.mission,
                               'safe_path': str(scene.path), 'orbit_path': str(orbit_path or ''),
                               'footprint_source': 'SAFE annotation geolocation grid',
                               'burst_id_ANX_source': 'precise orbit requested' if orbit_path else 'SAFE annotation'}.items():
                element(element(data, 'Data', name=key), 'value', value)
            geometry = element(placemark, 'MultiGeometry')
            borders = burst.native.border
            if not borders:
                raise ValueError(f'No annotation footprint for {burst.id}')
            for polygon in borders:  # Retains s1-reader's antimeridian-split parts.
                try:
                    append_polygon(geometry, polygon)
                except ValueError as exc:
                    raise ValueError(f'{burst.id}: {exc}') from exc
            center = burst.native.center
            if center is None or center.is_empty:
                raise ValueError(f'No annotation center for {burst.id}')
            point = element(geometry, 'Point'); element(point, 'altitudeMode', 'clampToGround')
            element(point, 'coordinates', coordinates(center.coords))
    ET.indent(root, space='  ')
    return ET.ElementTree(root)


def read_kml_bursts(safe, polarization=None, subswaths=(1, 2, 3), orbit_path=None, orbit_directory=None):
    scene = read_scene(safe)
    if polarization is None:
        if len(scene.polarizations) != 1:
            raise ValueError(f'Specify --polarization; available: {list(scene.polarizations)}')
        polarization = scene.polarizations[0]
    polarization = polarization.upper()
    if polarization not in scene.polarizations:
        raise ValueError(f'Polarization {polarization} is not present in this SAFE')
    if not subswaths or len(set(subswaths)) != len(subswaths) or any(s not in (1, 2, 3) for s in subswaths):
        raise ValueError('Subswaths must be distinct values from 1, 2, 3')
    if orbit_path and orbit_directory:
        raise ValueError('Specify either an orbit file or an orbit directory')
    if orbit_directory:
        orbit_path = OrbitCatalog(orbit_directory).resolve(scene).path
    elif orbit_path:
        orbit_path = Path(orbit_path).expanduser().resolve(strict=True)
    from s1reader.s1_reader import load_bursts
    from .annotation_geometry import normalize_burst_geometry
    bursts = []
    for swath in sorted(subswaths):
        native = load_bursts(str(scene.path), str(orbit_path) if orbit_path else '', swath,
                             polarization.lower(), flag_apply_eap=False)
        native = normalize_burst_geometry(scene.path, swath, polarization, native)
        bursts.extend(Burst(str(b.burst_id), swath, b.sensing_start, b) for b in native)
    if orbit_path:
        validate_orbit(orbit_path, bursts)
    return bursts, scene, polarization, orbit_path
