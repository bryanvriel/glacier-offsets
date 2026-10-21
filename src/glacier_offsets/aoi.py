"""Polygon AOIs and annotation-only, positive-area common-burst selection."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from shapely import force_2d, from_wkb, get_coordinates, normalize, to_wkb
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import explain_validity
from shapely.errors import ShapelyError

from .products import checksum, identity


def validate_specification(spec):
    if not isinstance(spec, dict) or set(spec) - {'path', 'bbox', 'layer'}:
        raise ValueError('aoi must be a mapping containing path or bbox, and optional layer')
    if (spec.get('path') is not None) == (spec.get('bbox') is not None):
        raise ValueError('aoi requires exactly one non-null path or bbox')
    if spec.get('layer') is not None and (not spec.get('path') or
            not isinstance(spec['layer'], str) or not spec['layer'].strip()):
        raise ValueError('aoi.layer requires a vector path and a nonempty layer name')
    if spec.get('path') is not None:
        if not isinstance(spec['path'], str) or not Path(spec['path']).is_file():
            raise ValueError('aoi.path must be an existing vector file')
    else:
        v = spec['bbox']
        if (not isinstance(v, (list, tuple)) or len(v) != 4 or
                any(type(x) not in (int, float) or not np.isfinite(x) for x in v)):
            raise ValueError('aoi.bbox needs four finite numbers [west, south, east, north]')
        w, s, e, n = v
        if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s < n <= 90):
            raise ValueError('aoi.bbox requires longitude in [-180,180] and -90 <= south < north <= 90')
        if w == e or (w == 180 and e == -180):
            raise ValueError('aoi.bbox must have nonzero width')


def polygon(geom, label):
    if not isinstance(geom, BaseGeometry) or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError(f'{label}: expected nonempty Polygon/MultiPolygon geometry')
    if not np.isfinite(get_coordinates(geom)).all():
        raise ValueError(f'{label}: nonfinite coordinates')
    if not geom.is_valid:
        raise ValueError(f'{label}: invalid geometry ({explain_validity(geom)})')
    return geom


def kml_polygon(geom, label):
    """Read KML polygon-only MultiGeometry in XY; altitude does not define an AOI."""
    if geom is not None and geom.geom_type == 'GeometryCollection' and not geom.is_empty:
        # Validate every part before union: mixed point/line placemarks must not
        # silently contribute only their polygon subset.
        return polygon(unary_union([kml_polygon(part, label) for part in geom.geoms]), label)
    return polygon(None if geom is None else force_2d(geom), label)


def wgs84_parts(geom, crs=4326):
    """Use GDAL's wrapdateline operation, with traditional GIS XY ordering.

    A generous dateline offset also handles polygons whose vertices lie farther
    than GDAL's default ten degrees from the meridian. No validity repair occurs.
    """
    from osgeo import gdal, ogr, osr
    gdal.UseExceptions()
    source = gdal.GetDriverByName('MEM').Create('', 0, 0, 0, gdal.GDT_Unknown)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(CRS.from_user_input(crs).to_wkt())
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    layer = source.CreateLayer('geometry', srs=srs, geom_type=ogr.wkbUnknown)
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
    layer.CreateFeature(feature)
    wrapped = gdal.VectorTranslate('', source, options=gdal.VectorTranslateOptions(
        format='MEM', dstSRS='EPSG:4326', options=['-wrapdateline', '-datelineoffset', '180']))
    if wrapped is None:
        raise ValueError('Cannot normalize geometry at the antimeridian')
    result = unary_union([from_wkb(bytes(f.GetGeometryRef().ExportToWkb()))
                          for f in wrapped.GetLayer(0)])
    polygon(result, 'Normalized WGS84 geometry')
    coords = get_coordinates(result)
    if (np.abs(coords[:, 0]) > 180 + 1e-8).any() or (np.abs(coords[:, 1]) > 90).any():
        raise ValueError('Geometry outside WGS84 longitude/latitude bounds')
    return result


@dataclass(frozen=True)
class AOI:
    geometry: object
    crs: CRS
    report_json: str

    @property
    def report(self):
        return json.loads(self.report_json)


def load_aoi(spec):
    validate_specification(spec)
    sources = {}
    if spec.get('bbox') is not None:
        w, s, e, n = spec['bbox']
        geom = box(w, s, e, n) if w < e else unary_union([
            p for p in (box(w, s, 180, n), box(-180, s, e, n)) if p.area > 0])
        crs = CRS.from_epsg(4326)
    else:
        from osgeo import gdal
        gdal.UseExceptions()
        try:
            ds = gdal.OpenEx(spec['path'], gdal.OF_VECTOR | gdal.OF_READONLY)
            if ds is None:
                raise ValueError('Cannot open vector dataset')
            if ds.GetLayerCount() != 1 and spec.get('layer') is None:
                names = [ds.GetLayer(i).GetName() for i in range(ds.GetLayerCount())]
                raise ValueError(f'Multiple layers {names}; declare aoi.layer explicitly')
            layer = ds.GetLayerByName(spec['layer']) if spec.get('layer') else ds.GetLayer(0)
            if layer is None:
                raise ValueError(f'No layer {spec.get("layer")!r}')
            srs = layer.GetSpatialRef()
            if srs is None:
                raise ValueError('AOI layer must declare a horizontal CRS')
            crs = CRS.from_wkt(srs.ExportToWkt()).to_2d()
            if not (crs.is_projected or crs.is_geographic):
                raise ValueError('AOI layer must declare a horizontal geographic or projected CRS')
            features = []
            is_kml = ds.GetDriver().ShortName in ('KML', 'LIBKML')
            for feature in layer:
                native = feature.GetGeometryRef()
                geom = None if native is None else from_wkb(bytes(native.ExportToWkb()))
                validate = kml_polygon if is_kml else polygon
                features.append(validate(geom, f'AOI feature {feature.GetFID()}'))
            if not features:
                raise ValueError('AOI layer has no features')
            # Normalize geographic features individually before union, so a
            # dateline-crossing polygon cannot swallow unrelated components.
            if crs.is_geographic:
                features = [wgs84_parts(g, crs) for g in features]
                crs = CRS.from_epsg(4326)
            geom = polygon(unary_union(features), 'AOI union')
            files = set(ds.GetFileList() or [spec['path']])
            path = Path(spec['path'])
            if path.suffix.lower() == '.shp':
                files.update(str(p) for p in path.parent.glob(path.stem + '.*') if p.is_file())
            sources = {str(Path(p).resolve()): checksum(p) for p in sorted(files)}
            ds = None
        except (RuntimeError, OSError, ShapelyError) as exc:
            raise ValueError(f'Cannot load AOI {spec["path"]}: {exc}') from exc
    geom = normalize(polygon(geom, 'AOI'))
    geometry_identity = identity({'crs': crs.to_wkt(), 'wkb': to_wkb(geom, hex=True)})
    report = {'specification': spec, 'crs': crs.to_wkt(), 'geometry': mapping(geom),
              'geometry_identity': geometry_identity, 'source_checksums': sources,
              'intersection_units': 'native projected units squared' if crs.is_projected else 'degrees squared'}
    return AOI(geom, crs, json.dumps(report, sort_keys=True))


def footprint(burst):
    label = f'Burst {burst.id} annotation footprint'
    borders = getattr(burst.native, 'border', None)
    if borders is None:
        raise ValueError(f'{label}: missing footprint')
    try:
        borders = list(borders)
    except TypeError as exc:
        raise ValueError(f'{label}: expected a collection of polygon parts') from exc
    if not borders:
        raise ValueError(f'{label}: missing footprint')
    parts = [wgs84_parts(polygon(g, label)) for g in borders]
    return polygon(unary_union(parts), label)


def project_footprint(geom, crs):
    if crs.is_geographic:
        return geom
    # Segmentize in degrees before transforming; projected AOI edges remain
    # untouched. Each split footprint part is retained in the transformation.
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    parts = geom.geoms if geom.geom_type == 'MultiPolygon' else (geom,)
    projected = [polygon(transform(transformer.transform, part.segmentize(0.01)),
                         'Projected footprint part') for part in parts]
    # Split dateline parts can meet along a shared edge in a polar projection;
    # union them before validating the combined polygon topology.
    return polygon(unary_union(projected), 'Projected footprint')


def select_aoi(cfg, common=None):
    from .inputs import discover_validated
    aoi = load_aoi(cfg['aoi'])
    common = discover_validated(cfg) if common is None else common
    selected, decisions, errors = [], [], []
    for pair in sorted(common, key=lambda p: (p[0].swath, p[0].id)):
        try:
            ref, sec = [footprint(b) for b in pair]
            a, b = [project_footprint(g, aoi.crs) for g in (ref, sec)]
            area = aoi.geometry.intersection(a).intersection(b).area
            decisions.append({'burst_id': pair[0].id, 'swath': pair[0].swath,
                              'reference_start': pair[0].start.isoformat(),
                              'secondary_start': pair[1].start.isoformat(),
                              'reference_footprint': mapping(ref), 'secondary_footprint': mapping(sec),
                              'shared_aoi_area': area, 'selected': area > 0})
            if area > 0:
                selected.append(pair)
        except (ValueError, RuntimeError, ShapelyError) as exc:
            errors.append(f'{pair[0].id}: {exc}')
    if errors:
        raise ValueError('Invalid candidate footprints: ' + '; '.join(errors))
    report = {'aoi': aoi.report, 'candidates': decisions, 'common_count': len(common),
              'selected_ids': [p[0].id for p in selected]}
    report['selection_identity'] = identity(report)
    return tuple(selected), report
