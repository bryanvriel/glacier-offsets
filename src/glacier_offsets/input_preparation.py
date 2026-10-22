#!/usr/bin/env python3
"""Download Sentinel-1 TOPS inputs (orbits, optional auxiliaries, and DEM).

Accepts an SLC .SAFE directory, SLC .SAFE.zip, or a directory containing scenes.

Examples
--------
  glacier-offsets prepare-inputs /data/SLCs -o ./inputs --dem-source REMA
  glacier-offsets prepare-inputs /data/SLCs --dem-source REMA --rema-resolution 10
  glacier-offsets prepare-inputs scene.SAFE -o inputs --dem-format ENVI
  glacier-offsets prepare-inputs /data/SLCs --skip-dem --aux cal-ins

REMA data are loaded from the public Polar Geospatial Center mosaic VRT on AWS,
not the OpenTopography globaldem API (which does not serve REMA rasters).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from collections import Counter
from urllib.parse import urljoin
import xml.etree.ElementTree as ET
import zipfile

from .initialization import OrbitCatalog, read_scene, ORBIT_START_MARGIN_SECONDS
from .products import checksum, write_json
from .stack import scan_safes

AUX_API = "https://sar-mpc.eu/api/v1/"
AUX_SITE = "https://sar-mpc.eu/"
REMA_VRT = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/rema/mosaics/v2.0/{resolution}m_dem_tiles.vrt"
VALIDITY_RX = re.compile(r"_V(\d{8}T\d{6})_(\d{8}T\d{6})", re.I)


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def scene_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".zip"):
        name = name[:-4]
    if name.lower().endswith(".safe"):
        name = name[:-5]
    return name


def product_info(safe: Path) -> tuple[str, datetime, datetime]:
    scene = read_scene(safe)
    return scene.mission, scene.start, scene.stop


def zip_problem(path: Path) -> str | None:
    """Cheap ZIP sanity check: only reads archive directory and a small header.

    Does not CRC-test imagery (which can be many gigabytes). Returns a
    diagnostic string if the file is not a readable ZIP container.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fp:
            header = fp.read(128)
        if size == 0:
            return "empty file (0 bytes): download did not complete"
        if zipfile.is_zipfile(path):
            return None
    except OSError as exc:
        return f"cannot read ZIP file: {exc}"

    data = header.lstrip().lower()
    if data.startswith((b"<!doctype html", b"<html", b"<?xml", b"<error", b"{")):
        return (f"not a ZIP ({size:,} bytes); looks like a server error or "
                "authentication response saved as a .zip file")
    if header.startswith(b"PK"):
        return (f"ZIP signature found, but central directory missing or invalid "
                f"({size:,} bytes); likely a truncated or corrupted download")
    if header.startswith(b"\x1f\x8b"):
        return "gzip-compressed file, not a ZIP archive"
    return f"not a recognizable ZIP archive ({size:,} bytes; check download source)"


def find_safes(path: Path, recursive: bool = False) -> tuple[list[Path], list[dict]]:
    """Discover valid SAFE scenes, retaining diagnostics for bad ZIPs.

    A broken ZIP should not prevent processing unrelated SAFE products.
    Extracted SAFE directories take precedence over same-named ZIP files.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        if not path.name.lower().endswith(".zip"):
            raise ValueError(f"Expected a Sentinel-1 SLC .SAFE.zip archive: {path}")
        problem = zip_problem(path)
        if problem:
            raise ValueError(f"Invalid SAFE ZIP: {path}: {problem}")
        return [path.resolve()], []
    if path.name.lower().endswith(".safe"):
        product_info(path)
        return [path.resolve()], []

    candidates = scan_safes(path, recursive=recursive)
    # Prefer extracted SAFE directories to duplicate ZIPs with same scene name.
    selected = {}
    invalid: list[dict] = []
    for p in sorted(candidates, key=lambda p: (scene_stem(p), p.is_file(), str(p))):
        key = (p.parent, scene_stem(p))
        if key in selected:
            continue
        if p.is_file():
            problem = zip_problem(p)
            if problem:
                invalid.append({"path": str(p.resolve()), "reason": problem})
                continue
        selected[key] = p.resolve()
    return sorted(selected.values()), invalid


def safe_xml(safe: Path, inner: str):
    """Read a small metadata XML from a directory or ZIP (no extraction)."""
    if safe.is_dir():
        with (safe / inner).open("rb") as f:
            return ET.parse(f).getroot()
    with zipfile.ZipFile(safe) as z:
        matches = [n for n in z.namelist() if n.endswith("/" + inner)]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected exactly one {inner} in {safe}")
        with z.open(matches[0]) as f:
            return ET.parse(f).getroot()


def annotation_paths(safe: Path) -> list[str]:
    if safe.is_dir():
        paths = [p.relative_to(safe).as_posix() for p in (safe / "annotation").glob("*.xml")]
    else:
        with zipfile.ZipFile(safe) as z:
            paths = ["/".join(n.split("/")[-2:]) for n in z.namelist()
                     if n.count("/annotation/") == 1
                     and n.split("/")[-2] == "annotation"
                     and n.lower().endswith(".xml")]
    return sorted(set(p for p in paths if "-slc-" in p.lower()))


def read_ipf_version(safe: Path) -> str | None:
    root = safe_xml(safe, "manifest.safe")
    for node in root.iter():
        if local_name(node.tag) == "software" and "ipf" in node.attrib.get("name", "").lower():
            return node.attrib.get("version")
    return None


def minimum_lon_arc(lons: list[float]) -> tuple[float, float]:
    """Smallest covering longitude arc, with east permitted to exceed 180."""
    if not lons:
        raise ValueError("No valid longitudes")
    xs = sorted(x % 360 for x in lons)
    gaps = [(xs[(i + 1) % len(xs)] + (360 if i == len(xs)-1 else 0) - x, i)
            for i, x in enumerate(xs)]
    _, i = max(gaps)
    west, east = xs[(i + 1) % len(xs)], xs[i]
    if east < west:
        east += 360
    if west > 180:
        west, east = west-360, east-360
    return west, east


def footprint(safe: Path) -> tuple[list[float], int | None]:
    """Read annotation geolocation grid; return (W,S,E,N) and instrument ID."""
    coords: list[tuple[float, float]] = []
    icid: int | None = None
    paths = annotation_paths(safe)
    if not paths:
        raise ValueError(f"No SLC annotation XMLs in {safe}")
    for path in paths:
        root = safe_xml(safe, path)
        for elem in root.iter():
            tag = local_name(elem.tag)
            if tag == "instrumentconfigurationid" and icid is None:
                try:
                    icid = int((elem.text or "").strip())
                except ValueError:
                    pass
            if tag == "geolocationgridpoint":
                fields = {local_name(child.tag): child.text for child in elem}
                try:
                    lat, lon = float(fields["latitude"]), float(fields["longitude"])
                except (KeyError, TypeError, ValueError):
                    continue
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    coords.append((lon, lat))
    if not coords:
        raise ValueError(f"No annotation geolocation points in {safe}")
    west, east = minimum_lon_arc([x for x, _ in coords])
    return [west, min(y for _, y in coords), east, max(y for _, y in coords)], icid


def union_bounds(boxes: list[list[float]]) -> list[float]:
    if not boxes:
        raise ValueError("No footprints to combine")
    west, east = minimum_lon_arc([lon for box in boxes for lon in (box[0], box[2])])
    if east - west > 180:
        raise ValueError("Scenes are geographically scattered (>180 deg). Use --bbox or --skip-dem")
    return [west, min(b[1] for b in boxes), east, max(b[3] for b in boxes)]


def pad_bounds(bounds: list[float], km: float) -> list[float]:
    west, south, east, north = bounds
    if (not all(math.isfinite(x) for x in (*bounds, km)) or km < 0 or
            west >= east or not -90 <= south < north <= 90 or east-west > 180):
        raise ValueError("Invalid DEM bounds or negative padding")
    dy = km / 111.2
    dx = km / (111.2 * max(0.02, abs(math.cos(math.radians((south + north) / 2)))))
    result = [west-dx, max(-90, south-dy), east+dx, min(90, north+dy)]
    if result[2]-result[0] > 180:
        raise ValueError("Requested DEM longitude extent exceeds 180 degrees; use --bbox")
    return result


def cached_orbit(folder: Path, scene) -> Path | None:
    """Use precisely the same XML, state-vector and time checks as processing."""
    try:
        return OrbitCatalog(folder).resolve(scene).path
    except ValueError:
        return None


def get_orbit(safe: Path, folder: Path, use_cdse: bool, refresh: bool) -> str:
    scene = read_scene(safe)
    folder.mkdir(parents=True, exist_ok=True)
    old = cached_orbit(folder, scene)
    if old is not None and not refresh:
        print(f"    Reusing orbit: {old.name}")
        return str(old)
    try:
        from eof.download import download_eofs
    except ImportError as exc:
        raise ImportError('Orbit downloads require the preparation extra: '
                          'python -m pip install -e ".[preparation]"') from exc
    # Validate in a temporary directory before replacing any cached orbit.
    with tempfile.TemporaryDirectory(dir=folder, prefix='.orbit-') as scratch:
        options = {'orbit_dts':[scene.start,scene.stop],'missions':[scene.mission]*2,
                   'save_dir':scratch,'orbit_type':'precise','force_asf':not use_cdse}
        if 's1reader_compat' in inspect.signature(download_eofs).parameters:
            options['s1reader_compat'] = True
        else:
            # Older sentineleof queries have fixed 60-second margins. Request
            # the earlier endpoint too, then validate one file covers the whole
            # interval. Never concatenate two separately covering orbit files.
            options['orbit_dts'][0] -= timedelta(seconds=ORBIT_START_MARGIN_SECONDS-60)
        download_eofs(**options)
        downloaded = OrbitCatalog(scratch)
        downloaded.resolve(scene)  # Reject RESORB fallback or insufficient coverage.
        for orbit in downloaded.orbits:
            if orbit.mission == scene.mission:
                orbit.path.replace(folder/orbit.path.name)
    found = OrbitCatalog(folder).resolve(scene).path
    print(f"    Orbit: {found.name}")
    return str(found)


def aux_selection(ipf: str | None, mode: str) -> list[str]:
    if mode == "auto":
        return ["AUX_CAL"] if ipf == "002.36" else []
    return {"none": [], "cal": ["AUX_CAL"],
            "cal-ins": ["AUX_CAL", "AUX_INS"]}[mode]


def validity_from_name(name: str) -> tuple[datetime, datetime] | None:
    match = VALIDITY_RX.search(name)
    if not match:
        return None
    return tuple(datetime.strptime(x, "%Y%m%dT%H%M%S") for x in match.groups())


def download_aux(product: str, mission: str, acquired: datetime, icid: int | None,
                 folder: Path) -> Path:
    import requests

    when = acquired.strftime("%Y-%m-%dT%H:%M:%S")
    params = {"product_type": product, "sentinel1__mission": mission,
              "validity_start__lte": when, "adf__stop_time__gt": when,
              "mode": "extended", "page_size": 100, "ordering": "-validity_start"}
    if icid is not None:
        params["sentinel1__instr_conf_id"] = icid
    resp = requests.get(AUX_API, params=params, headers={"Accept": "application/json"}, timeout=40)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("results", []) if isinstance(payload, dict) else payload
    if not records:
        raise RuntimeError(f"No {product} for {mission} at {when}, ICID={icid}")
    records = sorted(records, key=lambda r: (r.get("validity_start") or "",
                                             r.get("creation_date") or ""), reverse=True)
    if icid is None and len(records) > 1:
        raise RuntimeError(f"Ambiguous {product} without instrument configuration ID: {mission} {when}")
    record = records[0]
    name = record.get("product_name") or record.get("physical_name")
    remote = record.get("remote_url")
    if not name or not remote:
        raise RuntimeError(f"Unexpected {product} metadata, missing product name or URL")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name.removesuffix(".zip")
    if target.is_dir() and any(target.rglob("*.xml")):
        print(f"    Reusing {product}: {target.name}")
        return target
    print(f"    Downloading {product}: {name}")
    with requests.get(urljoin(AUX_SITE, remote), stream=True, timeout=(15, 180)) as stream:
        stream.raise_for_status()
        with tempfile.TemporaryFile() as tmp:
            shutil.copyfileobj(stream.raw, tmp)
            tmp.seek(0)
            with zipfile.ZipFile(tmp) as archive:
                for member in archive.infolist():
                    path = Path(member.filename)
                    if path.is_absolute() or any(x in ("..", "") for x in path.parts):
                        raise RuntimeError("Unsafe auxiliary ZIP path")
                archive.extractall(folder)
    # SAR-MPC archive products contain a .SAFE folder.
    candidates = [p for p in folder.glob(f"{mission}_AUX_{product.split('_')[-1]}_*.SAFE")
                  if p.is_dir() and any(p.rglob("*.xml"))]
    if target in candidates:
        return target
    if len(candidates) == 1:
        return candidates[0]
    raise RuntimeError(f"Could not identify extracted {product} in {folder}")


def get_aux(product: str, mission: str, acquired: datetime, icid: int | None,
            folder: Path, memo: list[dict]) -> str:
    # Batch reuse is safe only if mission, product, ICID, AND validity agree.
    for record in memo:
        if (record["product"] == product and record["mission"] == mission
                and record["icid"] == icid and
                record["start"] <= acquired < record["stop"]):
            print(f"    Reusing {product}: {Path(record['path']).name}")
            return record["path"]
    path = download_aux(product, mission, acquired, icid, folder)
    dates = validity_from_name(path.name)
    if dates is not None and dates[0] <= acquired < dates[1]:
        memo.append({"product": product, "mission": mission, "icid": icid,
                     "start": dates[0], "stop": dates[1], "path": str(path)})
    return str(path)


def rema_geometry(bounds: list[float]):
    """Projected bounds of sampled WGS84 bbox edges, using native EPSG:3031."""
    from osgeo import osr

    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst.ImportFromEPSG(3031)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transformer = osr.CoordinateTransformation(src, dst)
    w, s, e, n = bounds
    points = []
    for i in range(65):
        t = i / 64
        points.extend([(w+(e-w)*t, s), (w+(e-w)*t, n),
                       (w, s+(n-s)*t), (e, s+(n-s)*t)])
    xy = [transformer.TransformPoint(lon, lat) for lon, lat in points]
    return [min(p[0] for p in xy), min(p[1] for p in xy),
            max(p[0] for p in xy), max(p[1] for p in xy)]


def estimate_pixels(box: list[float], dx: float, dy: float) -> int:
    return math.ceil((box[2]-box[0])/dx) * math.ceil((box[3]-box[1])/dy)


def generate_rema(output: Path, bounds: list[float], resolution: int,
                  fmt: str, max_pixels: int) -> int:
    from osgeo import gdal

    if bounds[3] > -55:
        raise ValueError("REMA only covers Antarctica; use COP or NISAR outside Antarctica")
    gdal.UseExceptions()
    vrt = "/vsicurl/" + REMA_VRT.format(resolution=resolution)
    # Polar GeoTIFF stays in native meters. ENVI is reprojected to geodetic
    # lat/lon, as expected by ISCE2 DEM readers.
    if fmt == "GTiff":
        box = rema_geometry(bounds)
        res_x = res_y = float(resolution)
        out_srs = "EPSG:3031"
    else:
        w, s, e, n = bounds
        if w < -180 or e > 180:
            raise ValueError("REMA ENVI across the date line is not supported; "
                             "use GTiff (EPSG:3031), or use an AOI on one side")
        box = bounds
        res_y = resolution / 111320.0
        res_x = resolution / (111320.0 * max(.03, abs(math.cos(math.radians((s+n)/2)))))
        out_srs = "EPSG:4326"
    pixels = estimate_pixels(box, res_x, res_y)
    if pixels > max_pixels:
        raise ValueError(f"REMA output would contain ~{pixels:,} pixels. "
                         "Use --bbox, 32m resolution, or raise --max-dem-pixels deliberately")
    print(f"  REMA {resolution} m: output {out_srs}, ~{pixels:,} pixels")
    print(f"  Source: {REMA_VRT.format(resolution=resolution)}")
    creation = ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"] if fmt == "GTiff" else []
    # The VRT references public /vsis3/ tiles. Warp's I/O workers must inherit
    # unsigned access and retries; thread-local options only reach the caller.
    with gdal.config_options({'AWS_NO_SIGN_REQUEST':'YES','GDAL_HTTP_MAX_RETRY':'3','GDAL_HTTP_RETRY_DELAY':'2'},
                             thread_local=False):
        result = gdal.Warp(str(output), vrt, options=gdal.WarpOptions(
            format=fmt, outputBounds=box, outputBoundsSRS=out_srs, dstSRS=out_srs,
            xRes=res_x, yRes=res_y, outputType=gdal.GDT_Float32, targetAlignedPixels=True,
            resampleAlg="bilinear", dstNodata=-9999, creationOptions=creation, multithread=True))
        if result is None:
            raise RuntimeError("GDAL Warp returned no dataset for REMA")
        result.FlushCache()
        result = None  # Close while the workers still have unsigned access.
    return pixels


def dem_filename(folder: Path, bounds: list[float], args: argparse.Namespace) -> Path:
    key = json.dumps({"bounds": [round(float(x), 9) for x in bounds], "source": args.dem_source,
                      "resolution": args.rema_resolution if args.dem_source == "REMA" else None,
                      "format": args.dem_format, "padding": args.padding_km}, sort_keys=True)
    digest = hashlib.sha256(key.encode()).hexdigest()[:12]
    ext = ".tif" if args.dem_format == "GTiff" else ".dem.wgs84"
    return folder / f"elevation_{args.dem_source}_{digest}{ext}"


def validate_dem(path, bounds, min_valid_fraction=.01, max_pixels=200_000_000):
    """Check actual raster coverage and elevations with bounded block reads.

    The fraction is over the output raster, not an assertion that every glacier
    pixel has an elevation. Ocean/void nodata is retained without filling.
    """
    from osgeo import gdal, osr
    import numpy as np
    from .raster_io import bounded_raster_cache

    gdal.UseExceptions()
    ds = gdal.Open(str(path))
    if ds is None or ds.RasterCount != 1 or not ds.GetProjection():
        raise ValueError('DEM must be a single-band georeferenced raster')
    band = ds.GetRasterBand(1)
    if gdal.DataTypeIsComplex(band.DataType):
        raise ValueError('DEM elevations must be real-valued')
    width, height = ds.RasterXSize, ds.RasterYSize
    if width*height == 0 or width*height > max_pixels:
        raise ValueError(f'DEM has {width*height:,} pixels; limit is {max_pixels:,}')
    gt = ds.GetGeoTransform(can_return_null=True)
    inv = gdal.InvGeoTransform(gt) if gt is not None and all(math.isfinite(v) for v in gt) else None
    if inv is None:
        raise ValueError('DEM has no usable geotransform')
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(4326); dst.ImportFromWkt(ds.GetProjection())
    for crs in (src, dst):
        crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    w,s,e,n = bounds
    center = gt[0]+width*gt[1]/2
    for i in range(65):
        t = i/64
        for lon,lat in ((w+(e-w)*t,s),(w+(e-w)*t,n),(w,s+(n-s)*t),(e,s+(n-s)*t)):
            x,y,_ = transform.TransformPoint(lon,lat)
            if dst.IsGeographic():
                x = center+(x-center+180)%360-180
            col,row = gdal.ApplyGeoTransform(inv,x,y)
            if not (math.isfinite(col) and math.isfinite(row) and
                    -1e-5 <= col <= width+1e-5 and -1e-5 <= row <= height+1e-5):
                raise ValueError('DEM extent does not cover requested buffered bounds')
    unit = band.GetUnitType().lower()
    if unit and unit not in ('m','metre','meter','metres','meters'):
        raise ValueError(f'DEM elevations must be in metres; got {unit}')
    valid_count = 0
    minimum, maximum = math.inf, -math.inf
    with bounded_raster_cache():
        for row in range(0,height,256):
            for col in range(0,width,1024):
                nx,ny = min(1024,width-col),min(256,height-row)
                data = band.ReadAsArray(col,row,nx,ny)
                good = np.isfinite(data)
                if band.GetMaskFlags() != gdal.GMF_ALL_VALID:
                    good &= band.GetMaskBand().ReadAsArray(col,row,nx,ny)!=0
                if good.any():
                    valid_count += int(good.sum())
                    minimum = min(minimum,float(data[good].min()))
                    maximum = max(maximum,float(data[good].max()))
    fraction = valid_count/(width*height)
    if not valid_count or fraction < min_valid_fraction:
        raise ValueError(f'DEM valid fraction {fraction:.3%} is below required {min_valid_fraction:.3%}; '
                         'check source coverage and nodata')
    return {'shape':[height,width],'crs':ds.GetProjection(),'valid_pixels':valid_count,
            'valid_fraction':fraction,'min_elevation_m':minimum,'max_elevation_m':maximum,
            'bounds_wsen':bounds,'height_reference':'WGS84 ellipsoid',
            'height_reference_basis':'REMA native heights or sardem keep_egm=False; not inferred from pixel values'}


def get_dem(bounds: list[float], folder: Path, args: argparse.Namespace) -> str:
    bbox = pad_bounds(bounds, args.padding_km)
    args = argparse.Namespace(**vars(args))
    if args.dem_source == 'auto':
        args.dem_source = 'REMA' if bbox[3] <= -55 else 'COP'
    folder.mkdir(parents=True, exist_ok=True)
    print(f"  DEM W,S,E,N: {[round(x, 5) for x in bbox]}")
    output = dem_filename(folder, bbox, args)
    if output.exists() and output.stat().st_size > 0 and not args.overwrite_dem:
        try:
            report = validate_dem(output,bbox,args.min_dem_valid_fraction,args.max_dem_pixels)
            if args.isce2_xml and not Path(str(output)+'.xml').is_file():
                raise ValueError('ENVI DEM lacks its ISCE XML companion')
        except (ValueError,RuntimeError,OSError) as exc:
            print(f'  Cached DEM is unusable; rebuilding: {exc}')
        else:
            report.update(source=args.dem_source,rema_resolution=args.rema_resolution if args.dem_source=='REMA' else None)
            write_json(str(output)+'.preparation.json',report)
            print(f"  Reusing validated DEM: {output}")
            return str(output)
    xml_writer = None
    if args.isce2_xml:
        try:
            import isce
            import isceobj
            from sardem.utils import gdal2isce_xml
            xml_writer = gdal2isce_xml
        except ImportError as exc:
            raise ImportError('--isce2-xml requires ISCE2 and sardem; omit it for the ISCE3 workflow') from exc
    with tempfile.TemporaryDirectory(dir=folder,prefix='.dem-') as scratch:
        staged = Path(scratch)/output.name
        if args.dem_source == "REMA":
            generate_rema(staged,bbox,args.rema_resolution,args.dem_format,args.max_dem_pixels)
        else:
            if bbox[2]-bbox[0] > 90:
                raise ValueError("DEM geographic width is >90 degrees. Use --bbox to avoid a huge raster")
            try:
                from sardem.dem import main as sardem_main
            except ImportError as exc:
                raise ImportError('DEM downloads require glacier-offsets[preparation]') from exc
            sardem_main(output_name=str(staged),bbox=bbox,data_source=args.dem_source,
                        output_format=args.dem_format,output_type="float32",make_isce_xml=False,keep_egm=False)
        report = validate_dem(staged,bbox,args.min_dem_valid_fraction,args.max_dem_pixels)
        from osgeo import gdal
        ds = gdal.Open(str(staged))
        files = [Path(p) for p in ds.GetFileList()]; ds = None
        if xml_writer:
            xml_writer(str(staged),keep_egm=False)
            xml_path = Path(str(staged)+'.xml')
            tree = ET.parse(xml_path)
            for element in tree.iter():
                if element.text:
                    element.text = element.text.replace(str(staged),str(output))
            tree.write(xml_path,encoding='utf-8',xml_declaration=True)
            files.append(xml_path)
        for p in sorted(files,key=lambda p:p==staged):
            p.replace(folder/p.name)  # Promote the binary last, after companions.
    report.update(source=args.dem_source,rema_resolution=args.rema_resolution if args.dem_source=='REMA' else None)
    write_json(str(output)+'.preparation.json',report)
    print(f"  DEM created: {output}")
    return str(output)


def prepare(args: argparse.Namespace) -> int:
    input_path = args.safe.expanduser().resolve()
    safes, invalid_zips = find_safes(input_path, args.recursive)
    for item in invalid_zips:
        print(f"  WARNING: Skipping invalid SAFE ZIP: {item['path']}\n"
              f"           Reason: {item['reason']}", file=sys.stderr)
    if not safes:
        if invalid_zips:
            raise ValueError(f"No usable SAFE products; {len(invalid_zips)} invalid ZIP(s). "
                             "Check warnings above and re-download those scenes.")
        raise ValueError("No Sentinel-1 SLC .SAFE directories or .SAFE.zip archives found")
    root = args.output.expanduser().resolve()
    orbit_dir, aux_dir, dem_dir = [root / n for n in ("orbits", "aux", "dem")]
    for folder in (root, orbit_dir, aux_dir, dem_dir):
        folder.mkdir(parents=True, exist_ok=True)
    # An interrupted refresh must not leave an earlier success manifest current.
    write_json(root/'manifest.json',{'method':'s1-input-preparation-v1','status':'running',
                                    'input':str(input_path),'scenes':[]})
    print(f"Found {len(safes)} usable SAFE product(s) in {args.safe}; "
          f"skipped {len(invalid_zips)} invalid ZIP(s)")
    counts = Counter(scene_stem(p) for p in safes)
    reports = {}
    for p in safes:
        stem = scene_stem(p)
        suffix = '_'+hashlib.sha256(str(p).encode()).hexdigest()[:8] if counts[stem]>1 else ''
        reports[p] = root/f'inputs_{stem}{suffix}.json'
    scenes: list[dict] = []
    aux_memo: list[dict] = []
    failures = len(invalid_zips)
    for idx, safe in enumerate(safes, 1):
        print(f"\n[{idx}/{len(safes)}] {safe.name}")
        info = {"safe": str(safe), "status": "ok", "orbit": None,
                "aux": {}, "dem": None, "footprint_wsen": None,'report':str(reports[safe])}
        scenes.append(info)
        try:
            mission, start, stop = product_info(safe)
            info.update(mission=mission, start=start.isoformat(), stop=stop.isoformat())
            ipf = read_ipf_version(safe)
            info["ipf_version"] = ipf
            try:
                box, icid = footprint(safe)
                info["footprint_wsen"] = box
            except (ValueError, FileNotFoundError) as exc:
                if args.bbox is None and not args.skip_dem:
                    raise
                print(f"  Footprint unavailable (using --bbox or skipping DEM): {exc}")
                icid = None
            info["instrument_configuration_id"] = icid
            print(f"  {mission} {start.isoformat()} IPF={ipf or '?'} ICID={icid}")
            if not args.skip_orbit:
                info["orbit"] = get_orbit(safe,orbit_dir,args.use_cdse,args.refresh_orbits)
                info['orbit_sha256'] = checksum(info['orbit'])
            kinds = aux_selection(ipf, args.aux)
            if args.aux == "auto" and ipf is None:
                print("  NOTE: Unknown IPF; automatic AUX_CAL selection skipped")
            for product in kinds:
                info["aux"][product] = get_aux(product, mission, start, icid, aux_dir, aux_memo)
        except Exception as exc:
            info["status"] = "error"
            info["error"] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr)
            failures += 1
            if args.fail_fast:
                break
        finally:
            write_json(reports[safe],info)

    dems: list[dict] = []
    if not args.skip_dem and not (args.fail_fast and failures):
        try:
            if args.dem_mode == "combined":
                boxes = [list(args.bbox)] if args.bbox else [s["footprint_wsen"] for s in scenes
                                                          if s['status']=='ok' and s["footprint_wsen"]]
                if not boxes:
                    raise ValueError("No usable SAFE footprints; supply --bbox")
                common = union_bounds(boxes)
                dem = get_dem(common, dem_dir, args)
                dems.append({"bounds_wsen":pad_bounds(common,args.padding_km),"path":dem,
                             'validation':json.loads(Path(dem+'.preparation.json').read_text())})
                for scene in scenes:
                    if scene['status']=='ok':
                        scene["dem"] = dem
            else:
                found: dict[str, str] = {}
                for scene in scenes:
                    if scene['status']!='ok':
                        continue
                    try:
                        box = list(args.bbox) if args.bbox else scene["footprint_wsen"]
                        if box is None:
                            raise ValueError("No footprint for DEM; supply --bbox")
                        signature = json.dumps(box)
                        if signature not in found:
                            found[signature] = get_dem(box, dem_dir, args)
                            dems.append({"bounds_wsen":pad_bounds(box,args.padding_km),"path":found[signature],
                                         'validation':json.loads(Path(found[signature]+'.preparation.json').read_text())})
                        scene["dem"] = found[signature]
                    except Exception as exc:
                        failures += 1
                        scene["dem_error"] = str(exc)
                        scene['status'] = 'error'
                        print(f"  DEM ERROR ({Path(scene['safe']).name}): {exc}", file=sys.stderr)
                        if args.fail_fast:
                            break
        except Exception as exc:
            failures += 1
            print(f"  DEM ERROR: {exc}", file=sys.stderr)
            dems.append({"error": str(exc)})
            for scene in scenes:
                if scene['status']=='ok':
                    scene.update(status='error',dem_error=str(exc))
    for scene in scenes:
        write_json(scene['report'],scene)
    manifest = {"method":"s1-input-preparation-v1",'status':'failed' if failures else 'complete',
                "input":str(input_path),"count":len(scenes),
                'safe_directory':str(input_path) if input_path.is_dir() and not input_path.name.lower().endswith('.safe') else None,
                'orbit_directory':str(orbit_dir),
                'dem':dems[0]['path'] if args.dem_mode=='combined' and dems and 'path' in dems[0] else None,
                "invalid_safe_zips": invalid_zips,
                "failed_steps": failures, "dem_source": args.dem_source,
                "dem_mode": args.dem_mode, "dems": dems,
                "dem_height_reference": "WGS84 ellipsoid", "scenes": scenes}
    manifest_path = root / "manifest.json"
    write_json(manifest_path,manifest)
    print(f"\nManifest: {manifest_path}\nPrepared {len(scenes)} products, failed steps: {failures}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog='glacier-offsets prepare-inputs',description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("safe", type=Path, help="SLC SAFE/.SAFE.zip OR directory containing SLC SAFE products")
    parser.add_argument("-o", "--output", type=Path, default=Path("s1_inputs"))
    parser.add_argument("--recursive",action=argparse.BooleanOptionalAction,default=True,
                        help="Search nested visible directories (default, matching stack discovery)")
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                        help="DEM AOI in W,S,E,N order (not ISCE2 SNWE); east may exceed 180")
    parser.add_argument("--padding-km", type=float, default=5.0)
    parser.add_argument("--dem-source",choices=("auto","COP","NISAR","REMA"),default="auto",
                        help='auto selects REMA south of 55S, otherwise COP')
    parser.add_argument("--rema-resolution", type=int, choices=(10, 32), default=32,
                        help="REMA mosaic resolution in meters (default 32)")
    parser.add_argument("--dem-mode", choices=("combined", "per-safe"), default="combined",
                        help="One shared DEM for all scenes (default) or separate DEM for each")
    parser.add_argument("--dem-format", choices=("GTiff", "ENVI"), default="GTiff",
                        help="REMA GeoTIFF is native EPSG:3031; ENVI is EPSG:4326 for ISCE2")
    parser.add_argument('--isce2-xml',action='store_true',help='Optional ENVI ISCE2 XML companion; requires ISCE2')
    parser.add_argument("--max-dem-pixels", type=int, default=200_000_000,
                        help="Maximum validated DEM size; also checked before REMA download")
    parser.add_argument('--min-dem-valid-fraction',type=float,default=.01,
                        help='Minimum finite, unmasked elevation fraction (default 0.01); nodata is not filled')
    parser.add_argument("--aux", choices=("auto", "none", "cal", "cal-ins"), default="none",
                        help="Auto downloads AUX_CAL only for historical IPF 002.36 scenes")
    parser.add_argument("--use-cdse", action="store_true", help="Use Copernicus Data Space for orbit downloads")
    parser.add_argument("--refresh-orbits", action="store_true", help="Query/download precise orbits even with a valid cache")
    parser.add_argument("--skip-orbit", action="store_true")
    parser.add_argument("--skip-dem", action="store_true")
    parser.add_argument("--overwrite-dem", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop scene processing on first error")
    args = parser.parse_args(argv)
    if args.isce2_xml and args.dem_format!='ENVI':
        parser.error('--isce2-xml requires --dem-format ENVI')
    if (not math.isfinite(args.min_dem_valid_fraction) or not 0 < args.min_dem_valid_fraction <= 1
            or args.max_dem_pixels < 1):
        parser.error('DEM valid fraction must be in (0,1] and max-dem-pixels must be positive')
    if not math.isfinite(args.padding_km) or args.padding_km < 0:
        parser.error('padding-km must be finite and nonnegative')
    if args.bbox:
        try:
            pad_bounds(args.bbox,args.padding_km)
        except ValueError as exc:
            parser.error(str(exc))
    try:
        return prepare(args)
    except KeyboardInterrupt:
        print('Interrupted; validated cached inputs are preserved',file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
