"""Narrow metadata repairs for newly reconstructed, unpublished burst SAFEs."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import re


def repair_annotation_widths(safe, group, scratch):
    """Repair burst2safe's inherited width only with source and TIFF evidence.

    Mixed source-slice widths can leave numberOfSamples one sample narrower
    than both samplesPerBurst and the padded TIFF. Never alter image samples,
    height, burst timing, or valid-sample arrays. Full SAFE validation follows.
    """
    from lxml import etree as ET
    from osgeo import gdal
    from burst2safe.utils import get_subxml_from_metadata

    safe, scratch = Path(safe), Path(scratch).resolve()
    pending = []
    for annotation in sorted((safe / 'annotation').glob('*.xml')):
        if annotation.name.startswith('._'):
            continue
        root = ET.parse(str(annotation)).getroot()
        bursts = root.find('swathTiming/burstList')
        if bursts is None or not len(bursts):
            continue
        width = int(root.findtext('imageAnnotation/imageInformation/numberOfSamples'))
        samples = int(root.findtext('swathTiming/samplesPerBurst'))
        if width == samples:
            continue
        swath, pol = root.findtext('adsHeader/swath'), root.findtext('adsHeader/polarisation')
        height = int(root.findtext('imageAnnotation/imageInformation/numberOfLines'))
        lines = int(root.findtext('swathTiming/linesPerBurst'))

        def reject():
            raise ValueError(f'Unrecognized annotation width mismatch (not repaired): {annotation.name}')

        rows = sorted((r for r in group['bursts'] if r['swath'] == swath), key=lambda r: r['start'])
        if (width <= 0 or samples != width + 1 or lines <= 0 or height != lines * len(bursts)
                or int(bursts.get('count', '-1')) != len(bursts) or len(rows) != len(bursts)
                or any(r['polarization'] != pol for r in rows)):
            reject()
        parents = list(dict.fromkeys(r['parent'] for r in rows))
        if len(parents) < 2:
            reject()
        source_widths = {}
        for parent in parents:
            metadata = (scratch / f'{parent}_{pol}.xml').resolve()
            if not metadata.is_relative_to(scratch) or metadata.is_relative_to(safe.resolve()):
                reject()
            source = get_subxml_from_metadata(metadata, 'product', swath, pol)
            if source is None:
                reject()
            source_width = int(source.findtext('swathTiming/samplesPerBurst'))
            if (source_width != int(source.findtext('imageAnnotation/imageInformation/numberOfSamples'))
                    or lines != int(source.findtext('swathTiming/linesPerBurst'))):
                reject()
            source_widths[parent] = source_width
        if set(source_widths.values()) != {width, samples} or source_widths[parents[0]] != width:
            reject()
        matches = list((safe / 'measurement').glob(annotation.stem + '.tif*'))
        if len(matches) != 1:
            reject()
        ds = gdal.Open(str(matches[0]), gdal.GA_ReadOnly)
        if (ds is None or ds.RasterCount != 1 or (ds.RasterXSize, ds.RasterYSize) != (samples, height)
                or not gdal.DataTypeIsComplex(ds.GetRasterBand(1).DataType)):
            reject()
        ds = None
        # Preserve the rest of the annotation byte-for-byte, including prefixes.
        before = annotation.read_bytes()
        after, count = re.subn(
            rb'(<(?:[\w.-]+:)?numberOfSamples\b[^>]*>\s*)[0-9]+(\s*</(?:[\w.-]+:)?numberOfSamples>)',
            lambda m: m[1] + str(samples).encode('ascii') + m[2], before)
        if count != 1:
            reject()
        pending.append((annotation, before, after, {
            'method': 'burst2safe-inherited-width-v1',
            'annotation': annotation.relative_to(safe).as_posix(),
            'old_number_of_samples': width, 'number_of_samples': samples,
            'source_widths': source_widths}))
    if not pending:
        return []

    # Validate every original manifest reference before modifying any files.
    manifest_path = safe / 'manifest.safe'
    manifest = ET.parse(str(manifest_path))
    for annotation, before, after, _ in pending:
        entries = [node for node in manifest.xpath('//*[local-name()="byteStream"]')
                   if any((safe / location.get('href', '')).resolve() == annotation.resolve()
                          for location in node.xpath('./*[local-name()="fileLocation"]'))]
        if len(entries) != 1:
            raise ValueError(f'Missing/ambiguous SAFE manifest entry: {annotation.name}')
        stream = entries[0]
        digests = stream.xpath('./*[local-name()="checksum"]')
        if (stream.get('size') != str(len(before)) or len(digests) != 1
                or digests[0].get('checksumName', '').upper() != 'MD5'
                or (digests[0].text or '').lower() != hashlib.md5(before).hexdigest()):
            raise ValueError(f'Invalid original SAFE manifest checksum/size: {annotation.name}')
        stream.set('size', str(len(after)))
        digests[0].text = hashlib.md5(after).hexdigest()
    for annotation, _, after, repair in pending:
        annotation.write_bytes(after)
        logging.warning('Repaired inherited annotation width %s: %s -> %s', annotation.name,
                        repair['old_number_of_samples'], repair['number_of_samples'])
    manifest.write(str(manifest_path), encoding='UTF-8', xml_declaration=True)
    return [repair for _, _, _, repair in pending]
