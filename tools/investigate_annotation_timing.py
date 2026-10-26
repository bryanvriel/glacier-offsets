"""Read-only timing census of the frozen Larsen experiment's SAFE annotations.

python -m tools.investigate_annotation_timing --stack STACK --experiment EXPERIMENT

No image matching, timing calibration fitting, or production correction is done.
Metadata checksums and per-source timestamps are retained for reproduction.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np


CALIBRATION_URL = (
    'https://sentiwiki.copernicus.eu/__attachments/'
    'a_94c458a0d37f5d99e3df2448a4f893e788b3c64a242f16e7a8c28c1049a48400/'
    'DI-MPC-SETAP%20-%20Update%20of%20S-1%20Instrument%20Timing%20Calibration'
    '%20for%20ETAD%202023%20-%201.2.pdf?cb=bcdd82d5d96a49345142271dc461f983')
SPECIFICATION_URL = (
    'https://sentinels.copernicus.eu/documents/247904/1877131/'
    'Sentinel-1-Product-Specification.pdf/'
    '49c514c3-1574-4d94-aae2-d8061a3baebd?t=1584020315000')
ALGORITHM_URL = (
    'https://sentiwiki.copernicus.eu/__attachments/'
    'a_756fb6a7b0748da60a84b616db0d25be9f6a5115eeb185b38bfd75c6382d562e/'
    'DI-MPC-IPFDPM%20-%20Sentinel-1%20Level%201%20Detailed%20Algorithm%20Definition'
    '%202022%20-%202.5.pdf?cb=68b1b7de2b2941fa5cfc5745b54238dd')
IMAGE_FIELDS = (
    'productFirstLineUtcTime', 'productLastLineUtcTime', 'ascendingNodeTime',
    'anchorTime', 'productComposition', 'slantRangeTime', 'rangePixelSpacing',
    'azimuthPixelSpacing', 'azimuthTimeInterval', 'azimuthFrequency',
    'zeroDopMinusAcqTime', 'numberOfSamples', 'numberOfLines')
DOWNLINK_FIELDS = ('pri', 'rank', 'instrumentConfigId', 'txPulseLength',
                   'txPulseStartFrequency', 'txPulseRampRate')


def phase(seconds, interval):
    """Centered remainder; a grid-placement diagnostic, NOT a calibration."""
    return seconds - round(seconds / interval) * interval


def seconds(a, b):
    return (datetime.fromisoformat(a) - datetime.fromisoformat(b)).total_seconds()


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path):
        path = Path(path).resolve()
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
        stamp = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
        if stamp(before) != stamp(after):
            raise ValueError(f'Input changed during read: {path}')
        record = {'sha256': hashlib.sha256(data).hexdigest(), 'size': after.st_size,
                  'mtime_ns': after.st_mtime_ns}
        if str(path) in self.records and self.records[str(path)] != record:
            raise ValueError(f'Input changed between reads: {path}')
        self.records[str(path)] = record
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def verify(self):
        # Re-read consumed metadata, without consulting a changing progress catalog.
        for path in list(self.records):
            self.read(path)


def numbers(values):
    values = np.asarray(values, dtype=float)
    return {'n': len(values), 'mean': float(values.mean()),
            'min': float(values.min()), 'max': float(values.max())} if len(values) else None


def leaf_census(root):
    """All timing/calibration-like leaf paths, including non-timing I/Q biases."""
    found = defaultdict(list)

    def walk(node, path):
        path = path + '/' + node.tag.rsplit('}', 1)[-1]
        if not len(node) and re.search('time|prf|pri$|rank|bias|delay|calibration|offset',
                                       node.tag, re.I):
            found[path].append((node.text or '').strip())
        for child in node:
            walk(child, path)
    walk(root, '')
    return {p: {'count': len(v), 'unique_count': len(set(v)),
                'examples': sorted(set(v))[:3]} for p, v in sorted(found.items())}


def annotation_record(root, source_times, iw2_mid_tau):
    swath = root.findtext('adsHeader/swath').lower()
    info = {k: root.findtext('imageAnnotation/imageInformation/' + k) for k in IMAGE_FIELDS}
    dt = float(info['azimuthTimeInterval'])
    lines = int(root.findtext('swathTiming/linesPerBurst'))
    lookup = {t: key for key, t in source_times.items() if key.endswith(swath)}
    bursts = []
    for index, burst in enumerate(root.findall('swathTiming/burstList/burst')):
        az = burst.findtext('azimuthTime')
        if az not in lookup:
            raise ValueError(f'Unregistered annotated source {swath} {az}')
        anx_error = seconds(az, info['ascendingNodeTime']) - float(burst.findtext('azimuthAnxTime'))
        bursts.append({'source_id': lookup[az], 'index': index, 'azimuthTime': az,
                       'sensingTime': burst.findtext('sensingTime'),
                       'azimuthAnxTime': burst.findtext('azimuthAnxTime'),
                       'sensing_minus_zero_doppler_seconds': seconds(burst.findtext('sensingTime'), az),
                       'anx_identity_error_seconds': anx_error})
    if set(lookup) != {b['azimuthTime'] for b in bursts}:
        raise ValueError('Registered sources absent from original annotation')
    grid_checks, terminal_count = [], 0
    for point in root.findall('geolocationGrid/geolocationGridPointList/geolocationGridPoint'):
        line = int(point.findtext('line'))
        if line == len(bursts) * lines:
            # burst2safe's terminal footprint row is not an image-line origin.
            # No following retained burst exists to check this boundary.
            terminal_count += 1
            continue
        candidates = [i for i in (line // lines, line // lines - 1)
                      if 0 <= i < len(bursts) and 0 <= line - i * lines <= lines]
        if not candidates:
            raise ValueError(f'No burst associated with geolocation line {line}')
        adjusted = (float(point.findtext('slantRangeTime')) - iw2_mid_tau) / 2
        raw = [seconds(point.findtext('azimuthTime'), bursts[i]['azimuthTime'])
               - (line - i * lines) * dt for i in candidates]
        chosen = min(range(len(raw)), key=lambda i: abs(raw[i] - adjusted))
        grid_checks.append({'line': line, 'sample': int(point.findtext('pixel')),
                            'azimuthTime': point.findtext('azimuthTime'),
                            'source_id': bursts[candidates[chosen]]['source_id'],
                            'grid_minus_burst_line_seconds': raw[chosen],
                            'half_range_minus_iw2_mid_seconds': adjusted,
                            'remaining_discrepancy_seconds': raw[chosen] - adjusted})
    downlinks = []
    for node in root.findall('generalAnnotation/downlinkInformationList/downlinkInformation'):
        values = node.find('downlinkValues')
        downlinks.append({'swath': node.findtext('swath'), 'azimuthTime': node.findtext('azimuthTime'),
                          'prf': node.findtext('prf'), **{k: values.findtext(k) for k in DOWNLINK_FIELDS},
                          'swst': [{'azimuthTime': z.findtext('azimuthTime'), 'value': z.findtext('value')}
                                   for z in values.findall('swstList/swst')]})
    replicas = []
    for node in root.findall('generalAnnotation/replicaInformationList/replicaInformation'):
        replicas.append({'swath': node.findtext('swath'),
                         'pgSource': node.findtext('referenceReplica/pgSource'),
                         'reference_timeDelay_seconds': float(node.findtext('referenceReplica/timeDelay')),
                         'internalTimeDelay_seconds': [float(z.text) for z in node.findall(
                             'replicaList/replica/internalTimeDelay')]})
    return {'image': info, 'bursts': bursts, 'downlinks': downlinks, 'replicas': replicas,
            'bistaticDelayCorrectionApplied': root.findtext(
                'imageAnnotation/processingInformation/bistaticDelayCorrectionApplied'),
            'geolocation_timing_diagnostic': {
                'iw2_mid_two_way_range_time_seconds': iw2_mid_tau,
                'unsupported_terminal_points': terminal_count, 'points': grid_checks,
                'max_remaining_discrepancy_seconds': max(
                    (abs(p['remaining_discrepancy_seconds']) for p in grid_checks), default=0)},
            'leaf_census': leaf_census(root)}


def run(stack, experiment, output):
    inputs = Inputs()
    plan = inputs.json(experiment / 'plan.json')
    fit = inputs.json(experiment / 'fit.json')
    analysis_path = (experiment / 'analysis/primary.json').resolve()
    analysis = inputs.json(analysis_path)
    expected = fit['products'].get(str(analysis_path))
    if expected != inputs.records[str(analysis_path)]['sha256']:
        raise ValueError('Frozen primary analysis checksum does not match fit.json')
    frame = inputs.json(stack / 'geometry/manifest.json')
    catalog = inputs.json(stack / 'plan.json')
    dates = sorted({e[k] for e in plan['links'] for k in ('earlier', 'later')})
    inventory = {x['date']: x for x in catalog['inventory'] if x.get('eligible')}
    records = []
    for day in dates:
        scene = inventory[day]
        aligned = inputs.json(stack / 'acquisitions' / day / 'manifest.json')
        source_times = aligned['specification']['burst_times']
        safe = Path(scene['path'])
        manifest = ET.fromstring(inputs.read(safe / 'manifest.safe'))
        versions = sorted({n.get('version') for n in manifest.iter()
                           if n.tag.endswith('software') and n.get('name') == 'Sentinel-1 IPF'})
        if versions != [fit['metadata'][day]['ipf']]:
            raise ValueError('Original IPF version differs from frozen experiment')
        resources = sorted({(n.get('role'), Path(n.get('name')).name)
                            for n in manifest.iter() if n.tag.endswith('resource')
                            and n.get('role', '').startswith('AUX_')})
        swaths, roots = {}, {}
        for path in sorted(safe.glob('annotation/s*-iw[12]-slc-hh-*.xml')):
            root = ET.fromstring(inputs.read(path))
            swath = root.findtext('adsHeader/swath').lower()
            if swath in roots:
                raise ValueError('Duplicate swath annotation')
            roots[swath] = root
        if set(roots) != {'iw1', 'iw2'}:
            raise ValueError(f'Missing original swath annotation: {day}')
        iw2 = roots['iw2']
        iw2_mid_tau = float(iw2.findtext('imageAnnotation/imageInformation/slantRangeTime')) + (
            int(iw2.findtext('swathTiming/samplesPerBurst')) - 1) * float(
                iw2.findtext('imageAnnotation/imageInformation/rangePixelSpacing')) / 299792458.
        for swath, root in roots.items():
            swaths[swath] = annotation_record(root, source_times, iw2_mid_tau)
        dt = float(swaths['iw1']['image']['azimuthTimeInterval'])
        delta = (float(swaths['iw2']['image']['zeroDopMinusAcqTime'])
                 - float(swaths['iw1']['image']['zeroDopMinusAcqTime']))
        records.append({'date': day, 'mission': scene['mission'], 'ipf': versions[0],
                        'safe': str(safe), 'auxiliary_resources': resources, 'swaths': swaths,
                        'zeroDopMinusAcqTime_IW2_minus_IW1_seconds': delta,
                        'zeroDopMinusAcqTime_centered_phase_seconds': phase(delta, dt)})

    prf = frame['coreg']['reference_grid']['prf']
    beta = float(analysis['models']['spacecraft']['parameters'][0])
    strata = {}
    for ipf in ('all', '002.72', '002.82'):
        strata[ipf] = {}
        for mission in ('S1A', 'S1B'):
            rows = [r for r in records if r['mission'] == mission and (ipf == 'all' or r['ipf'] == ipf)]
            strata[ipf][mission] = numbers([r['zeroDopMinusAcqTime_centered_phase_seconds'] * 1e6
                                           for r in rows])
        strata[ipf]['B_minus_A_mean_microseconds'] = (strata[ipf]['S1B']['mean']
                                                      - strata[ipf]['S1A']['mean'])
    summary = {'dates': len(records), 'swath_annotations': 2 * len(records),
               'registered_bursts': sum(len(s['bursts']) for r in records for s in r['swaths'].values()),
               'reference_prf_hz': prf, 'descriptive_spacecraft_beta_pixels': beta,
               'equivalent_reference_time_magnitude_microseconds': abs(beta / prf) * 1e6,
               'phase_diagnostic_by_ipf_microseconds': strata,
               'max_anx_identity_error_seconds': max(abs(b['anx_identity_error_seconds'])
                   for r in records for s in r['swaths'].values() for b in s['bursts']),
               'max_geolocation_time_remaining_discrepancy_seconds': max(
                   s['geolocation_timing_diagnostic']['max_remaining_discrepancy_seconds']
                   for r in records for s in r['swaths'].values())}
    constants = {}
    for mission in ('S1A', 'S1B'):
        constants[mission] = {}
        for swath in ('iw1', 'iw2'):
            ss = [r['swaths'][swath] for r in records if r['mission'] == mission]
            constants[mission][swath] = {
                'azimuthTimeInterval_seconds': sorted({float(s['image']['azimuthTimeInterval']) for s in ss}),
                'azimuthFrequency_hz': sorted({float(s['image']['azimuthFrequency']) for s in ss}),
                'slantRangeTime_seconds': sorted({float(s['image']['slantRangeTime']) for s in ss}),
                'pri_seconds': sorted({float(d['pri']) for s in ss for d in s['downlinks']}),
                'rank': sorted({int(d['rank']) for s in ss for d in s['downlinks']}),
                'instrumentConfigId': sorted({int(d['instrumentConfigId']) for s in ss for d in s['downlinks']}),
                'bistaticDelayCorrectionApplied': sorted({s['bistaticDelayCorrectionApplied'] for s in ss}),
                'reference_replica_timeDelay_seconds': numbers([d['reference_timeDelay_seconds']
                                                               for s in ss for d in s['replicas']])}
    summary['sampling_and_hardware'] = constants
    summary['explicit_azimuth_calibration_paths_found'] = sorted({p
        for r in records for s in r['swaths'].values() for p in s['leaf_census']
        if re.search('azimuth.*(calibration|offset|bias|delay)|(timing|time)(Calibration|Correction)', p, re.I)})
    summary['aux_itc_references_found'] = [r['date'] for r in records
        if any(role == 'AUX_ITC' for role, name in r['auxiliary_resources'])]
    lineage = Counter((r['mission'], r['ipf'], role, name) for r in records
                      for role, name in r['auxiliary_resources'] if role != 'AUX_RES')
    summary['auxiliary_lineage'] = [dict(mission=m, ipf=i, role=role, name=name, dates=n)
                                  for (m, i, role, name), n in sorted(lineage.items())]
    summary['checks'] = {
        'registered_source_timestamps_exact': True,
        'frozen_primary_checksum_verified': True,
        'burst_anx_identity_within_1_microsecond': summary['max_anx_identity_error_seconds'] <= 1e-6,
        'grid_range_convention_within_1_microsecond':
            summary['max_geolocation_time_remaining_discrepancy_seconds'] <= 1e-6,
        'caution': 'Internal consistency does not validate absolute image timing.'}
    # Published location errors are external evidence, never local physical predictions.
    external = {'source': CALIBRATION_URL, 'printed_page': 11, 'table': 4,
                'site': 'Surat Basin', 'period': '2016-10 to 2022-12',
                'redefined_ITC_azimuth_ALE_metres': {'S1A': {'IW1': -.272, 'IW2': .010},
                                                   'S1B': {'IW1': .013, 'IW2': -.031}},
                'IW2_minus_IW1_metres': {'S1A': .282, 'S1B': -.044},
                'B_minus_A_contrast_metres': -.326,
                'interpretation': 'Magnitude comparison only. ALE sign, targets, polarization, '
                                  'and site differ; not a signed prediction for Larsen.'}
    inputs.verify()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'annotations.json', records)
    write_json(output / 'summary.json', summary)
    write_json(output / 'external_evidence.json', external)
    plot(output, records, summary, external)
    report(output, summary)
    inputs.verify()
    write_json(output / 'record.json', {
        'completed_utc': datetime.now(timezone.utc).isoformat(), 'argv': sys.argv,
        'inputs': inputs.records, 'inputs_unchanged': True,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir() if p.is_file() and p.name != 'record.json'},
        'scope': 'Metadata-only investigation; no fitted calibration or production changes'})
    print(json.dumps(summary, indent=2))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def plot(output, records, summary, external):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for mission, color in (('S1A', '#3366aa'), ('S1B', '#dd7733')):
        rows = [r for r in records if r['mission'] == mission]
        axes[0].scatter([datetime.fromisoformat(r['date']) for r in rows],
                        [r['zeroDopMinusAcqTime_centered_phase_seconds'] * 1e6 for r in rows],
                        color=color, label=mission, s=32)
    axes[0].axvline(datetime(2017, 4, 1), color='.5', ls=':', label='IPF 2.82 starts')
    axes[0].set(title='Annotated grid phase: IW2 − IW1',
                ylabel='zeroDopMinusAcqTime modulo line interval (µs)')
    axes[0].xaxis.set_major_formatter(DateFormatter('%b'))
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel('Grid-placement diagnostic; not a timing-bias estimate', fontsize=8)
    required = summary['equivalent_reference_time_magnitude_microseconds']
    external_us = abs(external['B_minus_A_contrast_metres']) / 14.06382 / summary['reference_prf_hz'] * 1e6
    axes[1].bar(['Larsen observed\nspacecraft term', 'Published Surat\nlocation contrast'],
                [required, external_us], color=['#555555', '#8b6f9e'], width=.55)
    axes[1].set(title='Independent evidence: comparable size', ylabel='Equivalent azimuth time magnitude (µs)')
    axes[1].set_xlabel('Magnitude only; signs and site transfer unvalidated\n'
                      'External metres converted with representative Larsen sampling', fontsize=8)
    for i, value in enumerate((required, external_us)):
        axes[1].text(i, value + 1, f'{value:.1f} µs', ha='center')
    axes[1].set_ylim(0, 70)
    fig.suptitle('S1A/S1B annotation timing investigation')
    fig.savefig(output / 'annotation_timing.png', dpi=180)
    plt.close(fig)


def report(output, summary):
    s = summary
    phase_summary = s['phase_diagnostic_by_ipf_microseconds']
    text = f'''# S1A/S1B annotation timing investigation

The retained SLC annotations do not expose an explicit per-swath azimuth timing
calibration explaining the spacecraft-order seam. Independent calibration data
make this a plausible hypothesis, but do not establish its cause or sign at Larsen.

## Local annotation evidence

- Read {s['swath_annotations']} IW1/IW2 annotations and 31 SAFE manifests covering
  {s['dates']} dates and {s['registered_bursts']} registered source bursts. Exact
  source IDs are joined to original timestamps using each alignment manifest.
- The observed descriptive spacecraft term is {s['descriptive_spacecraft_beta_pixels']:.7f}
  reference azimuth pixels, equivalent in magnitude to
  {s['equivalent_reference_time_magnitude_microseconds']:.2f} µs at
  {s['reference_prf_hz']:.8f} Hz. A **differential** satellite timing effect between
  IW2 and IW1 is required. This conversion does not establish the sign of a tagged-time correction.
- Both satellites and both swaths use an identical output line interval
  0.0020555563 s. Within each swath, annotated PRI and rank are identical across
  satellites: IW1 rank 9, PRI 0.000582367437282 s; IW2 rank 8,
  PRI 0.000688882145835 s. These static values cannot by themselves supply an
  alternating spacecraft-order effect.
- Instrument configuration ID is 5 for S1A and 1 for S1B throughout this sample.
  Hardware configuration and spacecraft are confounded; this is provenance, not
  an independently estimated timing term. Auxiliary PP1/CAL/INS filenames and
  their IPF-specific usage are retained in summary.json.
- All annotations report `bistaticDelayCorrectionApplied=true`. This describes
  IPF processing, not proof that residual bistatic or FM mismatch corrections
  were applied by COMPASS. The earlier executed-path audit and predictions remain authoritative.
- `zeroDopMinusAcqTime` is an image-line acquisition/zero-Doppler time relation.
  Its IW2−IW1 difference includes whole output-line changes. After centering
  modulo the line interval, the B−A difference of date means is
  {phase_summary['all']['B_minus_A_mean_microseconds']:.3f} µs overall,
  {phase_summary['002.72']['B_minus_A_mean_microseconds']:.3f} µs for IPF 2.72,
  and {phase_summary['002.82']['B_minus_A_mean_microseconds']:.3f} µs for IPF 2.82.
  This diagnostic is not a physical correction and does not isolate an unknown
  absolute calibration. No prediction was made from it.
- Internal replica `timeDelay`/`internalTimeDelay` fields concern the range chirp
  replica, not a slow-time azimuth calibration. I/Q bias fields are also not
  azimuth timing offsets. Exact paths, counts, examples and replica delays are retained.
- No explicit azimuth-calibration leaf or AUX_ITC resource reference was found.
  Missing calibration information remains unknown, never a zero prediction.
- Cross-checks of burst UTC versus ANX-relative time differ by at most
  {s['max_anx_identity_error_seconds'] * 1e6:.3f} µs. Geolocation-grid timestamps
  are **not** identical to the corresponding image-line tags: they contain a
  range-dependent shift. Subtracting `(two_way_range_time − IW2_mid_range_time)/2`
  leaves at most {s['max_geolocation_time_remaining_discrepancy_seconds'] * 1e6:.3f} µs.
  At assembled joins either adjacent burst is considered; the terminal footprint
  row has no following source origin and is explicitly excluded. Every checked
  point and remaining discrepancy is retained. This describes the observed
  annotation convention; it is not a new calibrated physical prediction. These
  internal checks cannot detect a bias shared by image and annotation geometry.

`azimuthTime` identifies the first output line at zero Doppler; `sensingTime`
identifies the first input line. Their difference includes acquisition and
focusing geometry and must not be treated as a small calibration error.
See the [product specification]({SPECIFICATION_URL}), tables 6-55, 6-57, 6-65 and 6-85.
The [IPF algorithm description]({ALGORITHM_URL}), section 9.23, distinguishes
coarse midpoint tagging, fine bistatic shifts, and instrument timing correction.
Its later release is contextual documentation, not proof that a particular
correction was executed in IPF 2.72/2.82.

## External evidence and limits

The [ESA/SAR-MPC calibration study]({CALIBRATION_URL}) covers October 2016–December
2022. Its Surat Basin table 4 gives IW2−IW1 azimuth location-error contrasts of
+0.282 m for S1A and −0.044 m for S1B: a satellite difference of −0.326 m,
comparable in magnitude to Larsen's approximately 0.36 m. This is supporting
context, not a Larsen prediction: site, targets, polarization and ALE sign conventions
need separate validation. Other sites show smaller contrasts (table 5).

The study identifies S1A swath-dependent behavior, discusses the June 2016
antenna tile event as a possible origin, and supports post-June-2016 calibration
validity. It does not prove that mechanism here. Its AUX_ITC design supports swath
offsets, but those offsets were zero in the documented 2023 baseline. The common
per-satellite constants affect both swaths equally in native time; a large
satellite-wide constant alone is not an IW2−IW1 explanation. Transport through
the actual resampling map would be needed to assess any small mapping difference.
This investigation did not inspect current AUX_ITC files or apply ETAD.

## Interpretation and next discriminating check

The local annotation search is negative for an explicit usable calibration;
it does **not** refute a real S1A/S1B swath-specific localization bias. The known
S1A behavior strengthens that hypothesis. The existing small bistatic/FM terms
remain separate and cannot be replaced by external site means.

An independent test needs an external position reference or a stable-ground
S1A/S1B control with the same HH beam geometry and processing lineage. A useful
hardware test would compare pre- and post-June-2016 S1A controls while tracing the
original antenna-pattern/auxiliary processing. The 2017 sample alone cannot
separate spacecraft, hardware configuration and their processing calibration.
No production correction was made.

## Reproduction and retained artifacts

Run with the ISCE3 environment (NumPy/Matplotlib):

```sh
MPLCONFIGDIR=/tmp/glacier-mpl /Users/briel/miniforge3/envs/isce3/bin/python \\
  -m tools.investigate_annotation_timing \\
  --stack work/larsen/stack \\
  --experiment work/larsen/swath-experiment-2017h1
```

`annotations.json` retains exact image/burst/downlink timing and per-file path
censuses; `summary.json` holds satellite/IPF summaries and auxiliary lineage;
`external_evidence.json` identifies the published table; `annotation_timing.png`
shows diagnostic phase and independent magnitude context. `record.json` contains
SHA-256 input/output identities and confirms consumed metadata did not change.
Production imagery and the frozen experiment outputs were only read.
'''
    (output / 'REPORT.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', type=Path, default=Path('work/larsen/stack'))
    parser.add_argument('--experiment', type=Path,
                        default=Path('work/larsen/swath-experiment-2017h1'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run(args.stack.resolve(), args.experiment.resolve(),
        (args.output or args.experiment / 'annotation-timing').resolve())


if __name__ == '__main__':
    main()
