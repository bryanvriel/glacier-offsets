#!/usr/bin/env python
"""Run an AOI configuration and compare selected bursts with standalone runs.

Uses a separate --output-directory. Repeat with a velocity-mode configuration to
exercise real M5 geometry/orbit projection. Motion, masks and geometry are exact;
quality/formal-uncertainty fields allow small, reported numerical differences.
Use --strict for exact equality throughout, or --compare-only to inspect saved
outputs without processing or reading the original acquisition inputs.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pyproj import Transformer
from shapely.geometry import shape
import yaml
from glacier_offsets.aoi import load_aoi, project_footprint, select_aoi
from glacier_offsets.cli import main as workflow
from glacier_offsets.config import load_config
from glacier_offsets.products import checksum, write_json
from glacier_offsets.comparison import compare_burst, ANCILLARY_RTOL, ANCILLARY_ATOL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--backend', choices=['cpu','gpu','auto'], default='cpu')
    parser.add_argument('--compare-only', action='store_true', help='Compare existing outputs without processing or reading source acquisitions')
    parser.add_argument('--strict', action='store_true', help='Require exact equality for ancillary fields too')
    args = parser.parse_args(argv)
    output = args.output_directory.expanduser().resolve()
    if args.compare_only:
        # The saved selection, manifests and products are sufficient. In
        # particular, load_config would unnecessarily require external SAFE data.
        try:
            selection = json.loads((output/'batch/selection.json').read_text())
        except (OSError, ValueError) as exc:
            parser.error(f'Cannot read saved selection: {exc}')
        report = comparison_report(output, selection, strict=args.strict)
        report['scope'] = 'existing_products'
        report['aoi_restart_checks'] = 'not_checked'
        write_json(output/'comparison.json', report)
        print_report(report)
        return 1 if report['outcome'] == 'failed' else 0
    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if cfg.get('aoi') is None:
        parser.error('Configuration must select an AOI')
    if Path(cfg['output']['directory']) == output:
        parser.error('Choose a separate validation output directory')
    pairs, selection = select_aoi(cfg)
    if len(pairs) < 2:
        parser.error('Validation requires at least two selected common bursts')
    output.mkdir(parents=True, exist_ok=True)
    cfg['output']['directory'] = str(output/'batch')
    cfg['matching']['backend'] = args.backend
    # This validator compares independent burst products against standalone runs.
    # Common-grid SLC assembly has separate boundary/assembled-workflow validation.
    cfg['burst_merge']={'enabled':False}
    def run(document, name):
        path = output/(name+'.yaml')
        path.write_text(yaml.safe_dump(document,sort_keys=False))
        status = workflow(['run',str(path)])
        if status:
            raise RuntimeError(f'{name} failed with workflow exit {status}; inspect its manifests/logs')
    run(cfg,'batch')
    for pair in pairs:
        key = pair[0].id
        single = deepcopy(cfg)
        single.pop('aoi')
        single['inputs']['burst_id'] = key
        single['output']['directory'] = str(output/'standalone'/key)
        run(single,'single-'+key)
    report = comparison_report(output, selection, strict=args.strict)
    report.update(scope='full_workflow', backend_requested=args.backend,
                  gross_offsets_mode=cfg['gross_offsets']['mode'], aoi_restart_checks='not_checked')
    if report['outcome'] == 'passed':
        report['outcome'] = 'pending_restart'
    write_json(output/'comparison.json', report)
    if report['outcome'] == 'failed':
        print_report(report)
        return 1
    files = [p for pair in pairs for directory in ('radar','geo')
             for p in (output/'batch'/pair[0].id/directory).rglob('*') if p.is_file()]
    def snapshot():
        return {str(p):{'sha256':checksum(p),'mtime_ns':p.stat().st_mtime_ns} for p in files}
    before = snapshot()
    # Choose a tiny AOI inside the actual shared three-way intersection.
    aoi = load_aoi(cfg['aoi'])
    candidate = next(c for c in selection['candidates'] if c['selected'])
    ref,sec = [project_footprint(shape(candidate[k]),aoi.crs)
               for k in ('reference_footprint','secondary_footprint')]
    point = aoi.geometry.intersection(ref).intersection(sec).representative_point()
    lon,lat = Transformer.from_crs(aoi.crs,4326,always_xy=True).transform(point.x,point.y)
    modified = deepcopy(cfg)
    modified['aoi'] = {'bbox':[lon-.000001,lat-.000001,lon+.000001,lat+.000001]}
    try:
        run(modified,'changed-aoi')
        if before != snapshot():
            raise RuntimeError('AOI-only change modified reusable scientific products')
        run(cfg,'restored-aoi')
        if before != snapshot():
            raise RuntimeError('Restored AOI modified reusable scientific products')
    except Exception as exc:
        report.update(outcome='failed', aoi_restart_checks='failed', error=str(exc))
        write_json(output/'comparison.json', report)
        raise
    report['outcome'] = 'passed'
    report['aoi_restart_checks'] = 'passed'
    report['aoi_restart_preserved_checksums_and_mtimes'] = True
    write_json(output/'comparison.json',report)
    print_report(report)
    return 0


def comparison_report(output, selection, *, strict=False):
    comparisons = []
    for key in selection['selected_ids']:
        result = compare_burst(output/'batch'/key, output/'standalone'/key/key, strict=strict)
        comparisons.append({'burst_id': key} | result)
    failed = not comparisons or any(c['result'] == 'failed' for c in comparisons)
    return {'outcome': 'failed' if failed else 'passed',
            'selection_identity': selection['selection_identity'], 'comparisons': comparisons,
            'strict': strict, 'ancillary_tolerance': {'rtol': 0. if strict else ANCILLARY_RTOL,
                'atol': 0. if strict else ANCILLARY_ATOL, 'units': 'stored units of each field',
                'rule': 'abs(a-b) <= atol + rtol * max(abs(a),abs(b))'},
            'required_exact': 'coordinates, masks, gross/residual/total offsets, displacement/velocity and center geometry',
            'limitation': 'Ancillary agreement within tolerance is numerical agreement, not bitwise reproducibility or independent scientific accuracy; GPU/CPU agreement is a separate check.'}


def print_report(report):
    for comparison in report['comparisons']:
        print(f"{comparison['burst_id']}: {comparison['result']}")
        if comparison.get('error'):
            print('  ' + comparison['error'])
        for name, value in comparison.get('arrays', {}).items():
            if value['status'] != 'exact':
                print(f"  {name}: {value['status']}; differing={value.get('different_values', '?')}; max_abs={value.get('max_abs_difference', '?')}")
    print(f"Outcome: {report['outcome']}; AOI restart checks: {report.get('aoi_restart_checks', 'not_checked')}")
    print('Details: comparison.json in the validation output directory')


if __name__ == '__main__':
    raise SystemExit(main())
