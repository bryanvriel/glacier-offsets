"""Compare split-year swath calibrations from a checksummed overlap experiment.

Run after extending the independent swath experiment; report only, no correction
publication or widening of the existing calibration. The temporal fitter itself
continues to read netCDFs only.
"""
import argparse
import json
from pathlib import Path

from glacier_offsets.velocity_calibration import compare_calibrations
import numpy as np
from glacier_offsets.velocity_io import digest, write_json, Collection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    parser.add_argument('--catalog',type=Path,required=True,help='Saved stack catalog supplying acquisition spacecraft')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inputs',nargs='+',help='Optional corrected horizontal netCDFs to bind a map-gate report')
    args=parser.parse_args(argv)
    fit=json.loads((args.experiment/'fit.json').read_text())
    plan=json.loads((args.experiment/'plan.json').read_text())
    if fit.get('specification',{}).get('experiment')!=plan.get('identity'):
        raise ValueError('Calibration fit does not belong to the frozen overlap plan')
    analysis=args.experiment/'analysis'/'primary.json'
    expected=fit.get('products',{}).get(str(analysis.resolve()))
    if fit.get('status')!='complete' or not expected or digest(analysis)!=expected:
        raise ValueError('Independent overlap analysis is incomplete or its checksum differs')
    # Verify recorded same-ground measurements rather than accepting an arbitrary
    # file of purported residuals as independent calibration evidence.
    for path,sha in fit['specification']['measurements'].items():
        p=Path(path)
        if not p.is_absolute():
            p=args.experiment/'measurements'/path/'report.json'
        if digest(p)!=sha:
            raise ValueError('Independent measurement provenance checksum differs')
        measured=json.loads(p.read_text())
        for product,checksum in measured.get('products',{}).items():
            if digest(product)!=checksum:
                raise ValueError('Independent measured contrast checksum differs')
    catalog=json.loads(args.catalog.read_text())
    missions={item['date']:item['mission'] for item in catalog['inventory'] if item.get('eligible')}
    rows=json.loads(analysis.read_text())['rows']
    dates=sorted({r[key] for r in rows for key in ('earlier','later')})
    result=compare_calibrations(rows,dates,missions)
    result['evidence']=[{'path':str(p.resolve()),'sha256':digest(p)} for p in (analysis,args.experiment/'fit.json',args.experiment/'plan.json',args.catalog)]
    if args.inputs:
        blockers=[]
        shared=result['shared_calibration_supported']
        with Collection(args.inputs) as collection:
            for source,h in zip(collection.sources,collection.handles):
                if any(str(h.attrs.get(k,''))!=str(plan['specification'][k]) for k in ('geometry_identity','layout_identity')):
                    blockers.append('Input reference geometry/layout differs from the independent overlap experiment')
                applied=source['correction_history'].get('calibration_json')
                if not applied:
                    blockers.append('A validated applied spacecraft calibration is missing')
                    continue
                for key,value in applied.get('scope',{}).get('matching',{}).items():
                    if plan['specification']['config']['matching'].get(key)!=value:
                        blockers.append('Applied calibration/overlap matching treatment differs: '+key)
                early=h['earlier_burst_time'][:];late=h['later_burst_time'][:]
                finite=np.isfinite(early+late)&(late>early)
                years={str(np.datetime64('1970-01-01')+np.timedelta64(int(t//86400),'D'))[:4] for t in np.r_[early[finite],late[finite]]}
                for year in years:
                    estimate=result['joint'] if shared else result['split_year_estimates'].get(year,{})
                    if not shared and not estimate.get('independently_supported'):
                        blockers.append('Year '+year+' lacks independent held-out validation')
                    expected=estimate.get('beta_px',np.nan)
                    if not np.isclose(applied['beta_px'],expected,atol=1e-10,rtol=0):
                        blockers.append('Applied coefficient differs from validated '+('shared' if shared else year)+' estimate; regenerate corrections/packages with a new calibration')
                if not shared:
                    for a,b in zip(early[finite],late[finite]):
                        ya=str(np.datetime64('1970-01-01')+np.timedelta64(int(a//86400),'D'))[:4]
                        yb=str(np.datetime64('1970-01-01')+np.timedelta64(int(b//86400),'D'))[:4]
                        if ya!=yb:
                            blockers.append('Cross-year pair has no validated correction; exclude it')
                            break
            result.update(schema='glacier-offsets-calibration-validation-v1',passed=not blockers,
                          input_sha256=[s['sha256'] for s in collection.sources],blockers=blockers,
                          cross_boundary_policy='validated' if shared else 'excluded')
    write_json(args.output,result)
    print(json.dumps({'report':str(args.output),'shared_calibration_supported':result['shared_calibration_supported']},indent=2))


if __name__=='__main__':main()
