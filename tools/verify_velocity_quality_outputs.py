"""Verify immutable pilot baselines and completed revised-validation artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
from glacier_offsets.velocity_io import digest,write_json


def verify(root):
    out=root/'2017-quality-regional';checks={}
    for label,folder,manifest in (
        ('original_2017',root/'2017','completion.json'),
        ('quality_regional',out,'completion.json'),
        ('sensitivity',out,'supplement_completion.json')):
        record=json.loads((folder/manifest).read_text())
        bad=[name for name,sha in record['products'].items() if digest(folder/name)!=sha]
        if bad:raise ValueError(f'{label}: product checksum mismatch: {bad}')
        checks[label+'_product_checksums']=len(record['products'])
    recipe=json.loads((out/'recipe.json').read_text())
    bad=[name for name,sha in recipe['software'].items() if digest(name)!=sha]
    if bad:raise ValueError('Main experiment source changed: '+str(bad))
    for name,source in (('supplement_completion.json','tools/assess_velocity_quality_sensitivity.py'),
                        ('period_scans_conditioned.json','tools/check_velocity_period_conditioning.py')):
        record=json.loads((out/name).read_text())
        if digest(source)!=record['source_sha256']:raise ValueError('Supplement source changed: '+source)
    checks['source_fingerprints_current']=True
    validation=json.loads((out/'simulator_validation.json').read_text())
    if len(validation['rows'])!=108 or any(r['successful_draws']!=200 or r['failed_fits_region'] for r in validation['rows']):
        raise ValueError('Incomplete regional validation')
    checks['validation_rows']=108;checks['successful_draws_per_row']=200
    regional=json.loads((out/'regional_fits.json').read_text())['rows']
    if len(regional)!=18 or any(r['bootstrap_draws']!=1000 for r in regional):raise ValueError('Incomplete bootstrap')
    for r in regional:
        cov=np.array(r['coefficient_covariance'])
        if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov).min()<=0:raise ValueError('Invalid covariance')
    checks['regional_rows']=18;checks['bootstrap_draws']=1000
    screen=json.loads((out/'screen_validation.json').read_text())
    for r in screen['heldout']:
        # Missing burst timestamps serialize as null and are unavailable in the
        # fit masks. A shared null marker is not a shared acquisition date.
        train={v for v in r['training_acquisition_days'] if v is not None}
        test={v for v in r['test_acquisition_days'] if v is not None}
        if train&test:raise ValueError('Held-out acquisition leakage')
    checks['heldout_acquisitions_disjoint']=True
    scans=json.loads((out/'period_scans_conditioned.json').read_text())
    for r in scans['rows']:
        if np.max(r['target_information_conditions'])>scans['target_information_condition_limit']:raise ValueError('Bad scan conditioning')
        if len(r['excluded_ill_conditioned_periods'])!=1 or abs(r['excluded_ill_conditioned_periods'][0]['period_days']-12)>1e-10:
            raise ValueError('Unexpected sampling-limit exclusion')
    checks['scan_conditioning_guard_verified']=True
    checks['maps_generated']=False
    if list(out.glob('map_*.nc')):raise ValueError('Unexpected production map')
    write_json(out/'verification.json',{'status':'passed','checks':checks,'source_sha256':digest(__file__)})
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('.runtime/velocity-recovery-larsen'))
    print(verify(p.parse_args().root.resolve()))

if __name__=='__main__':main()
