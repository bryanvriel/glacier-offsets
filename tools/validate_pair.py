"""Execute fresh native zero/constant matching and audit real product contracts."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import h5py
import numpy as np
from osgeo import gdal
from glacier_offsets.config import load_config, gross_values
from glacier_offsets.products import checksum, write_json


def audit_products(root,cfg):
    manifest=json.loads((root/'manifest.json').read_text())
    with h5py.File(root/'radar/offsets.h5') as h:
        for k,axis in enumerate(['azimuth','range']):
            np.testing.assert_array_equal(h[f'total/{axis}_px'][:],h[f'gross/{axis}_px'][:]+h[f'residual/{axis}_px'][:])
            assert np.all(h[f'gross/{axis}_px'][:]==gross_values(cfg)[k])
        assert h.attrs['mergeGrossOffset']==0
        metadata=json.loads(h.attrs['metadata_json'])
        assert metadata['burst_id']==cfg['inputs']['burst_id']
        assert metadata['matching']==cfg['matching']
        assert metadata['input_provenance']['reference_orbit']['sha256']
        raw_shape=h['valid'].shape
    valid_raster=gdal.Open(str(root/'geo/valid_mask.tif'))
    validity=valid_raster.ReadAsArray().astype(bool)
    finite_count={};uncertainty={}
    days=manifest['interval_seconds']/86400
    for axis in ['range','azimuth']:
        displacement_ds=gdal.Open(str(root/'geo'/f'{axis}_displacement_m.tif'))
        velocity_ds=gdal.Open(str(root/'geo'/f'{axis}_velocity_m_per_day.tif'))
        assert displacement_ds.GetMetadataItem('burst_id')==cfg['inputs']['burst_id']
        assert displacement_ds.GetMetadataItem('acquisition_times_UTC')
        assert displacement_ds.GetMetadataItem('motion').startswith('TOTAL')
        assert displacement_ds.GetMetadataItem('units')=='m'
        assert velocity_ds.GetMetadataItem('units')=='m/day'
        assert displacement_ds.GetGeoTransform()==valid_raster.GetGeoTransform()
        assert displacement_ds.GetProjection()==valid_raster.GetProjection()
        displacement=displacement_ds.ReadAsArray();velocity=velocity_ds.ReadAsArray()
        assert validity.any(),f'No valid geocoded observations for {axis}'
        np.testing.assert_array_equal(np.isfinite(displacement),validity)
        np.testing.assert_array_equal(np.isfinite(velocity),validity)
        np.testing.assert_allclose(velocity[validity],displacement[validity]/days,rtol=2e-6,atol=1e-6)
        finite_count[axis]=int(validity.sum())
        sigma=gdal.Open(str(root/'geo'/f'{axis}_displacement_sigma_m.tif')).ReadAsArray()
        sigma_velocity=gdal.Open(str(root/'geo'/f'{axis}_velocity_sigma_m_per_day.tif')).ReadAsArray()
        available=np.isfinite(sigma)
        assert not np.any(available&~validity),'Uncertainty exists outside valid motion coverage'
        assert np.all(sigma[available]>=0)
        np.testing.assert_array_equal(np.isfinite(sigma_velocity),available)
        np.testing.assert_allclose(sigma_velocity[available],sigma[available]/days,rtol=2e-6,atol=1e-6)
        uncertainty[axis]={'finite_cells':int(available.sum()),'median_sigma_m':float(np.median(sigma[available])) if available.any() else None}
    assert (root/'diagnostics/uncertainty.png').is_file()
    return {'matching':manifest['matching_summary'],'parameters':manifest['matching']['matching'],
            'geocoding':manifest['geocoding'],'backend':manifest['matching']['backend'],
            'elapsed_matching_seconds':manifest['matching']['elapsed_seconds'],
            'finite_geocoded_cells':finite_count,'formal_uncertainty':uncertainty,
            'coreg_identity':manifest['coreg_identity'],'raw_grid_shape':raw_shape,
            'units_total_gross_nodata_checks':'passed'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pair.yaml')
    parser.add_argument('--constant-config',default='configs/pair.constant.yaml')
    parser.add_argument('--report-directory',type=Path,default=Path('docs/m4-closeout'))
    parser.add_argument('--snapshot-name',default='m4-closeout')
    parser.add_argument('--overwrite',action='store_true',help='Refresh this validation set, preserving earlier named baselines')
    args=parser.parse_args()
    cfg=load_config(args.config);constant=load_config(args.constant_config)
    assert cfg['inputs']==constant['inputs'] and cfg['output']==constant['output']
    assert cfg['matching']==constant['matching'] and cfg['geocoding']==constant['geocoding']
    assert cfg.get('gross_offsets',{}).get('mode','zero')=='zero'
    assert constant['gross_offsets']['mode']=='constant'
    assert Path(args.snapshot_name).name==args.snapshot_name and args.snapshot_name not in {'.','..'}
    root=Path(cfg['output']['directory'])/cfg['inputs']['burst_id']
    snapshot=root/'validation'/args.snapshot_name
    if snapshot.exists() and not args.overwrite:parser.error(f'{snapshot} exists; select a new --snapshot-name or use --overwrite')
    args.report_directory.mkdir(parents=True,exist_ok=True)
    launcher=Path('bin/glacier-offsets').resolve()
    env=os.environ.copy();env['GLACIER_OFFSETS_PYTHON']=sys.executable
    def command(label,stage,config,extra=()):
        cmd=[str(launcher),stage,config,*extra]
        with (args.report_directory/f'{label}.txt').open('w') as log:
            log.write('Command: '+ ' '.join(cmd)+'\n');log.flush()
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    gdal.UseExceptions()
    if not (root/'manifest.json').is_file():command('initial-coregister','coregister',args.config)
    initial=json.loads((root/'manifest.json').read_text())
    coreg_identity=initial['coreg_identity']
    slcs=[Path(p) for p in initial['coreg']['slcs'].values()]
    slc_mtimes={str(p):p.stat().st_mtime_ns for p in slcs}
    report={'config':str(Path(args.config).resolve()),'constant_config':str(Path(args.constant_config).resolve())}
    try:
        for mode,config,settings in [('zero',args.config,cfg),('constant',args.constant_config,constant)]:
            command(f'{mode}-match','match',config,['--force'])
            command(f'{mode}-run','run',config)
            report[mode]=audit_products(root,settings)
            assert report[mode]['coreg_identity']==coreg_identity
            dest=snapshot/mode;dest.mkdir(parents=True,exist_ok=True)
            (dest/'radar').mkdir(exist_ok=True)
            for name in ['offsets.h5','matching_grid.h5']:shutil.copy2(root/'radar'/name,dest/'radar'/name)
            for source,target in [(root/'radar/native',dest/'radar/native'),(root/'geo',dest/'geo'),(root/'diagnostics',dest/'diagnostics')]:
                shutil.copytree(source,target,dirs_exist_ok=args.overwrite)
            shutil.copy2(root/'manifest.json',dest/'manifest.json')
            write_json(args.report_directory/'real-validation.json',report)
        with h5py.File(snapshot/'zero/radar/offsets.h5') as a,h5py.File(snapshot/'constant/radar/offsets.h5') as b:
            for name in ['center_line','center_sample']:np.testing.assert_array_equal(a[name][:],b[name][:])
            common=a['valid'][:]&b['valid'][:]
            assert common.any(),'No common quality-valid windows for gross-mode sensitivity check'
            report['comparison']={'common_valid_windows':int(common.sum()),'ground_truth':'not supplied; sensitivity comparison only'}
            for axis in ['range','azimuth']:
                delta=b[f'total/{axis}_px'][:][common]-a[f'total/{axis}_px'][:][common]
                report['comparison'][axis]={'median_difference_px':float(np.median(delta)),
                                           'p95_absolute_difference_px':float(np.quantile(np.abs(delta),.95)),
                                           'maximum_absolute_difference_px':float(np.max(np.abs(delta))),
                                           'fraction_within_0_25_px':float(np.mean(np.abs(delta)<=.25))}
    finally:
        command('restore-zero','run',args.config)
    assert all(p.stat().st_mtime_ns==slc_mtimes[str(p)] for p in slcs),'Coregistered SLCs were unexpectedly recomputed'
    products=slcs+[root/'radar/offsets.h5',root/'radar/matching_grid.h5',root/'geo/center_geometry.h5',root/'geo/geocoding.json']
    products+=sorted((root/'geo').glob('*.tif'))+sorted((root/'radar/native').glob('*'))
    before={str(p):{'mtime_ns':p.stat().st_mtime_ns,'sha256':checksum(p)} for p in products if p.is_file()}
    command('unchanged-restart','run',args.config)
    after={p:{'mtime_ns':Path(p).stat().st_mtime_ns,'sha256':checksum(p)} for p in before}
    assert before==after,'An unchanged restart modified a scientific product'
    current=json.loads((root/'manifest.json').read_text())
    source=Path('src/glacier_offsets')
    hashes=current['workflow_source_at_validation']['sha256']
    assert all(checksum(source/name)==value for name,value in hashes.items())
    report['restart']={'cached_products_unchanged':len(before),'source_hashes_verified':len(hashes),
                       'coregistered_slcs_unchanged_across_modes':True,'products':before}
    report['snapshots']=str(snapshot.resolve())
    report['passed']=True
    write_json(args.report_directory/'real-validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
