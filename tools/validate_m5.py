"""Audit one velocity/zero real pair, including an independent radar Jacobian.

Run after matching/geocoding both configs. This is a local validation tool,
not a stack driver or an independent displacement truth assessment.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import isce3
from glacier_offsets.config import load_config
from glacier_offsets.inputs import select_pair
from glacier_offsets.velocity_prior import radar_prediction
from glacier_offsets.products import write_json


def audit(velocity_config,zero_config,output):
    cfg,zcfg=load_config(velocity_config),load_config(zero_config)
    roots=[Path(c['output']['directory'])/c['inputs']['burst_id'] for c in (cfg,zcfg)]
    manifests=[json.loads((r/'manifest.json').read_text()) for r in roots]
    assert manifests[0]['coreg_identity']==manifests[1]['coreg_identity']
    assert manifests[0]['coreg']['slcs']==manifests[1]['coreg']['slcs']
    with h5py.File(roots[0]/'radar/offsets.h5') as v,h5py.File(roots[1]/'radar/offsets.h5') as z:
        for name in ('center_line','center_sample'):np.testing.assert_array_equal(v[name][:],z[name][:])
        total=np.stack([v[f'total/{axis}_px'][:] for axis in ('azimuth','range')],-1)
        baseline=np.stack([z[f'total/{axis}_px'][:] for axis in ('azimuth','range')],-1)
        gross=np.stack([v[f'gross/{axis}_px'][:] for axis in ('azimuth','range')],-1)
        residual=np.stack([v[f'residual/{axis}_px'][:] for axis in ('azimuth','range')],-1)
        np.testing.assert_array_equal(total,gross+residual)
        eligible=v['prior/valid'][:];common=v['valid'][:]&z['valid'][:]
        assert common.any()
        flags=v['prior/rejection_flags'][:]
        assert np.isnan(total[~eligible]).all()
        lon,lat,height=[v[f'prior/{name}'][:] for name in ('longitude_deg','latitude_deg','height_m')]
        # Sparse directed checks over this burst, including both signs and zero.
        positions=np.argwhere(eligible);positions=positions[np.linspace(0,len(positions)-1,9).astype(int)]
        ll=np.array([lon[tuple(i)] for i in positions])[None,:]
        la=np.array([lat[tuple(i)] for i in positions])[None,:]
        hh=np.array([height[tuple(i)] for i in positions])[None,:]
        predicted=np.stack([v[f'prior/predicted_px/{axis}'][:] for axis in ('azimuth','range')],-1)
        report={'coreg_identity':manifests[0]['coreg_identity'],'same_coreg_slcs':True,
                'interval_seconds':manifests[0]['interval_seconds'],
                'windows':int(eligible.size),'prior_eligible':int(eligible.sum()),
                'prior_rejections':{name:int(np.count_nonzero(flags&bit)) for bit,name in
                                    [(1,'geometry'),(2,'missing_velocity'),(4,'projection'),(8,'overflow'),(16,'outside_secondary')]},
                'zero':manifests[1]['matching_summary'],'velocity':manifests[0]['matching_summary'],
                'common_valid':int(common.sum()),'total_difference_median_abs_px':np.median(np.abs(total[common]-baseline[common]),axis=0).tolist(),
                'total_difference_p95_abs_px':np.percentile(np.abs(total[common]-baseline[common]),95,axis=0).tolist(),
                'fraction_common_with_difference_over_1px':float(np.mean(np.any(np.abs(total[common]-baseline[common])>1.,axis=-1))),
                'predicted_min_max_px':[np.nanmin(predicted,axis=(0,1)).tolist(),np.nanmax(predicted,axis=(0,1)).tolist()],
                'gross_min_max_px':[np.nanmin(gross,axis=(0,1)).tolist(),np.nanmax(gross,axis=(0,1)).tolist()],
                'exact_total_equals_gross_plus_residual':True,'missing_prior_is_masked':True,
                'comparison_limit':'Disagreement is not an accuracy or outlier rate without independent truth. Permissive cutoffs retain false matches.',
                'geocoding':manifests[0].get('geocoding')}
        strict=common&(v['snr'][:]>=2)&(z['snr'][:]>=2)&(v['correlation_peak'][:]>=.2)&(z['correlation_peak'][:]>=.2)
        report['posthoc_snr2_peak02']={'common':int(strict.sum()),
            'difference_median_abs_px':np.median(np.abs(total[strict]-baseline[strict]),axis=0).tolist() if strict.any() else None,
            'fraction_difference_over_1px':float(np.mean(np.any(np.abs(total[strict]-baseline[strict])>1.,axis=-1))) if strict.any() else None}
    burst=select_pair(cfg)[0].native;grid=burst.as_isce3_radargrid();ellipsoid=isce3.core.Ellipsoid()
    # Independent zero-Doppler differential: dt/dp = v dot dp /
    # (v dot v - (p-s) dot acceleration); dR/dp = LOS dot dp.
    expected=[]
    for longitude,latitude,h in zip(ll.ravel(),la.ravel(),hh.ravel()):
        l,b=np.deg2rad([longitude,latitude])
        p=ellipsoid.lon_lat_to_xyz([l,b,h])
        t,_=isce3.geometry.geo2rdr([l,b,h],ellipsoid,burst.orbit,isce3.core.LUT2d(),grid.wavelength,grid.lookside)
        s,vel=burst.orbit.interpolate(t)
        _,vm=burst.orbit.interpolate(t-.01);_,vp=burst.orbit.interpolate(t+.01)
        acc=(vp-vm)/.02;los=(p-s)/np.linalg.norm(p-s)
        en=[np.array([-np.sin(l),np.cos(l),0.]),np.array([-np.sin(b)*np.cos(l),-np.sin(b)*np.sin(l),np.cos(b)])]
        expected.append([[np.dot(vel,d)/(np.dot(vel,vel)-np.dot(p-s,acc))*grid.prf,
                          np.dot(los,d)/grid.range_pixel_spacing] for d in en])
    expected=np.array(expected)
    errors=[]
    for axis,(east,north) in enumerate([(1.,0.),(0.,1.)]):
        plus,ok=radar_prediction(ll,la,hh,np.full(ll.shape,east),np.full(ll.shape,north),grid,burst.orbit,np.ones(ll.shape,bool))
        minus,okm=radar_prediction(ll,la,hh,np.full(ll.shape,-east),np.full(ll.shape,-north),grid,burst.orbit,np.ones(ll.shape,bool))
        assert ok.all() and okm.all()
        np.testing.assert_allclose(plus[0],expected[:,axis],atol=3e-5,rtol=0)
        np.testing.assert_allclose(minus[0],-expected[:,axis],atol=3e-5,rtol=0)
        errors.append(float(np.max(np.abs(plus[0]-expected[:,axis]))))
    zero,ok=radar_prediction(ll,la,hh,np.zeros(ll.shape),np.zeros(ll.shape),grid,burst.orbit,np.ones(ll.shape,bool))
    assert ok.all();np.testing.assert_array_equal(zero,np.zeros_like(zero))
    report['independent_zero_doppler_jacobian']={'locations':len(positions),'directed_motion_m':1.,
        'max_abs_error_px_east_north':errors,'tolerance_px':3e-5,'negative_signs_and_zero_verified':True}
    # Independently reproduce nearest radar-cell mapping from TOTAL fields.
    from glacier_offsets.geocode import make_mapper,physical_fields
    from pyproj import Transformer
    from osgeo import gdal
    with h5py.File(roots[0]/'geo/center_geometry.h5') as h,h5py.File(roots[0]/'radar/offsets.h5') as v:
        longitude,latitude=h['longitude_deg'][:],h['latitude_deg'][:]
        valid=v['input_valid'][:]&((v['rejection_flags'][:]&7)==0)&h['geometry_valid'][:]
        x,y=Transformer.from_crs(4326,cfg['geocoding']['epsg'],always_xy=True).transform(longitude,latitude)
        mapper=make_mapper(x,y,valid,cfg['geocoding']['spacing_m'],cfg['geocoding']['resampling'])
        fields=physical_fields(v['total/range_px'][:],v['total/azimuth_px'][:],
            manifests[0]['coreg']['reference_grid']['range_pixel_spacing'],h['azimuth_ground_spacing_m_per_slc_line'][:],manifests[0]['interval_seconds'])
        for field in fields:
            ds=gdal.Open(str(roots[0]/'geo'/f'{field}.tif'))
            np.testing.assert_array_equal(ds.ReadAsArray(),mapper.map(fields[field]))
            assert 'TOTAL measured gross + residual' in ds.GetMetadataItem('motion')
            assert ds.GetMetadataItem('velocity_prior_sources_json')
    report['geocoded_total_and_velocity_audit']='exact mapped total, physical scales, interval, NaN masks and prior provenance verified'
    write_json(output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('velocity_config');parser.add_argument('zero_config');parser.add_argument('output')
    args=parser.parse_args();audit(args.velocity_config,args.zero_config,args.output)
