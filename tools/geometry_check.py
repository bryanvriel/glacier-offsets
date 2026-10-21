"""Independent native geometry round trips at three actual matching centers."""
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
from pyproj import Geod
from glacier_offsets.config import load_config
from glacier_offsets.inputs import select_pair
from glacier_offsets.geocode import center_geometry
from glacier_offsets.products import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/pair.yaml')
    parser.add_argument('--report',type=Path,default=Path('docs/m4-closeout/geometry.json'))
    args=parser.parse_args()
    # Pyre, imported by ISCE3, interprets --config as its own startup option.
    # Keep this script's parsed options out of the native library's initialization.
    original_argv=sys.argv
    try:
        sys.argv=original_argv[:1]
        import isce3
    finally:
        sys.argv=original_argv
    cfg=load_config(args.config)
    pair=select_pair(cfg);native=pair[0].native
    root=Path(cfg['output']['directory'])/pair[0].id
    manifest=json.loads((root/'manifest.json').read_text())
    with h5py.File(root/'radar/offsets.h5') as h:
        line,sample=h['center_line'][:],h['center_sample'][:]
    lon,lat,z,scale,valid=center_geometry(manifest['coreg'],line,sample)
    candidates=np.argwhere(valid)
    assert len(candidates)>=3,'Fewer than three geometry-valid centers'
    normalized=candidates/np.maximum(np.asarray(valid.shape)-1,1)
    indices=[]
    for fraction in [.25,.5,.75]:
        for index in np.argsort(np.sum((normalized-fraction)**2,axis=1)):
            ij=tuple(candidates[index])
            if ij not in indices:
                indices.append(ij);break
    grid=native.as_isce3_radargrid()
    shift_epoch=(grid.ref_epoch-native.orbit.reference_epoch).total_seconds()
    dem=isce3.geometry.DEMInterpolator(isce3.io.Raster(cfg['inputs']['dem']))
    samples=[]
    for i,j in indices:
        t,r=isce3.geometry.geo2rdr([np.deg2rad(lon[i,j]),np.deg2rad(lat[i,j]),z[i,j]],
                                 isce3.core.Ellipsoid(),native.orbit,isce3.core.LUT2d(),grid.wavelength,grid.lookside)
        line_error=(t-shift_epoch-grid.sensing_start)*grid.prf-line[i,j]
        sample_error=(r-grid.starting_range)/grid.range_pixel_spacing-sample[i,j]
        tcenter=grid.sensing_start+shift_epoch+float(line[i,j])/grid.prf
        rcenter=grid.starting_range+float(sample[i,j])*grid.range_pixel_spacing
        llhm=isce3.geometry.rdr2geo(tcenter-.5/grid.prf,rcenter,native.orbit,grid.lookside,
                                   dem=dem,threshold=1e-4,maxiter=100,extraiter=10)
        llhp=isce3.geometry.rdr2geo(tcenter+.5/grid.prf,rcenter,native.orbit,grid.lookside,
                                   dem=dem,threshold=1e-4,maxiter=100,extraiter=10)
        _,_,native_scale=Geod(ellps='WGS84').inv(*np.rad2deg(llhm[:2]),*np.rad2deg(llhp[:2]))
        samples.append({'native_azimuth_scale_m_per_line':float(native_scale),
                        'scale_difference_m_per_line':float(native_scale-scale[i,j]),
                        'center_line':float(line[i,j]),'center_sample':float(sample[i,j]),
                        'longitude':float(lon[i,j]),'latitude':float(lat[i,j]),'height_m':float(z[i,j]),
                        'azimuth_scale_m_per_line':float(scale[i,j]),
                        'native_geo2rdr_line_error_px':float(line_error),
                        'native_geo2rdr_range_error_px':float(sample_error)})
    passed=all(abs(p['native_geo2rdr_line_error_px'])<.02 and abs(p['native_geo2rdr_range_error_px'])<.05
               and abs(p['scale_difference_m_per_line'])<.02 for p in samples)
    report={'config':str(Path(args.config).resolve()),'burst_id':pair[0].id,
            'center_selection':'nearest geometry-valid centers to 25%, 50%, 75% of sparse-grid axes',
            'tolerances':{'line_px':.02,'range_px':.05,'azimuth_scale_m_per_line':.02},
            'rdr2geo_parameters':{'threshold_m':1e-4,'maxiter':100,'extraiter':10},
            'passed':passed,'samples':samples}
    write_json(args.report,report)
    print(json.dumps(report,indent=2))
    assert passed,'Native geometry round-trip or local azimuth-scale tolerance failed'


if __name__=='__main__':main()
