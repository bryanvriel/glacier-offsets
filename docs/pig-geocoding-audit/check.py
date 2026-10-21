"""Read-only comparison of PIG radar masks and existing geocoded total motion.

Run from the workspace with PYTHONPATH=src:.runtime/deps in the isce3 env.
Additional mask choices are previews in memory, not revised scientific products.
"""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from osgeo import gdal
from pyproj import Transformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glacier_offsets.geocode import RadarNearestMapper

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE / 'work/pig/jan12_jan18/t065_139122_iw2'
OUT = Path(__file__).resolve().parent
gdal.UseExceptions()
files = [ROOT/'radar/offsets.h5', ROOT/'manifest.json', *sorted((ROOT/'geo').glob('*'))]
before = {str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
          for p in files if p.is_file()}
manifest = json.loads((ROOT/'manifest.json').read_text())
with h5py.File(ROOT/'radar/offsets.h5') as h:
    raw = {key: h[key][:] for key in ['total/range_px', 'total/azimuth_px',
           'snr', 'correlation_peak', 'input_valid', 'valid', 'rejection_flags']}
    metadata = json.loads(h.attrs['metadata_json'])
with h5py.File(ROOT/'geo/center_geometry.h5') as h:
    geo = {key: h[key][:] for key in ['longitude_deg', 'latitude_deg',
           'geometry_valid', 'azimuth_ground_spacing_m_per_slc_line']}
x, y = Transformer.from_crs(4326,3031,always_xy=True).transform(
    geo['longitude_deg'],geo['latitude_deg'])
finite = np.isfinite(raw['total/range_px']) & np.isfinite(raw['total/azimuth_px'])
eligible = raw['input_valid'] & ((raw['rejection_flags'] & 7)==0) & finite
masks = {'input':eligible, 'quality':raw['valid'] & finite}
mapping = {k:RadarNearestMapper(x,y,m & geo['geometry_valid'],200) for k,m in masks.items()}
support = RadarNearestMapper(x,y,np.ones_like(finite,bool) & geo['geometry_valid'],200)
quality = mapping['quality']; input_map = mapping['input']
rgscale = manifest['coreg']['reference_grid']['range_pixel_spacing']
azscale = geo['azimuth_ground_spacing_m_per_slc_line']
days = manifest['interval_seconds']/86400
for axis,scale in [('range',rgscale),('azimuth',azscale)]:
    expected = quality.map(raw[f'total/{axis}_px']*scale)
    ds = gdal.Open(str(ROOT/'geo'/f'{axis}_displacement_m.tif'))
    actual = ds.ReadAsArray()
    assert ds.GetGeoTransform()==quality.transform
    np.testing.assert_array_equal(np.isfinite(actual), quality.valid)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=0,equal_nan=True)
    velocity = gdal.Open(str(ROOT/'geo'/f'{axis}_velocity_m_per_day.tif')).ReadAsArray()
    np.testing.assert_allclose(velocity,quality.map(raw[f'total/{axis}_px']*scale/days),rtol=0,atol=0,equal_nan=True)

snr_ok = eligible & (raw['snr']>=metadata['matching']['snr_threshold'])
peak = raw['correlation_peak']
sensitivity = []
for cutoff in [0, .05, .1, .15, .2, .25]:
    accepted = snr_ok & (peak>=cutoff)
    mapper = RadarNearestMapper(x,y,accepted & geo['geometry_valid'],200)
    sensitivity.append({'peak_cutoff':cutoff,'accepted_radar_windows':int(accepted.sum()),
                        'preview_map_cells':int(mapper.valid.sum())})
# A visually coherent range region, not an independent glacier/stationary mask.
coherent = eligible.copy()
with h5py.File(ROOT/'radar/offsets.h5') as h:
    coherent &= (h['center_sample'][:]>=17000) & (h['center_sample'][:]<=21000)
    coherent &= (h['center_line'][:]>=300) & (h['center_line'][:]<=1000)
report = {
    'radar_shape':list(finite.shape),'raw_finite_windows':int(finite.sum()),
    'input_valid_windows':int(raw['input_valid'].sum()),
    'input_eligible_windows':int(eligible.sum()),'quality_windows':int(masks['quality'].sum()),
    'geometry_rejected_quality_windows':int((masks['quality'] & ~geo['geometry_valid']).sum()),
    'unique_quality_sources_represented_on_map':int(np.unique(quality.source_indexes).size),
    'map_rectangle_cells':int(quality.valid.size),'geometry_supported_map_cells':int(support.valid.sum()),
    'quality_map_cells':int(quality.valid.sum()),'input_preview_map_cells':int(input_map.valid.sum()),
    'snr_passing_eligible_windows':int(snr_ok.sum()),
    'peak_only_rejected_snr_passing_windows':int((snr_ok & (peak<.2)).sum()),
    'peak_threshold_sensitivity_previews':sensitivity,
    'coherent_region_description':'range samples 17000–21000, azimuth lines 300–1000; visual example only',
    'coherent_region':{'input_eligible_windows':int(coherent.sum()),
        'quality_windows':int((coherent & raw['valid']).sum()),
        'snr_quantiles_10_50_90':np.quantile(raw['snr'][coherent],[.1,.5,.9]).tolist(),
        'peak_quantiles_10_50_90':np.quantile(peak[coherent],[.1,.5,.9]).tolist()},
    'existing_geo_displacement_velocity_equal_source_total':True,
    'previews_only':'No mask/config/native/radar/geo products changed; lower thresholds are not validated acceptance criteria.'}

# All offset panels share axis-specific limits and physical displacement units.
fig, axes = plt.subplots(2,4,figsize=(20,8),layout='constrained')
for row,(axis,scale) in enumerate([('range',rgscale),('azimuth',azscale)]):
    values = raw[f'total/{axis}_px']*scale
    panels = [np.where(eligible,values,np.nan),np.where(masks['quality'],values,np.nan),
              quality.map(values),input_map.map(values)]
    limit = float(np.quantile(np.abs(values[eligible]),.98))
    for col,data in enumerate(panels):
        cmap = plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e6e6e6')
        kwargs = {}
        if col>=2:
            gt=quality.transform;ny,nx=quality.shape
            kwargs['extent']=[gt[0]/1000,(gt[0]+nx*gt[1])/1000,
                              (gt[3]+ny*gt[5])/1000,gt[3]/1000]
        im = axes[row,col].imshow(data,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto',**kwargs)
        title = ['Radar: input eligible','Radar: quality accepted','Existing geo: quality','Geo preview: input eligible'][col]
        axes[row,col].set_title(f'{title}\nTOTAL {axis} displacement [m]')
        axes[row,col].set_xlabel('Projected x [km]' if col>=2 else 'Sparse range column')
        axes[row,col].set_ylabel('Projected y [km]' if col>=2 else 'Sparse azimuth row')
        fig.colorbar(im,ax=axes[row,col],shrink=.8,extend='both')
fig.suptitle('PIG Jan 12–18 · identical color scales per axis · gray = excluded/nodata\n'
             'Input previews include rejected observations; existing products remain unchanged')
fig.savefig(OUT/'offset-comparison.png',dpi=150);plt.close(fig)

fig,axes=plt.subplots(3,1,figsize=(15,8),layout='constrained')
for ax,data,title,limit in zip(axes,[raw['snr'],peak,raw['valid'].astype(float)],
        ['SNR (cutoff 2)','Refined amplitude-correlation peak (cutoff 0.2)',
         'Accepted radar mask'],[np.quantile(raw['snr'][eligible],.98),.4,1]):
    selected = np.where(raw['input_valid'],data,np.nan)
    im=ax.imshow(selected,aspect='auto',vmin=0,vmax=limit)
    ax.set_title(title);ax.set_ylabel('Sparse azimuth row')
    fig.colorbar(im,ax=ax,shrink=.8)
axes[-1].set_xlabel('Sparse range column')
fig.savefig(OUT/'quality-comparison.png',dpi=150);plt.close(fig)
after = {str(p): (p.stat().st_mtime_ns,hashlib.sha256(p.read_bytes()).hexdigest())
         for p in files if p.is_file()}
assert before==after,'Audit changed a processing product'
report['unchanged_processing_files_checked']=len(before)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
