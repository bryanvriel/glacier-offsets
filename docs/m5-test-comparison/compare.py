"""Read-only comparison of the two user test runs; original products unchanged."""
import json
from pathlib import Path
import h5py
import numpy as np
from osgeo import gdal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glacier_offsets.config import load_config
from glacier_offsets.products import checksum,write_json

OUT=Path(__file__).parent
configs=[load_config('configs/'+p) for p in ['pair.test.yaml','pair.test.velocity.yaml']]
roots=[Path(c['output']['directory'])/c['inputs']['burst_id'] for c in configs]
manifest=[json.loads((r/'manifest.json').read_text()) for r in roots]
records=[]
for root in roots:
 with h5py.File(root/'radar/offsets.h5') as h:
  names=['center_line','center_sample','snr','correlation_peak','valid','input_valid','rejection_flags']
  a={k:h[k][:] for k in names}
  for field in ['total','residual','gross']:
   a[field]=np.stack([h[field+'/'+axis+'_px'][:] for axis in ['azimuth','range']],-1)
  if 'prior' in h:
   a['prior_flags']=h['prior/rejection_flags'][:];a['prior_valid']=h['prior/valid'][:]
  records.append(a)
z,v=records;difference=v['total']-z['total']
for name in ['center_line','center_sample']:np.testing.assert_array_equal(z[name],v[name])
np.testing.assert_array_equal(v['total'],v['gross']+v['residual'])
common=z['valid']&v['valid'];strong=common&(z['snr']>=2)&(v['snr']>=2)&(z['correlation_peak']>=.2)&(v['correlation_peak']>=.2)
gross_zero=np.all(v['gross']==0,axis=-1)
changed=common&np.any(abs(difference)>.1,axis=-1)
report={'configs':['configs/pair.test.yaml','configs/pair.test.velocity.yaml'],'roots':list(map(str,roots)),
 'same_coreg_identity':manifest[0]['coreg_identity']==manifest[1]['coreg_identity'],
 'same_matching_settings':manifest[0]['matching']['matching']==manifest[1]['matching']['matching'],
 'same_radar_centers':True,'gross_added_exactly_once':True,
 'matching':[m['matching_summary'] for m in manifest], 'geocoding':[m['geocoding'] for m in manifest],
 'common_valid':int(common.sum()),'posthoc_strong_common':int(strong.sum()),
 'total_difference_quantiles_px_azimuth_range':np.percentile(np.abs(difference[common]),[50,90,95,99,100],axis=0).tolist(),
 'changed_over_0p1_px':int(changed.sum()),'changed_over_0p25_px':int((common&np.any(abs(difference)>.25,axis=-1)).sum()),
 'changed_over_1_px':int((common&np.any(abs(difference)>1,axis=-1)).sum()),
 'strong_total_exactly_identical':bool(np.array_equal(z['total'][strong],v['total'][strong])),
 'common_zero_gross_count':int((common&gross_zero).sum()),
 'common_zero_gross_exactly_identical':bool(np.array_equal(z['total'][common&gross_zero],v['total'][common&gross_zero])),
 'prior_rejections':{name:int(np.count_nonzero(v['prior_flags']&bit)) for bit,name in [(1,'geometry'),(2,'missing_velocity'),(4,'projection'),(8,'overflow'),(16,'outside_secondary')]},
 'changed_quality':{'zero_peak_p50_p95':np.percentile(z['correlation_peak'][changed],[50,95]).tolist(),
                    'velocity_peak_p50_p95':np.percentile(v['correlation_peak'][changed],[50,95]).tolist(),
                    'zero_snr_p50_p95':np.percentile(z['snr'][changed],[50,95]).tolist(),
                    'velocity_snr_p50_p95':np.percentile(v['snr'][changed],[50,95]).tolist()},
 'source_slcs':{},'geometry_equal':{}}
# File hashes and pixel hashes are distinct: container metadata need not agree.
for side in ['reference','secondary']:
 paths=[m['coreg']['slcs'][side] for m in manifest]
 ds=[gdal.Open(p) for p in paths];same=True
 for line in range(0,ds[0].RasterYSize,128):
  a,b=[d.ReadAsArray(0,line,d.RasterXSize,min(128,d.RasterYSize-line)) for d in ds]
  same &= np.array_equal(a,b,equal_nan=True)
 report['source_slcs'][side]={'paths':paths,'file_sha256':[checksum(p) for p in paths],'pixels_exactly_equal':bool(same)}
for name in ['x','y','z','layover_shadow_mask']:
 ds=[gdal.Open(m['coreg']['geometry'][name]) for m in manifest];same=True
 for line in range(0,ds[0].RasterYSize,128):
  a,b=[d.ReadAsArray(0,line,d.RasterXSize,min(128,d.RasterYSize-line)) for d in ds]
  same &= np.array_equal(a,b,equal_nan=True)
 report['geometry_equal'][name]=bool(same)
# All displayed differences use total; residual step patterns are expected.
for axis,k in [('range',1),('azimuth',0)]:
 fig,axes=plt.subplots(5,1,figsize=(15,12),layout='constrained')
 goodvals=np.r_[z['total'][strong,k],v['total'][strong,k]]
 limits=np.quantile(goodvals,[.01,.99]);limits=[float(limits[0]-.25),float(limits[1]+.25)]
 panels=[(np.where(z['input_valid'],z['total'][...,k],np.nan),'Zero-prior TOTAL offsets',limits,'RdBu_r'),
         (np.where(v['input_valid'],v['total'][...,k],np.nan),'Velocity-prior TOTAL offsets',limits,'RdBu_r'),
         (np.where(common,difference[...,k],np.nan),'Velocity minus zero TOTAL (common valid), ±1 pixel scale',[-1,1],'RdBu_r'),
         (v['gross'][...,k],'Applied integer gross offset [pixels]',None,'RdBu_r'),
         (v['prior_flags'].astype(float),'Prior rejection bits: 1 geometry, 2 missing velocity, 16 outside secondary',[0,31],'viridis')]
 for ax,(data,title,bounds,cmap) in zip(axes,panels):
  artist=ax.imshow(data,aspect='auto',cmap=cmap,vmin=bounds[0] if bounds else None,vmax=bounds[1] if bounds else None)
  ax.set_title(title);ax.set_ylabel('Radar grid row');ax.set_xlabel('Radar grid range column')
  fig.colorbar(artist,ax=ax,label=f'{axis} pixels' if 'bits' not in title else 'bit flags')
 fig.suptitle(f'{axis.capitalize()} comparison; same reference chips; color scales shared for total panels')
 fig.savefig(OUT/f'{axis}-radar-comparison.png',dpi=150);plt.close(fig)
# Geographic QA and displacement share the exact same geogrid.
geo={}
for axis in ['range','azimuth']:
 datasets=[gdal.Open(str(r/'geo'/f'{axis}_displacement_m.tif')) for r in roots]
 assert datasets[0].GetGeoTransform()==datasets[1].GetGeoTransform()
 arrays=[d.ReadAsArray() for d in datasets];a,b=arrays;both=np.isfinite(a)&np.isfinite(b)
 geo[axis]={'common_cells':int(both.sum()),'lost_cells':int((np.isfinite(a)&~np.isfinite(b)).sum()),
            'gained_cells':int((~np.isfinite(a)&np.isfinite(b)).sum()),
            'difference_quantiles_m':np.percentile(abs(b[both]-a[both]),[50,95,99,100]).tolist()}
 peaks=[gdal.Open(str(r/'geo/correlation_peak.tif')).ReadAsArray() for r in roots]
 snrs=[gdal.Open(str(r/'geo/snr.tif')).ReadAsArray() for r in roots]
 good=both&(peaks[0]>=.2)&(peaks[1]>=.2)&(snrs[0]>=2)&(snrs[1]>=2)
 geo[axis]['strong_common_cells']=int(good.sum());geo[axis]['strong_values_exactly_equal']=bool(np.array_equal(a[good],b[good]))
 limits=np.quantile(np.r_[a[good],b[good]],[.01,.99]);fig,axes=plt.subplots(3,1,figsize=(13,14),layout='constrained')
 for ax,data,title,bounds in zip(axes,[a,b,np.where(both,b-a,np.nan)],['Zero-prior TOTAL','Velocity-prior TOTAL','Velocity minus zero TOTAL'],[limits,limits,[-5,5]]):
  im=ax.imshow(data,cmap='RdBu_r',vmin=bounds[0],vmax=bounds[1]);ax.set_title(title+' '+axis+' displacement [m]');fig.colorbar(im,ax=ax)
 fig.savefig(OUT/f'{axis}-geo-comparison.png',dpi=130);plt.close(fig)
report['geocoded_comparison']=geo
write_json(OUT/'comparison.json',report)
print(json.dumps(report,indent=2))
