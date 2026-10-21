"""Controlled native replays at a few reference windows; no original products overwritten."""
import json
from pathlib import Path
import h5py,numpy as np
from glacier_offsets.ampcor import native_match
from glacier_offsets.config import load_config
from glacier_offsets.matching_grid import MatchingGrid
from glacier_offsets.products import write_json

out=Path(__file__).parent
cfg=load_config('configs/pair.test.velocity.yaml');root=Path(cfg['output']['directory'])/cfg['inputs']['burst_id']
zroot=Path(load_config('configs/pair.test.yaml')['output']['directory'])/cfg['inputs']['burst_id']
m=json.loads((root/'manifest.json').read_text())
with h5py.File(root/'radar/offsets.h5') as v,h5py.File(zroot/'radar/offsets.h5') as z:
 azrg=lambda h,f:np.stack([h[f+'/'+a+'_px'][:] for a in ['azimuth','range']],-1)
 a,b,g=azrg(z,'total'),azrg(v,'total'),azrg(v,'gross');delta=np.max(abs(b-a),axis=-1)
 common=z['valid'][:]&v['valid'][:]
 strong=common&(z['correlation_peak'][:]>=.2)&(v['correlation_peak'][:]>=.2)&(z['snr'][:]>=2)&(v['snr'][:]>=2)
 changed=common&(delta>.1);nonzero=np.any(g!=0,axis=-1)
 groups={'largest_difference':common,'small_difference':changed&(delta<.5),'nonzero_strong':strong&nonzero,'zero_strong':strong&~nonzero}
 selected=[]
 for name,mask in groups.items():
  if not mask.any():continue
  scores=np.where(mask,delta,-np.inf);i,j=np.unravel_index(np.argmax(scores),mask.shape)
  selected.append((name,i,j))
 reports=[]
 for name,i,j in selected:
  p=cfg['matching'];start=[int(v['center_line'][i,j]-(p['window'][0]-1)/2),int(v['center_sample'][i,j]-(p['window'][1]-1)/2)]
  grid=MatchingGrid(np.array([start[0]],dtype='int32'),np.array([start[1]],dtype='int32'),tuple(p['window']),tuple(p['search']),tuple(p['stride']))
  rr=[]
  for label,gross in [('zero',[0,0]),('constant',g[i,j].astype(int).tolist()),('varying',g[i,j].astype(int).tolist())]:
   kwargs={'_grid':grid}
   if label=='varying':kwargs['_gross']=np.array([[gross]],dtype='int32')
   c={'gross_offsets':{'mode':'constant','constant':{'azimuth_px':gross[0],'range_px':gross[1]}}}
   r=native_match(m['coreg']['slcs']['reference'],m['coreg']['slcs']['secondary'],out/'replays'/name/label,p,c,'cpu',**kwargs)
   rr.append(r)
  np.testing.assert_allclose(rr[1]['total'],rr[2]['total'],atol=1/64)
  np.testing.assert_allclose(rr[0]['total'][0,0],a[i,j],atol=1/64)
  np.testing.assert_allclose(rr[2]['total'][0,0],b[i,j],atol=1/64)
  reports.append({'name':name,'row':int(i),'col':int(j),'gross_azimuth_range':g[i,j].tolist(),
   'original_zero':a[i,j].tolist(),'original_velocity':b[i,j].tolist(),
   'replayed_total_zero_constant_varying':[r['total'][0,0].tolist() for r in rr],
   'peak_zero_constant_varying':[float(r['correlation_peak'][0,0]) for r in rr],
   'snr_zero_constant_varying':[float(r['snr'][0,0]) for r in rr]})
write_json(out/'replay.json',reports);print(json.dumps(reports,indent=2))
