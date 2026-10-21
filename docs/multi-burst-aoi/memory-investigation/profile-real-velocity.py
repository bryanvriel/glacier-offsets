from pathlib import Path
import gc,json,resource,sys,tempfile,collections
import h5py,numpy as np
from glacier_offsets.ampcor import native_match
from glacier_offsets.matching_grid import MatchingGrid

def main():
 key='t038_080975_iw2'
 root=Path('/Users/briel/data/insar/glacier_offsets/work/multi-burst-validation/velocity/batch')/key
 m=json.loads((root/'manifest.json').read_text())
 cfg=__import__('yaml').safe_load((root.parent.parent/'batch.yaml').read_text())
 with h5py.File(root/'radar/offsets.h5') as h:
  line,sample=h['center_line'][:],h['center_sample'][:]
  window,search,stride=[tuple(h.attrs[k]) for k in ['window','search','stride']]
  grid=MatchingGrid((line[:,0]-(window[0]-1)/2).astype('int32'),(sample[0]-(window[1]-1)/2).astype('int32'),window,search,stride)
  prior={'grid':grid,'integer_px':np.stack([h['prior/integer_px/'+axis][:] for axis in ['azimuth','range']],axis=-1),
         'valid':h['prior/valid'][:],'rejection_flags':h['prior/rejection_flags'][:]}
 with tempfile.TemporaryDirectory() as d:
  result=native_match(m['coreg']['slcs']['reference'],m['coreg']['slcs']['secondary'],Path(d)/'native',cfg['matching'],cfg,'cpu',prior=prior)
  checks={}
  np.savez_compressed('/tmp/velocity-memory-result.npz',**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
  with h5py.File(root/'radar/offsets.h5') as h:
   for name in ['input_valid','finite','search_boundary','valid','snr','correlation_peak','covariance']:
    x,y=result[name],h[name][:]
    exact=np.array_equal(x,y,equal_nan=True)
    close=np.allclose(x,y,rtol=1e-5,atol=1e-6,equal_nan=True)
    same=(x==y)|(np.isnan(x)&np.isnan(y))
    checks[name]={'exact':exact,'within_tolerance':close,'different_values':int(np.count_nonzero(~same))}
    print('CHECK',name,checks[name],flush=True)
   for field in ['gross','residual','total']:
    for i,axis in enumerate(['azimuth','range']):
     name=field+'/'+axis+'_px';x,y=result[field][...,i],h[name][:];same=(x==y)|(np.isnan(x)&np.isnan(y));accepted=h['valid'][:];checks[name]={'exact':bool(same.all()),'different_values':int(np.count_nonzero(~same)),'different_accepted':int(np.count_nonzero(~same&accepted))};print('CHECK',name,checks[name],flush=True)
  repeat=native_match(m['coreg']['slcs']['reference'],m['coreg']['slcs']['secondary'],Path(d)/'repeat',cfg['matching'],cfg,'cpu',prior=prior)
  repeat_checks={}
  for name in ['gross','residual','total','input_valid','finite','search_boundary','valid','snr','correlation_peak','covariance']:
   exact=np.array_equal(result[name],repeat[name],equal_nan=True)
   repeat_checks[name]=exact
   print('NEW_REPEAT',name,'exact',exact,flush=True)
   assert exact,name
  segments=result['effective_parameters']['segments']+repeat['effective_parameters']['segments']
  pids=collections.Counter(s['worker_pid'] for s in segments)
  summary={'burst_id':key,'segments':len(segments),'workers':len(pids),'max_calls_per_worker':max(pids.values()),
    'parent_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
    'max_worker_peak_rss_mib':max(s['worker_peak_rss_bytes'] for s in segments)/1024**2,'legacy_comparisons':checks,'repeat_exact':repeat_checks}
  print('MEMORY_RESULT',json.dumps(summary,sort_keys=True),flush=True)
  Path('/Users/briel/data/insar/glacier_offsets/docs/multi-burst-aoi/memory-investigation/velocity-rematch.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
