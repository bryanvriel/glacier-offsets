"""Audit official VRT precedence and compare saved unspliced burst views.

This isolates merge policy using existing reference rectangles. It does not
reproduce ISCE2 resampling or its stack-common validity computation.
"""
import argparse
import importlib.util
import json
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
from osgeo import gdal

gdal.UseExceptions()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('experiment',type=Path)
parser.add_argument('--isce2',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
experiment=args.experiment.resolve()
pair=Path(json.loads((experiment/'experiment.json').read_text())['specification']['pair'])
out=args.output.resolve()
stack=pair.parent.parent
if out.exists() or stack == out or stack in out.parents or out in stack.parents:
 raise ValueError('Use a new directory outside the production stack')
out.mkdir(parents=True)
isce2=args.isce2.resolve()
m=json.loads((pair/'manifest.json').read_text())
entries=m['assembly']['sources']
grid=m['coreg']['reference_grid']
shape=(grid['length'],grid['width'])
spec=importlib.util.spec_from_file_location('official_vrt',isce2/'contrib/stack/topsStack/VRTManager.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
b=module.VRTConstructor(*shape)
# Integer placements are deliberately held fixed: isolate the merge policy,
# using our reference valid rectangles rather than inventing ISCE2 stack XMLs.
b.setReferenceTime(datetime(2000,1,1));b.setReferenceRange(0)
b.setTimeSpacing(1);b.setRangeSpacing(1);b.setDataType('FLOAT')
b.initVRT();b.initBand(1)
for i,e in enumerate(entries,1):
 g=e['coreg']['reference_grid'];y0,y1,x0,x1=e['valid_window'];dy,dx=e['offset']
 path=out/f'index-{i}.tif'
 ds=gdal.GetDriverByName('GTiff').Create(str(path),g['width'],g['length'],1,gdal.GDT_Byte,options=['TILED=YES','COMPRESS=DEFLATE'])
 ds.GetRasterBand(1).Fill(i);ds=None
 ds=gdal.Translate(str(path)+'.vrt',str(path),format='VRT');ds=None
 burst=SimpleNamespace(numberOfLines=g['length'],numberOfSamples=g['width'],
  firstValidLine=y0,firstValidSample=x0,numValidLines=y1-y0,numValidSamples=x1-x0,
  sensingStart=b.tref+timedelta(seconds=dy),startingRange=dx)
 b.addSwath(SimpleNamespace(prod=SimpleNamespace(bursts=[burst])),[str(path)],validOnly=True)
b.finishBand();b.finishVRT();b.writeVRT(str(out/'official-owner.vrt'))
actual=gdal.Open(str(out/'official-owner.vrt'))
current=gdal.Open(m['coreg']['source_index'])
counts={'covered_pixels':0,'different_owner_pixels':0,'coverage_disagreements':0,'numpy_precedence_disagreements':0}
for row in range(0,shape[0],64):
 h=min(64,shape[0]-row)
 official=actual.ReadAsArray(0,row,shape[1],h).astype('uint16')
 numpy=np.zeros((h,shape[1]),'uint16')
 for i,e in enumerate(entries,1):
  y0,y1,x0,x1=e['valid_window'];dy,dx=e['offset']
  lo,hi=max(row,y0+dy),min(row+h,y1+dy)
  if lo<hi:numpy[lo-row:hi-row,x0+dx:x1+dx]=i
 ours=current.ReadAsArray(0,row,shape[1],h)
 counts['covered_pixels']+=int((official>0).sum())
 counts['different_owner_pixels']+=int(((official>0)&(ours>0)&(official!=ours)).sum())
 counts['coverage_disagreements']+=int(((official>0)!=(ours>0)).sum())
 counts['numpy_precedence_disagreements']+=int((numpy!=official).sum())
with h5py.File(pair/'radar/offsets.h5') as h:
 cy,cx=h['center_line'][:],h['center_sample'][:]
 owner=h['source_burst_index'][:]
 yi,xi=np.rint(cy).astype(int),np.rint(cx).astype(int)
 official=np.zeros_like(owner)
 for r in range(owner.shape[0]):
  line=actual.ReadAsArray(0,int(yi[r,0]),shape[1],1)[0]
  official[r]=line[xi[r]]
 counts['changed_matching_center_owners']=int(((owner>0)&(official>0)&(owner!=official)).sum())

reports={}
for mode in ('complex','magnitude'):
 sources=[];params=m['matching']['matching'];wy,wx=params['window'];sy,sx=params['stride']
 ybase=cy[0,0]-(wy-1)/2;xbase=cx[0,0]-(wx-1)/2
 for i,e in enumerate(entries,1):
  p=experiment/f'unspliced_{mode}/radar/native'/e['burst_id']/'result.npz'
  with np.load(p) as h:
   d={k:h[k] for k in ('valid','total','correlation_peak')};eff=json.loads(str(h['effective_json']))
  dy,dx=e['offset'];ny,nx=d['valid'].shape
  rr=(eff['referenceStartPixelDownStatic']+dy+np.arange(ny)*sy-ybase)/sy
  cc=(eff['referenceStartPixelAcrossStatic']+dx+np.arange(nx)*sx-xbase)/sx
  assert np.allclose(rr,np.rint(rr)) and np.allclose(cc,np.rint(cc))
  d.update(rows=np.rint(rr).astype(int),cols=np.rint(cc).astype(int),id=e['burst_id'])
  sources.append(d)
 records=[]
 for a,c in combinations(sources,2):
  rows=np.intersect1d(a['rows'],c['rows']);cols=np.intersect1d(a['cols'],c['cols'])
  if not len(rows) or not len(cols):continue
  sections=[np.ix_(np.searchsorted(s['rows'],rows),np.searchsorted(s['cols'],cols)) for s in (a,c)]
  aa,cc=[s['total'][section] for s,section in zip((a,c),sections)]
  entry={'sources':[a['id'],c['id']], 'kind':'along_track' if a['id'][-3:]==c['id'][-3:] else 'cross_swath', 'thresholds':{}}
  for threshold in (0,.3,.5):
   good=np.ones(aa.shape[:2],bool)
   for s,section in zip((a,c),sections):good &= s['valid'][section]&(s['correlation_peak'][section]>=threshold)
   if not good.any():continue
   delta=(cc-aa)[good]
   entry['thresholds'][str(threshold)]={'n':int(good.sum()),'axes':{
    axis:{'median_signed':float(np.median(delta[:,k])), 'median_absolute':float(np.median(abs(delta[:,k]))),
    'p90_absolute':float(np.quantile(abs(delta[:,k]),.9)), 'p99_absolute':float(np.quantile(abs(delta[:,k]),.99))}
    for k,axis in enumerate(('azimuth','range'))}}
  records.append(entry)
 reports[mode]=records
import hashlib
import subprocess
provenance={str(isce2/path):hashlib.sha256((isce2/path).read_bytes()).hexdigest() for path in ('contrib/stack/topsStack/mergeBursts.py','contrib/stack/topsStack/VRTManager.py','contrib/stack/topsStack/Stack.py')}
report={'pair':pair.name, 'official_source_sha256':provenance,
 'isce2_git_commit':subprocess.check_output(['git','-C',str(isce2),'rev-parse','HEAD'],text=True).strip(),'scope':'Direct execution of official VRTManager using our fixed placements and reference rectangles; not original ISCE2 coregistered products or stack-common XML validity.',
 'owner_comparison':counts,'official_cross_swath_switches_from_reference_rectangles':sorted({e['offset'][1]+e['valid_window'][2] for e in entries if e['burst_id'].endswith('_iw2')}),
 'current_cross_swath_cuts':json.loads((stack/'geometry/manifest.json').read_text())['ownership']['cuts'],
 'overlap_differences_without_agreement_selection':reports}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(counts,indent=2))
for mode,entries in reports.items():
 print(mode)
 for e in entries:
  if '0.5' in e['thresholds']:
   d=e['thresholds']['0.5']
   print(e['kind'],e['sources'],d['n'],d['axes'])
