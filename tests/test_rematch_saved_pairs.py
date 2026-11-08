"""Metadata/control checks; full native rematching requires retained server inputs."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
import io,json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,h5py
from tools import rematch_saved_pairs as runner

class RematchChecks(unittest.TestCase):
 def cfg(self,root):
  return {'output':{'directory':str(root)},'matching':{'engine':'isce3','backend':'cpu','window':[64,256],'search':[10,40],'stride':[32,128],'raw_oversampling':2,'zoom':16,'deramp':0,'range_oversampling':2},'gross_offsets':{'mode':'zero','constant':{'azimuth_px':0,'range_px':0}},'stack':{'processing':'isce3_merge_first'}}
 def test_center_lattice(self):
  from glacier_offsets.matching_grid import build_grid
  grids=[build_grid((5000,6000),(5000,6000),w,[10,40],[32,128],first_center=[137.5,551.5]) for w in [[64,256],[128,512],[256,1024]]]
  for axis in [0,1]:
   center=[np.unique(g.centers[axis]) for g in grids]
   self.assertTrue(np.isin(center[2],center[0]).all());self.assertTrue(np.isin(center[2],center[1]).all())
 def test_settings_isolated_and_search_checked(self):
  cfg=self.cfg('/tmp/production');old=deepcopy(cfg)
  a=runner.parser().parse_args(['x','--pair','2018-01-01_2018-01-07','--output','/tmp/trial','--window','128','512'])
  got=runner.select_settings(cfg,a);self.assertEqual(got['matching']['window'],[128,512]);self.assertEqual(cfg,old)
  a.search=[2,12]
  with self.assertRaises(ValueError):runner.select_settings(cfg,a)
 def test_production_output_guard(self):
  for path in ['/tmp/production','/tmp/production/trial','/tmp']:
   with self.assertRaises(ValueError):runner.outside(Path(path),Path('/tmp/production'))
  runner.outside(Path('/tmp/trial'),Path('/tmp/production'))
 def test_comparison_joins_physical_centers(self):
  with tempfile.TemporaryDirectory() as temp:
   paths=[Path(temp)/'a.h5',Path(temp)/'b.h5']
   for p,x,offset in [(paths[0],[1.,2.,3.],0.),(paths[1],[3.,1.,4.],1.)]:
    with h5py.File(p,'w') as h:
     h['center_line']=np.zeros((1,3));h['center_sample']=np.array([x]);h['input_valid']=np.ones((1,3),bool);h['valid']=np.ones((1,3),bool);h['correlation_peak']=np.ones((1,3))*.2
     for axis in ['azimuth','range']:h['total/'+axis+'_px']=np.array([x])+offset
   result=runner.comparison(*paths);self.assertEqual(result['shared_centers'],2);self.assertEqual(result['offset_difference_on_common_input_support_px']['range']['median'],1.)
 def test_dry_run_no_writes(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp)/'production';out=Path(temp)/'trial';pair=root/'pairs/2018-01-01_2018-01-07';pair.mkdir(parents=True)
   cfg=self.cfg(root);m={'stack_pair':{'acquisition_dates':['2018-01-01','2018-01-07']},'matching':{'gross_offsets':cfg['gross_offsets']},'coreg':{'reference_grid':{'length':2000,'width':2000}}}
   (pair/'manifest.json').write_text(json.dumps(m));(pair/'radar').mkdir();(pair/'radar/offsets.h5').touch();(root/'geometry').mkdir();(root/'geometry/manifest.json').write_text('{}')
   for day in m['stack_pair']['acquisition_dates']:
    d=root/'acquisitions'/day;d.mkdir(parents=True);(d/'aligned.tif').touch();(d/'manifest.json').write_text(json.dumps({'slcs':{'burst':str(d/'aligned.tif')}}))
   before={p:p.stat().st_mtime_ns for p in root.rglob('*') if p.is_file()}
   args=runner.parser().parse_args(['x','--pair',pair.name,'--output',str(out),'--dry-run'])
   with patch('glacier_offsets.stack_config.load_status_config',return_value=cfg),redirect_stdout(io.StringIO()) as captured:
    self.assertEqual(runner.run(args),0)
   self.assertFalse(out.exists());self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before});self.assertEqual(json.loads(captured.getvalue())['missing_inputs'],[])

class VelocityRematchChecks(unittest.TestCase):
 def test_velocity_settings_accept_smaller_residual_search(self):
  with tempfile.TemporaryDirectory() as temp:
   path=Path(temp)/'velocity.tif';path.touch()
   cfg=RematchChecks().cfg(Path(temp)/'stack')
   cfg['gross_offsets']={'mode':'velocity','velocity':{'vx':{'path':str(path),'band':1},'vy':{'path':str(path),'band':1},'basis':'east_north','units':'m/day','interpolation':'bilinear','product':'synthetic','version':'1'}}
   args=runner.parser().parse_args(['x','--pair','2018-01-01_2018-01-07','--output',temp+'/trial','--search','4','12'])
   selected=runner.select_settings(cfg,args)
   self.assertEqual(selected['gross_offsets']['mode'],'velocity')
   self.assertEqual(selected['matching']['search'],[4,12])

 def test_retained_orbit_selected_without_source_eof_or_writes(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);(root/'geometry').mkdir()
   frame={'specification':{'inputs':{'orbit':{'path':str(root/'missing.EOF'),'sha256':'pinned'}}}}
   snapshot=root/'geometry/reference_orbit.h5';snapshot.write_bytes(b'kept')
   record=snapshot.with_suffix('.json');record.write_text(json.dumps({'source_sha256':'pinned','sha256':'snapshot-digest'}))
   before={p:p.stat().st_mtime_ns for p in (snapshot,record)}
   selected,hashes,paths=runner.orbit_inputs(root,frame)
   self.assertEqual(selected,snapshot);self.assertEqual(hashes[str(snapshot)],'snapshot-digest')
   self.assertEqual(set(paths),set(before));self.assertEqual(before,{p:p.stat().st_mtime_ns for p in before})
   record.write_text(json.dumps({'source_sha256':'wrong','sha256':'snapshot-digest'}))
   with self.assertRaisesRegex(ValueError,'pinned reference'):runner.orbit_inputs(root,frame)

 def test_old_stack_requires_pinned_eof_for_experiment_snapshot(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);source=root/'orbit.EOF'
   frame={'specification':{'inputs':{'orbit':{'path':str(source),'sha256':'pinned'}}}}
   selected,hashes,paths=runner.orbit_inputs(root,frame)
   self.assertIsNone(selected);self.assertEqual(paths,[source]);self.assertEqual(hashes,{str(source):'pinned'})
   self.assertFalse((root/'geometry').exists())

 def test_reference_adapters_bind_each_burst_grid_and_shared_orbit(self):
  grids=[{'prf':100.},{'prf':200.}];orbit=object()
  frame={'sources':[{'burst_id':str(i),'coreg':{'reference_grid':g}} for i,g in enumerate(grids)]}
  with patch('glacier_offsets.package_geometry.native_orbit',return_value=orbit),patch('glacier_offsets.stack_mosaic.radar_grid',side_effect=lambda g:SimpleNamespace(**g)):
   references=runner.velocity_references(frame,Path('snapshot.h5'))
   for i,g in enumerate(grids):
    self.assertIs(references[str(i)].native.orbit,orbit)
    self.assertEqual(references[str(i)].native.as_isce3_radargrid().prf,g['prf'])

 def test_reference_adapters_read_native_orbit_and_burst_radar_grids(self):
  with tempfile.TemporaryDirectory() as temp:
   snapshot=Path(temp)/'reference_orbit.h5'
   epoch=1577836800.
   t=np.arange(-120.,240.,10.)
   with h5py.File(snapshot,'w') as h:
    h['time']=epoch+t;h['position']=np.column_stack([np.full(len(t),7e6),7500*t,np.zeros(len(t))])
    h['velocity']=np.tile([0.,7500.,0.],(len(t),1))
   grid={'sensing_start':0.,'wavelength':.055,'prf':100.,'starting_range':800000.,'range_pixel_spacing':2.,'lookside':'right','length':100,'width':512,'ref_epoch':'2020-01-01T00:00:00'}
   sources=[{'burst_id':'first','geometry_reference_time':'2020-01-01T00:00:00','coreg':{'reference_grid':grid}},
            {'burst_id':'second','geometry_reference_time':'2020-01-01T00:00:05','coreg':{'reference_grid':grid|{'sensing_start':5.,'prf':200.}}}]
   before=snapshot.stat().st_mtime_ns
   references=runner.velocity_references({'sources':sources},snapshot)
   self.assertIs(references['first'].native.orbit,references['second'].native.orbit)
   self.assertEqual(references['first'].native.as_isce3_radargrid().prf,100.)
   self.assertEqual(references['second'].native.as_isce3_radargrid().sensing_start,5.)
   self.assertEqual(references['second'].native.as_isce3_radargrid().prf,200.)
   self.assertEqual(snapshot.stat().st_mtime_ns,before)

 def test_match_dispatch_regenerates_prior_and_passes_reference_bursts(self):
  cfg={'gross_offsets':{'mode':'velocity'}};manifest={'velocity_provenance':'old'};references={'burst':object()}
  with patch('glacier_offsets.ampcor.match') as match:
   runner.match_trial(cfg,Path('trial'),manifest,references)
   self.assertNotIn('velocity_provenance',manifest)
   self.assertIs(match.call_args.kwargs['native_pair'],references)
  with self.assertRaisesRegex(ValueError,'pinned reference orbit'):
   runner.match_trial(cfg,Path('trial'),{})
  with patch('glacier_offsets.ampcor.match') as match:
   runner.match_trial({'gross_offsets':{'mode':'zero'}},Path('trial'),{'velocity_provenance':'old'})
   self.assertEqual(match.call_args.kwargs,{})


if __name__=='__main__':unittest.main()
