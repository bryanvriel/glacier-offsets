"""Metadata/control checks; full native rematching requires retained server inputs."""
from pathlib import Path
from copy import deepcopy
from contextlib import redirect_stdout
import io,json,tempfile,unittest
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

if __name__=='__main__':unittest.main()
