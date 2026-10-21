from contextlib import chdir
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import yaml
from glacier_offsets.config import load_config, gross_values, load_defaults, config_overrides


class ConfigTests(unittest.TestCase):
    def full_config(self, directory):
        p=directory/'input';p.touch()
        return {'inputs':{k:str(p) for k in ['reference_safe','secondary_safe','reference_orbit','secondary_orbit','dem']}
                | {'burst_id':'t038_080975_iw2'},'output':{'directory':str(directory/'work')},
                'geocoding':{'epsg':3031}}

    def test_relative_paths_work_from_another_cwd_and_after_sync(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            project = root/'laptop'
            (project/'configs').mkdir(parents=True)
            (project/'data').mkdir()
            cfg = self.full_config(project/'data')
            cfg['inputs'].update({key: '../data/input' for key in (
                'reference_safe', 'secondary_safe', 'reference_orbit', 'secondary_orbit', 'dem')})
            cfg['output']['directory'] = '../work/new_pair'
            pair = project/'configs/pair.yaml'
            pair.write_text(yaml.safe_dump(cfg))
            shutil.copytree(project, root/'server')
            with chdir(root):
                for machine in ('laptop', 'server'):
                    loaded = load_config(Path(machine)/'configs/pair.yaml')
                    expected = (root/machine/'data/input').resolve()
                    self.assertTrue(all(loaded['inputs'][key] == str(expected)
                                        for key in ('reference_safe', 'secondary_safe',
                                                    'reference_orbit', 'secondary_orbit', 'dem')))
                    self.assertEqual(loaded['output']['directory'], str((root/machine/'work/new_pair').resolve()))
                    self.assertFalse((root/machine/'work').exists())
                    self.assertEqual(load_config(Path(machine)/'configs/pair.yaml', discovery=True), loaded)

    def test_inherited_paths_belong_to_the_yaml_that_declares_them(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'base').mkdir()
            (root/'child').mkdir()
            base = self.full_config(root/'base')
            base['inputs'].update({key: 'input' for key in (
                'reference_safe', 'secondary_safe', 'reference_orbit', 'secondary_orbit', 'dem')})
            base['output']['directory'] = 'work'
            (root/'base/pair.yaml').write_text(yaml.safe_dump(base))
            (root/'child/dem.tif').touch()
            pair = root/'child/pair.yaml'
            pair.write_text(yaml.safe_dump({'extends': '../base/pair.yaml',
                'inputs': {'dem': 'dem.tif'}, 'matching': {'backend': 'cpu'}}))
            cfg = load_config(pair)
            self.assertEqual(cfg['inputs']['reference_safe'], str((root/'base/input').resolve()))
            self.assertEqual(cfg['inputs']['dem'], str((root/'child/dem.tif').resolve()))
            self.assertEqual(cfg['output']['directory'], str((root/'base/work').resolve()))

    def test_home_paths_and_invalid_path_values(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cfg = self.full_config(root)
            cfg['inputs']['dem'] = '~/input'
            cfg['output']['directory'] = '~/work'
            pair = root/'pair.yaml'
            pair.write_text(yaml.safe_dump(cfg))
            with patch.dict('os.environ', {'HOME': str(root)}):
                loaded = load_config(pair)
            self.assertEqual(loaded['inputs']['dem'], str(root/'input'))
            self.assertEqual(loaded['output']['directory'], str(root/'work'))
            for value in ('', False, 123, [], {}):
                cfg['inputs']['dem'] = value
                pair.write_text(yaml.safe_dump(cfg))
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'inputs.dem'):
                    load_config(pair)
            cfg['inputs']['dem'] = 'missing.tif'
            pair.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, 'inputs.dem.*missing.tif'):
                load_config(pair)

    def test_velocity_paths_resolve_before_validation_and_inheritance(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root/'nested').mkdir()
            base = self.full_config(root)
            base['gross_offsets'] = {'mode': 'velocity', 'velocity': {
                'vx': {'path': 'input', 'variable': 'VX'},
                'vy': {'path': 'input', 'variable': 'VY'},
                'basis': 'east_north', 'units': 'm/s', 'interpolation': 'nearest',
                'product': 'test', 'version': '1'}}
            (root/'base.yaml').write_text(yaml.safe_dump(base))
            (root/'nested/velocity.tif').touch()
            pair = root/'nested/pair.yaml'
            pair.write_text(yaml.safe_dump({'extends': '../base.yaml', 'gross_offsets': {
                'velocity': {'vy': {'path': 'velocity.tif', 'variable': 'north'}}}}))
            velocity = load_config(pair)['gross_offsets']['velocity']
            self.assertEqual(velocity['vx']['path'], str((root/'input').resolve()))
            self.assertEqual(velocity['vy']['path'], str((root/'nested/velocity.tif').resolve()))
            self.assertEqual(velocity['vy']['variable'], 'north')

    def test_recursive_inheritance_relative_to_yaml_and_list_replacement(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'nested').mkdir()
            base=self.full_config(d)
            base['matching']={'window':[80,320],'snr_threshold':2.}
            (d/'base.yaml').write_text(yaml.safe_dump(base))
            child=d/'nested/pair.yaml'
            child.write_text(yaml.safe_dump({'extends':'../base.yaml',
                'matching':{'correlation_threshold':.1,'stride':[48,192]},'inputs':{'subswaths':[2]}}))
            cfg=load_config(child)
            self.assertNotIn('extends',cfg)
            self.assertEqual(cfg['matching']['window'],[80,320])
            self.assertEqual(cfg['matching']['stride'],[48,192])
            self.assertEqual(cfg['matching']['snr_threshold'],2.)
            self.assertEqual(cfg['matching']['correlation_threshold'],.1)
            self.assertEqual(cfg['inputs']['subswaths'],[2])
            self.assertEqual(cfg['geocoding']['mask'],'input')
            self.assertEqual(cfg['geocoding']['spacing_m'],200)
            # Resolved configs are independent; no mutation leaks between loads.
            cfg['matching']['window'][0]=999
            self.assertEqual(load_config(child)['matching']['window'],[80,320])
            compact={'extends':'defaults'}|config_overrides(load_config(child))
            (d/'compact.yaml').write_text(yaml.safe_dump(compact))
            self.assertEqual(load_config(d/'compact.yaml'),load_config(child))

    def test_live_defaults_changes_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);defaults=load_defaults()
            defaults['matching']['snr_threshold']=3.
            default_path=d/'defaults.yaml';default_path.write_text(yaml.safe_dump(defaults))
            pair=d/'pair.yaml';cfg=self.full_config(d);pair.write_text(yaml.safe_dump(cfg))
            with patch('glacier_offsets.config.DEFAULT_CONFIG_PATH',default_path):
                self.assertEqual(load_config(pair)['matching']['snr_threshold'],3.)
                defaults['matching']['snr_threshold']=4.
                default_path.write_text(yaml.safe_dump(defaults))
                self.assertEqual(load_config(pair)['matching']['snr_threshold'],4.)
                cfg['matching']={'snr_threshold':2.};pair.write_text(yaml.safe_dump(cfg))
                self.assertEqual(load_config(pair)['matching']['snr_threshold'],2.)

    def test_inheritance_errors(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            a=d/'a.yaml';b=d/'b.yaml'
            a.write_text('extends: b.yaml\n');b.write_text('extends: a.yaml\n')
            with self.assertRaisesRegex(ValueError,'inheritance cycle'):load_config(a)
            for text,reason in [('extends: missing.yaml','Cannot read'),('extends: [a.yaml]','extends must'),
                                ('extends: null','extends must'),('[]','YAML mapping'),
                                ('matching: null','matching must')]:
                if text.startswith('matching:'):
                    cfg=self.full_config(d);cfg['matching']=None;a.write_text(yaml.safe_dump(cfg))
                else:a.write_text(text+'\n')
                with self.subTest(text=text),self.assertRaisesRegex(ValueError,reason):load_config(a)

    def test_gross_modes(self):
        self.assertEqual(gross_values({}), (0,0))
        self.assertEqual(gross_values({'gross_offsets':{'mode':'constant','constant':{'azimuth_px':-4,'range_px':7}}}), (-4,7))
        for v in [True,1.2,None]:
            with self.assertRaises(ValueError):
                gross_values({'gross_offsets':{'mode':'constant','constant':{'azimuth_px':v,'range_px':0}}})
        with self.assertRaisesRegex(ValueError,'reference geometry'):
            gross_values({'gross_offsets':{'mode':'velocity'}})

    def test_discovery_minimum(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input'; p.touch()
            cfg={'inputs':{k:str(p) for k in ['reference_safe','secondary_safe','reference_orbit','secondary_orbit']}}
            f=Path(d)/'pair.yaml'; f.write_text(yaml.safe_dump(cfg))
            self.assertNotIn('burst_id',load_config(f,discovery=True)['inputs'])
            with self.assertRaisesRegex(ValueError,'dem'):
                load_config(f)
            cfg['inputs'].update(dem=str(p),burst_id='t038_080975_iw2')
            cfg['output']={'directory':str(Path(d)/'work')}
            f.write_text(yaml.safe_dump(cfg))
            defaults=load_config(f)['matching']
            self.assertEqual(defaults['snr_threshold'],0.0)
            self.assertEqual(defaults['correlation_threshold'],0.0)
            cfg['matching']={'snr_threshold':2.0,'correlation_threshold':0.1}
            f.write_text(yaml.safe_dump(cfg))
            explicit=load_config(f)['matching']
            self.assertEqual(explicit['snr_threshold'],2.0)
            self.assertEqual(explicit['correlation_threshold'],0.1)
