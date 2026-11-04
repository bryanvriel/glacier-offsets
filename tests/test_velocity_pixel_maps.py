"""Check mapping, phase convention and units independently of rendering."""
from contextlib import chdir
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import h5netcdf
import numpy as np
from numpy.testing import assert_allclose,assert_array_equal

from glacier_offsets.velocity_io import digest
from glacier_offsets.velocity_pixel_maps import read_maps,plot_maps
from glacier_offsets.cli import main as cli_main


def source_fixture(root):
    root.mkdir(exist_ok=True)
    index=np.array([[0,0,1,-1],[0,1,1,2],[2,2,-1,0]])
    with h5py.File(root/'observations.h5','w') as h:
        h['x']=[100,300,500,700];h['y']=[900,700,500]
        h['map_source_index']=index;h['source_id']=[10,20,30]
    with h5py.File(root/'source_coefficients.h5','w') as h:
        h.attrs['status']='complete';h['source_id']=[10,20,30]
        h['standard_coverage']=[True,False,True];h['mean_flow_speed']=[1.,.05,1.]
        h['year_direction_change_deg']=[0.,0.,40.];h['azimuth_flow_projection_by_year']=np.full((3,2),.6)
        for name in ('residual_vertical','azimuth_direct'):
            g=h.create_group(name);g['estimable']=[True,True,False]
            g['amplitude']=[.1,.2,np.nan];g['phase_deg']=[-120,45,np.nan]
            g['amplitude_sigma']=[.01,.02,np.nan];g['phase_sigma_deg']=[40,10,np.nan]
            coefficients=np.zeros((3,12));coefficients[:,6:8]=[.3,.4];coefficients[:,8:10]=[.12,.05]
            g['coefficients']=coefficients
    return index


class MapTests(unittest.TestCase):
    def test_yaml_relative_output_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();data=root/'data';source_fixture(data)
            config=root/'experiment.yaml'
            # Input stacks need not remain present; plotting uses only saved output.
            # Edits to the YAML's display settings must not relabel existing fits.
            config.write_text('input_directory: absent-cache\noutput: data\ndisplay:\n  min_speed_m_day: 2\n  proxy_min_speed_m_day: 2\n')
            elsewhere=root/'elsewhere';elsewhere.mkdir()
            with chdir(elsewhere):
                actual=read_maps(config);expected=read_maps(data)
            assert_array_equal(actual.selected,expected.selected)
            assert_allclose(actual.amplitude,expected.amplitude,equal_nan=True)
            self.assertIn(config,actual.files)

    def test_maps_cli_accepts_yaml_and_records_resolved_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();data=root/'data';source_fixture(data)
            config=root/'experiment.yml';config.write_text('input_directory: absent-cache\noutput: data\n')
            elsewhere=root/'elsewhere';elsewhere.mkdir()
            with chdir(elsewhere):
                self.assertEqual(cli_main(['velocity-pixels','maps',str(config),'--dpi','40']),0)
            settings=json.loads((data/'amplitude_phase_maps/plot-settings.json').read_text())
            self.assertEqual(settings['input_directory'],str(data))
            self.assertEqual(settings['config_file'],str(config))
            self.assertEqual(settings['input_sha256'][str(config)],digest(config))

    def test_source_mapping_masks_and_copied_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);index=source_fixture(root)
            maps=read_maps(root)
            assert_array_equal(maps.selected,index==0)
            assert_array_equal(maps.footprint,index>=0)
            assert_allclose(maps.amplitude[maps.selected],.1)
            assert_allclose(maps.phase[maps.selected],-120)
            assert_array_equal(read_maps(root,mask='estimable').selected,(index==0)|(index==1))
            self.assertFalse(read_maps(root,mask='phase-quality').selected.any())

    def test_vertical_height_units_and_phase_from_saved_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_fixture(root)
            m2=read_maps(root,'vertical-m2');o1=read_maps(root,'vertical-o1')
            assert_allclose(m2.amplitude[m2.selected],.5)
            assert_allclose(m2.phase[m2.selected],np.degrees(np.arctan2(.4,.3)))
            assert_allclose(o1.amplitude[o1.selected],.13)

    def test_netcdf_only_input_and_coordinate_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);amplitude=np.arange(6).reshape(2,3)*.1
            with h5netcdf.File(root/'coefficient_maps.nc','w') as h:
                h.dimensions={'y':2,'x':3};h.attrs.update(status='complete',phase_origin='2000-01-01T00:00:00Z')
                h.create_variable('x',('x',),data=[500.,300.,100.])
                h.create_variable('y',('y',),data=[500.,700.])
                for name,value in {'source_observation_index':np.array([[10,20,-1],[30,40,50]]),
                    'fit_estimable':np.ones((2,3),bool),'along_display_mask':np.ones((2,3),bool),
                    'msf_along_amplitude':amplitude,'msf_along_phase_deg':np.full((2,3),-120.)}.items():
                    h.create_variable(name,('y','x'),data=np.asarray(value,dtype='f8'))
            maps=read_maps(root)
            assert_array_equal(maps.x,[100,300,500]);assert_array_equal(maps.y,[700,500])
            assert_array_equal(maps.amplitude,amplitude[::-1,::-1])
            self.assertFalse(maps.selected[1,0]);self.assertEqual(maps.selected.sum(),5)

    def test_rendering_from_another_directory_and_saved_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_fixture(root);elsewhere=root/'elsewhere';elsewhere.mkdir()
            before={name:digest(root/name) for name in ('observations.h5','source_coefficients.h5')}
            with chdir(elsewhere):
                records=plot_maps(root,component='vertical-m2',phase_range='positive',amplitude_max=.7,dpi=50)
            self.assertEqual(records[0]['amplitude_units'],'m')
            self.assertEqual(records[0]['phase_limits_deg'],(0,360))
            self.assertEqual(records[0]['unique_sources_displayed'],1)
            for path in records[0]['files']:self.assertGreater(Path(path).stat().st_size,1000)
            settings=json.loads((root/'amplitude_phase_maps/plot-settings.json').read_text())
            self.assertEqual(settings['maps'][0]['amplitude_limits'],[0,.7])
            self.assertEqual(before,{name:digest(root/name) for name in before})

    def test_invalid_options_and_changed_fits_fail_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source_fixture(root)
            for options in ({'amplitude_max':0},{'dpi':0},{'mask':'unknown'},{'component':'unknown'}):
                with self.subTest(options=options),self.assertRaises(ValueError):plot_maps(root,**options)
            self.assertFalse((root/'amplitude_phase_maps').exists())
            completion={'output_sha256':digest(root/'source_coefficients.h5')}
            (root/'fit-completion.json').write_text(json.dumps(completion))
            with h5py.File(root/'source_coefficients.h5','r+') as h:h['residual_vertical/amplitude'][0]=.5
            with self.assertRaisesRegex(ValueError,'fits changed'):read_maps(root)


if __name__=='__main__':unittest.main()
