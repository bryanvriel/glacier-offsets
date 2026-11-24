"""Exact joins and exploratory joint Sentinel-1 fitting."""
from contextlib import ExitStack
from datetime import timedelta
import json

import h5netcdf
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from pyproj import CRS

from glacier_offsets.velocity_collection import year_start
from glacier_offsets.velocity_io import variable_hashes
from glacier_offsets.velocity_multitrack import JointLayout, common_grid, covariances_for, ordered_tracks, overlap_axis, pair_rows
from glacier_offsets.velocity_multitrack_pilot import acquisition_folds
from glacier_offsets.velocity_pixel_general import design, predict, solve
from glacier_offsets.velocity_pixel_maps import read_maps
from glacier_offsets.velocity_solid_earth import Interrupted
from glacier_offsets.velocity_workflow import run_workflow

from velocity_synthetic import GRID, Truth, config_text, pair_list, write_package


def test_exact_overlap_and_missing_group():
    overlap, slices, spacing = overlap_axis([[-400., -200., 0., 200.], [-200., 0., 200., 400.]])
    assert_array_equal(overlap, [-200., 0., 200.])
    assert slices == [slice(1, 4), slice(0, 3)]
    assert spacing == 200.
    with pytest.raises(ValueError, match='same lattice'):
        overlap_axis([[0., 200.], [100., 300.]])
    with pytest.raises(ValueError, match='no common extent'):
        overlap_axis([[0., 200.], [400., 600.]])
    with pytest.raises(ValueError, match='S1:T065:2018'):
        covariances_for([{'noise_group': 'S1:T065:2018'}], {37: {'covariance_ra_m2_by_year': {'2018': np.eye(2)}}})


def test_repeated_native_chips_fit_once_per_tuple(tmp_path):
    with ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(tmp_path/f'{i}.h5','w')) for i in range(2)]
        for i, h in enumerate(handles):
            h['x'], h['y'], h['crs'] = [-200.,0.,200.], [200.,0.], np.int32(0)
            h.attrs['crs_wkt'] = CRS.from_epsg(3031).to_wkt()
            h['map_source_index'] = [[0,0,1], [1,1,-1]]
            h['source_id'] = [1,2]
            h['longitude'], h['latitude'] = [0.+i,1.+i], [-77.,-77.]
        grid = common_grid(handles)
        assert_array_equal(grid.back, [[0,0,1],[1,1,-1]])
        assert_array_equal(grid.native_id, [[1,1],[2,2]])
        assert (grid.chip_separation_m > 0).all()


def test_namespaced_pairs_and_exact_acquisition_purge():
    start = year_start(2018)
    common = {'pair_id':'a_b', 'calendar_early_day':start, 'calendar_late_day':start+6, 'year':2018,
              'earlier_spacecraft':'S1A', 'later_spacecraft':'S1B', 'order':1}
    next_pair = common | {'pair_id':'b_c','calendar_early_day':start+6,'calendar_late_day':start+12,
                          'earlier_spacecraft':'S1B','later_spacecraft':'S1A', 'order':-1}
    rows = pair_rows([(37,'a','DESCENDING'), (65,'b','DESCENDING')],
                     [{'pairs':[common,next_pair]}, {'pairs':[common]}])
    assert len({r['pair_id'] for r in rows}) == 3
    assert rows[0]['late_acquisition'] == rows[1]['early_acquisition']
    assert rows[0]['early_acquisition'] != rows[2]['early_acquisition']
    train,test = next(acquisition_folds(rows, width=6))
    assert_array_equal(test, [True,False,True])
    assert_array_equal(train, [False,False,False])


def test_exact_joint_injection_and_rank_gate():
    years = [2017,2018]
    first = year_start(2017)+182
    early = np.concatenate([first+6*np.arange(75), first+.11+6*np.arange(75)])[None]
    late = early+6.
    track = np.repeat([37,65],75)
    year = ((early[0]+3) >= year_start(2018)).astype(int)
    order = np.tile(np.where(np.arange(75)%2 == 0, 1., -1.),2)
    rows = [dict(track=t,year=years[k],order=p) for t,k,p in zip(track,year,order)]
    layout = JointLayout(years,rows)
    order_indices = [layout.joint_names.index(name) for name in
                     ('order_T037_2017','order_T037_2018','order_T065_2017','order_T065_2018')]
    g = np.array([[[[-.55,.12,-.83],[.11,.99,.003]]],
                  [[[-.48,-.30,-.84],[-.48,.87,-.002]]]])
    axis = np.array([[.8,.6]])
    field = dict(early=early,late=late,calendar_early=early[0],calendar_late=late[0],track=track,year_index=year,
                 order=order,geometry_by_file=g,file_index=(track==65).astype(int),
                 data=np.zeros((*early.shape,2)))
    cov = {'by_pair':np.broadcast_to(np.array([[.04,.01],[.01,.09]]),(len(track),2,2))}
    wanted = np.array([[.8,.6,1.2,.9,.1,.2,-.4,.3,.04,-.03,.2,.07,.08,-.06]])
    whitened, _, _, _ = design(field,layout,axis,cov)
    for t,j in enumerate(order_indices):
        mask = (track==[37,37,65,65][t]) & (year==t%2)
        assert_array_equal(whitened[0, ~mask, :, j],0)
    signal = predict(field,layout,wanted,layout.joint_names,axis)
    fitted = solve(field | {'data':signal},layout,np.ones_like(early,bool),axis,cov,
                   annual_uncertainty=True,batched=True)
    assert fitted['estimable'][0]
    assert_allclose(fitted['coefficients'],wanted,rtol=0,atol=1e-8)
    assert fitted['target_min_information'][0] > 0
    assert_allclose(fitted['annual_en_covariance'], 0., atol=1e-15)
    # Single-frequency collinear geometry with no azimuth motion does not identify EN or Msf.
    dead = g.copy()
    dead[..., :, :2] = 0.
    deficient = solve(field | {'geometry_by_file':dead},layout,np.ones_like(early,bool),axis,cov,batched=True)
    assert not deficient['estimable'][0]


class TrackTruth(Truth):
    def __init__(self, track):
        super().__init__([2017,2018])
        self.track = track
        for y in self.years:
            self.velocity[y][:] = [.8,.6]
        self.msf[:] = [.08,-.06]
        self.vertical[:] = [.04,-.03,.02,.01]
        self.order = {2017: .3 if track == 37 else -.2, 2018: .15 if track == 37 else -.4}

    def geometry(self):
        association, owner, sensitivity = super().geometry()
        if self.track == 65:
            sensitivity[1,0] -= .4
            sensitivity[0,1] += .25
        return association, owner, sensitivity


def package(path, track):
    pairs = pair_list('2017-07-01',60)
    if track==65:
        lag = timedelta(hours=2)
        pairs = [(e+lag,l+lag,a,b) for e,l,a,b in pairs]
    write_package(path,TrackTruth(track),pairs,noise=[[.003,.0005],[.0005,.006]],seed=track,
                  grid_shift=0. if track==37 else 200.,dtype='f8')
    with h5netcdf.File(path,'a') as h:
        for which in ('earlier','later'):
            h.create_variable(f'{which}_relative_orbit',('time',),data=np.full(len(pairs),track,'i4'))
            h.create_variable(f'{which}_orbit_direction',('time',),dtype=h5py.string_dtype(),
                              data=np.full(len(pairs),'DESCENDING',object))
        h.attrs['content_hashes_json'] = json.dumps(variable_hashes(h))


def test_workflow_resume_export_verify_and_independent_coverage(tmp_path):
    package(tmp_path/'t37.nc',37)
    package(tmp_path/'t65.nc',65)
    with pytest.raises(ValueError, match='distinct Sentinel-1 relative orbit'):
        ordered_tracks([tmp_path/'t37.nc',tmp_path/'t37.nc'])
    conf = tmp_path/'config.yaml'
    conf.write_text(config_text(['t65.nc','t37.nc'],'cache','joint',
                                'mode: controls\nregions: regions.json\ngroup_by: track_year',
                                min_span=330,extra='collection: {mode: sentinel1_multitrack}\ndiagnostics: {regions: null}\n'))
    # Two control boxes provide ample held-out residuals.
    boxes = [dict(id=f'c{j}',role='control',reviewed=True,
                  bounds=[GRID['x0']-25,GRID['y0']-700+j*350,GRID['x0']+1250,GRID['y0']-350+j*350])
             for j in range(2)]
    (tmp_path/'regions.json').write_text(json.dumps(dict(EPSG=3031,regions=boxes)))
    with pytest.raises(Interrupted,match='fit interrupted'):
        run_workflow(conf,'fit',interrupt=dict(stage='fit',after=5))
    out = tmp_path/'joint'
    with h5py.File(out/'source_coefficients.partial.h5') as h:
        assert h.attrs['completed_sources']==5
        first = h['residual_vertical/coefficients'][:5]
    assert run_workflow(conf,'run')['stage']=='complete'
    with h5py.File(out/'source_coefficients.h5') as h, h5netcdf.File(out/'coefficient_maps.nc') as nc:
        assert_array_equal(h['track'][:],[37,65])
        assert_array_equal(h['residual_vertical/coefficients'][:5],first)
        assert h['residual_vertical/estimable'][:].all()
        assert 'azimuth_track_037' in h and 'azimuth_track_065' in h
        assert h['map_source_index'].shape==(4,5)
        assert_array_equal(nc['source_observation_index'][:],h['map_source_index'][:])
        assert_allclose(nc['annual_EN_velocity'][:].mean(axis=(2,3)),[[.8,.6],[.8,.6]],atol=.01)
        assert nc['msf_azimuth_amplitude_by_track'].shape==(2,4,5)
    collection = json.loads((out/'collection-manifest.json').read_text())
    assert len({r['pair_id'] for r in collection['pairs']})==120
    assert len({r['native_pair_id'] for r in collection['pairs']})==60
    assert json.loads((out/'verification.json').read_text())['status']=='passed'
    for track in (37,65):
        assert read_maps(out,'msf-azimuth',track=track).selected.any()
    assert read_maps(out,'vertical-m2',mask='estimable').selected.any()
