"""Evaluate stored point fits without mixing instantaneous and pair velocities.

The along-flow observation is an azimuth-derived equivalent, conditional on
fitted order, vertical and cross-flow secular terms. It never uses range or
subtracts the target Msf harmonic from the observation. Coefficients are read
by name; annual secular terms are integrated over each pair's overlap with
every calendar-year regime.
"""
import numpy as np
from .velocity_basis import harmonic,PERIOD
from .velocity_vertical import vertical_columns

VERTICAL=('vertical_M2_C','vertical_M2_S','vertical_O1_C','vertical_O1_S')


def _pair_geometry(field):
    if 'geometry_by_file' in field:return np.moveaxis(field['geometry_by_file'][field['file_index']],0,1)
    return np.moveaxis(field['geometry_by_year'][field['year_index']],0,1)


def evaluate_named(field,coefficients,names,axis,years,overlap=None):
    """Physical radar predictions and comparable interval velocities.

    ``names`` labels the coefficient columns (``east_velocity_<year>``,
    ``order_<year>``, vertical and ``Msf_C``/``Msf_S``). ``overlap`` gives the
    exact days of each pair inside each year regime; by default the whole
    interval belongs to the pair's labeled year.
    """
    coef=np.asarray(coefficients);axis=np.asarray(axis);index={n:j for j,n in enumerate(names)}
    early,late=field['early'],field['late'];dt=late-early;year=field['year_index']
    geometry=_pair_geometry(field)
    if overlap is None:overlap=dt[...,None]*(year[None,:,None]==np.arange(len(years)))
    velocity=np.stack([coef[:,[index[f'east_velocity_{y}'],index[f'north_velocity_{y}']]] for y in years])
    baseline_by_year=np.einsum('ksi,si->ks',velocity,axis)
    cross_by_year=velocity-baseline_by_year[...,None]*axis[None]
    secular=np.einsum('stk,stci,ksi->stc',np.nan_to_num(overlap),geometry[...,:2],velocity)
    cross_secular=np.einsum('stk,stci,ksi->stc',np.nan_to_num(overlap),geometry[...,:2],cross_by_year)
    order=np.zeros_like(field['data'])
    for k,y in enumerate(years):
        if f'order_{y}' in index:order[...,1]+=field['order'][None]*(year==k)[None]*coef[:,index[f'order_{y}'],None]
    height=np.einsum('stj,sj->st',vertical_columns(early,late),coef[:,[index[n] for n in VERTICAL]])
    vertical=height[...,None]*geometry[...,2]
    target=coef[:,[index['Msf_C'],index['Msf_S']]]
    displacement=np.einsum('stj,sj->st',harmonic(early,late),target)
    projection=np.einsum('stci,si->stc',geometry[...,:2],axis)
    msf=displacement[...,None]*projection
    prediction=secular+order+vertical+msf
    with np.errstate(invalid='ignore',divide='ignore'):
        baseline=np.einsum('stk,ks->st',overlap,baseline_by_year)/dt
        observed_radar=field['data']/dt[...,None]
        modeled_radar=prediction/dt[...,None]
        denominator=dt*projection[...,1]
        along_observed=(field['data'][...,1]-order[...,1]-vertical[...,1]-cross_secular[...,1])/denominator
        along_msf=displacement/dt
        along_modeled=baseline+along_msf
    mid=(early+late)/2
    omega=2*np.pi/PERIOD
    instantaneous=baseline+target[:,0,None]*np.cos(omega*mid)+target[:,1,None]*np.sin(omega*mid)
    return {'duration_days':dt,'midpoint_days':mid,'predicted_radar_displacement_m':prediction,
        'observed_radar_velocity_m_day':observed_radar,'modeled_radar_velocity_m_day':modeled_radar,
        'secular_radar_displacement_m':secular,'order_radar_displacement_m':order,
        'vertical_radar_displacement_m':vertical,'cross_secular_radar_displacement_m':cross_secular,
        'azimuth_flow_projection':projection[...,1],
        'observed_along_equivalent_m_day':along_observed,'modeled_along_pair_average_m_day':along_modeled,
        'modeled_along_secular_m_day':baseline,'modeled_along_Msf_pair_average_m_day':along_msf,
        'modeled_along_instantaneous_at_midpoint_m_day':instantaneous}


def evaluate_series(field,coefficients,axis):
    """Version-1 entry point: two labeled years in the historical column order."""
    names=('east_velocity_0','north_velocity_0','east_velocity_1','north_velocity_1','order_0','order_1',
           *VERTICAL,'Msf_C','Msf_S')
    return evaluate_named(field,coefficients,names,axis,(0,1))
