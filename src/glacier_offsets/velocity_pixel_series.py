"""Evaluate stored point fits without mixing instantaneous and pair velocities.

The along-flow observation is an azimuth-derived equivalent, conditional on
fitted order, vertical and cross-flow secular terms. It never uses range or
subtracts the target Msf harmonic from the observation.
"""
import numpy as np
from .velocity_basis import harmonic,PERIOD
from .velocity_vertical import vertical_columns


def evaluate_series(field,coefficients,axis):
    """Return physical radar predictions and comparable interval velocities.

    field is the same source/time structure used by velocity_pixel.design.
    coefficients follow JOINT_NAMES: year EN secular (4), year order (2),
    residual height M2/O1 C/S (4), then shared along-flow Msf velocity C/S (2).
    """
    coef=np.asarray(coefficients);axis=np.asarray(axis)
    early,late=field['early'],field['late'];dt=late-early;year=field['year_index']
    geometry=np.moveaxis(field['geometry_by_year'][year],0,1)
    velocity_by_year=np.stack([coef[:,:2],coef[:,2:4]],axis=0)
    velocity=np.moveaxis(velocity_by_year[year],0,1)
    baseline=np.einsum('sti,si->st',velocity,axis)
    cross_velocity=velocity-baseline[...,None]*axis[:,None]
    secular=dt[...,None]*np.einsum('stci,sti->stc',geometry[...,:2],velocity)
    cross_secular=dt[...,None]*np.einsum('stci,sti->stc',geometry[...,:2],cross_velocity)
    order=np.zeros_like(field['data'])
    order[...,1]=field['order'][None]*coef[:,4:6][:,year]
    height=np.einsum('stj,sj->st',vertical_columns(early,late),coef[:,6:10])
    vertical=height[...,None]*geometry[...,2]
    displacement=np.einsum('stj,sj->st',harmonic(early,late),coef[:,-2:])
    projection=np.einsum('stci,si->stc',geometry[...,:2],axis)
    msf=displacement[...,None]*projection
    prediction=secular+order+vertical+msf
    with np.errstate(invalid='ignore',divide='ignore'):
        observed_radar=field['data']/dt[...,None]
        modeled_radar=prediction/dt[...,None]
        denominator=dt*projection[...,1]
        along_observed=(field['data'][...,1]-order[...,1]-vertical[...,1]-cross_secular[...,1])/denominator
        along_msf=displacement/dt
        along_modeled=baseline+along_msf
    mid=(early+late)/2
    omega=2*np.pi/PERIOD
    instantaneous=baseline+coef[:,-2,None]*np.cos(omega*mid)+coef[:,-1,None]*np.sin(omega*mid)
    return {'duration_days':dt,'midpoint_days':mid,'predicted_radar_displacement_m':prediction,
        'observed_radar_velocity_m_day':observed_radar,'modeled_radar_velocity_m_day':modeled_radar,
        'secular_radar_displacement_m':secular,'order_radar_displacement_m':order,
        'vertical_radar_displacement_m':vertical,'cross_secular_radar_displacement_m':cross_secular,
        'azimuth_flow_projection':projection[...,1],
        'observed_along_equivalent_m_day':along_observed,'modeled_along_pair_average_m_day':along_modeled,
        'modeled_along_secular_m_day':baseline,'modeled_along_Msf_pair_average_m_day':along_msf,
        'modeled_along_instantaneous_at_midpoint_m_day':instantaneous}
