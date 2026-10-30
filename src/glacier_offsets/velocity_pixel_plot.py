"""Measured versus stored modeled pair velocities at four shelf target points.

No refitting or model changes. Geographic/coverage selection precedes reading
harmonic amplitudes. Along-flow observations are azimuth-derived equivalents
conditional on fitted nuisance terms; raw radar comparisons are also exported.
"""
import argparse
import csv
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'glacier-offsets-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import h5py
import numpy as np
from pyproj import Transformer
from glacier_offsets.velocity_basis import PERIOD,utc_days
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_pixel import screen_years
from glacier_offsets.velocity_pixel_config import parameters_from_recipe
from glacier_offsets.velocity_pixel_series import evaluate_series
from .velocity_pixel_inversion import get_field
from .velocity_pixel_runtime import auxiliary_path,module_fingerprints
from .velocity_pixel_mixed import check_mixed_support

EPOCH=datetime(2000,1,1)


def dates(values):
    return np.array([EPOCH+timedelta(days=float(v)) for v in values])


def plot_observations(ax,t,y,modeled,valid,good,limits=None):
    finite=np.isfinite(y+t);used=valid&good&finite;rejected=valid&~good&finite
    if limits is None:
        all_values=np.r_[y[used],modeled[np.isfinite(modeled)]]
        low,high=np.min(all_values),np.max(all_values);margin=max(.04,.12*(high-low));limits=(low-margin,high+margin)
    inside=rejected&(y>=limits[0])&(y<=limits[1])
    ax.scatter(dates(t[inside]),y[inside],s=14,marker='x',color='.65',alpha=.65,zorder=1)
    below=rejected&(y<limits[0]);above=rejected&(y>limits[1])
    for mask,marker,edge in [(below,'v',limits[0]),(above,'^',limits[1])]:
        ax.scatter(dates(t[mask]),np.full(mask.sum(),edge),s=24,marker=marker,facecolors='none',edgecolors='.55',clip_on=False,zorder=2)
    ax.scatter(dates(t[used]),y[used],s=17,color='#176d9b',alpha=.85,zorder=4)
    ax.set_ylim(*limits);ax.grid(axis='y',alpha=.2)
    return {'retained':int(used.sum()),'rejected':int(rejected.sum()),'rejected_outside_y_limits':int(below.sum()+above.sum())}


def draw_pair_model(ax,t,model,year,valid_time):
    for yr in (0,1):
        mask=(year==yr)&valid_time&np.isfinite(model)
        ax.plot(dates(t[mask]),model[mask],color='#d66c20',lw=1.2,marker='.',ms=3,zorder=3)


def draw_along_model(ax,t,model,year,valid_time,coef,axis,duration):
    velocity=np.stack([coef[:2],coef[2:4]]);baseline=velocity@axis
    for yr in (0,1):
        mask=(year==yr)&valid_time&np.isfinite(model)
        if not mask.any():continue
        dense=np.linspace(t[mask].min(),t[mask].max(),2500);interval=float(np.median(duration[mask]));w=2*np.pi/PERIOD
        prediction=baseline[yr]+np.sinc(interval/PERIOD)*(coef[-2]*np.cos(w*dense)+coef[-1]*np.sin(w*dense))
        ax.plot(dates(dense),prediction,color='#d66c20',lw=1.15,zorder=3)
        ax.scatter(dates(t[mask]),model[mask],s=10,facecolors='none',edgecolors='#d66c20',linewidths=.6,zorder=3)


def run(root,regions=None):
    region_file=auxiliary_path(root,'regions',regions,required=True)
    out=root/'point_timeseries';out.mkdir(exist_ok=True)
    input_meta=json.loads((root/'inputs.json').read_text());completion=json.loads((root/'fit-completion.json').read_text())
    recipe=json.loads((root/'recipe.json').read_text());parameters=parameters_from_recipe(recipe)
    check_mixed_support(root,recipe)
    if digest(root/'recipe.json')!=completion['recipe_sha256']:raise ValueError('Stored fit recipe changed')
    if digest(root/'source_coefficients.h5')!=completion['output_sha256']:raise ValueError('Stored fits changed')
    regions=[r for r in json.loads(region_file.read_text())['regions'] if r['role']=='target']
    if len(regions)!=4:raise ValueError('Point-series diagnostics require four reviewed target regions')
    with h5py.File(root/'observations.h5') as h,h5py.File(root/'solid_earth_tides.h5') as body,h5py.File(root/'source_coefficients.h5') as fits,h5py.File(input_meta['inputs'][0]['path']) as original:
        tr=Transformer.from_crs(4326,3031,always_xy=True);xx,yy=tr.transform(h['longitude'][:],h['latitude'][:])
        tide_response=original['tidal_response_factor'][:][h['row'][:],h['col'][:]]
        eligible=(fits['standard_coverage'][:]&fits['residual_vertical/estimable'][:]
            &(fits['mean_flow_speed'][:]>=.1)&(np.min(abs(fits['azimuth_flow_projection_by_year'][:]),axis=1)>=.5)
            &(fits['year_direction_change_deg'][:]<=20)&(tide_response>=.8))
        points=[]
        for region in regions:
            west,south,east,north=region['bounds'];cx,cy=region['center_xy']
            candidates=np.flatnonzero(eligible&(xx>=west)&(xx<=east)&(yy>=south)&(yy<=north))
            if not len(candidates):raise ValueError('No coverage/geometry-eligible point in '+region['id'])
            i=int(candidates[np.argmin((xx[candidates]-cx)**2+(yy[candidates]-cy)**2)])
            points.append({'label':region['id'].replace('target_','T'),'region':region['id'],'source_index':i,
                'source_id':int(h['source_id'][i]),'row':int(h['row'][i]),'col':int(h['col'][i]),
                'longitude':float(h['longitude'][i]),'latitude':float(h['latitude'][i]),'x_m':float(xx[i]),'y_m':float(yy[i]),
                'distance_from_reviewed_center_m':float(np.hypot(xx[i]-cx,yy[i]-cy)),
                'mean_flow_m_day':float(fits['mean_flow_speed'][i]),'clean_pairs_by_year':fits['clean_pairs_by_year'][i],
                'tidal_response_factor':float(tide_response[i])})
        indices=np.array(sorted(p['source_index'] for p in points));lookup={index:j for j,index in enumerate(indices)}
        field=get_field(h,body,indices,parameters['screening']['remove_mixed_windows']);good,axis,_,_,_=screen_years(field,parameters['screening']);stored_axis=fits['flow_axis_EN'][indices]
        if not np.allclose(axis,stored_axis,rtol=0,atol=1e-12):raise ValueError('Stored secular direction does not reproduce')
        coefficients=fits['residual_vertical/coefficients'][indices]
        series=evaluate_series(field,coefficients,stored_axis);dt=series['duration_days'];year=field['year_index']
        actual_mid=series['midpoint_days'];calendar_mid=(field['calendar_early']+field['calendar_late'])/2
        t=np.where(np.isfinite(actual_mid),actual_mid,calendar_mid[None])
        replay=[]
        for point in points:
            j=lookup[point['source_index']];coef=coefficients[j];point['Msf_amplitude_m_day']=float(np.hypot(*coef[-2:]));point['Msf_phase_deg']=float(np.degrees(np.arctan2(coef[-1],coef[-2])))
            residual=field['data'][j]-series['predicted_radar_displacement_m'][j]
            rms=np.sqrt(np.mean(residual[good[j]]**2,axis=0));expected=fits['residual_vertical/rms_m'][point['source_index']]
            error=float(np.max(abs(rms-expected)));replay.append(error)
            if error>1e-6:raise ValueError('Stored radar RMS does not reproduce')
            valid=field['valid'][j];time_valid=np.isfinite(actual_mid[j])
            columns={'pair_id':h['pair_id'].asstr()[:],'early_UTC':[str(v) if np.isfinite(d) else '' for v,d in zip(dates(np.nan_to_num(field['early'][j])),field['early'][j])],
                'late_UTC':[str(v) if np.isfinite(d) else '' for v,d in zip(dates(np.nan_to_num(field['late'][j])),field['late'][j])],
                'midpoint_UTC':[str(v) for v in dates(t[j])],'duration_days':dt[j],
                'input_valid':valid.astype(int),'used_in_fit':good[j].astype(int),'correlation_peak':field['peak'][j],
                'snr':h['snr'][point['source_index']],
                'measured_range_velocity_m_day':series['observed_radar_velocity_m_day'][j,:,0],
                'modeled_range_velocity_m_day':series['modeled_radar_velocity_m_day'][j,:,0],
                'measured_azimuth_velocity_m_day':series['observed_radar_velocity_m_day'][j,:,1],
                'modeled_azimuth_velocity_m_day':series['modeled_radar_velocity_m_day'][j,:,1],
                'azimuth_order_displacement_m':series['order_radar_displacement_m'][j,:,1],
                'azimuth_residual_vertical_displacement_m':series['vertical_radar_displacement_m'][j,:,1],
                'azimuth_cross_secular_displacement_m':series['cross_secular_radar_displacement_m'][j,:,1],
                'azimuth_flow_projection':series['azimuth_flow_projection'][j],
                'measured_along_equivalent_m_day':series['observed_along_equivalent_m_day'][j],
                'modeled_along_pair_average_m_day':series['modeled_along_pair_average_m_day'][j],
                'modeled_secular_along_m_day':series['modeled_along_secular_m_day'][j],
                'modeled_Msf_along_pair_average_m_day':series['modeled_along_Msf_pair_average_m_day'][j],
                'modeled_instantaneous_along_at_midpoint_m_day':series['modeled_along_instantaneous_at_midpoint_m_day'][j]}
            with (out/(point['label']+'.csv')).open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(columns)
                for k in range(len(year)):writer.writerow([value[k] for value in columns.values()])
        # Four pointwise along-flow views; all retained observations are visible.
        fig,axes=plt.subplots(4,1,figsize=(14,10.5),sharex=True,layout='constrained')
        for ax,point in zip(axes,points):
            j=lookup[point['source_index']];time_valid=np.isfinite(actual_mid[j]);observed=series['observed_along_equivalent_m_day'][j];modeled=series['modeled_along_pair_average_m_day'][j]
            counts=plot_observations(ax,t[j],observed,modeled,field['valid'][j],good[j]);draw_along_model(ax,t[j],modeled,year,time_valid,coefficients[j],stored_axis[j],dt[j]);point['plot_counts']=counts
            ax.axvline(datetime(2018,1,1),color='.5',ls='--',lw=.8);ax.set_ylabel('m/day')
            ax.set_title(f"{point['label']} | source {point['source_id']} | {point['mean_flow_m_day']:.2f} m/day mean | {counts['retained']} used pairs",fontsize=10,loc='left')
        axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3));axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));axes[-1].set_xlabel('UTC pair midpoint')
        handles=[plt.Line2D([],[],marker='o',ls='',color='#176d9b',label='Measured: used in fit'),plt.Line2D([],[],color='#d66c20',label='Stored model: pair-average; line uses a nominal 6-day window'),plt.Line2D([],[],marker='x',ls='',color='.65',label='Screened out; triangles mark out-of-range values')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('Individual shelf points: azimuth-derived along-flow equivalent velocity\nFitted spacecraft offset, residual vertical and secular cross-flow contributions removed; no spatial averaging',fontsize=12)
        fig.savefig(out/'along_flow_timeseries.png',dpi=180);fig.savefig(out/'along_flow_timeseries.pdf');plt.close(fig)
        # Direct radar-space comparisons include every fitted nuisance contribution.
        fig,axes=plt.subplots(4,2,figsize=(15,11),sharex=True,layout='constrained')
        for row,point in enumerate(points):
            j=lookup[point['source_index']];time_valid=np.isfinite(actual_mid[j])
            for channel in (0,1):
                ax=axes[row,channel];observed=series['observed_radar_velocity_m_day'][j,:,channel];modeled=series['modeled_radar_velocity_m_day'][j,:,channel]
                plot_observations(ax,t[j],observed,modeled,field['valid'][j],good[j]);draw_pair_model(ax,t[j],modeled,year,time_valid)
                ax.axvline(datetime(2018,1,1),color='.5',ls='--',lw=.8);ax.set_ylabel('m/day');ax.set_title(point['label']+' — '+('range','azimuth')[channel],loc='left',fontsize=10)
        for ax in axes[-1]:ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'));ax.set_xlabel('UTC pair midpoint')
        fig.legend(handles=handles[:1]+[plt.Line2D([],[],color='#d66c20',marker='.',label='Stored full radar model / interval')]+handles[2:],loc='outside lower center',ncol=3,fontsize=8)
        fig.suptitle('Direct measured versus modeled radar velocities = pair displacement / actual interval\nCATS and full ENU body tides already removed; model includes secular, order, residual vertical and Msf terms',fontsize=12)
        fig.savefig(out/'radar_velocity_timeseries.png',dpi=180);fig.savefig(out/'radar_velocity_timeseries.pdf');plt.close(fig)
        # Predetermined equal-season windows expose the six-day measurements.
        fig,axes=plt.subplots(4,2,figsize=(14,10),layout='constrained')
        for row,point in enumerate(points):
            j=lookup[point['source_index']];observed=series['observed_along_equivalent_m_day'][j];modeled=series['modeled_along_pair_average_m_day'][j]
            for column,yr in enumerate((2017,2018)):
                ax=axes[row,column];period=(t[j]>=float(utc_days(f'{yr}-01-01')))&(t[j]<float(utc_days(f'{yr}-04-01')))
                plot_observations(ax,t[j],observed,modeled,field['valid'][j]&period,good[j]&period)
                draw_along_model(ax,t[j],modeled,year,np.isfinite(actual_mid[j])&period,coefficients[j],stored_axis[j],dt[j])
                ax.set(xlim=(datetime(yr,1,1),datetime(yr,4,1)),ylabel='m/day',title=point['label']+' | '+str(yr))
                ax.xaxis.set_major_locator(mdates.MonthLocator());ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fig.suptitle('Same pointwise along-flow comparison: January–March windows\nOrange curve is an interval-average prediction; it is not the instantaneous harmonic',fontsize=12)
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=8);fig.savefig(out/'along_flow_zoom.png',dpi=180);fig.savefig(out/'along_flow_zoom.pdf');plt.close(fig)
        # Source positions and selection domain.
        back=h['map_source_index'][:];foot=back>=0;safe=np.maximum(back,0);x,y=h['x'][:],h['y'][:]
        speed=fits['mean_flow_speed'][:];background=np.where(foot&fits['standard_coverage'][:][safe],speed[safe],np.nan)
        fig,ax=plt.subplots(figsize=(8.3,7),layout='constrained');im=ax.imshow(background,extent=[(x[0]-100)/1000,(x[-1]+100)/1000,(y[-1]-100)/1000,(y[0]+100)/1000],origin='upper',cmap='Greys',vmin=0,vmax=1.2)
        for point in points:
            ax.plot(point['x_m']/1000,point['y_m']/1000,'o',color='#d66c20',ms=6);ax.annotate(point['label'],(point['x_m']/1000,point['y_m']/1000),xytext=(5,6),textcoords='offset points',fontsize=10,bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        ax.set(title='Selected single-source points near the reviewed shelf targets',xlabel='EPSG:3031 x (km)',ylabel='EPSG:3031 y (km)');fig.colorbar(im,ax=ax,label='Robust mean speed (m/day)',shrink=.8)
        fig.savefig(out/'point_locations.png',dpi=180);plt.close(fig)
        np.savez_compressed(out/'series.npz',source_indices=indices,good=good,valid=field['valid'],early=field['early'],late=field['late'],axis_EN=stored_axis,coefficients=coefficients,**series)
    write_json(out/'points.json',{'points':points,'selection':'nearest eligible physical source to each reviewed target centre, inside ROI; standard coverage, speed>=.1, |azimuth projection|>=.5, annual direction change<=20, CATS response>=.8; no Msf amplitude, phase or fit residual criterion',
        'inputs':{'observations_sha256':digest(root/'observations.h5'),'body_tides_sha256':digest(root/'solid_earth_tides.h5'),'source_fits_sha256':digest(root/'source_coefficients.h5'),'recipe_sha256':digest(root/'recipe.json')},
        'model':'current stored two-year pixel fits, original H1-trained annual noise weights; no refit or half-year reweighting',
        'comparison':'all model symbols use exact endpoint pair averages; smooth along curves use the median interval (~6 days) within each year',
        'along_observation':'azimuth-derived equivalent after fitted year-order, residual-vertical and secular-crossflow subtraction; depends on fitted nuisance and local flow direction, not independent horizontal truth',
        'screen':'stored-run yearwise screen; retained observations untrimmed; out-of-range rejected values explicitly marked',
        'inversion_parameters':parameters,
        'maximum_reconstructed_RMS_difference_m':max(replay),'regions_sha256':digest(region_file),
        'software_sha256':module_fingerprints(('velocity_pixel_plot','velocity_pixel_series'))})
    print(json.dumps(points,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--regions',type=Path)
    a=p.parse_args(argv);run(a.root.resolve(),a.regions)


if __name__=='__main__':main()
