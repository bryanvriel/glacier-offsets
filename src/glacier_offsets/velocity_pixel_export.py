"""Export native-grid coefficient NetCDF, GeoTIFFs and exploratory figures."""
import argparse
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'glacier-offsets-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import h5py
import h5netcdf
import numpy as np
import rasterio
from rasterio.transform import from_origin
from glacier_offsets.velocity_io import digest,write_json
from glacier_offsets.velocity_inspect import M2_PERIOD,O1_PERIOD
from glacier_offsets.velocity_pixel_config import parameters_from_recipe,display_masks
from .velocity_pixel_runtime import auxiliary_path


def export(out,regions=None):
    region_file=auxiliary_path(out,'regions',regions)
    meta=json.loads((out/'inputs.json').read_text());recipe=json.loads((out/'recipe.json').read_text())
    parameters=parameters_from_recipe(recipe);display=parameters['display']
    fitmeta=json.loads((out/'fit-completion.json').read_text())
    if digest(out/'recipe.json')!=fitmeta['recipe_sha256']:raise ValueError('Stored fit recipe changed')
    if digest(out/'source_coefficients.h5')!=fitmeta['output_sha256']:raise ValueError('Changed source fits')
    with h5py.File(out/'observations.h5') as obs,h5py.File(out/'source_coefficients.h5') as fits:
        x,y=obs['x'][:],obs['y'][:];index=obs['map_source_index'][:];footprint=index>=0;safe=np.maximum(index,0);shape=index.shape
        map_chunks=(min(128,len(y)),min(128,len(x)))
        def mapped(value):
            v=np.asarray(value,dtype=float)[safe].copy();v[~footprint]=np.nan;return v
        standard=fits['standard_coverage'][:];speed=fits['mean_flow_speed'][:];angle=fits['year_direction_change_deg'][:]
        projection=fits['azimuth_flow_projection_by_year'][:]
        joint=fits['residual_vertical'];az=fits['azimuth_direct'];estimable=joint['estimable'][:];azestimable=az['estimable'][:]
        a=joint['amplitude'][:];sigma=joint['amplitude_sigma'][:]
        masks=display_masks(estimable,azestimable,standard,speed,angle,projection,a,sigma,joint['phase_sigma_deg'][:],display)
        along_display=masks['along'];along_proxy=masks['proxy'];phase_quality=masks['phase_quality']
        azdisplay=masks['azimuth'];azmoving=masks['azimuth_moving']
        coef=joint['coefficients'][:];azcoef=az['coefficients'][:]
        fields={
            'msf_along_C':(coef[:,-2],'m day-1'),'msf_along_S':(coef[:,-1],'m day-1'),
            'msf_along_amplitude':(a,'m day-1'),'msf_along_phase_deg':(joint['phase_deg'][:],'degree'),
            'msf_along_amplitude_diagnostic_se':(sigma,'m day-1'),'msf_along_phase_diagnostic_se_deg':(joint['phase_sigma_deg'][:],'degree'),
            'msf_azimuth_C':(azcoef[:,-2],'m day-1'),'msf_azimuth_S':(azcoef[:,-1],'m day-1'),
            'msf_azimuth_amplitude':(az['amplitude'][:],'m day-1'),'msf_azimuth_phase_deg':(az['phase_deg'][:],'degree'),
            'msf_azimuth_amplitude_diagnostic_se':(az['amplitude_sigma'][:],'m day-1'),
            'msf_azimuth_phase_diagnostic_se_deg':(az['phase_sigma_deg'][:],'degree'),
            'mean_flow_speed':(speed,'m day-1'),'year_direction_change_deg':(angle,'degree'),
            'flow_axis_east':(fits['flow_axis_EN'][:,0],'1'),'flow_axis_north':(fits['flow_axis_EN'][:,1],'1'),
            'azimuth_flow_projection_2017':(projection[:,0],'1'),'azimuth_flow_projection_2018':(projection[:,1],'1'),
            'clean_pairs_2017':(fits['clean_pairs_by_year'][:,0],'1'),'clean_pairs_2018':(fits['clean_pairs_by_year'][:,1],'1'),
            'span_days':(fits['span_days'][:],'day'),'msf_phase_bins_occupied':(fits['phase_bins'][:],'1'),
            'joint_target_condition':(joint['target_condition'][:],'1'),'azimuth_target_condition':(az['target_condition'][:],'1'),
            'range_residual_rms':(joint['rms_m'][:,0],'m'),'azimuth_residual_rms':(joint['rms_m'][:,1],'m'),
            'Msf_partial_R2':(joint['partial_r2'][:],'1'),'calendar_clusters':(joint['calendar_clusters'][:],'1')}
        for year,k in [(2017,0),(2018,2)]:
            fields['east_velocity_'+str(year)]=(coef[:,k],'m day-1');fields['north_velocity_'+str(year)]=(coef[:,k+1],'m day-1')
            fields['order_displacement_'+str(year)]=(coef[:,4+k//2],'m')
        for name,start in [('M2',6),('O1',8)]:
            fields['residual_vertical_'+name+'_C']=(coef[:,start],'m');fields['residual_vertical_'+name+'_S']=(coef[:,start+1],'m')
            fields['residual_vertical_'+name+'_amplitude']=(np.hypot(coef[:,start],coef[:,start+1]),'m')
            fields['residual_vertical_'+name+'_phase_deg']=(np.degrees(np.arctan2(coef[:,start+1],coef[:,start])),'degree')
        masks={'fit_estimable':estimable,'azimuth_fit_estimable':azestimable,'standard_coverage_mask':standard,
            'along_display_mask':along_display,'along_flow_proxy_mask':along_proxy,'phase_quality_display_mask':phase_quality,
            'known_mixed_source_mask':obs['known_mixed_source'][:]}
        nc=out/'coefficient_maps.nc';temporary=out/'coefficient_maps.partial.nc'
        with h5netcdf.File(temporary,'w') as dst,h5netcdf.File(meta['inputs'][0]['path'],'r') as original:
            dst.dimensions={'y':len(y),'x':len(x),'parameter':12,'azimuth_parameter':6,'target':2,'target2':2}
            dst.attrs.update(Conventions='CF-1.8',status='complete',EPSG=3031,title='Exploratory 2017+2018 local Msf and residual vertical tide coefficients',
                phase_origin='2000-01-01T00:00:00Z',phase_convention='v=C cos(omega*tau)+S sin(omega*tau); phase=atan2(S,C) degrees',
                horizontal_Msf_period_days=14.765294,vertical_M2_period_days=M2_PERIOD,vertical_O1_period_days=O1_PERIOD,
                order_sign='B_later - B_earlier; azimuth displacement metres',secular_vertical_velocity_m_day=0.,vertical_basis='height endpoint differences, M2 and O1; coefficients in metres',
                uncertainty=recipe['diagnostic_uncertainty'],
                spatial_fit='independent unique radar sources, copied native cells share source fits; no spatial smoothing',recipe_json=json.dumps(recipe,sort_keys=True))
            for name,v,dimension in [('x',x,'x'),('y',y,'y')]:
                variable=dst.create_variable(name,(dimension,),dtype='f8');variable[:]=v;variable.attrs.update(units='m',axis=name.upper())
                variable.attrs['standard_name']='projection_x_coordinate' if name=='x' else 'projection_y_coordinate'
            crs=dst.create_variable('crs',(),dtype='i4');crs[()]=0;crs.attrs.update(dict(original['crs'].attrs))
            for name,(value,unit) in fields.items():
                variable=dst.create_variable(name,('y','x'),dtype='f4',fillvalue=np.nan,compression='gzip',compression_opts=2,chunks=map_chunks);variable[:]=mapped(value)
                variable.attrs.update(units=unit,grid_mapping='crs')
            for name,value in masks.items():
                variable=dst.create_variable(name,('y','x'),dtype='u1',compression='gzip',compression_opts=2);variable[:]=np.where(footprint,value[safe],False).astype('u1')
                variable.attrs.update(units='1',grid_mapping='crs')
            variable=dst.create_variable('source_observation_index',('y','x'),dtype='i8',fillvalue=-1);variable[:]=np.where(footprint,obs['source_id'][:][safe],-1)
            for group,dimension,count in [('residual_vertical','parameter',12),('azimuth_direct','azimuth_parameter',6)]:
                names=json.loads(fits[group].attrs['parameter_names']);v=dst.create_variable(group+'_parameter_name',(dimension,),dtype=h5py.string_dtype());v[:]=names
                units=['m day-1']*4+['m']*6+['m day-1']*2 if count==12 else ['m day-1']*2+['m']*2+['m day-1']*2
                unit_var=dst.create_variable(group+'_parameter_units',(dimension,),dtype=h5py.string_dtype());unit_var[:]=units
                v=dst.create_variable(group+'_coefficients',(dimension,'y','x'),dtype='f8',fillvalue=np.nan,compression='gzip',compression_opts=2,chunks=(1,*map_chunks))
                v[:]=np.moveaxis(mapped(fits[group]['coefficients'][:]),-1,0)
                v=dst.create_variable(group+'_Msf_covariance',('target','target2','y','x'),dtype='f8',fillvalue=np.nan,compression='gzip',compression_opts=2,chunks=(1,1,*map_chunks))
                v[:]=np.moveaxis(mapped(fits[group]['coefficient_covariance'][:]),(2,3),(0,1))
        temporary.replace(nc)
        # Explicit center-to-edge transform, checked by a rasterio/iceutils readback.
        transform=from_origin(x[0]-(x[1]-x[0])/2,y[0]-(y[1]-y[0])/2,x[1]-x[0],-(y[1]-y[0]))
        exports={'msf_along_amplitude':along_display,'msf_along_phase_deg':along_display,
            'msf_azimuth_amplitude':azdisplay,'msf_azimuth_phase_deg':azdisplay,
            'msf_along_phase_diagnostic_se_deg':along_display,'mean_flow_speed':standard,
            'residual_vertical_M2_amplitude':along_display,'residual_vertical_M2_phase_deg':along_display,
            'residual_vertical_O1_amplitude':along_display,'residual_vertical_O1_phase_deg':along_display}
        tiffs=[]
        for name,mask in exports.items():
            image=mapped(np.where(mask,fields[name][0],np.nan)).astype('f4');path=out/(name+'.tif')
            with rasterio.open(path,'w',driver='GTiff',height=len(y),width=len(x),count=1,dtype='float32',crs='EPSG:3031',transform=transform,nodata=np.nan,compress='deflate',tiled=True) as dst:
                dst.write(image,1);dst.update_tags(units=fields[name][1],mask='see coefficient_maps.nc and report.md',phase_origin='2000-01-01 UTC')
            with rasterio.open(path) as check:
                if not np.allclose(check.xy(0,0),(x[0],y[0])):raise ValueError('GeoTIFF pixel-center shift')
                if not np.array_equal(check.read(1),image,equal_nan=True):raise ValueError('GeoTIFF values changed')
            tiffs.append(path.name)
        # iceutils round-trip validates polar georeferencing used by downstream tools.
        import iceutils as ice
        check=ice.Raster(rasterfile=str(out/'msf_along_amplitude.tif'))
        # iceutils' default GDAL convention reports the pixel-corner grid.
        if check.hdr.epsg!=3031 or not np.allclose(check.hdr.xcoords+check.hdr.dx/2,x) or not np.allclose(check.hdr.ycoords+check.hdr.dy/2,y):raise ValueError('iceutils GeoTIFF coordinate mismatch')
        extent=[(x[0]-(x[1]-x[0])/2)/1000,(x[-1]+(x[1]-x[0])/2)/1000,(y[-1]+(y[1]-y[0])/2)/1000,(y[0]-(y[1]-y[0])/2)/1000]
        regions=[] if region_file is None else json.loads(region_file.read_text())['regions']
        def panel(ax,values,mask,title,cmap,limits,label):
            ax.imshow(np.where(footprint,1.,np.nan),extent=extent,origin='upper',cmap='gray',vmin=0,vmax=1.15)
            im=ax.imshow(mapped(np.where(mask,values,np.nan)),extent=extent,origin='upper',cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
            ax.set(title=title,xlabel='EPSG:3031 x (km)',ylabel='EPSG:3031 y (km)');ax.set_aspect('equal')
            for r in regions:
                w,s,e,n=np.array(r['bounds'])/1000;ax.add_patch(Rectangle((w,s),e-w,n-s,fill=False,edgecolor='k',lw=.6))
                ax.text(w,n,r['id'].replace('target_','T').replace('control_','C'),fontsize=6,va='bottom',color='k',clip_on=True,bbox=dict(facecolor='white',alpha=.6,edgecolor='none',pad=.2))
            plt.colorbar(im,ax=ax,shrink=.85,label=label);return im
        fig,axes=plt.subplots(2,2,figsize=(13,10.5),layout='constrained')
        panel(axes[0,0],a,along_display,'Along-flow Msf amplitude: coverage + moving ice','viridis',(0,.2),'m/day')
        panel(axes[0,1],joint['phase_deg'][:],along_display,'Along-flow Msf phase: same support','twilight_shifted',(-180,180),'degrees from 2000-01-01')
        panel(axes[1,0],a,phase_quality,'Amplitude: additional phase/direction diagnostic screen','viridis',(0,.2),'m/day')
        panel(axes[1,1],joint['phase_deg'][:],phase_quality,f"Phase: diagnostic SE ≤{display['max_phase_sigma_deg']:g}°, amplitude/SE ≥{display['min_amplitude_over_sigma']:g}",'twilight_shifted',(-180,180),'degrees from 2000-01-01')
        fig.suptitle('2017+2018 independent-source Msf + residual vertical M2/O1\nYear-specific secular motion and spacecraft order; no spatial smoothing',fontsize=13)
        fig.savefig(out/'msf_amplitude_phase.png',dpi=180);fig.savefig(out/'msf_amplitude_phase.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        panel(axes[0],a,along_display,'Along-flow Msf velocity amplitude','viridis',(0,.2),'m/day')
        panel(axes[1],joint['phase_deg'][:],along_display,'Along-flow Msf velocity phase','twilight_shifted',(-180,180),'degrees from 2000-01-01')
        fig.suptitle('2017+2018 pointwise inversion: residual vertical M2/O1 + year-specific secular/order terms',fontsize=12)
        fig.savefig(out/'msf_quicklook.png',dpi=180);fig.savefig(out/'msf_quicklook.pdf');plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
        panel(axes[0],az['amplitude'][:],azmoving,'Direct satellite-azimuth Msf amplitude','viridis',(0,.15),'m/day in satellite azimuth')
        panel(axes[1],az['phase_deg'][:],azmoving,'Direct satellite-azimuth Msf phase','twilight_shifted',(-180,180),'degrees from 2000-01-01')
        fig.suptitle('Direction-independent comparator; phase flips ~180° when flow projects negatively into azimuth',fontsize=11)
        fig.savefig(out/'azimuth_amplitude_phase.png',dpi=180);fig.savefig(out/'azimuth_amplitude_phase.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
        diagnostic=[('clean_pairs_2017','2017 clean pairs','viridis',(0,60),'pairs'),('clean_pairs_2018','2018 clean pairs','viridis',(0,60),'pairs'),
            ('mean_flow_speed','Mean robust secular speed','viridis',(0,1.2),'m/day'),('year_direction_change_deg','2017–2018 secular direction change','magma',(0,45),'degrees'),
            ('azimuth_flow_projection_2017','Azimuth projection of flow direction','coolwarm',(-1,1),'dimensionless'),
            ('msf_along_phase_diagnostic_se_deg','Calendar-cluster phase diagnostic SE','magma',(0,90),'degrees')]
        for ax,(name,title,cmap,limits,label) in zip(axes.ravel(),diagnostic):panel(ax,fields[name][0],standard,title,cmap,limits,label)
        fig.suptitle('Support, flow geometry and conditional uncertainty diagnostics',fontsize=13)
        fig.savefig(out/'fit_diagnostics.png',dpi=180);fig.savefig(out/'fit_diagnostics.pdf');plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
        for row,name in enumerate(('M2','O1')):
            panel(axes[row,0],fields['residual_vertical_'+name+'_amplitude'][0],along_display,'Residual vertical '+name+' height amplitude','viridis',(0,.3),'metres')
            panel(axes[row,1],fields['residual_vertical_'+name+'_phase_deg'][0],along_display,'Residual vertical '+name+' height phase','twilight_shifted',(-180,180),'degrees from 2000-01-01')
        fig.suptitle('Residual height coefficients after CATS and full ENU solid-Earth correction\nDescriptive nuisance estimates, not total ocean-tide amplitudes',fontsize=12)
        fig.savefig(out/'residual_vertical_amplitude_phase.png',dpi=180);fig.savefig(out/'residual_vertical_amplitude_phase.pdf');plt.close(fig)
        # Summary statistics count independent sources and native copied cells separately.
        def count(mask):return {'unique_sources':int(mask.sum()),'map_cells':int(np.sum(footprint&mask[safe]))}
        def percentiles(value):
            return np.nanpercentile(value,[10,25,50,75,90,95,99]) if np.isfinite(value).any() else np.full(7,np.nan)
        summary={'unique_sources':len(a),'native_grid_shape':list(shape),'pairs_by_year':[60,59],
            'source_coverage':count(standard&estimable),'along_display':count(along_display),'phase_quality_display':count(phase_quality),'fast_stable_flow_proxy':count(along_proxy),
            'along_amplitude_percentiles_m_day':percentiles(a[along_display]),
            'azimuth_amplitude_percentiles_m_day':percentiles(az['amplitude'][:][azmoving]),
            'median_phase_diagnostic_se_deg':float(np.nanmedian(joint['phase_sigma_deg'][:][along_display])),
            'median_range_rms_m':float(np.nanmedian(joint['rms_m'][:][along_display,0])),
            'median_azimuth_rms_m':float(np.nanmedian(joint['rms_m'][:][along_display,1])),
            'GeoTIFFs':tiffs,'netCDF_sha256':digest(nc),'GeoTIFF_center_and_data_roundtrip':True,'iceutils_georeference_roundtrip':'default corner grid converted to pixel centers with dx/2,dy/2',
            'parameters':parameters,'regions_sha256':None if region_file is None else digest(region_file),
            'mask_definitions':{'along_display':f"standard coverage, estimable, mean speed>={display['min_speed_m_day']:g}m/day, minimum yearly absolute azimuth-flow projection>={display['min_abs_azimuth_projection']:g}",
                'phase_quality_display':f"along display plus direction change<={display['max_direction_change_deg']:g}deg, min projection>={display['diagnostic_min_abs_azimuth_projection']:g}, cluster diagnostic phase SE<={display['max_phase_sigma_deg']:g}deg, amplitude/SE>={display['min_amplitude_over_sigma']:g}",
                'fast_stable_flow_proxy':f"along display plus speed>={display['proxy_min_speed_m_day']:g}m/day, direction change<={display['max_direction_change_deg']:g}deg, min projection>={display['diagnostic_min_abs_azimuth_projection']:g}"},
            'uncertainty_warning':'diagnostic pixelwise cluster covariance, no FDR/significance or calibrated phase-confidence claim'}
        write_json(out/'map-summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k in ['source_coverage','along_display','phase_quality_display','fast_stable_flow_proxy']},indent=2))

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--regions',type=Path)
    a=p.parse_args(argv);export(a.output.resolve(),a.regions)


if __name__=='__main__':main()
