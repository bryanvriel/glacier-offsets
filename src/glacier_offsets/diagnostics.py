"""Scientific QC of raw observations, quality masks and geocoded TOTAL motion."""
import json
import logging
from pathlib import Path
import h5py
import numpy as np
from osgeo import gdal
from .products import write_json, checksum
from .storage import preview_array


def diagnostics(cfg,root,manifest):
    gdal.UseExceptions()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=root/'diagnostics';directory.mkdir(exist_ok=True)
    from . import __version__
    manifest['workflow_source_at_validation']={'version':__version__,'sha256':{p.name:checksum(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    manifest['diagnostics_software']={'matplotlib':matplotlib.__version__}
    with h5py.File(root/'radar/offsets.h5') as h:
        arrays={k:h[k][:] for k in ['valid','input_valid','snr','correlation_peak','covariance','center_line','center_sample','rejection_flags']}
        for field in ['gross','residual','total']:
            for axis in ['azimuth','range']:arrays[f'{field}/{axis}']=h[f'{field}/{axis}_px'][:]
        assembly=({name:h[name][:] for name in ('source_burst_index','seam_crossing','interval_seconds')}
                  if 'source_burst_index' in h else None)
        prior=({key:h[f'prior/{key}'][:] for key in ['vx','vy','predicted_px/azimuth','predicted_px/range','valid','rejection_flags']}
               if 'prior' in h else None)
    def image(ax,data,title,mask=None,cmap='viridis',extent=None,limits=None):
        display=np.where(mask,data,np.nan) if mask is not None else data
        finite=display[np.isfinite(display)]
        bounds=limits if limits is not None else (np.quantile(finite,[.02,.98]) if len(finite)>1 else [None,None])
        artist=ax.imshow(display,aspect='auto',cmap=cmap,vmin=bounds[0],vmax=bounds[1],extent=extent)
        ax.set_title(title,fontsize=10)
        if finite.size:
            plt.colorbar(artist,ax=ax,shrink=.7)
        if not finite.size:
            ax.text(.5,.5,'No finite selected observations',ha='center',va='center',transform=ax.transAxes,
                    bbox={'facecolor':'white','alpha':.85,'edgecolor':'none'})
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    grid=manifest['coreg']['reference_grid'];extent=[0,grid['width'],grid['length'],0]
    for j,side in enumerate(['reference','secondary']):
        a=preview_array(manifest, side)
        image(axes[0,j],np.log1p(np.abs(a)),f'{side} log amplitude (radar grid)',extent=extent)
        image(axes[1,j],(np.isfinite(a)&(np.abs(a)>0)).astype(float),f'{side} finite nonzero sample mask (preview)',extent=extent,limits=(0,1))
        for ax in axes[:,j]:ax.set_xlabel('reference range sample');ax.set_ylabel('reference azimuth line')
    fig.savefig(directory/'overview.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,5),layout='constrained')
    preview=preview_array(manifest, 'reference')
    ax.imshow(np.log1p(np.abs(preview)),aspect='auto',extent=extent,cmap='gray')
    for mask,label,color in [(~arrays['input_valid'],'Invalid input windows','gray'),
                              (arrays['input_valid']&~arrays['valid'],'Quality rejected','orange'),
                              (arrays['valid'],'Quality valid','cyan')]:
        ax.scatter(arrays['center_sample'][mask],arrays['center_line'][mask],s=7,c=color,label=f'{label}: {int(mask.sum())}')
    ax.set_xlabel('Full-resolution reference range sample');ax.set_ylabel('Full-resolution reference azimuth line')
    ax.set_title('Matching-window centers on the reference radar grid');ax.legend(loc='upper right',fontsize=8)
    fig.savefig(directory/'window_locations.png',dpi=140,bbox_inches='tight');plt.close(fig)
    if assembly is not None:
        fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
        image(axes[0,0],assembly['source_burst_index'],'Owning burst index (0 = no source)',cmap='tab10')
        image(axes[0,1],assembly['seam_crossing'].astype(float),'Chip/search crosses a source boundary',limits=(0,1))
        for ax,axis in zip(axes[1],('azimuth','range')):
            image(ax,arrays[f'total/{axis}'],f'TOTAL {axis} [SLC pixels]; quality-valid',mask=arrays['valid'],cmap='RdBu_r')
        for ax in axes.ravel():ax.set_xlabel('Matching-grid range column');ax.set_ylabel('Matching-grid azimuth row')
        fig.suptitle('SLC assembly before tracking: source ownership and seam-crossing observations')
        fig.savefig(directory/'burst_seams.png',dpi=140,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,10),layout='constrained')
    for i,field in enumerate(['gross','residual','total']):
        for j,axis in enumerate(['range','azimuth']):
            image(axes[i,j],arrays[f'{field}/{axis}'],f'RAW {field} {axis} [SLC pixels]',cmap='RdBu_r')
    image(axes[0,2],arrays['snr'],'Coarse amplitude-correlation SNR')
    peak_title = ('Correlation peak unavailable (ISCE2)' if manifest['matching']['matching'].get('engine') == 'isce2'
                  else 'Refined amplitude-correlation peak')
    image(axes[0,3],arrays['correlation_peak'],peak_title)
    image(axes[1,2],arrays['input_valid'].astype(float),'Input-window validity',limits=(0,1))
    image(axes[1,3],arrays['valid'].astype(float),'Matching quality validity',limits=(0,1))
    for j,axis in enumerate(['range','azimuth']):
        image(axes[2,j+2],arrays[f'total/{axis}'],f'Filtered TOTAL {axis} [SLC pixels]',mask=arrays['valid'],cmap='RdBu_r')
    fig.suptitle('Gross guides searches; total = gross + residual. Raw observations are retained.')
    fig.savefig(directory/'offset_quality.png',dpi=140,bbox_inches='tight');plt.close(fig)
    if prior is not None:
        units=manifest['matching']['gross_offsets']['velocity']['units']
        fig,axes=plt.subplots(3,2,figsize=(14,9),layout='constrained')
        for ax,key,title in zip(axes.ravel(),prior,
                               [f'Sampled vx [{units}]',f'Sampled vy [{units}]','Predicted azimuth [SLC pixels]',
                                'Predicted range [SLC pixels]','Prior eligible (0/1)','Prior rejection bits']):
            limits=(0,1) if key=='valid' else (0,31) if key=='rejection_flags' else None
            image(ax,prior[key].astype(float),title,limits=limits)
            ax.set_xlabel('Matching-grid range column');ax.set_ylabel('Matching-grid azimuth row')
        fig.suptitle('External prior guides searches only; missing coverage is masked')
        fig.savefig(directory/'velocity_prior.png',dpi=140,bbox_inches='tight');plt.close(fig)
    if (root/'geo/center_geometry.h5').is_file():
        with h5py.File(root/'geo/center_geometry.h5') as h:geo={k:h[k][:] for k in ['longitude_deg','latitude_deg','azimuth_ground_spacing_m_per_slc_line','geometry_valid']}
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,key,title in zip(axes[0],['longitude_deg','latitude_deg','azimuth_ground_spacing_m_per_slc_line'],['Center longitude [deg]','Center latitude [deg]','Local azimuth ground scale [m/SLC line]']):
            image(ax,geo[key],title,mask=geo['geometry_valid'])
        for j,axis in enumerate(['range','azimuth']):
            path=root/'geo'/f'{axis}_velocity_m_per_day.tif';ds=gdal.Open(str(path));a=ds.ReadAsArray();gt=ds.GetGeoTransform()
            image(axes[1,j],a,f'TOTAL {axis} velocity [m/day]',cmap='RdBu_r',extent=[gt[0],gt[0]+gt[1]*a.shape[1],gt[3]+gt[5]*a.shape[0],gt[3]])
            axes[1,j].set_xlabel('Projected x [m]');axes[1,j].set_ylabel('Projected y [m]')
        quality_path=root/'geo/quality_valid_mask.tif'
        if quality_path.is_file():
            ds=gdal.Open(str(quality_path));a=ds.ReadAsArray();gt=ds.GetGeoTransform()
            image(axes[1,2],a,'Quality accepted map cells (0/1)',
                  extent=[gt[0],gt[0]+gt[1]*a.shape[1],gt[3]+gt[5]*a.shape[0],gt[3]],limits=(0,1))
            axes[1,2].set_xlabel('Projected x [m]');axes[1,2].set_ylabel('Projected y [m]')
        else:
            axes[1,2].axis('off');axes[1,2].text(.02,.9,'Radar observables, not east/north velocity.\n\nNo stationary-ground mask supplied.\nAbsolute accuracy is not established.',va='top',wrap=True)
        fig.suptitle(f"Radar observables; absolute accuracy unverified. Source mask: {manifest.get('geocoding',{}).get('source_mask','quality')}")
        fig.savefig(directory/'geometry_checks.png',dpi=140,bbox_inches='tight');plt.close(fig)
    uncertainty_summary={}
    uncertainty_paths=[root/'geo'/f'{axis}_{quantity}.tif' for quantity in ['displacement_sigma_m','velocity_sigma_m_per_day']
                       for axis in ['range','azimuth']]
    if all(path.is_file() for path in uncertainty_paths):
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,path in zip(axes.ravel(),uncertainty_paths):
            ds=gdal.Open(str(path));data=ds.ReadAsArray();gt=ds.GetGeoTransform()
            finite=data[np.isfinite(data)];units=ds.GetMetadataItem('units')
            image(ax,data,f'{path.stem.replace("_"," ")} [{units}]',
                  extent=[gt[0],gt[0]+gt[1]*data.shape[1],gt[3]+gt[5]*data.shape[0],gt[3]])
            ax.set_xlabel('Projected x [m]');ax.set_ylabel('Projected y [m]')
            uncertainty_summary[path.stem]={'finite_cells':int(finite.size),'units':units,
                                           'median':float(np.median(finite)) if finite.size else None,
                                           'p95':float(np.quantile(finite,.95)) if finite.size else None}
        fig.suptitle('Native formal 1σ uncertainty; independent vertices assumed\n'
                     'Excludes systematic bias and spatial correlation, including overlapping chips')
        fig.savefig(directory/'uncertainty.png',dpi=140,bbox_inches='tight');plt.close(fig)
    valid=arrays['valid']
    mapping=manifest.get('geocoding',{})
    map_cells=int(np.prod(mapping['shape'])) if mapping.get('shape') else 0
    source_centers=mapping.get('quality_valid_source_centers',mapping.get('valid_source_centers'))
    report={'windows':int(valid.size),'input_valid':int(arrays['input_valid'].sum()),'quality_valid':int(valid.sum()),
            'quality_coverage_fraction':float(valid.mean()),'backend':manifest['matching']['backend'],
            'matching_outcome':'quality_valid_observations' if valid.any() else 'no_quality_valid_matches',
            'mapping_outcome':'valid_geocoded_cells' if manifest.get('geocoding',{}).get('valid_output_cells',0)>0 else 'no_valid_geocoded_cells',
            'geocoding_coverage':{'quality_valid_centers':int(valid.sum()),
                                 'quality_and_geometry_valid_centers':source_centers,
                                 'quality_centers_rejected_by_geometry':int(valid.sum())-source_centers if source_centers is not None else None,
                                 'map_cells_in_bounding_rectangle':map_cells,
                                 'valid_map_cells':mapping.get('valid_output_cells',0),
                                 'valid_map_fraction':mapping.get('valid_output_cells',0)/map_cells if map_cells else None,
                                 'quality_valid_map_cells':mapping.get('quality_valid_output_cells',mapping.get('valid_output_cells',0)),
                                 'resampling':mapping.get('resampling','linear'),
                                 'source_mask':mapping.get('source_mask','quality'),
                                 'policy':'Maps use the configured resampling and source mask. Nearest selects the original radar sample before masking; linear requires adjacent selected vertices. Invalid geometry and space outside the center mesh remain nodata.'},
            'rejection_counts':{name:int(np.count_nonzero(arrays['rejection_flags']&bit)) for bit,name in
                                [(1,'invalid_input'),(2,'nonfinite'),(4,'search_boundary'),(8,'low_snr'),(16,'low_correlation_peak')]},
            'rejection_counts_note':'Flags overlap; counts are not mutually exclusive',
            'formal_uncertainty':uncertainty_summary,
            'gross_offsets':manifest['matching']['gross_offsets'],
            'stationary_ground':{'status':'not assessed','reason':'No independently identified stationary-ground mask supplied'},
            'limits':['Quality filtering does not establish absolute accuracy','Search-edge flags cannot detect every out-of-search false match',
                      'Formal covariance excludes bias, decorrelation and spatial correlation','No tide/vertical-motion/ionosphere correction'],
            'total_median_px':{axis:float(np.median(arrays[f'total/{axis}'][valid])) if valid.any() else None for axis in ['range','azimuth']},
            'geocoding':manifest.get('geocoding')}
    if prior is not None:
        report['velocity_prior']={'eligible_windows':int(prior['valid'].sum()),
            'rejection_counts':{name:int(np.count_nonzero(prior['rejection_flags']&bit)) for bit,name in
                                [(1,'geometry'),(2,'missing_velocity'),(4,'projection'),(8,'overflow'),(16,'outside_secondary')]},
            'missing_policy':'masked; no zero fallback','motion':'total measured gross + residual; prior not subtracted'}
    if assembly is not None:
        report['burst_assembly']={'method':manifest['assembly']['method'],
            'source_burst_ids':manifest['assembly']['source_burst_ids'],
            'seam_crossing_windows':int(assembly['seam_crossing'].sum()),
            'seam_crossing_input_valid':int((assembly['seam_crossing']&arrays['input_valid']).sum()),
            'seam_crossing_quality_valid':int((assembly['seam_crossing']&valid).sum()),
            'interval_policy':manifest['assembly']['interval_policy'],
            'sources':[{'index':i,'burst_id':e['burst_id'],'interval_seconds':e['interval_seconds'],
                        'matching_centers':int((assembly['source_burst_index']==i).sum()),
                        'quality_valid':int(((assembly['source_burst_index']==i)&valid).sum())}
                       for i,e in enumerate(manifest['assembly']['sources'],1)],
            'interpretation':'Seam crossing is a provenance flag, not an accuracy assessment or automatic rejection.'}
    write_json(directory/'report.json',report)
    if not valid.any():
        logging.warning('No quality-valid matches; any input-mask mapped results are unaccepted raw observations')
    manifest['stages']['diagnostics']='complete';write_json(root/'manifest.json',manifest)
