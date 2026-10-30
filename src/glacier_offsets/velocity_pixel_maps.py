"""Plot amplitude/phase from a pixel-inversion directory without refitting."""
from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np

from .velocity_io import digest,write_json
from .velocity_pixel_config import parameters_from_recipe,display_masks,load_config
from .velocity_pixel_runtime import auxiliary_path


COMPONENTS = {
    'msf-along': ('msf_along','Along-flow Msf velocity','m/day',0.20,None),
    'msf-azimuth': ('msf_azimuth','Satellite-azimuth Msf velocity','m/day',0.15,None),
    'vertical-m2': ('residual_vertical_M2','Residual vertical M2 height','m',0.30,'vertical_M2_C'),
    'vertical-o1': ('residual_vertical_O1','Residual vertical O1 height','m',0.30,'vertical_O1_C'),
}
# Historical two-year fits without stored names used these coefficient columns.
LEGACY_COLUMNS = {'vertical_M2_C': 6, 'vertical_O1_C': 8}
MASKS = ('default','estimable','standard','along','phase-quality','flow-proxy')


@dataclass
class MapData:
    x: np.ndarray
    y: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    footprint: np.ndarray
    selected: np.ndarray
    source_id: np.ndarray
    mask: str
    phase_origin: str
    files: tuple
    epsg: int = 3031


def _mask_name(component,mask):
    if component not in COMPONENTS:raise ValueError(f'Unknown component: {component}')
    if mask not in MASKS:raise ValueError(f'Unknown display mask: {mask}')
    return ('standard' if component=='msf-azimuth' else 'along') if mask=='default' else mask


def resolve_map_input(path):
    """Use the inversion YAML only to locate output; saved fit settings remain authoritative."""
    path=Path(path).expanduser().resolve()
    if path.is_dir():return path,None
    if path.suffix.lower() in ('.yaml','.yml'):
        from .velocity_workflow_config import config_version,load_workflow_config
        if config_version(path)==2:return Path(load_workflow_config(path)['output']),path
        return Path(load_config(path)['output']),path
    raise ValueError(f'Expected a velocity output directory or inversion YAML: {path}')


def _source_maps(directory,component,mask):
    run_manifest=directory/'run-manifest.json'
    # Version-2 fits carry their own map association; version 1 reads observations.h5.
    files=[directory/'source_coefficients.h5' if run_manifest.exists() else directory/'observations.h5',directory/'source_coefficients.h5']
    recipe_file=directory/'recipe.json';recipe=json.loads(recipe_file.read_text()) if recipe_file.exists() else {}
    if recipe_file.exists():files.append(recipe_file)
    completion=directory/'fit-completion.json'
    if completion.exists():
        record=json.loads(completion.read_text());files.append(completion)
        if digest(directory/'source_coefficients.h5')!=record['output_sha256']:raise ValueError('Stored source fits changed')
        if run_manifest.exists():
            from .velocity_cache import identity_digest
            if identity_digest(recipe)!=record['recipe_sha256']:raise ValueError('Stored fit recipe changed')
        elif recipe_file.exists() and digest(recipe_file)!=record['recipe_sha256']:raise ValueError('Stored fit recipe changed')
    if run_manifest.exists():
        # Display settings belong to the current run configuration, not the fit recipe.
        display=json.loads(run_manifest.read_text())['config']['display'];files.append(run_manifest)
    else:display=parameters_from_recipe(recipe)['display']
    with h5py.File(files[0]) as obs,h5py.File(files[1]) as fits:
        if fits.attrs.get('status')!='complete':raise ValueError('Source fits are incomplete')
        index=obs['map_source_index'][:];foot=index>=0;safe=np.maximum(index,0)
        if np.any(index[foot]>=len(fits['source_id'])):raise ValueError('Map has invalid source indices')
        if not np.array_equal(obs['source_id'][:],fits['source_id'][:]):raise ValueError('Map and fit source IDs differ')
        joint=fits['residual_vertical'];az=fits['azimuth_direct']
        is_az=component=='msf-azimuth';group=az if is_az else joint
        estimable=group['estimable'][:];standard=fits['standard_coverage'][:]
        if mask in ('along','phase-quality','flow-proxy'):
            choices=display_masks(joint['estimable'][:],az['estimable'][:],standard,
                fits['mean_flow_speed'][:],fits['year_direction_change_deg'][:],fits['azimuth_flow_projection_by_year'][:],
                joint['amplitude'][:],joint['amplitude_sigma'][:],joint['phase_sigma_deg'][:],display)
            selected=estimable&choices[{'along':'along','phase-quality':'phase_quality','flow-proxy':'proxy'}[mask]]
        else:selected=estimable&(standard if mask=='standard' else True)
        name=COMPONENTS[component][4]
        if name is None:
            amplitude=group['amplitude'][:];phase=group['phase_deg'][:]
        else:
            names=json.loads(joint.attrs['parameter_names']) if 'parameter_names' in joint.attrs else None
            start=names.index(name) if names is not None else LEGACY_COLUMNS[name]
            cs=joint['coefficients'][:,start:start+2]
            amplitude=np.hypot(cs[:,0],cs[:,1]);phase=np.degrees(np.arctan2(cs[:,1],cs[:,0]))
        return MapData(obs['x'][:],obs['y'][:],amplitude[safe],phase[safe],foot,
            foot&selected[safe],fits['source_id'][:][safe],mask,
            recipe.get('phase_origin','2000-01-01 00:00 UTC; phase=atan2(S,C)'),tuple(dict.fromkeys(files)),
            int(np.asarray(obs.attrs.get('EPSG',3031)).ravel()[0]))


def _netcdf_maps(directory,component,mask):
    path=directory/'coefficient_maps.nc';prefix=COMPONENTS[component][0]
    with h5py.File(path) as h:
        if h.attrs.get('status')!='complete':raise ValueError('Coefficient maps are incomplete')
        source_id=h['source_observation_index'][:];foot=source_id>=0
        estimable=h['azimuth_fit_estimable' if component=='msf-azimuth' else 'fit_estimable'][:].astype(bool)
        names={'standard':'standard_coverage_mask','along':'along_display_mask',
               'phase-quality':'phase_quality_display_mask','flow-proxy':'along_flow_proxy_mask'}
        selected=foot&estimable
        if mask!='estimable':selected&=h[names[mask]][:].astype(bool)
        origin=h.attrs.get('phase_origin','2000-01-01 00:00 UTC')
        if isinstance(origin,bytes):origin=origin.decode()
        return MapData(h['x'][:],h['y'][:],h[prefix+'_amplitude'][:],h[prefix+'_phase_deg'][:],
            foot,selected,source_id,mask,str(origin),(path,),int(np.asarray(h.attrs.get('EPSG',3031)).ravel()[0]))


def read_maps(directory,component='msf-along',mask='default'):
    """Prefer source fits; also support a directory containing only exported NetCDF."""
    directory,config_file=resolve_map_input(directory);mask=_mask_name(component,mask)
    if (directory/'source_coefficients.h5').exists() and (directory/'observations.h5').exists():
        result=_source_maps(directory,component,mask)
    elif (directory/'coefficient_maps.nc').exists():result=_netcdf_maps(directory,component,mask)
    else:raise ValueError('Directory needs observations.h5 + source_coefficients.h5, or coefficient_maps.nc')
    if result.x.ndim!=1 or result.y.ndim!=1 or min(len(result.x),len(result.y))<2:
        raise ValueError('Map needs one-dimensional x/y coordinates with at least two cells each')
    for coordinate in (result.x,result.y):
        delta=np.diff(coordinate)
        if not np.isfinite(coordinate).all() or delta[0]==0 or not np.allclose(delta,delta[0],rtol=1e-6,atol=1e-6):
            raise ValueError('Map coordinates must be finite and uniformly spaced')
    shape=(len(result.y),len(result.x))
    if any(getattr(result,name).shape!=shape for name in ('amplitude','phase','footprint','selected','source_id')):
        raise ValueError('Map fields do not match the x/y grid')
    # Normalize orientation, preserving the association between cell centres and values.
    rows=slice(None,None,-1) if result.y[0]<result.y[-1] else slice(None)
    cols=slice(None,None,-1) if result.x[0]>result.x[-1] else slice(None)
    result.x=result.x[cols];result.y=result.y[rows]
    for name in ('amplitude','phase','footprint','selected','source_id'):
        setattr(result,name,getattr(result,name)[rows,cols])
    result.selected&=np.isfinite(result.amplitude)&np.isfinite(result.phase)
    if config_file is not None:result.files+= (config_file,)
    return result


def plot_maps(directory,*,component='msf-along',mask='default',amplitude_max=None,
              phase_range='signed',output_dir=None,dpi=180,regions=None):
    if component not in (*COMPONENTS,'all'):raise ValueError(f'Unknown component: {component}')
    if amplitude_max is not None and (not np.isfinite(amplitude_max) or amplitude_max<=0):raise ValueError('amplitude-max must be positive')
    if phase_range not in ('signed','positive'):raise ValueError('phase-range must be signed or positive')
    if isinstance(dpi,bool) or not isinstance(dpi,int) or dpi<=0:raise ValueError('dpi must be a positive integer')
    directory,config_file=resolve_map_input(directory)
    components=tuple(COMPONENTS) if component=='all' else (component,)
    data={name:read_maps(directory,name,mask) for name in components}
    if regions is None and (directory/'run-manifest.json').exists():
        regions=json.loads((directory/'run-manifest.json').read_text())['config']['diagnostics']['regions']
    region_file=auxiliary_path(directory,'regions',regions)
    overlays=[] if region_file is None else json.loads(region_file.read_text())['regions']
    # Matplotlib is only needed when rendering, and no external GIS runtime is required.
    import os
    import tempfile
    os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'glacier-offsets-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    out=Path(output_dir).expanduser().resolve() if output_dir is not None else directory/'amplitude_phase_maps'
    out.mkdir(parents=True,exist_ok=True);records=[]
    for name,maps in data.items():
        prefix,title,units,default_max,_=COMPONENTS[name];maximum=default_max if amplitude_max is None else amplitude_max
        x,y=maps.x,maps.y;dx=x[1]-x[0];dy=y[1]-y[0]
        extent=np.array([x[0]-dx/2,x[-1]+dx/2,y[-1]+dy/2,y[0]-dy/2])/1000
        phase=(maps.phase+180)%360-180 if phase_range=='signed' else maps.phase%360
        phase_limits=(-180,180) if phase_range=='signed' else (0,360)
        fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
        for ax,values,cmap,limits,label,heading in zip(axes,(maps.amplitude,phase),('viridis','twilight_shifted'),
                ((0,maximum),phase_limits),(units,'degrees'),('Amplitude','Phase')):
            ax.set_facecolor('white')
            ax.imshow(np.where(maps.footprint,1.,np.nan),extent=extent,origin='upper',cmap='gray',vmin=0,vmax=1.15,interpolation='nearest')
            im=ax.imshow(np.where(maps.selected,values,np.nan),extent=extent,origin='upper',cmap=cmap,vmin=limits[0],vmax=limits[1],interpolation='nearest')
            ax.set(title=heading,xlabel=f'EPSG:{maps.epsg} x (km)',ylabel=f'EPSG:{maps.epsg} y (km)');ax.set_aspect('equal')
            for region in overlays:
                west,south,east,north=np.array(region['bounds'])/1000
                ax.add_patch(Rectangle((west,south),east-west,north-south,fill=False,edgecolor='k',lw=.7))
                ax.text(west,north,region['id'],fontsize=6,va='bottom',clip_on=True)
            if not maps.selected.any():ax.text(.5,.5,'No cells pass this mask',transform=ax.transAxes,ha='center',bbox={'facecolor':'white','alpha':.8})
            fig.colorbar(im,ax=ax,shrink=.85,label=label)
        fig.suptitle(f'{title} | mask: {maps.mask}\nPhase = atan2(S, C); origin 2000-01-01 UTC. Gray: excluded; white: outside footprint.',fontsize=11)
        paths=[out/(prefix+'_amplitude_phase.'+extension) for extension in ('png','pdf')]
        for path in paths:fig.savefig(path,dpi=dpi)
        plt.close(fig)
        records.append({'component':name,'mask':maps.mask,'amplitude_units':units,'amplitude_limits':[0,maximum],
            'phase_limits_deg':phase_limits,'phase_origin':maps.phase_origin,'native_cells_displayed':int(maps.selected.sum()),
            'unique_sources_displayed':len(np.unique(maps.source_id[maps.selected])),
            'files':[str(path) for path in paths]})
        print(paths[0])
    inputs={str(path):digest(path) for maps in data.values() for path in maps.files}
    if config_file is not None:inputs[str(config_file)]=digest(config_file)
    if region_file is not None:inputs[str(region_file)]=digest(region_file)
    write_json(out/'plot-settings.json',{'input_directory':str(directory),'config_file':None if config_file is None else str(config_file),'input_sha256':inputs,'maps':records,
        'dpi':dpi,'interpretation':'Exploratory fitted coefficients; weak-signal phases and diagnostic masks are not calibrated detections.'})
    return records


def add_arguments(parser):
    parser.add_argument('directory',type=Path,metavar='DIRECTORY_OR_YAML',help='Velocity output directory or the inversion YAML (uses its output path)')
    parser.add_argument('--component',choices=(*COMPONENTS,'all'),default='msf-along')
    parser.add_argument('--mask',choices=MASKS,default='default',help='default: along for joint components, standard for direct azimuth')
    parser.add_argument('--amplitude-max',type=float,help='Upper color limit in m/day (horizontal) or m (vertical); clips colors only')
    parser.add_argument('--phase-range',choices=('signed','positive'),default='signed',help='signed: -180..180; positive: 0..360 degrees')
    parser.add_argument('--output-dir',type=Path,help='Default: DIRECTORY/amplitude_phase_maps')
    parser.add_argument('--dpi',type=int,default=180)
    parser.add_argument('--regions',type=Path,help='Optional reviewed ROI overlay; otherwise use regions staged in DIRECTORY')


def run_cli(args):
    return plot_maps(args.directory,component=args.component,mask=args.mask,amplitude_max=args.amplitude_max,
        phase_range=args.phase_range,output_dir=args.output_dir,dpi=args.dpi,regions=args.regions)


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);add_arguments(parser)
    args=parser.parse_args(argv)
    try:run_cli(args)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))


if __name__=='__main__':main()
