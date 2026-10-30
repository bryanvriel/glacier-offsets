"""Map descriptive Larsen domains where an along-flow hypothesis is plausible.

This computes robust secular motion and coverage only, never an Msf detection.
Rutford speed thresholds are proxies to transfer for testing, not calibrated
Larsen validity criteria. Inputs and production map gates are unchanged.
"""
import json
import logging
from pathlib import Path
import numpy as np
from pyproj import Transformer,Proj
from glacier_offsets.velocity_io import Collection,write_json,digest
from glacier_offsets.velocity_quality import screen,crossfit_residual,quiet_median


def run(root):
    out=root/'along-flow-domain';out.mkdir(exist_ok=True)
    input_path=root/'frozen-2017/displacements_swath_corrected_tide_corrected_horizontal.nc'
    threshold=json.loads((root/'2017-quality-regional/screen_validation.json').read_text())['threshold']
    with Collection([input_path],start='2017-01-01',end='2018-01-01') as c:
        supported=c.association>=0;flat=np.flatnonzero(supported)
        ids,first=np.unique(c.association[supported],return_index=True)
        positions=flat[first];rr,cc=np.unravel_index(positions,c.shape)
        size=len(ids);results={k:np.full(size,np.nan) for k in ('speed','east','north','clean_pairs','span','phase_bins','halfyear_angle_deg','first_half_pairs','second_half_pairs')}
        for index,section in enumerate(c.sections(128)):
            row,col=section
            chosen=np.flatnonzero((rr>=row.start)&(rr<row.stop)&(cc>=col.start)&(cc<col.stop))
            if not len(chosen):continue
            tile=c.tile(section);r=rr[chosen]-row.start;s=cc[chosen]-col.start
            f={k:tile[k][:,r,s].T for k in ('early','late','valid','snr','peak')}
            en=tile['data'][:,r,s].transpose(1,0,2)
            geometry=tile['geometry'][r,s]
            f['data']=np.einsum('sij,stj->sti',geometry[...,:2],en)
            good=screen(f,threshold,crossfit_residual(f))
            rate=np.where(good[...,None],en/(f['late']-f['early'])[...,None],np.nan)
            velocity=quiet_median(rate,axis=1)
            results['east'][chosen]=velocity[:,0];results['north'][chosen]=velocity[:,1]
            results['speed'][chosen]=np.linalg.norm(velocity,axis=-1)
            results['clean_pairs'][chosen]=good.sum(1)
            span=np.max(np.where(good,f['late'],-np.inf),axis=1)-np.min(np.where(good,f['early'],np.inf),axis=1)
            results['span'][chosen]=np.where(good.any(1),span,0)
            ph=np.where(np.isfinite(f['early']+f['late']),np.floor(np.mod((f['early']+f['late'])/2,14.765294)/14.765294*12),-1)
            results['phase_bins'][chosen]=np.array([np.any(good&(ph==k),axis=1) for k in range(12)]).sum(0)
            split=6391. # replaced by explicit UTC conversion below
            from glacier_offsets.velocity_basis import utc_days
            split=float(utc_days('2017-07-01'))
            h1=good&(f['late']<split);h2=good&(f['early']>=split)
            v1=quiet_median(np.where(h1[...,None],rate,np.nan),axis=1)
            v2=quiet_median(np.where(h2[...,None],rate,np.nan),axis=1)
            with np.errstate(divide='ignore',invalid='ignore'):
                dot=np.sum(v1*v2,axis=-1)/(np.linalg.norm(v1,axis=-1)*np.linalg.norm(v2,axis=-1))
            angle=np.degrees(np.arccos(np.clip(dot,-1,1)))
            results['halfyear_angle_deg'][chosen]=angle
            results['first_half_pairs'][chosen]=h1.sum(1);results['second_half_pairs'][chosen]=h2.sum(1)
            if index%8==0:logging.info('Processed secular-motion tile %d; unique sources %d',index+1,size)
        h=c.handles[0];response=h['tidal_response_factor'][:]
        x,y=np.meshgrid(c.x,c.y)
        lon,lat=Transformer.from_crs(3031,4326,always_xy=True).transform(x,y)
        area=abs(np.median(np.diff(c.x))*np.median(np.diff(c.y)))/np.asarray(Proj('EPSG:3031').get_factors(lon,lat).areal_scale)
        index_map=np.searchsorted(ids,np.where(supported,c.association,ids[0]))
        mapped={k:np.where(supported,v[index_map],np.nan) for k,v in results.items()}
        base=supported&(mapped['clean_pairs']>=20)&(mapped['span']>=270)&np.isfinite(mapped['speed'])
        moving=base&(mapped['speed']>=.1)
        tide=base&np.isfinite(response)&(response>=.8)
        adequate=base&(mapped['clean_pairs']>=40)&(mapped['phase_bins']>=9)
        stable=(mapped['first_half_pairs']>=8)&(mapped['second_half_pairs']>=8)&(mapped['halfyear_angle_deg']<=10)
        stats=[]
        def summary(label,mask,denominator):
            return {'label':label,'area_km2':float(area[mask].sum()/1e6),'fraction_of_denominator':float(area[mask].sum()/area[denominator].sum()),
                'map_cells':int(mask.sum()),'unique_sources':len(np.unique(c.association[mask]))}
        footprints={'supported_source_footprint':supported,'reliable_secular_support':base,'moving_proxy':moving,'high_tide_response_proxy':tide}
        denominators={name:summary(name,mask,supported) for name,mask in footprints.items()}
        for name,domain in footprints.items():
            if name=='supported_source_footprint':continue
            for speed in (.3,.5,.8,1.):
                fast=domain&(mapped['speed']>=speed)
                for restriction,extra in (('speed_only',True),('stable_direction',stable),('original_fit_coverage',adequate),('stable_and_fit_coverage',stable&adequate)):
                    r=summary(f'{name}; speed>={speed}; {restriction}',fast&extra,domain)
                    r.update(domain=name,speed_threshold=speed,restriction=restriction);stats.append(r)
        regions=json.loads((root/'regions.reviewed.json').read_text())['regions'];regional=[]
        for r in regions:
            if r['role']!='target':continue
            w,s,e,n=r['bounds'];box=(x>=w)&(x<=e)&(y>=s)&(y<=n)&base
            fast=box&(mapped['speed']>=.5)
            regional.append({'region':r['id'],'covered_area_km2':float(area[box].sum()/1e6),
                'median_speed':float(np.nanmedian(mapped['speed'][box])),
                'speed_ge_05_area_fraction':float(area[fast].sum()/area[box].sum()),
                'stable_direction_and_fast_area_fraction':float(area[fast&stable].sum()/area[box].sum())})
        write_json(out/'summary.json',{'scope':'candidate structural-assumption domain, not validated Msf or production eligibility',
            'input':c.sources[0],'source_sha256':digest(__file__),'peak_threshold':threshold,
            'base_definition':'at least20 screened pairs, span>=270 days; robust local EN median velocity',
            'moving_definition':'base and secular speed>=0.1 m/day; proxy, not an independent ice-stream outline',
            'high_tide_response_definition':'base and tidal_response_factor>=0.8; model-based proxy, not independently confirmed floating-ice mask',
            'stable_definition':'>=8 clean pairs in each half, angle between H1/H2 secular medians<=10 degrees',
            'fit_coverage_definition':'>=40 clean pairs and >=9/12 Msf phase bins, in addition to base span',
            'area_definition':'native 200m grid footprint with EPSG3031 areal scale correction; copied cells count toward footprint, not independent observations',
            'denominators':denominators,'rows':stats,'regions':regional})
        np.savez_compressed(out/'domain_fields.npz',**mapped,x=c.x,y=c.y,response=response,base=base,moving=moving,stable=stable,adequate=adequate,area=area)
        plots(out,mapped,c.x,c.y,moving,stable,adequate,regions)
        c.check_revisions()
    print('denominators',denominators)
    for r in stats:
        if r['domain'] in ('moving_proxy','high_tide_response_proxy') and r['speed_threshold']==.5:print(r)
    print('regions',regional)


def plots(out,d,x,y,moving,stable,adequate,regions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    extent=[x.min()/1000,x.max()/1000,y.min()/1000,y.max()/1000]
    im=axes[0].imshow(np.where(moving,d['speed'],np.nan),extent=extent,origin='upper',vmin=.1,vmax=1.2,cmap='viridis')
    axes[0].set_title('Robust secular speed (m/day)');fig.colorbar(im,ax=axes[0],shrink=.8)
    candidate=moving&(d['speed']>=.5)
    labels=np.where(moving,0,np.nan);labels[candidate]=1;labels[candidate&stable]=2
    im=axes[1].imshow(labels,extent=extent,origin='upper',cmap=ListedColormap(['#d9d9d9','#e7ae49','#327c76']),vmin=-.5,vmax=2.5)
    cb=fig.colorbar(im,ax=axes[1],ticks=[0,1,2],shrink=.8);cb.ax.set_yticklabels(['0.1–0.5','>=0.5','>=0.5 + stable'])
    axes[1].set_title('Along-flow hypothesis candidates')
    im=axes[2].imshow(np.where(moving,d['clean_pairs'],np.nan),extent=extent,origin='upper',vmin=20,vmax=56,cmap='viridis')
    axes[2].contour(x/1000,y/1000,(candidate&adequate).astype(int),levels=[.5],colors=['orange'],linewidths=.5)
    axes[2].set_title('Clean pairs; orange: fast + fit coverage');fig.colorbar(im,ax=axes[2],shrink=.8)
    for ax in axes:
        ax.set_xlabel('EPSG:3031 x (km)');ax.set_ylabel('y (km)')
        for r in regions:
            if r['role']!='target':continue
            w,s,e,n=np.array(r['bounds'])/1000
            ax.add_patch(Rectangle((w,s),e-w,n-s,fill=False,edgecolor='red',lw=.8));ax.text(w,n,r['id'],fontsize=6,color='red')
    fig.suptitle('Larsen 2017: candidate area for testing an along-flow restriction\nSpeed and directional stability do not validate the Msf direction')
    fig.savefig(out/'candidate_domain.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    run(Path('.runtime/velocity-recovery-larsen').resolve())
