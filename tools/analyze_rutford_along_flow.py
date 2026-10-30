"""Resolve native Rutford Msf velocity harmonics along/cross secular flow.

Amplitude units follow the user's older_version convention: m/day despite the
'displacement' filenames. Phases are degrees on a shared reference. Sources are
read-only. Spatial summaries are descriptive, not independent-pixel inference.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Geod,Transformer
from glacier_offsets.velocity_io import digest,write_json


NAMES=['secular_east_velocity','secular_north_velocity','Msf_east_displacement_amplitude',
       'Msf_north_displacement_amplitude','Msf_east_displacement_phase','Msf_north_displacement_phase']


def decompose(ve,vn,ae,an,pe,pn):
    speed=np.hypot(ve,vn)
    with np.errstate(invalid='ignore',divide='ignore'):
        ue,un=ve/speed,vn/speed
        # cos(omega*t + phase); changing the common sign/time origin does not
        # change along/cross amplitudes, ellipse orientation or energy fractions.
        e=ae*np.exp(1j*np.radians(pe));n=an*np.exp(1j*np.radians(pn))
        along=ue*e+un*n;cross=-un*e+ue*n
        total2=ae**2+an**2;fraction=abs(along)**2/total2
        ee=abs(e)**2;nn=abs(n)**2;en=np.real(e*np.conj(n))
        gap=np.sqrt((ee-nn)**2+4*en**2)
        major=np.sqrt(np.maximum((ee+nn+gap)/2,0));minor=np.sqrt(np.maximum((ee+nn-gap)/2,0))
        theta=.5*np.arctan2(2*en,ee-nn)
        angle=np.degrees(np.arccos(np.clip(abs(ue*np.cos(theta)+un*np.sin(theta)),0,1)))
        return {'speed':speed,'along':abs(along),'cross':abs(cross),'fraction':fraction,
                'total':np.sqrt(total2),'relative_rms_residual':np.sqrt(np.maximum(1-fraction,0)),
                'major_axis_angle_deg':angle,'minor_major_ratio':minor/major,'cross_secular_ratio':abs(cross)/speed}


def weighted_quantile(x,w,q=(.1,.25,.5,.75,.9)):
    ix=np.argsort(x);v=x[ix];weight=w[ix];position=(np.cumsum(weight)-.5*weight)/weight.sum()
    return np.interp(q,position,v)


def statistics(d,mask,area,label):
    w=area[mask];along=d['along'][mask];cross=d['cross'][mask];total=d['total'][mask]
    f=float(np.sum(w*along**2)/np.sum(w*total**2))
    return {'label':label,'pixels':int(mask.sum()),'area_km2':float(w.sum()/1e6),
        'area_weighted_harmonic_energy_fraction_along':f,
        'integrated_relative_RMS_residual_of_along_only_approximation':float(np.sqrt(1-f)),
        'quantiles_probability':[.1,.25,.5,.75,.9],
        'pixel_energy_fraction_along_quantiles':weighted_quantile(d['fraction'][mask],w),
        'major_axis_misalignment_deg_quantiles':weighted_quantile(d['major_axis_angle_deg'][mask],w),
        'minor_major_ratio_quantiles':weighted_quantile(d['minor_major_ratio'][mask],w),
        'along_amplitude_m_per_day_quantiles':weighted_quantile(along,w),
        'cross_amplitude_m_per_day_quantiles':weighted_quantile(cross,w),
        'cross_to_secular_speed_quantiles':weighted_quantile(d['cross_secular_ratio'][mask],w),
        'fraction_area_at_least_90pct_along_energy':float(np.sum(w[d['fraction'][mask]>=.9])/w.sum()),
        'fraction_area_at_least_95pct_along_energy':float(np.sum(w[d['fraction'][mask]>=.95])/w.sum()),
        'fraction_area_major_axis_within_10deg':float(np.sum(w[d['major_axis_angle_deg'][mask]<=10])/w.sum()),
        'fraction_area_major_axis_within_20deg':float(np.sum(w[d['major_axis_angle_deg'][mask]<=20])/w.sum())}


def verify():
    u=np.array([.6,.8]);phi=123.;amp=.2
    v=u*1.1;z=u*amp*np.exp(1j*np.radians(phi))
    d=decompose(*v,*abs(z),*np.degrees(np.angle(z)))
    assert np.isclose(d['fraction'],1) and d['cross']<1e-14
    # Independent cycle-average check including out-of-phase EN components.
    ve=np.array([.8,.3]);vn=np.array([.6,-.7]);ae=np.array([.2,.1]);an=np.array([.1,.3]);pe=np.array([20,80]);pn=np.array([110,-30])
    d=decompose(ve,vn,ae,an,pe,pn)
    t=np.linspace(0,2*np.pi,4096,endpoint=False)
    e=ae[:,None]*np.cos(t+np.radians(pe)[:,None]);n=an[:,None]*np.cos(t+np.radians(pn)[:,None]);speed=np.hypot(ve,vn)
    along=(ve[:,None]*e+vn[:,None]*n)/speed[:,None]
    f=np.mean(along**2,axis=1)/np.mean(e**2+n**2,axis=1)
    assert np.allclose(f,d['fraction'],atol=1e-14)
    assert np.allclose(d['along']**2+d['cross']**2,ae**2+an**2,atol=1e-14)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('/Users/briel/data/glaciers/antarctica/rutford/minyan_data/older_version'))
    p.add_argument('--output',type=Path,default=Path('.runtime/rutford-along-flow-older'))
    args=p.parse_args();source=args.input.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    verify();arrays=[];provenance={};reference=None
    for name in NAMES:
        file=source/(name+'.tif')
        with rasterio.open(file) as h:
            signature=(h.shape,h.transform,h.crs)
            if reference is None:reference=signature;profile=h.profile
            if signature!=reference:raise ValueError('Coefficient grids must match exactly')
            a=h.read(1,masked=True).filled(np.nan).astype(float);arrays.append(a)
            provenance[name]={'path':str(file),'sha256':digest(file),'finite':int(np.isfinite(a).sum())}
    ve,vn,ae,an,pe,pn=arrays
    valid=np.isfinite(np.stack(arrays)).all(0)&(np.hypot(ve,vn)>0)&(ae>=0)&(an>=0)&(np.hypot(ae,an)>0)
    d=decompose(*arrays);transform=profile['transform'];ny,nx=ve.shape
    if str(profile['crs'])!='EPSG:4326':raise ValueError('This audit expects the native geographic grid')
    lon=transform.c+np.arange(nx+1)*transform.a;lat=transform.f+np.arange(ny+1)*transform.e
    geod=Geod(ellps='WGS84');rowarea=[]
    for j in range(ny):
        area,_=geod.polygon_area_perimeter([lon[0],lon[1],lon[1],lon[0]],[lat[j],lat[j],lat[j+1],lat[j+1]])
        rowarea.append(abs(area))
    area=np.broadcast_to(np.array(rowarea)[:,None],ve.shape)
    rows=[]
    for low,high in ((0,.1),(.1,.5),(.5,1.),(1.,np.inf),(.5,np.inf)):
        for signal in (0,.02,.05,.10):
            mask=valid&(d['speed']>=low)&(d['speed']<high)&(d['total']>=signal)
            if mask.any():
                r=statistics(d,mask,area,f'speed {low}–{high} m/day; total Msf >= {signal} m/day')
                r.update(speed_min=low,speed_max=None if not np.isfinite(high) else high,min_total_Msf=signal);rows.append(r)
    # Check disproportionate influence from the largest amplitudes without
    # silently removing them from the primary result.
    primary=valid&(d['speed']>=.5)
    cutoff=float(np.quantile(d['total'][primary],.99));trimmed=primary&(d['total']<=cutoff)
    sensitivity=statistics(d,trimmed,area,'fast flow, excluding highest 1% total amplitudes')
    sensitivity['excluded_above_total_amplitude_m_per_day']=cutoff
    report={'input':provenance,'source_sha256':digest(__file__),'amplitude_units':'m/day, per direct user instruction; no 2*pi/P factor applied',
        'phase_units':'degrees, per direct user instruction; native values span -180..180 and are not individually zero-mean',
        'assumption':'east/north phases share a common reference; common phase sign/origin does not affect these metrics',
        'method':'project complex harmonic coefficients onto local geographic secular-flow and its 90-degree CCW normal',
        'energy_definition':'F_parallel=|u_E z_E+u_N z_N|^2/(|z_E|^2+|z_N|^2), equal to retained cycle-mean squared horizontal speed',
        'valid_pixels':int(valid.sum()),'weighting':'ellipsoidal native-grid cell area; no polar resampling, no independent-pixel uncertainty claims',
        'rows':rows,'high_amplitude_sensitivity':sensitivity,'verification':'known along-flow harmonic, independent cycle-average projection, and energy conservation passed',
        'limitations':['no coefficient uncertainty or independent glacier/grounding mask provided',
            'speed thresholds are descriptive proxies for moving ice, not validated region classifications',
            'summaries assess the inferred coefficient field, not proof that every cross-flow component is physical']}
    write_json(out/'summary.json',report)
    np.savez_compressed(out/'decomposition.npz',**{k:np.where(valid,v,np.nan) for k,v in d.items()},valid=valid)
    # Native-grid scientific products, rather than duplicated 100-m polar pixels.
    product=profile.copy();product.update(dtype='float32',count=1,nodata=np.nan,compress='deflate')
    for key in ('along','cross','fraction','major_axis_angle_deg','minor_major_ratio'):
        with rasterio.open(out/(key+'.tif'),'w',**product) as h:
            h.write(np.where(valid,d[key],np.nan).astype('float32'),1)
            h.update_tags(metric=key,phase_units='degrees',amplitude_units='m/day',source='older_version native rasters')
    plots(out,d,valid,lon,lat,area)
    for r in rows:
        if r['min_total_Msf']==0:print(r)
    print('trimmed',sensitivity)


def plots(out,d,valid,lon,lat,area):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    transform=Transformer.from_crs(4326,3031,always_xy=True)
    x,y=transform.transform(*np.meshgrid(lon,lat));x/=1000;y/=1000
    primary=valid&(d['speed']>=.5)
    fig,axes=plt.subplots(2,3,figsize=(13,9),layout='constrained')
    views=[('speed','Secular speed (m/day)',valid,'viridis',0,1.2),
           ('total','Total Msf amplitude norm (m/day)',primary,'magma',0,.3),
           ('fraction','Msf energy retained along flow',primary,'viridis',0,1),
           ('cross','Cross-flow Msf amplitude (m/day)',primary,'magma',0,.15),
           ('major_axis_angle_deg','Ellipse axis / secular-flow angle (degrees)',primary,'viridis',0,45),
           ('minor_major_ratio','Ellipse minor / major axis',primary,'viridis',0,1)]
    for ax,(key,title,mask,cmap,lo,hi) in zip(axes.ravel(),views):
        image=ax.pcolormesh(x,y,np.where(mask,d[key],np.nan),shading='flat',cmap=cmap,vmin=lo,vmax=hi,rasterized=True)
        ax.set_aspect('equal');ax.set_title(title,fontsize=10);ax.set_xlabel('EPSG:3031 x (km)');ax.set_ylabel('y (km)')
        fig.colorbar(image,ax=ax,shrink=.8,extend='max' if key not in ('fraction','minor_major_ratio') else 'neither')
    fig.suptitle('Rutford older_version: full phase-aware Msf projection\nTidal panels show secular speed >=0.5 m/day; no significance mask')
    fig.savefig(out/'along_flow_maps.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for lo,hi,label in ((.1,.5,'0.1–0.5 m/day'),(.5,1,'0.5–1 m/day'),(1,np.inf,'>=1 m/day')):
        take=valid&(d['speed']>=lo)&(d['speed']<hi)
        for ax,key,title in zip(axes,('fraction','major_axis_angle_deg','minor_major_ratio'),('Along-flow retained energy','Major-axis misalignment (degrees)','Ellipse minor/major ratio')):
            v=d[key][take];w=area[take];ix=np.argsort(v);cdf=np.cumsum(w[ix])/w.sum()
            ax.plot(v[ix],cdf,label=label);ax.set_xlabel(title);ax.set_ylabel('Area fraction');ax.grid(alpha=.2)
    axes[0].set_xlim(0,1);axes[1].set_xlim(0,90);axes[2].set_xlim(0,1);axes[0].legend(fontsize=8)
    fig.suptitle('Descriptive distributions by secular speed; native-cell area weighting')
    fig.savefig(out/'alignment_distributions.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
