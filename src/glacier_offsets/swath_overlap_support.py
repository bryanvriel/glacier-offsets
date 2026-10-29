"""Fixed unspliced overlap ownership and matching-result association."""
from copy import deepcopy
import numpy as np


def burst_interior_mask(sources, line, sample, owners, dual, window, search):
    """Exclude chip/search footprints crossing an along-track ownership change."""
    start_y=(line-(window[0]-1)/2).astype('int32')-search[0]
    start_x=(sample-(window[1]-1)/2).astype('int32')-search[1]
    height,width=[window[k]+2*search[k] for k in (0,1)]
    if not dual.any():
        return np.array(dual,copy=True)
    y0,x0=int(start_y[dual].min()),int(start_x[dual].min())
    y1,x1=int(start_y[dual].max()+height),int(start_x[dual].max()+width)
    within=np.array(dual,copy=True)
    ordered=sorted(sources,key=lambda e:(e['offset'][0],e['burst_id']))
    for swath in ('iw1','iw2'):
        raster=np.zeros((y1-y0,x1-x0),'uint16')
        for i,e in enumerate(ordered,1):
            if not e['burst_id'].endswith(swath):continue
            dy,dx=e['offset'];a,b,c,d=e['valid_window']
            a,b=max(a+dy,y0),min(b+dy,y1);c,d=max(c+dx,x0),min(d+dx,x1)
            if a<b and c<d:raster[a-y0:b-y0,c-x0:d-x0]=i
        for row,col in zip(*np.nonzero(dual)):
            a,b=int(start_y[row,col]-y0),int(start_x[row,col]-x0)
            within[row,col] &= bool((raster[a:a+height,b:b+width]==owners[swath][row,col]).all())
    return within


def select_views(sources, records, shape):
    """Freeze source ownership using support alone, before native matching."""
    views={s:{'owner':np.zeros(shape,'int16'),'support':np.zeros(shape,bool),
        'geometry':np.full((*shape,5,2),np.nan)} for s in ('iw1','iw2')}
    source_numbers={}
    for i,e in enumerate(sorted(sources,key=lambda e:(e['offset'][0],e['burst_id'])),1):
        key=e['burst_id'];source_numbers[i]=key
        if key not in records:continue
        r=records[key];v=views[key[-3:]];section=np.ix_(r['rows'],r['cols']);take=r['support']
        for field,value in [('owner',np.full(take.shape,i)),('support',take),('geometry',r['geometry'])]:
            block=v[field][section];block[take]=value[take];v[field][section]=block
    return views,source_numbers


def attach_results(selection, records, source_numbers, mode):
    """Attach each mode to fixed owners; a failed owner never falls back."""
    views=deepcopy(selection)
    for v in views.values():
        shape=v['support'].shape
        v.update({key:np.zeros(shape,bool) for key in ('valid','finite','search_boundary','native_completed')})
        v.update(total=np.full((*shape,2),np.nan),peak=np.full(shape,np.nan),snr=np.full(shape,np.nan))
    for number,key in source_numbers.items():
        if key not in records:continue
        r=records[key];result=r['results'].get(mode)
        if result is None:continue
        v=views[key[-3:]];section=np.ix_(r['rows'],r['cols'])
        take=v['owner'][section]==number
        for field,name in [('valid','valid'),('finite','finite'),('search_boundary','search_boundary'),
                           ('total','total'),('peak','correlation_peak'),('snr','snr')]:
            block=v[field][section];block[take]=result[name][take];v[field][section]=block
        block=v['native_completed'][section];block[take]=True;v['native_completed'][section]=block
    return views
