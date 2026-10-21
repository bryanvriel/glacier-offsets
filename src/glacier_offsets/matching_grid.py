"""Sparse samples tied to exact full-resolution reference chip centers."""
from dataclasses import dataclass
import numpy as np


@dataclass
class MatchingGrid:
    starts_line: np.ndarray
    starts_sample: np.ndarray
    window: tuple
    search: tuple
    stride: tuple

    @property
    def shape(self):
        return (len(self.starts_line), len(self.starts_sample))

    @property
    def centers(self):
        return np.meshgrid(self.starts_line + (self.window[0]-1)/2,
                           self.starts_sample + (self.window[1]-1)/2, indexing='ij')


def build_grid(reference_shape, secondary_shape, window, search, stride, gross=(0,0), first_center=None):
    starts=[]
    for axis in (0,1):
        n,m=reference_shape[axis],secondary_shape[axis]
        w,h,s,g=window[axis],search[axis],stride[axis],gross[axis]
        lower=max(0,h-g)
        upper=min(n-w,m-w-h-g)
        if first_center is not None:
            begin=first_center[axis]-(w-1)/2
            if not float(begin).is_integer() or begin<lower:
                raise ValueError('first_center must imply an integer chip start within both search bounds')
            lower=int(begin)
        if upper<lower:
            raise ValueError('No matching windows fit after applying gross offsets and residual search bounds')
        starts.append(np.arange(lower,upper+1,s,dtype=np.int32))
    return MatchingGrid(*starts,tuple(window),tuple(search),tuple(stride))


def valid_windows(reference, secondary, grid, gross):
    """Reject nonfinite samples and connected zero-fill, allowing isolated quantized zero observations.

    A 3x3 all-zero region or a zero connected to an image boundary seeds
    zero-fill detection. Propagation includes its whole connected component.
    One isolated CInt16 (0+0j) speckle observation is not evidence of missing data.
    """
    from scipy.ndimage import binary_opening, binary_propagation
    result=np.ones(grid.shape,dtype=bool)
    gross=np.broadcast_to(np.asarray(gross),(*grid.shape,2))
    for side,ds in enumerate((reference,secondary)):
        from osgeo import gdal
        from .raster_io import bounded_raster_cache
        band=ds.GetRasterBand(1)
        flags=band.GetMaskFlags()
        # COMPASS complex nodata=0 masks valid quantized zero real components;
        # retain the existing connected-complex-zero policy instead.
        complex_zero_nodata=(flags==gdal.GMF_NODATA and band.GetNoDataValue()==0 and gdal.DataTypeIsComplex(band.DataType))
        explicit_mask=flags!=gdal.GMF_ALL_VALID and not complex_zero_nodata
        zero=np.empty((ds.RasterYSize,ds.RasterXSize),dtype=bool)
        bad=np.empty_like(zero)
        with bounded_raster_cache():
            for row in range(0,ds.RasterYSize,64):
                height=min(64,ds.RasterYSize-row)
                data=ds.ReadAsArray(0,row,ds.RasterXSize,height)
                zero[row:row+height]=(data.real==0)&(data.imag==0)
                bad[row:row+height]=~np.isfinite(data)
                if explicit_mask:
                    bad[row:row+height] |= band.GetMaskBand().ReadAsArray(0,row,ds.RasterXSize,height)==0
            del data
        # Connectivity must still span the whole image, including read-strip
        # boundaries. Only the complex data read and cumulative sums change.
        seeds=binary_opening(zero,structure=np.ones((3,3),bool))
        seeds[0,:]|=zero[0,:];seeds[-1,:]|=zero[-1,:]
        seeds[:,0]|=zero[:,0];seeds[:,-1]|=zero[:,-1]
        bad |= binary_propagation(seeds,mask=zero)
        del zero,seeds
        integral=np.zeros((ds.RasterYSize+1,ds.RasterXSize+1),dtype=np.uint32)
        interior=integral[1:,1:]
        interior[:]=bad
        del bad
        np.cumsum(interior,axis=0,dtype=np.uint32,out=interior)
        np.cumsum(interior,axis=1,dtype=np.uint32,out=interior)
        del interior
        for i,line in enumerate(grid.starts_line):
            for j,sample in enumerate(grid.starts_sample):
                if side==0:
                    y,x,h,w=int(line),int(sample),*grid.window
                else:
                    y=int(line)+int(gross[i,j,0])-grid.search[0];x=int(sample)+int(gross[i,j,1])-grid.search[1]
                    h=grid.window[0]+2*grid.search[0];w=grid.window[1]+2*grid.search[1]
                if y<0 or x<0 or y+h>ds.RasterYSize or x+w>ds.RasterXSize:
                    result[i,j]=False
                    continue
                count=int(integral[y+h,x+w])+int(integral[y,x])-int(integral[y+h,x])-int(integral[y,x+w])
                result[i,j]&=count==0
        del integral
    return result
