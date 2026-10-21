"""Measure CPU Ampcor with OMP_NUM_THREADS set before process startup.

Synthetic amplitude tracking only. No real processing products are touched.
This is a small local check, not a portable performance benchmark.
"""
import json
import os
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch
import isce3
import numpy as np
from scipy.ndimage import gaussian_filter, shift
from osgeo import gdal
from glacier_offsets.ampcor import native_match
from glacier_offsets.config import load_defaults

out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(20261011)
amp=gaussian_filter(rng.uniform(.1,1,(512,1536)),.6)
ref=(amp*np.exp(1j*rng.normal(0,.2,amp.shape))).astype('complex64')
sec=shift(ref,(3,-6),order=3,mode='constant',cval=0)
for name,array in [('reference',ref),('secondary',sec)]:
    ds=gdal.GetDriverByName('GTiff').Create(str(out/f'{name}.tif'),array.shape[1],array.shape[0],1,gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(array);ds=None
native_timing={}
class MeasuredCPU(isce3.matchtemplate.PyCPUAmpcor):
    def runAmpcor(self):
        start=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF)
        cpu0=usage.ru_utime+usage.ru_stime
        super().runAmpcor()
        wall=time.perf_counter()-start;usage=resource.getrusage(resource.RUSAGE_SELF)
        native_timing.update(wall_seconds=wall,cpu_seconds=usage.ru_utime+usage.ru_stime-cpu0)

usage=resource.getrusage(resource.RUSAGE_SELF)
cpu0=usage.ru_utime+usage.ru_stime;wall0=time.perf_counter()
with patch('glacier_offsets.ampcor.select_backend',return_value=('cpu',MeasuredCPU)):
    result=native_match(out/'reference.tif',out/'secondary.tif',out/'native',
                        load_defaults()['matching'],{'gross_offsets':{'mode':'zero'}},'cpu')
wall=time.perf_counter()-wall0
usage=resource.getrusage(resource.RUSAGE_SELF)
cpu=usage.ru_utime+usage.ru_stime-cpu0
np.testing.assert_allclose(np.median(result['total'],axis=(0,1)),[3,-6],atol=.15)
np.save(out/'total.npy',result['total'])
report={'OMP_NUM_THREADS':os.environ['OMP_NUM_THREADS'],'windows':int(result['grid'].shape[0]*result['grid'].shape[1]),
        'native_wall_seconds':result['elapsed_seconds'],'match_including_preparation_wall_seconds':wall,
        'match_including_preparation_cpu_seconds':cpu,'cpu_seconds_per_wall_second':cpu/wall,
        'native_cpu_seconds':native_timing['cpu_seconds'],
        'native_cpu_seconds_per_wall_second':native_timing['cpu_seconds']/native_timing['wall_seconds'],
        'known_shift_recovered':True}
(out/'timing.json').write_text(json.dumps(report,indent=2)+'\n')
for name in ['reference','secondary']:
    (out/f'{name}.tif').unlink()  # Recreated from the fixed seed on each run.
print(json.dumps(report,indent=2))
