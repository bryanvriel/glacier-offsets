"""Standalone ISCE2 Ampcor worker; compatible with a separate Python 3.9 runtime.

Use the same native Ampcor kernel and magnitude inputs as DenseAmpcor, with
explicit chip starts instead of DenseAmpcor's automatic grid and fork wrapper.
"""
import hashlib
import json
from pathlib import Path
import sys


def provenance():
    import isce
    from mroipac.ampcor import ampcor
    from mroipac.ampcor.Ampcor import Ampcor
    import inspect
    files = [Path(ampcor.__file__), Path(inspect.getfile(Ampcor))]
    return {'engine': 'isce2-magnitude-ampcor-v1', 'version': isce.__version__,
            'python': sys.version, 'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def run(request):
    import isce  # Sets the installed component paths; must precede isceobj.
    import isceobj
    from mroipac.ampcor.Ampcor import Ampcor
    import numpy as np

    p = request['matching']
    images = []
    try:
        for filename in request['images']:
            image = isceobj.createSlcImage()
            image.setFilename(filename)  # Local filename.vrt, never an input sidecar.
            image.setWidth(request['shape'][1])
            image.setLength(request['shape'][0])
            image.setAccessMode('read')
            image.createImage()
            images.append(image)
        a = Ampcor()
        a.configure()
        a.setImageDataType1('mag')
        a.setImageDataType2('mag')
        a.setWindowSizeHeight(p['window'][0])
        a.setWindowSizeWidth(p['window'][1])
        a.setSearchWindowSizeHeight(p['search'][0])
        a.setSearchWindowSizeWidth(p['search'][1])
        a.oversamplingFactor = p['oversampling']
        a.zoomWindowSize = p['zoom']
        a.setDownGrossOffset(request['gross'][0])
        a.setAcrossGrossOffset(request['gross'][1])
        a.scaleFactorX = a.scaleFactorY = 1.
        # The Fortran kernel adds the search half-width to these 1-based
        # parameters before reading the reference chip. Preserve exact starts.
        a.firstSampleDown = request['rows'][0] + 1 - p['search'][0]
        a.lastSampleDown = request['rows'][-1] + 1 - p['search'][0]
        a.firstSampleAcross = request['cols'][0] + 1 - p['search'][1]
        a.lastSampleAcross = request['cols'][-1] + 1 - p['search'][1]
        a.skipSampleDown, a.skipSampleAcross = p['stride']
        a.numberLocationDown = a.numberLocationAcross = None
        # Explicit strides support a single row/column; upstream checkSkip
        # derives strides by dividing by numberLocation-1 instead.
        a.checkSkip = lambda: None
        a.thresholdSNR = 1.e-6  # DenseAmpcor's native rejection settings.
        a.thresholdCov = 1000.
        a.acrossLooks = a.downLooks = 1
        a.winsizeFilt, a.oversamplingFactorFilt = 1, 64
        a.band1 = a.band2 = 0
        a.debugFlag = a.displayFlag = False
        a.ampcor(*images)
        rows = np.asarray(a.locationDown, dtype='int64') - (p['window'][0]-1)//2 - 1
        cols = np.asarray(a.locationAcross, dtype='int64') - (p['window'][1]-1)//2 - 1
        np.savez_compressed(request['result'], rows=rows, cols=cols,
            total=np.column_stack([a.locationDownOffset, a.locationAcrossOffset]).astype('float32'),
            snr=np.asarray(a.snrRet, dtype='float32'),
            covariance=np.column_stack([a.cov2Ret, a.cov1Ret, a.cov3Ret]).astype('float32'))
    finally:
        for image in images:
            image.finalizeImage()


if __name__ == '__main__':
    if sys.argv[1] == '--probe':
        Path(sys.argv[2]).write_text(json.dumps(provenance()))
    else:
        run(json.loads(Path(sys.argv[1]).read_text()))
