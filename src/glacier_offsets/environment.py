"""Runtime provenance and native backend detection."""
import importlib
import json
import logging
import platform
import sys
from pathlib import Path


def select_backend(requested="auto"):
    import isce3
    if requested not in {"cpu", "gpu", "auto"}:
        raise ValueError("backend must be cpu, gpu, or auto")
    if requested != "cpu":
        try:
            cls = isce3.cuda.matchtemplate.PyCuAmpcor
            if not isce3.core.gpu_check.cuda_device_valid(0):
                raise RuntimeError("no usable CUDA device")
            return "gpu", cls
        except (AttributeError, RuntimeError, ImportError, ValueError, OSError) as exc:
            if requested == "gpu":
                raise RuntimeError(f"CUDA Ampcor unavailable: {exc}") from exc
            logging.info("CUDA unavailable (%s); using native CPU Ampcor", exc)
    return "cpu", isce3.matchtemplate.PyCPUAmpcor


def inspect_environment():
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("isce3", "compass", "s1reader", "osgeo.gdal", "numpy", "scipy", "h5py", "yaml", "pyproj"):
        mod = importlib.import_module(name)
        result["packages"][name] = {"version": getattr(mod, "__version__", "unknown"), "path": mod.__file__}
    result["backend"] = select_backend()[0]
    cls = select_backend("cpu")[1]
    cls()  # instantiate, not merely import
    result["ampcor_methods"] = {n: getattr(cls, n).__doc__ for n in ("setConstantGrossOffset", "setVaryingGrossOffset")}
    result["conda_packages"] = []
    for path in sorted((Path(sys.prefix) / "conda-meta").glob("*.json")):
        data = json.loads(path.read_text())
        result["conda_packages"].append({k: data.get(k) for k in ("name", "version", "build", "url", "sha256")})
    return result
