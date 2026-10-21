import platform
import isce3
import compass

print("Architecture:", platform.machine())
print("ISCE3:", isce3.__file__)
print("COMPASS:", compass.__file__)

ampcor = isce3.matchtemplate.PyCPUAmpcor()
print("Ampcor:", type(ampcor).__name__)
print("Installation successful!")
