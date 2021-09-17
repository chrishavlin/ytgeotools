import matplotlib.pyplot as plt
import yt
from numpy import isnan

from ytgeotools.seismology.datasets import XarrayGeoSpherical

filename = "IRIS/NA07_percent.nc"
ds = XarrayGeoSpherical(filename)
x, y, z, dvs = ds.interpolate_to_uniform_cartesian(["dvs"])

plt.hist(dvs[~isnan(dvs)].ravel(), bins=100)
plt.show()

yt_ds = ds.interpolate_to_uniform_cartesian(["dvs"], return_yt=True)
slc = yt.SlicePlot(yt_ds, "x", "dvs")
slc.set_log("dvs", False)
slc.save("test_lowres_interpolation")


yt_ds = ds.interpolate_to_uniform_cartesian(["dvs"], N=300, return_yt=True)
slc = yt.SlicePlot(yt_ds, "x", "dvs")
slc.set_log("dvs", False)
slc.save("test_highres_interpolation")
