import matplotlib.pyplot as plt
from numpy import isnan

import ytgeotools

filename = "IRIS/NA07_percent.nc"
ds = ytgeotools.open_dataset(filename)
x, y, z, dvs = ds.profiler.interpolate_to_uniform_cartesian(["dvs"])

plt.hist(dvs[~isnan(dvs)].ravel(), bins=100)
plt.show()

x, y, z, dvs = ds.profiler.interpolate_to_uniform_cartesian(["dvs"], N=300)
plt.hist(dvs[~isnan(dvs)].ravel(), bins=100)
plt.show()
