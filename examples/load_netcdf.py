import matplotlib.pyplot as plt

import ytgeotools

filename = "IRIS/GYPSUM_percent.nc"

ds = ytgeotools.open_dataset(filename)
ds.profiler.surface_gpd.plot()
plt.show()
