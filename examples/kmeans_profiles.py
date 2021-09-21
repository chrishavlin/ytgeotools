from ytgeotools.seismology.datasets import XarrayGeoSpherical
from ytgeotools.seismology.collections import ProfileCollection
from dask.distributed import Client
import matplotlib.pyplot as plt


vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = XarrayGeoSpherical(vs_file)
profs, x, y = ds.get_profiles("dvs")
depth = ds.get_coord("depth")
P = ProfileCollection(profs, depth, x, y, crs=ds.crs)

model = P.fit_kmeans(3)
df = P.get_classified_coords(model)
df.plot("labels")

kmeans_stats = P.kmeans_depth_stats(model)

plt.figure()
c = ["r", "g", "b", "c", "m"]
for i in range(3):
    plt.plot(model.cluster_centers_[i, :], depth, color=c[i])
    plt.plot(kmeans_stats[i]["one_sigma_min"], depth, color=c[i], linestyle="--")
    plt.plot(kmeans_stats[i]["one_sigma_max"], depth, color=c[i], linestyle="--")
plt.gca().invert_yaxis()
plt.show()





c = Client(threads_per_worker=1, n_workers=4)

c_range = range(1, 11)
models, inertia = P.multi_kmeans_fit(c_range)

plt.plot(c_range, inertia)
plt.show()
