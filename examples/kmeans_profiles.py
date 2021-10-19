import matplotlib.pyplot as plt

from ytgeotools.seismology.collections import DepthSeriesKMeans
from ytgeotools.seismology.datasets import XarrayGeoSpherical

vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = XarrayGeoSpherical(vs_file)
P = ds.get_profiles("dvs")

model = DepthSeriesKMeans(P, n_clusters=3)
model.fit()
df = model.get_classified_coordinates()
df.plot("labels")

kmeans_stats = model.depth_stats()
plt.figure()
c = ["r", "g", "b", "c", "m"]
for i in range(model.n_clusters):
    minvals = kmeans_stats[i]["two_sigma_min"]
    maxvals = kmeans_stats[i]["two_sigma_max"]
    plt.plot(model.cluster_centers_[i, :], model.profile_collection.depth, color=c[i])
plt.gca().invert_yaxis()


print("calculating kmeans inertia vs number of clusters")
plt.figure()
c_range = range(1, 11)
models, inertia = model.multi_kmeans_fit(c_range)

plt.plot(c_range, inertia)
plt.xlabel("number of clusters")
plt.ylabel("kmeans inertia")
plt.show()
