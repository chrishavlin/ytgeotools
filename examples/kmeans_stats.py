import matplotlib.pyplot as plt

import ytgeotools
from ytgeotools.geo_points.datasets import EarthChem
from ytgeotools.seismology.collections import DepthSeriesKMeans

vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = ytgeotools.open_dataset(vs_file)
P = ds.profiler.get_profiles("dvs")

model = DepthSeriesKMeans(P, n_clusters=5)
model.fit()

file = "data/earthchem/earthchem_download_90561.csv"
echem = EarthChem(file, drop_duplicates_by=["latitude", "longitude", "age"])

df = model.classify_points(echem.df)

colors = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 0.2, 1.0, 1.0),
    (1.0, 0.75, 0.5, 1.0),
    (1.0, 0.5, 0.75, 1.0),
    (0.0, 0.75, 0.2, 1.0),
]

fig, ax = plt.subplots(1)
model.bounding_polygons.plot("label", ax=ax, color=colors)

fig = plt.figure(figsize=(12, 5))
for iclust in range(model.n_clusters):
    plt.subplot(1, model.n_clusters, iclust + 1)
    plt.hist(df[df.label == iclust].age, bins=50, color=colors[iclust])
    plt.xlabel("age")
    plt.title(f"kmeans label: {iclust}")
plt.show()
