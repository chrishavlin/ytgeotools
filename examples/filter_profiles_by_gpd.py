from ytgeotools.geo_points import EarthChem
from ytgeotools.seismology.datasets import XarrayGeoSpherical
import matplotlib.pyplot as plt
import numpy as np

file = "data/geo_points/earthchem_download_90561.csv"

# load with initial filters
initial_filters = [
    {"column": "age", "comparison": "<=", "value": 30},
]
echem = EarthChem(file, initial_filters=initial_filters)

_, volcanic_bound_df, _ = echem.build_volcanic_extent(radius_deg=0.5)


vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = XarrayGeoSpherical(vs_file)

surface_gpd = ds.surface_gpd
volcanic_surface = ds.filter_surface_gpd(volcanic_bound_df)
volcanic_surface_pts = ds.filter_surface_gpd(volcanic_bound_df, drop_null=True)

profs, coords = ds.get_profiles("dvs", df_gpd=volcanic_bound_df)

profs_nv, coords_nv = ds.get_profiles(
    "dvs", df_gpd=volcanic_bound_df, invert_selection=True
)
depth = ds.get_coord("depth")

nvolc = profs.shape[0]
n_nvolc = profs_nv.shape[0]
titlestr = f"N volc: {nvolc}, N non volc: {n_nvolc}"

# plot the mean and 1-std spread in each region
dvs_mean = np.mean(profs, axis=0)
dvs_std = np.std(profs, axis=0)
dvs_nv_mean = np.mean(profs_nv, axis=0)
dvs_nv_std = np.std(profs_nv, axis=0)

plt.plot(dvs_mean, depth, "r")
plt.plot(dvs_mean - dvs_std, depth, "--r")
plt.plot(dvs_mean + dvs_std, depth, "--r")
plt.plot(dvs_nv_mean, depth, "b")
plt.plot(dvs_nv_mean - dvs_nv_std, depth, "--b")
plt.plot(dvs_nv_mean + dvs_nv_std, depth, "--b")
plt.xlabel("dvs")
plt.gca().invert_yaxis()
plt.ylabel("depth")
plt.title(titlestr)
plt.show()
