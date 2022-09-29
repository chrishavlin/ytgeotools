import matplotlib.pyplot as plt
import numpy as np

from ytgeotools import open_dataset
from ytgeotools.geo_points import EarthChem

file = "data/earthchem/earthchem_download_90561.csv"

# load with initial filters
initial_filters = [
    {"column": "age", "comparison": "<=", "value": 30},
]
echem = EarthChem(
    file,
    initial_filters=initial_filters,
    drop_duplicates_by=["latitude", "longitude", "age"],
)

_, volcanic_bound_df, _ = echem.build_volcanic_extent(radius_deg=0.5)


vs_file = "IRIS/wUS-SH-2010_percent.nc"
ds = open_dataset(vs_file)

surface_gpd = ds.profiler.surface_gpd
volcanic_surface = ds.profiler.filter_surface_gpd(volcanic_bound_df, drop_null=True)
volcanic_surface_pts = ds.profiler.filter_surface_gpd(
    volcanic_bound_df, drop_inside=True
)

profs_v = ds.profiler.get_profiles(
    "dvs",
    df_gpds=[
        volcanic_bound_df,
    ],
    drop_null=True,
)

profs_nv = ds.profiler.get_profiles(
    "dvs",
    df_gpds=[
        volcanic_bound_df,
    ],
    drop_inside=True,
)

nvolc = profs_v.profiles.shape[0]
n_nvolc = profs_nv.profiles.shape[0]
titlestr = f"N volc: {nvolc}, N non volc: {n_nvolc}"

# plot the mean and 1-std spread in each region
dvs_mean = np.mean(profs_v.profiles, axis=0)
dvs_std = np.std(profs_v.profiles, axis=0)
dvs_nv_mean = np.mean(profs_nv.profiles, axis=0)
dvs_nv_std = np.std(profs_nv.profiles, axis=0)

plt.plot(dvs_mean, profs_v.depth, "r")
plt.plot(dvs_mean - dvs_std, profs_v.depth, "--r")
plt.plot(dvs_mean + dvs_std, profs_v.depth, "--r")
plt.plot(dvs_nv_mean, profs_nv.depth, "b")
plt.plot(dvs_nv_mean - dvs_nv_std, profs_nv.depth, "--b")
plt.plot(dvs_nv_mean + dvs_nv_std, profs_nv.depth, "--b")
plt.xlabel("dvs")
plt.gca().invert_yaxis()
plt.ylabel("depth")
plt.title(titlestr)
plt.show()
