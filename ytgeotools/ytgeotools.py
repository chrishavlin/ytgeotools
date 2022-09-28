"""Main module."""
import warnings
from typing import Type

import geopandas as gpd
import numpy as np
import xarray as xr
import yt_xarray  # NOQA
from yt import load_uniform_grid as lug

from ytgeotools.data_manager import data_manager as _dm
# profiler: sampling, binning, geopandas aggregate data
from ytgeotools.mapping import default_crs, successive_joins, validate_lons
from ytgeotools.seismology.collections import ProfileCollection


@xr.register_dataset_accessor("profiler")
class ProfilerAccessor:
    def __init__(self, xarray_obj):
        # __init__ can ONLY have xarray_obj here
        self._obj = xarray_obj
        self.crs = default_crs

    def set_crs(self, new_crs):
        self.crs = new_crs

    _surface_gpd = None

    @property
    def surface_gpd(self):

        if self._surface_gpd is None:
            lat, lg = self.latlon_grid
            lg = lg.ravel()
            lat = lat.ravel()

            df = gpd.GeoDataFrame(
                {"latitude": lat, "longitude": lg},
                geometry=gpd.points_from_xy(lg, lat),
                crs=self.crs,
            )
            self._surface_gpd = df
        return self._surface_gpd

    _latlon_grid = None

    @property
    def latlon_grid(self):
        if self._latlon_grid is None:
            lon = self.get_coord("longitude")
            lat = self.get_coord("latitude")

            long, latg = np.meshgrid(lon, lat)
            self._latlon_grid = {"longitude": long, "latitude": latg}

        return self._latlon_grid["latitude"], self._latlon_grid["longitude"]

    def _validate_coord_name(self, name: str) -> str:
        if name in self._obj.coords:
            return name
        if name in coord_aliases:
            for candidate in coord_aliases[name]:
                if candidate in self._obj.coords:
                    return candidate
        raise RuntimeError(
            f"Could not find {name} coordinate or equivalent in dataset. If it "
            f"exists by another name, add it to the ytgeotools.coord_aliases"
            f"dictionary."
        )

    def get_coord(self, name: str):
        name = self._validate_coord_name(name)
        coord = self._obj[name]
        if name in coord_aliases["longitude"]:
            coord = validate_lons(coord.values)
        return coord

    def filter_surface_gpd(self, df_gpds, drop_null=False, drop_inside=False):
        df = self.surface_gpd
        return successive_joins(
            df, df_gpds, drop_null=drop_null, drop_inside=drop_inside
        )

    def get_profiles(
        self,
        field: str,
        df_gpds: list = None,
        depth_mask=None,
        drop_null=False,
        drop_inside=False,  # returns
    ) -> ProfileCollection:

        # yikes, need to handle coord order properly!!!!
        warnings.warn("has the coord order issue been fixed yet?")

        if df_gpds is not None:
            surface_df = self.filter_surface_gpd(
                df_gpds, drop_null=drop_null, drop_inside=drop_inside,
            )
        else:
            surface_df = self.surface_gpd

        raw_profiles = []
        crds = []
        lon = self.get_coord("longitude")
        lat = self.get_coord("latitude")

        var = getattr(self._obj, field)
        for _, row in surface_df.iterrows():
            # this is slow and should/could be vectorized
            lon_id = np.where(lon == row["longitude"])[0][0]
            lat_id = np.where(lat == row["latitude"])[0][0]

            # the following:
            if depth_mask is None:
                fvars = var[:, lat_id, lon_id]
            else:
                fvars = var[depth_mask, lat_id, lon_id]

            raw_profiles.append(fvars[:])
            crds.append((row["latitude"], row["longitude"]))

        crds = np.array(crds)

        return ProfileCollection(
            np.array(raw_profiles),
            self.get_coord("depth"),
            crds[:, 1],
            crds[:, 0],
            crs=self.crs,
        )


_latnames = ["lat", "latitude", "lats"]
_lonnames = ["lon", "long", "longitude", "lons"]
coord_aliases = {}
for ref_name, aliases in zip(["latitude", "longitude"], [_latnames, _lonnames]):
    full_list = (
        aliases + [a.upper() for a in aliases] + [a.capitalize() for a in aliases]
    )
    coord_aliases[ref_name] = full_list


# abstract class
class Dataset:

    geometry = "cartesian"

    def __init__(self, data: dict, coords: dict):
        """
        coords : dict
            {0: {"values": xvals, "name":x},
             1: {"values": yvals, "name":y},
             2: {"values": zvals, "name":z},
            }
        """

        self.fields = list(data.keys())

        for fld, vals in data.items():
            if len(vals.shape) != 3:
                raise ValueError(f"data arrays must be 3d, {fld} is {len(vals.shape)}d")
            setattr(self, fld, vals)

        if all([dim in coords.keys() for dim in [0, 1, 2]]) is False:
            raise ValueError("coords dictionary must contain values for 0, 1, 2")

        self.coords = [coords[dim]["values"] for dim in range(3)]

        self.bbox = np.array(
            [[self.coords[dim].min(), self.coords[dim].max()] for dim in range(3)]
        )
        self._coord_order = [coords[dim]["name"] for dim in range(3)]
        self._coord_hash = {dim: coords[dim]["name"] for dim in range(3)}
        self._coord_hash_r = {coords[dim]["name"]: dim for dim in range(3)}

    def get_coord(self, name: str):
        dim_index = self._coord_hash_r[name]
        return self.coords[dim_index]

    def data_dict(self, field_subset: list = None) -> dict:
        if field_subset is None:
            field_subset = self.fields
        return {f: getattr(self, f) for f in field_subset}

    def load_uniform_grid(self):
        return load_uniform_grid(self)


def load_uniform_grid(ds: Type[Dataset], *args, **kwargs):

    data = ds.data_dict()
    sizes = data[list(data.keys())[0]].shape
    dims = tuple(ds._coord_order)
    geometry = (ds.geometry, (dims))
    return lug(data, sizes, 1.0, *args, bbox=ds.bbox, geometry=geometry, **kwargs)


def open_dataset(file, *args, **kwargs):
    file = _dm.validate_file(file)
    return xr.open_dataset(file, *args, **kwargs)
