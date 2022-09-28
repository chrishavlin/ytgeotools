"""Main module."""
import warnings
from typing import List, Type, Union

import geopandas as gpd
import numpy as np
import xarray as xr
import yt_xarray  # NOQA
from scipy import spatial

import ytgeotools.mapping as ygm
import ytgeotools.seismology.datasets as sds
from ytgeotools.coordinate_transformations import geosphere2cart
from ytgeotools.data_manager import data_manager as _dm
from ytgeotools.seismology.collections import ProfileCollection


# profiler: sampling, binning, geopandas aggregate data
@xr.register_dataset_accessor("profiler")
class ProfilerAccessor:
    def __init__(self, xarray_obj):
        # __init__ can ONLY have xarray_obj here
        self._obj = xarray_obj
        self.crs = ygm.default_crs
        self.max_radius = ygm.default_radius
        self._interp_trees = {}

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
            coord = ygm.validate_lons(coord.values)
        return coord

    def filter_surface_gpd(self, df_gpds, drop_null=False, drop_inside=False):
        df = self.surface_gpd
        return ygm.successive_joins(
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

    _cartesian_coords = None

    @property
    def cartesian_coords(self) -> tuple:
        """
        returns 3d arrays representing earth-centered cartesian coordinates of
        every grid point
        """

        if self._cartesian_coords is None:
            depth, lat, lon = self._get_lat_lon_depth_grid()
            radius = self.max_radius - depth
            x, y, z = geosphere2cart(lat, lon, radius)
            self._cartesian_coords = x, y, z

        return self._cartesian_coords

    def _get_lat_lon_depth_grid(self):

        depth_ = self.get_coord("depth")
        lat_ = self.get_coord("latitude")
        lon_ = self.get_coord("longitude")
        depth, lat, lon = np.meshgrid(depth_, lat_, lon_, indexing="ij")
        return depth, lat, lon

    _cartesian_bbox = None

    @property
    def cartesian_bbox(self):
        if self._cartesian_bbox is None:
            x, y, z = self.cartesian_coords
            cbbox = np.array([[d.min(), d.max()] for d in [x, y, z]])
            self._cartesian_bbox = cbbox
        return self._cartesian_bbox

    def convert_to_cartesian(self, latitude, longitude, depth, rescale):
        radius = self.max_radius - depth

        x, y, z = geosphere2cart(latitude, longitude, radius)
        if rescale:
            cart_bbox = self.cartesian_bbox
            wids = np.abs(cart_bbox[:, 1] - cart_bbox[:, 0])
            x = (x - cart_bbox[0, 0]) / wids[0]
            y = (y - cart_bbox[1, 0]) / wids[1]
            z = (z - cart_bbox[2, 0]) / wids[2]

        return x, y, z

    def interpolate_to_uniform_cartesian(
        self,
        fields: List[str],
        N: int = 50,
        max_dist: Union[int, float] = 100,
        interpChunk: int = 500000,
        recylce_trees: bool = False,
        return_yt: bool = False,
        rescale_coords: bool = False,
        apply_functions: dict = None,
    ):
        """
        moves geo-spherical data (radius/depth, lat, lon) to earth-centered
        cartesian coordinates using a kdtree with inverse distance weighting (IDW)

        fields: List[str]
            fields to interpolate
        N: int
            number of points in shortest dimension (default 50)
        max_dist : int or float
            the max distance away for nearest neighbor search (default 100)
        interpChunk : int
            the chunk size for querying the kdtree (default 500000)
        recylce_trees : bool
            if True, will store the generated kdtree(s) in memory (default False)
        return_yt: bool
            if True, will return a yt dataset (default False)
        rescale_coords: bool
            if True, will rescale the dimensions to 1. in smallest range,
            maintaining aspect ration in other dimensions (default False)
        apply_functions: dict
            a dictionary with fields as keys, pointing to a list of callable
            functions that get applied to the the interpolated field. Functions
            must accept and return an ndarray.
        """

        if apply_functions is None:
            apply_functions = {}

        x, y, z = self.cartesian_coords  # the actual xyz at which we have data
        cart_bbox = self.cartesian_bbox

        # drop points at which we don't have data
        wids = np.abs(cart_bbox[:, 1] - cart_bbox[:, 0])
        dx = wids.min() / N
        Ngrid = np.floor(wids / dx).astype(int)

        fillval = np.nan
        xdata = x.ravel()
        ydata = y.ravel()
        zdata = z.ravel()
        trees = {}
        interpd = {}
        for field in fields:
            data = getattr(self._obj, field).values.ravel()

            if recylce_trees and field in self._interp_trees:
                trees[field] = self._interp_trees[field]
            else:
                x_fi = xdata[data != fillval]
                y_fi = ydata[data != fillval]
                z_fi = zdata[data != fillval]
                data = data[data != fillval]
                xyz = np.column_stack((x_fi, y_fi, z_fi))
                print("building kd tree for " + field)
                trees[field] = {"tree": spatial.cKDTree(xyz), "data": data}
                print("    kd tree built")

            interpd[field] = np.full((Ngrid[0], Ngrid[1], Ngrid[2]), np.nan)

        # interpolate the field data from x, y, z to xyz_int
        xyz = [
            np.linspace(cart_bbox[d, 0], cart_bbox[d, 1], Ngrid[d]) for d in range(3)
        ]
        xdata, ydata, zdata = np.meshgrid(*xyz, indexing="ij")
        orig_shape = xdata.shape
        xdata = xdata.ravel(order="C")
        ydata = ydata.ravel(order="C")
        zdata = zdata.ravel(order="C")

        parallel_query = False
        if parallel_query:
            raise NotImplementedError
        else:
            interpd = _query_trees(
                xdata,
                ydata,
                zdata,
                interpChunk,
                fields,
                trees,
                interpd,
                orig_shape,
                max_dist,
            )

        if recylce_trees:
            self._interp_trees.update(trees)

        if rescale_coords:
            max_wid = wids.max()
            for dim in range(3):
                xyz[dim] = (xyz[dim] - cart_bbox[dim, 0]) / max_wid

        for key, funchandles in apply_functions.items():
            for funchandle in funchandles:
                interpd[key] = funchandle(interpd[key])

        if return_yt:
            raise NotImplementedError

        if len(interpd) == 1:
            interpd = interpd[fields[0]]
        return xyz[0], xyz[1], xyz[2], interpd

    def _perturbation_calcs(
        self,
        ref_model: Union[Type[sds.ReferenceModel1D], Type[sds.ReferenceCollection]],
        field: str,
        ref_model_field: str = None,
        perturbation_type: str = "percent",
        to_perturbation: bool = True,
    ):

        field_data = getattr(self._obj, field)
        depth, lat, lon = self._get_lat_lon_depth_grid()

        if type(ref_model) == sds.ReferenceModel1D:
            # evaluate interpolated reference model at depths
            ref_data = ref_model.evaluate(depth)
        elif type(ref_model) == sds.ReferenceCollection:
            model = getattr(ref_model, ref_model_field)
            ref_data = model.evaluate(depth)

        if to_perturbation:
            # field is in reference, calculate the perturbation
            return_data = sds._calculate_perturbation(
                ref_data, field_data, perturbation_type
            )
        else:
            # calculate absolute from perturbation
            return_data = sds._calculate_absolute(
                ref_data, field_data, perturbation_type
            )

        return return_data

    def get_perturbation(
        self,
        ref_model: Union[Type[sds.ReferenceModel1D], Type[sds.ReferenceCollection]],
        field: str,
        ref_model_field: str = None,
        perturbation_type: str = "percent",
    ):

        return self._perturbation_calcs(
            ref_model,
            field,
            ref_model_field=ref_model_field,
            perturbation_type=perturbation_type,
            to_perturbation=True,
        )

    def get_absolute(
        self,
        ref_model: Union[Type[sds.ReferenceModel1D], Type[sds.ReferenceCollection]],
        field: str,
        ref_model_field: str = None,
        perturbation_type: str = "percent",
    ):

        return self._perturbation_calcs(
            ref_model,
            field,
            ref_model_field=ref_model_field,
            perturbation_type=perturbation_type,
            to_perturbation=False,
        )


def _query_trees(
    xdata: np.ndarray,
    ydata: np.ndarray,
    zdata: np.ndarray,
    interpChunk: int,
    fields: List[str],
    trees: dict,
    interpd: dict,
    orig_shape: tuple,
    max_dist: float,
) -> dict:
    # query the tree at each new grid point and weight nearest neighbors
    # by inverse distance. proceed in chunks.
    N_grid = len(xdata)
    print("querying kdtree on interpolated grid")
    N_chunks = int(N_grid / interpChunk) + 1
    print(f"breaking into {N_chunks} chunks")
    for i_chunk in range(0, N_chunks):
        print(f"   processing chunk {i_chunk + 1} of {N_chunks}")
        i_0 = i_chunk * interpChunk
        i_1 = i_0 + interpChunk
        if i_1 > N_grid:
            i_1 = N_grid
        pts = np.column_stack((xdata[i_0:i_1], ydata[i_0:i_1], zdata[i_0:i_1]))
        indxs = np.array(range(i_0, i_1))  # the linear indeces of this chunk
        for fi in fields:
            (dists, tree_indxs) = trees[fi]["tree"].query(
                pts, k=8, distance_upper_bound=max_dist
            )

            # remove points with all infs (no NN's within max_dist)
            m = np.all(~np.isinf(dists), axis=1)
            tree_indxs = tree_indxs[m]
            indxs = indxs[m]
            dists = dists[m]

            # IDW with array manipulation
            # Build weighting matrix
            wts = 1 / dists
            wts = wts / np.sum(wts, axis=1)[:, np.newaxis]  # shape (N,8)
            vals = trees[fi]["data"][tree_indxs]  # shape (N,8)
            vals = vals * wts
            vals = np.sum(vals, axis=1)  # shape (N,)

            # store in proper indeces
            full_indxs = np.unravel_index(indxs, orig_shape, order="C")
            interpd[fi][full_indxs] = vals

    return interpd


_latnames = ["lat", "latitude", "lats"]
_lonnames = ["lon", "long", "longitude", "lons"]
coord_aliases = {}
for ref_name, aliases in zip(["latitude", "longitude"], [_latnames, _lonnames]):
    full_list = (
        aliases + [a.upper() for a in aliases] + [a.capitalize() for a in aliases]
    )
    coord_aliases[ref_name] = full_list


def open_dataset(file, *args, **kwargs):
    file = _dm.validate_file(file)
    return xr.open_dataset(file, *args, **kwargs)
