from abc import ABC, abstractmethod
from typing import Any, List, Type, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy import spatial
from scipy.interpolate import interp1d
from unyt import unyt_quantity

from ytgeotools.coordinate_transformations import geosphere2cart
from ytgeotools.data_manager import data_manager as _dm
from ytgeotools.dependencies import dependency_checker
from ytgeotools.mapping import default_crs, successive_joins, validate_lons
from ytgeotools.seismology.collections import ProfileCollection
from ytgeotools.typing import all_numbers
from ytgeotools.ytgeotools import Dataset


def _calculate_perturbation(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    return_data = field_data - ref_data
    if perturbation_type in ["percent", "fractional"]:
        return_data = return_data / ref_data
        if perturbation_type == "percent":
            return_data = return_data * 100
    return return_data


def _calculate_absolute(
    ref_data: np.ndarray, field_data: np.ndarray, perturbation_type: str
) -> np.ndarray:
    # field_data is a perturbation, ref frame value
    if perturbation_type == "absolute":
        return_data = ref_data + field_data
    elif perturbation_type == "fractional":
        return_data = ref_data * (1 + field_data)
    elif perturbation_type == "percent":
        return_data = ref_data * (1 + field_data / 100)
    return return_data


class ReferenceModel(ABC):
    @abstractmethod
    def interpolate_func(self):
        pass

    @abstractmethod
    def evaluate(self):
        # return model values at a point
        pass

    def _validate_array(self, vals: np.typing.ArrayLike) -> np.ndarray:
        if type(vals) == np.ndarray:
            return vals
        return np.asarray(vals)


def _sanitize_ndarray(input_array: all_numbers) -> all_numbers:
    if type(input_array) == np.ndarray:
        if input_array.shape == ():
            return input_array.item()
    return input_array


class ReferenceModel1D(ReferenceModel):
    """
    A one-dimensional reference model

    Parameters
    ----------
    fieldname : str
        the name of the reference fild
    depth : ArrayLike
        array-like depth values for the reference model
    vals : Arraylike
        array-like model values
    disc_correction : bool
        if True (the default), will apply a discontinuity correction before
        creating the interpolating function. This looks for points at the same
        depth and offsets them by a small value.
    """

    def __init__(
        self,
        fieldname: str,
        depth: np.typing.ArrayLike,
        vals: np.typing.ArrayLike,
        disc_correction: bool = True,
    ):
        self.fieldname = fieldname
        self.depth = self._validate_array(depth)
        self.depth_range = (np.min(self.depth), np.max(self.depth))
        self.vals = self._validate_array(vals)
        self.disc_correction = disc_correction
        self.disc_off_eps = np.finfo(float).eps

    _interpolate_func = None

    @property
    def interpolate_func(self):
        if self._interpolate_func is None:

            depth = self.depth
            vals = self.vals

            if self.disc_correction:
                # deal with discontinuities
                # offset disc depths by a small number
                # disc_vals=[]
                eps_off = self.disc_off_eps
                d_diffs = depth[1:] - depth[0:-1]  # will be 1 element smaller
                disc_i = np.where(d_diffs == 0)[0]  # indices of discontinuties
                depth[disc_i + 1] = depth[disc_i + 1] + eps_off

            # build and return the interpolation function
            self._interpolate_func = interp1d(depth, vals)
        return self._interpolate_func

    def evaluate(self, depths: np.typing.ArrayLike, method: str = "interp") -> Any:
        if method == "interp":
            return _sanitize_ndarray(self.interpolate_func(depths))
        elif method == "nearest":
            raise NotImplementedError

    def perturbation(
        self,
        depths: np.typing.ArrayLike,
        abs_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        pert = _calculate_perturbation(ref_vals, abs_vals, perturbation_type)
        return _sanitize_ndarray(pert)

    def absolute(
        self,
        depths: np.typing.ArrayLike,
        pert_vals: np.typing.ArrayLike,
        method: str = "interp",
        perturbation_type: str = "percent",
    ) -> np.ndarray:

        ref_vals = self.evaluate(depths, method=method)
        abs_vals = _calculate_absolute(ref_vals, pert_vals, perturbation_type)
        return _sanitize_ndarray(abs_vals)


class ReferenceCollection:
    def __init__(self, ref_models: List[ReferenceModel1D]):
        self.reference_fields = []
        for ref_mod in ref_models:
            setattr(self, ref_mod.fieldname, ref_mod)
            self.reference_fields.append(ref_mod.fieldname)


class GeoSpherical(Dataset):
    def __init__(
        self,
        data: dict,
        latitude: ArrayLike,
        longitude: ArrayLike,
        depth: ArrayLike,
        coord_order: list = None,
        use_neg_lons: bool = False,
        max_radius: float = 6371.0,
        crs: dict = default_crs,
    ):

        if coord_order is None:
            coord_order = ["latitude", "longitude", "depth"]

        self.max_radius = max_radius

        longitude = validate_lons(longitude, use_negative_lons=use_neg_lons)
        coord_init = {"latitude": latitude, "longitude": longitude, "depth": depth}
        coords = {
            idim: {"values": coord_init[dim], "name": dim}
            for idim, dim in enumerate(coord_order)
        }
        self.max_radius = max_radius
        self.crs = crs
        self._interp_trees = {}
        super().__init__(data, coords)

    _cartesian_coords = None
    _cartesian_bbox = None

    def _get_lat_lon_depth_grid(self):
        cmesh = np.meshgrid(
            self.coords[0], self.coords[1], self.coords[2], indexing="ij"
        )

        depth = cmesh[self._coord_hash_r["depth"]]
        lat = cmesh[self._coord_hash_r["latitude"]]
        lon = cmesh[self._coord_hash_r["longitude"]]
        return depth, lat, lon

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

    def get_bounding_edges(self, points_per_edge=3, cartesian=False, rescale=False):

        minlat = self.bbox[self._coord_hash_r["latitude"]][0]
        maxlat = self.bbox[self._coord_hash_r["latitude"]][1]
        minlon = self.bbox[self._coord_hash_r["longitude"]][0]
        maxlon = self.bbox[self._coord_hash_r["longitude"]][1]
        mindep = self.bbox[self._coord_hash_r["depth"]][0]
        maxdep = self.bbox[self._coord_hash_r["depth"]][1]

        def edge_line(lat0, lon0, dep0, lat1, lon1, dep1):
            xyz = np.column_stack(
                [
                    np.linspace(lon0, lon1, ppe),
                    np.linspace(lat0, lat1, ppe),
                    np.linspace(dep0, dep1, ppe),
                ]
            )

            if cartesian:
                x, y, z = self.convert_to_cartesian(
                    xyz[:, 1], xyz[:, 0], xyz[:, 2], rescale=rescale
                )
                xyz = np.column_stack([x, y, z])

            return xyz

        ppe = points_per_edge
        curves = []
        for dep in [mindep, maxdep]:
            curves.append(edge_line(minlat, minlon, dep, maxlat, minlon, dep))
            curves.append(edge_line(maxlat, minlon, dep, maxlat, maxlon, dep))
            curves.append(edge_line(maxlat, maxlon, dep, minlat, maxlon, dep))
            curves.append(edge_line(minlat, maxlon, dep, minlat, minlon, dep))

        for lat in [minlat, maxlat]:
            curves.append(edge_line(lat, minlon, mindep, lat, minlon, maxdep))
            curves.append(edge_line(lat, maxlon, mindep, lat, maxlon, maxdep))

        return curves

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
            data = getattr(self, field).ravel()

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
            coords = {
                d: {"values": xyz[d], "name": dim} for d, dim in zip(range(3), "xyz")
            }
            return Dataset(interpd, coords,).load_uniform_grid()

        if len(interpd) == 1:
            interpd = interpd[fields[0]]
        return xyz[0], xyz[1], xyz[2], interpd

    _latlon_grid = None

    @property
    def latlon_grid(self):
        if self._latlon_grid is None:
            lon = self.get_coord("longitude")
            lat = self.get_coord("latitude")

            long, latg = np.meshgrid(lon, lat)
            self._latlon_grid = {"longitude": long, "latitude": latg}

        return self._latlon_grid["latitude"], self._latlon_grid["longitude"]

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
        drop_inside=True,
    ) -> ProfileCollection:

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

        var = getattr(self, field)
        for _, row in surface_df.iterrows():
            lon_id = np.where(lon == row["longitude"])[0][0]
            lat_id = np.where(lat == row["latitude"])[0][0]
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

    @dependency_checker.requires_cartopy
    def load_uniform_grid(self):
        return super().load_uniform_grid()

    def _perturbation_calcs(
        self,
        ref_model: Union[Type[ReferenceModel1D], Type[ReferenceCollection]],
        field: str,
        ref_model_field: str = None,
        perturbation_type: str = "percent",
        to_perturbation: bool = True,
    ):

        field_data = getattr(self, field)
        depth, lat, lon = self._get_lat_lon_depth_grid()

        if type(ref_model) == ReferenceModel1D:
            # evalulate interpolated reference model at depths
            ref_data = ref_model.evaluate(depth)
        elif type(ref_model) == ReferenceCollection:
            model = getattr(ref_model, ref_model_field)
            ref_data = model.evaluate(depth)

        if to_perturbation:
            # field is in reference, calculate the perturbation
            return_data = _calculate_perturbation(
                ref_data, field_data, perturbation_type
            )
        else:
            # calculate absolute from perturbation
            return_data = _calculate_absolute(ref_data, field_data, perturbation_type)

        return return_data

    def get_perturbation(
        self,
        ref_model: Union[Type[ReferenceModel1D], Type[ReferenceCollection]],
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
        ref_model: Union[Type[ReferenceModel1D], Type[ReferenceCollection]],
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


class XarrayGeoSpherical(GeoSpherical):
    """uses xarray to read in a dataset with latitude, longitude, depth coordinates"""

    geometry = "internal_geographic"

    def __init__(
        self,
        filename: str,
        field_subset: Union[list] = None,
        coord_aliases: dict = None,
        max_radius: Union[float, int] = 6371.0,
        radius_units: str = "km",
        use_neg_lons: bool = False,
    ):

        max_radius = unyt_quantity(max_radius, radius_units)

        filename = _dm.validate_file(filename)
        with xr.open_dataset(filename) as ds:

            coord_list = list(ds.coords)
            var_list = [v for v in list(ds.variables) if v not in coord_list]

            if field_subset is None:
                field_subset = var_list

            # check that all fields have same coordinate order
            coord_order = list(getattr(ds, field_subset[0]).coords)
            co_hash = "_".join(coord_order)
            for f in field_subset:
                if "_".join(list(getattr(ds, f).coords)) != co_hash:
                    raise ValueError(
                        f"all fields must have the same coordinate ordering but "
                        f"{f} does not. Please provide a field_subset containing "
                        f"only fields that have the same coordinate ordering."
                    )

            # pull out our data values
            data = {f: getattr(ds, f).values for f in field_subset}

            # pull out our coordinates, normalize the name
            if coord_aliases is None:
                coord_aliases = {}
            c_aliases = {"lat": "latitude", "lon": "longitude", "depth": "depth"}
            c_aliases.update(coord_aliases)

            latitude = getattr(ds, c_aliases["lat"]).values
            longitude = getattr(ds, c_aliases["lon"]).values
            depth = getattr(ds, c_aliases["depth"]).values
            depth_units = getattr(ds, c_aliases["depth"]).units
            max_radius = float(max_radius.to(depth_units).value)

        super().__init__(
            data,
            latitude,
            longitude,
            depth,
            coord_order=coord_order,
            max_radius=max_radius,
            use_neg_lons=use_neg_lons,
        )


def load_1d_csv_ref(
    filename: str, depth_column: str, value_column: str, **kwargs: Any
) -> Type[ReferenceModel1D]:
    """

    loads a 1D reference model from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns :str
        the column of the reference values
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceModel1D

    Examples
    --------
    from ytgeotools.seismology.datasets import load_1d_csv_ref
    import numpy as np
    ref = load_1d_csv_ref("IRIS/refModels/AK135F_AVG.csv", 'depth_km', 'Vs_kms')
    ref.evaluate([100., 150.])
    depth_new = np.linspace(ref.depth_range[0], ref.depth_range[1], 400)
    vs = ref.evaluate(depth_new)
    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    v = df[value_column].to_numpy()
    return ReferenceModel1D(value_column, d, v, disc_correction=True)


def load_1d_csv_ref_collection(
    filename: str, depth_column: str, value_columns: List[str] = None, **kwargs: Any
) -> Type[ReferenceCollection]:
    """

    loads a 1D reference model collection from a CSV file

    Parameters
    ----------
    filename : str
        filename
    depth_column : str
        the name of the depth column
    value_columns : List[str]
        list of columns to load as reference curves.
    **kwargs : Any
        all kwargs forwarded to pandas.read_csv()

    Returns
    -------
    ReferenceCollection

    Examples
    --------
    from ytgeotools.seismology.datasets import load_1d_csv_ref_collection
    import matplotlib.pyplot as plt
    import numpy as np

    refs = load_1d_csv_ref_collection("IRIS/refModels/AK135F_AVG.csv", 'depth_km')
    print(refs.reference_fields)

    depth_new = np.linspace(0, 500, 50000)
    vs = refs.Vs_kms.evaluate(depth_new)
    vp = refs.Vp_kms.evaluate(depth_new)
    rho = refs.rho_kgm3.evaluate(depth_new)

    f, ax = plt.subplots(1)
    ax.plot(vs, depth_new, label='V_s')
    ax.plot(refs.Vs_kms.vals, refs.Vs_kms.depth,'.k', label='V_s')
    ax.plot(vp, depth_new, label='V_p')
    ax.plot(refs.Vp_kms.vals, refs.Vp_kms.depth,'.k', label='V_p')
    ax.set_ylim(0, 500)
    ax.invert_yaxis()

    """
    filename = _dm.validate_file(filename)
    df = pd.read_csv(filename, **kwargs)
    d = df[depth_column].to_numpy()
    if value_columns is None:
        value_columns = [c for c in df.columns if c != depth_column]

    ref_mods = []
    for vcol in value_columns:
        vals = df[vcol].to_numpy()
        ref_mods.append(ReferenceModel1D(vcol, d, vals))

    return ReferenceCollection(ref_mods)
