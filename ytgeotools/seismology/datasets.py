import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Type
import xarray as xr
from unyt import unyt_quantity
from ytgeotools.ytgeotools import Dataset
from ytgeotools.coordinate_transformations import geosphere2cart
from scipy import spatial


class GeoSpherical(Dataset):
    def __init__(
        self,
        data: dict,
        latitude: ArrayLike,
        longitude: ArrayLike,
        depth: ArrayLike,
        coord_order: list = ["latitude", "longitude", "depth"],
        max_radius: float = 6371.0,
    ):

        self.max_radius = max_radius
        coord_init = {"latitude": latitude, "longitude": longitude, "depth": depth}
        coords = {
            idim: {"values": coord_init[dim], "name": dim}
            for idim, dim in enumerate(coord_order)
        }
        self.max_radius = max_radius
        self._interp_trees = {}
        super().__init__(data, coords)

    def cartesian_coords(self) -> tuple[ArrayLike]:
        "returns 3d arrays representing earth-centered cartesian coordinates of every grid point"

        cmesh = np.meshgrid(
            self.coords[0], self.coords[1], self.coords[2], indexing="ij"
        )

        radius = self.max_radius - cmesh[self._coord_hash_r["depth"]]
        lat = cmesh[self._coord_hash_r["latitude"]]
        lon = cmesh[self._coord_hash_r["longitude"]]

        return geosphere2cart(lat, lon, radius)

    def interpolate_to_uniform_cartesian(
        self,
        fields: list,
        N: int = 50,
        max_dist: Union[int, float] = 100,
        interpChunk: int = 500000,
        recylce_trees: bool = False,
        return_yt: bool = False,
    ):
        """
        moves geo-spherical data (radius/depth, lat, lon) to earth-centered
        cartesian coordinates using a kdtree with inverse distance weighting (IDW)


        max_dist : int or float
            the max distance away for nearest neighbor search (default 100)
        interpChunk : int
            the chunk size for querying the kdtree (default 500000)
        recylce_trees : boolean
            if True, will store the kdtree(s) generated (default False)
        """

        x, y, z = self.cartesian_coords()  # the actual xyz at which we have data

        # drop points at which we don't have data

        cart_bbox = np.array([[d.min(), d.max()] for d in [x, y, z]])

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
            interpd = query_trees(
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

        if return_yt:
            coords = {
                d: {"values": xyz[d], "name": dim} for d, dim in zip(range(3), "xyz")
            }
            return Dataset(
                interpd,
                coords,
            ).load_uniform_grid()

        if len(interpd) == 1:
            interpd = interpd[fields[0]]
        return *xyz, interpd


def query_trees(
    xdata, ydata, zdata, interpChunk, fields, trees, interpd, orig_shape, max_dist
):

    # query the tree at each new grid point and weight nearest neighbors
    # by inverse distance. proceed in chunks.
    N_grid = len(xdata)
    print("querying kdtree on interpolated grid")
    chunk = interpChunk
    N_chunks = int(N_grid / chunk) + 1
    print("breaking into " + str(N_chunks) + " chunks")
    for i_chunk in range(0, N_chunks):
        print("   processing chunk " + str(i_chunk + 1) + " of " + str(N_chunks))
        i_0 = i_chunk * chunk
        i_1 = i_0 + chunk
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
        max_radius: Type[unyt_quantity] = unyt_quantity(6371.0, "km"),
    ):

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
                        f"all fields must have the same coordinate ordering but {f} does not."
                        " Please provide a field_subset containing only fields that have "
                        "the same coordinate ordering."
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
        )
