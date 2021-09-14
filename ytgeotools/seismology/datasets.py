import numpy as np
from numpy.typing import ArrayLike
from typing import Union, Type
import xarray as xr
from unyt import unyt_quantity
from yt import load_uniform_grid as lug
from ytgeotools.coordinate_transformations import geosphere2cart
from ytgeotools.ytgeotools import Dataset


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

    def interpolate_to_uniform_cartesian(self, field_subset: list = None, N: int = 500):

        x, y, z = self.cartesian_coords()  # the actual xyz at which we have data

        # drop points at which we don't have data

        cart_bbox = np.array([[d.min(), d.max()] for d in [x, y, z]])

        wids = np.abs(cart_bbox[:, 1] - cart_bbox[:, 0])
        dx = wids.min() / N
        Ngrid = np.floor(wids / dx)

        xyz_int = [
            np.linspace(cart_bbox[0, d], cart_bbox[1, d], Ngrid[d]) for d in range(3)
        ]

        # interpolate the field data from x, y, z to xyz_int


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


def load_uniform_grid(ds: Type[Dataset], *args, **kwargs):

    data = ds.data_dict()
    sizes = data[list(data.keys())[0]].shape
    dims = tuple(ds._coord_order)
    geometry = (ds.geometry, (dims))
    return lug(data, sizes, 1.0, *args, bbox=ds.bbox, geometry=geometry, **kwargs)


# ds = yt.load_uniform_grid(data, sizes, 1.0, geometry=("internal_geographic", dims),
#                           bbox=bbox)
