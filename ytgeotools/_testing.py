from typing import List

import numpy as np
import xarray as xr


def create_fake_ds(fields: List[str] = None):

    if fields is None:
        fields = ["dvs"]

    nlon = 10
    nlat = 11
    ndepth = 15
    lons = np.linspace(-180.0, 180.0, nlon)
    lats = np.linspace(-90.0, 90.0, nlat)
    depths = np.linspace(60, 660.0, ndepth)

    dim_order = ("depth", "latitude", "longitude")
    field_dict = {
        field: (dim_order, np.random.random((ndepth, nlat, nlon))) for field in fields
    }

    return xr.Dataset(
        field_dict, {"depth": depths, "latitude": lats, "longitude": lons}
    )


def save_fake_ds(filename, *args, **kwargs):
    ds = create_fake_ds(*args, **kwargs)
    ds.to_netcdf(filename)
