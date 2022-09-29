from typing import List

import numpy as np
import xarray as xr

from ytgeotools.mapping import build_bounding_df


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


def correct_lon_value(lon):
    if lon < 0:
        lon = lon + 360
    if lon > 360.0:
        lon = lon - 360.0
    return lon


def geo_df_for_testing(
    center_lon=-120.0, center_lat=40, dlon=60.0, dlat=50.0, correct_neg_lons=True
):
    maxlat = center_lat + dlat / 2.0
    minlat = center_lat - dlat / 2.0
    maxlon = center_lon + dlon / 2.0
    minlon = center_lon - dlon / 2.0
    if correct_neg_lons:
        maxlon = correct_lon_value(maxlon)
        minlon = correct_lon_value(minlon)

    lats = (minlat, maxlat, maxlat, minlat)
    lons = (minlon, minlon, maxlon, maxlon)

    df, _ = build_bounding_df(lats, lons)
    return df
