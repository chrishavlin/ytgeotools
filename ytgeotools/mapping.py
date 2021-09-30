import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import affinity as aff
from shapely.geometry import Point, Polygon
from shapely.ops import cascaded_union
from pandas import isnull as pd_isnull
from ytgeotools.data_manager import data_manager

default_crs = {"init": "epsg:4326"}


def validate_lons(lons, use_negative_lons=False):
    if use_negative_lons:
        lons[lons > 180] = lons[lons > 180] - 360.0
    else:
        lons[lons < 0] = lons[lons < 0] + 360.0
    return lons


class OnDiskGeometry:
    def __init__(
        self,
        filename: str,
        crs: dict = default_crs,
        ftype: str = None,
        latname: str = "latitude",
        lonname: str = "longitude",
        use_negative_lons: bool = False,
    ):
        self.filename = data_manager.validate_file(filename)
        self.crs = crs
        self.use_negative_lons = use_negative_lons
        self.lonname = lonname
        self.latname = latname

        if ftype is None:
            ftype = os.path.splitext(self.filename)[-1].replace(".", "")

        read_engines = {
            "csv": self.read_csv,
        }
        self._read = read_engines[ftype]
        self.ftype = ftype

    def read_csv(self, *args, **kwargs):
        return pd.read_csv(self.filename, *args, **kwargs)

    def _validate_lons(self):
        self.df[self.lonname] = validate_lons(
            self.df[self.lonname], self.use_negative_lons
        )


class PolygonFile(OnDiskGeometry):
    def __init__(
        self,
        filename: str,
        *args,
        crs: dict = default_crs,
        ftype: str = None,
        latname: str = "latitude",
        lonname: str = "longitude",
        use_negative_lons: bool = False,
        description: str = None,
        smooth_factor: int = 1,
        affine_scale: int = 1,
        **kwargs
    ):

        super().__init__(
            filename,
            crs=crs,
            ftype=ftype,
            latname=latname,
            lonname=lonname,
            use_negative_lons=use_negative_lons,
        )

        self.df = self._read(*args, **kwargs)
        self._validate_lons()

        if description is None:
            description = filename
        self.description = description

        self.bounding_polygon = self.build_gpd_df(smooth_factor, affine_scale)

    def build_gpd_df(self, smooth_factor: float = 1, affine_scale: float = 1):

        poly = Polygon(
            [[p[0], p[1]] for p in zip(self.df[self.lonname], self.df[self.latname])]
        )

        if affine_scale != 1:
            poly = aff.scale(poly, xfact=affine_scale, yfact=affine_scale)

        if smooth_factor != 1:
            poly = poly.buffer(smooth_factor, join_style=1).buffer(
                -smooth_factor, join_style=1
            )

        gpd_rows = []
        gpd_rows.append({"geometry": poly, "description": self.description})
        gpd_df = gpd.GeoDataFrame(gpd_rows, crs=self.crs)
        return gpd_df


def build_bounding_df(
    latitudes, longitudes, crs={"init": "epsg:4326"}, description="bounding_poly"
):

    poly = Polygon([[p[0], p[1]] for p in zip(longitudes, latitudes)])

    gpd_rows = [{"shape_id": 0, "geometry": poly, "description": description}]
    return gpd.GeoDataFrame(gpd_rows, crs=crs), poly


class BoundingPolies(object):
    """class for processing bounding polygons of point data

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        dataframe of point data
    b_df : GeoDataFrame
        bounding GeoDataFrame to limit df initially, can be None (default)
    radius_deg : float
        the radius for circles around each point
    lonname: str
        name of longitude field in the dataframe (default "longitude")
    latname: str
        name of latitude field in the dataframe (default "latitude")
    crs : dict
        coordinate reference dictionary, default is ytgeotools.mapping.default_crs

    Attributes
    ----------
    df_raw : DataFrame or GeoDataFrame
        the input dataframe
    df : GeoDataFrame
        the input dataframe after initial filter by b_df
    df_gp : GeoDataFrame
        dataframe containing polygons for each point
    df_bound : GeoSeries
        the union of polygons in df_gp

    """

    def __init__(
        self,
        df,
        b_df=None,
        radius_deg=0.5,
        crs=default_crs,
        lonname="longitude",
        latname="latitude",
    ):
        self.df_raw = df
        self.b_df = b_df

        self.crs = crs
        self.lonname = lonname
        self.latname = latname
        self._build_extents(radius_deg=radius_deg)

    def _build_extents(self, radius_deg=0.5):
        """builds bounding extent (bounding polygon of all points)

        Parameters
        ----------
        radius_deg : float
            the radius from each center point, in degrees (default 0.5)

        Returns
        -------
        tuple, (df,df_gp,volc_bound)
            df : GeoDataFrame, the volcanic data within boundary_df
            df_gp: GeoDataFrame, same as df but with polygons as geometry
            volc_bound: GeoSeries, the union of all polygons in df_gp

        """
        # ceate circle of radius radius_deg for every point, finds union
        # of all

        if self.b_df is not None:
            df = self.df_raw.copy(deep=True)
            df = filter_by_bounds(df, self.b_df)
            self.df = df
        else:
            self.df = self.df_raw
            df = self.df

        polies = []
        lons = self.lonname
        lats = self.latname
        for rowid, row in df.iterrows():
            polies.append(
                Polygon(circ(row[lats], row[lons], radius_deg, radius_deg / 10))
            )

        self.df_gp = gpd.GeoDataFrame(geometry=polies, crs=self.crs)
        self.df_bound = gpd.GeoSeries(cascaded_union(polies))


def circ(c_lat_y, c_lon_x, radius_deg, max_arc_length_deg=0.01):
    """builds circle Polygon around a center lat/lon point

    Parameters
    ----------
    c_lat_y : float
        center latitude, degrees
    c_lon_x : float
        center longitude, degrees
    radius_deg : float
        radius of circle in degrees
    max_arc_length_deg : float
        max arc length between points on circle (the default is .01).

    Returns
    -------
    Polygon
        shapely polygon built from points on circle

    """
    circumf = 2 * np.pi * radius_deg
    Npts = int(circumf / max_arc_length_deg)
    angle = np.linspace(0, 2 * np.pi, Npts)
    lat_pts = c_lat_y + radius_deg * np.sin(angle)
    lon_pts = c_lon_x + radius_deg * np.cos(angle)
    return Polygon(zip(lon_pts, lat_pts))


def filter_by_bounds(df, b_df, return_interior=True, crs=default_crs):
    """finds dataframe points within polygons of a second dataframe

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        dataframe of points, can be pandas or geopandas dataframe.
    b_df : GeoDataFrame
        dataframe of polygons.
    return_interior : boolean
        will only return points inside bounding polygons if True (default)

    Returns
    -------
    DataFrame
        a right join on the boundary dataframe, 'shape_id' column will be null
        for points outside the bounding polygons unless return_interior is
        True

    """

    # create geodataframe of raw points
    if type(df) == pd.DataFrame:
        geo = [Point([p[0], p[1]]) for p in zip(df["lon"], df["lat"])]
        df_gpd = gpd.GeoDataFrame(df, crs=crs, geometry=geo)
    elif type(df) == gpd.GeoDataFrame:
        df_gpd = df

    # spatial join of the two geodataframes
    df_s = gpd.sjoin(b_df, df_gpd, how="right", op="intersects")
    if return_interior:
        return df_s[~pd.isnull(df_s["index_left"])]
    else:
        return df_s


def successive_joins(df_left, df_right_list, drop_null=False, drop_inside=False):

    df = df_left.copy()

    if type(drop_null) is bool:
        drop_null = [drop_null] * len(df_right_list)

    if type(drop_inside) is bool:
        drop_inside = [drop_inside] * len(df_right_list)

    for df_r, dnull, dins in zip(df_right_list, drop_null, drop_inside):

        if dnull and dins:
            raise ValueError("Only one of drop_na and drop_inside can be True")

        df = gpd.sjoin(df, df_r, how="left", op="intersects")

        if dnull:
            df = df[~pd_isnull(df["index_right"])]
        if dins:
            df = df[pd_isnull(df["index_right"])]

        if len(df_right_list) > 1:
            df = df.drop(columns=["index_right"])

    return df
