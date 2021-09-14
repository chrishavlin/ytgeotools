from ytgeotools import data_manager as dm
from shapely.geometry import Polygon, Point
from shapely import affinity as aff
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import cascaded_union

default_crs = {"init": "epsg:4326"}
preferredproj = "robinson"


def setDb(db=None):
    if type(db) is str or db is None:
        return dm.filesystemDB(dataDir=db)
    else:
        return db


class ColoradoPlateau(object):
    def __init__(self, db=None, CP_scales=[1], smooth_fac=None, use_neg_lons=False):
        self.db = setDb(db)
        self.file = self.db.fullpath("ColoradoPlateauBoundary.csv")
        self.CP = pd.read_csv(self.file)
        if use_neg_lons:
            self.CP.loc[self.CP.lon > 180, "lon"] = (
                self.CP.loc[self.CP.lon > 180, "lon"] - 360.0
            )
        else:
            self.CP.loc[self.CP.lon < 0, "lon"] = (
                self.CP.loc[self.CP.lon < 0, "lon"] + 360.0
            )
        (self.CP_gpd, self.CP_polies) = self.buildCPgpd(CP_scales, smooth_fac)

    def buildCPgpd(self, CP_scales=[1], smooth_fac=None):
        poly = Polygon([[p[0], p[1]] for p in zip(self.CP.lon, self.CP.lat)])
        self.CP_raw_poly = poly
        # Dom_gp= gpd.GeoDataFrame(crs=crs)

        gpd_rows = []
        polies = []
        sid = 0
        for sc in CP_scales:
            CP_sc = aff.scale(poly, xfact=sc, yfact=sc)
            if smooth_fac is not None:
                CP_sc = CP_sc.buffer(smooth_fac, join_style=1).buffer(
                    -smooth_fac, join_style=1
                )
            polies.append(CP_sc)
            # Dom_gp=Dom_gp.append({'shape_id':sid,'geometry':CP_sc,'sc':sc},ignore_index=True)
            gpd_rows.append({"shape_id": sid, "geometry": CP_sc, "sc": sc})
            sid = sid + 1

        Dom_gp = gpd.GeoDataFrame(gpd_rows, crs=crs)
        return Dom_gp, polies


def build_bounding_df(latitudes, longitudes, crs = {"init": "epsg:4326"}, description="bounding_poly"):

    poly = Polygon([[p[0], p[1]] for p in zip(longitudes, latitudes)])

    gpd_rows = [
        {"shape_id": 0, "geometry": poly, "description": description}
    ]
    return gpd.GeoDataFrame(gpd_rows, crs=crs), poly



class BoundingPolies(object):
    """class for processing bounding polygons of point data

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        dataframe of point data, must have 'lat' and 'lon' fields
    b_df : GeoDataFrame
        bounding GeoDataFrame to limit df initially, can be None (default)
    **kwargs
        'radius_deg' float
            the radius for circles around each point

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

    def __init__(self, df, b_df=None, radius_deg=0.5, crs=default_crs):
        self.df_raw = df
        self.b_df = b_df

        self.crs = crs
        self._buildExtents(radius_deg=radius_deg)

    def _buildExtents(self, radius_deg=0.5):
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
            df = filterByBounds(df, self.b_df)
            self.df = df
        else:
            self.df = self.df_raw
            df = self.df

        polies = []
        for rowid, row in df.iterrows():
            polies.append(
                Polygon(circ(row["lat"], row["lon"], radius_deg, radius_deg / 10))
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


def filterByBounds(df, b_df, return_interior=True, crs=default_crs):
    """finds dataframe points within polygons of a second dataframe

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        dataframe of points, can be pandas or geopandas dataframe.
    b_df : GeoDataFrame
        dataframe of polygons. If None (default), will use Colorado Plateau
        with affinity scale of 2, smoothing factor of 5.
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
        return df_s[~pd.isnull(df_s["shape_id"])]
    else:
        return df_s
