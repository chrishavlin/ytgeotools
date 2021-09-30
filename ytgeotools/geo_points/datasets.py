import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame
from abc import ABC, abstractmethod

from ytgeotools.data_manager import data_manager as _dm
from ytgeotools.mapping import BoundingPolies, default_crs, validate_lons
from typing import Tuple


def _apply_filter(df, filter: dict):
    col = filter["column"]
    if filter["comparison"] == "==":
        df = df[df[col] == filter["value"]]
    elif filter["comparison"] == "<=":
        df = df[df[col] <= filter["value"]]
    elif filter["comparison"] == "<":
        df = df[df[col] < filter["value"]]
    elif filter["comparison"] == ">=":
        df = df[df[col] >= filter["value"]]
    elif filter["comparison"] == ">":
        df = df[df[col] > filter["value"]]
    return df


class _GeoPoint(ABC):
    @abstractmethod
    def load_data(self):
        pass


class CSVData(_GeoPoint):

    file_sep = ","

    def __init__(
        self,
        filename: str,
        use_neg_lons: bool = False,
        initial_filters: list = None,
        drop_duplicates_by: list = None,
        lonname: str = "longitude",
        latname: str = "latitude",
        file_sep: str = ",",
        crs: dict = default_crs,
    ):
        """
        load an on-disk csv file of geo-referenced points.

        Parameters
        ----------
        filename : str
            csv file to load
        use_neg_lons : bool
            if False (default), enforces longitude values are in 0, 360
        initial_filters : list
            list of filter-dictionaries to apply after initial load. Default is
            None. Should have the form:

            [
                {"column": "age", "value":100, "comparison": "<="},
                {"column": "rock_name", "value":"RHYOLITE", "comparison": "=="},
            ]

            Filters will be applied in the order supplied.

        drop_duplicates_by : list
            list of columns to drop duplicates by, default None
        lonname : str
            the on-disk name for the longitude column (default "longitude").
            This will get renamed to "longitude" in the loaded dataframe.
        latname : str
            the on-disk name for the latitude column (default "latitude")
            This will get renamed to "latitude" in the loaded dataframe.
        file_sep : str
            the file separator, default ","
        crs : dict
            the coordinate reference system dictionary, default is
            ytgeotools.mapping.default_crs
        """
        self.crs = crs
        self.file = _dm.validate_file(filename)
        if drop_duplicates_by is None:
            drop_duplicates_by = []
        self.drop_duplicates_by = drop_duplicates_by
        self.file_sep = file_sep
        self.filters = []

        if initial_filters is not None:
            self.filters += initial_filters

        self.use_neg_lons = use_neg_lons
        self.lonname = lonname
        self.latname = latname
        self.df, self.bounds = self.load_data()

    def load_data(self, filters: list = None, **kwargs):

        if filters is None:
            filters = []

        df = pd.read_csv(self.file, sep=self.file_sep, low_memory=False)
        df = df.rename(columns={self.lonname: "longitude", self.latname: "latitude"})

        lonvals = df["longitude"].values
        lonvals = validate_lons(lonvals, use_negative_lons=self.use_neg_lons)
        df["longitude"] = lonvals

        if self.drop_duplicates_by:
            df = df.drop_duplicates(subset=self.drop_duplicates_by)

        for filter_dict in self.filters + filters:
            df = _apply_filter(df, filter_dict)

        dims = "latitude", "longitude"
        bounds = {dim: [df[dim].min(), df[dim].max()] for dim in dims}

        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs=self.crs,
        )

        return df, bounds


class EarthChem(CSVData):
    def __init__(
        self,
        filename: str,
        use_neg_lons: bool = False,
        initial_filters: list = None,
        drop_duplicates_by: list = ["latitude", "longitude", "age"],
        lonname: str = "lon",
        latname: str = "lat",
    ):
        super().__init__(
            filename,
            use_neg_lons=use_neg_lons,
            initial_filters=initial_filters,
            drop_duplicates_by=drop_duplicates_by,
            lonname=lonname,
            latname=latname,
            file_sep="|",
        )

    def build_volcanic_extent(self, boundary_df=None, radius_deg=0.5):
        """builds volcanic extent (bounding polygon of all volcs)

        Parameters
        ----------
        boundary_df : GeoDataFrame
            the GeoDataFrame to initially limit the volcanic data.
        radius_deg : float
            the radius from each volcanic point, in degrees (default 0.5)

        Returns
        -------
        tuple, (df,df_gp,volc_bound)
            df : GeoDataFrame, the volcanic data within boundary_df
            df_gp: GeoDataFrame, same as df but with polygons as geometry
            volc_bound: GeoSeries, the union of all polygons in df_gp

        """

        boundingPoly = BoundingPolies(self.df,
                                      b_df=boundary_df,
                                      radius_deg=radius_deg)
        vbf = boundingPoly.df_bound
        vbf = GeoDataFrame(geometry=vbf.geometry, crs=self.crs)

        return boundingPoly.df, boundingPoly.df_gp, vbf
