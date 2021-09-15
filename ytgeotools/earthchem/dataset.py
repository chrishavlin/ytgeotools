import pandas as pd
import geopandas as gpd
from ytgeotools.mapping import BoundingPolies, default_crs


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


class Dataset(object):
    def __init__(
        self,
        filename,
        use_neg_lons=False,
        initial_filters=None,
        drop_duplicates_by=["lat", "lon", "age"],
    ):
        self.file = filename
        self.drop_duplicates_by = drop_duplicates_by

        self.filters = []
        # format of filter
        # filters = [
        #     {"column": "age", "value":100, "comparison": "<="},
        #     {"column": "rock_name", "value":"RHYOLITE", "comparison": "=="},
        # ]

        if initial_filters is not None:
            self.filters += initial_filters

        self.use_neg_lons = use_neg_lons
        self.df, self.bounds = self.loadData()

    def loadData(self, filters: list = None):

        if filters is None:
            filters = []

        df = pd.read_csv(self.file, sep="|", low_memory=False)

        if self.drop_duplicates_by:
            df = df.drop_duplicates(subset=self.drop_duplicates_by)

        for filter_dict in self.filters + filters:
            df = _apply_filter(df, filter_dict)

        bounds = {
            "lat": [df.lat.min(), df.lat.max()],
            "lon": [df.lon.min(), df.lon.max()],
        }
        if self.use_neg_lons:
            df.loc[df.lon > 180, "lon"] = df.loc[df.lon > 180, "lon"] - 360.0
        else:
            df.loc[df.lon < 0, "lon"] = df.loc[df.lon < 0, "lon"] + 360.0

        df = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=default_crs
        )

        return df, bounds

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

        boundingPoly = BoundingPolies(self.df, b_df=boundary_df, radius_deg=radius_deg)

        return boundingPoly.df, boundingPoly.df_gp, boundingPoly.df_bound
