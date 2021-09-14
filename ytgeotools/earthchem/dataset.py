import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
import numpy as np
from . import mapping as MS, pointData as pdd, dataManager as dm


class volc(object):
    def __init__(self, dataDir=None, use_neg_lons=False, **kwargs):
        self.db = dm.filesystemDB(dataDir)
        self.file = self.db.fullpath("earthchem_download_90561.csv")
        self.max_age_Ma = kwargs.get("max_age_Ma", None)
        self.min_age_Ma = kwargs.get("min_age_Ma", None)
        self.age_sigma_Ma = kwargs.get("age_sigma_Ma", None)
        self.use_neg_lons = use_neg_lons

    def loadData(self, max_age_Ma=None, min_age_Ma=None, age_sigma_Ma=None):
        df = pd.read_csv(self.file, sep="|", low_memory=False)
        df = df.drop_duplicates(subset=["lat", "lon", "age"])
        # limit by age
        # note: 'age' column is the calculate age, 'age_min' and 'age_max' are the
        # min/max ages for the calculated age. The age_min, age_max inputs to
        # this function act on 'age' column.
        if max_age_Ma is not None:
            df = df[df["age"] <= max_age_Ma]
        if min_age_Ma is not None:
            df = df[df["age"] >= min_age_Ma]
        df["dt_sigma"] = df["age_max"] - df["age_min"]  # age uncertainty
        if age_sigma_Ma is not None:
            df = df[df["dt_sigma"] <= age_sigma_Ma]

        self.bounds = {
            "lat": [df.lat.min(), df.lat.max()],
            "lon": [df.lon.min(), df.lon.max()],
        }
        if self.use_neg_lons:
            df.loc[df.lon > 180, "lon"] = df.loc[df.lon > 180, "lon"] - 360.0
        else:
            df.loc[df.lon < 0, "lon"] = df.loc[df.lon < 0, "lon"] + 360.0
        self.df = df

    def setCPfilter(self, CP_scales=[2], smooth_fac=5):
        self.CP = MS.CP(db=self.db, CP_scales=CP_scales, smooth_fac=smooth_fac)

    def buildVolcExtent(self, radius_deg=0.5, boundary_df=None):
        """builds volcanic extent (bounding polygon of all volcs)

        Parameters
        ----------
        radius_deg : float
            the radius from each volcanic point, in degrees (default 0.5)
        boundary_df : GeoDataFrame or None
            If a GeoDataFrame, will use this to find the outer bound to limit
            the volcanic data. If None, will use Colorado Plateau with scale
            of 2, smoothing factor of 5 (default None).

        Returns
        -------
        tuple, (df,df_gp,volc_bound)
            df : GeoDataFrame, the volcanic data within boundary_df
            df_gp: GeoDataFrame, same as df but with polygons as geometry
            volc_bound: GeoSeries, the union of all polygons in df_gp

        """
        # ceate circle of radius radius_deg for every volc center, find union
        # of all
        if hasattr(self, "df") is False:
            self.loadData(self.max_age_Ma, self.min_age_Ma, self.age_sigma_Ma)
        df = self.df

        if boundary_df is None:
            CP = MS.CP(db=self.db, CP_scales=[2], smooth_fac=5)
            boundary_df = CP.CP_gpd

        boundingPoly = pdd.boundingPolies(df, boundary_df, radius_deg=radius_deg)

        return (boundingPoly.df, boundingPoly.df_gp, boundingPoly.df_bound)
