from typing import Optional, Type

import numpy as np
from dask import compute, delayed
from geopandas import GeoDataFrame, points_from_xy, sjoin
from tslearn.clustering import TimeSeriesKMeans

from ytgeotools.mapping import BoundingPolies, default_crs


class ProfileCollection:
    def __init__(self, profiles, depth, x, y, crs=default_crs):
        self.profiles = profiles
        self.depth = depth
        self.x = x
        self.y = y
        self.crs = crs

    _surface_df = None

    @property
    def surface_df(self):
        if self._surface_df is None:

            df = GeoDataFrame(
                {"longitude": self.x, "latitude": self.y},
                geometry=points_from_xy(self.x, self.y),
                crs=self.crs,
            )
            self._surface_df = df
        return self._surface_df

    def get_surface_union(self, *args, **kwargs):
        bp = BoundingPolies(self.surface_df, *args, **kwargs)
        return GeoDataFrame(geometry=bp.df_bound.geometry, crs=self.crs)

    _use_negative_lons = False

    def toggle_negative_lons(self, new_value: Optional[bool] = None):
        self._surface_df = None
        if new_value is not None:
            self._use_negative_lons = new_value
        else:
            self._use_negative_lons = not self._use_negative_lons

        xvals = self.x
        if self._use_negative_lons:
            xmask = xvals > 180.0
            xvals[xmask] = xvals[xmask] - 360.0
        else:
            xmask = xvals < 0.0
            xvals[xmask] = xvals[xmask] + 360.0
        self.x = xvals


def fit_kmeans(profile_collection, n_clusters=3, **kwargs):
    """
    instantiate and fit a DepthSeriesKMeans data

    Parameters
    ----------
    profile_collection : ProfileCollection
        an instance of a ProfileCollection
    n_clusters : int
        clusters to use
    kwargs
        any kwarg to DepthSeriesKMeans

    Returns
    -------
    DepthSeriesKMeans
        an instance of DepthSeriesKMeans after running fit()

    """
    kmeans_model = DepthSeriesKMeans(
        profile_collection, n_clusters=n_clusters, **kwargs
    )
    kmeans_model.fit()
    return kmeans_model


def requires_fit(func):
    def wrapper(*args, **kwargs):
        if args[0]._fit_exists is False:
            raise ValueError("You must run model.fit() before using this method.")
        return func(*args, **kwargs)

    return wrapper


class DepthSeriesKMeans(TimeSeriesKMeans):

    depth_range = None
    depth_mask = None
    depth = None

    def __init__(
        self,
        profile_collection: Type[ProfileCollection],
        depth_min=None,
        depth_max=None,
        radius_deg=0.2,
        n_clusters=3,
        max_iter=50,
        tol=1e-06,
        n_init=1,
        metric="euclidean",
        max_iter_barycenter=100,
        metric_params=None,
        n_jobs=None,
        dtw_inertia=False,
        verbose=0,
        random_state=None,
        init="k-means++",
    ):
        super().__init__(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            n_init=n_init,
            metric=metric,
            max_iter_barycenter=max_iter_barycenter,
            metric_params=metric_params,
            n_jobs=n_jobs,
            dtw_inertia=dtw_inertia,
            verbose=verbose,
            random_state=random_state,
            init=init,
        )

        self.profile_collection = profile_collection
        self._set_depth_attrs(depth_min, depth_max)
        self._fit_exists = False
        self.radius_deg = radius_deg

    def _set_depth_attrs(self, depth_min, depth_max):

        d = self.profile_collection.depth
        if depth_min is None:
            depth_min = d.min()
        if depth_max is None:
            depth_max = d.max()

        self.depth_range = (depth_min, depth_max)
        self.depth_mask = (d >= depth_min) & (d <= depth_max)
        self.depth = d[self.depth_mask]

    def fit(self):
        super().fit(self.profile_collection.profiles[:, self.depth_mask])
        self._fit_exists = True

    @requires_fit
    def get_classified_coordinates(self):
        p = self.profile_collection
        return GeoDataFrame(
            {"labels": self.labels_, "latitude": p.y, "longitude": p.x},
            geometry=points_from_xy(p.x, p.y),
            crs=p.crs,
        )

    _bounding_polygons = None

    @property
    def bounding_polygons(self):
        if self._bounding_polygons is None:
            df = self.get_classified_coordinates()
            geoms = []
            geomdata = {"label": []}
            for iclust in range(self.n_clusters):
                df_members = df[df.labels == iclust]
                bounds = BoundingPolies(df_members, radius_deg=self.radius_deg)
                geoms.append(bounds.df_bound.geometry[0])
                geomdata["label"].append(iclust)

            crs = self.profile_collection.crs
            df = GeoDataFrame(geomdata, geometry=geoms, crs=crs)
            self._bounding_polygons = df
        return self._bounding_polygons

    def set_bounding_radius(self, radius_deg):
        if radius_deg != self.radius_deg:
            self._bounding_polygons = None
        self.radius_deg = radius_deg

    @requires_fit
    def classify_points(self, df_gpd):
        b_df = self.bounding_polygons
        return sjoin(df_gpd, b_df, how="left", op="intersects")

    @requires_fit
    def depth_stats(self):

        cstats = {}
        for lab in range(self.n_clusters):
            label_mask = self.labels_ == lab
            p = self.profile_collection.profiles
            vals = p[label_mask, :][:, self.depth_mask]
            cval = self.cluster_centers_[lab, :].squeeze()
            stdvals = np.std(vals, axis=0)
            labstats = {
                "std": stdvals,
                "cluster_center": cval,
                "two_sigma_min": cval - stdvals * 2.0,
                "two_sigma_max": cval + stdvals * 2.0,
                "one_sigma_min": cval - stdvals,
                "one_sigma_max": cval + stdvals,
            }
            cstats[lab] = labstats

        return cstats

    def multi_kmeans_fit(
        self, cluster_range, max_iter=50, metric="euclidean", **kwargs
    ):
        """

        Parameters
        ----------
        cluster_range : array_like
            the clusters to run with
        max_iter : int
            max iterations, used across all, default 50
        metric : str
            metric used, default "euclidean"
        **kwargs
            any other kwarg for DepthSeriesKMeans initialization


        Returns
        -------

        """
        models = []

        for n_clusters in cluster_range:
            models.append(
                delayed(fit_kmeans)(
                    self.profile_collection,
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    metric=metric,
                    **kwargs,
                )
            )

        computed_models = compute(*models)
        inertia = [c.inertia_ for c in computed_models]

        return computed_models, inertia
