from tslearn.clustering import TimeSeriesKMeans
from dask import delayed, compute
from ytgeotools.mapping import default_crs
from geopandas import GeoDataFrame, points_from_xy
import numpy as np


class ProfileCollection:
    def __init__(self, profiles, depth, x, y, crs=default_crs):
        self.profiles = profiles
        self.depth = depth
        self.x = x
        self.y = y
        self.crs = crs


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
    def __init__(
        self,
        profile_collection,
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
        self._fit_exists = False

    def fit(self):
        super().fit(self.profile_collection.profiles)
        self._fit_exists = True

    @requires_fit
    def get_classified_coordinates(self):
        p = self.profile_collection
        return GeoDataFrame(
            {"labels": self.labels_}, geometry=points_from_xy(p.x, p.y), crs=p.crs
        )

    @requires_fit
    def get_stats_in_labels(self, df_gpd):
        df = self.get_classified_coordinates()

    @requires_fit
    def depth_stats(self):

        cstats = {}
        for lab in range(self.n_clusters):
            label_mask = self.labels_ == lab
            vals = self.profile_collection.profiles[label_mask, :]

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
