from tslearn.clustering import TimeSeriesKMeans
from dask import delayed, compute
from ytgeotools.mapping import default_crs
from geopandas import GeoDataFrame, points_from_xy
import numpy as np

def fit_kmeans(profiles, n_clusters, max_iter=10, metric="euclidean"):
    model = TimeSeriesKMeans(n_clusters=n_clusters,
                             metric=metric,
                             max_iter=max_iter)
    model.fit(profiles)

    return model


class ProfileCollection:

    def __init__(self, profiles, depth, x, y, crs=default_crs):
        self.profiles = profiles
        self.depth = depth
        self.x = x
        self.y = y
        self.crs = crs

    def get_classified_coords(self, kmeans_model):

        labs = kmeans_model.labels_

        df = GeoDataFrame({"labels": labs},
                          geometry=points_from_xy(self.x, self.y),
                          crs=self.crs)

        return df

    def kmeans_depth_stats(self, kmeans_model):

        cstats = {}
        for lab in range(kmeans_model.n_clusters):
            label_mask = kmeans_model.labels_ == lab
            vals = self.profiles[label_mask, :]

            cval = kmeans_model.cluster_centers_[lab, :].squeeze()
            diff = [vals[:, i] - cval[i] for i in range(len(cval))]
            diff = np.array(diff).transpose()
            mindiff = diff.min(axis=0)
            maxdiff = diff.max(axis=0)
            stdvals = np.std(self.profiles, axis=0)
            labstats = {
                "std": stdvals,
                "cluster_center": cval,
                "two_sigma_min": cval - stdvals * 2.,
                "two_sigma_max": cval + stdvals * 2.,
                "one_sigma_min": cval - stdvals,
                "one_sigma_max": cval + stdvals,
            }
            cstats[lab] = labstats

        return cstats

    def fit_kmeans(self, n_clusters, max_iter=10, metric="euclidean"):
        model = fit_kmeans(self.profiles,
                           n_clusters,
                           max_iter=max_iter,
                           metric=metric,
                           )
        return model

    def multi_kmeans_fit(self, cluster_range, max_iter=10, metric="euclidean"):

        models = []

        for n_clusters in cluster_range:
            models.append(
                            delayed(fit_kmeans)(self.profiles,
                                                n_clusters,
                                                max_iter=max_iter,
                                                metric=metric,
                                               )
                          )

        computed_models = compute(*models)
        inertia = [c.inertia_ for c in computed_models]

        return computed_models, inertia





