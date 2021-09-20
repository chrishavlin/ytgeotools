from tslearn.clustering import TimeSeriesKMeans


class ProfileCollection:

    def __init__(self, profiles):
        self.profiles = profiles

    def fit_kmeans(self, n_clusters, max_iter=10, metric="euclidean"):
        model = TimeSeriesKMeans(n_clusters=n_clusters,
                                 metric=metric,
                                 max_iter=max_iter)
        model.fit(self.profiles)
        return model





