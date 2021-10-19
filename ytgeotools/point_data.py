"""
class for processing point data
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from sklearn.cluster import KMeans


class pointData(object):
    def __init__(self, df=None, xname="x", yname="y"):
        self.df = df
        self.xname = xname
        self.yname = yname

        return

    def create2dGrid(self, dx, dy, xmin, xmax, ymin, ymax):
        """creates a 2d grid"""
        Nx = int(np.ceil((xmax - xmin) / dx))
        Ny = int(np.ceil((ymax - ymin) / dy))

        # grid nodes
        x = np.linspace(xmin, xmax, Nx + 1)
        y = np.linspace(ymin, ymax, Ny + 1)
        setattr(self, self.xname, x)
        setattr(self, self.yname, y)

        # cell centers
        x_c = (x[1:] + x[0:-1]) / 2
        y_c = (y[1:] + y[0:-1]) / 2
        setattr(self, self.xname + "_c", x_c)
        setattr(self, self.yname + "_c", y_c)

    def assignDfToGrid(self, binfields=None):
        """finds stats within bins. binfield is the field to bin"""

        if binfields is None:
            binfields = []

        if hasattr(self, self.xname):
            xedges = getattr(self, self.xname)
            yedges = getattr(self, self.yname)

            # initialize
            gridded = {}

            for binfield in binfields:
                gridded[binfield] = {}
                stats_list = ["mean", "median", "max", "min", "std", "count"]
                Nx = xedges.size - 1
                Ny = yedges.size - 1
                for stat in stats_list:
                    gridded[binfield][stat] = np.zeros((Nx, Ny))
                    gridded[binfield][stat][:] = np.nan

            # restrict to grid min/max
            df0 = self.df[
                (self.df[self.xname] >= xedges.min())
                & (self.df[self.xname] <= xedges.max())
            ]
            df0 = df0[
                (df0[self.yname] >= yedges.min()) & (df0[self.yname] <= yedges.max())
            ]

            # need stats beyond mean, hist2d won't work. Loop over 1 spatial dim,
            # use pandas cut
            for i_x in range(0, Nx):

                # find all values within this x
                x1 = xedges[i_x]
                x2 = xedges[i_x + 1]
                df = df0[(df0[self.xname] >= x1) & (df0[self.xname] < x2)]

                if len(df) > 0:

                    # cut and aggregate along y at this x
                    bins = pd.cut(
                        df[self.yname], yedges, include_lowest=True, right=True
                    )

                    for binfield in binfields:
                        aggd = df.groupby(bins)[binfield].agg(stats_list)

                        # store each stat
                        for stat in stats_list:
                            gridded[binfield][stat][i_x, :] = aggd[stat]

                else:
                    print("grid contains no data at this x1,x2,i_x:")
                    print([x1, x2, i_x])

            if "max" in gridded.keys() and "min" in gridded.keys():
                gridded["span"] = gridded["max"] - gridded["min"]

            gridded[self.xname] = xedges
            gridded[self.yname] = yedges
            gridded[self.xname + "_c"] = (xedges[1:] + xedges[0:-1]) / 2.0
            gridded[self.yname + "_c"] = (yedges[1:] + yedges[0:-1]) / 2.0
        else:
            print("grid required for assignDfToGrid")
            gridded = None

        return gridded


def KmeansSensitivity(max_N, X1, X2, min_N=1):
    """iterative Kmeans clustering using clusters 1 through max_N for 2 variables

    Parameters
    ----------
    max_N : int
        max number of clusters to use
    X1 : ndarray
        first variable for clustering (assumed to be normalized)
    X2 : type
        second variable for clustering (assumed to be normalized)

    Returns
    -------
    dict
        dictionary of results with following keys
            'clusters'  ndarray, the cluster range
            'inertia'   ndarray, inertia value for each clustering
            'bounds'    dict with bounding polygons by cluster, label within cluster


    Example Usage
    -------------
    results=pdd.KmeansSensitivity(18,X1,X2)

    where X1, X2 are normalized observations of the same length

    to pull out bounding polygons of a cluster:

    results['bounds'][2][0]
    """

    results = {"bounds": {}, "X1": X1, "X2": X2}
    Xcluster = np.column_stack((X1, X2))

    Nclusters = range(min_N, max_N + 1)
    inert = []

    for nclust in Nclusters:
        clustering = KMeans(n_clusters=nclust).fit(Xcluster)
        inert.append(clustering.inertia_)
        results["bounds"][nclust] = {}

        # find bounding polygon of each cluster
        for lev in np.unique(clustering.labels_):
            x_1 = X1[clustering.labels_ == lev]
            x_2 = X2[clustering.labels_ == lev]
            b = MultiPoint(np.column_stack((x_1, x_2))).convex_hull
            results["bounds"][nclust][lev] = b

    results["inertia"] = np.array(inert)
    results["clusters"] = np.array(Nclusters)

    return results


def plotKmeansSensitivity(kMeansResults, cmapname="hot", N_best=None):
    """builds plots of KmeansSensitivity results

    Parameters
    ----------
    kMeansResults : dict
        the dict returned from KmeansSensitivity
    cmapname : string
        name of matplotlib colormap to use (the default is 'hot').
    N_best : int
        if not None, will highlight best_N in inertia plot (the default is None)

    Returns
    -------
    fig1,fig2
        figure handles for composite histogram plot and inertia plot

    """
    fig1 = plt.figure()
    maxClusters = max(kMeansResults["clusters"])
    Ntests = len(kMeansResults["clusters"])
    Ncols = int(np.ceil(maxClusters / 2))
    Ncols = 5 if Ncols > 5 else Ncols
    Nrows = np.ceil(Ntests / (Ncols * 1.0))

    X1 = kMeansResults["X1"]
    X2 = kMeansResults["X2"]
    for nclust in kMeansResults["clusters"]:
        ax = plt.subplot(Nrows, Ncols, nclust)
        ax.hist2d(X1, X2, bins=100, density=True, cmap=cmapname)
        for lev in kMeansResults["bounds"][nclust].keys():
            b = kMeansResults["bounds"][nclust][lev]
            ax.plot(b.boundary.xy[0], b.boundary.xy[1], color="w")
        plt.title(str(nclust))

    fig2 = plt.figure()
    plt.plot(kMeansResults["clusters"], kMeansResults["inertia"], "k", marker=".")

    if N_best is not None and N_best in kMeansResults["clusters"]:
        inertval = kMeansResults["inertia"][kMeansResults["clusters"] == N_best]
        plt.plot(N_best, inertval, "r", marker="o")

    plt.xlabel("N")
    plt.ylabel("kmeans inertia")
    return fig1, fig2


def calcKmeans(best_N, X1vals, X2vals):
    def scaleFunc(X_raw):
        return (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())

    def unscaleFunc(Xsc, X_raw):
        return Xsc * (X_raw.max() - X_raw.min()) + X_raw.min()

    X1 = scaleFunc(X1vals)
    X2 = scaleFunc(X2vals)
    Xcluster = np.column_stack((X1, X2))
    clustering = KMeans(n_clusters=best_N).fit(Xcluster)
    return {"X1": X1, "X2": X2, "clustering": clustering}
