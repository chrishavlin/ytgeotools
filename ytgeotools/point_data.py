"""
class for processing point data
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, MultiPoint, Polygon, LineString
from shapely.ops import cascaded_union
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from . import mapping as MS


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

    def assignDfToGrid(self, binfields=[]):
        """finds stats within bins. binfield is the field to bin"""

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
        try:
            # find bounding polygon of each cluster
            for lev in np.unique(clustering.labels_):
                x_1 = X1[clustering.labels_ == lev]
                x_2 = X2[clustering.labels_ == lev]
                b = MultiPoint(np.column_stack((x_1, x_2))).convex_hull
                results["bounds"][nclust][lev] = b
        except:
            pass

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


class boundingPolies(object):
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

    def __init__(self, df, b_df=None, **kwargs):
        self.df_raw = df
        self.b_df = b_df

        rad_deg = kwargs.get("radius_deg", 0.5)
        self._buildExtents(radius_deg=rad_deg)
        return

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
        # ceate circle of radius radius_deg for every volc center, find union
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

        self.df_gp = gpd.GeoDataFrame(geometry=polies, crs=MS.crs)
        self.df_bound = gpd.GeoSeries(cascaded_union(polies))

        return


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


def filterByBounds(df, b_df, return_interior=True):
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
    if type(df) == type(pd.DataFrame()):
        geo = [Point([p[0], p[1]]) for p in zip(df["lon"], df["lat"])]
        df_gpd = gpd.GeoDataFrame(df, crs=MS.crs, geometry=geo)
    elif type(df) == type(gpd.GeoDataFrame):
        df_gpd = df

    # spatial join of the two geodataframes
    df_s = gpd.sjoin(b_df, df_gpd, how="right", op="intersects")
    if return_interior:
        return df_s[~pd.isnull(df_s["shape_id"])]
    else:
        return df_s
