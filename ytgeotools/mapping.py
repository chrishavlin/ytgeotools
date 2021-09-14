from . import dataManager as dm
from shapely.geometry import Polygon
from shapely import affinity as aff
import pandas as pd
import geopandas as gpd
import numpy as np
import os

crs = {"init": "epsg:4326"}
preferredproj = "robinson"


def setDb(db=None):
    if type(db) == type("") or db is None:
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
        return (Dom_gp, polies)


class etopo(object):
    """
    class for loading dem topo files from https://www.ngdc.noaa.gov/mgg/global/global.html
    """

    def __init__(self, filename, db=None, loadFile=True):
        self.db = setDb(db)
        if os.path.isfile(filename):
            self.filename = filename
        else:
            self.filename = self.db.fullpath(filename)

        self.filetype = filename.split(".")[-1]

        if loadFile:
            if self.filetype == "asc":
                self.loadGriddedAscii()
            else:
                print("filetype " + self.filetype + " not supported")

        return

    def loadGriddedAscii(self):
        """loads the gridded etopo file"""

        # load the gridded data
        self.topo = np.loadtxt(self.filename, skiprows=5)

        # load the header
        headervals = []
        header = ["ncols", "nrows", "lon1", "lat1", "d_deg"]
        with open(self.filename, "r") as f:
            for i in range(5):
                line = next(f)  # .next()
                headervals.append(float(line.split(" ")[-1]))
        header_dict = dict(zip(header, headervals))
        header_dict["lat2"] = (
            header_dict["lat1"] + header_dict["nrows"] * header_dict["d_deg"]
        )
        header_dict["lon2"] = (
            header_dict["lon1"] + header_dict["ncols"] * header_dict["d_deg"]
        )

        # calculate lat, lon arrays
        self.lats = np.linspace(
            header_dict["lat2"], header_dict["lat1"], header_dict["nrows"]
        )
        self.lons = np.linspace(
            header_dict["lon1"], header_dict["lon2"], header_dict["ncols"]
        )
        self.topo_range = header_dict
        return
