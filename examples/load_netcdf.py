import yt

from ytgeotools.seismology.datasets import XarrayGeoSpherical

filename = "IRIS/GYPSUM_percent.nc"

ds = XarrayGeoSpherical(filename)
x, y, z = ds.cartesian_coords()

ds_yt = XarrayGeoSpherical(filename).load_uniform_grid()


slc = yt.SlicePlot(ds_yt, "depth", "dvs")
slc.save()
