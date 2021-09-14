from ytgeotools.datasets import XarrayGeoSpherical

filename = "/home/chris/hdd/data/yt_data/IRIS_models/NA07_percent.nc"

ds = XarrayGeoSpherical(filename)
x,y,z = ds.cartesian_coords()



import yt

ds_yt = XarrayGeoSpherical(filename).load_uniform_grid()


slc = yt.SlicePlot(ds_yt, "depth", "dvs")
slc.save()


