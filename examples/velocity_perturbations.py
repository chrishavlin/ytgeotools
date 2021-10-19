from ytgeotools.seismology import datasets as ysd

filename = "IRIS/wUS-SH-2010_percent.nc"
refmodel = "IRIS/refModels/AK135F_AVG.csv"

refAK = ysd.load_1d_csv_ref_collection(refmodel, "depth_km")
ds = ysd.XarrayGeoSpherical(filename)
abs_vs = ds.get_absolute(refAK, "dvs", "Vs_kms")
