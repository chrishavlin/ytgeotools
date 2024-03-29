import matplotlib.pyplot as plt

from ytgeotools.mapping import PolygonFile

fname = "data/CP_boundary/ColoradoPlateauBoundary.csv"
CP = PolygonFile(fname, lonname="lon", latname="lat")
CP.bounding_polygon
CP.bounding_polygon.plot()
plt.show()


# apply some transformations: smoothing and scaling
fig, axs = plt.subplots(ncols=3, nrows=3)
for irow, sf in enumerate([1.0, 2, 4.0]):
    for icol, sc in enumerate([0.5, 1.0, 2.0]):
        poly_gpd = CP.build_gpd_df(smooth_factor=sf, affine_scale=sc)
        poly_gpd.plot(ax=axs[irow, icol])
        axs[irow, icol].set_xlim([240, 260])
        axs[irow, icol].set_ylim([30, 45])
        axs[irow, icol].set_title(f"sf={sf}, sc={sc}")

plt.show()
