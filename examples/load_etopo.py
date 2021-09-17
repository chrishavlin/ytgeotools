import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from ytgeotools.topography import etopo

topo = etopo.Etopo("data/elevation_data/etopo1.asc")

ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(topo.longitude, topo.latitude, topo.topo, 60, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()
