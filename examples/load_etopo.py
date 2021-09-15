from ytgeotools.topography import etopo
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

topo = etopo.Etopo("/home/chavlin/hdd/data/LAB_obs/data/elevation_data/etopo1.asc")

ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(topo.longitude, topo.latitude, topo.topo, 60, transform=ccrs.PlateCarree())
ax.coastlines()
plt.show()
