var = "v10"
fig, ax = set_projection_figure(projection=ccrs.PlateCarree(), figsize=(9, 5.5))
ax.set_title("Mean " + str(var), loc="left")
dataplot = ax.contourf(
    ERA5_ANN.longitude,
    ERA5_ANN.latitude,
    ERA5_ANN[var],
    levels=colorlevels_clim,
    transform=ccrs.PlateCarree(),
    cmap=plt.cm.coolwarm,
)
fig.colorbar(dataplot, orientation="vertical", label="m/s", shrink=0.55, pad=0.08)