var = "v10"
fig, ax = set_projection_figure(projection=ccrs.PlateCarree())
ax.set_title("Mean " + str(var) + " – (1980-2005)", loc="left")
dataplot = ax.contourf(
    ERA5_mean.longitude,
    ERA5_mean.latitude,
    ERA5_mean[var],
    levels=colorlevels_clim,
    transform=ccrs.PlateCarree(),
    cmap=plt.cm.coolwarm,
)
fig.colorbar(dataplot, orientation="vertical", label="Wind Speed (m/s)", shrink=0.55, pad=0.11)