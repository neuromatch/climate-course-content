var='u10'
season='DJF'
F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season_u10.longitude, ERA5_season_u10.latitude, ERA5_season_u10.sel(season=season),
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'Wind speed (m/s)', shrink= 0.55 , pad = 0.11) # colorbar
plt.show()

season='JJA'
F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season_u10.longitude, ERA5_season_u10.latitude, ERA5_season_u10.sel(season=season),
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'Wind speed (m/s)', shrink= 0.55 , pad = 0.11) # colorbar
plt.show()

# difference:
pdata = ERA5_season_u10.sel(season='DJF') - ERA5_season_u10.sel(season='JJA')

F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title('Zonal Wind DJF mean - JJA mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season_u10.longitude, ERA5_season_u10.latitude,pdata ,
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'Wind speed (m/s)', shrink= 0.55 , pad = 0.11) # colorbar
plt.show()