# note, this codes takes a while to run
ERA5_season = ERA5_mm.groupby('time.season').mean()

var='u10'
season='DJF'
F, ax = set_projection_figure(projection = ccrs.PlateCarree(), figsize = (9, 5.5) )
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season.longitude, ERA5_season.latitude, ERA5_season.sel(season=season)[var],
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'm/s', shrink= 0.55 , pad = 0.08) # colorbar
plt.show()


var='u10'
season='JJA'
F, ax = set_projection_figure(projection = ccrs.PlateCarree(), figsize = (9, 5.5) )
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season.longitude, ERA5_season.latitude, ERA5_season.sel(season=season)[var],
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'm/s', shrink= 0.55 , pad = 0.08) # colorbar
plt.show()

# difference:
var='u10'
pdata = ERA5_season.sel(season='DJF')[var] - ERA5_season.sel(season='JJA')[var]

F, ax = set_projection_figure(projection = ccrs.PlateCarree(), figsize = (9, 5.5) )
ax.set_title('Zonal Wind DJF mean - JJA mean | '+ var , loc ='left')
dataplot = ax.contourf(ERA5_season.longitude, ERA5_season.latitude,pdata ,
                       levels = colorlevels_clim,
                       transform=ccrs.PlateCarree(), cmap= plt.cm.coolwarm)
_ = plt.colorbar(dataplot, orientation='vertical', label = 'm/s', shrink= 0.55 , pad = 0.08) # colorbar
plt.show()