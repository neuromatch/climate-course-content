var='u10'
season='DJF'
F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ...
_ = ... # colorbar
plt.show()

season='JJA'
F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title('Zonal Wind DJF mean | '+ var , loc ='left')
dataplot = ...
_ = ... # colorbar
plt.show()

# difference:
pdata = ...

F, ax = set_projection_figure(projection = ccrs.PlateCarree())
ax.set_title(... , loc ='left')
dataplot = ...
_ = ... # colorbar
plt.show()