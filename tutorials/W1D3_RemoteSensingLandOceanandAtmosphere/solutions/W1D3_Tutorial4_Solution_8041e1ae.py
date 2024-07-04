
# select data for the month of interest
data = ds.precip_error.sel(time="1979-01-01", method="nearest")

# initiate plot
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})

# add coastal lines to indicate land/ocean
_ = ax.coastlines()

# add grid lines for latitude and longitude
_ = ax.gridlines()

# add the precipitation data
_ = data.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(shrink=0.4, label="GPCP Monthly\nPrecipitation Error\n(mm/day)"),
    robust=True
)