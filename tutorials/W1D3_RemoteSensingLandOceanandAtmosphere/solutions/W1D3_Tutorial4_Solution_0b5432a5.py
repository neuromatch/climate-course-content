
# select data for the month of interest
data = ds.precip_error.sel(time="1979-01-01", method="nearest")

# initate plot with the specific figure size
fig, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()}, figsize=(9, 6))

# add coastal lines to indicate land/ocean
_ = ax.coastlines()

# add grid lines for latitude and longitute
_ = ax.gridlines()

# add the precipitation data for
data.plot(
    ax=ax,
    transform=ccrs.PlateCarree(),
    cbar_kwargs=dict(shrink=0.4, label="GPCP Monthly Precipitation Error\n(mm/day)"),
)