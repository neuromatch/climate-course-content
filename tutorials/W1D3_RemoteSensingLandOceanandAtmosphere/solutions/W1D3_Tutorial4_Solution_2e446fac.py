
# extract 30 year data for 1991-2020
precip_30yr_exercise = ds.precip.sel(time=slice("1991-01-01", "2020-12-30"))

# calculate climatology for 1991-2020
precip_clim_exercise = precip_30yr_exercise.groupby("time.month").mean(dim="time")

# find difference in climatologies: (1981-2010) minus (1991-2020)
precip_diff_exercise = precip_clim_exercise - precip_clim

# Compare the climatology for four different seasons by generating the
#         difference maps for January, April, July, and October with colorbar max and min = 1,-1

# Define the figure and each axis for the 2 rows and 2 columns
fig, axs = plt.subplots(
    nrows=2, ncols=2, subplot_kw={"projection": ccrs.Robinson()}, figsize=(12, 8)
)

# axs is a 2-dimensional array of `GeoAxes`. We will flatten it into a 1-D array
axs = axs.flatten()

# Loop over selected months (Jan, Apr, Jul, Oct)
for i, month in enumerate([1, 4, 7, 10]):

    # Draw the coastlines and major gridline for each subplot
    axs[i].coastlines()
    axs[i].gridlines()

    # Draw the precipitation data
    precip_diff_exercise.sel(month=month).plot(
        ax=axs[i],
        transform=ccrs.PlateCarree(),
        vmin=-1,
        vmax=1,
        cbar_kwargs=dict(shrink=0.5, label="GPCP Climatology\nDifference (mm/day)"),
    )