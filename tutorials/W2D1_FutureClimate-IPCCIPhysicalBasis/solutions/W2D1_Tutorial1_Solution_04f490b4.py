# select just a single model and experiment
sst_ssp585 = dt["TaiESM1"]["ssp585"].ds.tos

fig, ([ax_present, ax_future], [ax_diff_july, ax_diff_jan]) = plt.subplots(
    ncols=2, nrows=2, figsize=[12, 6], subplot_kw={"projection": ccrs.Robinson()}
)

# plot a timestep for 2023
sst_present = sst_ssp585.sel(time="2023-07").squeeze()
sst_present.plot(
    ax=ax_present,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    vmin=-10,
    vmax=30,
    cmap="magma",
    robust=True,
)
ax_present.coastlines()
ax_present.set_title("July 2023")

# repeat for 2100
# complete the following line to extract data for July 2100
sst_future = sst_ssp585.sel(time="2100-07").squeeze()
_ = sst_future.plot(
    ax=ax_future,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    vmin=-10,
    vmax=30,
    cmap="magma",
    robust=True,
)
ax_future.coastlines()
ax_future.set_title("July 2100")

# now find the difference between July 2100 and July 2023
# complete the following line to extract the July difference
sst_difference_july = sst_future - sst_present
_ = sst_difference_july.plot(
    ax=ax_diff_july,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    vmin=-7.5,
    vmax=7.5,
    cmap="coolwarm",
)
ax_diff_july.coastlines()
ax_diff_july.set_title("2100 vs. 2023 Difference (July)")

# finally, find the difference between January of the two years used above
# complete the following line to extract the January difference
sst_difference_jan = (
    sst_ssp585.sel(time="2100-01").squeeze() - sst_ssp585.sel(time="2023-01").squeeze()
)
_ = sst_difference_jan.plot(
    ax=ax_diff_jan,
    x="lon",
    y="lat",
    transform=ccrs.PlateCarree(),
    vmin=-7.5,
    vmax=7.5,
    cmap="coolwarm",
)
ax_diff_jan.coastlines()
ax_diff_jan.set_title("2100 vs. 2023 Difference (January)")