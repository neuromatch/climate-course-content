
def global_mean(ds: xr.Dataset) -> xr.Dataset:
    """Global average, weighted by the cell area"""
    return ds.weighted(ds.areacello.fillna(0)).mean(["x", "y"], keep_attrs=True)

# average every dataset in the tree globally via map_over_subtree() function introduced in Tutorial
dt_gm = dt_with_area.map_over_subtree(global_mean)

# create plot
fig, ax = plt.subplots()
for experiment in ["historical", "ssp126", "ssp585"]:
    # slice tos data array of experiment
    da = dt_gm["TaiESM1"][experiment].ds.tos
    # draw data
    _ = da.plot(label=experiment, ax=ax)
ax.set_title("Global Mean SST from TaiESM1")
ax.set_ylabel("Global Mean SST [$^\circ$C]")
ax.set_xlabel("Time (years)")
ax.legend()