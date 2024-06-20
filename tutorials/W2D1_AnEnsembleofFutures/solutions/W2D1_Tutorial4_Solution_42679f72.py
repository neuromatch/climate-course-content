
fig, ax = plt.subplots()
for experiment, color in zip(["historical", "ssp126", "ssp585"], ["C0", "C1", "C2"]):
    datasets = []
    for model in dt_gm_anomaly.keys():
        # calculate annual mean
        annual_sst = (
            dt_gm_anomaly[model][experiment]
            .ds.tos.coarsen(time=12)
            .mean()
            .assign_coords(source_id=model)
            .load()
        )
        datasets.append(
            annual_sst.sel(time=slice(None, "2100")).load()
        )  # the french model has a long running member for ssp 126
    # concatenate all along source_id dimension
    da = xr.concat(datasets, dim="source_id", join="override").squeeze()
    # compute ensemble mean and draw time series
    da.mean("source_id").plot(color=color, label=experiment, ax=ax)
    # extract time coordinates
    x = da.time.data
    # calculate the lower and upper bound of the likely range
    da_lower = da.squeeze().quantile(0.17, dim="source_id")
    da_upper = da.squeeze().quantile(0.83, dim="source_id")
    # shade via quantile boundaries
    ax.fill_between(x, da_lower, da_upper, alpha=0.5, color=color)


# but now add observations (https://pangeo-forge.org/dashboard/feedstock/43)
store = "https://ncsa.osn.xsede.org/Pangeo/pangeo-forge/HadISST-feedstock/hadisst.zarr"
ds_obs = xr.open_dataset(store, engine="zarr", chunks={}).convert_calendar(
    "standard", use_cftime=True
)
# mask missing values
ds_obs = ds_obs.where(ds_obs > -1000)
weights = np.cos(
    np.deg2rad(ds_obs.latitude)
)  # In a regular lon/lat grid, area is ~cos(latitude)
# calculate weighted global mean for observations
sst_obs_gm = ds_obs.sst.weighted(weights).mean(["longitude", "latitude"])
# calculate anomaly for observations
sst_obs_gm_anomaly = sst_obs_gm - sst_obs_gm.sel(time=slice("1950", "1980")).mean()

sst_obs_gm_anomaly.coarsen(time=12, boundary="trim").mean().plot(
    color="0.3", label="Observations", ax=ax
)
ax.set_ylabel("Global Mean SST with respect to 1950-1980 (°C)")
ax.set_xlabel("Time (years)")
ax.legend()