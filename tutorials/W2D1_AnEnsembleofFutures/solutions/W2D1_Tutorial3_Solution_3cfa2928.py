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
        )
        datasets.append(
            annual_sst.sel(time=slice(None, "2100")).load()
        )  # the french model has a long running member for ssp126
    # concatenate all along source_id dimension
    da = xr.concat(datasets, dim="source_id", join="override").squeeze()
    # compute ensemble mean and draw time series
    da.mean("source_id").plot(color=color, label=experiment, ax=ax)
    # extract time coordinates
    x = da.time.data
    # Calculate the lower bound of the likely range
    da_lower = da.squeeze().quantile(0.17, dim="source_id")
    # Calculate the upper bound of the likely range
    da_upper = da.squeeze().quantile(0.83, dim="source_id")
    # shade via quantile boundaries
    ax.fill_between(x, da_lower, da_upper, alpha=0.5, color=color)

# aesthetics
ax.set_title(
    "Global Mean SST Anomaly from five-member CMIP6 ensemble\n(base period: 1950 to 1980)"
)
ax.set_ylabel("Global Mean SST Anomaly (°C)")
ax.set_xlabel("Time (years)")
ax.legend()