
def plot_historical_ssp126_combined(dt, ax):
    for model in dt.keys():
        datasets = []
        for experiment in ["historical", "ssp126"]:
            datasets.append(dt[model][experiment].tos)

        # for each of the models, concatenate its historical and future data
        da_combined = xr.concat(datasets, dim="time")
        # plot annual averages
        da_combined.coarsen(time=12).mean().plot(label=model, ax=ax)


fig, ax = plt.subplots()
# plot_historical_ssp126_combined
_ = plot_historical_ssp126_combined(dt_gm, ax)

ax.set_title("Global Mean SST from five CMIP6 models (annually smoothed)")
ax.set_ylabel("Global Mean SST [$^\circ$C]")
ax.set_xlabel("Year")
ax.legend()