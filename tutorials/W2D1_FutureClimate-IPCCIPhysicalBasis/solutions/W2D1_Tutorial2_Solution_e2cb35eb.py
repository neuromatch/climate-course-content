
# calculate anomaly to reference period
def datatree_anomaly(dt):
    dt_out = DataTree()
    for model, subtree in dt.items():
        # find the temporal average over the desired reference period
        ref = dt[model]["historical"].ds.sel(time=slice("1950", "1980")).mean()
        dt_out[model] = subtree - ref
    return dt_out

# Uncomment below once you finish the above
# dt_gm_anomaly = datatree_anomaly(dt_gm)
dt_gm_anomaly = datatree_anomaly(dt_gm)

fig, ax = plt.subplots()
# Uncomment below once you finish the above
# plot_historical_ssp126_combined(dt_gm_anomaly, ax)
plot_historical_ssp126_combined(dt_gm_anomaly, ax)

ax.set_title(
    "Global Mean SST Anomaly from five CMIP6 models (base period: 1950 to 1980)"
)
ax.set_ylabel("Global Mean SST Anomaly [$^\circ$C]")
ax.set_xlabel("Year")
ax.legend()