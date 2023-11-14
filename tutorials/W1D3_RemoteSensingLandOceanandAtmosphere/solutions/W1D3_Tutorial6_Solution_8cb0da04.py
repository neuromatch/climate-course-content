
# select data from 1991-2020.
sst_30yr_later = ds.sst.sel(time=slice("1991-01-01", "2020-12-30"))

# calculate climatology
sst_clim_later = sst_30yr_later.groupby("time.month").mean()

# calculate anomaly
sst_anom_later = ds.sst.groupby("time.month") - sst_clim_later

# calculate mean over Nino 3.4 region
nino34_later = sst_anom_later.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(
    dim=["lat", "lon"]
)

# compute 3 month rolling mean
oni_later = nino34_later.rolling(time=3, center=True).mean()

# compare the two ONI time series and visualize the difference as a time series plot
fig, ax = plt.subplots(figsize=(12, 6))
oni.plot(color="k", label="ONI (1982-2011)", ax=ax)
oni_later.plot(color="r", label="ONI (1991-2020)", ax=ax)
ax.set_ylabel("Anomaly (degC)")
ax.axhline(y=0, color="k", linestyle="dashed")
ax.legend()