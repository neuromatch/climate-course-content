
# calculate climatology for the 1981-2010 period for both GPCP and nClimGrid
sat_clim = (
    sat.sel(time=slice("1981-01-01", "2010-12-01"))
    .groupby("time.month")
    .mean(dim="time")
)
obs_clim = (
    obs.sel(time=slice("1981-01-01", "2010-12-01"))
    .groupby("time.month")
    .mean(dim="time")
)

# calculate anomaly of the NYC time series for both GPCP and nClimGrid
sat_clim_anom = sat.groupby("time.month") - sat_clim
obs_clim_anom = obs.groupby("time.month") - obs_clim

# plot time series and scatter plot between two time series
fig, ax = plt.subplots(figsize=(12, 6))
obs_clim_anom.sel(time=slice("2011-01-01", "2015-12-01")).plot(
    label="nClimGrid anomaly", ax=ax
)
sat_clim_anom.sel(time=slice("2011-01-01", "2015-12-01")).plot(
    marker="o", label="GPCP Monthly anomaly", ax=ax
)
ax.legend()

# plot the scatter plot between nClimGrid and GPCP monthly precipitation CDR
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("GPCP Precipitaion v.s. nClimGrid")
ax.scatter(sat_clim_anom, obs_clim_anom, alpha=0.6)
# Add 1:1 line
y_lim = (0, 15)
x_lim = (0, 15)
ax.plot((0, 15), (0, 15), "r-")
ax.set_ylim(y_lim)
ax.set_xlim(x_lim)
ax.set_xlabel("GPCP Precipitation anomaly (mm/day)")
ax.set_ylabel("nClimGrid anomaly (mm/day)")

# calculate and print correlation coefficient and p-value
r, p = stats.pearsonr(sat_clim_anom, obs_clim_anom)
print("Corr Coef: " + str(r) + ", p-val: " + str(p))