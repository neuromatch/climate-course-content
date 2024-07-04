
# define region of interest for the precipitation anomaly
italy_lon = [6, 19]
italy_lat = [36, 48]

# calculate regional mean time series
precip_nino34_italy = precip.sel(
    latitude=slice(italy_lat[0], italy_lat[1]),
    longitude=slice(italy_lon[0], italy_lon[1]),
    time=slice("1981-09-01", "2022-12-01"),
).mean(dim=("latitude", "longitude"))

# plot the time series of precipitation anomaly and ONI for the same time period on different subplots
fig, axs = plt.subplots(2, sharex=True)
# set title
fig.suptitle("GPCP Precipitation Anomaly v.s. Oceanic Niño Index")

# draw precipitation data
_ = axs[0].plot(precip_nino34_italy.time, precip_nino34_italy.precip)
# upper plot aesthetics
axs[0].set_ylabel("Precipitation\n(mm/day)")
axs[0].axhline(y=0, color="k", linestyle="dashed")

# draw ONI data
_ = axs[1].plot(nino34.time, nino34.sst)
# lower plot aesthetics
axs[1].set_ylabel("ONI (°C)")
axs[1].set_xlabel("Time (months)")
axs[1].axhline(y=0, color="k", linestyle="dashed")

# El Nino Data, select ONI values above 0.5 via boolean indexing
# (https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#boolean-indexing)
italy_el_nino_sst = nino34.sst[nino34.sst > 0.5]
italy_el_nino_precip = precip_nino34_italy.precip[nino34.sst > 0.5]

# La Nina Data, select ONI values below 0.5 via boolean indexing
italy_la_nina_sst = nino34.sst[nino34.sst < -0.5]
italy_la_nina_precip = precip_nino34_italy.precip[nino34.sst < -0.5]

# correlation for El Nino data
italy_el_nino_r = stats.pearsonr(italy_el_nino_sst, italy_el_nino_precip)[0]
italy_el_nino_p = stats.pearsonr(italy_el_nino_sst, italy_el_nino_precip)[1]
print("El Niño - Corr Coef: " + str(italy_el_nino_r) + ", p-val: " + str(italy_el_nino_p))

# correlation for La Nina data
italy_la_nina_r = stats.pearsonr(italy_la_nina_sst, italy_la_nina_precip)[0]
italy_la_nina_p = stats.pearsonr(italy_la_nina_sst, italy_la_nina_precip)[1]
print("La Niña - Corr Coef: " + str(italy_la_nina_r) + ", p-val: " + str(italy_la_nina_p))

# scatter plot: precipitation vs. ONI
fig, ax = plt.subplots()
fig.suptitle("GPCP Precipitation Anomaly v.s. Oceanic Nino Index")
_ = ax.scatter(italy_el_nino_sst, italy_el_nino_precip, c="r", alpha=0.6)
_ = ax.scatter(italy_la_nina_sst, italy_la_nina_precip, c="b", alpha=0.6)

# add horizontal and vertical lines of 0 values
ax.axhline(y=0, linestyle="dashed", color="k", alpha=0.6)
ax.axvline(x=0, linestyle="dashed", color="k", alpha=0.6)
ax.axvline(x=0.5, linestyle="dashed", color="r", alpha=0.6)  # El Nino threshold
ax.axvline(x=-0.5, linestyle="dashed", color="b", alpha=0.6)  # La Nina threshold
# labels
ax.set_xlabel("ONI (°C)")
ax.set_ylabel("Precipitation Anomaly\n(mm/day)")