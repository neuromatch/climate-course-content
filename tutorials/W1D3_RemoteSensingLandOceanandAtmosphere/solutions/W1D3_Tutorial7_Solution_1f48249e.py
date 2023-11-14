
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
fig.suptitle("GPCP Precipitaion Anomaly v.s. Oceanic Nino Index")
axs[0].plot(precip_nino34_italy.time, precip_nino34_italy.precip)
axs[0].set_ylabel("Precip (mm/day)")
axs[0].axhline(y=0, color="k", linestyle="dashed")
axs[1].plot(nino34.time, nino34.sst)
axs[1].set_ylabel("ONI (degC)")
axs[1].set_xlabel("Time")
axs[1].axhline(y=0, color="k", linestyle="dashed")

# El Nino Data, logically index to keep ONI values above 0.5
italy_el_nino_sst = nino34.sst[nino34.sst > 0.5]
italy_el_nino_precip = precip_nino34_italy.precip[nino34.sst > 0.5]

# La Nina Data, logically index to keep ONI values below -0.5
italy_la_nina_sst = nino34.sst[nino34.sst < -0.5]
italy_la_nina_precip = precip_nino34_italy.precip[nino34.sst < -0.5]

# correlation for El Nino data
italy_el_nino_r, italy_el_nino_p = stats.pearsonr(
    italy_el_nino_sst, italy_el_nino_precip
)

print(
    "El Nino - Corr Coef: " + str(italy_el_nino_r) + ", p-val: " + str(italy_el_nino_p)
)

# correlation for La Nina data
italy_la_nina_r, italy_la_nina_p = stats.pearsonr(
    italy_la_nina_sst, italy_la_nina_precip
)

print(
    "La Nina - Corr Coef: " + str(italy_la_nina_r) + ", p-val: " + str(italy_la_nina_p)
)

# plot scatter plot between precipitation and ONI
fig, ax = plt.subplots(figsize=(7, 7))
fig.suptitle("GPCP Precipitaion Anomaly v.s. Oceanic Nino Index")
ax.scatter(italy_el_nino_sst, italy_el_nino_precip, c="r", alpha=0.6)
ax.scatter(italy_la_nina_sst, italy_la_nina_precip, c="b", alpha=0.6)

# add horizontal and vertical lines of 0 values
ax.axhline(y=0, linestyle="dashed", color="k", alpha=0.6)
ax.axvline(x=0, linestyle="dashed", color="k", alpha=0.6)
ax.axvline(x=0.5, linestyle="dashed", color="r", alpha=0.6)  # El Nino threshold
ax.axvline(x=-0.5, linestyle="dashed", color="b", alpha=0.6)  # La Nina threshold
ax.set_xlabel("ONI (degC)")
ax.set_ylabel("Precipitation Anomaly (mm/day)")