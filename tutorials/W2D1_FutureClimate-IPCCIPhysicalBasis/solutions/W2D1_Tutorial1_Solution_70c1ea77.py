fig, ax = plt.subplots()
dt_ensemble_gm_anomaly_hist.plot(hue='member_id')

ax.set_title(
    "Global Mean SST Anomaly from a 5-member single-model ensemble"
)

plt.axhline(0, linestyle='dashed', color='lightgrey')
ax.set_ylabel("Global Mean SST Anomaly [°C]")
ax.set_xlabel("Year")