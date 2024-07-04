fig, ax = plt.subplots()

# calculate the mean across ensemble members and plot it
da_mean = da.mean("member_id")
_ = da_mean.plot(color='C0', label='Ensemble mean', ax=ax)

# shading representing spread between members
x = da.time.data
# diagnose the lower range of the likely bounds
da_lower = da.squeeze().quantile(0.17, dim="member_id")
# diagnose the upper range of the likely bounds
da_upper = da.squeeze().quantile(0.83, dim="member_id")
# use ax.fill_between() and above bounds to shade likely range
_ = ax.fill_between(x, da_lower, da_upper, alpha=0.5, color='C0')

# aesthetics
plt.axhline(0, linestyle='dashed', color='lightgrey')
ax.set_title(
    "Global Mean SST Anomaly from a 5-member single-model ensemble"
)
ax.set_ylabel("Global Mean SST Anomaly (°C)")
ax.set_xlabel("Time (years)")
ax.legend()