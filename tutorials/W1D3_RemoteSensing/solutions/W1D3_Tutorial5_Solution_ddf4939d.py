
# calculate unweighted global mean
global_unweighted_mean = precip_anom.mean(("latitude", "longitude"))

# calculate the difference between weighted and unweighted global mean
global_diff = global_weighted_mean - global_unweighted_mean

# plot the time series of the difference
fig, ax = plt.subplots()
_ = global_weighted_mean.plot(label="Monthly anomaly", ax=ax)
_ = global_diff.rolling(time=12, center=True).mean(("latitude", "longitude")).plot(
    color="k", label="12-mon rolling diff mean", ax=ax
)

# aesthetics
ax.axhline(y=0, color="y", linestyle="-")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Precipitation anomaly (mm/day)")
ax.legend()
ax.grid(True)