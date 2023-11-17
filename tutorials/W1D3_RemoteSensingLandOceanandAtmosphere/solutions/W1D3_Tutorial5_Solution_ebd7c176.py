
# calculate 24-month rolling mean
grid_rolling_24m = grid_month.rolling(time=24, center=True).mean()

# plot all three time series together with different colors
fig, ax = plt.subplots(figsize=(12, 6))
grid_month.plot(label="Monthly Anomaly", ax=ax)
grid_rolling.plot(color="k", label="12-mon rolling mean", ax=ax)
grid_rolling_24m.plot(color="r", label="24-mon rolling mean", ax=ax)
ax.axhline(y=0, color="y", linestyle="-")
ax.set_ylabel("Precipitation Anomaly (mm/day)")
ax.legend()
ax.set_xlabel("Time")
# remove the automatically generated title
ax.set_title("")