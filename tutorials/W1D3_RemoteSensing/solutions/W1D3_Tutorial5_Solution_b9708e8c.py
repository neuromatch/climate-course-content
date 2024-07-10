
# calculate 24-month rolling mean
grid_rolling_24m = grid_month.rolling(time=24, center=True).mean()

# plot all three time series together with different colors
fig, ax = plt.subplots()
_ = grid_month.plot(label="Monthly anomaly", ax=ax)
_ = grid_rolling.plot(color="k", label="12-mon rolling mean", ax=ax)
_ = grid_rolling_24m.plot(color="r", label="24-mon rolling mean", ax=ax)

# aesthetics
ax.axhline(y=0, color="y", linestyle="-")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Precipitation anomaly (mm/day)")
ax.legend()
ax.set_title("") # remove the automatically generated title
ax.grid(True)