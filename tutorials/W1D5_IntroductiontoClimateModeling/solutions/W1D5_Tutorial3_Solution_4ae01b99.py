
# plot the time series
fig, ax = plt.subplots()
_ = ax.plot(t_series_forced, T_series_forced)

ax.set_xlabel("Time (years)")
ax.set_ylabel("Global mean temperature (K)")