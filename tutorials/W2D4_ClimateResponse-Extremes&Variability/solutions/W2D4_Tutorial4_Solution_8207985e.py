
# setup plots
fig, ax = plt.subplots()

# get empirical return levels
_ = empirical_return_level(precipitation).plot(ax=ax, marker=".", linestyle="None")

# create vector of years
years = np.arange(1.1, 100, 0.1)

# calculate and plot the normal return levels
ax.plot(
    years,
    stats.norm.ppf(1 - 1 / years, loc=precipitation.mean(), scale=precipitation.std()),
)

# calculate and plot the GEV distribution, note the negtive shape parameter
ax.plot(years, gev.ppf(1 - 1 / years, shape, loc=loc, scale=scale))
# set x axis to log scale
ax.set_xscale("log")

# show legend
ax.legend(["empirical", "normal", "GEV"])