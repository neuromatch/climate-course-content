
# initalize list to store parameters from samples
params = []

# generate 1000 samples by resampling data with replacement
for i in range(1000):
    params.append(
        gev.fit(np.random.choice(precipitation, size=precipitation.size, replace=True))
    )

# print the estimate of the mean of each parameter and it's confidence intervals
print(
    "Mean estimate: ",
    np.mean(np.array(params), axis=0),
    " and 95% confidence intervals: ",
    np.quantile(np.array(params), [0.025, 0.975], axis=0),
)

# generate years vector
years = np.arange(1.1, 1000, 0.1)

# intialize list for return levels
levels = []

# calculate return levels for each of the 1000 samples
for i in range(1000):
    levels.append(gev.ppf(1 - 1 / years, *params[i]))
levels = np.array(levels)

# setup plots
fig, ax = plt.subplots()

# find empirical return levels
_ = empirical_return_level(precipitation).plot(ax=ax, marker=".", linestyle="None")

# plot return mean levels
_ = ax.plot(years, levels.mean(axis=0))

# plot confidence intervals
_ = ax.plot(years, np.quantile(levels, [0.025, 0.975], axis=0).T, "k--")

# aesthetics
ax.set_xlim(1.5, 1000)
ax.set_ylim(20, 110)
ax.set_xscale("log")
ax.set_xlabel("Return Period (years)")
ax.set_ylabel("Return Level (mm/day)")