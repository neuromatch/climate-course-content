
# instantiate a GEV distribution
law_ns_shape = sd.GEV()

# fit the GEV to the data, while specifying that the shape parameter ('shape') is meant to be a covariate ('_c') of the time axis (data.index)
_ = law_ns_shape.fit(data.ssh.values, c_shape=np.arange(data.index.size))

# plot results
fig, ax = plt.subplots()
data.ssh.plot(c="k", ax=ax)

# years of interest
x_years = [2, 10, 50, 100, 500]

# repeat plotting of effective return levels for all years of interest
for x in x_years:
    _ = ax.plot(
        data.index, estimate_return_level(1 - 1 / x, law_ns), label=f"{x}-year return level"
    )

ax.legend()
ax.grid(True)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Annual Maximum Sea Surface \nHeight Anomaly (mm)")
ax.set_title("Shape as Function of Time")