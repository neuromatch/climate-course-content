
# initialize a GEV distribution
law_ns_scale = sd.GEV()

# fit the GEV to the data, while specifying that the scale parameter ('scale') is meant to be a covariate ('_c') of the time axis (data.index)
_ = law_ns_scale.fit(data.ssh.values, c_scale=np.arange(data.index.size))

# plot results
fig, ax = plt.subplots()
data.ssh.plot(c="k", ax=ax)
ax.plot(
    data.index, estimate_return_level(1 - 1 / 2, law_ns), label="2-year return level"
)
ax.plot(
    data.index, estimate_return_level(1 - 1 / 10, law_ns), label="10-year return level"
)
ax.plot(
    data.index, estimate_return_level(1 - 1 / 50, law_ns), label="50-year return level"
)
ax.plot(
    data.index,
    estimate_return_level(1 - 1 / 100, law_ns),
    label="100-year return level",
)
ax.plot(
    data.index,
    estimate_return_level(1 - 1 / 500, law_ns),
    label="500-year return level",
)
ax.legend()
ax.grid(True)
ax.set_title("Scale as Function of Time")