
# initialize a GEV distribution
law_ns_loc_scale = sd.GEV()

# fit the GEV to the data using c_loc and c_scale
_ = law_ns_loc_scale.fit(
    data.ssh.values,
    c_loc=np.arange(data.index.size),
    c_scale=np.arange(data.index.size),
)

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