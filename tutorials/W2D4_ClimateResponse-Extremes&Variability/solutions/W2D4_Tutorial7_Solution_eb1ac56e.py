
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
ax.set_title("Scale and Location as Functions of Time")