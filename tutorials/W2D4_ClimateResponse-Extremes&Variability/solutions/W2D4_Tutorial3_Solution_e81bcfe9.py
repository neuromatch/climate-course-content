
# set range of scale values to use
range_scale = np.arange(4, 11, 1)

# set shape parameter
shape = 0

# set location parameter
loc = 26

# setup plots
fig, ax = plt.subplots()

# setup colors to use for lines
colors_scale = plt.cm.coolwarm(np.linspace(0, 1, range_scale.size))

# generate pdf for each scale value
for idx, scalei in enumerate(range_scale):
    ax.plot(
        x_r80,
        gev.pdf(x_r80, shape, loc=loc, scale=scalei),
        color=colors_scale[idx],
        label="scale = %.2f" % scalei,
    )

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation \n(mm/day)")
ax.set_ylabel("Density")