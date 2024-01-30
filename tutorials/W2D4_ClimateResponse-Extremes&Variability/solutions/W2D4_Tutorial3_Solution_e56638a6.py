
# set range of shape values to use
range_shape = np.arange(-0.4, 0.4 + 0.1, 0.1)

# set scale parameter
scale = 7

# set location parameter
loc = 26

# create precipitation array
x_r80 = np.arange(80,step=0.5)

# setup plots
fig, ax = plt.subplots()

# setup colors to use for lines
colors_shape = plt.cm.coolwarm(np.linspace(0, 1, range_shape.size))

# generate pdf for each shape value
for idx, shapei in enumerate(range_shape):
    ax.plot(
        x_r80,
        gev.pdf(x_r80, shapei, loc=loc, scale=scale),
        color=colors_shape[idx],
        label="shape = %.2f" % shapei,
    )

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation \n(mm/day)")
ax.set_ylabel("Density")