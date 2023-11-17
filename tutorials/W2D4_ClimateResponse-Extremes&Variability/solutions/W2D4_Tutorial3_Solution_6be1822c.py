
# set range of location values to use
range_loc = np.arange(20, 41, 4)

# set shape parameter
shape = 0

# set scale parameter
scale = 7

# set x values
x_r80 = np.linspace(0, 80, 1000)

# setup plots
fig, ax = plt.subplots()

# setup colors to use for lines
colors_loc = plt.cm.coolwarm(np.linspace(0, 1, range_loc.size))

# generate pdf for each location value
for idx, loci in enumerate(range_loc):
    ax.plot(
        x_r80,
        gev.pdf(x_r80, shape, loc=loci, scale=scale),
        color=colors_loc[idx],
        label="loc = %i" % loci,
    )

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation \n(mm/day)")
ax.set_ylabel("Density")