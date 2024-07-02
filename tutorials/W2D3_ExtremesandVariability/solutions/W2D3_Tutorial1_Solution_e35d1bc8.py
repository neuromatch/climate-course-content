
fig, ax = plt.subplots()

# make histogram
_ = sns.histplot(precipitation, bins=bins, stat="density", ax=ax)

# set x limits
_ = ax.set_xlim(bins[0], bins[-1])

# get y lims for plotting mean line
ylim = ax.get_ylim()

# add vertical line with mean
_ = ax.vlines(mean_pr, ymin=ylim[0], ymax=ylim[1], color="C3", lw=3)

# plot pdf
_ = ax.plot(x_r100, stats.norm.pdf(x_r100, mean_pr, std_pr), c="k", lw=3)

# plot 95th percentile
_ = ax.plot(x_r100, np.quantile(pdfs, 0.95, axis=1), "--", lw=2, color="k")

# plot 5th percentile
_ = ax.plot(x_r100, np.quantile(pdfs, 0.05, axis=1), "--", lw=2, color="k")

# set xlabel
ax.set_xlabel("Annual Maximum Daily Precipitation \n(mm/day)")