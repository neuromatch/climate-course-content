
# setup plot
fig, ax = plt.subplots()

# collect data of all scenarios, labels, colors in lists, and define bin_range
scenario_data = [data_hist, data_ssp126, data_ssp245, data_ssp585]
scenario_labels = ["Historical, 1850-2014","SSP-126, 2071-2100","SSP-245, 2071-2100","SSP-585, 2071-2100"]
colors = ["k","C0","C1","C2"]
bin_range = np.arange(30, 100, 5)

# create histograms/ PDFs for each scenario and historical
for counter, data_src in enumerate(scenario_data):
    sns.histplot(
        data_src,
        bins=bin_range,
        color=colors[counter],
        element="step",
        stat="density",
        alpha=0.3,
        lw=0.5,
        line_kws=dict(lw=3),
        kde=True,
        label=scenario_labels[counter],
        ax=ax,
    )

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation (mm/day)")

# calculate moments
periods_stats = pd.DataFrame(index=["Mean", "Standard Deviation", "Skew"])
column_names = ["hist","ssp126","ssp245","ssp585"]

for counter, data_src in enumerate(scenario_data):
    periods_stats[column_names[counter]] = [data_src.mean(), data_src.std(), data_src.skew()]

periods_stats = periods_stats.T
periods_stats