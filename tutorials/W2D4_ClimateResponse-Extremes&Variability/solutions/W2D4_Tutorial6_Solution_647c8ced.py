
# setup plot
fig, ax = plt.subplots()

# create histograms for each scenario and historical
# Uncomment and complete
sns.histplot(
    data_hist,
    bins=np.arange(30, 100, 5),
    color="k",
    element="step",
    stat="density",
    alpha=0.3,
    lw=0.5,
    line_kws=dict(lw=3),
    kde=True,
    label="Historical, 1850-2014",
    ax=ax,
)
sns.histplot(
    data_ssp126,
    bins=np.arange(30, 100, 5),
    color="C0",
    element="step",
    stat="density",
    alpha=0.3,
    lw=0.5,
    line_kws=dict(lw=3),
    kde=True,
    label="SSP-126, 2071-2100",
    ax=ax,
)
sns.histplot(
    data_ssp126,
    bins=np.arange(30, 100, 5),
    color="C0",
    element="step",
    stat="density",
    alpha=0.3,
    lw=0.5,
    line_kws=dict(lw=3),
    kde=True,
    label="SSP-126, 2071-2100",
    ax=ax,
)
sns.histplot(
    data_ssp245,
    bins=np.arange(30, 100, 5),
    color="C1",
    element="step",
    stat="density",
    alpha=0.3,
    lw=0.5,
    line_kws=dict(lw=3),
    kde=True,
    label="SSP-245, 2071-2100",
    ax=ax,
)
sns.histplot(
    data_ssp585,
    bins=np.arange(30, 100, 5),
    color="C2",
    element="step",
    stat="density",
    alpha=0.3,
    lw=0.5,
    line_kws=dict(lw=3),
    kde=True,
    label="SSP-585, 2071-2100",
    ax=ax,
)

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation (mm/day)")

# calculate moments
periods_stats = pd.DataFrame(index=["Mean", "Standard Deviation", "Skew"])
periods_stats["hist"] = [data_hist.mean(), data_hist.std(), data_hist.skew()]
periods_stats["ssp126"] = [data_ssp126.mean(), data_ssp126.std(), data_ssp126.skew()]
periods_stats["ssp245"] = [data_ssp245.mean(), data_ssp245.std(), data_ssp245.skew()]
periods_stats["ssp585"] = [data_ssp585.mean(), data_ssp585.std(), data_ssp585.skew()]
periods_stats = periods_stats.T
periods_stats