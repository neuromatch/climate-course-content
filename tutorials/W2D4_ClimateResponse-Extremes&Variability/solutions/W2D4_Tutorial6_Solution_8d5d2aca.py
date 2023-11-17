
# find return levels
fit_hist = fit_return_levels(
    data_hist, np.arange(1.1, 200, 0.1), N_boot=100, alpha=0.05
)
fit_ssp126 = fit_return_levels(
    data_ssp126, np.arange(1.1, 200, 0.1), N_boot=100, alpha=0.05
)
fit_ssp245 = fit_return_levels(
    data_ssp245, np.arange(1.1, 200, 0.1), N_boot=100, alpha=0.05
)
fit_ssp585 = fit_return_levels(
    data_ssp585, np.arange(1.1, 200, 0.1), N_boot=100, alpha=0.05
)


# plot
fig, ax = plt.subplots()
plot_return_levels(fit_hist, c="k", label="historical, 1850-2014", ax=ax)
plot_return_levels(fit_ssp126, c="C0", label="SSP-126, 2071-2100", ax=ax)
plot_return_levels(fit_ssp245, c="C1", label="SSP-245, 2071-2100", ax=ax)
plot_return_levels(fit_ssp585, c="C2", label="SSP-585, 2071-2100", ax=ax)
ax.set_xlim(1, 200)
ax.set_ylim(30, 110)
ax.set_ylabel("Return Level (mm/day)")
ax.set_xlabel("Return Period (years)")