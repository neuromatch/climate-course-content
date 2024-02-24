
# collect data of all scenarios in a list, define labels and colors
scenario_data = [data_hist, data_ssp126, data_ssp245, data_ssp585]
scenario_labels = ["Historical, 1850-2014", "SSP-126, 2071-2100", "SSP-245, 2071-2100", "SSP-585, 2071-2100"]
colors = ["k", "C0", "C1", "C2"]

fit_all_scenarios = [0, 0, 0, 0]

# setup plot
fig, ax = plt.subplots()

# repeat fitting and plotting of the return levels for all scenarios
# using fit_return_levels() and plot_return_levels()
for counter, scenario in enumerate(scenario_data):
    fit_all_scenarios[counter] = fit_return_levels(
        scenario, np.arange(1.1, 200, 0.1), N_boot=100, alpha=0.05
    )
    _ = plot_return_levels(
        fit_all_scenarios[counter], c=colors[counter], label=scenario_labels[counter], ax=ax
    )

# aesthetics
ax.set_xlim(1, 200)
ax.set_ylim(30, 110)
ax.set_ylabel("Return Level (mm/day)")
ax.set_xlabel("Return Period (years)")