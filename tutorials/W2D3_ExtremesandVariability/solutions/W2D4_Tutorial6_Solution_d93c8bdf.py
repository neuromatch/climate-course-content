
# collect data of all scenarios in a list, define labels and colors
scenario_data = [data_hist, data_ssp126, data_ssp245, data_ssp585]
scenario_labels = ["Historical, 1850-2014", "SSP-126, 2071-2100", "SSP-245, 2071-2100", "SSP-585, 2071-2100"]
colors = ["k", "C0", "C1", "C2"]

# initialize list of scenario_data length
shape_all = [x*0 for x in range(len(scenario_data))]
loc_all = [x*0 for x in range(len(scenario_data))]
scale_all = [x*0 for x in range(len(scenario_data))]

fig, ax = plt.subplots()
x = np.linspace(20, 120, 1000)

# repeat fitting and plotting for all scenarios
for counter, scenario in enumerate(scenario_data):
    # fit GEV distribution
    shape_all[counter],loc_all[counter], scale_all[counter] = gev.fit(scenario)
    # make plots
    _ = ax.plot(
            x,
            gev.pdf(x, shape_all[counter], loc=loc_all[counter], scale=scale_all[counter]),
            c=colors[counter],
            lw=3,
            label=scenario_labels[counter],
        )

# aesthetics
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation (mm/day)")
ax.set_ylabel("Density");