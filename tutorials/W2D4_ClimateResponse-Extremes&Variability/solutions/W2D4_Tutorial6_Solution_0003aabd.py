
# fit GEV distribution
shape_hist, loc_hist, scale_hist = gev.fit(data_hist)
shape_ssp126, loc_ssp126, scale_ssp126 = gev.fit(data_ssp126)
shape_ssp245, loc_ssp245, scale_ssp245 = gev.fit(data_ssp245)
shape_ssp585, loc_ssp585, scale_ssp585 = gev.fit(data_ssp585)

# make plots
fig, ax = plt.subplots()
x = np.linspace(20, 120, 1000)
ax.plot(
    x,
    gev.pdf(x, shape_hist, loc=loc_hist, scale=scale_hist),
    c="k",
    lw=3,
    label="historical, 1850-2014",
)
ax.plot(
    x,
    gev.pdf(x, shape_ssp126, loc=loc_ssp126, scale=scale_ssp126),
    c="C0",
    lw=3,
    label="SSP-126, 2071-2100",
)
ax.plot(
    x,
    gev.pdf(x, shape_ssp245, loc=loc_ssp245, scale=scale_ssp245),
    c="C1",
    lw=3,
    label="SSP-245, 2071-2100",
)
ax.plot(
    x,
    gev.pdf(x, shape_ssp585, loc=loc_ssp585, scale=scale_ssp585),
    c="C2",
    lw=3,
    label="SSP-585, 2071-2100",
)
ax.legend()
ax.set_xlabel("Annual Maximum Daily Precipitation (mm/day)")
ax.set_ylabel("Density");