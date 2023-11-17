fig, ax = plt.subplots()
for experiment, color in zip(["historical", "ssp126", "ssp585"], ["C0", "C1", "C2"]):
    da = (
        dt_ensemble_gm_anomaly["MPI-ESM1-2-LR"][experiment]
        .ds.tos.coarsen(time=12)
        .mean()
        .load()
    )
    # calculate the mean across ensemble members
    da.mean("member_id").plot(color=color, label=experiment, ax=ax)

    # shading representing spread between members
    x = da.time.data
    # diagnose the lower range of the likely bounds
    da_lower = da.squeeze().quantile(0.17, dim="member_id")
    # diagnose the upper range of the likely bounds
    da_upper = da.squeeze().quantile(0.83, dim="member_id")
    # Uncomment below after filling in above ellipses
    # ax.fill_between(x, da_lower, da_upper, alpha=0.5, color=color)
    ax.fill_between(x, da_lower, da_upper, alpha=0.5, color=color)

ax.set_title(
    "Global Mean SST Anomaly in SSP1-2.6 from a 5-member single-model ensemble"
)
ax.set_ylabel("Global Mean SST Anomaly [$^\circ$C]")
ax.set_xlabel("Year")
ax.legend()