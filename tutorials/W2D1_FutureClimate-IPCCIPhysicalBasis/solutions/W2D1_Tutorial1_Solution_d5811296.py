fig, ax = plt.subplots()

da = (
    dt_ensemble_gm_anomaly["MPI-ESM1-2-LR"]['historical']
    .ds.tos.coarsen(time=12)
    .mean()
    .load()
)
# calculate the mean across ensemble members
da.mean("member_id").plot(color='C0', label='ensemble mean', ax=ax)

# shading representing spread between members
x = da.time.data
# diagnose the lower range of the likely bounds
da_lower = da.squeeze().quantile(0.17, dim="member_id")
# diagnose the upper range of the likely bounds
da_upper = da.squeeze().quantile(0.83, dim="member_id")
# Uncomment below after filling in above ellipses
ax.fill_between(x, da_lower, da_upper, alpha=0.5, color='C0')

ax.set_title(
    "Global Mean SST Anomaly from a 5-member single-model ensemble"
)
ax.set_ylabel("Global Mean SST Anomaly [°C]")
ax.set_xlabel("Year")
ax.legend()