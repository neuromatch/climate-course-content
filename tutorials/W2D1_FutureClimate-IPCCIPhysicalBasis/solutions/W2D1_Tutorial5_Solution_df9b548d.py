col = intake.open_esm_datastore(
    "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
)  # open an intake catalog containing the Pangeo CMIP cloud data

cat_ensemble = col.search(
    source_id="MPI-ESM1-2-LR",
    variable_id="tos",
    table_id="Omon",
    # select the 5 ensemble members described above
    member_id=["r1i1p1f1", "r2i1p1f1", "r3i1p1f1", "r4i1p1f1", "r5i1p1f1"],
    grid_label="gn",
    experiment_id=["historical", "ssp126", "ssp585"],
    require_all_on=["source_id", "member_id"],
)

# convert the sub-catalog into a datatree object, by opening each dataset into an xarray.Dataset (without loading the data)
kwargs = dict(
    preprocess=combined_preprocessing,  # apply xMIP fixes to each dataset
    xarray_open_kwargs=dict(
        use_cftime=True
    ),  # ensure all datasets use the same time index
    storage_options={
        "token": "anon"
    },  # anonymous/public authentication to google cloud storage
)
# hopefully we can implement https://github.com/intake/intake-esm/issues/562 before the
# actual tutorial, so this would be a lot cleaner
cat_ensemble.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
dt_ensemble = cat_ensemble.to_datatree(**kwargs)

cat_area = col.search(
    source_id=["MPI-ESM1-2-LR"],
    # for the coding exercise, ellipses will go after the equals on this line
    variable_id="areacello",
    member_id="r1i1p1f1",
    table_id="Ofx",  # for the coding exercise, ellipses will go after the equals on this line
    grid_label="gn",
    experiment_id=[
        "historical"
    ],  # for the coding exercise, ellipses will go after the equals on this line
    require_all_on=["source_id"],
)
# hopefully we can implement https://github.com/intake/intake-esm/issues/562 before the
# actual tutorial, so this would be a lot cleaner
cat_area.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
dt_area = cat_area.to_datatree(**kwargs)

# add the area (we can reuse the area from before, since for a given model the horizontal are does not vary between members)
dt_ensemble_with_area = DataTree()
for model, subtree in dt_ensemble.items():
    metric = dt_area["MPI-ESM1-2-LR"]["historical"].ds["areacello"].squeeze()
    dt_ensemble_with_area[model] = subtree.map_over_subtree(_parse_metric, metric)

# global average
# average every dataset in the tree globally
dt_ensemble_gm = dt_ensemble_with_area.map_over_subtree(global_mean)

# calculate anomaly
dt_ensemble_gm_anomaly = datatree_anomaly(dt_ensemble_gm)


def plot_historical_ssp126_ensemble_combined(dt, ax):
    for model in dt.keys():
        datasets = []
        for experiment in ["historical", "ssp126"]:
            datasets.append(dt[model][experiment].ds.coarsen(time=12).mean().tos)

        # concatenate the historical and ssp126 timeseries for each ensemble member
        da_combined = xr.concat(datasets, dim="time")
        # plot annual averages
        da_combined.plot(hue="member_id", ax=ax)


fig, ax = plt.subplots()
# Uncomment below after filling in above ellipses
# plot_historical_ssp126_ensemble_combined(dt_ensemble_gm_anomaly, ax)
plot_historical_ssp126_ensemble_combined(dt_ensemble_gm_anomaly, ax)

ax.set_title(
    "Global Mean SST Anomaly in SSP1-2.6 from a 5-member single-model ensemble"
)
ax.set_ylabel("Global Mean SST Anomaly [$^\circ$C]")
ax.set_xlabel("Year")