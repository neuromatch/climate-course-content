cat_area = col.search(
    source_id = "TaiESM1",
    # Add the appropriate variable_id
    variable_id = "areacello",
    member_id = "r1i1p1f1",
    # Add the appropriate table_id
    table_id = "Ofx",
    grid_label = "gn",
    # Add the appropriate experiment_id
    experiment_id = ["historical"],
    require_all_on = ["source_id"],
)
# construct sub-catalog datatree with attributes
cat_area.esmcat.aggregation_control.groupby_attrs = ["source_id", "experiment_id"]
dt_area = cat_area.to_datatree(**kwargs)

# instantiate a DataTree object
dt_with_area = DataTree()

# add the sub-catalog to the DataTree object via map_over_subtree()
for model, subtree in dt.items():
    metric = dt_area[model]["historical"].ds["areacello"]
    dt_with_area[model] = subtree.map_over_subtree(_parse_metric, metric)