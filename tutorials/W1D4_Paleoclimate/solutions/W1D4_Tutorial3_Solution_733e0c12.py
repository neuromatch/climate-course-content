
# initate plot with the specific figure size
fig = plt.figure(figsize=(9, 6))

# set base map projection
ax = plt.axes(projection=ccrs.Robinson())

ax.set_global()

# add land fratures using gray color
ax.add_feature(cfeature.LAND, facecolor="k")

# add coastlines
ax.add_feature(cfeature.COASTLINE)

# add the proxy locations
# Uncomment and complete following line
_ = ax.scatter(
    euro_lake_lon,
    euro_lake_lat,
    transform=ccrs.Geodetic(),
    label="lake sediment",
    s=50,
    marker="d",
    color=[0.52734375, 0.8046875, 0.97916667],
    edgecolor="k",
    zorder=2,
)
_ = ax.scatter(
    euro_tree_lon,
    euro_tree_lat,
    transform=ccrs.Geodetic(),
    label="tree ring",
    s=50,
    marker="p",
    color=[0.73828125, 0.71484375, 0.41796875],
    edgecolor="k",
    zorder=2,
)
_ = ax.scatter(
    euro_spel_lon,
    euro_spel_lat,
    transform=ccrs.Geodetic(),
    label="speleothem",
    s=50,
    marker="8",
    color=[1, 0, 0],
    edgecolor="k",
    zorder=2,
)

# change the map view to zoom in on central Pacific
ax.set_extent((0, 360, 0, 90), crs=ccrs.PlateCarree())

ax.legend(
    scatterpoints=1,
    bbox_to_anchor=(0, -0.4),
    loc="lower left",
    ncol=3,
    fontsize=15,
)