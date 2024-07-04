
def plot_spatial_distribution(data, col_name, c_label):
    """
    Plot the spatial distribution of a variable of interest.

    Args:
        data (DataFrame): DataFrame containing latitude, longitude, and data of interest.
        col_name (str): Name of the column containing data of interest.
        c_label (str): Label to describe quantity and unit for the colorbar labeling.

    Returns:
        None
    """
    # create a xarray dataset from the pandas dataframe
    # for convenient plotting with cartopy afterwards
    ds = xr.Dataset({col_name: ('points', data[col_name])},
                    coords={'lon': ('points', data['lon']),
                            'lat': ('points', data['lat'])}
                   )

    # create geoaxes
    ax = plt.axes(projection=ccrs.PlateCarree())

    # add coastlines
    ax.coastlines()

    # plot the data with ax.scatter(..., ..., ..., cmap='coolwarm', transform=ccrs.PlateCarree())
    p = ax.scatter(ds['lon'], ds['lat'], c=ds[col_name], cmap='coolwarm', transform=ccrs.PlateCarree())

    # add a colorbar
    cbar = plt.colorbar(p, orientation='vertical')
    cbar.set_label(c_label)

    # add a grid and labels
    ax.gridlines(draw_labels={"bottom": "x", "left": "y"})

    # add title
    plt.title('Spatial Distribution of\n Annual Mean Anomalies\n')
    plt.show()

# test your function along precipitation data
_ = plot_spatial_distribution(training_data, 'pr_2015', 'Precipitation (mm)')