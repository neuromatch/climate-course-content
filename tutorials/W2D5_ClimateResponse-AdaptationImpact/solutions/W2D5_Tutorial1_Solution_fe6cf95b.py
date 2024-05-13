
def plot_spatial_distribution(data):
    """Plot spatial distribution of total precipitation.

    Args:
        data (DataFrame): DataFrame containing latitude, longitude, and precipitation data.
        col_name (str): Name of the column containing precipitation data.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))

    # Fill with the name of appropriate column name to plot the spatial distribution of total precipitation
    col_name = 'pr_2015'

    # fill in inputs to functions
    plt.scatter(data['lon'], data['lat'], c=data[col_name], cmap='coolwarm', alpha=0.6)

    plt.colorbar(label='Precipitation (mm/day)')
    plt.title('Spatial Distribution of Precipitation')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.show()

# Uncomment the code below to test your function

plot_spatial_distribution(training_data)