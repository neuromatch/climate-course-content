
def scatter_plot_predicted_vs_true(spatial_test_data, true_values):
    """Create a scatter plot of predicted vs true temperature values.

    Args:
        spatial_test_data: Test features.
        true_values (ndarray): True temperature values.

    Returns:
        None
    """
    #################################################
    ## TODO for students: Implement scatter plot of predicted vs true temperature values ##
    # Use plt.scatter() to plot the data points
    # Use plt.plot() to plot the diagonal line y=x
    # Label the x-axis as 'Predicted Temperatures' and y-axis as 'True Temperatures'
    #################################################

    # Make predictions using the random forest regressor
    spatial_test_predicted = rf_regressor.predict(spatial_test_data)

    spatial_test_score = rf_regressor.score(spatial_test_data, true_values)
    print("Spatial Test Data Score:", spatial_test_score)

    # Implement scatter plot of predicted vs true temperature values
    plt.scatter(spatial_test_predicted, true_values, color='b', label='Comparison of Predicted and True Temperatures')
    plt.plot([min(spatial_test_predicted), max(spatial_test_predicted)], [min(true_values), max(true_values)], color='r', label='Ideal Line')
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.title('Annual mean temperature anomaly', fontsize=14)

    # Add a caption with adjusted y-coordinate to create space
    caption_text = 'The anomalies are calculated by subtracting the annual means of the pre-industrial scenario from \nthe annual means of the respective scenario.'
    plt.figtext(0.5, -0.03, caption_text, ha='center', fontsize=7)  # Adjusted y-coordinate to create space
    plt.legend(loc='upper left')
    plt.show()

# Remove the comment to run this code
scatter_plot_predicted_vs_true(spatial_test_data,spatial_test_target)