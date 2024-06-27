
def scatter_plot_predicted_vs_true(spatial_test_data, true_values):
    """Create a scatter plot of predicted vs true temperature values.

    Args:
        spatial_test_data: Test features.
        true_values (ndarray): True temperature values.

    Returns:
        None
    """

    # make predictions using the random forest regressor
    spatial_test_predicted = rf_regressor.predict(spatial_test_data)

    spatial_test_score = rf_regressor.score(spatial_test_data, true_values)
    print("\nSpatial Test Data Score:", spatial_test_score)

    # implement plt.scatter() to compare predicted and true temperature values
    _ = plt.scatter(spatial_test_predicted, true_values, color='b', label='Comparison of Predicted and True Temperatures')
    # implement plt.plot() to plot the diagonal line y=x
    _ = plt.plot([min(spatial_test_predicted), max(spatial_test_predicted)], [min(true_values), max(true_values)], color='r', label='Ideal Line')

    # aesthetics
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.title('Annual mean temperature anomaly')

    # add a caption with adjusted y-coordinate to create space
    caption_text = 'The anomalies are calculated by subtracting the annual means of the pre-industrial scenario from \nthe annual means of the respective scenario.'
    plt.figtext(0.5, -0.03, caption_text, ha='center', fontsize=10)  # Adjusted y-coordinate to create space
    plt.legend(loc='upper left')
    plt.show()

# test your function
_ = scatter_plot_predicted_vs_true(spatial_test_data,spatial_test_target)