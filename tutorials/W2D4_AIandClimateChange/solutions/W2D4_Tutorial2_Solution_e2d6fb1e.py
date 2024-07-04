
def fit_and_visualize_rf(training_data, target):
    """Fit a random forest regressor to the training data and visualize the results.

    Args:
        training_data (array-like): Input data for training the model.
        target (array-like): Target variable for training the model.

    Returns:
        None
    """

    # fit the random forest regressor to the training data
    _ = rf_regressor.fit(training_data, target)

    # print the R-squared score of the model
    _ = print(rf_regressor.score(training_data, target))

    # predict the target variable using the trained model
    predicted = rf_regressor.predict(training_data)

    # Create scatter plot
    plt.scatter(predicted,target,color='b',label='Comparison of Predicted and True Temperatures')
    plt.plot([0,4],[0,4],color='r', label='Ideal Line') # add a diagonal line for reference
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.legend()
    plt.title('Annual mean temperature anomaly')
    # add a caption with adjusted y-coordinate to create space
    caption_text = 'The anomalies are calculated by subtracting the annual means of the pre-industrial scenario from \nthe annual means of the respective scenario.'
    plt.figtext(0.5, -0.03, caption_text, ha='center', fontsize=10)  # adjusted y-coordinate to create space

# test your function
_ = fit_and_visualize_rf(training_data, target)