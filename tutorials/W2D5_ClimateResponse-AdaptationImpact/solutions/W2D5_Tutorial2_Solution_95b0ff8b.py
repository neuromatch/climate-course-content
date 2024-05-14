
def fit_and_visualize_rf(training_data, target):
    """Fit a random forest regressor to the training data and visualize the results.

    Args:
        training_data (array-like): Input data for training the model.
        target (array-like): Target variable for training the model.

    Returns:
        None
    """

    # Fit the random forest regressor to the training data
    rf_regressor.fit(training_data, target)

    # Print the R-squared score of the model
    print(rf_regressor.score(training_data, target))

    # Predict the target variable using the trained model
    predicted = rf_regressor.predict(training_data)

    # Create scatter plot
    plt.scatter(predicted,target,color='b',label='Comparison of Predicted and True Temperatures')
    plt.plot([0,4],[0,4],color='r', label='Ideal Line') # Add a diagonal line for reference
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.legend()
    plt.title('Annual mean temperature anomaly', fontsize=14)
    # Add a caption with adjusted y-coordinate to create space
    caption_text = 'The anomalies are calculated by subtracting the annual means of the pre-industrial scenario from \nthe annual means of the respective scenario.'
    plt.figtext(0.5, -0.03, caption_text, ha='center', fontsize=7)  # Adjusted y-coordinate to create space

## Uncomment the code below to test your function

fit_and_visualize_rf(training_data, target)