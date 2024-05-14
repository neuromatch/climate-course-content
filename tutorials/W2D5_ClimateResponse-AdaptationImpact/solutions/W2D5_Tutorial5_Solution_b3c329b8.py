
def evaluate_and_plot_scenario_performance(rf_regressor, scenario_test_data, scenario_test_target):
    """Evaluate the performance of the pre-trained model on the new emissions scenario
    and create a scatter plot of predicted vs. true temperature values.

    Args:
        rf_regressor (RandomForestRegressor): Pre-trained Random Forest regressor model.
        scenario_test_data (ndarray): Test features for the new emissions scenario.
        scenario_test_target (ndarray): True temperature values for the new emissions scenario.

    Returns:
        float: Score of the model on the new emissions scenario.
    """

    # Evaluate the model on the new emissions scenario
    scenario_test_score = rf_regressor.score(scenario_test_data, scenario_test_target)
    print("Scenario Test Score:", scenario_test_score)

    # Predict temperature values for the new emissions scenario
    scenario_test_predicted = rf_regressor.predict(scenario_test_data)

    # Create a scatter plot of predicted vs. true temperature values
    plt.figure()
    plt.scatter(scenario_test_predicted, scenario_test_target, color='b',  label='Comparison of Predicted and True Temperatures')
    plt.plot([min(scenario_test_predicted), max(scenario_test_predicted)], [min(scenario_test_target), max(scenario_test_target)], color='r',label='Ideal Line')
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.title('Annual mean temperature anomaly (New Emissions Scenario)')
    plt.grid(True)
    plt.show()

    return scenario_test_score

# Example usage:
scenario_test_score = evaluate_and_plot_scenario_performance(rf_regressor, scenario_test_data, scenario_test_target)