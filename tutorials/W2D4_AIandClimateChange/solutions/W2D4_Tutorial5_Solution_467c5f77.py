
def evaluate_and_plot_scenario_performance(rf_regressor, scenario_test_data, scenario_test_target):
    """Evaluate the performance of the pre-trained model on the new emissions scenario
    and create a scatter plot of predicted vs. true temperature values.

    Args:
        rf_regressor (RandomForestRegressor): Pre-trained Random Forest regressor model.
        scenario_test_data (ndarray): Test features for the new emissions scenario.
        scenario_test_target (ndarray): True temperature values of the new emissions scenario.

    Returns:
        float: Score of the model on the new emissions scenario.
    """

    # predict temperature values for the new emissions scenario
    scenario_test_predicted = rf_regressor.predict(scenario_test_data)

    # evaluate the model on the new emissions scenario
    scenario_test_score = rf_regressor.score(scenario_test_data, scenario_test_target)
    print("Scenario Test Score:", scenario_test_score)

    # implement plt.scatter() to compare predicted and true temperature values
    plt.figure()
    _ = plt.scatter(scenario_test_predicted, scenario_test_target, color='b',  label='Comparison of Predicted and True Temperatures')
    # implement plt.plot() to plot the diagonal line y=x
    _ = plt.plot([min(scenario_test_predicted), max(scenario_test_predicted)], [min(scenario_test_target), max(scenario_test_target)], color='r',label='Ideal Line')

    # aesthetics
    plt.xlabel('Predicted Temperatures (K)')
    plt.ylabel('True Temperatures (K)')
    plt.title('Annual mean temperature anomaly\n(New Emissions Scenario)')
    plt.grid(True)
    plt.show()

    return scenario_test_score

# test your function
scenario_test_score = evaluate_and_plot_scenario_performance(rf_regressor, scenario_test_data, scenario_test_target)