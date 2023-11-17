
def evaluate_model_performance(trained_model, X_test, y_test):
    """
    Evaluate the performance of a trained model on a given test set.

    Parameters:
    trained_model (sklearn estimator): A trained scikit-learn estimator.
    X_test (array-like): Test input data.
    y_test (array-like): True labels for the test data.

    Returns:
    test_recall (float): The recall score on the test set.
    test_precision (float): The precision score on the test set.
    """
    # use the trained model to make predictions on the test set using predict()
    pred_test = trained_model.predict(X_test)

    # calculate the recall and precision scores on the test set using recall_score() and precision_score()
    test_recall = skm.recall_score(y_test, pred_test)
    test_precision = skm.precision_score(y_test, pred_test)

    # return the recall and precision scores
    return test_recall, test_precision


# evaluate the performance of the trained model on the test set
test_recall, test_precision = evaluate_model_performance(trained_model, X_test, y_test)
print("Test Recall: ", test_recall)
print("Test Precision: ", test_precision)