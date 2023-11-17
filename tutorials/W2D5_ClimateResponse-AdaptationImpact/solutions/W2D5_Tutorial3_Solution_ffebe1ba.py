
def calculate_performance(X_train, X_test, y_train, y_test):
    """
    Calculates performance with missing features and only one feature using logistic regression.

    Args:
        X_train (ndarray): The training data.
        X_test (ndarray): The testing data.
        y_train (ndarray): The training labels.
        y_test (ndarray): The testing labels.

    Returns:
        missing_feature_performance (list): A list of performance scores for each missing feature.
        only_feature_performance (list): A list of performance scores for each individual feature.
    """

    missing_feature_performance = (
        []
    )  # create empty list to store performance scores for each missing feature
    only_feature_performance = (
        []
    )  # create empty list to store performance scores for each individual feature

    for feature in range(
        X_train.shape[1]
    ):  # iterate through each feature in the dataset
        # remove the feature from both the training and test set
        X_train_reduced = np.delete(
            X_train, feature, 1
        )  # remove feature from training set
        X_test_reduced = np.delete(X_test, feature, 1)  # remove feature from test set
        reduced_trained_model = LogisticRegression().fit(
            X_train_reduced, y_train
        )  # train a logistic regression model on the reduced training set
        missing_feature_performance.append(
            reduced_trained_model.score(X_test_reduced, y_test)
        )  # calculate the score on the reduced test set and append to list

        # select only the feature from both the training and test set
        X_train_reduced = X_train[:, feature].values.reshape(
            -1, 1
        )  # select only feature from training set
        X_test_reduced = X_test[:, feature].values.reshape(
            -1, 1
        )  # select only feature from test set
        # train a logistic regression model on the reduced training set
        reduced_trained_model = LogisticRegression().fit(
            X_train_reduced, y_train
        )
        only_feature_performance.append(reduced_trained_model.score(X_test_reduced, y_test))
        # calculate the score on the reduced test set and append to list

    return missing_feature_performance, only_feature_performance


missing_feature_performance, only_feature_performance = calculate_performance(
    X_train, X_test, y_train, y_test
)  # Call calculate_performance() function with the training and testing data and the corresponding labels
# Uncomment next line
# plot_feature_performance(missing_feature_performance, only_feature_performance, feature_names) # Plot the performance scores for each missing feature and each individual feature