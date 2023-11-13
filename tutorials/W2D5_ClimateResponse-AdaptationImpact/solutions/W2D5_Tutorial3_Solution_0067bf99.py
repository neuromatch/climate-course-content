
def evaluate_model(X_train, y_train, X_test, y_test):
    """
    Fits a logistic regression model to the training data and evaluates its accuracy on the training and test sets.

    Parameters:
    X_train (numpy.ndarray): The feature data for the training set.
    y_train (numpy.ndarray): The target data for the training set.
    X_test (numpy.ndarray): The feature data for the test set.
    y_test (numpy.ndarray): The target data for the test set.

    Returns:
    tuple: A tuple containing the training accuracy and test accuracy of the trained model.
    """
    # create an instance of the LogisticRegression class and fit it to the training data
    trained_model = LogisticRegression().fit(X_train, y_train)

    # calculate the training and test accuracy of the trained model
    train_accuracy = trained_model.score(X_train, y_train)
    test_accuracy = trained_model.score(X_test, y_test)

    # return the trained_model, training and test accuracy of the trained model
    return trained_model, train_accuracy, test_accuracy


trained_model, train_acc, test_acc = evaluate_model(X_train, y_train, X_test, y_test)
print("Training Accuracy: ", train_acc)
print("Test Accuracy: ", test_acc)