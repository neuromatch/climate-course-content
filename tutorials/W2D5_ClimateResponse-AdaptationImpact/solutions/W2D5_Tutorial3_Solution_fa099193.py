def mlp_performance(X_train, y_train, X_test, y_test):
    """
    This function trains multiple MLP classifiers with different numbers of hidden units and returns the training and testing
    accuracy for each classifier.

    Args:
    X_train (ndarray): An array of shape (N_train, M) that contains the training data
    y_train (ndarray): An array of shape (N_train,) that contains the labels for the training data
    X_test (ndarray): An array of shape (N_test, M) that contains the testing data
    y_test (ndarray): An array of shape (N_test,) that contains the labels for the testing data

    Returns:
    tuple: A tuple of three arrays: hidden_units, train_accuracy, and test_accuracy
    """

    # define the range of hidden units to test
    hidden_units = range(5, 300, 15)  # Test 5 to 300 hidden units in steps of 15

    # create empty lists to store training and testing accuracy for different numbers of hidden units
    train_accuracy = []
    test_accuracy = []

    # loop over different numbers of hidden units
    for units in hidden_units:
        # set a seed for random number generation to ensure consistent results
        np.random.seed(144)

        # create an MLP classifier with the current number of hidden units
        classifier = MLPClassifier(hidden_layer_sizes=(units))

        # train the classifier on the training data
        classifier.fit(X_train, y_train)

        # calculate the training and testing accuracy of the classifier
        train_acc = classifier.score(X_train, y_train)
        test_acc = classifier.score(X_test, y_test)

        # add the training and testing accuracy to the corresponding lists
        train_accuracy.append(train_acc)
        test_accuracy.append(test_acc)

    return hidden_units, train_accuracy, test_accuracy


# call mlp_performance function
# Uncomment next line
# hidden_units, train_accuracy, test_accuracy = mlp_performance(X_train, y_train, X_test, y_test)
# plot the training and testing accuracy as a function of the number of hidden units
# Uncomment next line
# plot_mlp_performance(hidden_units, train_accuracy, test_accuracy)