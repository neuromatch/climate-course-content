

def train_epochs(
    X_train, y_train, X_test, y_test, start_epoch=5, end_epoch=500, step=20
):
    """
    Train an MLPClassifier for a range of epochs and return the training and test accuracy for each epoch.

    Parameters:
    -----------
    X_train: array-like of shape (n_samples, n_features)
        Training input samples.
    y_train: array-like of shape (n_samples,)
        Target values for the training set.
    X_test: array-like of shape (n_samples, n_features)
        Test input samples.
    y_test: array-like of shape (n_samples,)
        Target values for the test set.
    start_epoch: int, optional
        The first epoch to train for. Default is 5.
    end_epoch: int, optional
        The last epoch to train for. Default is 500.
    step: int, optional
        The step size between epochs. Default is 20.

    Returns:
    --------
    tuple
        Two lists, the first containing the training accuracy for each epoch, and the second containing the test
        accuracy for each epoch.
    """

    # Initialize empty lists to store training and test performance
    train_accuracy = []
    test_accuracy = []

    # Loop through a range of epochs and train the MLPClassifier model for each epoch
    for m in range(start_epoch, end_epoch, step):
        # Set random seed for reproducibility
        np.random.seed(144)

        # Fit the MLPClassifier model to the training data for the current epoch
        trained_model = MLPClassifier(max_iter=m).fit(X_train, y_train)

        # Calculate and store the training and test accuracy for the current epoch
        train_accuracy.append(trained_model.score(X_train, y_train))
        test_accuracy.append(trained_model.score(X_test, y_test))

    # Return the lists of training and test performance for each epoch
    return train_accuracy, test_accuracy


# call the train_epochs function and store the returned values in variables train_perfs and test_perfs
# Uncomment next line
train_perfs, test_perfs = train_epochs(X_train, y_train, X_test, y_test)

# calling plotting function to plot the performance vs epochs plot
# Uncomment next line
plot_training_performance(train_perfs, test_perfs)