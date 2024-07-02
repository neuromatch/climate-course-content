
def train_random_forest_model(X_train, y_train, X_test, y_test, random_state):
    """Train a Random Forest model and evaluate its performance.

    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        X_test (ndarray): Test features.
        y_test (ndarray): Test labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        RandomForestRegressor: Trained Random Forest regressor model.
    """

    # train the model on the training data
    rf_regressor = RandomForestRegressor(random_state=random_state)

    # fit the model
    _ = rf_regressor.fit(X_train, y_train)

    print('Performance on training data :', rf_regressor.score(X_train, y_train))
    print('Performance on test data     :', rf_regressor.score(X_test, y_test))

    return rf_regressor

# test the function
rf_model = train_random_forest_model(X_train, y_train, X_test, y_test, random_state=42)