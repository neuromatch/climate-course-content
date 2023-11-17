
# set a random seed to ensure reproducibility of results
np.random.seed(144)

# fit the MLPClassifier model on the training data by calling .fit() on MLPClassifier()
trained_model = MLPClassifier().fit(X_train, y_train)

# print the training accuracy and test accuracy of the model
print(" Training Accuracy: ", trained_model.score(X_train, y_train))
print(" Test Accuracy:     ", trained_model.score(X_test, y_test))