
# generate 100 numbers from a Gaussian distribution with zero mean
x_1_test1 = np.random.randn(data_points)
x_2_test1 = np.random.randn(data_points)

# create an array of x values
regressors_test1 = np.array([x_1_test1, x_2_test1])

# assuming the same relationship of beta =.5 and alpha = .1 as set above,calculate the true y values:
y_test1 = y_func(np.array([beta, 1 - beta]), regressors_test1, alpha)

# use the logistic regression model to make predictions on the new x values
preds = logreg_model.predict(regressors_test1.T)

# calculate the confusion matrix for the model's predictions
cm = confusion_matrix(y_test1, preds)

# visualize the confusion matrix as an image with labeled ticks
fig, ax = plt.subplots()
im = ax.imshow(cm)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])  # plot the confusion matrix using pyplot
ax.set_ylabel("true category")
ax.set_xlabel("predicted category")  # add x and y axis labels
# add colorbar to the plot (assign im to know on what color to base it)
_ = fig.colorbar(im)

# print the percent of correct predictions by the logistic regression model
"Percent correct is " + str(100 * logreg_model.score(regressors_test1.T, y_test1))