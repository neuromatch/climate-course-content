
# create a new instance of the LinearRegression class
reg_model = LinearRegression()

# train the model on the training data i.e on df_cleaned_train,cases_train
reg_model.fit(df_cleaned_train, cases_train)

# print the R^2 score of the trained model on the training data
print("r^2 on training data is: ")
print(reg_model.score(df_cleaned_train, cases_train))

# print the R^2 score of the trained model on the test data
print("r^2 on test data is: ")
print(reg_model.score(df_cleaned_test, cases_test))

fig, ax = plt.subplots()
# create a scatter plot of the predicted values vs. the actual values for the test data
_ = ax.scatter(cases_test, reg_model.predict(df_cleaned_test))

# add 1:1 line
ax.plot(np.array(ax.get_xlim()), np.array(ax.get_xlim()), "k-")

# add axis labels to the scatter plot
ax.set_xlabel("Actual Number of Dengue Cases")
ax.set_ylabel("Predicted Number of Dengue Cases")
ax.set_title("Predicted values vs. the actual values for the test data")