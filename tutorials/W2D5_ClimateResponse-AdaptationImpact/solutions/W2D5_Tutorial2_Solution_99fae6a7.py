
# train a random forest regression model

# use the RandomForestRegressor we imported earlier
rf = RandomForestRegressor()

# run fit on 'df_cleaned_train' and 'cases_train'
_ = rf.fit(df_cleaned_train, cases_train)

# evaluate the model's performance on the training and testing data
# calculate accuracy by calling rf.score() on 'df_cleaned_train' and 'cases_train'
print("R^2 on training data is: ")
print(rf.score(df_cleaned_train, cases_train))

print("R^2 on test data is: ")
print(rf.score(df_cleaned_test, cases_test))


fig, ax = plt.subplots()
# plot the predicted vs. actual total cases on the test data
_ = ax.scatter(cases_test, rf.predict(df_cleaned_test))

# add 1:1 line
ax.plot(np.array(ax.get_xlim()), np.array(ax.get_xlim()), "k-")

ax.set_xlabel("Actual Total Cases")
ax.set_ylabel("Predicted Total Cases")
ax.set_title("Random Forest Regression")