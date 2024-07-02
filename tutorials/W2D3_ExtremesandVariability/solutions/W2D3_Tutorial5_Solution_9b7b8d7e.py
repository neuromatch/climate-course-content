
# setup dataframe with titles for each parameter
parameters = pd.DataFrame(index=["Location", "Scale", "Shape"])

# add in 1931-1960 parameters
parameters["1931-1960"] = [loc_period1, scale_period1, shape_period1]
parameters["1931-1960"] = gev.fit(data_period1.ssh.values, 0)

# add in 1961-1990 parameters
parameters["1961-1990"] = [loc_period2, scale_period2, shape_period2]
parameters["1961-1990"] = gev.fit(data_period2.ssh.values, 0)

# add in 1991-202 parameters
parameters["1991-2020"] = [loc_period3, scale_period3, shape_period3]
parameters["1991-2020"] = gev.fit(data_period3.ssh.values, 0)

# transpose the dataset so the time periods are rows
parameters = parameters.T

# round the values for viewing
_ = parameters.round(4)  # .astype('%.2f')