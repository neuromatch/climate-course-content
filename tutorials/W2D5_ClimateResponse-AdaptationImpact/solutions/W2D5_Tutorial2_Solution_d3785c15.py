
# creating a scatter plot of 'year' vs 'total_cases' in the DataFrame 'df_dengue'.
_ = df_labels.plot.scatter("year", "total_cases")

# creating a scatter plot of 'weekofyear' vs 'total_cases' in the DataFrame 'df_dengue'.
_ = df_labels.plot.scatter("weekofyear", "total_cases")

# creating a new DataFrame named 'new' that contains only the columns 'total_cases' and 'city' from 'df_dengue'.
new = df_labels[["total_cases", "city"]].copy()

# creating histograms of the 'total_cases' column in 'new' separated by the values in the 'city' column.
new.hist(by="city", bins=30)