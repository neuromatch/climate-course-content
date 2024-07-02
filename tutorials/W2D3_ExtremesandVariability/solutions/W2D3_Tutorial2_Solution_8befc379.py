
# define a function to calculate the empirical return period by using the code from the cells above.
# add period and sorted to a dataframe as before
def empirical_period(data):
    df = pd.DataFrame(index=np.arange(data.size))
    df["sorted"] = np.sort(data)[::-1]
    df["ranks"] = np.arange(data.size)
    df["ranks_sp"] = np.sort(stats.rankdata(-data))
    n = data.size
    P = df["ranks_sp"] / (n + 1)
    df["exceedance"] = P
    df["period"] = 1 / df["exceedance"]

    return df[["period", "sorted"]].set_index("period")["sorted"]


# setup figure
fix, ax = plt.subplots()

# create 1000 resamples of the data, with replacement set to true.
for i in range(1000):
    empirical_period(
        # select the randome sample with replacement and plot
        np.random.choice(
            data.precipitation.values, size=data.precipitation.size, replace=True
        )
    ).plot(style="C0-", alpha=0.1, ax=ax)

# aesthetics
ax.plot(precip_df["period"], precip_df["sorted"], "ko")
ax.grid(True)
ax.set_xlabel("Return Period (years)")
ax.set_ylabel("Return Level (mm/day)")
ax.set_xscale("log")