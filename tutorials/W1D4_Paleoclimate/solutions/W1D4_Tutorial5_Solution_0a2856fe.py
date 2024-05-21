
# bin size of 500
tang_bin_500 = ts_tang.bin(bin_size=500)

# bin size of 1000
tang_bin_1000 = ts_tang.bin(bin_size=1000)

# plot
fig, ax = plt.subplots()  # assign a new plot axis
_ = ts_tang.plot(ax=ax, label="Original", invert_yaxis=True)
_ = tang_bin.plot(ax=ax, label="Binned Default")
_ = tang_bin_500.plot(ax=ax, label="Binned 500 yrs")
_ = tang_bin_1000.plot(ax=ax, label="Binned 1000 yrs")