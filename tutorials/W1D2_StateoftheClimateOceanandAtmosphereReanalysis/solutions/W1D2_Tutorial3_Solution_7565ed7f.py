
ERA5_mm.mean("longitude")["v10"].groupby("time.month").mean().plot()