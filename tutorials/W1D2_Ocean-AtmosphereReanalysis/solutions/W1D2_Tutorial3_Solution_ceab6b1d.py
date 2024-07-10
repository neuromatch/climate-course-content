# select the variable of interest, group the data by season, and average over all seasons.
# note, this code takes a while to run,
# to test your solution and to reduce the duration select a smaller period
ERA5_season_u10 = ERA5_mm.u10.sel(time=slice('2000','2005')).groupby('time.season').mean()