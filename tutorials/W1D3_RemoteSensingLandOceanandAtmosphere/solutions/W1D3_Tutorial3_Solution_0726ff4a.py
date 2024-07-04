
# select a desired date and hours (midnight is zero)
exercise_date_sel = datetime.datetime(2019, 9, 1, 0)

# automatic filename from data_sel. we use strftime (string format time) to get the text format of the file in question.
exercise_file_location = fs.glob(
    "s3://noaa-cdr-ndvi-pds/data/"
    + exercise_date_sel.strftime("%Y")
    + "/VIIRS-Land_v001_NPP13C1_S-NPP_*"
    + exercise_date_sel.strftime("%Y%m%d")
    + "*.nc"
)

# now let's check if there is a file that matches the pattern of the date that we are interested in.
exercise_file_location
# VIIRS-Land_v001_NPP13C1_S-NPP_20190901_c20240125203558.nc