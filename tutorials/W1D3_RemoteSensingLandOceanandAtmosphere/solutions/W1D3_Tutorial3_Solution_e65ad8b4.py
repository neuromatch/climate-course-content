
# select a desired date and hours (midnight is zero)
exercise_date_sel = datetime.datetime(2020, 1, 1, 0)

# automatic filename from data_sel. we use strftime (string format time) to get the text format of the file in question.
exercise_file_location = fs.glob(
    "s3://noaa-cdr-ndvi-pds/data/"
    + exercise_date_sel.strftime("%Y")
    + "/VIIRS-Land_v001-preliminary_NPP13C1_S-NPP_*"
    + exercise_date_sel.strftime("%Y%m%d")
    + "*.nc"
)

# now let's check if there is a file match the pattern of the date that we are interested in.
exercise_file_location