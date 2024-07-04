
# define the date of your interest YYYYMMDD (e.g., 20030701)
# select a desired date and hours (midnight is zero)
date_sel_exercise = datetime.datetime(
    2022, 1, 1
)

# locate the data in the AWS S3 bucket
# hint: use the file pattern that we described
file_location_exercise = fs.glob(
    "s3://noaa-cdr-ndvi-pds/data/"
    + date_sel_exercise.strftime("%Y")
    + "/VIIRS-Land_v001*_NPP13C1_S-NPP_*"
    + date_sel_exercise.strftime("%Y%m%d")
    + "_c*.nc"
)

# open file connection to the file in AWS S3 bucket and use xarray to open the NDVI CDR file
# open the file
ds_exercise = xr.open_dataset(
    pooch.retrieve(
        "http://s3.amazonaws.com/" + file_location_exercise[0], known_hash=None
    ),
    decode_times=False # to address overflow issue
)


# get the QA value and extract the high quality data mask and Mask NDVI data to keep only high quality value
# hint: reuse the get_quality_info helper function we defined
ndvi_masked_exercise = ds_exercise.NDVI.where(get_quality_info(ds_exercise.QA))

# plot high quality NDVI data
# hint: use plot() function
# ndvi_masked_exercise.coarsen(latitude=5).mean().coarsen(longitude=5).mean().plot(
#     vmin=-0.1, vmax=1.0, aspect=1.8, size=5
# )