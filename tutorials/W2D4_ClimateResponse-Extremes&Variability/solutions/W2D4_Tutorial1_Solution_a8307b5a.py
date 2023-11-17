
# take 1000 records of 100 samples
random_samples = np.random.normal(mean_pr, std_pr, size=[100, 1000])

# create placeholder for pdfs
pdfs = np.zeros([x_r100.size, 1000])

# loop through all 1000 records and create a pdf of each sample
for i in range(1000):
    # find pdfs
    pdfi = stats.norm.pdf(
        x_r100, random_samples[:, i].mean(), random_samples[:, i].std()
    )

    # add to array
    pdfs[:, i] = pdfi