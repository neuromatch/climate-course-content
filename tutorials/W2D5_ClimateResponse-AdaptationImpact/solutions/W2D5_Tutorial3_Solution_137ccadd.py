
# separate the positive and negative classes
pos_inds = np.where(
    y_train == 1
)  # indices of positive class (i.e positive class will have y_train as 1)
neg_inds = np.where(
    y_train == 0
)  # indices of negative class (i.e positive class will have y_train as 0)

# create subplots for each feature
fig, ax = plt.subplots(3, 4)
ax = ax.flatten()
for i in range(12):
    # plot histograms of the positive and negative classes for the current feature
    n, bins, patches = ax[i].hist(
        X_train[:, i][pos_inds], alpha=0.5
    )  # histogram of the positive class
    ax[i].hist(
        X_train[:, i][neg_inds], bins=bins, alpha=0.5
    )  # histogram of the negative class
    ax[i].set_title(
        feature_names[i]
    )  # set the title of the subplot to the current feature number

fig.tight_layout()  # adjust spacing between subplots

# calculate % by (number of positive/(number of positive + number of negative)); can use len to
print("Percentage of positive samples: " + str(len(pos_inds) / (len(pos_inds) + len(neg_inds))))