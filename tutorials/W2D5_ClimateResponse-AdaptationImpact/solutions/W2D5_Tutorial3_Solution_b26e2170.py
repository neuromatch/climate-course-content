# Evaluate and plot feature importance with the permutation method

# calculate the permutation feature importance using trained_model, X_test and y_test
perm_feat_imp = permutation_importance(
    trained_model, X_test, y_test, n_repeats=10, random_state=0
)

# Uncomment next line
# plot_permutation_feature_importance(perm_feat_imp, X_test, feature_names)