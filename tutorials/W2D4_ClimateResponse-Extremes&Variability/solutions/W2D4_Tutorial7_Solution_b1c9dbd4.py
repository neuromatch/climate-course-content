
aics = pd.Series(
    index=["Location", "Scale", "Shape", "Location and Scale"],
    data=[
        compute_aic(law_ns),
        compute_aic(law_ns_scale),
        compute_aic(law_ns_shape),
        compute_aic(law_ns_loc_scale),
    ],
)

aics