from copy import deepcopy


def inlier_outlier_dicts(conformal_dict, suspect_features):
    """
    For each feature, we create a dataframe that contains the true value, the lower bound, the upper
    bound, and the confidence interval. We then create a column called "outlier" that is True if the
    true value is not within the confidence interval. We use the CIs to assign two dictionaries one for the
    inliers and one for the outliers

    Args:
      conformal_dict: a dictionary of dataframes, where each dataframe contains the conformal prediction
    intervals for a given feature.
      suspect_features: a list of features that you want to check for outliers

    Returns:
      A dictionary of inliers and a dictionary of outliers.
    """
    inliers_dict = {}
    outliers_dict = {}

    for feature in suspect_features:
        feature = int(feature)

        mydf = deepcopy(conformal_dict[feature])

        def func(truth, min_val, max_val, interval):
            epsilon = 0.05 * interval
            return not ((truth > min_val - epsilon) & (truth < max_val + epsilon))

        mydf["outlier"] = mydf.apply(
            lambda x: func(
                x["true_val"],
                x["min"],
                x["max"],
                x["conf_interval"],
            ),
            axis=1,
        )

        outlier_df = mydf[mydf["outlier"]]

        inlier_df = mydf[~mydf["outlier"]]

        outlier_ids = outlier_df.index.values
        inlier_ids = inlier_df.index.values

        outliers_dict[feature] = outlier_ids
        inliers_dict[feature] = inlier_ids

    return inliers_dict, outliers_dict


def sort_cis_synth(conformal_dict, inliers_dict, suspect_features, proportion=0.1):
    """
    > This function takes a dictionary of conformal intervals, a dictionary of inlier ids, and a list of suspect
    features. It then creates a dataframe of the conformal intervals for the first suspect feature, and
    then adds the conformal intervals for the other suspect features to the dataframe. It then sorts the
    dataframe by the norm_interval column, and returns the ids of the top and bottom proportion of the
    dataframe

    Args:
      conformal_dict: a dictionary of dataframes, where each dataframe is the conformal intervals for a
    feature
      inliers_dict: a dictionary of inlier ids for each feature
      suspect_features: a list of features that are suspected to be problematic
      proportion: the proportion of the data to use as certain and uncertain

    Returns:
      the indices of the samples with the smallest and largest confidence intervals.
    """
    feature = suspect_features[0]

    df_conformal = conformal_dict[feature]

    inlier_ids = inliers_dict[feature]

    df_inlier = df_conformal.iloc[inlier_ids, :]
    df_inlier[f"{feature}_contrib"] = df_inlier["norm_interval"]

    nsamples = int(len(df_conformal) * proportion)

    if len(suspect_features) > 1:
        for feat in suspect_features[1:]:
            print(f"suspect - {feat}")
            df_conformal = conformal_dict[feat]

            # inlier_ids = inliers_dict[feat]

            df_inlier_feat = df_conformal  # .iloc[inlier_ids,:]

            df_inlier[f"{feat}_contrib"] = df_inlier_feat["norm_interval"]

            df_inlier = df_inlier.add(df_inlier_feat, fill_value=0)

    df_sorted = df_inlier.sort_values(by=["norm_interval"], ascending=True)

    # small_ci_ids = sorted_ids[0:nsamples] #df_sorted.index.values[0:nsamples]
    small_ci_ids = df_sorted.index.values[0:nsamples]

    # df_sorted = df_inlier.sort_values(by=['norm_interval'], ascending=False)

    large_ci_ids = df_sorted.index.values[-nsamples:]

    return small_ci_ids, large_ci_ids, df_sorted


def sort_cis_all(conformal_dict, inliers_dict, suspect_features):
    feature = suspect_features[0]

    proportion = 0.5

    df_conformal = conformal_dict[feature]

    inlier_ids = inliers_dict[feature]

    df_inlier = df_conformal.iloc[inlier_ids, :]

    nsamples = int(len(df_conformal) * proportion)

    if len(suspect_features) > 1:
        for feat in suspect_features[1:]:
            print(f"suspect - {feat}")
            df_conformal = conformal_dict[feat]

            inlier_ids = inliers_dict[feat]

            df_inlier_feat = df_conformal.iloc[inlier_ids, :]

            df_inlier = df_inlier.add(df_inlier_feat, fill_value=0)

    df_sorted = df_inlier.sort_values(by=["norm_interval"], ascending=True)

    # small_ci_ids = sorted_ids[0:nsamples] #df_sorted.index.values[0:nsamples]
    small_ci_ids = df_sorted.index.values[0:nsamples]

    # df_sorted = df_inlier.sort_values(by=['norm_interval'], ascending=False)

    large_ci_ids = df_sorted.index.values[-nsamples:]

    return small_ci_ids, large_ci_ids, df_sorted


def sort_ci_vals(conformal_dict, inliers_dict, suspect_features, proportion=0.1):
    """
    > This function takes in a dictionary of conformal inference results, a dictionary of inlier results, a list of
    suspect features, and a proportion of the data to be used for the analysis.

    It then returns the indices of the data points with the smallest and largest confidence intervals,
    and a dataframe with the sorted confidence intervals.

    Args:
      conformal_dict: a dictionary of dataframes, where each dataframe is the conformal intervals for a
    feature
      inliers_dict: a dictionary of inlier ids for each feature
      suspect_features: a list of features that are suspected to be problematic
      proportion: the proportion of the data to use as certain and uncertain

    Returns:
      the indices of the samples with the smallest and largest confidence intervals.
    """
    feature = suspect_features[0]

    df_conformal = conformal_dict[feature]

    inliers_dict[feature]

    df_inlier = df_conformal
    df_inlier[f"{feature}_contrib"] = df_inlier["norm_interval"]

    nsamples = int(len(df_conformal) * proportion)

    if len(suspect_features) > 1:
        for feat in suspect_features[1:]:
            print(f"Evaluating feature - {feat}")
            df_conformal = conformal_dict[feat]

            df_inlier_feat = df_conformal

            df_inlier[f"{feat}_contrib"] = df_inlier_feat["norm_interval"]

            df_inlier = df_inlier.add(df_inlier_feat, fill_value=0)

    df_sorted = df_inlier.sort_values(by=["norm_interval"], ascending=True)

    small_ci_ids = df_sorted.index.values[0:nsamples]

    large_ci_ids = df_sorted.index.values[-nsamples:]

    return small_ci_ids, large_ci_ids, df_sorted
