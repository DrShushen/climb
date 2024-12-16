from difflib import SequenceMatcher

from climb.tool.impl.tool_descriptive_stats import (
    check_normal_distribution,
    format_descriptive_statistics_table_for_print,
    summarize_dataframe,
    top_n_with_other,
)

EXPECTED_EDA_OUTPUT = "Dataset Shape: 1000 rows and 8 columns\nColumn Names and Types:\nUnnamed: 0          int64\nnormal_col        float64\nnon_normal_col    float64\ncat_data            int64\nstr_data           object\nnan_data          float64\ncorr_data         float64\nnon_corr_data     float64\n\nDescriptive Statistics for Numerical Features:\n        Unnamed: 0   normal_col  non_normal_col     cat_data  nan_data    corr_data  non_corr_data\ncount  1000.000000  1000.000000     1000.000000  1000.000000       0.0  1000.000000    1000.000000\nmean    499.500000    -0.045257        1.047098     3.825000       NaN     0.054743      -0.033296\nstd     288.819436     0.987527        1.022801     0.587128       NaN     0.987527       0.949684\nmin       0.000000    -3.046143        0.000074     1.000000       NaN    -2.946143      -4.484641\n25%     249.750000    -0.698420        0.312780     4.000000       NaN    -0.598420      -0.410167\n50%     499.500000    -0.058028        0.751197     4.000000       NaN     0.041972      -0.012606\n75%     749.250000     0.606951        1.465758     4.000000       NaN     0.706951       0.335221\nmax     999.000000     2.759355        6.520167     4.000000       NaN     2.859355       7.530807\nskew      0.000000     0.033910        1.743307    -3.661486       NaN     0.033910       0.506095\nkurt     -1.200000    -0.040977        3.858616    13.038093       NaN    -0.040977       8.585442\n\nIdentified numeric value columns that should most likely be considered categoricals:\n['cat_data', 'nan_data'].\nThis is done by checking whether the column contains only integers and has a low number of unique values (<20 or <5% of total examples).\n\nDetailed Information on Categorical Variables:\nnan_data - Unique Values: 0 \nTop 5 Values:\nSeries([], )\n\nstr_data - Unique Values: 5 \nTop 5 Values:\nstr_data\na    200\nb    200\nc    200\nd    200\ne    200\n\ncat_data - Unique Values: 4 \nTop 5 Values:\ncat_data\n4    900\n3     50\n1     25\n2     25\n\nMissing Values Analysis:\nnan_data    1000\n\nCount of columns with all NaN values: 1\nCorrelation Analysis:\n\nMost Positively Correlated Features:\n        Feature 1       Feature 2  Correlation\n0      normal_col       corr_data     1.000000\n1      Unnamed: 0        cat_data     0.477777\n2        cat_data   non_corr_data     0.040232\n3      normal_col  non_normal_col     0.023444\n4  non_normal_col       corr_data     0.023444\n5      Unnamed: 0   non_corr_data     0.000423\n\nMost Negatively Correlated Features:\n        Feature 1       Feature 2  Correlation\n0       corr_data   non_corr_data    -0.095358\n1      normal_col   non_corr_data    -0.095358\n2      normal_col        cat_data    -0.056641\n3        cat_data       corr_data    -0.056641\n4  non_normal_col        cat_data    -0.038787\n5      Unnamed: 0  non_normal_col    -0.031014\n6  non_normal_col   non_corr_data    -0.005799\n7      Unnamed: 0      normal_col    -0.005452\n8      Unnamed: 0       corr_data    -0.005452\n\nOutlier Identification for Numerical Features:\nUnnamed: 0 - Outliers Count: 0\n[Lower Bound: -624, Upper Bound: 1.62e+03]\nnormal_col - Outliers Count: 1\n[Lower Bound: -2.98, Upper Bound: 2.89]\nnon_normal_col - Outliers Count: 32\n[Lower Bound: -1.7, Upper Bound: 3.48]\ncat_data - Outliers Count: 100\n[Lower Bound: 3, Upper Bound: 3]\nnan_data - Outliers Count: 0\n[Lower Bound: -1, Upper Bound: -1]\ncorr_data - Outliers Count: 1\n[Lower Bound: -2.88, Upper Bound: 2.99]\nnon_corr_data - Outliers Count: 74\n[Lower Bound: -1.71, Upper Bound: 1.64]\n\nDuplicate Records: 0\n\n"


def test_check_normal_distribution(df_numerical):
    """Test function checking normal distribution"""
    norm_dist_return = check_normal_distribution(df_numerical)

    assert norm_dist_return == (["normal_col"], ["non_normal_col"])


def test_top_n_with_other(df_mixed_types):
    """Test modiifed value counts from top_n_with_other"""
    modified_value_counts = top_n_with_other(df_mixed_types, "int_categorical")

    for i in range(5):
        assert modified_value_counts[i] == 125

    # Test Other collapse works
    assert modified_value_counts["Other"] == 375


def test_descriptive_statistics_pipeline(df_mixed_types, test_summary_df):
    """This test covers summarize_dataframe and create_descriptive_statistics_table"""

    summary_df, categorical_columns, numeric_columns, normal, _ = summarize_dataframe(df_mixed_types)

    assert set(categorical_columns) == set(["int_categorical", "strings"])
    assert set(numeric_columns) == set(["floats", "int_numerical"])
    assert normal == ["floats"]
    summary_str = format_descriptive_statistics_table_for_print(summary_df)
    test_summary_str = format_descriptive_statistics_table_for_print(test_summary_df)
    print(summary_str)
    print(test_summary_str)
    print(SequenceMatcher(None, summary_str, test_summary_str).ratio())
    assert SequenceMatcher(None, summary_str, test_summary_str).ratio() > 0.95
