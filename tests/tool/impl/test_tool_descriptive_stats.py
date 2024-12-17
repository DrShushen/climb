from difflib import SequenceMatcher

import pandas as pd

from climb.tool.impl.tool_descriptive_stats import (
    check_normal_distribution,
    format_descriptive_statistics_table_for_print,
    summarize_dataframe,
    top_n_with_other,
)

EXPECTED_SUMMARY_DF = {
    "Variable": {
        "0": "int_categorical",
        "1": "<<int_categorical>> 0",
        "2": "<<int_categorical>> 2",
        "3": "<<int_categorical>> 3",
        "4": "<<int_categorical>> 4",
        "5": "<<int_categorical>> 1",
        "6": "<<int_categorical>> Other",
        "7": "int_numerical",
        "8": "floats",
        "9": "strings",
        "10": "<<strings>> a",
        "11": "<<strings>> b",
        "12": "<<strings>> c",
        "13": "<<strings>> d",
    },
    "Summary": {
        "0": None,
        "1": "125/1000 (12.5)",
        "2": "125/1000 (12.5)",
        "3": "125/1000 (12.5)",
        "4": "125/1000 (12.5)",
        "5": "125/1000 (12.5)",
        "6": "375/1000 (37.5)",
        "7": "4.00 (2.00 - 249.25)",
        "8": "0.0495 ± 0.9945",
        "9": None,
        "10": "250/1000 (25.0)",
        "11": "250/1000 (25.0)",
        "12": "250/1000 (25.0)",
        "13": "250/1000 (25.0)",
    },
}


def test_check_normal_distribution(df_numerical):
    """Test function checking normal distribution"""
    norm_dist_return = check_normal_distribution(df_numerical)

    assert norm_dist_return == (["normal_col"], ["non_normal_col"])


def test_top_n_with_other(df_mixed_types):
    """Test modified value counts from top_n_with_other"""
    modified_value_counts = top_n_with_other(df_mixed_types, "int_categorical")

    for i in range(5):
        assert modified_value_counts[i] == 125

    # Test Other collapse works
    assert modified_value_counts["Other"] == 375


def test_descriptive_statistics_pipeline(df_mixed_types):
    """This test covers summarize_dataframe and create_descriptive_statistics_table"""

    summary_df, categorical_columns, numeric_columns, normal, _ = summarize_dataframe(df_mixed_types)

    assert set(categorical_columns) == set(["int_categorical", "strings"])
    assert set(numeric_columns) == set(["floats", "int_numerical"])
    assert normal == ["floats"]
    summary_str = format_descriptive_statistics_table_for_print(summary_df)

    test_summary_df = pd.DataFrame(EXPECTED_SUMMARY_DF).set_index("Variable")
    test_summary_str = format_descriptive_statistics_table_for_print(test_summary_df)

    assert SequenceMatcher(None, summary_str, test_summary_str).ratio() > 0.95
