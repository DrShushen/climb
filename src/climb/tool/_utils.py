import math
from typing import Any, List

import numpy as np
import pandas as pd


def id_numerics_actually_categoricals(
    df: pd.DataFrame, unique_values_threshold: int = 20, unique_values_ratio_threshold: float = 0.05
) -> List[str]:
    """
    Identifies numeric columns that should be considered categorical based on whether they are all integers, and if so,
    whether the number of unique values meets certain criteria: either a maximum number of unique values or a maximum
    ratio of unique values to total number of rows.

    Args:
        df (pd.DataFrame):
            DataFrame to analyze.
        unique_values_threshold (int):
            Maximum number of unique values for a column to be considered categorical. Default is 20.
        unique_values_ratio_threshold (float):
            Maximum ratio of unique values to total rows for a column to be considered categorical. Default is 0.05.

    Returns:
        List[str]: List of column names that should be considered categorical.
    """
    # TODO: This heuristic should perhaps be refined.

    column_classification = []
    for column in df.select_dtypes(include=[np.number]).columns:  # type: ignore
        # Check if all values in the column are integers
        if df[column].dropna().apply(lambda x: float.is_integer(float(x))).all():
            unique_values = df[column].nunique()
            total_rows = len(df)

            # Check against unique value thresholds
            if unique_values <= min(unique_values_threshold, unique_values_ratio_threshold * total_rows):
                column_classification.append(column)  # Considered categorical
                continue

    print(f"Identified numeric columns that should be considered categorical:\n{column_classification}")
    return column_classification


def decimal_places_for_sf(x: Any, n_sf: int = 3) -> int:
    """
    Determine the number of decimal places required to display a float to 3 significant figures.
    """
    if x == 0:
        return 0  # No decimal places needed for 0

    # Calculate the order of magnitude of the number:
    order_of_magnitude = math.floor(math.log10(abs(x)))

    # Calculate decimal places needed for 3 significant figures:
    decimal_places = (n_sf - 1) - order_of_magnitude

    return max(decimal_places, 0)  # Ensure non-negative number of decimal places.
