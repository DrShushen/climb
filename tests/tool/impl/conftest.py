import os

import numpy as np
import pandas as pd
import pytest

N_SAMPLES = 1000


@pytest.fixture
def df_numerical() -> pd.DataFrame:
    """Test dataframe with one normally distributed column and one not normally distributed column."""
    np.random.seed(0)
    normal_data = np.random.normal(loc=0, scale=1, size=N_SAMPLES)
    non_normal_data = np.random.exponential(scale=1, size=N_SAMPLES)
    df = pd.DataFrame({"normal_col": normal_data, "non_normal_col": non_normal_data})
    return df


@pytest.fixture
def df_mixed_types() -> pd.DataFrame:
    """Test dataframe with mixed data types, including integer for categories and numerical data"""
    df = pd.DataFrame(
        {
            "int_categorical": [0, 1, 2, 3, 4] * 100 + np.arange(20).tolist() * 25,
            "int_numerical": [0, 1, 2, 3, 4] * 100 + np.arange(500).tolist(),
            "floats": np.random.normal(size=1000),
            "strings": ["a", "b", "c", "d"] * 250,
        }
    )
    return df


@pytest.fixture(scope="session")
def tmp_workspace(tmp_path_factory):
    """Temporary workspace for serving dynamically generated test fixtures"""
    return tmp_path_factory.mktemp("tmp")


@pytest.fixture(scope="session")
def df_eda_path(tmp_workspace) -> str:
    """Test dataframe with mixed columns for EDA. It is generated with the code below,
    and stored as a static file in tmpworkspace"""
    np.random.seed(0)
    normal_data = np.random.normal(loc=0, scale=1, size=1000)
    non_normal_data = np.random.exponential(scale=1, size=1000)
    cat_data = [1] * 25 + [2] * 25 + [3] * 50 + [4] * 900
    str_data = ["a", "b", "c", "d", "e"] * 200
    nan_data = [np.nan] * 1000
    corr_data = normal_data + 0.1
    non_corr_data = normal_data * np.random.normal(loc=0, scale=1, size=1000)

    df = pd.DataFrame(
        {
            "normal_col": normal_data,
            "non_normal_col": non_normal_data,
            "cat_data": cat_data,
            "str_data": str_data,
            "nan_data": nan_data,
            "corr_data": corr_data,
            "non_corr_data": non_corr_data,
        }
    )

    filepath = os.path.join(tmp_workspace, "test_eda.csv")

    df.to_csv(filepath, index=False)

    return filepath


@pytest.fixture(scope="session")
def df_classification_path(tmp_workspace):
    """Test dataframe for regression task. Correlated features include X1, X2, X5
    The target is T/F depends if it's > 5 after adding up X1, X2, X5"""

    np.random.seed(0)

    # Relevant features
    x1 = np.random.normal(loc=0, scale=1, size=N_SAMPLES)
    x2 = np.random.normal(loc=3, scale=2, size=N_SAMPLES)

    # Irrelevant features
    x3 = np.random.normal(loc=0, scale=1, size=N_SAMPLES)
    x4 = np.random.normal(loc=0, scale=1, size=N_SAMPLES)

    # Categorical feature (relevant)
    x5 = np.random.choice([0, 1, 2], size=N_SAMPLES)

    # Target variable influenced by x1, x2, x5
    y = (x1 + x2 + x5 + np.random.normal(0, 1, N_SAMPLES)) > 5

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "target": y.astype(int)})

    filepath = os.path.join(tmp_workspace, "test_classification.csv")

    df.to_csv(filepath, index=False)

    return filepath


@pytest.fixture(scope="session")
def df_regression_path(tmp_workspace):
    """Test dataframe for regression task. Correlated features include X1, X2, X5.
    Target is just a linear combination of relevant features."""

    np.random.seed(0)

    # Relevant features
    x1 = np.random.uniform(0, 10, N_SAMPLES)
    x2 = np.random.uniform(-5, 5, N_SAMPLES)

    # Irrelevant features
    x3 = np.random.uniform(0, 1, N_SAMPLES)
    x4 = np.random.uniform(0, 1, N_SAMPLES)

    # Categorical feature (relevant)
    x5 = np.random.choice([0, 1, 2], N_SAMPLES)

    # Target variable influenced by x1, x2, x5
    y = x1 + 3 * x2 + 5 * x5 + np.random.normal(0, 1, N_SAMPLES)

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4, "X5": x5, "target": y})
    filepath = os.path.join(tmp_workspace, "test_regression.csv")

    df.to_csv(filepath, index=False)

    return filepath


# TODO: Survival analysis with mock data
@pytest.fixture(scope="session")
def df_survival_path(tmp_workspace):
    filepath = os.path.join(tmp_workspace, "test_survival.csv")

    return filepath


@pytest.fixture
def df_missing_path(tmp_workspace) -> pd.DataFrame:
    """Test dataframe with missing values in its columns"""

    np.random.seed(0)

    single_missing = np.arange(N_SAMPLES).astype(np.float32)
    single_missing[500] = np.nan

    # 10% missing data
    multiple_missing_numerical = np.random.uniform(0, 10, N_SAMPLES)
    multiple_missing_numerical[np.arange(0, N_SAMPLES, 10)] = np.nan

    multiple_missing_categorical = np.random.choice(np.arange(0, 9).tolist() + [np.nan], size=N_SAMPLES)

    data = {
        "single_missing": single_missing,
        "multiple_missing_numerical": multiple_missing_numerical,
        "multiple_missing_categorical": multiple_missing_categorical,
    }

    filepath = os.path.join(tmp_workspace, "test_missing.csv")
    df = pd.DataFrame(data)

    df.to_csv(filepath, index=False)

    return filepath
