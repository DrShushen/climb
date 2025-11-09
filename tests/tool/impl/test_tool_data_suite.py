import os
from typing import Any

import numpy as np
import pandas as pd

from climb.tool.impl.tool_data_suite import data_suite_insights


class MockToolCommunicator:
    def __init__(self):
        self.print_args_list = []
        self.tool_return = None

    """Mock tool communicator."""

    def set_returns(self, tool_return: Any) -> None:
        """Set the returns of the tool."""
        self.tool_return = tool_return

    def print(self, *args: Any) -> None:
        """Print the arguments."""
        self.print_args_list.append(args)


def test_with_synthetic_dataframe(tmp_path):
    """Test the data_suite_insights tool with a synthetic dataframe."""
    # Generate a dataframe with 20 random columns and 1000 rows
    np.random.seed(42)
    df = pd.DataFrame(np.random.rand(1000, 20))
    df.columns = [f"feature_{i}" for i in range(20)]
    # Add a target column that is a linear combination of the features
    df["target"] = df.iloc[:, 0] + df.iloc[:, 1] + df.iloc[:, 2] + df.iloc[:, 3] + df.iloc[:, 4]
    # Make some boolean and categorical columns
    df["bool_col"] = df["feature_0"] > 0.5
    df["cat_col"] = df["feature_1"].apply(lambda x: round(x, 2)).astype(str)
    df.to_csv(tmp_path / "test_data_suite.csv", index=False)

    mock_tc = MockToolCommunicator()

    # Execute the toolL
    data_suite_insights(
        tc=mock_tc, data_file_path=tmp_path / "test_data_suite.csv", target_column="target", workspace=tmp_path
    )

    assert mock_tc.tool_return is not None
    assert "may perform poorly" in mock_tc.tool_return
    assert "Data_suite_examples_to_collect.csv" in mock_tc.tool_return
    assert len(mock_tc.print_args_list) > 0
    assert "Running analysis for feature = 0" in mock_tc.print_args_list[0]
    assert os.path.exists(tmp_path / "Data_suite_examples_to_collect.csv")
