import math
import os

import pandas as pd

from climb.tool.impl.tool_imputation import hyperimpute_impute
from climb.tool.tool_comms import ToolCommunicator


def test_hyperimpute_impute(tmp_workspace, df_missing_path):
    mock_tc = ToolCommunicator()
    imputed_file_path = os.path.join(tmp_workspace, "df_imputed.csv")

    # Execute function with mock_tc
    hyperimpute_impute(mock_tc, df_missing_path, imputed_file_path, workspace=tmp_workspace)

    missing_df = pd.read_csv(df_missing_path)
    imputed_df = pd.read_csv(imputed_file_path)

    # The single missing value ~should~ be 500. Assert the imputed value is close enough
    assert math.isclose(500, imputed_df.single_missing[500], rel_tol=0.1)

    # For columns with multiple missing variables, we mostly just want to make sure the
    # distribution is not too different after imputation
    for col in ["multiple_missing_categorical", "multiple_missing_numerical"]:
        assert math.isclose(missing_df[col].mean(), imputed_df[col].mean(), rel_tol=0.01)
        assert math.isclose(missing_df[col].std(), imputed_df[col].std(), rel_tol=0.1)
