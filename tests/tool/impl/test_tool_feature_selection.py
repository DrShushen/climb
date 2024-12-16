import pytest
from utils import get_tool_output

from climb.tool.impl.tool_feature_selection import feature_selection
from climb.tool.tool_comms import ToolCommunicator


# TODO: To add survival analysis here later
@pytest.mark.parametrize(
    "task_type",
    [
        "classification",
        "regression",
    ],
)
def test_feature_selection_classification(df_classification_path, df_regression_path, task_type):
    """This tests the feature_selection() function in tools. X1, X2, X5 are features that
    are correlated with the target in their respective tasks and should be selected.
    Task coverage include classification, regression, and survival (TODO)"""

    mock_tc = ToolCommunicator()

    if task_type == "classification":
        df_filepath = df_classification_path
    elif task_type == "regression":
        df_filepath = df_regression_path
    else:
        raise ValueError("Task Type Error")

    # Execute function with mock_tc
    feature_selection(
        mock_tc,
        data_file_path=df_filepath,
        workspace="",
        task_type=task_type,
        target_column="target",
    )

    tool_return = get_tool_output(mock_tc).tool_return

    assert "X1" in tool_return
    assert "X2" in tool_return
    assert "X5" in tool_return
