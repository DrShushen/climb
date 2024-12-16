import os
from difflib import SequenceMatcher

import pandas as pd
from utils import get_tool_output, train_and_save_model

from climb.tool.impl.tool_feature_importance import shap_explainer
from climb.tool.tool_comms import ToolCommunicator


def test_shap_explainer_classification(tmp_workspace, df_classification_path, shap_output):
    df = pd.read_csv(df_classification_path)

    # Train and save a classification model for testing
    model_path = os.path.join(tmp_workspace, "classification_model.pkl")
    train_and_save_model(df, "classification", model_path, "target")

    mock_tc = ToolCommunicator()

    # Execute function with mock_tc
    shap_explainer(
        mock_tc,
        data_file_path=df_classification_path,
        model_path=model_path,
        target_variable="target",
        problem_type=None,
        workspace=tmp_workspace,
    )

    tool_return = get_tool_output(mock_tc).tool_return

    assert "shap_bar.png" in os.listdir(tmp_workspace)
    assert "shap_beeswarm.png" in os.listdir(tmp_workspace)

    assert SequenceMatcher(None, tool_return, shap_output).ratio() > 0.9


# TODO: To test permutation explainer
