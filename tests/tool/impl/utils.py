import joblib
import pandas as pd
from autoprognosis.utils.serialization import save_model_to_file
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from climb.tool.tool_comms import ToolCommunicator, ToolOutput


def train_and_save_model(df, task_type, model_path, target_column, time_column=None):
    X = df.drop(columns=[target_column])
    if time_column and time_column in X.columns:
        X = X.drop(columns=[time_column])
    y = df[target_column]

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    elif task_type == "regression":
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    else:
        # For survival analysis, you'd need a survival model
        from lifelines import CoxPHFitter

        X[time_column] = df[time_column]
        model = CoxPHFitter()
        model.fit(pd.concat([X, y], axis=1), duration_col=time_column, event_col=target_column)
        joblib.dump(model, model_path)
        return

    model.fit(X, y)
    save_model_to_file(model_path, model)


# NOTE: There's some issue with mocking ToolCommunicator; potentially due to the mixed
# usage of stdout and set_returns. This is why we need to instantiate one over the
# function and get its output with this helper function
def get_tool_output(tc: ToolCommunicator) -> ToolOutput:
    """Helper function to extract the ToolOutput from a ToolCommunicator."""
    p = tc.comm_queue.get()
    while isinstance(p, str):
        p = tc.comm_queue.get()
    if isinstance(p, ToolOutput):
        return p
    else:
        raise ValueError("ToolCommunicator missing ToolOutput")
