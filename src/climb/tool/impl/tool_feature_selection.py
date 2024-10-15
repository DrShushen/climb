import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def feature_selection(
    tc: ToolCommunicator,
    data_file_path: str,
    workspace: str,  # pylint: disable=unused-argument
    task_type: str,
    target_column: str,
    time_column: Optional[str] = None,
) -> None:
    tc.print("Setting up feature pruner...")

    if task_type == "survival_analysis" and time_column is None:
        raise ValueError('`time_column` was not provided, but it is required for "survival_analysis" `task_type`.')

    df = pd.read_csv(data_file_path)

    # check the percentage of nans in the dataset
    nan_percentage = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if nan_percentage > 0.1:
        tc.print(
            "The dataset has more than 10% missing values. Please impute the missing values with a tool like HyperImpute before running this tool."
        )
        tc.set_returns(
            "The dataset has more than 10% missing values. Please impute the missing values with a tool like HyperImpute before running this tool."
        )
        return
    elif nan_percentage > 0:
        # Impute missing values with s simple approach as nans to too few to have a significant impact
        # impute categorical columns with the mode
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        # impute all other columns with median
        try:
            df = df.fillna(df.median())
        except TypeError:
            df = df.fillna(df.mean(numeric_only=True))

    if len(df) > 100000:  # HACK: limit the number of rows to 100,000 for speed. Could run multiple subsets instead?
        df = df.sample(n=100000, random_state=1)

    if task_type == "survival_analysis":
        # convert to classification using time horizon
        time_horizon = df[time_column].median()
        df["TEMP_CLASSIFICATION_FROM_SURVIVAL_EVENT_COL"] = np.where(
            (df[time_column] < time_horizon) & (df[target_column] == 1), 1, 0
        )
        df.drop(time_column, axis=1, inplace=True)
        df.drop(target_column, axis=1, inplace=True)
        # change target column to the new classification column
        target_column = "TEMP_CLASSIFICATION_FROM_SURVIVAL_EVENT_COL"
        task_type = "classification"

    # convert categorical columns to ordinal
    enc = OrdinalEncoder()
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        df[col] = df[col].astype(str)
        if df[col].nunique() > 1:  # Only encode columns with more than one unique value
            # Reshape and transform:
            df[col] = enc.fit_transform(df[col].values.reshape(-1, 1))  # pyright: ignore
        else:
            df[col] = 1.0  # If only one unique value, set to 1.0

    y = df[target_column]
    X = df.drop(target_column, axis=1)
    if task_type == "classification":
        rfc = RandomForestClassifier(random_state=1, n_estimators=1000, max_depth=5)
    elif task_type == "regression":
        rfc = RandomForestRegressor(random_state=1, n_estimators=1000, max_depth=5)
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    boruta_selector = BorutaPy(
        rfc,
        n_estimators="auto",  # type: ignore
        verbose=0,
        random_state=1,
    )
    boruta_selector.fit(np.array(X.values), np.array(y.values))

    important_features = X.columns[boruta_selector.support_].to_list()  # type: ignore
    if len(important_features) == 0:
        important_features = X.columns.to_list()
    out_message = f"Here are the selected features: {', '.join(important_features)}"
    tc.print(out_message)
    tc.set_returns(out_message)


class BorutaFeatureSelection(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            feature_selection,
            wd=self.working_directory,
            data_file_path=real_path,
            task_type=kwargs["task_type"],
            target_column=kwargs["target_column"],
            time_column=kwargs.get("time_column"),
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "feature_selection"

    @property
    def description(self) -> str:
        return """
        Uses an **automated feature selection** library to suggest the most important features in the dataset. Users may want to use this tool to
        drop the features that are not in the list of important features. Reducing the number of features in their dataset,
        which can help to reduce overfitting and improve the performance of machine learning models. They may not want to drop
        all unimportant features, as they may have domain knowledge that suggests that some features are important.
        """

    @property
    def specification(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_file_path": {"type": "string", "description": "Path to the data file."},
                        "task_type": {
                            "type": "string",
                            "description": "The type of task that the user is working on. This can one of the following: 'classification', 'regression', 'survival_analysis'.",
                        },
                        "target_column": {
                            "type": "string",
                            "description": "The target column to predict in the research task. For survival analysis this should be the event column.",
                        },
                        "time_column": {
                            "type": "string",
                            "description": "The time to event column. This is only applicable for survival analysis tasks, where it is mandatory.",
                        },
                    },
                    "required": ["data_file_path", "task_type", "target_column"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "uses the **automated feature selection** library to automatically find the most important features in your data."
