import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase
from .data_suite.models.conformal import conformal_class
from .data_suite.models.copula import fit_sample_copula
from .data_suite.models.representation import compute_representation
from .data_suite.utils.helpers import inlier_outlier_dicts, sort_cis_synth


def clean_dataframe(df, unique_threshold=15):
    # Identify column data types
    inferred_categorical_columns = []
    inferred_numerical_columns = []
    inferred_boolean_columns = []

    for col in df.columns:
        unique_values = df[col].dropna().unique()  # Drop NA to get unique values
        num_unique_values = len(unique_values)

        if df[col].dtype == "bool":
            inferred_boolean_columns.append(col)
        elif num_unique_values < unique_threshold or df[col].dtype == "object":
            inferred_categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            inferred_numerical_columns.append(col)
        else:
            # Handle mixed or unexpected data types
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                inferred_numerical_columns.append(col)
            except ValueError:
                inferred_categorical_columns.append(col)

    numerical_columns = [
        col
        for col in inferred_numerical_columns
        if col not in inferred_categorical_columns and col not in inferred_boolean_columns
    ]
    categorical_columns = inferred_categorical_columns
    boolean_columns = inferred_boolean_columns

    # Convert categorical columns to category indices
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col]).codes

    # Clean numerical columns
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Handle missing values - example: fill with the median
        df[col] = df[col].fillna(df[col].median())

    # Convert boolean columns to integers
    for col in boolean_columns:
        df[col] = df[col].astype(int)

    return df


def data_suite_insights(
    tc: ToolCommunicator,
    data_file_path: str,
    target_column: str,
    workspace: str,  # pylint: disable=unused-argument
) -> None:
    """data_suite

    Args:
        tc (ToolCommunicator): The tool communicator object.
        data_file_path (str): The path to the input CSV file.
        workspace (str): The workspace directory path.
    """
    workspace = Path(workspace)  # pyright: ignore
    # Load the data
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)

    # Shuffle the data
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]
    all_features = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    # Use alias test and train
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    train = X_train
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    test = X_test

    # define suspect_features as all features
    suspect_features = list(range(train.shape[1]))

    # define parameters for the copula and representer
    copula_n_samples = 1000
    rep_type = "pca"

    # Step 1: fit and sample a copula on the dataset.
    copula_samples = fit_sample_copula(
        clean_corpus=train,
        copula="gauss",
        copula_n_samples=copula_n_samples,
    )

    # Step 2: REPRESENTER - learns a low dimensional representation of the data. The representation dimension is half, but can be adjusted as a hyperparameter
    rep_dim = int(np.ceil(train.shape[1] / 2))
    pcs_train, pcs_test, pcs_copula = compute_representation(
        train,
        test,
        copula_samples,
        n_components=rep_dim,
        rep_type=rep_type,
    )

    # Step 3: CONFORMAL PREDICTOR - a feature-wise conformal predictor is fit and each reconstruction assessed
    conformal_dict = {}
    for feat in suspect_features:
        feat = int(feat)
        dim = pcs_copula.shape[1]
        conf = conformal_class(conformity_score="sign", input_dim=dim)
        conf.fit(x_train=pcs_copula, y_train=copula_samples[:, feat])
        conformal_dict[feat] = conf.predict(x_test=pcs_test, y_test=test[:, feat])
        tc.print(f"Running analysis for feature = {feat}")

    # Step 4: PROCESS CONFORMAL INTERVALS - we need to process the intervals
    inliers_dict, outliers_dict = inlier_outlier_dicts(conformal_dict, suspect_features)
    # Define the threshold for the proportion of inliers
    proportion = 0.4
    small_ci_ids, large_ci_ids, df_sorted = sort_cis_synth(
        conformal_dict, inliers_dict, suspect_features=[0], proportion=proportion
    )

    if len(large_ci_ids) == 0:
        tc.set_returns(
            tool_return=("The model is performing well on all the data points."),
        )
    else:
        df_large_ci = df.loc[large_ci_ids]

        cluster_model = KMeans(n_clusters=math.ceil(math.sqrt(len(large_ci_ids))))
        cluster_model.fit(df_large_ci)
        df_large_ci["Cluster"] = cluster_model.labels_
        # Calculate mean of each feature for each cluster
        cluster_means = df_large_ci.groupby("Cluster")[all_features].mean()
        cluster_means.reset_index(inplace=False).to_csv(workspace / "Data_suite_examples_to_collect.csv", index=False)  # pyright: ignore

        tc.set_returns(
            tool_return=(
                f"The following features have large conformal intervals and these are the data points that the model may perform poorly on: {large_ci_ids}.\n"
                f"Here are records that approximates the data points that the model may perform poorly on: {cluster_means}.\n"
                f"It is therefore advised that you collect more records that are similar to the example above in order to improve the model's performance.\n"
                f"The exemplar records have also been saved to the workspace directory as 'Data_suite_examples_to_collect.csv'."
            ),
        )


class DataSuiteInsights(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            data_suite_insights,
            wd=self.working_directory,
            data_file_path=real_path,
            target_column=kwargs["target_column"],
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "data_suite_insights"

    @property
    def description(self) -> str:
        return """
        Uses the data_suite_insights tool to gain insights regions of the dataset that the model may perform poorly on.
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
                        "target_column": {"type": "string", "description": "Name of the target column."},
                    },
                    "required": ["data_file_path", "target_column"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return """
Uses the data_suite_insights tool to gain insights regions of the dataset that the model may perform poorly on.
The tool provides exemplar records that the user may want to collect more records similar to in order to improve the model's performance.
"""
