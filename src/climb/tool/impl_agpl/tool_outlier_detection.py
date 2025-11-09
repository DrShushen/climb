# Imports
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from cleanlab.outlier import OutOfDistribution  # noqa: F401  # type: ignore
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from climb.common.utils import raise_if_extra_not_available

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase

raise_if_extra_not_available()


def clean_dataframe(df, unique_threshold=15):
    """
    Cleans the dataframe by encoding categorical variables, handling missing values, and converting data types.

    Parameters:
    - df (pd.DataFrame): The input dataframe to clean.
    - unique_threshold (int): Threshold to decide if a numerical column should be treated as categorical.

    Returns:
    - pd.DataFrame: The cleaned dataframe.
    """
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
            df[col] = pd.to_numeric(df[col], errors="coerce")
            inferred_numerical_columns.append(col)

    numerical_columns = [
        col
        for col in inferred_numerical_columns
        if col not in inferred_categorical_columns and col not in inferred_boolean_columns
    ]
    categorical_columns = inferred_categorical_columns
    boolean_columns = inferred_boolean_columns

    # Convert categorical columns to category indices, handling NaNs
    for col in categorical_columns:
        df[col] = pd.Categorical(df[col].fillna("Missing")).codes

    # Clean numerical columns
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Handle missing values - example: fill with the median
        df[col] = df[col].fillna(df[col].median())

    # Convert boolean columns to integers
    for col in boolean_columns:
        df[col] = df[col].astype(int)

    return df


def cleanlab_outlier_detection(
    tc: ToolCommunicator,
    data_file_path: str,
    cleaned_file_path: str,
    target_variable: str,
    workspace: str,
    time_variable: Optional[str] = None,
    task_type: str = "classification",
) -> None:
    if task_type not in ["classification", "survival_analysis"]:
        raise ValueError(
            f"`task_type` must be 'classification' or 'survival_analysis'. Cleanlab does not support {task_type}."
        )

    # Get the data and target variable
    tc.print("Loading the data...")
    workspace = Path(workspace)  # pyright: ignore
    data_file_path = workspace / data_file_path  # pyright: ignore
    cleaned_file_path = workspace / cleaned_file_path  # pyright: ignore
    df = pd.read_csv(data_file_path)

    # Save the original data for later
    df_original = df.copy()
    original_target_variable = target_variable
    X_original = df.drop(columns=[original_target_variable]).values
    y_original = df[original_target_variable].values

    # Convert to classification using time horizon for the sake of the tool
    if task_type == "survival_analysis":
        if time_variable is None:
            raise ValueError("For survival analysis tasks, `time_column` must be provided.")
        # convert to classification using time horizon
        time_horizon = df[time_variable].median()
        df["TEMP_CLASSIFICATION_FROM_SURVIVAL_EVENT_COL"] = np.where(
            (df[time_variable] < time_horizon) & (df[target_variable] == 1), 1, 0
        )
        df.drop([time_variable, target_variable], axis=1, inplace=True)
        # change target column to the new classification column
        target_variable = "TEMP_CLASSIFICATION_FROM_SURVIVAL_EVENT_COL"

    # Process the df to clean it for outlier detection
    df = clean_dataframe(df)

    # Verify that the cleaned dataframe aligns with the original
    assert len(df) == len(df_original), "Row count mismatch after cleaning."
    assert all(df.index == df_original.index), "Row order mismatch after cleaning."

    # Split the data into features and target
    X = df.drop(columns=[target_variable]).values  # noqa: F841
    y = df[target_variable].values  # noqa: F841

    # NOTE: we need to use cross-validation to get the OOD scores for all data points, as this method relies on
    # test and train data.

    # Initialize Cross-Validation
    # We'll use 5-Fold Cross-Validation to create pseudo-train/test splits.
    # Each data point will be evaluated exactly once as part of the test set.
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    ood_scores = np.zeros(len(X))  # the OOD scores for each data point across folds.
    count = np.zeros(len(X))  # number of times each data point has been evaluated.

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        tc.print(f"Processing Fold {fold + 1}/{K}")

        # Split the data into training and testing sets for the current fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Initialize and train the model on the training set
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)
        ood = OutOfDistribution()
        ood_fold_scores = ood.fit_score(pred_probs=y_probs, labels=y_test)

        # Accumulate OOD scores and update counts for the test indices
        ood_scores[test_index] += ood_fold_scores
        count[test_index] += 1

    # Ensure that each data point was evaluated at least once to prevent division by zero
    count[count == 0] = 1  # This should not occur in standard K-Fold CV

    # Compute the average OOD score for each data point
    average_ood_scores = ood_scores / count

    # Define thresholds based on the 95th and 5th percentiles
    low_threshold = np.percentile(average_ood_scores, 5)

    tc.print(f"=Threshold (5th percentile): {low_threshold:.4f}")

    # Identify Outliers Based on Thresholds
    # Flag data points with OOD scores below the low threshold
    low_outliers = average_ood_scores < low_threshold

    # Retrieve indices of outliers
    low_outlier_indices = np.where(low_outliers)[0]

    tc.print(f"Number of Outliers (<{low_threshold:.4f}): {len(low_outlier_indices)}")

    # Visualize the Distribution of OOD Scores

    plt.figure()
    plt.hist(average_ood_scores, bins=50, alpha=0.7, color="skyblue", edgecolor="black")
    plt.axvline(
        low_threshold,  # pyright: ignore
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"5th Percentile ({low_threshold:.4f})",
    )
    plt.title("Distribution of Averaged OOD Scores with Low Threshold")
    plt.xlabel("OOD Score")
    plt.ylabel("Frequency")
    plt.legend()
    outlier_plot = plt.gcf()
    outlier_plot.savefig(os.path.join(workspace, "outlier_detection.png"), bbox_inches="tight")
    plt.close()

    # Remove Outliers from the Dataset
    # Create masks for outliers and inliers
    mask_outliers = low_outliers
    mask_inliers = ~low_outliers

    # Extract inlier and outlier data from original dataset
    cleaned_X = X_original[mask_inliers]
    cleaned_y = y_original[mask_inliers]
    outliers_X = X_original[mask_outliers]  # noqa: F841
    outliers_y = y_original[mask_outliers]

    # Save the Cleaned Data to CSV
    feature_columns = df_original.columns[df_original.columns != original_target_variable]
    df_cleaned = pd.DataFrame(cleaned_X, columns=feature_columns)
    df_cleaned[original_target_variable] = cleaned_y

    tc.print("Saving cleaned data...")
    df_cleaned.to_csv(cleaned_file_path, index=False)
    tc.set_returns(
        tool_return=(
            f"{len(outliers_y)} outliers were removed. The cleaned data has been saved to {cleaned_file_path}"
        ),
        user_report=[
            "📊 **Outlier Detection**",
            "Outliers plot:",
            outlier_plot,
        ],
        files_in=[os.path.basename(data_file_path)],
        files_out=[os.path.basename(cleaned_file_path)],
    )


class CleanlabOutlierDetection(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        data_file_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        cleaned_file_path = os.path.join(self.working_directory, kwargs["cleaned_file_path"])
        thrd, out_stream = execute_tool(
            cleanlab_outlier_detection,
            wd=self.working_directory,
            data_file_path=data_file_path,
            cleaned_file_path=cleaned_file_path,
            target_variable=kwargs["target_variable"],
            time_variable=kwargs.get("time_variable"),
            task_type=kwargs["task_type"],
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "outlier_detection"

    @property
    def description(self) -> str:
        return """
Identifies and removes outliers from a dataset, enhancing the quality and reliability of the data for subsequent analytical or modeling tasks.
By leveraging Cleanlab's out_of_distribution function, this function systematically detects anomalous data points through a K-Fold Cross-Validation
approach. The cleaned dataset is then saved for further use, ensuring that outliers do not skew the results of downstream processes. This function is
only compatible with classification and survival analysis problems.
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
                        "cleaned_file_path": {
                            "type": "string",
                            "description": "Path to the cleaned data file that we create by calling this function.",
                        },
                        "target_variable": {"type": "string", "description": "Name of the target variable."},
                        "time_variable": {
                            "type": "string",
                            "description": "The time to event column. This is only applicable for survival analysis tasks, where it is mandatory.",
                        },
                        "task_type": {
                            "type": "string",
                            "enum": ["classification", "survival_analysis"],
                            "description": "Type of problem (classification or survival_analysis).",
                        },
                    },
                    "required": ["data_file_path", "cleaned_file_path", "target_variable", "task_type"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "Identifies and removes outliers from a dataset"
