import copy
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from autoprognosis.plugins.explainers import Explainers
from autoprognosis.utils.serialization import load_model_from_file

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


# TODO: abstract this into a shared module
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
        df[col].fillna(df[col].median(), inplace=True)

    # Convert boolean columns to integers
    for col in boolean_columns:
        df[col] = df[col].astype(int)

    return df


# ----------------------------------------------
# INVASE Explainer Tool
# ----------------------------------------------
def autoprognosis_explainer_invase(
    tc: ToolCommunicator,
    model_file_path: str,
    data_file_path: str,
    target_variable: str,
    workspace: str,
    feature_names: Optional[List[str]] = None,
    n_epoch: int = 200,
    n_folds: int = 1,
    task_type: str = "classification",
    time_variable: Optional[str] = None,  # used for risk estimation tasks
) -> None:
    tc.print("Setting up INVASE explainer...")
    model = load_model_from_file(model_file_path)
    tc.print("Loaded autoprognosis model from file")
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    if feature_names is None:
        feature_names = list(X.columns)
    # If risk estimation is selected, compute eval_times automatically and pass time_to_event.
    if task_type == "risk_estimation":
        if time_variable is None:
            raise RuntimeError("time_variable must be provided for risk estimation tasks")
        T = df[time_variable]
        # Compute evaluation horizons using the 25th, 50th, and 75th percentiles of T for events (y == 1)
        eval_times = [
            int(T[y == 1].quantile(0.25)),
            int(T[y == 1].quantile(0.50)),
            int(T[y == 1].quantile(0.75)),
        ]
        tc.print("Prepared data for survival analysis explanation")
        tc.print("Fitting invase explainer")
        exp = Explainers().get(
            "invase",
            copy.deepcopy(model),
            X,
            y,
            feature_names=feature_names,
            n_folds=n_folds,
            n_epoch=n_epoch,
            task_type=task_type,
            eval_times=eval_times,  # automatically computed evaluation times
            time_to_event=df[time_variable],  # pass the time-to-event values
        )
    else:
        # For classification (or other supported tasks)
        tc.print("Fitting invase explainer")
        exp = Explainers().get(
            "invase",
            copy.deepcopy(model),
            X,
            y,
            feature_names=feature_names,
            n_folds=n_folds,
            n_epoch=n_epoch,
            task_type=task_type,
        )
    # Generate the explanation.
    explanation = exp.explain(X)
    # Force conversion to a NumPy array.
    explanation_arr = np.array(explanation)
    # If explanation is 3D (risk estimation), average over the evaluation horizons to get a 2D array.
    if explanation_arr.ndim == 3:
        explanation_2d = np.mean(explanation_arr, axis=2)
    else:
        explanation_2d = explanation_arr
    # Plot using the plugin's plot method (which expects 2D input).
    exp.plot(explanation_2d)
    plot_path = os.path.join(workspace, "invase_explanation_plot.png")
    invase_plot = plt.gcf()
    invase_plot.savefig(plot_path)
    result = {
        "explanation": explanation_2d.tolist(),
        "plot_path": plot_path,
    }
    tc.print("INVASE explanation generated successfully.")
    tc.set_returns(
        tool_return=json.dumps(result, indent=2),
        user_report=[
            f"INVASE explanation complete. Check the saved plot at '{plot_path}'.",
            "INVASE plot: ",
            invase_plot,
        ],
    )


class AutoprognosisExplainerInvase(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_model_path = os.path.join(self.working_directory, kwargs["model_file_path"])
        real_data_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            autoprognosis_explainer_invase,
            model_file_path=real_model_path,
            data_file_path=real_data_path,
            target_variable=kwargs["target_variable"],
            workspace=self.working_directory,
            feature_names=kwargs.get("feature_names"),
            n_epoch=kwargs.get("n_epoch", 200),
            n_folds=kwargs.get("n_folds", 1),
            task_type=kwargs.get("task_type", "classification"),
            time_variable=kwargs.get("time_variable"),  # pass time_variable if provided
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_explainer_invase"

    @property
    def description(self) -> str:
        return (
            "Uses the INVASE algorithm to generate feature importance explanations for a trained AutoPrognosis model. "
            "For classification tasks, it explains class predictions. For risk estimation tasks, it computes evaluation "
            "time horizons (25th, 50th, and 75th percentiles) from the provided time-to-event variable and passes the time-to-event data."
        )

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
                        "model_file_path": {
                            "type": "string",
                            "description": "Path to the trained model file.",
                        },
                        "data_file_path": {
                            "type": "string",
                            "description": "Path to the CSV data file.",
                        },
                        "target_variable": {
                            "type": "string",
                            "description": "Name of the target variable (or event indicator for risk estimation).",
                        },
                        "feature_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of feature names.",
                        },
                        "n_epoch": {
                            "type": "integer",
                            "description": "Number of epochs for INVASE training.",
                            "default": 50,
                        },
                        "n_folds": {
                            "type": "integer",
                            "description": "Number of folds for cross-validation.",
                            "default": 1,
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Task type (e.g., 'classification' or 'risk_estimation').",
                            "default": "classification",
                        },
                        "time_variable": {
                            "type": "string",
                            "description": "Name of the time-to-event variable (required for risk estimation tasks).",
                        },
                    },
                    "required": ["model_file_path", "data_file_path", "target_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Uses the INVASE algorithm to generate feature importance explanations for your AutoPrognosis model. "
            "For risk estimation, the evaluation time horizons (25th, 50th, and 75th percentiles) are computed automatically "
            "from the provided time variable."
        )


# ----------------------------------------------
# Symbolic Pursuit Explainer Tool
# ----------------------------------------------
def autoprognosis_explainer_symbolic_pursuit(
    tc: ToolCommunicator,
    model_file_path: str,
    data_file_path: str,
    target_variable: str,
    workspace: str,
    feature_names: Optional[List[str]] = None,
    n_epoch: int = 10000,
    subsample: int = 10,
    task_type: str = "classification",
    prefit: bool = False,
    time_variable: Optional[str] = None,  # used for risk estimation tasks
) -> None:
    tc.print("Setting up Symbolic Pursuit explainer...")
    model = load_model_from_file(model_file_path)
    tc.print("Loaded autoprognosis model from file")
    df = pd.read_csv(data_file_path)
    df = clean_dataframe(df)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    if feature_names is None:
        feature_names = list(X.columns)
    if task_type == "risk_estimation":
        if time_variable is None:
            raise RuntimeError("time_variable must be provided for risk estimation tasks")
        T = df[time_variable]
        eval_times = [
            int(T[y == 1].quantile(0.25)),
            int(T[y == 1].quantile(0.50)),
            int(T[y == 1].quantile(0.75)),
        ]
        tc.print("Prepared data for survival analysis explanation")
        tc.print("Fitting Explainer")
        exp = Explainers().get(
            "symbolic_pursuit",
            copy.deepcopy(model),
            X,
            y,
            feature_names=feature_names,
            task_type=task_type,
            prefit=prefit,
            n_epoch=n_epoch,
            subsample=subsample,
            eval_times=eval_times,  # computed evaluation horizons
            time_to_event=df[time_variable],  # pass the actual time-to-event values
        )
    else:
        tc.print("Fitting Explainer")
        exp = Explainers().get(
            "symbolic_pursuit",
            copy.deepcopy(model),
            X,
            y,
            feature_names=feature_names,
            task_type=task_type,
            prefit=prefit,
            n_epoch=n_epoch,
            subsample=subsample,
        )
    # Generate the symbolic explanation.
    symbolic_explanation = exp.explain(X)
    symbolic_explanation_list = [[float(val) for val in row] for row in symbolic_explanation]
    # Create a heatmap of the feature importances.
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        np.array(symbolic_explanation_list),
        annot=True,
        cmap="coolwarm",
        fmt=".4f",
        xticklabels=feature_names,
        yticklabels=[f"Instance {i + 1}" for i in range(np.array(symbolic_explanation_list).shape[0])],
    )
    ax.set_title("Symbolic Explanation Heatmap")
    feature_importance = plt.gcf()
    # Also obtain a projection (a symbolic expression string) from the explainer.
    plot_str, projections = exp.plot(X)
    result = {
        "symbolic_explanation": symbolic_explanation_list,
        "symbolic_projection": plot_str,
        "projections": str(projections),
    }
    tc.print("Symbolic Pursuit explanation generated successfully.")
    tc.set_returns(
        tool_return=json.dumps(result, indent=2),
        user_report=[
            "Symbolic Pursuit explanation complete.",
            "**Feature Importance Heatmap**",
            feature_importance,
        ],
    )


class AutoprognosisExplainerSymbolicPursuit(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_model_path = os.path.join(self.working_directory, kwargs["model_file_path"])
        real_data_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            autoprognosis_explainer_symbolic_pursuit,
            model_file_path=real_model_path,
            data_file_path=real_data_path,
            target_variable=kwargs["target_variable"],
            workspace=self.working_directory,
            feature_names=kwargs.get("feature_names"),
            n_epoch=50,
            # n_epoch=kwargs.get("n_epoch", 10000),
            subsample=10,
            # subsample=kwargs.get("subsample", 10),
            task_type=kwargs.get("task_type", "risk_estimation"),
            prefit=kwargs.get("prefit", False),
            time_variable=kwargs.get("time_variable"),  # pass time_variable if provided
            wd=self.working_directory,
        )
        return out_stream

    @property
    def name(self) -> str:
        return "autoprognosis_explainer_symbolic_pursuit"

    @property
    def description(self) -> str:
        return (
            "Uses the Symbolic Pursuit algorithm to generate an interpretable, symbolic explanation of a trained AutoPrognosis model. "
            "For classification and regression tasks, it fits the model (if needed) and computes symbolic feature importances. "
            "For risk estimation tasks, it automatically computes evaluation time horizons from the provided time-to-event variable."
        )

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
                        "model_file_path": {
                            "type": "string",
                            "description": "Path to the trained model file.",
                        },
                        "data_file_path": {
                            "type": "string",
                            "description": "Path to the CSV data file.",
                        },
                        "target_variable": {
                            "type": "string",
                            "description": "Name of the target variable (or event indicator for risk estimation).",
                        },
                        "feature_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of feature names.",
                        },
                        # "n_epoch": {
                        #     "type": "integer",
                        #     "description": "Number of epochs for Symbolic Pursuit.",
                        #     "default": 10000,
                        # },
                        # "subsample": {
                        #     "type": "integer",
                        #     "description": "Subsample size for Symbolic Pursuit.",
                        #     "default": 10,
                        # },
                        "task_type": {
                            "type": "string",
                            "description": "Task type (e.g., 'classification', 'regression', or 'risk_estimation') risk_estimation should be used for survival_analysis problems.",
                            "default": "risk_estimation",
                        },
                        "prefit": {
                            "type": "boolean",
                            "description": "If true, the estimator is assumed prefit.",
                            "default": False,
                        },
                        "time_variable": {
                            "type": "string",
                            "description": "Name of the time-to-event variable (required for risk estimation tasks).",
                        },
                    },
                    "required": ["model_file_path", "data_file_path", "target_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "Uses the Symbolic Pursuit algorithm to generate an interpretable, symbolic explanation for your AutoPrognosis model. "
            "For risk estimation tasks, the evaluation time horizons (25th, 50th, and 75th percentiles) are computed automatically "
            "from the provided time variable."
        )
