# Imports
import multiprocessing
import numbers
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import shap
from autoprognosis.utils.serialization import load_model_from_file
from autoprognosis.utils.tester import evaluate_survival_estimator
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn.ensemble._bagging import _generate_indices  # pyright: ignore
from sklearn.inspection._permutation_importance import _create_importances_bunch, _weights_scorer  # pyright: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection._validation import _aggregate_score_dicts  # pyright: ignore
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import _safe_indexing, check_array, check_random_state  # pyright: ignore

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    # Initialize a OneHotEncoder
    try:
        # Scikit-learn <1.4 uses `sparse` parameter.
        encoder = OneHotEncoder(sparse=False, drop="if_binary")
    except TypeError as e:
        if "sparse" in str(e):
            # Scikit-learn >=1.4 uses `sparse_output` parameter.
            encoder = OneHotEncoder(sparse_output=False, drop="if_binary")
        else:
            raise

    # Process each categorical column
    for col in categorical_cols:
        if df[col].nunique() < 5:
            # Perform one-hot encoding
            encoded_df = pd.DataFrame(encoder.fit_transform(df[[col]]))
            encoded_df.columns = encoder.get_feature_names_out([col])

            # Drop the original column from df
            df = df.drop(col, axis=1)

            # Join the encoded DataFrame with the original DataFrame
            df = pd.concat([df, encoded_df], axis=1)
        else:
            # Drop the column if it has 5 or more categories
            df = df.drop(col, axis=1)

    return df


def shap_explainer(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    problem_type: str,
    workspace: str,
) -> None:
    tc.print("Loading the data...")
    df = pd.read_csv(data_file_path)
    df = process_dataframe(df)
    X, y = df[[c for c in df.columns if c != target_variable]], df[target_variable]

    tc.print("Setting up the SHAP explainer...")

    tc.print("Running the SHAP explainer, this can take a while...")
    if problem_type == "classification":
        model = LogisticRegression()
    else:
        model = LinearRegression()
    model.fit(X, y)
    explainer = shap.Explainer(model.predict, X, max_evals=1000)
    shap_values = explainer(X)

    # Get numerical values for the mean absolute SHAP values per feature.
    mean_abs_shap_values = np.abs(shap_values.values).mean(axis=0)  # pylint: disable=no-member
    shap_df = pd.DataFrame(mean_abs_shap_values, index=X.columns, columns=["Mean Abs SHAP Value"])
    shap_df.sort_values(by="Mean Abs SHAP Value", ascending=False, inplace=True)
    shap_info_str = shap_df.to_json(indent=2)
    tc.print(f"SHAP values:\n{shap_info_str}")

    plt.figure()
    shap.plots.bar(shap_values, show=False)
    shap_bar = plt.gcf()
    # shap_bar_out = copy.deepcopy(shap_bar)
    shap_bar.savefig(os.path.join(workspace, "shap_bar.png"), bbox_inches="tight")
    tc.print("SHAP bar plot saved to `shap_bar.png`")

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    shap_beeswarm = plt.gcf()
    # shap_beeswarm_out = copy.deepcopy(shap_beeswarm)
    shap_beeswarm.savefig(os.path.join(workspace, "shap_beeswarm.png"), bbox_inches="tight")
    tc.print("SHAP beeswarm plot saved to `shap_beeswarm.png`")

    tc.print("SHAP explainer completed!")
    tc.set_returns(
        tool_return=(
            f"SHAP explainer completed. Mean absolute SHAP values are:\n{shap_info_str}. "
            "The user can see the SHAP bar plot and beeswarm plot in the UI."
        ),
        user_report=[
            "📊 **SHAP Explainer**",
            "SHAP bar plot:",
            shap_bar,
            "SHAP beeswarm plot:",
            shap_beeswarm,
        ],
    )


# TODO: THIS IS A DUMMY EXPLAINER - THIS NEEDS TO BE MADE WORK WITH THE AUTOPROGNOSIS MODEL!
# TODO: Also give the model the shap values / summary of the important features.
class ShapExplainer(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target_variable = kwargs["target_variable"]
        problem_type = kwargs["problem_type"]
        thrd, out_stream = execute_tool(
            shap_explainer,  # TODOpermutation
            wd=self.working_directory,
            data_file_path=real_path,
            target_variable=target_variable,
            problem_type=problem_type,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "shap_explainer"

    @property
    def description(self) -> str:
        return """
        Performs SHAP analysis on the user's data and the model from the library AutoPrognosis 2.0.
        
        Must ONLY be called AFTER `autoprognosis_regression` or `autoprognosis_classification` has been called. 
        This tool DOES NOT work for survival analysis tasks and should NEVER be used for survival tasks.
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
                        "target_variable": {"type": "string", "description": "Name of the target variable."},
                        "model_path": {"type": "string", "description": "Path to the model file."},
                        "problem_type": {
                            "type": "string",
                            "enum": ["classification", "regression"],
                            "description": "Type of problem (classification or regression).",
                        },
                    },
                    "required": ["data_file_path", "target_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "performs SHAP feature importance analysis with the prediction model and your data"


def permutation_importance_explainer(
    tc: ToolCommunicator,
    data_file_path: str,
    workspace: str,
    model_path: Any,
    target_variable: str,
    time_variable: str,
) -> None:
    # Ensure you have this clean_dataframe function
    def clean_dataframe(df, unique_threshold=15):
        categorical_columns = [
            col for col in df.columns if df[col].nunique() < unique_threshold or df[col].dtype == "object"
        ]
        numerical_columns = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
        boolean_columns = [col for col in df.columns if df[col].dtype == "bool"]
        categorical_columns = [
            col for col in categorical_columns if col not in numerical_columns and col not in boolean_columns
        ]

        for col in categorical_columns:
            df[col] = df[col].astype("category").cat.codes

        for col in numerical_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median(), inplace=True)

        for col in boolean_columns:
            df[col] = df[col].astype(int)

        return df

    # Define the scorer function correctly
    def survival_scorer(estimator, X, y):
        T = y["time"]
        E = y["event"]

        if not isinstance(T, pd.Series):
            T = pd.Series(T)
        if not isinstance(E, pd.Series):
            E = pd.Series(E)

        eval_time_horizons = [
            int(T[E == 1].quantile(0.25)),
            int(T[E == 1].quantile(0.50)),
            int(T[E == 1].quantile(0.75)),
        ]
        n_folds = 3
        metrics = evaluate_survival_estimator(
            [estimator] * n_folds,
            X,
            T,
            E,
            eval_time_horizons,  # type: ignore
            n_folds=n_folds,
            pretrained=True,
        )
        score = metrics["raw"]["c_index"][0] - metrics["raw"]["brier_score"][0]
        return score

    def _calculate_permutation_scores(
        estimator,
        X,
        y,
        sample_weight,
        col_idx,
        random_state,
        n_repeats,
        scorer,
        max_samples,
        working_dir,  # Added parameter
    ):
        """Calculate score when `col_idx` is permuted."""
        original_dir = os.getcwd()  # Save the original working directory
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)  # Change to the process-specific working directory

        try:
            random_state = check_random_state(random_state)

            # Work on a copy of X to ensure thread-safety in case of threading based
            # parallelism. Furthermore, making a copy is also useful when the joblib
            # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
            # if X is large it will be automatically be backed by a readonly memory map
            # (memmap). X.copy() on the other hand is always guaranteed to return a
            # writable data-structure whose columns can be shuffled inplace.
            if max_samples < X.shape[0]:
                row_indices = _generate_indices(
                    random_state=random_state,
                    bootstrap=False,
                    n_population=X.shape[0],
                    n_samples=max_samples,
                )
                X_permuted = _safe_indexing(X, row_indices, axis=0)
                y = _safe_indexing(y, row_indices, axis=0)
            else:
                X_permuted = X.copy()

            scores = []
            shuffling_idx = np.arange(X_permuted.shape[0])
            for _ in range(n_repeats):
                random_state.shuffle(shuffling_idx)
                if hasattr(X_permuted, "iloc"):
                    col = X_permuted.iloc[shuffling_idx, col_idx]
                    col.index = X_permuted.index
                    X_permuted[X_permuted.columns[col_idx]] = col
                else:
                    X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
                scores.append(_weights_scorer(scorer, estimator, X_permuted, y, sample_weight))

            if isinstance(scores[0], dict):
                scores = _aggregate_score_dicts(scores)
            else:
                scores = np.array(scores)

            return scores
        finally:
            os.chdir(original_dir)  # Restore the original working directory

    def permutation_importance(
        estimator,
        X,
        y,
        *,
        scoring=None,
        workspace=Path().cwd(),  # Added parameter for the base directory
        n_repeats=5,
        n_jobs=None,
        random_state=None,
        sample_weight=None,
        max_samples=1.0,
    ):
        if not hasattr(X, "iloc"):
            X = check_array(X, force_all_finite="allow-nan", dtype=None)

        random_state = check_random_state(random_state)
        random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

        if not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])
        elif max_samples > X.shape[0]:
            raise ValueError("max_samples must be <= n_samples")

        baseline_score = _weights_scorer(scoring, estimator, X, y, sample_weight)

        # Set up directories
        base_dir = workspace / "model_checkpoints"
        directories = [base_dir / f"dir_{i}" for i in range(multiprocessing.cpu_count())]
        for dir in directories:
            os.makedirs(dir, exist_ok=True)

        scores = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(_calculate_permutation_scores)(
                estimator,
                X,
                y,
                sample_weight,
                col_idx,
                random_seed,
                n_repeats,
                scoring,
                max_samples,
                directories[col_idx % len(directories)],  # Assign each job a directory in a round-robin fashion
            )
            for col_idx in range(X.shape[1])
        )

        if isinstance(baseline_score, dict):
            return {
                name: _create_importances_bunch(
                    baseline_score[name],
                    np.array([scores[col_idx][name] for col_idx in range(X.shape[1])]),  # pyright: ignore
                )
                for name in baseline_score
            }
        else:
            return _create_importances_bunch(baseline_score, np.array(scores))

    workspace, model_path = Path(workspace), Path(model_path)  # type: ignore
    if workspace not in model_path.parents:
        model_path = workspace / model_path
    tc.print("Loading the data...")
    df = pd.read_csv(data_file_path)
    tc.print(f"Data loaded with shape: {df.shape[0]} rows and {df.shape[1]} columns.")
    threshold = int(100000 / df.shape[1])
    if len(df) > threshold:
        df = df.sample(threshold)

    df = clean_dataframe(df)

    X = df.drop([time_variable, target_variable], axis=1)
    Y = df[target_variable]
    T = df[time_variable]

    # TODO: make robust - Create a structured array for y
    y = np.array([(e, t) for e, t in zip(Y, T)], dtype=[("event", int), ("time", float)])

    tc.print(f"Loading model from file: {model_path}")
    try:
        model = load_model_from_file(model_path)
    except Exception as e:
        raise TypeError(
            "Model file is not a valid AutoPrognosis 2.0 file. This tool only supports AutoPrognosis 2.0 models."
        ) from e

    # TODO: Support sklearn models.
    # Use permutation_importance to find feature importances
    tc.print("Running the permutation explainer, this can take a while...")
    tc.print("""
        This tool takes approximately 1 minute per column. If this is too slow, consider reducing the number of columns.
        Reducing the number of column can be done with the feature selection tool. You can cancel this tool with the 
        `Restart from last reasoning step` button. Then use the feature selection tool to reduce the number of columns.""")

    result = permutation_importance(
        model,
        X,
        y,
        scoring=survival_scorer,
        workspace=workspace,  # pyright: ignore
        n_repeats=3,
        random_state=42,
        n_jobs=-1,
    )

    # Create a DataFrame to display the importances
    importance_df = pd.DataFrame(
        {
            "importances_mean": result.importances_mean,  # pyright: ignore
            "importances_std": result.importances_std,  # pyright: ignore
        },
        index=X.columns,
    ).sort_values(by="importances_mean", ascending=False)

    tc.print("Permutation explainer completed!")
    tc.print(importance_df)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df.index,
        importance_df["importances_mean"],
        xerr=importance_df["importances_std"],
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=5,
    )
    plt.xlabel("Importance Mean")
    plt.title("Feature Importances with Error Bars")
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest importance at the top
    perm_importance_plot = plt.gcf()
    perm_importance_plot.savefig(os.path.join(workspace, "permutation_importance.png"), bbox_inches="tight")
    tc.print("Permutation bar plot saved to `permutation_importance.png`")

    tc.set_returns(
        tool_return=(
            f"permutation explainer completed. Mean absolute permutation values are:\n{importance_df['importances_mean']}. "
            "The user can see the permutation bar plot in the UI."
        ),
        user_report=[
            "📊 **permutation Explainer**",
            "permutation bar plot:",
            perm_importance_plot,
        ],
    )


class PermutationExplainer(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            permutation_importance_explainer,
            wd=self.working_directory,
            data_file_path=real_path,
            model_path=kwargs["model_path"],
            target_variable=kwargs["target_variable"],
            time_variable=kwargs["time_variable"],
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "permutation_explainer"

    @property
    def description(self) -> str:
        return """
        Performs permutation analysis on the user's data and the model from the library AutoPrognosis 2.0. It is used for feature importance analysis on survival models.
        
        Must ONLY be called AFTER `autoprognosis_survival` has been called.
        This tool should not be used for classification or regression tasks.
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
                        "data_file_path": {
                            "type": "string",
                            "description": "Path to the data file.",
                        },
                        "model_path": {"type": "string", "description": "Path to the model file."},
                        "target_variable": {
                            "type": "string",
                            "description": "Name of the target variable.",
                        },
                        "time_variable": {
                            "type": "string",
                            "description": "The time to event column. This is only applicable for survival analysis tasks, where it is mandatory.",
                        },
                    },
                    "required": ["data_file_path", "model_path", "target_variable", "time_variable"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "performs permutation feature importance analysis with the prediction model and your data"
