import os
from typing import Any, Dict, List, Optional

import pandas as pd
from hyperimpute.plugins.imputers import Imputers
from hyperimpute.utils.serialization import save

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


def update_dataframe(df_orig: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    # Get the list of columns in df_new that exist in df_orig.
    common_columns = [col for col in df_new.columns if col in df_orig.columns]

    # Update df_orig with values from df_new for the common columns.
    df_out = df_orig.copy()
    df_out.update(df_new[common_columns])

    return df_out


class BasicHook:
    def __init__(self, iters: int) -> None:
        self.iters = iters
        self.count = 0

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.count += 1
        print(f"HyperImpute: Iterations completed {self.count}/{self.iters}")


def hyperimpute_impute(
    tc: ToolCommunicator,
    data_file_path: str,
    imputed_file_path: str,
    workspace: str,  # pylint: disable=unused-argument
    subset: Optional[List[str]] = None,
) -> None:
    df = pd.read_csv(data_file_path)
    df_use = df[subset] if subset else df
    if subset:
        tc.print("Imputing subset of columns:")
        for col in subset:
            tc.print(f"* {col}")
    else:
        tc.print("Imputing all columns")

    tc.print("Setting up HyperImpute Imputer...")
    n_inner_iter = 3
    imputer = Imputers().get(
        "hyperimpute",
        n_inner_iter=n_inner_iter,
        classifier_seed=[
            "random_forest",
            "logistic_regression",
            "catboost",
            # "xgboost",
            # NOTE: xgboost fails for e.g. CF dataset - fails with error related to lack of values of certain classes.
            # TODO: requires proper investigation and fixing.
        ],
        inner_loop_hook=BasicHook(n_inner_iter),
    )

    tc.print("Imputing data...")
    tc.print("Missing values before imputation:")
    tc.print(df_use.isnull().sum())
    n_missing_before = df_use.isnull().sum().sum()

    out = imputer.fit_transform(df_use.copy())
    if subset:
        out = update_dataframe(df, out)

    # Save the imputation model:
    imputed_file_basename = os.path.splitext(os.path.basename(imputed_file_path))[0]
    model_file_path = os.path.join(workspace, f"hyperimpute__{imputed_file_basename}.pkl")
    tc.print("Saving imputation model to:", model_file_path)
    bytes = save(imputer)
    with open(model_file_path, "wb") as f:
        f.write(bytes)

    tc.print("Missing values after imputation:")
    tc.print(out.isnull().sum())
    n_missing_after = out.isnull().sum().sum()

    tc.print("Saving imputed data...")
    out.to_csv(imputed_file_path, index=False)

    tc.set_returns(
        tool_return=(
            f"{n_missing_before - n_missing_after} missing values were imputed. "
            f"The imputed data has been saved to {imputed_file_path}"
        ),
        files_in=[os.path.basename(data_file_path)],
        files_out=[os.path.basename(imputed_file_path)],
    )


class HyperImputeImputation(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        real_imputed_path = os.path.join(self.working_directory, kwargs["imputed_file_path"])
        subset = kwargs.get("subset", None)
        thrd, out_stream = execute_tool(
            hyperimpute_impute,
            wd=self.working_directory,
            data_file_path=real_path,
            imputed_file_path=real_imputed_path,
            workspace=self.working_directory,
            subset=subset,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "hyperimpute_imputation"

    @property
    def description(self) -> str:
        return """
        Uses the **HyperImpute** library to automatically impute missing values in your data.
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
                        "imputed_file_path": {
                            "type": "string",
                            "description": "Path to the imputed data file.",
                        },
                        "subset": {
                            "type": "array",
                            "description": "Optional subset of columns to impute. If not provided, all columns will be imputed.",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["data_file_path", "imputed_file_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "uses the **HyperImpute** library to automatically impute missing values in your data."
