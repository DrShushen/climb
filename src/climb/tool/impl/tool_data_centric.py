import os
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
from autoprognosis.utils.serialization import load_model_from_file
from data_iq import DataIQ_SKLearn

from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase


class APWrapperForDataIQ:
    def __init__(self, model: Any) -> None:
        self.model = model

    def predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> Any:  # pylint: disable=unused-argument
        # Swallow any unknown kwargs.
        return self.model.predict_proba(X).to_numpy()  # Also return as numpy array, not DF.


def dataiq_insights(
    tc: ToolCommunicator,
    data_file_path: str,
    target_variable: str,
    model_path: str,
    workspace: str,
) -> None:
    df = pd.read_csv(data_file_path)
    X = df[[c for c in df.columns if c != target_variable]]
    y = df[target_variable].to_numpy()

    tc.print("Loading the model...")
    try:
        model = load_model_from_file(model_path)
    except Exception as e:
        raise TypeError(
            "Model file is not a valid AutoPrognosis 2.0 file. This tool only supports AutoPrognosis 2.0 models."
        ) from e
    # TODO: Support sklearn models.
    clf = APWrapperForDataIQ(model)

    tc.print("Running DataIQ...")
    dataiq = DataIQ_SKLearn(X=X, y=y)
    nest = 10
    for i in range(1, nest):
        dataiq.on_epoch_end(clf=clf, iteration=i)
    aleatoric_uncertainty = dataiq.aleatoric
    confidence = dataiq.confidence

    # Determine easy/hard/ambiguous samples.
    tc.print("Determining easy/hard/ambiguous samples...")
    # NOTE: The thresholds here are heuristics.
    percentile_thresh = 90  # Originally 50
    thresh = max(
        0.25,
        (np.max(confidence) - np.min(confidence)) * 0.25 + np.min(confidence),
    )  # Originally just 0.25
    conf_thresh_low = thresh
    conf_thresh_high = 1 - thresh

    hard_train = np.where(
        (confidence <= conf_thresh_low)
        & (aleatoric_uncertainty <= np.percentile(aleatoric_uncertainty, percentile_thresh))
    )[0]
    easy_train = np.where(
        (confidence >= conf_thresh_high)
        & (aleatoric_uncertainty <= np.percentile(aleatoric_uncertainty, percentile_thresh))
    )[0]

    hard_easy = np.concatenate((hard_train, easy_train))
    ambig_train_ = []
    for id_ in range(len(confidence)):
        if id_ not in hard_easy:
            ambig_train_.append(id_)

    ambig_train = np.array(ambig_train_)

    # Save the results.
    tc.print("Saving the results...")
    results = {
        "aleatoric_uncertainty": aleatoric_uncertainty,
        "confidence": confidence,
        "easy_samples": easy_train,
        "hard_samples": hard_train,
        "ambiguous_samples": ambig_train,
        "df": df,
        "target_variable": target_variable,
    }
    results_path = os.path.join(workspace, "dataiq_results.p")
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    # Load in all the data.
    tc.print("Preparing the plot...")
    easy_samples = results["easy_samples"]
    hard_samples = results["hard_samples"]
    ambiguous_samples = results["ambiguous_samples"]
    aleatoric_uncertainty = results["aleatoric_uncertainty"]
    confidence = results["confidence"]
    df = results["df"]
    target_variable = results["target_variable"]
    features = [c for c in df.columns if c != target_variable]

    # Count the different groups and put into text.
    easy_count = len(easy_samples)
    hard_count = len(hard_samples)
    ambiguous_count = len(ambiguous_samples)
    counts = f"Easy: {easy_count} samples\nHard: {hard_count} samples\nAmbiguous: {ambiguous_count} samples"

    # Put into a dataframe for plotly.
    df["aleatoric_uncertainty"] = aleatoric_uncertainty
    df["confidence"] = confidence
    df["data_iq_group"] = None
    df.loc[easy_samples, "data_iq_group"] = "Easy"
    df.loc[ambiguous_samples, "data_iq_group"] = "Ambiguous"
    df.loc[hard_samples, "data_iq_group"] = "Hard"
    df["Row Index"] = np.arange(len(df))

    hover_show = features + ["Row Index", target_variable]

    fig = px.scatter(
        df,
        x="aleatoric_uncertainty",
        y="confidence",
        color="data_iq_group",
        color_discrete_sequence=["#27ae60", "#f8c471", "#e74c3c"],
        hover_data=hover_show,
        title="Data-IQ Insights",
        labels={
            "aleatoric_uncertainty": "Aleatoric Uncertainty",
            "confidence": "Confidence",
            "data_iq_group": "Data-IQ Group",
        },
        width=1000,
        height=800,
    )

    # The below text is left-aligned to avoid strange formatting in the UI.

    # Additional explanation.
    explanation_of_axes = """
In DataIQ the two axes are:
- **Confidence** represents the model's confidence in the predictions
- **Aleatoric uncertainty** is the inherent uncertainty or ambiguity in a sample. That is, even if we add more \
samples, we'd still be confused. For example, in a tabular dataset, two patients with the same features but \
different labels.
"""
    explanation_of_groups = """
In DataIQ, the samples are categorized into three groups:
- **Easy**: Samples which we are confident about and are clear-cut (low data uncertainty), hence easy to learn.
- **Ambiguous**: Samples for which there is just uncertainty in the data itself. For instance, for tabular data \
in the medical setting, this could be: "the only way we'll get better predictions is if we get more information, \
e.g. patients for whom we need to run more tests).
- **Hard**: Possibly mislabeled data. Since these are really clear-cut samples (low data uncertainty), but the \
model just can't learn them, i.e. samples which are unlearnable in their current state.
"""

    tc.set_returns(
        tool_return=f"""
Results saved to: `{results_path}`.
This is a pickle file containing a dictionary with keys:
{{
"easy_samples": numpy array with indices of easy samples,
"hard_samples": numpy array with indices of hard samples,
"ambiguous_samples": numpy array with indices of ambiguous samples,
}}

Sample counts summary:
{counts}

{explanation_of_axes}

{explanation_of_groups}

The user can now explore different groups of samples on an interactive chart.
""",
        user_report=[
            fig,
            f"""
**Additional explanation:**

{explanation_of_axes}

{explanation_of_groups}
""",
        ],
    )


class DataIQInsights(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_data_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        real_model_path = os.path.join(self.working_directory, kwargs["model_path"])
        target_variable = kwargs["target_variable"]
        thrd, out_stream = execute_tool(
            dataiq_insights,
            wd=self.working_directory,
            data_file_path=real_data_path,
            target_variable=target_variable,
            model_path=real_model_path,
            workspace=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "dataiq_insights"

    @property
    def description(self) -> str:
        return """
        DataIQ Insights for a classification task.

        Given a dataset and a classifier model, this tool provides data-centric insights for a classification task.
        In particular it categorizes the samples into "easy", "hard" and "ambiguous" for classification. 
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
                    },
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return (
            "provides insights for your classification task - which samples were 'easy', 'hard' or 'ambiguous' "
            "for classification."
        )


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
