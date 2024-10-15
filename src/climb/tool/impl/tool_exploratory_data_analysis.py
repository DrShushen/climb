import os
import time
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .._utils import id_numerics_actually_categoricals
from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase, get_str_up_to_marker


def generate_correlogram(
    df: pd.DataFrame,
    numerics_that_are_categoricals: List[str],
    workspace: str,
    show_n_corr: int = 10,  # Number of most correlated pairs to show.
    target: Optional[str] = None,
) -> Tuple[str, matplotlib.figure.Figure]:
    # Convert likely categorical numerics to categorical type:
    for col in numerics_that_are_categoricals:
        df[col] = pd.Categorical(df[col]).codes

    # Select only numerical columns
    num_df = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr_matrix = num_df.corr()

    # Flatten the matrix to vector and sort by absolute value
    corr_unstacked = corr_matrix.abs().unstack()
    sorted_corr = corr_unstacked.sort_values(kind="quicksort", ascending=False)  # type: ignore

    # Include target feature in the top correlations if specified
    if target and target in num_df.columns:
        target_corr = corr_unstacked[target].drop(target).abs().sort_values(ascending=False).head(show_n_corr)
        sorted_corr = pd.concat([sorted_corr, target_corr]).drop_duplicates()

    # Select the top most correlated pairs
    top_correlations = sorted_corr.drop_duplicates().iloc[1 : show_n_corr + 1]  # skip the first (self-correlation)

    # Find the features involved in these top correlations
    features = set([item for sublist in top_correlations.index for item in sublist])

    # Ensure target is included in features set if specified
    if target and target in num_df.columns:
        features.add(target)

    # Filter the correlation matrix to include only these features
    filtered_corr_matrix = corr_matrix.filter(items=features).reindex(features)  # type: ignore

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(filtered_corr_matrix, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(11, 9))

    # Draw the heatmap with the mask and the smaller set of features
    sns.heatmap(
        filtered_corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=1.0,
        vmin=-1.0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
    )

    # Add a title
    plt.title("Correlation Matrix of Top Correlated Features")

    # Save the figure
    fig = plt.gcf()
    plt.savefig(os.path.join(workspace, "correlogram.png"))
    plt.close()

    # Create a DataFrame from the upper triangle of the correlation matrix, without the diagonal.
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Stack the DataFrame and reset index.
    stacked_corr = upper_tri.stack().reset_index()
    stacked_corr.columns = ["Feature 1", "Feature 2", "Correlation"]

    # Sort by absolute values of correlation, descending.
    sorted_corr = stacked_corr.reindex(stacked_corr.Correlation.abs().sort_values(ascending=False).index)

    # Extract the top most positively correlated feature pairs.
    most_positive_corr = sorted_corr[sorted_corr["Correlation"] > 0].head(show_n_corr).reset_index(drop=True)

    # Extract the top most negatively correlated feature pairs.
    most_negative_corr = sorted_corr[sorted_corr["Correlation"] < 0].head(show_n_corr).reset_index(drop=True)

    # Combine and return the most correlated pairs.
    as_text = f"""
Most Positively Correlated Features:
{most_positive_corr}

Most Negatively Correlated Features:
{most_negative_corr}
"""

    return as_text, fig


def exploratory_data_analysis(
    tc: ToolCommunicator,
    data_file_path: str,
    target: Optional[str],
    workspace: str,
) -> None:
    """Perform exploratory data analysis (EDA) on a CSV file, outputting a detailed textual summary.

    Key features:
    1. Dataset Overview:
        Reports the dataset's dimensions and column data types.
    2. Numerical Feature Analysis:
        Provides statistics (mean, median...), including skewness and kurtosis, to detail numerical data distribution.
    3. Categorical Variable Analysis:
        Lists unique counts, top and rare categories, aiding in the assessment of categorical data distribution.
    4. Missing Values Analysis:
        Identifies and counts missing values per column, essential for data cleaning.
    5. Correlation Analysis:
        Calculates most (anti-)correlated features, creates a correlogram.
    6. Outliers Identification:
        Detects outliers using IQR, reporting counts and bounds, crucial for data quality assessment.
    7. Duplicate Records Analysis:
        Checks and reports the count of duplicate records, important for ensuring data integrity.

    Args:
        tc (ToolCommunicator): tool communicator object.
        data_file_path (str): path to the data file.
        target (str): target feature name.

    Returns:
        str: Detailed EDA report.
    """
    df = pd.read_csv(data_file_path)
    analysis_summary = ""

    # Dataset basic info
    tc.print("Getting dataset basic info...")
    time.sleep(0.4)
    analysis_summary += f"Dataset Shape: {df.shape[0]} rows and {df.shape[1]} columns\n"
    analysis_summary += f"Column Names and Types:\n{df.dtypes.to_string()}\n\n"

    # Enhanced Descriptive statistics for numerical features
    tc.print("Getting descriptive statistics for numerical features...")
    analysis_summary += "Descriptive Statistics for Numerical Features:\n"
    numerical_stats = df.describe(include=[np.number])
    numerical_stats.loc["skew"] = df.select_dtypes(include=[np.number]).skew()  # type: ignore
    numerical_stats.loc["kurt"] = df.select_dtypes(include=[np.number]).kurtosis()  # type: ignore
    analysis_summary += f"{numerical_stats.to_string()}\n\n"

    # Detailed information on categorical variables
    tc.print("Getting detailed information on categorical variables...")
    time.sleep(0.4)
    numerics_that_are_categoricals = id_numerics_actually_categoricals(df)
    numerics_that_are_categoricals_info = (
        "Identified numeric value columns that should most likely be considered categoricals:"
        f"\n{numerics_that_are_categoricals}.\n"
        "This is done by checking whether the column contains only integers and "
        "has a low number of unique values (<20 or <5% of total examples).\n"
    )
    categorical_columns = df.select_dtypes(include=["object"]).columns
    categorical_columns = list(set(categorical_columns).union(set(numerics_that_are_categoricals)))
    analysis_summary += f"{numerics_that_are_categoricals_info}\n"
    analysis_summary += "Detailed Information on Categorical Variables:\n"
    for col in categorical_columns:
        analysis_summary += (
            f"{col} - Unique Values: {df[col].nunique()} \nTop 5 Values:\n{df[col].value_counts().head().to_string()}"
        )
        if df[col].nunique() > 5:
            analysis_summary += f"\nRare Categories:\n{df[col].value_counts().tail(5).to_string()}\n\n"
        else:
            analysis_summary += "\n\n"

    # Missing values analysis
    tc.print("Performing missing values analysis...")
    time.sleep(0.4)
    missing_values = df.isnull().sum()
    analysis_summary += "Missing Values Analysis:\n"
    if missing_values.any():
        analysis_summary += f"{missing_values[missing_values > 0].to_string()}\n\n"
    else:
        analysis_summary += "No missing values found.\n\n"
    # Count all NaN rows
    all_nan_rows = df.isna().all(axis=1).sum()
    # Count all NaN columns
    all_nan_columns = df.isna().all(axis=0).sum()
    if all_nan_rows > 0:
        analysis_summary += f"Count of rows with all NaN values: {all_nan_rows}\n"
    if all_nan_columns > 0:
        analysis_summary += f"Count of columns with all NaN values: {all_nan_columns}\n"

    # Correlation analysis
    tc.print("Performing correlation analysis...")
    time.sleep(0.4)
    # Old code, for info:
    # analysis_summary += "Correlation Analysis (Numerical Features):\n"
    # if df.shape[1] <= 15:
    #     analysis_summary += f"{df.select_dtypes(include=[np.number]).corr().to_string()}\n\n"  # type: ignore
    # else:
    #     analysis_summary += "Too many columns to calculate correlations.\n\n"
    try:
        correlogram_text, correlogram_fig = generate_correlogram(
            df,
            numerics_that_are_categoricals=numerics_that_are_categoricals,
            show_n_corr=10,
            workspace=workspace,
            target=target,
        )
        analysis_summary += "Correlation Analysis:\n"
        analysis_summary += correlogram_text
    except Exception:
        analysis_summary += "There was a problem generating the correlogram, skipped."
        correlogram_text = ""
        correlogram_fig = None

    # Potential outliers identification with more details
    tc.print("Performing potential outliers identification...")
    time.sleep(0.4)
    analysis_summary += "\nOutlier Identification for Numerical Features:\n"
    for col in df.select_dtypes(include=[np.number]).columns:  # type: ignore
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.75 * IQR
        upper_bound = Q3 + 1.75 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        analysis_summary += (
            f"{col} - Outliers Count: {outliers.shape[0]}\n[Lower Bound: {lower_bound:.3g}, "
            f"Upper Bound: {upper_bound:.3g}]\n"
        )

    # Duplicate records analysis
    tc.print("Performing duplicate records analysis...")
    time.sleep(0.4)
    duplicates = df.duplicated().sum()
    analysis_summary += f"\nDuplicate Records: {duplicates}\n\n"

    if correlogram_fig is not None:
        user_report = [
            "Here is a correlogram showing the correlation between features:",
            correlogram_fig,
            """A correlogram is a visual tool that shows the relationships between different variables (or features) \
in a dataset. It presents a grid of color-coded squares, where each square represents the strength and direction \
of the relationship between two variables. Brighter or darker colors indicate stronger relationships. \
Positive relationships (where variables increase together) and negative relationships (where one variable increases \
as the other decreases) are shown with different colors. Here we use reds to denote positive, and blue to denote \
negative relationships. This makes it easy to see which pairs of variables are related, and how closely they are \
connected.
""",
        ]
    else:
        user_report = None
    tc.set_returns(
        tool_return=analysis_summary,
        user_report=user_report,
    )


class ExploratoryDataAnalysis(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        real_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        target = kwargs.get("target", None)
        thrd, out_stream = execute_tool(
            exploratory_data_analysis,
            wd=self.working_directory,
            data_file_path=real_path,
            workspace=self.working_directory,
            target=target,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "EDA"

    @property
    def description(self) -> str:
        return get_str_up_to_marker(exploratory_data_analysis.__doc__, "Args")  # type: ignore

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
                        "target": {
                            "type": "string",
                            "description": "Target feature name.",
                            "default": None,
                        },
                    },
                    "required": ["data_file_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "performs exploratory data analysis on your data, providing a summary of its characteristics."
