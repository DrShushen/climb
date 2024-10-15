import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from climb.common.utils import make_filename_path_safe

from .._utils import decimal_places_for_sf, id_numerics_actually_categoricals
from ..tool_comms import ToolCommunicator, ToolReturnIter, execute_tool
from ..tools import ToolBase, get_str_up_to_marker


def run_with_time_limit(func, time_limit=10, **kwargs) -> Any:
    def wrapper():
        try:
            result[0] = func(**kwargs)
        except Exception as e:
            result[0] = str(e)  # type: ignore

    result = [None]
    thread = threading.Thread(target=wrapper)
    thread.start()
    thread.join(timeout=time_limit)

    if thread.is_alive():
        return "Out of time"
    else:
        return result[0]


def check_normal_distribution(
    df: pd.DataFrame,
    max_rows: int = 5000,
    p_value_thresh: float = 1e-5,
    random_state: int = 0,
    subset_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    # NOTE: P value threshold is set to 1e-5 as this seems to be similar to what a human would conclude when looking
    # at a plot of the data. This is somewhat subjective.

    # Downsample DataFrame to max_rows rows if it has more than 5000 rows:
    if df.shape[0] > max_rows:
        print(
            f"Downsampling DataFrame to {max_rows} rows as Shapiro-Wilk test is not well-suited for large sample sizes."
        )
        df = df.sample(max_rows, random_state=random_state, replace=False)

    features_normally_distributed = []
    features_not_normally_distributed = []

    if subset_cols is not None:
        # If subset_cols is provided, only consider those columns:
        df = df[subset_cols]

    # Iterate over columns in DataFrame:
    for column in df.select_dtypes(include=[np.number]).columns:  # type: ignore
        # Apply Shapiro-Wilk test:
        try:
            stat, p_value = stats.shapiro(df[column].dropna())  # pylint: disable=unused-variable
        except ValueError as e:
            # If there are not enough data points after dropping NaNs, assume not normally distributed:
            if "at least length" in str(e):
                features_not_normally_distributed.append(column)
                continue

        # Categorize based on p-value:
        # print(f"Shapiro-Wilk test for {column}: p-value = {p_value}")
        if p_value > p_value_thresh:
            features_normally_distributed.append(column)
        else:
            features_not_normally_distributed.append(column)

    print(f"Normally distributed features:\n{features_normally_distributed}")
    print(f"Not normally distributed features:\n{features_not_normally_distributed}")
    return features_normally_distributed, features_not_normally_distributed


def top_n_with_other(df: pd.DataFrame, column: str, n_explicit: int = 5, other_name: str = "Other") -> pd.Series:
    # Get the value counts:
    value_counts = df[column].value_counts()

    # Select the top n_explicit categories:
    top_n = value_counts.nlargest(n_explicit)

    if len(value_counts) <= n_explicit:
        return value_counts

    # Create a new Series where categories not in the top 5 are labeled as "Other":
    modified_series = df[column].apply(lambda x: x if x in top_n.index else other_name)

    # Get value counts for the modified series, including 'Other':
    modified_value_counts = modified_series.value_counts()

    # Sort so that "Other" is at the end:
    modified_value_counts = modified_value_counts.reindex(top_n.index.tolist() + [other_name])

    return modified_value_counts


def summarize_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
    # Returns:
    # - summary_df: DataFrame containing the summary statistics.
    # - categorical_columns: List of categorical columns.
    # - numeric_columns: List of all numeric columns.
    # - normal: List of normally distributed numeric columns subset.
    # - non_normal: List of non-normally distributed numeric columns subset.

    summary_data = []

    numerics_that_are_categoricals = id_numerics_actually_categoricals(df)

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    categorical_columns = categorical_columns.union(numerics_that_are_categoricals)

    numeric_columns = df.select_dtypes(include=[np.number]).columns  # type: ignore
    numeric_columns = numeric_columns.difference(numerics_that_are_categoricals)

    normal, non_normal = check_normal_distribution(df, subset_cols=numeric_columns)  # type: ignore

    for column in df.columns:
        if column in categorical_columns:
            total_count = df[column].count()
            category_counts = top_n_with_other(df, column)
            summary_data.append(
                {
                    "Variable": column,
                    "Summary": "",
                }
            )
            for category, count in category_counts.to_dict().items():
                percentage = (count / total_count) * 100
                summary_data.append(
                    {"Variable": f"<<{column}>> {category}", "Summary": f"{count}/{total_count} ({percentage:.1f})"}
                )

        elif column in numeric_columns:
            if column in normal:
                mean = df[column].mean()
                std = df[column].std()
                dps = decimal_places_for_sf(mean, n_sf=3)
                summary_data.append({"Variable": column, "Summary": f"{mean:.{dps}f} ± {std:.{dps}f}"})
            elif column in non_normal:
                median = df[column].median()
                q4, q1 = df[column].quantile(0.75), df[column].quantile(0.25)
                dps = decimal_places_for_sf(median, n_sf=3)
                summary_data.append({"Variable": column, "Summary": f"{median:.{dps}f} ({q1:.{dps}f} - {q4:.{dps}f})"})

    # NOTE:
    # If the columns are neither numeric nor categorical, these will be skipped.

    summary_df = pd.DataFrame(summary_data).set_index("Variable", drop=True)

    return (summary_df, list(categorical_columns), list(numeric_columns), normal, non_normal)


def format_descriptive_statistics_table_for_print(df: pd.DataFrame) -> str:
    # Prepare for display:
    df.reset_index(inplace=True, drop=False)
    # Replace any content in the "Variable" column that matches <<CONTENT>> with "   " using a regex:
    df["Variable"] = df["Variable"].apply(lambda x: re.sub(r"<<.*?>>", "   ", x))

    # See: https://stackoverflow.com/a/69737077
    formatters = dict()
    len_max = df["Variable"].str.len().max()
    formatters["Variable"] = lambda _: f"{_:<{len_max}s}"

    # Convert summary_df to string without collapsing any rows or columns:
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        # pd.set_properties(subset=["col1", "col2"], **{'text-align': 'right'})
        summary_str = df.to_string(index=False, formatters=formatters)

    return summary_str


def plot_and_save_columns(
    dataframe: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    workspace: str,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    TIME_LIMIT = 4

    def _plot_categorical_column(column: str) -> Tuple[str, matplotlib.figure.Figure]:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=dataframe[column].value_counts().index, y=dataframe[column].value_counts().values)
        ax.set_title(f"Bar plot of {column}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        plot_filename = make_filename_path_safe(f"descr__bar_plot__{column}.png", remove_slashes=True)
        # filename_dict[column] = plot_filename
        plt.savefig(os.path.join(workspace, plot_filename))
        plt.close()

        # Store the figure object in the dictionary
        # plots_dict[column] = ax.get_figure()

        return (
            plot_filename,
            ax.get_figure(),  # pyright: ignore
        )

    def _plot_numeric_column(column: str) -> Tuple[str, matplotlib.figure.Figure]:
        # Create a figure with a 2x1 grid of axes
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

        # Histogram
        sns.histplot(dataframe[column], ax=axs[0], kde=True)  # type: ignore
        axs[0].set_title(f"Histogram of {column}")

        # Box plot
        sns.boxplot(x=dataframe[column], ax=axs[1])
        axs[1].set_title(f"Box plot of {column}")

        plt.tight_layout()

        # Save the plot
        plot_filename = make_filename_path_safe(f"descr__hist_box_plot__{column}.png", remove_slashes=True)
        # filename_dict[column] = plot_filename
        plt.savefig(os.path.join(workspace, plot_filename))
        plt.close()

        # Store the figure object in the dictionary
        # plots_dict[column] = fig

        return plot_filename, fig

    # Returns:
    # - plots_dict: Dictionary containing the figure objects for each column.
    # - plot_filenames: Dictionary containing the filenames for each plot.
    plots_dict = {}
    filename_dict = {}

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Plot for each categorical column
    for column in categorical_columns:
        out = run_with_time_limit(_plot_categorical_column, time_limit=TIME_LIMIT, column=column)
        if out == "Out of time":
            print(f"Plotting of '{column}' took too long and was skipped.")
            continue
        else:
            plot_filename, fig = out
            filename_dict[column] = plot_filename
            plots_dict[column] = fig
            print(f"Plotted a bar plot for: '{column}'")

    # Plot for each numeric column
    for column in numeric_columns:
        out = run_with_time_limit(_plot_numeric_column, time_limit=TIME_LIMIT, column=column)
        if out == "Out of time":
            print(f"Plotting of '{column}' took too long and was skipped.")
            continue
        else:
            plot_filename, fig = out
            filename_dict[column] = plot_filename
            plots_dict[column] = fig
            print(f"Plotted a histogram and box plot for: '{column}'")

    return plots_dict, filename_dict


def create_descriptive_statistics_table(
    tc: ToolCommunicator,
    data_file_path: str,
    workspace: str,
) -> None:
    """Create a medical paper style descriptive statistics table for a dataset.

    Details:
    - Categorical variables are summarized by listing unique values and showing: count / total (percentage).
    - Numerical variables are summarized by showing: mean ± std (if normally distributed) or median (Q1 - Q3) (if not).
    - The user will also be shown plots of the data:
        - bar plots for categorical variables,
        - and histograms and box plots for numerical variables.

    Args:
        tc (ToolCommunicator): tool communicator object.
        data_file_path (str): path to the data file.
        workspace (str): path to the workspace directory.
    """

    tc.print("Creating the descriptive statistics table...")

    df = pd.read_csv(data_file_path)
    summary_df, categoricals, numerics, *_ = summarize_dataframe(df)

    save_file_name = f"{data_file_path}__descriptive_stats.csv"
    tc.print(f"Saving the summary table to:\n{save_file_name}")
    summary_df.to_csv(os.path.join(workspace, save_file_name), index=True)

    summary_str = format_descriptive_statistics_table_for_print(summary_df)

    tc.print("Creating plots for the data...")
    # pylint: disable-next=unused-variable
    plots_dict, filenames_dict = plot_and_save_columns(df, categoricals, numerics, workspace)
    summary_str += "\n\nThe following plots have also been created and saved:\n"
    for column, filename in filenames_dict.items():
        summary_str += f"- {column}: {filename}\n"

    for_user = [
        "📊 **Descriptive Statistics Plots**",
        (
            """
            > ⚠️ Since there may be a very large number of plots, these are not shown here, as this can slow down the UI.

            To view the plots, please select any of the images in the "Working Directory" tab on the right.
            """
        ),
    ]
    # TODO: Find a way to display the plots without slowing down the UI.
    # for column, plot, filename in zip(plots_dict.keys(), plots_dict.values(), filenames_dict.values()):
    #     for_user.append(f"*{column}*:")
    #     for_user.append(f"Saved in `{filename}`")
    #     for_user.append(plot)

    tc.set_returns(
        tool_return=summary_str,
        user_report=for_user,  # type: ignore
    )


class DescriptiveStatistics(ToolBase):
    def _execute(self, **kwargs: Any) -> ToolReturnIter:
        data_file_path = os.path.join(self.working_directory, kwargs["data_file_path"])
        thrd, out_stream = execute_tool(
            create_descriptive_statistics_table,
            data_file_path=data_file_path,
            workspace=self.working_directory,
            # ---
            wd=self.working_directory,
        )
        self.tool_thread = thrd
        return out_stream

    @property
    def name(self) -> str:
        return "descriptive_statistics"

    @property
    def description(self) -> str:
        return get_str_up_to_marker(create_descriptive_statistics_table.__doc__, "Args")  # type: ignore

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
                    },
                    "required": ["data_file_path"],
                },
            },
        }

    @property
    def description_for_user(self) -> str:
        return "produce medical paper -style descriptive statistics table for the dataset."
