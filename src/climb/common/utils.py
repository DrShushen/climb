import os
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple, TypeVar

import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.representer import RoundTripRepresenter

# from ruamel.yaml.comments import CommentedMap as ordereddict


def ui_log(*args: Any, **kwargs) -> None:
    print("[UI]  >>>", *args, **kwargs)


def engine_log(*args: Any, **kwargs) -> None:
    print("[ENG] >>>", *args, **kwargs)


def convert_size(size_bytes: float) -> Tuple[float, str]:
    """Convert file size to a more readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if size_bytes < 1024.0:
            return (size_bytes, unit)
        size_bytes /= 1024.0
    return (size_bytes, "PB")


def make_filename_path_safe(s: str, remove_slashes: bool = False) -> str:
    """Make a string safe to use as a path. A simple implementation, only applicable to file name, not the full path."""
    # Replace any characters that are not letters, numbers, underscores, or dot with a dash:
    if remove_slashes:
        s = s.replace("/", "").replace("\\", "")
    elif "/" in s or "\\" in s:
        raise ValueError("The input string should not contain path separators.")
    return re.sub(r"[^\.\w]", "-", s)


def log_messages_to_file(
    messages: List[Dict],
    tools: Optional[List[Dict]],
    metadata: Optional[Dict],
    path: str,
) -> None:
    # Why use ruamel.yaml instead of PyYAML?
    # PyYAML fails to properly handle long multiline strings with |, but we want to always have actual new lines
    # and not \n characters. This is not possible with PyYAML.
    # Note however that the key order is not preserved in ruamel.yaml. No workaround for this for now.

    # Save the messages to a file.
    dump = {"METADATA": metadata, "MESSAGES": messages, "TOOLS": tools}

    # Recursively go through `dump` and replace any dict with ordered dict. Iterate over lists.
    # Not currently using, as this approach, though preserves key order, causes lots of confusing
    # !!omap tags in the YAML.
    # def ordered_dict(d: Dict):
    #     for k, v in d.items():
    #         if isinstance(v, dict):
    #             d[k] = ordered_dict(v)
    #         elif isinstance(v, list):
    #             for i, item in enumerate(v):
    #                 if isinstance(item, dict):
    #                     v[i] = ordered_dict(item)
    #     return OrderedDict(d)
    # dump = ordered_dict(dump)

    def repr_str(dumper: RoundTripRepresenter, data: str):
        # Ensure multiline strings with yaml | only when there is indeed a newline in the string.
        if "\n" in data:
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
        return dumper.represent_scalar("tag:yaml.org,2002:str", data)

    yaml = YAML(typ="safe", pure=True)
    yaml.default_flow_style = False
    yaml.representer.add_representer(str, repr_str)

    with open(path, "w") as f:
        yaml.dump(dump, f)


def replace_str_from_dict(s: str, d: Dict[str, Any]) -> str:
    """Replace all instances of keys in a string with their corresponding values. The dictionary values that are not \
    strings will be converted to strings ``str(value)`` before replacement.

    Args:
        s (str): The string to modify.
        d (Dict[str, Any]): The dictionary that contains the replacement information. Keys are the strings to be \
            replaced, and values are the strings to replace them with.

    Returns:
        str: The modified string.
    """
    for key, value in d.items():
        s = s.replace(key, str(value))
    return s


def similar(a: Any, b: Any) -> float:
    return SequenceMatcher(None, a, b).ratio()


def attempt_imputation_match(df1: pd.DataFrame, df2: pd.DataFrame, nan_sentinel: Any) -> List[int]:
    # Find the rows that were imputed from df1 to df2. Imputed = cell changes like "was NaN, no longer NaN" ONLY.
    comparison = pd.DataFrame.compare(df1, df2, keep_shape=True, keep_equal=True)

    changed = comparison.loc[:, (slice(None), "self")].values != comparison.loc[:, (slice(None), "other")].values  # type: ignore
    changed_df = df1.copy()
    changed_df[:] = changed

    source_nan = comparison.loc[:, (slice(None), "self")] == nan_sentinel  # type: ignore

    # imputed = source_nan.values & changed_df.values
    not_imputed = ~source_nan.values & changed_df.values
    # imputed_df = df1.copy()
    # imputed_df[:] = imputed
    not_imputed_df = df1.copy()
    not_imputed_df[:] = not_imputed

    original_index = df1.index
    not_imputed_index = not_imputed_df[not_imputed_df.sum(axis=1) != 0].index
    imputed_indexes = set(original_index) - set(not_imputed_index)

    return list(imputed_indexes)


def analyze_df_modifications(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    row_similarity_threshold: float = 0.8,
) -> Dict[str, Any]:
    success = True

    try:
        # NaNs cause matching issues, replace them with a dummy value -987.654:
        NAN_SENTINEL = -987.654
        df_before = df_before.copy(deep=True)
        df_before.fillna(NAN_SENTINEL, inplace=True)
        df_after = df_after.copy(deep=True)
        df_after.fillna(NAN_SENTINEL, inplace=True)

        # Get columns removed:
        cols_removed = list(set(df_before.columns) - set(df_after.columns))

        # Get columns added:
        cols_added = list(set(df_after.columns) - set(df_before.columns))

        # Get common columns:
        common_columns = list(set(df_before.columns).intersection(set(df_after.columns)))

        # DataFrames with only common columns:
        df_before_common = df_before[common_columns]
        df_after_common = df_after[common_columns]
        # print("df_before_common:\n", df_before_common)

        rows_removed = []
        rows_added = []
        modified_values = []

        before_rows_set = set([tuple(row) for row in df_before_common.values])
        after_rows_set = set([tuple(row) for row in df_after_common.values])

        # Attempt to detect imputations:
        try:
            imputed_rows = attempt_imputation_match(df_before_common, df_after_common, NAN_SENTINEL)
        except Exception as e:
            print("ERROR in attempt_imputation_match:\n", e)
            imputed_rows = []
        # print("imputed_rows:\n", imputed_rows)

        modified_rows_set__after = [tuple(df_after_common.loc[x, :]) for x in imputed_rows]
        modified_rows_set__before = [tuple(df_before_common.loc[x, :]) for x in imputed_rows]
        for row in after_rows_set:
            idx_after = df_after_common.index[df_after_common.apply(tuple, axis=1) == row][0]
            most_similar = None
            highest_similarity = 0

            for idx_before, before_row in df_before_common.iterrows():
                similarity = similar(row, tuple(before_row))
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    most_similar = idx_before

            if highest_similarity > row_similarity_threshold:  # Threshold for considering a row as modified.
                modified_rows_set__after.append(row)
                modified_rows_set__before.append(tuple(df_before_common.loc[most_similar, :]))  # type: ignore
                for col in common_columns:
                    if df_before.at[most_similar, col] != df_after.at[idx_after, col]:
                        before = df_before.at[most_similar, col]
                        after = df_after.at[idx_after, col]
                        # Restore NaNs:
                        if before == NAN_SENTINEL:
                            before = None
                        if after == NAN_SENTINEL:
                            after = None
                        # ---
                        modified_values.append((idx_after, col, f"{before} → {after}"))

        modified_rows_set__after = set(modified_rows_set__after)
        modified_rows_set__before = set(modified_rows_set__before)

        non_modified_row_set__after = after_rows_set - modified_rows_set__after
        non_modified_row_set__before = before_rows_set - modified_rows_set__before

        removed_rows_set = non_modified_row_set__before
        added_rows_set = non_modified_row_set__after

        for row in removed_rows_set:
            rows_removed.append(df_before_common.index[df_before_common.apply(tuple, axis=1) == row][0])

        for row in added_rows_set:
            rows_added.append(df_after_common.index[df_after_common.apply(tuple, axis=1) == row][0])

    except Exception as e:
        print("ERROR in analyze_modifications:\n", e)
        success = False
        raise

    return {
        "columns_removed": cols_removed,
        "columns_added": cols_added,
        "rows_removed": rows_removed,
        "rows_added": rows_added,
        "modified_values": modified_values,
        "success": success,
    }


def fix_windows_path_backslashes(path: str) -> str:
    """Modify the path string for Windows with correct escape sequences.
    Converts single backslashes to double backslashes.

    Args:
        path (str): The original file path string.

    Returns:
        str: Modified path with double backslashes where needed.
    """
    if os.name == "nt":  # Checks if the OS is Windows.
        # Replace single backslashes with double backslashes.
        # This regex looks for single backslashes that are not followed or preceded by another backslash.
        path = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", path)
    return path


def dedent(text: str) -> str:
    """Remove the same amount of leading spaces from each line of the text.

    Args:
        text (str): Multiline string to process.

    Returns:
        str: Processed text with leading spaces removed.
    """
    # Split the text into lines:
    lines = text.splitlines()
    if not lines:
        return text  # Return the original text if it's empty or has no lines

    first_line_index = 0 if lines[0].strip() or len(lines) == 1 else 1

    # Find the number of leading spaces in the first line:
    first_line_spaces = len(lines[first_line_index]) - len(lines[first_line_index].lstrip(" "))

    # Remove that many spaces from the start of each line, if present:
    processed_lines = [line[first_line_spaces:] if line.startswith(" " * first_line_spaces) else line for line in lines]

    # Join the lines back into a single string:
    return "\n".join(processed_lines)


def truncate_dict_values(d: Dict, max_len: int = 50) -> Dict:
    """
    Recursively truncates dictionary values whose string representation exceeds max_len characters. To be used for
    shorter representations of possibly large nested dictionaries.

    Args:
        d (Dict): Input dictionary, possibly nested.
        max_len (int): Maximum length of the string representation of the values.

    Returns:
        Dict: New dictionary with possibly truncated values.
    """
    new_dict = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = truncate_dict_values(value, max_len)
        else:
            value_str = repr(value)
            if len(value_str) > max_len:
                new_dict[key] = f"<long repr truncated>:{value_str[:max_len]}..."
            else:
                new_dict[key] = value
    return new_dict


T = TypeVar("T")


def d2m(d: Dict[str, Any], model: T) -> T:
    """`d(ictionary)_to_m(odel)`. Convert a dictionary to a Pydantic model instance`.

    Args:
        d (Dict[str, Any]): The dictionary to convert.
        model (pydantic.BaseModel): The Pydantic model class to convert to.

    Returns:
        pydantic.BaseModel: The Pydantic model instance.
    """
    return model(**d)  # type: ignore


def m2d(model: Any) -> Dict[str, Any]:
    """`m(odel)_to_d(ictionary)`. Convert a Pydantic model instance to a dictionary.

    Args:
        model (Any): The Pydantic model instance to convert.

    Returns:
        Dict[str, Any]: The dictionary.
    """
    return model.model_dump()


def update_templates(body_text: str, templates: Dict[str, str]) -> str:
    for k, v in templates.items():
        body_text = body_text.replace(f"{k}", v)
    return body_text


def filter_out_lines(in_str: str) -> str:
    # TODO: This is a temporary solution. A more robust filtering mechanism should be implemented.
    # A minimal filter for all tool logs (during streaming and final log saving).
    FILTERS = [
        "FutureWarning:",
        "`sparse` was renamed",
        "elementwise comparison failed;",
        "UserWarning:",
        "The least populated class in y has only 1 members",
        "RuntimeWarning:",
        "divide by zero encountered",
    ]
    lines = in_str.split("\n")
    return "\n".join([line for line in lines if not any([filt in line for filt in FILTERS])])
