import copy
import enum
import importlib
import os
import pickle
from typing import Any, Dict

import matplotlib.figure
import plotly.graph_objects

from . import Message, Session
from .utils import make_filename_path_safe


def encode_enum(obj: enum.Enum) -> str:
    """Store the module and the enum name and value in a string separated by a slash.

    Args:
        obj (enum.Enum): The enum object to encode.

    Returns:
        str: The encoded string.
    """
    # Note: we record the module name to ensure that the enum can be properly imported.
    encoding = f"{type(obj).__module__}/{str(obj)}"
    return encoding


def decode_enum(s: str) -> enum.Enum:
    """Recover the module and the enum name and value from the string, instantiate the enum and return it.

    Args:
        s (str): The encoded string.

    Returns:
        enum.Enum: The decoded enum object.
    """
    module_str, enum_part = s.split("/")
    enum_name, enum_value = enum_part.split(".")
    # Note: use the module name to dynamically import the module that has the enum.
    module = importlib.import_module(module_str)
    enum_cls = getattr(module, enum_name)
    return enum_cls[enum_value]


# TODO: This should be made properly modular etc. Currently it's just a quick hack.
def message_to_serializable_dict(message: Message, session_path: str) -> Dict[str, Any]:
    pickle_dir = os.path.join(session_path, "session_pickles", make_filename_path_safe(message.key))

    message_dump = message.model_dump(by_alias=True)
    new_message_dump = copy.deepcopy(message_dump)

    # Handle enum (ResponseKind), which isn't directly serializable.
    if message.engine_state is not None:
        new_message_dump["engine_state"]["response_kind"] = encode_enum(message.engine_state.response_kind_value)

    # Handle the figure objects.
    if message.tool_call_user_report is not None:
        serializable = []

        for idx, report_item in enumerate(message.tool_call_user_report):
            if isinstance(report_item, plotly.graph_objects.Figure):
                os.makedirs(pickle_dir, exist_ok=True)
                pickle_path = os.path.join(pickle_dir, f"{idx}__plotly_figure.pickle")
                with open(pickle_path, "wb") as f:
                    pickle.dump(report_item, f)

                serializable.append({"type": "plotly_figure", "report_item_idx": idx, "path": pickle_path})

            elif isinstance(report_item, matplotlib.figure.Figure):
                os.makedirs(pickle_dir, exist_ok=True)
                pickle_path = os.path.join(pickle_dir, f"{idx}__matplotlib_figure.pickle")
                with open(pickle_path, "wb") as f:
                    pickle.dump(report_item, f)

                serializable.append({"type": "matplotlib_figure", "report_item_idx": idx, "path": pickle_path})

            elif isinstance(report_item, str):
                serializable.append({"type": "str", "report_item_idx": idx, "content": report_item})

            else:
                raise ValueError(f"Message serialization failed. Unsupported report item type: {type(report_item)}")

        new_message_dump["tool_call_user_report"] = serializable

    return new_message_dump


def message_from_serializable_dict(message_dict: Dict[str, Any]) -> Message:
    message_dict_new = copy.deepcopy(message_dict)

    # Handle enum (ResponseKind), which isn't directly serializable.
    if message_dict["engine_state"] is not None:
        message_dict_new["engine_state"]["response_kind"] = decode_enum(message_dict["engine_state"]["response_kind"])

    # Handle the figure objects.
    if message_dict["tool_call_user_report"]:
        deserialized = []

        for report_item in message_dict["tool_call_user_report"]:
            if report_item["type"] == "plotly_figure":
                try:
                    with open(report_item["path"], "rb") as f:
                        deserialized.append(pickle.load(f))
                except Exception as e:
                    print(f"Failed to deserialize plotly figure from {report_item['path']}: {e}")
                    report_item["type"] = "str"
                    deserialized.append("< Failed to deserialize plotly figure >")

            elif report_item["type"] == "matplotlib_figure":
                try:
                    with open(report_item["path"], "rb") as f:
                        deserialized.append(pickle.load(f))
                except Exception as e:
                    print(f"Failed to deserialize matplotlib figure from {report_item['path']}: {e}")
                    report_item["type"] = "str"
                    deserialized.append("< Failed to deserialize matplotlib figure >")

            elif report_item["type"] == "str":
                deserialized.append(report_item["content"])

            else:
                raise ValueError(f"Message deserialization failed. Unsupported report item type: {report_item['type']}")

        message_dict_new["tool_call_user_report"] = deserialized

    return Message(**message_dict_new)


def session_to_serializable_dict(session: Session) -> Dict[str, Any]:
    session_dump = session.model_dump()

    if session.messages:
        serialized_messages = [
            message_to_serializable_dict(message, session.working_directory) for message in session.messages
        ]
        session_dump["messages"] = serialized_messages
    else:
        session_dump["messages"] = []

    return session_dump


def session_from_serializable_dict(session_dict: Dict[str, Any]) -> Session:
    session_dict_new = copy.deepcopy(session_dict)

    if session_dict["messages"]:
        session_dict_new["messages"] = [message_from_serializable_dict(message) for message in session_dict["messages"]]
    else:
        session_dict_new["messages"] = []

    return Session(**session_dict_new)
