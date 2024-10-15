import copy
import enum
import importlib
import json
import os
import pickle
from functools import partial
from io import StringIO
from typing import Any, Dict

import matplotlib.figure
import plotly.graph_objects
from nutree import Node, Tree

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


# === Tree-related serialization ===

USE_DATA_KEY = "data"


def _serialize_mapper(node: Node, data: Dict[str, Any], working_dir: str) -> Dict[str, Any]:
    # Validate that the node data is a Message.
    node_data = node.data
    if not isinstance(node_data, Message):
        raise ValueError(f"Node data is not a Message: {node_data}")
    # Save the message representation dictionary in the `USE_DATA_KEY` key of the `data` dictionary.
    # Pass the message data through our custom serialization function.
    data[USE_DATA_KEY] = message_to_serializable_dict(node_data, working_dir)
    return data


def serialize_message_tree(tree: Tree, working_dir: str) -> Dict[str, Any]:
    """Serialize a message tree to a dictionary.

    Args:
        tree (Tree): The tree to serialize.
        working_dir (str): The working directory that is used for storing some non-serializable data via pickling.

    Returns:
        Dict[str, Any]: The serialized tree.
    """
    # Use StringIO to simulate a file-like object that the Tree.save method expects.
    string_io = StringIO()
    # Use a mapper to serialize the data in a way we need.
    serialize_mapper_partial = partial(_serialize_mapper, working_dir=working_dir)
    tree.save(mapper=serialize_mapper_partial, target=string_io)
    string_io_value = string_io.getvalue()
    # Convert the string to a dictionary.
    return json.loads(string_io_value)


def _deserialize_mapper(parent: Node, data: Dict[str, Any]) -> Any:
    # The message data dictionary is stored in the `data` dictionary `USE_DATA_KEY` key.
    # Feed this data dictionary through the message_from_serializable_dict function to parse any complex objects.
    return message_from_serializable_dict(data[USE_DATA_KEY])


def deserialize_message_tree(serialized_tree: Dict[str, Any]) -> Tree:
    """Deserialize a message tree from its serialized form.

    Args:
        serialized_tree (Dict[str, Any]): The serialized tree.

    Returns:
        Tree: The deserialized tree.
    """
    # Use StringIO to simulate a file-like object that the Tree.load method expects.
    string_io = StringIO(json.dumps(serialized_tree))
    # Use a mapper to deserialize the data in a way we need.
    return Tree.load(target=string_io, mapper=_deserialize_mapper)


# === Tree-related serialization [END] ===


def session_to_serializable_dict(session: Session) -> Dict[str, Any]:
    session_dump = session.model_dump()

    if session.message_tree is not None:
        serialized_messages_tree = serialize_message_tree(session.messages, session.working_directory)
        session_dump["message_tree"] = serialized_messages_tree
    else:
        session_dump["message_tree"] = None

    return session_dump


def session_from_serializable_dict(session_dict: Dict[str, Any]) -> Session:
    session_dict_new = copy.deepcopy(session_dict)

    if session_dict["message_tree"] is not None:
        session_dict_new["message_tree"] = deserialize_message_tree(session_dict["message_tree"])
    else:
        session_dict_new["message_tree"] = None

    return Session(**session_dict_new)
