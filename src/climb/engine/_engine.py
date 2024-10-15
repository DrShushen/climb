import abc
import os
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, Optional, Tuple, Union, get_args

from nutree import Node, Tree
from openai import Stream

from climb.common import (
    Agent,
    EngineParameter,
    EngineParameterValue,
    EngineState,
    FileInfo,
    Message,
    Role,
    Session,
    ToolCallRecord,
    ToolSpecs,
)
from climb.common.utils import convert_size, engine_log, log_messages_to_file, make_filename_path_safe
from climb.db import DB
from climb.tool import ToolBase, ToolReturnIter, UserInputRequest

from ._code_execution import CodeExecReturn
from .common import get_dotenv_config

dotenv_config = get_dotenv_config()

# The maximum number of branches a message can have for the CoT.
BRANCH_LIMIT = int(dotenv_config.get("BRANCH_LIMIT", 2))  # type: ignore
BRANCH_ROLE = "user"

StreamLike = Union[Iterator, Generator, Stream]

ChunkSentinel = Literal["not_started", "text", "tool_call", "end_of_stream"]
LoadingIndicator = ("loading",)

PrivacyModes = Literal["default", "guardrail", "guardrail_with_approval"]

PRIVACY_MODE_PARAMETER_DESCRIPTION = """
Privacy mode to use for the engine.
- `default`: No additional privacy guardrails imposed. The LLM can access all messages and is not given any restrictions \
on what data it can view.
- `guardrail`: Guardrail privacy mode. The LLM receives an explicit guardrail instructions that it is not allowed to \
view the data directly when generating code etc. The LLM can still access metadata and various summary statistics. \
Note that this mode does not guarantee privacy, but it is a best-effort approach to limit the LLM's access to data. \
It is still possible to accidentally or purposefully leak the data. User caution is advised.
- `guardrail_with_approval`: Guardrail privacy mode with user approval. Each time the LLM needs is about to receive \
message history, the user is requested to review and approve the latest message to ensure that the data is safe to \
share with the LLM provider.
"""
PrivacyModeParameter = EngineParameter(
    name="privacy_mode",
    kind="enum",
    default="default",
    description=PRIVACY_MODE_PARAMETER_DESCRIPTION,
    enum_values=list(get_args(PrivacyModes)),
)

# === Tree-related functions ===


class tree_helpers:
    # This class is just to group methods in a namespace for clarity.

    @staticmethod
    def get_last_terminal_child(tree_or_node: Union[Tree, Node]) -> Union[Node, None]:
        """Get the last terminal child of a tree or node. Traverse the tree recursively by getting the `last_child()` of
        the current node until a terminal node (no children) is found.

        If the tree is empty, return `None`.

        Args:
            tree_or_node (Union[Tree, Node]): The tree or node to traverse to find the last terminal child.

        Returns:
            Union[Node, None]: The last terminal child of the tree or node, or `None` if the tree is empty.
        """
        if len(tree_or_node.children) == 0:
            if isinstance(tree_or_node, Tree):
                return None
            return tree_or_node
        else:
            node = tree_or_node.last_child()
            if node is None:
                raise IndexError("The node was None - last terminal child not found. Empty tree?")
            return tree_helpers.get_last_terminal_child(node)

    @staticmethod
    def get_message_list(to_node: Node) -> List["Message"]:
        """Get the list of messages from the root node to the given node (`to_node`), inclusive.

        Returns:
            List[Message]: The list of messages from the root node to the given node, inclusive.
        """
        return [node.data for node in to_node.get_parent_list(add_self=True)]

    @staticmethod
    def append_message_to_end_of_tree(tree: Tree, message: Message) -> Node:
        """Append a message to the end of the tree. If the tree is empty, add the message as the root node. Otherwise, find
        the last terminal child of the tree and add the message as a child of that node.

        Args:
            tree (Tree): The tree to which the message should be appended.
            message (Message): The message to append.

        Returns:
            Node: The new node to which the message was appended.
        """
        last_message_node = tree_helpers.get_last_terminal_child(tree)
        if last_message_node is None:
            return tree.add(message)
        else:
            return last_message_node.add(message)

    @staticmethod
    def append_multiple_messages_to_end_of_tree(tree: Tree, messages: List[Message]) -> Node:
        """Append multiple messages to the end of the tree. If the tree is empty, add the messages as the root node. Otherwise, find
        the last terminal child of the tree and add the messages as children of that node.

        Args:
            tree (Tree): The tree to which the messages should be appended.
            messages (List[Message]): The messages to append.

        Returns:
            Node: The new node to which the messages were appended.
        """
        node = None
        for message in messages:
            if node is None:
                node = tree_helpers.append_message_to_end_of_tree(tree, message)
            else:
                # To avoid re-traversing the tree, we use the last appended node to append the next message.
                node = node.add(message)
        if node is None:
            raise ValueError("No messages were appended to the tree")
        return node

    @staticmethod
    def get_linear_message_history_to_terminal_child(tree: Tree) -> List[Message]:
        """Get the linear message history (a list) from the root node to the last terminal child of the tree.

        Args:
            tree (Tree): The tree from which to get the linear message history.

        Returns:
            List[Message]: The message list.
        """
        last_message_node = tree_helpers.get_last_terminal_child(tree)
        if last_message_node is None:
            message_list = []
        else:
            message_list = tree_helpers.get_message_list(last_message_node)
        return message_list

    @staticmethod
    def get_last_branch_point_node(tree: Tree) -> Node:
        """Get the message at the last branch point of the tree.

        Args:
            tree (Tree): The tree from which to get the last branch point message.

        Returns:
            Message: The message at the last branch point.
        """

        def _ascend_tree_to_previous_branch_point(node: Node) -> Node:
            """Ascend the tree to find the previous branch point.
            If the parent of the current node is a branch point, return
            the parent. Otherwise, continue to ascend the tree until a branch point is found.

            Args: node (Node): The current node to check for a branch point.

            Returns: Node: The previous branch point.
            """
            parent = node.parent
            if parent is None:
                return None  # type: ignore
            if parent.data.role == BRANCH_ROLE:
                return parent
            return _ascend_tree_to_previous_branch_point(parent)

        last_terminal_child = tree_helpers.get_last_terminal_child(tree)
        if last_terminal_child is None:
            raise ValueError("No messages in the tree")

        potential_branch_point = _ascend_tree_to_previous_branch_point(last_terminal_child)
        if potential_branch_point is None:
            warnings.warn("No previous branch point found. Returning the root node with no action taken.")
            return last_terminal_child
        try:
            while not (
                len(potential_branch_point.children) < BRANCH_LIMIT and len(potential_branch_point.children) > 0
            ):
                potential_branch_point = _ascend_tree_to_previous_branch_point(potential_branch_point)
        except AttributeError:
            warnings.warn(f"potential_branch_point has no children. potential_branch_point: {potential_branch_point}")
            if potential_branch_point is None:
                return last_terminal_child

        if len(potential_branch_point.children) == 0:
            raise ValueError("parent message has no children - this should never be raised")
        return potential_branch_point


# === Tree-related functions [END] ===


class EngineAgent:
    def __init__(
        self,
        agent_type: Agent,
        system_message_template: str,
        first_message_content: str,
        first_message_role: Role,
        # -- --- ---
        # Callables - assign Engine methods to these. Note the parameters.
        # <method>(self, agent) -> <return_type>
        # The `agent` parameter will be the EngineAgent instance itself.
        set_initial_messages: Callable[["EngineBase", "EngineAgent"], List[Message]],
        gather_messages: Callable[["EngineBase", "EngineAgent"], Tuple[List[Message], ToolSpecs]],
        dispatch: Callable[["EngineBase", "EngineAgent"], EngineState],
    ):
        self.agent_type: Agent = agent_type
        self.system_message_template = system_message_template
        self.first_message_content = first_message_content
        self.first_message_role: Role = first_message_role

        self.set_initial_messages = set_initial_messages
        self.gather_messages = gather_messages
        self.dispatch = dispatch


class EngineBase(abc.ABC):
    def __init__(
        self,
        db: DB,
        session: Session,
        conda_path: Optional[str] = None,
    ) -> None:
        self.db = db
        self.session = session

        self.conda_path = conda_path

        self._new_session = not self.session.messages

        # List of message keys of the messages to send to the LLM, ONLY if in the current reasoning cycle.
        self.ephemeral_messages_to_send: List[str] = []

        # Make subdirectories.
        self.logs_path = os.path.join(self.session.working_directory, "logs")
        os.makedirs(self.logs_path, exist_ok=True)

        # For terminating an active tool:
        self.executing_tool: Optional[ToolBase] = None

        # Validate engine parameter are expected:
        for key in session.engine_params.keys():
            if key not in [param.name for param in self.get_engine_parameters()]:
                raise ValueError(f"Unknown engine parameter: {key}")
        # Fill in missing engine parameters with defaults:
        for engine_param in self.get_engine_parameters():
            if engine_param.name not in session.engine_params:
                session.engine_params[engine_param.name] = engine_param.default

        # Set up the agents.
        self._before_define_agents_hook()
        self.agents = self.define_agents()
        if not self.agents:
            raise ValueError("No agents defined for the engine. Must define at least one agent.")
        if self._new_session:  # If there are no messages in the session (new session)...
            # ... set the initial messages for the initial agent.
            initial_agent = self.agents[self.define_initial_agent()]
            engine_log(f"Setting initial messages for agent: {initial_agent.agent_type}")
            initial_agent.set_initial_messages(self, initial_agent)

    def _before_define_agents_hook(self) -> None:
        pass

    @property
    def working_directory(self) -> str:
        return self.session.working_directory

    @property
    def working_directory_abs(self) -> str:
        return os.path.realpath(self.session.working_directory)

    @property
    def engine_params(self) -> Dict[str, EngineParameterValue]:
        return self.session.engine_params

    @staticmethod
    @abc.abstractmethod
    def get_engine_name() -> str: ...

    @staticmethod
    @abc.abstractmethod
    def get_engine_parameters() -> List[EngineParameter]: ...

    @abc.abstractmethod
    def describe_tool_to_user(self, tool: ToolBase) -> str: ...

    def define_agents(self) -> Dict[Agent, EngineAgent]:
        # Default case - no agents.
        return dict()

    def define_initial_agent(self) -> Agent:
        return "worker"

    def get_current_plan(self) -> Any:
        return None

    def get_token_counts(self) -> Dict[Agent, int]:
        return dict()

    # TODO: Rethink.
    @abc.abstractmethod
    def _set_initial_messages(self, agent: Optional[EngineAgent] = None) -> List[Message]: ...

    @abc.abstractmethod
    def _append_message(self, message: Message) -> None: ...

    def stop_tool_execution(self) -> None:
        # If there is an active tool execution, make sure to terminate it:
        if self.executing_tool is not None:
            engine_log("Terminating active tool execution.")
            self.executing_tool.stop_execution()
            self.session.engine_state.executing_tool = None
            self.db.update_session(self.session)  # Record that no execution is active.
            self.executing_tool = None
            engine_log("Terminated active tool execution.")
        else:
            engine_log("No active tool execution to terminate.")

    def describe_working_directory_list(self) -> List[FileInfo]:
        file_infos: List[FileInfo] = []

        # List all files in the directory
        for file_name in os.listdir(os.path.realpath(self.working_directory)):
            # Construct full file path:
            file_path = os.path.realpath(os.path.join(self.working_directory, file_name))
            # Skip directories:
            if os.path.isdir(file_path):
                continue
            # Skip *.py files:
            if file_name.endswith(".py"):
                continue
            # Get file size and convert it to a more readable format:
            file_size, file_size_unit = convert_size(os.path.getsize(file_path))
            # Get last modification time:
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))

            file_infos.append(FileInfo(name=file_name, size=file_size, size_units=file_size_unit, modified=mod_time))

        return file_infos

    def describe_working_directory_str(self) -> str:
        files_info = self.describe_working_directory_list()
        return "\n".join(
            [
                f"{fi.name}, Size: {fi.size:3.1f} {fi.size_units}, Last Modified: {fi.modified.strftime('%Y-%m-%d %H:%M:%S')}"
                for fi in files_info
            ]
        )

    def _log_messages(self, messages: List[Dict], tools: Optional[List[Dict]], metadata: Optional[Dict]) -> None:
        messages_file_name = make_filename_path_safe(f"{self.get_last_message().key}.yaml")
        messages_file_path = os.path.join(self.logs_path, messages_file_name)
        log_messages_to_file(
            messages=messages,
            tools=tools,
            metadata=metadata,
            path=messages_file_path,
        )

    @abc.abstractmethod
    def get_message_history(self) -> List[Message]: ...

    @abc.abstractmethod
    def ingest_user_input(self, user_input: str) -> None: ...

    def reason(self) -> StreamLike:
        # Reset ephemeral message keys list.
        self.ephemeral_messages_to_send = []

        # Gather the messages and tools for the appropriate agent:
        agent = self.agents[self.session.engine_state.agent]
        messages, tools = agent.gather_messages(self, agent)

        # The ephemeral messages are only sent to the LLM if they are found in ephemeral_messages_to_send.
        messages = [
            message
            for message in messages
            if message.visibility != "llm_only_ephemeral" or message.key in self.ephemeral_messages_to_send
        ]

        yield from self._llm_call(messages, tools)

        # Handle the active agent dispatch:
        self.session.engine_state = agent.dispatch(self, agent)
        self.db.update_session(self.session)

    @abc.abstractmethod
    def _llm_call(self, messages: List[Message], tools: ToolSpecs) -> StreamLike: ...

    def get_state(self) -> EngineState:
        return self.session.engine_state

    def project_completed(self) -> bool:
        return False

    # Override as needed.
    def get_last_message(self) -> Message:
        """Get the last message from the session messages.

        Raises:
            ValueError: If there are no messages in the session.

        Returns:
            Message: The last message in the session.
        """
        last_terminal_child = tree_helpers.get_last_terminal_child(self.session.messages)
        if last_terminal_child is None:
            raise ValueError("No messages in the session")
        return last_terminal_child.data

    def update_state(self) -> None:
        # Propagate the engine state to the last message.
        if self.get_last_message().engine_state is not None:
            self.get_last_message().engine_state = self.session.engine_state

        # Update session.
        self.db.update_session(self.session)

    @abc.abstractmethod
    def discard_last(self) -> bool: ...

    # TODO: Possibly improve/rethink.
    def restart_at_user_message(self, key: str) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def create_new_message_branch(self) -> bool: ...

    @abc.abstractmethod
    def execute_tool_call(
        self,
        tool_call: ToolCallRecord,
        user_input: UserInputRequest,
    ) -> ToolReturnIter: ...

    @abc.abstractmethod
    def execute_generated_code(
        self,
    ) -> CodeExecReturn: ...


class ChunkTracker:
    def __init__(self) -> None:
        self.chunks: List[ChunkSentinel] = ["not_started"]

    def update(self, sentinel: ChunkSentinel) -> None:
        self.chunks.append(sentinel)

    def processing_required(self) -> bool:
        if self.chunks[-1:] == ["end_of_stream"]:
            return True
        if self.chunks[-2:] == ["text", "tool_call"]:
            # Moved from text to tool call, append the text message.
            return True
        if self.chunks[-2:] == ["tool_call", "text"]:
            # Moved from tool call to text, append the tool call.
            return True
        return False
