import abc
import os
from datetime import datetime
from typing import Any, Callable, Dict, Generator, Iterator, List, Literal, Optional, Tuple, Union, get_args

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
from ._config import get_dotenv_config

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


class EngineAgent:
    def __init__(
        self,
        agent_type: Agent,
        system_message_template: str,
        first_message_content: Optional[str],
        first_message_role: Optional[Role],
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
        self.first_message_role: Role = first_message_role  # pyright: ignore

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
        self.simulated_user = False

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
        if not self.session.messages:
            raise ValueError("No messages in the session")
        return self.session.messages[-1]

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
