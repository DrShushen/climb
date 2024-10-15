import copy
import enum
import getpass
import os
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import matplotlib.figure
import plotly.graph_objects
import pydantic
import rich.pretty
from nutree import Tree

from .utils import make_filename_path_safe, truncate_dict_values

if TYPE_CHECKING:
    from climb.db import DB

Role = Literal["system", "assistant", "user", "tool", "code_execution", "new_branch"]
# Explanation:
# - "system": Messages that are system messages from the LLM perspective.
# - "assistant": Messages that are assistant messages from the LLM perspective.
# - "user": Messages that are user messages from the LLM perspective.
# - "tool": Messages that are tool messages from the LLM perspective, and also used internally for tracking tools.
# - "code_execution": Messages that contain code execution information, used internally for managing code execution.
# - "new_branch": Messages that indicate a new branch (Tree/Chain of Thought feature).


MessageVisibility = Literal["all", "ui_only", "llm_only", "llm_only_ephemeral", "system_only"]
# Explanation:
# - "all": Message is given to all parties.
# - "ui_only": Message is given only to the UI.
# - "llm_only": Message is given only to the LLM.
# - "llm_only_ephemeral": Message is given only to the LLM, but only in the current reasoning cycle.
# - "system_only": Message is given only to the system (system being this tool, for its internal use,
#   e.g. for handling things like code execution.)

Agent = Literal["coordinator", "worker", "supervisor"]

ToolSpecs = Union[List[Dict[str, Any]], None]


class ResponseKind(enum.Enum):
    NOT_SET = enum.auto()  # An indicator to show while still streaming.
    TEXT_MESSAGE = enum.auto()
    TOOL_REQUEST = enum.auto()
    CODE_GENERATION = enum.auto()


class SessionSettings(pydantic.BaseModel):
    show_tool_call_logs: bool = True
    show_tool_call_return: bool = True
    show_code: bool = True
    show_code_out: bool = True
    show_planning_details: bool = False
    show_full_message_history: bool = False
    show_message_history_length: int = 15


class UserSettings(pydantic.BaseModel):
    user_name: str = getpass.getuser()

    # UI default settings:
    default_session_settings: SessionSettings = SessionSettings()

    # Internal functionality:
    active_session: Optional[str] = None


ToolUserOut = Union[
    str,
    plotly.graph_objects.Figure,
    matplotlib.figure.Figure,
]
ToolUserReportSeq = List[ToolUserOut]


class ToolCallRecord(pydantic.BaseModel):
    name: str
    arguments: str  # TODO: improve this.
    engine_id: Optional[str] = None


class Message(pydantic.BaseModel):
    key: str
    role: Role
    visibility: MessageVisibility = "all"
    agent: Agent = "worker"

    new_reasoning_cycle: bool = False

    # --- Pure text ---
    text: Optional[str]

    # --- Token tracking ---
    token_counts: Dict[Agent, Optional[int]] = dict()

    # --- Tool call ---
    incoming_tool_calls: Optional[List[ToolCallRecord]] = None
    outgoing_tool_call: Optional[ToolCallRecord] = None
    # Output:
    tool_call_success: Optional[bool] = None
    tool_call_logs: Optional[str] = None
    tool_call_return: Optional[str] = None
    tool_call_user_report: Optional[ToolUserReportSeq] = None

    # --- Code generation ---
    generated_code_dependencies: Optional[List[str]] = None
    generated_code: Optional[str] = None
    # Output:
    generated_code_success: Optional[bool] = None
    generated_code_stdout: Optional[str] = None
    generated_code_stderr: Optional[str] = None

    files_in: Optional[List[str]] = None
    files_out: Optional[List[str]] = None

    # Engine state record.
    engine_state_value: Optional["EngineState"] = pydantic.Field(default=None, alias="engine_state")

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    @property
    def engine_state(self) -> Optional["EngineState"]:
        return self.engine_state_value

    @engine_state.setter
    def engine_state(self, value: Optional["EngineState"]) -> None:
        # Store a "snapshot" copy of the engine state.
        self.engine_state_value = copy.deepcopy(value)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Message):
            return False
        return self.key == other.key


EngineParameterValue = Union[str, float, bool]


# TODO: Consistency validation.
class EngineParameter(pydantic.BaseModel):
    name: str
    description: str
    kind: Literal["float", "bool", "enum"]
    default: EngineParameterValue
    enum_values: Optional[List[str]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    set_by_static_method: Optional[str] = None
    disabled: Optional[bool] = False


InteractionStage = Literal["reason", "output", "await_user_input"]

UserInputKind = Literal["text", "file"]
UserInputRequestKey = str


class UploadedFileAbstraction(pydantic.BaseModel):
    name: str
    content: bytes


class UserInputRequest(pydantic.BaseModel):
    key: UserInputRequestKey
    kind: UserInputKind
    description: Optional[str] = None
    extra: Dict[str, Any] = dict()

    _received_input: Any = None

    @property
    def received_input(self) -> Any:
        return self._received_input

    @received_input.setter
    def received_input(self, value: Any) -> None:
        self._received_input = value
        self.check_received_input()

    def check_received_input(self) -> None:
        if self.kind == "file":
            if not isinstance(self.received_input, UploadedFileAbstraction):
                raise ValueError(
                    f"Expected 'received_input' to be of type '{UploadedFileAbstraction.__name__}' "
                    f"but got {type(self.received_input)}."
                )
        elif self.kind == "text":
            if not isinstance(self.received_input, str):
                raise ValueError("Expected 'received_input' to be of type 'str' but got something else.")


class UIControlledState(pydantic.BaseModel):
    # The state that the UI controls (can modify).
    interaction_stage: InteractionStage = "reason"
    input_request: Optional[UserInputRequest] = None
    input_placeholder: Optional[str] = None  # Only used for `restart_at_user_message`


class EngineState(pydantic.BaseModel):
    streaming: bool

    agent: Agent
    agent_switched: bool

    agent_state: Dict[Agent, Dict[str, Any]] = dict()

    executing_tool: Optional[str] = None

    user_message_requested: bool = False

    response_kind_value: ResponseKind = pydantic.Field(alias="response_kind", default=ResponseKind.NOT_SET)
    tool_request_value: Optional[ToolCallRecord] = pydantic.Field(alias="tool_request", default=None)

    ui_controlled_value: UIControlledState = pydantic.Field(alias="ui_controlled", default=UIControlledState())

    def __rich_repr__(self) -> Generator:
        yield "streaming", self.streaming
        yield "agent", self.agent
        yield "agent_switched", self.agent_switched
        yield "agent_state", truncate_dict_values(self.agent_state, max_len=50)
        yield "executing_tool", self.executing_tool
        yield "user_message_requested", self.user_message_requested
        yield "response_kind", self.response_kind_value
        yield "tool_request", self.tool_request_value
        yield "ui_controlled", self.ui_controlled_value

    def __repr__(self) -> str:
        return rich.pretty.pretty_repr(self)

    def __str__(self) -> str:
        return rich.pretty.pretty_repr(self)

    # Properties with self-consistency checks. ---

    @property
    def response_kind(self) -> ResponseKind:
        # Assert self-consistency.

        # Ensure that the response kind is set if streaming is done.
        if self.response_kind_value != ResponseKind.NOT_SET and self.streaming is True:
            raise ValueError(
                "EngineState self-consistency error: Response kind was set even though streaming is ongoing."
            )

        return self.response_kind_value

    @response_kind.setter
    def response_kind(self, value: ResponseKind) -> None:
        self.response_kind_value = value

    @property
    def tool_request(self) -> Optional[ToolCallRecord]:
        # Assert self-consistency.

        # Ensure that tool requests are set if the response kind is tool request.
        if self.response_kind_value == ResponseKind.TOOL_REQUEST and self.tool_request_value is None:
            raise ValueError(
                "EngineState self-consistency error: Tool request was not set even though response kind is "
                "tool request."
            )

        return self.tool_request_value

    @tool_request.setter
    def tool_request(self, value: Optional[ToolCallRecord]) -> None:
        self.tool_request_value = value

    # --- --- --- --- --- ---

    @property
    def ui_controlled(self) -> UIControlledState:
        return self.ui_controlled_value

    @ui_controlled.setter
    def ui_controlled(self, value: UIControlledState) -> None:
        # Always store a copy of the UI controlled state, so that modifications on the UI end
        # do not affect this snapshot value.
        self.ui_controlled_value = copy.deepcopy(value)

    # --- --- --- --- --- ---


class Session(pydantic.BaseModel):
    session_key: str
    working_directory: str

    started_at: datetime = datetime.now()
    friendly_name: str = ""

    engine_name: str
    engine_params: Dict[str, EngineParameterValue] = dict()

    # The Optional + exclude is used to avoid the nutree can't pickle lock error.
    # https://stackoverflow.com/questions/66419620/pydantic-settings-typeerror-cannot-pickle-thread-lock-object
    # In order to avoid always checking for None, the property `messages` is used.
    message_tree: Optional[Tree] = pydantic.Field(default=None, exclude=True)

    # NOTE: Notice the default values that are set here!
    engine_state: EngineState = EngineState(
        streaming=False,
        agent="worker",
        agent_switched=False,
        ui_controlled=UIControlledState(),
    )

    session_settings: SessionSettings = SessionSettings()

    @property
    def messages(self) -> Tree:
        if self.message_tree is None:
            self.message_tree = Tree()
        return self.message_tree

    @messages.setter
    def messages(self, value: Tree) -> None:
        self.message_tree = value

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)


def create_new_session(
    session_name: Optional[str],
    engine_name: str,
    engine_params: Dict[str, EngineParameterValue],
    db: "DB",
) -> Session:
    user_settings = db.get_user_settings()

    now = datetime.now()
    key = KeyGeneration.generate_session_key(use_time=now)
    wd_root = "./wd"
    wd_subdir = make_filename_path_safe(key)
    wd = os.path.join(wd_root, wd_subdir)
    friendly_name = " ".join(key.split("_")).capitalize() if session_name is None else session_name
    session = Session(
        session_key=key,
        working_directory=wd,
        friendly_name=friendly_name,
        engine_name=engine_name,
        engine_params=engine_params,
        session_settings=user_settings.default_session_settings,
    )

    os.makedirs(wd, exist_ok=True)

    # Update active session.
    user_settings.active_session = key
    db.update_user_settings(user_settings)

    # Update session.
    db.update_session(session)

    return session


class KeyGeneration:
    @staticmethod
    def generate_message_key(use_time: Optional[datetime] = None) -> str:
        if use_time is None:
            use_time = datetime.now()
        current_time = use_time.strftime("%Y-%m-%d_%H:%M:%S")
        timestamp = time.time()
        return f"{current_time}_{timestamp}"

    @staticmethod
    def generate_session_key(use_time: Optional[datetime] = None) -> str:
        if use_time is None:
            use_time = datetime.now()
        current_time = use_time.strftime("%Y-%m-%d_%H:%M:%S")
        return f"session_{current_time}"


FileInfoCategory = Literal["image", "data", "model", "other"]


def get_previewable(filetype: FileInfoCategory) -> bool:
    if filetype in ("image", "data"):
        return True
    return False


# NOTE: Update as needed.
FILETYPE_MAP: Dict[FileInfoCategory, Dict[str, Any]] = {
    "image": {
        "ext": ("png", "jpg", "jpeg"),  # Must match one of these extensions.
        "regex": None,  # Must match this regex, if specified.
    },
    "data": {
        "ext": ("csv",),
        "regex": None,
    },
    "model": {
        "ext": ("pkl", "pth", "pt", "p"),
        "regex": r"model(_[a-zA-Z]+)?",
    },
}


def get_category_from_name(name: str) -> FileInfoCategory:
    def _ext(exts: Tuple[str, ...]) -> Tuple[str, ...]:
        return tuple("." + ext for ext in exts)

    for filetype, filetype_map in FILETYPE_MAP.items():
        if name.lower().endswith(_ext(filetype_map["ext"])):
            if filetype_map["regex"] is None:
                return filetype
            else:
                if re.match(filetype_map["regex"], name):
                    return filetype

    return "other"


class FileInfo(pydantic.BaseModel):
    name: str
    size: float
    size_units: str
    modified: datetime

    @property
    def category(self) -> FileInfoCategory:
        return get_category_from_name(self.name)

    @property
    def previewable(self) -> bool:
        return get_previewable(self.category)
